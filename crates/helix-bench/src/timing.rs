//! The measurement core: an adaptive, interleaved, Instant-based sampler
//! implementing the protocol from `docs/research/benchmark-methodology.md`
//! (recommendations 1–3, pitfalls 4–5, 11).
//!
//! # Protocol per (kernel, variant)
//!
//! 1. **Pilot**: run the closure doubling its rep count until one batch takes
//!    at least [`PILOT_TARGET_MS`] — this picks `R`, the inner repetitions per
//!    sample, so every *sample* lands in the 100–250 ms window where QPC
//!    resolution (~100 ns) and `Instant::now()` overhead (~30 ns) are noise.
//!    Interpreter variants need far fewer reps than native ones; freezing the
//!    chosen `R` into the results keeps that choice auditable (pitfall 13).
//! 2. **Warmup**: three *untimed* batches at the frozen `R` absorb JIT
//!    first-call codegen, cache/branch-predictor state and page-table effects.
//! 3. **Sampling**: [`K`] = 15 timed samples, each `R` inner reps.
//! 4. **Quality gate**: coefficient of variation = stddev/mean; above 5% the
//!    whole sampling stage reruns **once** (methodology rec. 2) and the
//!    strictly tighter set wins — a noisier rerun never replaces the
//!    originals.
//!
//! # Interleaving
//!
//! Variants are measured round-robin (`interp, native-seq, native-par,
//! interp, …`) by [`run_interleaved`], never one variant to completion before
//! the next starts: fixed-order blocking lets thermal/background drift bias
//! whichever variant ran last (pitfall 11). All timing uses
//! [`std::time::Instant`] — QPC-backed on Windows, monotonic, ~100 ns ticks;
//! deliberately no `quanta`/TSC (drift issue, methodology fact 2).

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Timed samples taken per (kernel, variant) — Kalibera & Jones' "~30 reps"
/// halved for campaign wall-clock; CV reporting covers the slack.
pub const K_SAMPLES: usize = 15;

/// Untimed batches run after the pilot (absorbs JIT compile + caches).
pub const WARMUPS: usize = 3;

/// Target duration of ONE SAMPLE (R inner reps), milliseconds.
/// The 100–250 ms window makes timer resolution irrelevant.
pub const SAMPLE_MIN_MS: f64 = 100.0;
pub const SAMPLE_MAX_MS: f64 = 250.0;

/// Pilot phase stops as soon as a batch reaches this many milliseconds.
const PILOT_TARGET_MS: f64 = SAMPLE_MIN_MS;

/// Coefficient of variation (stddev/mean) above which sampling reruns once.
pub const CV_RERUN_THRESHOLD: f64 = 0.05;

/// One timed sample: total elapsed for R inner reps plus the rep count, kept
/// together so per-rep times can always be recomputed offline.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sample {
    /// Wall-clock duration of the whole sample (R reps included).
    pub elapsed: Duration,
    /// Inner repetitions inside this sample.
    pub reps: u32,
}

impl Sample {
    /// Mean duration of one inner repetition.
    #[must_use]
    pub fn per_rep(&self) -> Duration {
        self.elapsed / self.reps.max(1)
    }

    /// Per-rep time in milliseconds (the unit every report table uses).
    #[must_use]
    pub fn per_rep_ms(&self) -> f64 {
        self.per_rep().as_secs_f64() * 1_000.0
    }
}

/// Result of measuring one variant: raw samples plus derived statistics.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Measurement {
    /// Rep count frozen by the pilot (metadata, pitfall 13).
    pub reps_per_sample: u32,
    /// Raw per-sample durations (ms) — hyperfine-style, recomputable offline.
    pub samples_ms: Vec<f64>,
    /// Median per-rep time in ms (headline number).
    pub median_ms: f64,
    /// Minimum per-rep time in ms (noise floor / "best case").
    pub min_ms: f64,
    /// Mean per-rep time in ms (denominator of the CV).
    pub mean_ms: f64,
    /// Standard deviation of per-rep times in ms.
    pub stddev_ms: f64,
    /// Coefficient of variation `stddev/mean` (0 when mean is 0).
    pub cv: f64,
    /// True when the first sample batch exceeded [`CV_RERUN_THRESHOLD`] and a
    /// rerun was performed. The rerun's samples replaced the originals only
    /// when strictly tighter (lower CV); a noisier rerun never displaces them.
    pub reran_for_cv: bool,
}

/// Median of a numeric slice (average of the two middles for even lengths).
#[must_use]
pub fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    Some(if values.len().is_multiple_of(2) {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    })
}

/// Population standard deviation of a slice.
#[must_use]
pub fn stddev(values: &[f64]) -> Option<f64> {
    let n = values.len();
    if n == 0 {
        return None;
    }
    let mean = values.iter().sum::<f64>() / n as f64;
    let ss = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>();
    Some((ss / n as f64).sqrt())
}

/// Coefficient of variation: population stddev / |mean| (0 for zero mean).
#[must_use]
pub fn cv_of(values: &[f64]) -> Option<f64> {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let sd = stddev(values)?;
    Some(if mean == 0.0 { 0.0 } else { sd / mean.abs() })
}

/// Picks the inner rep count `R` so one sample lands in the target window.
///
/// Doubles from 1 until a batch reaches [`PILOT_TARGET_MS`]; if the batch
/// overshoots [`SAMPLE_MAX_MS`] it halves once to land inside the window
/// (only exact when durations scale linearly, which they do here since each
/// rep is identical work). Callers may pass `min_reps = 1`.
///
/// # Errors
/// Propagates the first batch failure verbatim — a pilot whose code cannot
/// execute has no rep count worth freezing.
pub fn pick_reps<F: FnMut(u32) -> Result<Duration, String>>(
    mut run_batch: F,
) -> Result<u32, String> {
    let mut reps: u32 = 1;
    loop {
        let t = run_batch(reps)?;
        if t.as_secs_f64() * 1_000.0 >= PILOT_TARGET_MS {
            // Overshoot correction: linear back-off toward the window.
            let secs = t.as_secs_f64();
            let ideal = (reps as f64 * (SAMPLE_MIN_MS / 1000.0) / secs).ceil();
            let clamped = ideal.clamp(1.0, (reps as f64) * (SAMPLE_MAX_MS / SAMPLE_MIN_MS));
            return Ok(clamped.max(1.0) as u32);
        }
        // Cap the geometric growth: beyond ~2^24 reps something is wrong
        // (e.g. a closure that does nothing); bail out rather than spin.
        if reps >= 1 << 24 {
            return Ok(reps);
        }
        reps = reps.saturating_mul(2);
    }
}

/// Times `reps` invocations of `f` as one batch.
fn time_batch(f: &dyn Fn(), reps: u32) -> Duration {
    let start = Instant::now();
    for _ in 0..reps {
        f();
    }
    start.elapsed()
}

/// Full measurement of one closure following the module protocol.
///
/// `f` must be deterministic in output (checksummed by the caller elsewhere)
/// and already "warmed" at the process level; the warmups here handle JIT
/// codegen and microarchitectural state, not allocator growth.
#[must_use]
pub fn measure(f: impl Fn()) -> Measurement {
    // Plain closures cannot fail; fallible variants call
    // `measure_with_reps` directly and handle the error.
    measure_with_reps(|r| Ok(time_batch(&f, r)))
        .expect("infallible closure cannot fail")
}

/// Like [`measure`] but the closure receives the rep count (lets drivers
/// interleave multiple variants without re-running pilots per sample round).
///
/// The pilot runs first (picking `R`), then [`WARMUPS`] untimed batches at
/// `R`, then up to two sampling rounds gated by the CV threshold.
///
/// # Errors
/// Propagates the closure's first execution failure verbatim — a variant that
/// cannot run must never contribute samples to any measurement.
pub fn measure_with_reps(
    run: impl Fn(u32) -> Result<Duration, String>,
) -> Result<Measurement, String> {
    // -- pilot ---------------------------------------------------------
    let reps = pick_reps(&run)?;

    // -- warmups (untimed) ----------------------------------------------
    for _ in 0..WARMUPS {
        std::hint::black_box(run(reps)?);
    }

    // -- sampling, with one CV-gated rerun -------------------------------
    let mut best: Option<Measurement> = None;
    let mut reran_for_cv = false;
    for attempt in 0..2 {
        let mut per_rep_ms = Vec::with_capacity(K_SAMPLES);
        for _ in 0..K_SAMPLES {
            per_rep_ms.push(
                black_box_duration(run(reps)?).as_secs_f64() / f64::from(reps) * 1_000.0,
            );
        }
        let m = summarize(per_rep_ms, reps, reran_for_cv || attempt > 0);
        match &best {
            Some(prev) if prev.cv <= CV_RERUN_THRESHOLD => break, // keep tighter earlier run
            Some(prev) if m.cv < prev.cv => best = Some(m),
            None => best = Some(m),
            _ => {}
        }
        if best.as_ref().is_some_and(|m| m.cv <= CV_RERUN_THRESHOLD) {
            break;
        }
        reran_for_cv = true;
    }
    // Honesty: once a rerun ran, the SURVIVING measurement must record it
    // even when the original batch won (mirrors run_interleaved's else arm).
    if reran_for_cv && let Some(m) = best.as_mut() {
        m.reran_for_cv = true;
    }
    Ok(best.unwrap_or_else(|| summarize(Vec::new(), reps, false)))
}

/// Small helper mirroring `std::hint::black_box` for durations so the
/// compiler cannot fold consecutive identical batches together.
#[inline]
fn black_box_duration(d: Duration) -> Duration {
    std::hint::black_box(d)
}

/// Derives statistics from raw per-rep milliseconds.
fn summarize(mut per_rep_ms: Vec<f64>, reps: u32, reran: bool) -> Measurement {
    let mean = if per_rep_ms.is_empty() {
        0.0
    } else {
        per_rep_ms.iter().sum::<f64>() / per_rep_ms.len() as f64
    };
    let sd = stddev(&per_rep_ms).unwrap_or(0.0);
    let median = median(&mut per_rep_ms).unwrap_or(0.0);
    Measurement {
        reps_per_sample: reps,
        median_ms: median,
        min_ms: per_rep_ms.iter().copied().fold(f64::INFINITY, f64::min),
        mean_ms: mean,
        stddev_ms: sd,
        cv: cv_of(&per_rep_ms).unwrap_or(0.0),
        reran_for_cv: reran,
        samples_ms: per_rep_ms,
    }
}

// -- Round-robin driver ------------------------------------------------------

/// Measures several variants interleaved, round-robin.
///
/// Each entry gets its own pilot + warmups first (so `R`s are frozen before
/// any sampling starts), then the driver walks `rounds × variants` taking one
/// sample per visit. This defeats thermal-drift bias: drift hits every
/// variant proportionally instead of only whichever finished last.
///
/// Returns one [`Measurement`] per variant, in input order.
///
/// # Errors
/// Propagates the first failing variant's error text (tagged with the
/// variant's name) — a variant that fails mid-sampling aborts the round
/// instead of silently contributing garbage or missing samples.
pub fn run_interleaved<F>(variants: &[&str], mut sample_once: F) -> Result<Vec<Measurement>, String>
where
    F: FnMut(usize, u32) -> Result<Duration, String>,
{
    // Per-variant pilot + warmup phase (untimed by us; the closures time
    // their own batches). Each variant pilots with its OWN index — rep
    // counts differ by orders of magnitude between interp and native.
    let mut pilot_err = None;
    let reps: Vec<u32> = (0..variants.len())
        .map(|vi| match pick_reps(|r| sample_once(vi, r)) {
            Ok(r) => r,
            Err(e) => {
                pilot_err.get_or_insert_with(|| format!("{}: {e}", variants[vi]));
                0
            }
        })
        .collect();
    if let Some(e) = pilot_err {
        return Err(e);
    }

    for (vi, _) in variants.iter().enumerate() {
        for _ in 0..WARMUPS {
            std::hint::black_box(sample_once(vi, reps[vi])?);
        }
    }

    // Interleaved sampling: rounds outermost, variants innermost.
    let mut buckets: Vec<Vec<f64>> = vec![Vec::with_capacity(K_SAMPLES); variants.len()];
    for _ in 0..K_SAMPLES {
        for (vi, _) in variants.iter().enumerate() {
            let d = black_box_duration(sample_once(vi, reps[vi])?);
            buckets[vi].push(d.as_secs_f64() / f64::from(reps[vi]) * 1_000.0);
        }
    }

    // CV gate per variant: rerun just the offending variant's bucket once,
    // sampling that same variant's closure again.
    Ok(buckets
        .into_iter()
        .zip(reps)
        .enumerate()
        .map(|(vi, (bucket, reps))| {
            let mut m = summarize(bucket, reps, false);
            if m.cv > CV_RERUN_THRESHOLD {
                let retry: Result<Vec<f64>, String> = (0..K_SAMPLES)
                    .map(|_| -> Result<f64, String> {
                        Ok(
                            black_box_duration(sample_once(vi, reps)?).as_secs_f64()
                                / f64::from(reps)
                                * 1_000.0,
                        )
                    })
                    .collect();
                let retry = retry?;
                let alt = summarize(retry, reps, true);
                if alt.cv < m.cv {
                    m = alt;
                } else {
                    m.reran_for_cv = true;
                }
            }
            Ok(m)
        })
        .collect::<Result<Vec<_>, String>>()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn median_odd_even_and_empty() {
        assert_eq!(median(&mut []), None);
        assert_eq!(median(&mut [3.0]), Some(3.0));
        assert_eq!(median(&mut [5.0, 1.0, 9.0]), Some(5.0));
        assert_eq!(median(&mut [4.0, 1.0]), Some(2.5));
        // Order independence: input gets sorted in place.
        assert_eq!(median(&mut [7.0, 2.0, 2.0, 8.0]), Some(4.5));
    }

    #[test]
    fn stddev_and_cv_math_on_synthetic_series() {
        // [2, 4, 4, 4, 5, 5, 7, 9]: textbook series, stddev = 2.
        let xs = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert!(approx(stddev(&xs).unwrap(), 2.0, 1e-12));
        assert!(approx(cv_of(&xs).unwrap(), 2.0 / 5.0, 1e-12));
        // Zero mean => CV defined as 0 (not NaN).
        assert_eq!(cv_of(&[-3.0, 3.0]).unwrap(), 0.0);
        assert_eq!(stddev(&[]), None);
    }

    #[test]
    fn pick_reps_scales_with_closure_cost() {
        // A closure costing ~10 ms per rep should settle near 100-250 ms /
        // 10 ms = 10-25 reps.
        let cost = |reps: u32| Ok(Duration::from_nanos(u64::from(reps) * 10_000_000));
        let r = pick_reps(cost).unwrap();
        assert!((10..=25).contains(&r), "picked {r}");
        // An instant closure hits the growth cap instead of spinning forever.
        let fast = pick_reps(|_: u32| Ok(Duration::ZERO)).unwrap();
        assert!(fast >= 1 << 20, "instant closure picked {fast}");
    }

    #[test]
    fn pick_reps_propagates_batch_errors() {
        let err = pick_reps(|_: u32| Err("boom".to_string())).unwrap_err();
        assert_eq!(err, "boom");
    }

    #[test]
    fn summarize_reports_consistent_stats() {
        let m = summarize(vec![10.0; K_SAMPLES], 7, false);
        assert_eq!(m.median_ms, 10.0);
        assert_eq!(m.min_ms, 10.0);
        assert_eq!(m.mean_ms, 10.0);
        assert_eq!(m.stddev_ms, 0.0);
        assert_eq!(m.cv, 0.0);
        assert!(!m.reran_for_cv);
        assert_eq!(m.reps_per_sample, 7);
        assert_eq!(m.samples_ms.len(), K_SAMPLES);
    }

    #[test]
    fn measure_times_a_known_sleep() {
        // 2 ms per call: pilot lands at ~25 reps (~250ms budget capped by the
        // window clamp); we only assert the order of magnitude and that all
        // samples arrived.
        let m = measure(|| std::thread::sleep(Duration::from_millis(2)));
        assert_eq!(m.samples_ms.len(), K_SAMPLES);
        assert!(m.median_ms > 1.0, "median {} too small", m.median_ms);
        assert!(m.median_ms < 50.0, "median {} too big", m.median_ms);
    }

    #[test]
    fn failing_variant_aborts_the_round_with_its_name() {
        // Variant 1 fails during its PILOT: the driver must surface the error
        // (tagged with the variant's name), never fabricate measurements for
        // it or silently drop it from the round.
        let runs_before_err = std::cell::Cell::new(0u32);
        let err = run_interleaved(&["good", "broken"], |vi, _| {
            if vi == 1 {
                Err("segv in jitted code".to_string())
            } else {
                runs_before_err.set(runs_before_err.get() + 1);
                Ok(Duration::from_millis(1))
            }
        })
        .unwrap_err();
        assert!(
            err.contains("broken") && err.contains("segv"),
            "error must name the failing variant and cause: {err}"
        );
    }

    #[test]
    fn mid_sampling_failure_propagates_too() {
        // A variant that survives its pilot but fails once sampling starts:
        // the whole round aborts rather than emitting a short bucket.
        let calls = std::cell::Cell::new(0u32);
        let err = run_interleaved(&["flaky"], |_, _| {
            calls.set(calls.get() + 1);
            // pilot(1 call reaching target is impossible; cap path) + warmups +
            // a few samples, then fail.
            if calls.get() > 10 {
                Err("late crash".to_string())
            } else {
                Ok(Duration::from_millis(150))
            }
        })
        .unwrap_err();
        assert_eq!(err, "late crash");
    }

    // -- CV-rerun contract ----------------------------------------------------
    //
    // Regression tests for the adversarial-review finding: a CV-gated rerun
    // must REPLACE the original samples only when the rerun is STRICTLY
    // tighter, and must never keep a statistically worse set. The closures are
    // fully scripted (call-count keyed), so both batches' statistics are
    // known exactly and the winner is decidable without any real timing.
    //
    // Call layout for every script (reps freeze at 1 via the 150 ms pilot):
    // call 0 = pilot, 1..=3 = warmups, 4..=18 = sample batch 1,
    // 19.. = sample batch 2 (the rerun).

    /// Batch alternating 80/120 ms: population stddev 20 over mean ~98.67,
    /// CV ~0.20 — comfortably above [`CV_RERUN_THRESHOLD`].
    fn noisy_80_120(n: usize) -> f64 {
        if (n - 4) % 2 == 0 { 80.0 } else { 120.0 }
    }

    #[test]
    fn cv_rerun_replaces_originals_when_strictly_tighter() {
        let calls = std::cell::Cell::new(0usize);
        let m = measure_with_reps(|_| {
            let n = calls.get();
            calls.set(n + 1);
            let ms = match n {
                0 => 150.0,
                1..=3 => 100.0,
                4..=18 => noisy_80_120(n),
                _ => 90.0, // rerun: perfectly flat, CV 0
            };
            Ok(Duration::from_millis(ms as u64))
        })
        .unwrap();
        assert!(m.reran_for_cv, "gate tripped; the rerun must be recorded");
        assert!(
            m.cv <= CV_RERUN_THRESHOLD,
            "tight rerun must win: cv {}",
            m.cv
        );
        assert_eq!(m.median_ms, 90.0, "stats must come from the rerun batch");
        assert!(m.samples_ms.iter().all(|&x| x == 90.0));
    }

    #[test]
    fn cv_rerun_never_keeps_a_statistically_worse_batch() {
        // Inverse script: the rerun comes back noisier (60/140 -> CV ~0.41)
        // than the originals (CV ~0.20). The ORIGINAL batch must survive
        // untouched; only the honesty flag flips.
        let calls = std::cell::Cell::new(0usize);
        let m = measure_with_reps(|_| {
            let n = calls.get();
            calls.set(n + 1);
            let ms = match n {
                0 => 150.0,
                1..=3 => 100.0,
                4..=18 => noisy_80_120(n),
                _ => {
                    if n % 2 == 0 {
                        60.0
                    } else {
                        140.0
                    }
                }
            };
            Ok(Duration::from_millis(ms as u64))
        })
        .unwrap();
        assert!(m.reran_for_cv, "a rerun ran and must be recorded");
        assert!(
            m.cv > CV_RERUN_THRESHOLD,
            "kept-batch honesty: cv {}",
            m.cv
        );
        assert_eq!(
            m.median_ms, 80.0,
            "original batch must survive a noisier rerun (its median is 80)"
        );
        assert!(
            m.samples_ms.iter().all(|&x| x == 80.0 || x == 120.0),
            "no rerun sample may leak into the kept set"
        );
    }

    #[test]
    fn interleaved_cv_gate_replaces_bucket_when_retry_is_tighter() {
        let flaky_calls = std::cell::Cell::new(0usize);
        let out = run_interleaved(&["stable", "flaky"], |vi, _| {
            if vi == 0 {
                return Ok(Duration::from_millis(50)); // flat: never trips
            }
            let n = flaky_calls.get();
            flaky_calls.set(n + 1);
            let ms = match n {
                0 => 150.0,
                1..=3 => 100.0,
                4..=18 => noisy_80_120(n),
                _ => 90.0,
            };
            Ok(Duration::from_millis(ms as u64))
        })
        .unwrap();
        assert_eq!(out.len(), 2);
        assert!(!out[0].reran_for_cv);
        assert!(out[0].cv <= CV_RERUN_THRESHOLD);
        assert!(out[1].reran_for_cv);
        assert!(out[1].cv <= CV_RERUN_THRESHOLD, "{}", out[1].cv);
        assert_eq!(out[1].median_ms, 90.0, "tight retry must replace");
    }

    #[test]
    fn interleaved_cv_gate_keeps_bucket_when_retry_is_worse() {
        let calls = std::cell::Cell::new(0usize);
        let out = run_interleaved(&["lucky-noise"], |_, _| {
            let n = calls.get();
            calls.set(n + 1);
            let ms = match n {
                0 => 150.0,
                1..=3 => 100.0,
                4..=18 => noisy_80_120(n),
                _ => {
                    if n % 2 == 0 {
                        60.0
                    } else {
                        140.0
                    }
                }
            };
            Ok(Duration::from_millis(ms as u64))
        })
        .unwrap();
        let m = &out[0];
        assert!(m.reran_for_cv, "the gate fired; flag must say so");
        assert_eq!(
            m.median_ms, 80.0,
            "original bucket must survive a noisier retry (its median is 80)"
        );
        assert!(
            m.samples_ms.iter().all(|&x| x == 80.0 || x == 120.0),
            "retry samples must be discarded"
        );
    }

    #[test]
    fn run_interleaved_returns_one_measurement_per_variant() {
        // Variant 0 "costs" 5 ms/rep, variant 1 costs 1 ms/rep.
        let out = run_interleaved(&["slow", "fast"], |vi, reps| {
            let per = if vi == 0 {
                Duration::from_millis(5)
            } else {
                Duration::from_millis(1)
            };
            Ok::<Duration, String>(per * reps)
        })
        .unwrap();
        assert_eq!(out.len(), 2);
        assert!(
            out[0].median_ms > out[1].median_ms * 3.0,
            "{} vs {}",
            out[0].median_ms,
            out[1].median_ms
        );
    }

    #[test]
    fn interleaved_driver_only_indexes_valid_variants() {
        // Regression: pilots previously all ran variant 0 and the CV rerun
        // used a sentinel index — both must target their OWN variant or the
        // driver indexes out of bounds on single-variant closures.
        let seen = std::cell::RefCell::new(Vec::new());
        let out = run_interleaved(&["only"], |vi, reps| {
            seen.borrow_mut().push(vi);
            Ok(Duration::from_millis(1) * reps.max(1))
        })
        .unwrap();
        assert_eq!(out.len(), 1);
        assert!(seen.borrow().iter().all(|&vi| vi == 0), "bad index seen");
    }
}
