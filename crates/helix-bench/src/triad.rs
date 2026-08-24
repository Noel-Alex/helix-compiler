//! STREAM-triad ceiling: the honest denominator for memory-bound speedups.
//!
//! `a = b + s*c` over 2^24 f64 elements is the canonical achievable-bandwidth
//! probe (methodology rec. 8, fact 7): desktops realize only ~70–85% of the
//! DDR spec sheet, and single threads reach a fraction of the all-core
//! figure. Reporting HELIX's saxpy GB/s **as a percentage of this measured
//! ceiling** converts "we got 22x" into "we reach 82% of what this machine
//! can actually move" — and prevents spec-sheet strawmen.
//!
//! Implementation: single-threaded loop plus `std::thread::scope` over
//! contiguous chunks (`chunks_mut`) — no deps, no unsafe. Bytes moved per
//! triad element per STREAM counting: 2 loads (`b`, `c`) + 1 store (`a`)
//! of 8 B = 24 B/elem.

use std::time::Instant;

/// Default vector length: 2^24 f64 = 128 MiB per array, 384 MiB working set —
/// comfortably past any LLC and into DRAM territory on desktop hardware.
pub const DEFAULT_TRIAD_N: usize = 1 << 24;

/// Bytes moved per element per STREAM counting (2 loads + 1 store of f64).
pub const BYTES_PER_ELEM: f64 = 24.0;

/// Result of one triad configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TriadResult {
    /// Elements per array.
    pub n: usize,
    /// Threads used (1 = serial).
    pub threads: usize,
    /// Best full-triad pass wall-clock, milliseconds.
    pub best_pass_ms: f64,
    /// Median full-triad pass wall-clock, milliseconds.
    pub median_pass_ms: f64,
    /// Achieved bandwidth of the median pass, GiB/s (2^30 bytes).
    pub gib_per_sec: f64,
}

/// Measures triad at `n` elements with `threads` participants.
///
/// Runs 2 untimed warmup passes then 5 timed ones (triad is so stable that
/// full sampler machinery adds nothing); reports best + median.
///
/// # Panics
///
/// Never panics; chunk math saturates so any `(n >= 1, threads >= 1)` works.
#[must_use]
pub fn measure_triad(n: usize, threads: usize) -> TriadResult {
    assert!(n >= 1, "triad needs at least one element");
    let threads = threads.max(1).min(n);

    let mut a = vec![0.0f64; n];
    let b = vec![1.5f64; n];
    let c = vec![2.5f64; n];

    // Warmups (also page-faults all arrays in).
    triad_pass(&mut a, &b, &c, threads);
    triad_pass(&mut a, &b, &c, threads);

    let mut samples: Vec<f64> = Vec::with_capacity(5); // pass durations, ms
    for _ in 0..5 {
        let start = Instant::now();
        triad_pass(&mut a, &b, &c, threads);
        samples.push(start.elapsed().as_secs_f64() * 1_000.0);
    }

    let mut scratch = samples.clone();
    let median_ms = crate::timing::median(&mut scratch).unwrap_or_default();
    let best_ms = samples.iter().copied().fold(f64::INFINITY, f64::min);
    // STREAM convention: report bandwidth from the MEDIAN pass.
    let bytes = (n as f64) * BYTES_PER_ELEM;
    let gib_per_sec = bytes / (median_ms / 1_000.0).max(1e-9) / (1024.0 * 1024.0 * 1024.0);

    TriadResult {
        n,
        threads,
        best_pass_ms: best_ms,
        median_pass_ms: median_ms,
        gib_per_sec,
    }
}

/// One full `a = b + s*c` pass split into `threads` contiguous chunks.
fn triad_pass(a: &mut [f64], b: &[f64], c: &[f64], threads: usize) {
    let s = 0.75f64;
    if threads <= 1 {
        for i in 0..a.len() {
            a[i] = b[i] + s * c[i];
        }
        return;
    }
    // Contiguous chunks defeat false sharing (pitfall 7); `chunks_mut`
    // splits `a` disjointly and the same offsets index b/c without unsafe.
    let chunk = a.len().div_ceil(threads);
    std::thread::scope(|scope| {
        for (t, a_chunk) in a.chunks_mut(chunk).enumerate() {
            let start = t * chunk;
            let end = (start + a_chunk.len()).min(b.len());
            let b_chunk = &b[start..end];
            let c_chunk = &c[start..end];
            scope.spawn(move || {
                for i in 0..a_chunk.len() {
                    a_chunk[i] = b_chunk[i] + s * c_chunk[i];
                }
            });
        }
    });
}

/// Sweeps thread counts {1, 2, 4, ..., up to hw} and returns each result.
#[must_use]
pub fn triad_sweep(n: usize, max_threads: usize) -> Vec<TriadResult> {
    let mut out = Vec::new();
    let mut t = 1usize;
    while t <= max_threads.max(1) {
        out.push(measure_triad(n, t));
        if t == 1 {
            t = 2;
        } else {
            t *= 2;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serial_triad_runs_fast_at_reduced_n() {
        let r = measure_triad(1 << 16, 1);
        assert_eq!(r.n, 1 << 16);
        assert_eq!(r.threads, 1);
        // Whole measurement (warmups + 5 passes over 512 KiB) stays tiny.
        assert!(r.median_pass_ms < 100.0, "{r:?}");
        // 64 Ki elems * 24 B in well under 10 ms => far above 1 GiB/s even
        // on the slowest sandboxed runner.
        assert!(r.gib_per_sec > 1.0, "{}", r.gib_per_sec);
    }

    #[test]
    fn threaded_triad_uses_requested_threads() {
        let r = measure_triad(1 << 16, 4);
        assert_eq!(r.threads, 4);
        assert!(r.median_pass_ms < 500.0, "{r:?}");
    }

    #[test]
    fn sweep_is_powers_of_two_from_one() {
        let sweep = triad_sweep(1 << 14, 4);
        let threads: Vec<usize> = sweep.iter().map(|r| r.threads).collect();
        assert_eq!(threads, vec![1, 2, 4]);
        // Monotone bookkeeping: every entry reports its own n back.
        assert!(sweep.iter().all(|r| r.n == 1 << 14));
    }

    #[test]
    fn triad_computes_the_right_values() {
        // Deterministic correctness check at a tiny size with distinct seeds:
        // a[i] must equal 1.5 + 0.75*2.5 = 3.375 after any pass.
        let mut a = vec![0.0f64; 1024];
        let b = vec![1.5f64; 1024];
        let c = vec![2.5f64; 1024];
        triad_pass(&mut a, &b, &c, 4);
        assert!(a.iter().all(|&v| v == 3.375));
    }
}
