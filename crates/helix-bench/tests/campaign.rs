//! Runnable (non-ignored) integration tests: everything the harness guarantees
//! WITHOUT the JIT backend — campaign JSON shape, parity gates, variant
//! isolation, and the ExecVariant contract.

use helix_bench::{
    BenchConfig, ExecVariant, NativeAvailability, RunOutputLike, interp_variant, kernels,
    parity_holds, run_campaign,
};

/// A toy ExecVariant used to prove the harness only depends on the trait.
struct Counting {
    tag: String,
    runs: std::cell::Cell<u32>,
}

impl ExecVariant for Counting {
    fn name(&self) -> &str {
        &self.tag
    }

    fn run_once(&self) -> Result<RunOutputLike, String> {
        self.runs.set(self.runs.get() + 1);
        Ok(RunOutputLike::new(vec!["42".into()], 7))
    }
}

#[test]
fn every_kernel_passes_the_parity_gate() {
    for k in kernels::registry() {
        assert!(
            parity_holds(&k),
            "{} failed its correctness gate (expected {:?})",
            k.name,
            k.expected_printed
        );
    }
}

#[test]
fn exec_variant_is_the_only_backend_dependency() {
    // The harness must drive ANY implementation of the trait identically.
    let v = Counting {
        tag: "toy".to_string(),
        runs: std::cell::Cell::new(0),
    };
    assert_eq!(v.name(), "toy");
    let out = v.run_once().unwrap();
    assert_eq!(out.printed, vec!["42"]);
    let d = v.time_batch(3).unwrap();
    assert_eq!(v.runs.get(), 4); // 1 explicit + 3 in the batch
    assert!(d.as_nanos() < 10_000_000); // trivial work, no sleeps
}

/// A variant whose executions always fail — the harness must surface its
/// error from time_batch, never swallow it into a Duration.
struct Failing {
    runs: std::cell::Cell<u32>,
}

impl ExecVariant for Failing {
    fn name(&self) -> &str {
        "failing"
    }

    fn run_once(&self) -> Result<RunOutputLike, String> {
        self.runs.set(self.runs.get() + 1);
        Err("runtime trap code=13".to_string())
    }
}

#[test]
fn timed_repetitions_propagate_execution_errors() {
    // Bug P1-7 regression: time_batch used to discard run_once results, so
    // hundreds of timed reps could execute crashing code and produce a
    // plausible-looking Duration.
    let v = Failing {
        runs: std::cell::Cell::new(0),
    };
    let err = v.time_batch(5).unwrap_err();
    assert_eq!(err, "runtime trap code=13");
    // FIRST failure aborts the batch (not all 5 reps).
    assert_eq!(v.runs.get(), 1);
}

#[test]
fn interp_variant_compiles_and_executes_isolated_units() {
    let a = interp_variant(kernels::registry()[0].perf_source.as_str());
    // Perf-size sources compile fine too (we never RUN them here).
    assert!(
        a.is_ok(),
        "perf source failed to compile: {}",
        a.err().unwrap_or_default()
    );

    // Two variants of the same program are independent compilation units:
    // mutating one cannot affect the other.
    let src = r#"
        fn main() {
            let x: [i64] = zeros(2);
            x[0] = 9;
            print(x[0]);
        }
    "#;
    let v1 = interp_variant(src).unwrap();
    let v2 = interp_variant(src).unwrap();
    assert_eq!(
        v1.run_once().unwrap().printed,
        v2.run_once().unwrap().printed
    );
}

#[test]
fn native_variant_construction_matches_interp() {
    match helix_bench::native_availability() {
        NativeAvailability::Ready => {
            // Real JIT construction: tiny program must run and agree with the
            // interpreter line-for-line.
            let src = r#"
                fn main() {
                    let n = 1000;
                    let a: [f64] = zeros(n);
                    let out: [f64] = zeros(n);
                    for i in 0..n {
                        out[i] = a[i] + 1.0;
                    }
                    print(out[999]);
                }
            "#;
            let interp = helix_bench::interp_variant(src).expect("interp");
            let native = helix_bench::native_variant(src).expect("native");
            assert_eq!(
                interp.run_once().unwrap().printed,
                native.run_once().unwrap().printed
            );
        }
        NativeAvailability::Unavailable(_) => {
            assert!(helix_bench::native_variant("fn main() {}").is_err());
        }
    }
}

#[test]
fn campaign_json_shape_round_trips() {
    let config = BenchConfig {
        thread_sweep: vec![1],
        max_threads: 1,
        skip_triad: true,
        triad_n: 1024,
        only_kernels: vec!["small_n".to_string()],
    };
    let report = run_campaign(&config);

    assert_eq!(report.schema, "helix-campaign-v1");
    assert!(!report.meta.rustc_version.is_empty());
    assert!(report.triad_ceilings.is_empty());
    assert!(report.triad_ceiling.is_none());

    let point = &report.points[0];
    assert_eq!(point.kernel, "small_n");
    assert_eq!(point.n, 1000);
    assert!(!point.variants.is_empty());

    let v = &point.variants[0];
    assert_eq!(v.name, "interp");
    assert_eq!(
        v.samples_ms.len(),
        helix_bench::timing::K_SAMPLES,
        "raw samples must be preserved for offline recomputation"
    );
    assert_eq!(v.reps_per_sample, point.variants[0].reps_per_sample);

    // Speedup vs itself == 1.
    assert!((point.speedups[0] - 1.0).abs() < 1e-9);
    // Thread sweep with p={1} yields exactly one efficiency row; speedup is
    // measured-vs-measured so it is ~1 but not bit-exact (two samplings).
    assert_eq!(point.efficiency.len(), 1);
    assert!(
        (point.efficiency[0].speedup - 1.0).abs() < 0.05,
        "p=1 self-speedup drifted: {}",
        point.efficiency[0].speedup
    );
    assert!((point.efficiency[0].efficiency - point.efficiency[0].speedup).abs() < 1e-12);

    // JSON round-trip preserves everything serde cares about.
    let dir = std::env::temp_dir().join(format!("helix-bench-test-{}", std::process::id()));
    let path = dir.join("nested").join("campaign.json");
    helix_bench::write_json(&report, &path).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(text.contains("\"schema\": \"helix-campaign-v1\""));
    assert!(text.contains("\"samples_ms\""));
    let back = helix_bench::read_json(&path).unwrap();
    assert_eq!(back.points.len(), report.points.len());
    assert_eq!(back.meta.cpu_brand, report.meta.cpu_brand);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn campaign_publishes_triad_at_one_thread_and_hw_width() {
    // Regression (adversarial review): the ceiling used to be measured at
    // 1 thread only yet consumed as THE machine bandwidth denominator. It is
    // now one row per measured width: always threads=1, plus the hardware
    // width when more than one core exists.
    let config = BenchConfig {
        thread_sweep: vec![1],
        max_threads: 1,
        skip_triad: false,
        triad_n: 1 << 14,
        only_kernels: vec!["small_n".to_string()],
    };
    let report = run_campaign(&config);
    assert!(!report.triad_ceilings.is_empty(), "triad must be measured");
    assert_eq!(report.triad_ceilings[0].threads, 1);
    let hw = std::thread::available_parallelism().map_or(1, |n| n.get());
    if hw > 1 {
        assert_eq!(report.triad_ceilings.len(), 2);
        assert_eq!(report.triad_ceilings[1].threads, hw);
        // No cross-row speed assertion here ON PURPOSE: at unit-test sizes
        // spawn/join overhead legitimately dwarfs the transfer, so the wide
        // row may honestly be slower. The fix under test is that BOTH widths
        // get measured and published, not which one wins.
    }
    assert!(report
        .triad_ceilings
        .iter()
        .all(|t| t.gib_per_sec.is_finite() && t.gib_per_sec > 0.0));
    // Legacy field mirrors row 0 so old readers keep working.
    assert_eq!(
        report.triad_ceiling.map(|t| t.gib_per_sec),
        report.triad_ceilings.first().map(|t| t.gib_per_sec)
    );
}

#[test]
fn read_json_accepts_legacy_single_row_triad_schema() {
    // Old campaign.json files carry only `triad_ceiling`; the reader must
    // accept them (serde defaults fill `triad_ceilings` with nothing).
    let legacy = r#"{
        "schema": "helix-campaign-v1",
        "timestamp_utc": "2026-08-25T00:00:00Z",
        "meta": {
            "cpu_brand": "Test CPU",
            "logical_cores": 8,
            "physical_cores": null,
            "total_ram_bytes": 17179869184,
            "os": "Windows 11 Home",
            "rustc_version": "rustc 1.93.0",
            "git_rev": null,
            "timestamp_utc": "2026-08-25T00:00:00Z",
            "helix_nthreads_env": null,
            "helix_runtime_env": null,
            "jit_available": true
        },
        "triad_ceiling": { "threads": 1, "gib_per_sec": 33.5 },
        "points": []
    }"#;
    let dir = std::env::temp_dir().join(format!("helix-bench-legacy-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("legacy.json");
    std::fs::write(&path, legacy).unwrap();
    let back = helix_bench::read_json(&path).unwrap();
    assert_eq!(back.triad_ceiling.unwrap().gib_per_sec, 33.5);
    assert!(back.triad_ceilings.is_empty(), "legacy file has no new field");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn campaign_includes_expected_verdict_labels() {
    let config = BenchConfig {
        skip_triad: true,
        triad_n: 1024,
        ..BenchConfig::default()
    };
    // Restrict to one fast kernel so CI stays quick.
    let config = BenchConfig {
        only_kernels: vec!["small_n".to_string(), "fib_recursion".to_string()],
        ..config
    };
    let report = run_campaign(&config);
    // fib is correctness-only => excluded from points.
    assert!(report.points.iter().all(|p| p.kernel != "fib_recursion"));
    let small_n = report
        .points
        .iter()
        .find(|p| p.kernel == "small_n")
        .unwrap();
    assert_eq!(small_n.expected_verdict, "SafeParallel");
}

// ---------------------------------------------------------------------------
// Rust-twin differential guarantee (wave-3 regression)
// ---------------------------------------------------------------------------

/// The module docs advertise `twin_checksum == interpreter_checksum` as a
/// real differential assertion. This test makes it one: every twin's
/// independently computed result must hash to the same checksum the HELIX
/// interpreter prints for the corresponding correctness-size kernel.
#[test]
fn rust_twins_match_interpreter_checksums() {
    use helix_bench::interp_variant;
    use helix_bench::rust_twins;
    use helix_bench::rust_twins::printed_checksum;

    // saxpy: twin inputs mirror kernels.rs SAXPY_SRC's seeded init
    // (sign/fraction mix), s=2.5; the HELIX program below re-derives the
    // same arrays from the same formulas, so checksum equality is a real
    // differential over nontrivial data.
    let y = rust_twins::saxpy(8, 2.5);
    let twin_saxpy = rust_twins::checksum_f64(y[7]);
    let saxpy_src = r#"
        fn main() {
            let n = 8;
            let x: [f64] = zeros(n);
            let y: [f64] = zeros(n);
            for i in 0..n {
                x[i] = ((i * 17 + 3) % 251) as f64 / 17.0 - 4.0;
                y[i] = ((i * 29 + 11) % 241) as f64 / 19.0 - 5.0;
            }
            let s = 2.5;
            for i in 0..n {
                y[i] = s * x[i] + y[i];
            }
            print(y[7]);
        }
    "#;
    let v = interp_variant(saxpy_src).expect("compile");
    let out = v.run_once().expect("run");
    assert_eq!(
        printed_checksum(&out.printed),
        twin_saxpy,
        "saxpy twin vs interpreter checksum"
    );

    // dot: twin inputs mirror DOT_SRC's seeded pair at n=16 (cancelling mix).
    let (a, b) = rust_twins::dot_inputs(16);
    let twin_dot = rust_twins::checksum_f64(rust_twins::dot(&a, &b));
    let dot_src = r#"
        fn main() {
            let n = 16;
            let a: [f64] = zeros(n);
            let b: [f64] = zeros(n);
            for i in 0..n {
                a[i] = ((i * 7 + 1) % 97) as f64 / 9.0 - 4.0;
                b[i] = ((i * 11 + 2) % 89) as f64 / 11.0 - 3.0;
            }
            let d = 0.0;
            for i in 0..n {
                d = d + a[i] * b[i];
            }
            print(d);
        }
    "#;
    let v = interp_variant(dot_src).expect("compile");
    let out = v.run_once().expect("run");
    assert_eq!(
        printed_checksum(&out.printed),
        twin_dot,
        "dot twin vs interpreter checksum"
    );

    // scale: twin inputs mirror SCALE_SRC's seeded init, multiplier 5.0; both
    // sides check element 10 of an n=16 run.
    let twin_scale = rust_twins::checksum_f64(rust_twins::scale_inputs(16)[10] * 5.0);
    let scale_src = r#"
        fn main() {
            let n = 16;
            let a: [f64] = zeros(n);
            let out: [f64] = zeros(n);
            for i in 0..n {
                a[i] = ((i * 13 + 5) % 199) as f64 / 7.0 - 12.0;
            }
            for i in 0..n {
                out[i] = a[i] * 5.0;
            }
            print(out[10]);
        }
    "#;
    let v = interp_variant(scale_src).expect("compile");
    let out = v.run_once().expect("run");
    assert_eq!(
        printed_checksum(&out.printed),
        twin_scale,
        "scale twin vs interpreter checksum"
    );

    // matmul: twin recomputes c[centre] with IDENTICAL i-j-k summation order,
    // so FP rounding must match the interpreted program bit-for-bit — the
    // strongest form of the advertised guarantee (exact checksum equality on
    // a reduction).
    let twin_matmul = rust_twins::checksum_f64(rust_twins::matmul_centre(8));
    let v = interp_variant(
        kernels::registry()
            .iter()
            .find(|k| k.name == "matmul")
            .unwrap()
            .correctness_source
            .as_str(),
    )
    .expect("compile");
    let out = v.run_once().expect("run");
    assert_eq!(
        printed_checksum(&out.printed),
        twin_matmul,
        "matmul twin vs interpreter checksum"
    );
}
