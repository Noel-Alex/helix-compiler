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
    let d = v.time_batch(3);
    assert_eq!(v.runs.get(), 4); // 1 explicit + 3 in the batch
    assert!(d.as_nanos() < 10_000_000); // trivial work, no sleeps
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
fn native_variant_reports_unavailability_without_panicking() {
    match helix_bench::native_availability() {
        NativeAvailability::Ready => {
            // Backend landed: the stub must be gone. This branch keeps the
            // test honest after M10 instead of asserting nothing.
            panic!(
                "bench-native is now available; promote native_variant to real JIT construction"
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
