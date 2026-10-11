//! Regression tests for the external soundness/validity audit.
//!
//! Each test pins one audit finding so a future refactor cannot silently
//! reintroduce it. They run against the real JIT (helix-backend is a hard
//! dependency of this crate), so a `native_variant` construction failure here
//! is a real build break, not a skipped assertion.
//!
//! The JIT host heap, the trap recorder and the runtime's dispatch record are
//! process-global, and cargo runs a binary's tests on parallel threads, so
//! every test that compiles or executes native code holds
//! [`helix_backend::testutil::serial_lock`]. Without it one thread's heap
//! reset frees another thread's live arrays.

use helix_bench::{
    BenchConfig, ExecVariant, NativeVariant, interp_variant, kernels, rust_twins, selected_sizes,
};

/// A saxpy-shaped kernel with a NON-TRIVIAL final array: the interpreter folds
/// `x` and `y` into the array half of its checksum, so a checksum that matched
/// printed lines alone would be visibly different from the interpreter's.
const SAXPY_LIKE: &str = r#"
fn main() {
    let n = 64;
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

/// Seeded integer reductions: a sum and a product over 65 536 elements. Integer
/// wraparound arithmetic is associative, so every participant count must
/// produce the identical printed value.
const SEEDED_REDUCTIONS: &str = r#"
fn main() {
    let n = 65536;
    let a: [i64] = zeros(n);
    for i in 0..n {
        a[i] = ((i * 31 + 7) % 97) as i64;
    }
    let s = 17;
    for i in 0..n {
        s = s + a[i];
    }
    print(s);
    let p = 3;
    for i in 0..n {
        p = p * a[i];
    }
    print(p);
}
"#;

/// Logical CPUs visible to this process (1 when the OS will not say).
fn hardware() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// A registered kernel by name (panics with the suite's names when absent).
fn kernel(name: &str) -> kernels::KernelDef {
    kernels::registry()
        .into_iter()
        .find(|k| k.name == name)
        .unwrap_or_else(|| panic!("{name} kernel is registered"))
}

// ---------------------------------------------------------------------------
// Finding 1: "native-seq" must really be sequential
// ---------------------------------------------------------------------------

/// The sequential baseline must be compiled from an EMPTY backend plan, so a
/// parallelizable kernel cannot run parallel inside the baseline and make
/// every speedup measured against it wrong.
#[test]
fn native_seq_compiles_with_an_empty_plan() {
    let _guard = helix_backend::testutil::serial_lock();
    let src = kernel("scale").source_at_size(1_000);

    let seq = NativeVariant::new(&src, false).expect("native-seq compiles");
    assert_eq!(
        seq.region_count(),
        0,
        "native-seq must compile with an empty parallel plan — otherwise it \
         labels a parallel run as sequential"
    );
    // Truly no regions means the engine name stays the baseline's.
    assert_eq!(seq.name(), "native-seq");

    // The same source, compiled from the real analyzer plan, DOES carry
    // regions: the contrast is the point of the assertion.
    let par = NativeVariant::new_parallel(&src, 2).expect("native-par compiles");
    assert!(
        par.region_count() > 0,
        "native-par on a DOALL kernel must compile with real regions"
    );
}

/// A sequential-only program has no regions under either build, so the empty
/// plan is not hiding a difference for non-parallelizable sources.
#[test]
fn native_seq_parallel_agree_when_nothing_is_parallelizable() {
    let _guard = helix_backend::testutil::serial_lock();
    let src = r#"
        fn main() {
            let a: [i64] = zeros(4);
            a[0] = 7;
            print(a[0]);
        }
    "#;
    assert_eq!(NativeVariant::new(src, false).unwrap().region_count(), 0);
    assert_eq!(
        NativeVariant::new_parallel(src, 4).unwrap().region_count(),
        0
    );
}

// ---------------------------------------------------------------------------
// Finding 2: the region hint must not cap the thread sweep
// ---------------------------------------------------------------------------

/// The engine's thread hint is a ceiling, never the governor: the requested
/// participant count is `min(request, hardware)`, so a sweep row labelled `p`
/// really runs `p` participants (clamped only by what the machine has).
#[test]
fn p_sixteen_pins_sixteen_participants() {
    // One engine per pin: the runtime reads HELIX_NTHREADS at dispatch time.
    // The whole loop is inside the lock so the dispatch record read at the end
    // of each iteration really describes that iteration's run.
    let _guard = helix_backend::testutil::serial_lock();
    let hw = hardware();

    for p in [1_usize, 2, 4, 8, 16] {
        let variant =
            NativeVariant::new_parallel(SEEDED_REDUCTIONS, p).expect("native-par compiles");
        let out = variant.run_once().expect("run");
        let dispatch = helix_runtime::take_last_dispatch()
            .expect("a region dispatched, so the runtime recorded it");
        assert_eq!(
            dispatch.participants,
            p.min(hw),
            "requested p={p} on {hw}-way hardware must resolve to {} participants",
            p.min(hw)
        );
        assert_eq!(
            dispatch.participants,
            want_participants(p, hw),
            "the requested count is what a sweep row records"
        );
        assert_eq!(out.printed.len(), 2, "both reductions printed");
    }
}

/// The participant count a requested sweep value must resolve to: the request,
/// clamped by hardware only. The engine's baked-in hint is a ceiling far above
/// any host, so it never appears in the result.
fn want_participants(requested: usize, hw: usize) -> usize {
    requested.min(hw)
}

// ---------------------------------------------------------------------------
// Finding 3: --quick is quick
// ---------------------------------------------------------------------------

/// Quick mode keeps only the smallest declared size per kernel, and a kernel
/// that declares no size at all fails loudly instead of contributing nothing.
#[test]
fn quick_selects_the_smallest_size_only() {
    let config = BenchConfig {
        quick: true,
        ..BenchConfig::default()
    };
    let mut checked = 0_usize;
    for registered in kernels::registry() {
        let smallest = registered
            .sizes
            .iter()
            .min()
            .expect("registry kernel has sizes");
        let sizes = selected_sizes(&registered, &config).expect("quick selection succeeds");
        assert_eq!(
            sizes,
            vec![*smallest],
            "{}: quick mode must measure only the smallest declared size",
            registered.name
        );
        assert!(
            sizes.len() <= 1,
            "quick mode must collapse each kernel to one point"
        );
        checked += 1;
    }
    assert!(checked >= 8, "the kernel suite should be exercised");

    // The same call in full mode returns the whole sweep untouched.
    let full = BenchConfig::default();
    let saxpy = kernel("saxpy");
    assert_eq!(selected_sizes(&saxpy, &full).unwrap(), saxpy.sizes.to_vec());

    // A kernel with no sizes at all has no smaller size to reduce to.
    let empty = kernels::KernelDef {
        name: "size_less",
        description: "registry mistake: no size sweep",
        perf_source: String::new(),
        correctness_source: String::new(),
        sizes: &[],
        expected_verdict: kernels::ExpectedVerdict::NotApplicable,
        tolerance: kernels::Tolerance::Exact,
        expected_printed: &[],
        correctness_only: false,
        interp_max_size: 0,
    };
    assert!(
        selected_sizes(&empty, &config).is_err(),
        "--quick must fail loudly when a kernel has no smaller size to select"
    );
}

/// Quick mode also shrinks the participant sweep to {1, hardware} and the
/// sample count, without touching the measurement procedure itself.
#[test]
fn quick_shrinks_the_sweep_and_the_sample_count() {
    let full = BenchConfig::default();
    assert_eq!(full.samples(), helix_bench::timing::K_SAMPLES);
    assert_eq!(
        full.effective_thread_sweep(),
        vec![1, 2, 4, 8, 16, 24, 32],
        "a full campaign keeps its sweep untouched"
    );

    let quick = BenchConfig {
        quick: true,
        ..BenchConfig::default()
    };
    assert_eq!(quick.samples(), helix_bench::timing::K_SAMPLES_QUICK);
    let hw = hardware();
    assert_eq!(
        quick.effective_thread_sweep(),
        if hw > 1 { vec![1, hw] } else { vec![1] },
        "quick mode sweeps 1 and the hardware width and nothing else"
    );
}

/// A quick campaign produces the same point shape as a full one at the
/// smallest size, with fewer samples per variant, and it runs the shared
/// `run_campaign` path rather than a shortcut of its own.
#[test]
fn quick_campaign_runs_the_shared_path_with_fewer_samples() {
    let _guard = helix_backend::testutil::serial_lock();
    let quick = BenchConfig {
        quick: true,
        skip_triad: true,
        only_kernels: vec!["small_n".to_string()],
        ..BenchConfig::default()
    };
    let report = helix_bench::run_campaign(&quick);
    let point = report
        .points
        .iter()
        .find(|p| p.kernel == "small_n")
        .expect("small_n is measured");
    // One point per kernel, at the smallest declared size.
    assert_eq!(report.points.len(), 1);
    assert_eq!(point.n, 1_000);
    assert!(
        point
            .variants
            .iter()
            .all(|v| v.samples_ms.len() == helix_bench::timing::K_SAMPLES_QUICK),
        "quick mode must take K_SAMPLES_QUICK samples per variant"
    );
    assert!(
        point.variants.iter().all(|v| v.compile_ms >= 0.0),
        "compile time is serialized for every variant"
    );
    // The JSON carries the new field so offline recomputation can see it.
    let json = serde_json::to_string(&point.variants[0]).expect("serialize");
    assert!(json.contains("\"compile_ms\""), "compile_ms must serialize");
}

// ---------------------------------------------------------------------------
// Finding 4: compile time is measured and reported
// ---------------------------------------------------------------------------

/// `native_variant` must not measure compile time as execution time.
#[test]
fn native_variants_report_their_own_compile_time() {
    let _guard = helix_backend::testutil::serial_lock();
    let src = SEEDED_REDUCTIONS;

    let interp = interp_variant(src).expect("interp compiles");
    assert_eq!(
        interp.compile_ms(),
        0.0,
        "the interpreter has no compile step to report"
    );

    for (label, variant) in [
        (
            "native-seq",
            NativeVariant::new(src, false).expect("native-seq compiles"),
        ),
        (
            "native-par",
            NativeVariant::new_parallel(src, 4).expect("native-par compiles"),
        ),
    ] {
        let compile_ms = variant.compile_ms();
        assert!(
            compile_ms.is_finite() && compile_ms > 0.0,
            "{label}: compile time must be measured, got {compile_ms}"
        );
        // The trait accessor the campaign serializes carries the same number.
        assert_eq!(ExecVariant::compile_ms(&variant), compile_ms);
    }
}

// ---------------------------------------------------------------------------
// Finding 5: the parallel-path checksum must cover final arrays
// ---------------------------------------------------------------------------

/// The interpreter's checksum folds main's final arrays; the native paths must
/// produce the SAME stream, not a printed-lines-only one.
#[test]
fn native_checksum_covers_final_arrays_like_the_interpreter() {
    let _guard = helix_backend::testutil::serial_lock();
    let oracle = interp_variant(SAXPY_LIKE)
        .expect("interp compiles")
        .run_once()
        .expect("interpreter runs");

    let seq_out = NativeVariant::new(SAXPY_LIKE, false)
        .expect("native-seq compiles")
        .run_once()
        .expect("native-seq runs");
    assert_eq!(
        seq_out.checksum, oracle.checksum,
        "native-seq must match the interpreter's FULL state checksum (printed \
         lines + final arrays)"
    );
    assert_eq!(seq_out.printed, oracle.printed);

    // DOALL parallel results are bit-exact, so the parallel build matches too.
    let par_out = NativeVariant::new_parallel(SAXPY_LIKE, 2)
        .expect("native-par compiles")
        .run_once()
        .expect("native-par runs");
    assert_eq!(par_out.checksum, oracle.checksum);
    assert_eq!(par_out.printed, oracle.printed);

    // The guard that makes the above assertions meaningful: folding ONLY the
    // printed lines yields a DIFFERENT number, so a regression to the old
    // printed-only checksum cannot pass this test.
    let printed_only = rust_twins::printed_checksum(&oracle.printed);
    assert_ne!(
        oracle.checksum, printed_only,
        "this kernel's final arrays must actually contribute to the checksum"
    );
}

/// A program with NO final arrays folds the printed stream and nothing else,
/// which is still the interpreter's checksum.
#[test]
fn native_checksum_matches_when_main_leaves_no_arrays() {
    let _guard = helix_backend::testutil::serial_lock();
    let src = r#"
        fn main() {
            let n = 16;
            let mut_total = 0;
            for i in 1..n {
                mut_total = mut_total + i;
            }
            print(mut_total);
        }
    "#;
    let oracle = interp_variant(src).unwrap().run_once().unwrap();
    let seq = NativeVariant::new(src, false).unwrap();
    assert_eq!(seq.run_once().unwrap().checksum, oracle.checksum);
}

/// Two programs that print the same lines but leave different final arrays
/// must NOT checksum alike — otherwise the array half of the stream proves
/// nothing. (Interpreter-only: no JIT state is touched.)
#[test]
fn different_final_arrays_produce_different_checksums() {
    let a = interp_variant(SAXPY_LIKE).unwrap().run_once().unwrap();
    let mutated = SAXPY_LIKE.replace("y[i] = s * x[i] + y[i];", "y[i] = s * x[i] + y[i] + 1.0;");
    let b = interp_variant(&mutated).unwrap().run_once().unwrap();
    assert_ne!(
        a.checksum, b.checksum,
        "a program whose final arrays differ must checksum differently"
    );
}

/// The full-state checksum rule the campaign's parity gate enforces, applied to
/// every registered kernel: `native-seq` must reproduce the interpreter's exact
/// full-state checksum, and `native-par` must too when the kernel documents no
/// reassociation slack. A kernel with a nonzero documented tolerance (`dot`,
/// `matmul`) is exempt from the parallel checksum comparison only.
#[test]
fn full_state_parity_holds_across_the_registry() {
    let _guard = helix_backend::testutil::serial_lock();
    for kernel in kernels::registry() {
        let src = &kernel.correctness_source;
        let interp = interp_variant(src)
            .unwrap_or_else(|e| panic!("{}: interp compile: {e}", kernel.name))
            .run_once()
            .unwrap_or_else(|e| panic!("{}: interp run: {e}", kernel.name));

        let seq = NativeVariant::new(src, false)
            .unwrap_or_else(|e| panic!("{}: native-seq compile: {e}", kernel.name))
            .run_once()
            .unwrap_or_else(|e| panic!("{}: native-seq run: {e}", kernel.name));
        assert_eq!(
            seq.checksum, interp.checksum,
            "{}: native-seq full-state checksum {:016x} != interpreter {:016x}",
            kernel.name, seq.checksum, interp.checksum
        );
        assert_eq!(
            seq.printed, interp.printed,
            "{}: printed lines",
            kernel.name
        );

        let par = NativeVariant::new_parallel(src, 2)
            .unwrap_or_else(|e| panic!("{}: native-par compile: {e}", kernel.name))
            .run_once()
            .unwrap_or_else(|e| panic!("{}: native-par run: {e}", kernel.name));
        if kernel.tolerance == kernels::Tolerance::Exact {
            assert_eq!(
                par.checksum, interp.checksum,
                "{}: native-par full-state checksum {:016x} != interpreter {:016x}",
                kernel.name, par.checksum, interp.checksum
            );
        }
        assert_eq!(par.printed.len(), interp.printed.len(), "{}", kernel.name);
    }
}

/// A program whose final arrays the JIT host heap cannot witness (a callee
/// allocates too, so the heap holds more buffers than `main` binds) must report
/// that instead of folding a stream that is not the interpreter's.
#[test]
fn unwitnessable_final_state_is_reported_not_guessed() {
    let _guard = helix_backend::testutil::serial_lock();
    let src = r#"
        fn spawn() -> i64 {
            let t: [i64] = zeros(4);
            t[0] = 5;
            return t[0];
        }
        fn main() {
            let a: [i64] = zeros(4);
            a[0] = spawn();
            print(a[0]);
        }
    "#;
    let variant = NativeVariant::new(src, false).expect("native-seq compiles");
    let err = variant
        .run_once()
        .expect_err("no checksum can be witnessed");
    assert!(
        err.contains("cannot witness final array state"),
        "the failure must name the reason, got: {err}"
    );
}

// ---------------------------------------------------------------------------
// Finding 2 (corollary): integer reductions stay exact across thread counts
// ---------------------------------------------------------------------------

/// A seeded reduction is order-independent under exact integer arithmetic, so
/// every participant count must print the identical value. If the parallel
/// runtime ever mis-combines reduction cells this fails.
#[test]
fn seeded_reductions_stay_exact_across_thread_counts() {
    let _guard = helix_backend::testutil::serial_lock();
    let mut results: Vec<(usize, Vec<String>)> = Vec::new();
    for p in [1_usize, 2, 4, 8] {
        let variant = NativeVariant::new_parallel(SEEDED_REDUCTIONS, p)
            .unwrap_or_else(|e| panic!("native-par{p} failed to compile: {e}"));
        let out = variant.run_once().expect("run");
        results.push((p, out.printed));
    }
    let (first_p, expected) = &results[0];
    for (p, printed) in &results {
        assert_eq!(
            printed, expected,
            "integer reductions must be bit-exact at every participant count \
             (p={p} disagrees with p={first_p})"
        );
    }
    // And the value is the interpreter's, not merely self-consistent.
    let interp = interp_variant(SEEDED_REDUCTIONS)
        .unwrap()
        .run_once()
        .unwrap();
    assert_eq!(&interp.printed, expected);
}
