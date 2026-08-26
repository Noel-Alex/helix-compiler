//! # helix-bench — campaign harness for the HELIX compiler
//!
//! Produces the report's numbers: a hyperfine-style JSON campaign over the
//! kernel suite (see [`kernels`]) comparing every execution variant, plus the
//! environment metadata ([`meta`]), STREAM-triad ceiling ([`triad`]) and
//! hand-written Rust twins ([`rust_twins`]) that make those numbers
//! defensible. Methodology: `docs/research/benchmark-methodology.md`.
//!
//! ```text
//! run_campaign(config)
//!   ├─ meta::SystemMeta::capture()          provenance
//!   ├─ triad::measure_triad()               bandwidth ceiling
//!   └─ per kernel × size:
//!        build ExecVariants (interp now; JIT when backend present)
//!        parity check at correctness size  (tolerance-gated)
//!        timing::run_interleaved()         round-robin sampling
//!        efficiency table from thread sweep
//!   └─ CampaignReport → write_json()
//! ```
//!
//! ## Isolation from in-flight backends
//!
//! The JIT side is being built **in parallel**; this crate compiles and tests
//! without it. All execution paths hide behind [`ExecVariant`] and are
//! obtained through [`native_variant`] (always available — helix-backend is
//! a hard dependency). The full pipeline integration test lives
//! behind `#[ignore]` (`tests/integration_full.rs`) and runs later with
//! `cargo test -p helix-bench -- --ignored`.
//!
//! ## Thread sweep
//!
//! Parallel candidates sweep p ∈ {1,2,4,8,16,24,32} (methodology rec. 9) by
//! setting `HELIX_NTHREADS` around each measurement — the same override
//! helix-runtime honours — clamped to available hardware. Efficiency
//! E_p = S_p / p is tabulated next to raw speedup so inefficiency cannot
//! hide (pitfall 12).

pub mod kernels;
pub mod meta;
pub mod rust_twins;
pub mod timing;
pub mod triad;

use serde::{Deserialize, Serialize};
use std::path::Path;

pub use kernels::{ExpectedVerdict, KernelDef, RunOutputLike, Tolerance};
pub use meta::SystemMeta;
pub use timing::Measurement;
pub use triad::TriadResult;

/// Campaign-wide configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchConfig {
    /// Thread counts swept for parallel candidates (p values).
    pub thread_sweep: Vec<usize>,
    /// Cap total threads regardless of sweep contents.
    pub max_threads: usize,
    /// Skip the triad ceiling measurement (unit tests do this).
    pub skip_triad: bool,
    /// Triad vector length (reduced in tests).
    pub triad_n: usize,
    /// Kernels to run; empty = all non-correctness-only entries.
    pub only_kernels: Vec<String>,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            thread_sweep: vec![1, 2, 4, 8, 16, 24, 32],
            max_threads: 32,
            skip_triad: false,
            triad_n: triad::DEFAULT_TRIAD_N,
            only_kernels: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Execution variants
// ---------------------------------------------------------------------------

/// One executable form of a kernel program: an isolated compilation unit the
/// harness can warm up, checksum, and time without knowing which backend
/// produced it.
///
/// Implementations own their compiled state (interpreter trees, JIT engines,
/// registered body pointers) so dropping the variant drops the code. The
/// trait is object-safe to let campaigns mix interpreter and JIT variants in
/// one interleaved round.
pub trait ExecVariant {
    /// Stable identifier used in JSON (`"interp"`, `"native-seq"`, …).
    fn name(&self) -> &str;

    /// Runs the program once; returns printed lines + checksum or the error
    /// text. Must be deterministic for identical inputs (the harness compares
    /// outputs across variants under the kernel's [`Tolerance`]).
    ///
    /// # Errors
    /// Backend-specific failure text (parse/sema/runtime), rendered readably.
    fn run_once(&self) -> Result<RunOutputLike, String>;

    /// Time `reps` consecutive executions as one batch (sampler hook).
    ///
    /// Default: loop [`run_once`](Self::run_once) and take wall-clock.
    /// Backends with cheaper re-invocation can override.
    ///
    /// # Errors
    /// The FIRST failed execution aborts the batch and is returned verbatim:
    /// timed repetitions of code that crashes are meaningless, so the batch
    /// must never silently swallow a mid-run failure.
    fn time_batch(&self, reps: u32) -> Result<std::time::Duration, String> {
        let start = std::time::Instant::now();
        for _ in 0..reps {
            std::hint::black_box(self.run_once()?);
            // Reclaim JIT-host allocations (each run's zeros(n) arrays would
            // otherwise accumulate across hundreds of timed samples).
            helix_backend::reset_host_heap();
        }
        Ok(start.elapsed())
    }
}

/// Builds the reference interpreter variant for one program.
///
/// Parse + sema happen once here; each `run_once` re-executes the checked
/// tree through `helix_engine::run_with_source` (source kept for line-numbered
/// runtime errors).
///
/// # Errors
/// Propagates parse/sema diagnostics as text.
pub fn interp_variant(src: &str) -> Result<InterpVariant, String> {
    InterpVariant::new(src)
}

/// Interpreter-backed [`ExecVariant`].
pub struct InterpVariant {
    src: String,
    typed: helix_sema::TypedProgram,
}

impl InterpVariant {
    /// Compiles (parse + check) once up front.
    ///
    /// # Errors
    /// Parse or semantic diagnostics, rendered as text.
    pub fn new(src: &str) -> Result<Self, String> {
        let ast = helix_syntax::parse_str(src).map_err(|e| format!("syntax error: {e}"))?;
        let typed = helix_sema::check(&ast).map_err(|ds| format!("{ds:#?}"))?;
        Ok(Self {
            src: src.to_string(),
            typed,
        })
    }

    /// One full interpretation.
    ///
    /// # Errors
    /// Rendered runtime error (bounds/div/stack).
    pub fn execute(&self) -> Result<RunOutputLike, String> {
        helix_engine::run_with_source(&self.src, &self.typed)
            .map(|out| RunOutputLike::from_engine(&out))
            .map_err(|e| e.render(&self.src))
    }
}

impl ExecVariant for InterpVariant {
    fn name(&self) -> &str {
        "interp"
    }

    fn run_once(&self) -> Result<RunOutputLike, String> {
        self.execute()
    }
}

/// What [`variant_factory`] reports about the native (JIT) path on this host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeAvailability {
    /// `helix_backend` linked in and self-reports functional.
    Ready,
    /// No working backend compiled into this binary; native variants are
    /// skipped and the campaign proceeds interpreter-only.
    Unavailable(&'static str),
}

/// Probes whether the JIT variant can be built right now.
///
/// The backend is a hard dependency of this crate (helix-backend is linked
/// unconditionally), so availability is always `Ready`; per-program failures
/// surface through [`native_variant`]'s error return instead. (A dead
/// `bench-native` cargo feature used to gate this and made
/// `meta.jit_available` report Unavailable in every default build — the
/// opposite of reality.)
#[must_use]
pub fn native_availability() -> NativeAvailability {
    NativeAvailability::Ready
}

/// Builds the sequential-native variant for one program, if possible.
///
/// Full pipeline: parse → check → IR build → SSA → loop analysis → parallel
/// plan → Cranelift compile. Each `run_once` executes the machine code once
/// (prints captured host-side).
///
/// # Errors
/// Backend unavailability or any pipeline-stage failure, as readable text.
pub fn native_variant(src: &str) -> Result<Box<dyn ExecVariant>, String> {
    match native_availability() {
        NativeAvailability::Ready => Ok(Box::new(NativeVariant::new(src, false)?)),
        NativeAvailability::Unavailable(why) => Err(why.to_string()),
    }
}

/// JIT-backed [`ExecVariant`] — sequential or parallel per the analysis plan.
pub struct NativeVariant {
    engine: helix_backend::JitEngine,
    /// "native-seq" or "native-par<P>" — par variants pin HELIX_NTHREADS
    /// around every execution (the runtime reads it at dispatch time).
    label: &'static str,
    /// When Some(p), HELIX_NTHREADS is set to p for each run/batch.
    threads: Option<usize>,
}

impl NativeVariant {
    /// Compile once up front (JIT compilation is measured separately from
    /// steady-state execution by the sampler's warmup discipline).
    ///
    /// # Errors
    /// Any pipeline stage or backend compile failure, as text.
    pub fn new(src: &str, unchecked: bool) -> Result<Self, String> {
        let ast = helix_syntax::parse_str(src).map_err(|e| format!("syntax error: {e}"))?;
        let typed = helix_sema::check(&ast).map_err(|ds| format!("{ds:#?}"))?;
        let mut funcs = helix_ir::build(&typed);
        for f in &mut funcs {
            helix_ir::to_ssa(f);
        }
        let loops: Vec<_> = funcs.iter().map(helix_analysis::find_loops).collect();
        let reports: Vec<_> = funcs
            .iter()
            .zip(&loops)
            .map(|(f, l)| helix_analysis::analyze(f, l))
            .collect();
        let plan = helix_analysis::build_plan(&funcs, &loops, &reports);
        // Convert to the backend seam type.
        let mut bplan = helix_backend::ParallelPlan::default();
        for r in &plan.regions {
            bplan.regions.push(helix_backend::RegionDesc {
                func_idx: r.func_idx,
                header: r.header,
                kind: match r.kind {
                    helix_analysis::RegionKind::DoAll => helix_backend::RegionKind::DoAll,
                    helix_analysis::RegionKind::Reduction(op) => {
                        helix_backend::RegionKind::Reduction(match op {
                            helix_analysis::ReductionOp::Add => {
                                helix_backend::engine::helix_analysis_stub::ReductionOp::Add
                            }
                            helix_analysis::ReductionOp::Mul => {
                                helix_backend::engine::helix_analysis_stub::ReductionOp::Mul
                            }
                            helix_analysis::ReductionOp::Min => {
                                helix_backend::engine::helix_analysis_stub::ReductionOp::Min
                            }
                            helix_analysis::ReductionOp::Max => {
                                helix_backend::engine::helix_analysis_stub::ReductionOp::Max
                            }
                        })
                    }
                },
                body_fn_name: r.body_fn_name.clone(),
            });
        }
        let engine = helix_backend::JitEngine::compile(&funcs, &bplan, unchecked)?;
        Ok(Self {
            engine,
            label: "native-seq",
            threads: None,
        })
    }

    /// Same compilation, but executions pin HELIX_NTHREADS=p (the parallel
    /// variant). The plan is baked at compile time; the runtime honours the
    /// env override per dispatch, so ONE engine serves the whole sweep.
    pub fn new_parallel(src: &str, p: usize) -> Result<Self, String> {
        let mut v = Self::new(src, false)?;
        v.label = "native-par";
        v.threads = Some(p);
        Ok(v)
    }

    /// One full native execution.
    ///
    /// # Errors
    /// Runtime trap or execution failure, as text.
    pub fn execute(&self) -> Result<RunOutputLike, String> {
        helix_backend::engine::arm_trap_recorder();
        let (printed, result) = helix_backend::engine::capture_prints(|| self.engine.run_main());
        helix_backend::reset_host_heap(); // reclaim this run's arrays
        result?;
        if let Some((code, a, b)) = helix_backend::engine::take_last_trap() {
            return Err(format!("runtime trap code={code} at ({a},{b})"));
        }
        // Match the interpreter's checksum definition loosely: bench parity is
        // asserted on printed lines; checksums recomputed FNV-1a style here.
        let mut checksum = 0xcbf29ce484222325u64;
        for line in &printed {
            for byte in line.bytes().chain(std::iter::once(b'\n')) {
                checksum ^= u64::from(byte);
                checksum = checksum.wrapping_mul(0x100000001b3);
            }
        }
        Ok(RunOutputLike { printed, checksum })
    }
}

impl ExecVariant for NativeVariant {
    fn name(&self) -> &str {
        self.label
    }

    fn run_once(&self) -> Result<RunOutputLike, String> {
        let _guard = self
            .threads
            .map(|p| env_guard("HELIX_NTHREADS", &p.to_string()));
        self.execute()
    }
}

// ---------------------------------------------------------------------------
// Campaign schema (hyperfine-style, per benchmark-methodology.md rec. 11)
// ---------------------------------------------------------------------------

/// One measured variant of one (kernel, size) point.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VariantResult {
    /// `"interp"` / `"native-seq"` / `"native-par"`.
    pub name: String,
    /// Median per-run time, milliseconds.
    pub median_ms: f64,
    /// Minimum per-run time, milliseconds.
    pub min_ms: f64,
    /// Coefficient of variation (stddev/mean).
    pub cv: f64,
    /// Raw samples (per-run ms) for offline recomputation.
    pub samples_ms: Vec<f64>,
    /// Inner rep count frozen by the pilot (auditable choice).
    pub reps_per_sample: u32,
    /// True when CV > 5% forced one rerun.
    pub reran_for_cv: bool,
}

/// One speedup/efficiency row of a thread sweep.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct EfficiencyRow {
    /// Participants p.
    pub threads: usize,
    /// Speedup vs the p=1 point of the SAME variant family.
    pub speedup: f64,
    /// Parallel efficiency E_p = speedup / p.
    pub efficiency: f64,
}

/// One (kernel, N) campaign point.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KernelPoint {
    /// Kernel registry name.
    pub kernel: String,
    /// Problem size (elements, or matrix edge for stencil/matmul).
    pub n: i64,
    /// Measured variants.
    pub variants: Vec<VariantResult>,
    /// Speedups relative to the FIRST variant (the baseline).
    pub speedups: Vec<f64>,
    /// Thread-sweep efficiency table (parallel candidates only).
    pub efficiency: Vec<EfficiencyRow>,
    /// Verdict the dependence engine produced during this run, when the
    /// analysis pipeline was exercised (`None` when not wired yet).
    pub observed_verdict: Option<String>,
    /// Expected verdict from the registry (parity assertion target).
    pub expected_verdict: String,
}

/// Full campaign artifact — what `write_json` serializes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CampaignReport {
    /// Schema tag for downstream tooling.
    pub schema: String,
    /// ISO-8601 UTC start timestamp.
    pub timestamp_utc: String,
    /// Machine/toolchain/environment block.
    pub meta: SystemMeta,
    /// Measured bandwidth ceilings, one row per measured thread count
    /// (1 and hardware width; absent when skipped). A single-thread-only
    /// number used to masquerade as the machine ceiling — saxpy at 8 threads
    /// was being divided by a denominator no 8-thread run can approach.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub triad_ceilings: Vec<TriadGibPerSec>,
    /// Legacy single-row field (kept for old report readers; always the
    /// threads=1 measurement when present).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub triad_ceiling: Option<TriadGibPerSec>,
    /// One entry per measured (kernel, size).
    pub points: Vec<KernelPoint>,
}

/// Triad ceiling serialized shape (GiB/s per thread count).
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct TriadGibPerSec {
    pub threads: usize,
    pub gib_per_sec: f64,
}

/// Serializes a report to pretty JSON.
///
/// # Errors
/// Passes through `serde_json` errors (should not occur for these types).
pub fn write_json(report: &CampaignReport, path: &Path) -> Result<(), String> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    let text = serde_json::to_string_pretty(report).map_err(|e| format!("serialize: {e}"))?;
    std::fs::write(path, text).map_err(|e| format!("write {}: {e}", path.display()))
}

/// Loads a report back (offline figure generation reads these files).
///
/// # Errors
/// IO or JSON parse errors, as text.
pub fn read_json(path: &Path) -> Result<CampaignReport, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
}

// ---------------------------------------------------------------------------
// Campaign driver
// ---------------------------------------------------------------------------

/// Runs the full benchmark campaign.
///
/// Every kernel is first validated at its CORRECTNESS size against the
/// registry's oracle lines; a mismatch aborts the whole campaign (wrong
/// numbers are worse than no numbers). Perf sizes then go through the
/// interleaved sampler. Native variants join automatically when the backend
/// becomes available.
#[must_use]
pub fn run_campaign(config: &BenchConfig) -> CampaignReport {
    // Whole-campaign serialization. reset_host_heap, the HELIX_NTHREADS
    // EnvGuard, and the trap recorder are PROCESS-GLOBAL; two concurrent
    // campaigns would free each other's live arrays (heap corruption,
    // reproduced as 0xc0000374) and race env writes. Tests that exercise
    // run_campaign concurrently therefore queue here.
    static CAMPAIGN_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _campaign_guard = CAMPAIGN_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .map(Path::to_path_buf);
    let meta = SystemMeta::capture(repo_root.as_deref());

    // Ceiling at 1 thread AND at full hardware width: memory-bound speedups
    // must be judged against a denominator measured at the SAME width.
    let triad_ceilings = if config.skip_triad {
        Vec::new()
    } else {
        let hw = std::thread::available_parallelism().map_or(1, |n| n.get());
        let mut rows = vec![triad::measure_triad(config.triad_n, 1)];
        if hw > 1 {
            rows.push(triad::measure_triad(config.triad_n, hw));
        }
        rows.into_iter()
            .map(|t| TriadGibPerSec {
                threads: t.threads,
                gib_per_sec: t.gib_per_sec,
            })
            .collect::<Vec<_>>()
    };
    let triad_ceiling = triad_ceilings.first().copied();

    let mut points = Vec::new();
    for kernel in kernels::registry() {
        if !config.only_kernels.is_empty() && !config.only_kernels.iter().any(|k| k == kernel.name)
        {
            continue;
        }
        if kernel.correctness_only {
            // Correctness-only kernels still get their parity check (they are
            // part of integration_full), but contribute no timed points.
            continue;
        }
        points.extend(measure_kernel(&kernel, config));
    }

    CampaignReport {
        schema: "helix-campaign-v1".to_string(),
        timestamp_utc: meta.timestamp_utc.clone(),
        triad_ceilings,
        triad_ceiling,
        meta,
        points,
    }
}

/// Short verdict label for diagnostics (`"SafeParallel"` etc.).
fn verdict_label(v: Option<&helix_analysis::Verdict>) -> String {
    match v {
        Some(helix_analysis::Verdict::SafeParallel) => "SafeParallel".into(),
        Some(helix_analysis::Verdict::ReductionParallel(r)) => {
            format!("ReductionParallel({})", r.op.symbol())
        }
        Some(helix_analysis::Verdict::Sequential(_)) => "Sequential".into(),
        None => "(no analyzable loop)".into(),
    }
}

/// What the verdict gate needs about one program's analysis.
struct PlanView {
    /// Verdicts of the loops the parallelizer APPROVED (plan regions), in
    /// region order. This — not "the first loop's report" — is what
    /// `native-par` executes, so it is what positive expectations check;
    /// keying on the first loop breaks the moment a kernel grows an init
    /// loop ahead of its compute loop.
    approved: Vec<helix_analysis::Verdict>,
    /// Total loops the analyzer examined across all functions (0 = the
    /// pipeline saw nothing analyzable, which must not pass silently).
    loops_analyzed: usize,
}

/// Runs the analysis pipeline on `src` and collects [`PlanView`].
fn analyze_plan(src: &str) -> PlanView {
    let empty = PlanView {
        approved: Vec::new(),
        loops_analyzed: 0,
    };
    let prog = match helix_syntax::parse_str(src) {
        Ok(p) => p,
        Err(_) => return empty,
    };
    let typed = match helix_sema::check(&prog) {
        Ok(t) => t,
        Err(_) => return empty,
    };
    let mut funcs = helix_ir::build(&typed);
    for f in &mut funcs {
        helix_ir::to_ssa(f);
    }
    let loops: Vec<_> = funcs.iter().map(helix_analysis::find_loops).collect();
    let total: usize = loops.iter().map(|l| l.loops.len()).sum();
    let reports: Vec<_> = funcs
        .iter()
        .zip(&loops)
        .map(|(f, l)| helix_analysis::analyze(f, l))
        .collect();
    let plan = helix_analysis::build_plan(&funcs, &loops, &reports);
    let approved = plan
        .regions
        .iter()
        .filter_map(|r| {
            // RegionDesc carries the loop's HEADER BLOCK; LoopReport is keyed
            // by the loop FOREST id, so resolve header -> loop -> report.
            let lp = loops.get(r.func_idx)?.loops.iter().find(|l| l.header == r.header)?;
            reports
                .get(r.func_idx)?
                .iter()
                .find(|rep| rep.loop_id == lp.id)
                .map(|rep| rep.verdict.clone())
        })
        .collect();
    PlanView {
        approved,
        loops_analyzed: total,
    }
}

/// Measures one kernel across its size sweep.
fn measure_kernel(kernel: &KernelDef, config: &BenchConfig) -> Vec<KernelPoint> {
    // Verdict gate: whatever the dependence engine APPROVES for this kernel's
    // perf program must satisfy the registry's expectation, or the campaign
    // aborts. Without this a silent analyzer regression would drain the
    // parallel numbers while every row still claimed SafeParallel (the
    // numbers would look real). Positive expectations check the APPROVED PLAN
    // regions; the Sequential (negative) expectation checks that loops WERE
    // analyzed and that NOTHING was approved — build_plan never emits
    // rejected loops, so "empty approval list" alone would be vacuous.
    let observed = if kernel.expected_verdict != kernels::ExpectedVerdict::NotApplicable {
        let view = analyze_plan(&kernel.perf_source);
        assert!(
            view.loops_analyzed > 0,
            "{}: analyzer saw no loops at all — pipeline regressed or the \
             kernel source drifted",
            kernel.name
        );
        let ok = match kernel.expected_verdict {
            ExpectedVerdict::SafeParallel => view
                .approved
                .iter()
                .any(|v| matches!(v, helix_analysis::Verdict::SafeParallel)),
            ExpectedVerdict::ReductionParallel => view.approved.iter().any(|v| {
                matches!(v, helix_analysis::Verdict::ReductionParallel(_))
            }),
            _ => view.approved.is_empty(),
        };
        assert!(
            ok,
            "{}: approved-region verdicts {} do not match expected {:?} — \
             analysis regressed or the kernel source drifted",
            kernel.name,
            verdict_labels(&view.approved),
            kernel.expected_verdict
        );
        Some(verdict_labels(&view.approved))
    } else {
        None
    };

    // PARITY GATE (once per kernel): EVERY variant shape that will be timed —
    // the interpreter AND each native form the campaign constructs below —
    // must execute once at the correctness size and agree with the oracle
    // under the kernel's tolerance. A variant that cannot even do that must
    // abort the campaign loudly; discovering the disagreement after 15 timed
    // samples would mean publishing numbers from code that computes garbage.
    // (interp_max_size gating stays untouched: it bounds TIMED cost only.)
    assert_parity_all_variants(kernel, config);

    let mut out = Vec::new();
    for size in kernel.sizes {
        let src = kernel.source_at_size(*size);
        let mut variants: Vec<Box<dyn ExecVariant>> = Vec::new();

        // Interpreter only up to `interp_max_size`: interpreting millions of
        // iterations costs minutes per sample and adds no information (the
        // interp/native ratio is read from the largest shared size).
        if *size <= kernel.interp_max_size {
            match interp_variant(&src) {
                Ok(v) => variants.push(Box::new(v)),
                Err(e) => panic!(
                    "{}: interpreter compile failed at N={size}: {e}",
                    kernel.name
                ),
            }
        }
        let has_regions = kernel.is_parallel_candidate();
        // A native compile failure on a kernel whose expected verdict implies
        // parallel capability is a CAMPAIGN failure, never silent absence:
        // dropping the row would look like "no data" when it is really "the
        // backend broke" (and would drain the report's headline numbers).
        match native_variant(&src) {
            Ok(native) => variants.push(native),
            Err(e) if has_regions => panic!(
                "{}: native-seq compile failed at N={size} (kernel expects a \
                 parallel-capable verdict, so this aborts the campaign): {e}",
                kernel.name
            ),
            Err(_) => {}
        }
        // Parallel variant: same compiled plan, executions pin HELIX_NTHREADS
        // to a mid-range count (the efficiency table carries the full sweep).
        if has_regions && config.thread_sweep.len() > 1 {
            let hw = std::thread::available_parallelism().map_or(1, |n| n.get());
            let p = config
                .thread_sweep
                .iter()
                .copied()
                .filter(|&x| x > 1)
                .find(|&x| x >= hw / 2)
                .unwrap_or(hw.min(config.max_threads).max(2));
            match NativeVariant::new_parallel(&src, p) {
                Ok(v) => variants.push(Box::new(v)),
                Err(e) if has_regions => panic!(
                    "{}: native-par{p} compile failed at N={size}: {e}",
                    kernel.name
                ),
                Err(_) => {}
            }
        }
        assert!(
            !variants.is_empty(),
            "{}: no executable variant could be built at N={size}",
            kernel.name
        );

        // Round-robin sampling across all present variants. Any execution
        // error during pilot/warmup/sampling aborts: never sample failing code.
        let names: Vec<&str> = variants.iter().map(|v| v.name()).collect();
        let measurements = timing::run_interleaved(&names, |vi, reps| {
            variants[vi].time_batch(reps)
        })
        .unwrap_or_else(|e| {
            panic!(
                "kernel {} size {} failed during timing: {e}",
                kernel.name, size
            )
        });

        let baseline = measurements
            .first()
            .map(|m| m.median_ms.max(f64::MIN_POSITIVE))
            .unwrap_or(1.0);
        let vrs: Vec<VariantResult> = measurements
            .iter()
            .zip(&names)
            .map(|(m, name)| VariantResult {
                name: (*name).to_string(),
                median_ms: m.median_ms,
                min_ms: m.min_ms,
                cv: m.cv,
                samples_ms: m.samples_ms.clone(),
                reps_per_sample: m.reps_per_sample,
                reran_for_cv: m.reran_for_cv,
            })
            .collect();

        // Thread sweep for parallel candidates: p=1 vs p via HELIX_NTHREADS.
        let efficiency = if kernel.is_parallel_candidate() {
            sweep_efficiency(kernel, &src, *size, config)
        } else {
            Vec::new()
        };

        out.push(KernelPoint {
            kernel: kernel.name.to_string(),
            n: *size,
            variants: vrs,
            speedups: measurements
                .iter()
                .map(|m| baseline / m.median_ms.max(f64::MIN_POSITIVE))
                .collect(),
            efficiency,
            observed_verdict: observed.clone(),
            expected_verdict: format!("{:?}", kernel.expected_verdict),
        });
    }
    out
}

/// Comma-separated labels for a list of verdicts (gate diagnostics).
fn verdict_labels(verdicts: &[helix_analysis::Verdict]) -> String {
    if verdicts.is_empty() {
        return "(nothing approved)".to_string();
    }
    verdicts
        .iter()
        .map(|v| verdict_label(Some(v)))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Executes EVERY variant shape the campaign will time for `kernel` once at
/// the correctness-size source and asserts agreement with the oracle under
/// the kernel's tolerance: the interpreter, native-seq, and every pinned
/// native-par<P> the thread sweep / mid-range point construct. The first
/// failing variant aborts with its name.
fn assert_parity_all_variants(kernel: &KernelDef, config: &BenchConfig) {
    let src = &kernel.correctness_source;
    let oracle: Vec<String> = kernel
        .expected_printed
        .iter()
        .map(|s| (*s).to_string())
        .collect();

    // Collect (label, variant) for every shape the campaign could time.
    let mut shapes: Vec<(String, Box<dyn ExecVariant>)> = Vec::new();
    match interp_variant(src) {
        Ok(v) => shapes.push(("interp".to_string(), Box::new(v))),
        Err(e) => panic!("{}: interp parity variant failed to compile: {e}", kernel.name),
    }
    // native-seq: required whenever the verdict implies parallel capability;
    // otherwise optional but still parity-checked WHEN PRESENT — the campaign
    // times it, so it must agree at correctness size too.
    let has_regions = kernel.is_parallel_candidate();
    match native_variant(src) {
        Ok(v) => shapes.push((v.name().to_string(), v)),
        Err(e) if has_regions => panic!(
            "{}: native-seq parity variant failed to compile: {e}",
            kernel.name
        ),
        Err(_) => {}
    }
    // Every pinned parallel variant the campaign constructs: the mid-range P
    // of the timed point plus EACH p of the efficiency sweep (including p=1,
    // which sweep_efficiency builds as its pinned baseline).
    let mut pins: Vec<usize> = Vec::new();
    if has_regions {
        let hw = std::thread::available_parallelism().map_or(1, |n| n.get());
        let cap = config.max_threads.min(hw.max(1));
        if config.thread_sweep.len() > 1 {
            let mid = config
                .thread_sweep
                .iter()
                .copied()
                .filter(|&x| x > 1)
                .find(|&x| x >= hw / 2)
                .unwrap_or(hw.min(config.max_threads).max(2));
            pins.push(mid);
        }
        for p in &config.thread_sweep {
            if *p >= 1 && *p <= cap && !pins.contains(p) {
                pins.push(*p);
            }
        }
    }
    for p in pins {
        match NativeVariant::new_parallel(src, p) {
            Ok(v) => shapes.push((format!("{}{p}", v.name()), Box::new(v))),
            Err(e) => panic!(
                "{}: native-par{p} parity variant failed to compile: {e}",
                kernel.name
            ),
        }
    }

    for (label, variant) in &shapes {
        let out = variant.run_once().unwrap_or_else(|e| {
            panic!(
                "{}: parity ABORT — variant [{label}] failed at correctness \
                 size: {e}",
                kernel.name
            )
        });
        assert!(
            kernel.tolerance.matches(&out.printed, &oracle),
            "{}: parity ABORT — variant [{label}] printed {:?}, expected {:?} \
             (tolerance {})",
            kernel.name,
            out.printed,
            oracle,
            kernel.tolerance.name()
        );
    }
}

/// Cross-variant output agreement at the kernel's correctness size.
///
/// With only the interpreter present this degenerates to "the program prints
/// the oracle lines", which is still the strongest check we can make until
/// the JIT lands.
#[must_use]
pub fn parity_holds(kernel: &KernelDef) -> bool {
    let Ok(interp) = interp_variant(&kernel.correctness_source) else {
        return false;
    };
    let Ok(out) = interp.run_once() else {
        return false;
    };
    if out.printed.len() != kernel.expected_printed.len() {
        return false;
    }
    let oracle: Vec<String> = kernel
        .expected_printed
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    kernel.tolerance.matches(&out.printed, &oracle)
}

/// Times one program's interpreter at p=1 vs the swept p values by pinning
/// `HELIX_NTHREADS` (the override helix-runtime honours), producing
/// [`EfficiencyRow`]s normalized to the p=1 point.
///
/// Variant construction or execution failures abort loudly (they would
/// otherwise masquerade as "no scaling data").
fn sweep_efficiency(
    kernel: &KernelDef,
    src: &str,
    size: i64,
    config: &BenchConfig,
) -> Vec<EfficiencyRow> {
    let hw = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let ps: Vec<usize> = config
        .thread_sweep
        .iter()
        .copied()
        .filter(|p| *p >= 1 && *p <= config.max_threads.min(hw.max(1)))
        .collect();
    if ps.first() != Some(&1) {
        return Vec::new();
    }

    // Every row (INCLUDING p=1) gets its own variant whose executions pin
    // HELIX_NTHREADS=p. Pinning the baseline matters: without it the engine's
    // baked-in hint (8) would run, and since the env override CAPS the hint,
    // every p >= 8 would tie the baseline at ~1.0x — exactly the flat-sweep
    // bug this replaces.
    let make_variant = |p: usize| -> Box<dyn ExecVariant> {
        NativeVariant::new_parallel(src, p)
            .map(|v| Box::new(v) as Box<dyn ExecVariant>)
            .unwrap_or_else(|e| {
                panic!(
                    "{}: efficiency-sweep native-par{p} failed to compile at \
                     N={size}: {e}",
                    kernel.name
                )
            })
    };
    let mut rows = Vec::with_capacity(ps.len());
    let mut base = f64::NAN;
    for &p in &ps {
        let variant = make_variant(p);
        let med = timing::measure_with_reps(|r| variant.time_batch(r))
            .unwrap_or_else(|e| {
                panic!(
                    "kernel {} size {} failed during timing \
                     (efficiency sweep p={p}): {e}",
                    kernel.name, size
                )
            })
            .median_ms;
        if p == 1 {
            base = med;
            rows.push(EfficiencyRow {
                threads: 1,
                speedup: 1.0,
                efficiency: 1.0,
            });
        } else {
            let speedup = base.max(f64::MIN_POSITIVE) / med.max(f64::MIN_POSITIVE);
            rows.push(EfficiencyRow {
                threads: p,
                speedup,
                efficiency: speedup / f64::from(p as u32),
            });
        }
    }
    rows
}

/// Sets `name=value`, returning a guard restoring the previous value on drop.
struct EnvGuard {
    name: &'static str,
    prev: Option<std::ffi::OsString>,
}

/// Installs a temporary environment override (thread-count sweeps).
fn env_guard(name: &'static str, value: &str) -> EnvGuard {
    let prev = std::env::var(name).ok().map(std::ffi::OsString::from);
    // SAFETY: same single-threaded campaign invariant as [`EnvGuard`].
    unsafe { std::env::set_var(name, value) };
    EnvGuard { name, prev }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        // SAFETY: bench-only process-global mutation, single-threaded by
        // construction (thread sweeps run outside the interleaved rounds),
        // and no other thread reads the environment concurrently.
        unsafe { std::env::remove_var(self.name) };
        if let Some(v) = self.prev.take() {
            // Same single-threaded invariant as above.
            unsafe { std::env::set_var(self.name, v) };
        }
    }
}

// ---------------------------------------------------------------------------
// CLI-facing entry (used later by helix-cli's `bench` subcommand)
// ---------------------------------------------------------------------------

/// Runs the default campaign and writes results into `out_dir`
/// (`campaign.json` + `meta.json`). Prints a progress banner to stdout.
///
/// # Errors
/// Returns write failures; measurement itself never fails hard.
pub fn campaign_main(out_dir: &Path) -> Result<(), String> {
    let config = BenchConfig::default();
    println!("HELIX benchmark campaign starting");
    let report = run_campaign(&config);
    println!("{}", report.meta.banner());

    let results_path = out_dir.join("campaign.json");
    write_json(&report, &results_path)?;
    // Standalone provenance file: just the meta block, no empty point list.
    let meta_path = out_dir.join("meta.json");
    let meta_text =
        serde_json::to_string_pretty(&report.meta).map_err(|e| format!("serialize: {e}"))?;
    std::fs::write(&meta_path, meta_text)
        .map_err(|e| format!("write {}: {e}", meta_path.display()))?;

    println!("results: {}", results_path.display());
    println!("meta:    {}", meta_path.display());

    // Console summary table: absolute numbers first (methodology rec. 9).
    println!(
        "{:<22}{:>12}{:>12}{:>10}",
        "kernel/N", "variant", "median ms", "CV %"
    );
    for p in &report.points {
        for v in &p.variants {
            println!(
                "{:<22}{:>12}{:>12.3}{:>10.2}",
                format!("{}/{}", p.kernel, p.n),
                v.name,
                v.median_ms,
                v.cv * 100.0
            );
        }
    }
    Ok(())
}
