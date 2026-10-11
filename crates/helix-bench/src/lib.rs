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
//! helix-runtime honours. The engine's baked-in region hint is deliberately
//! far above any real host, so the override is what governs and a row labelled
//! `p` really runs `p` participants (clamped only by hardware). Efficiency
//! E_p = S_p / p is tabulated next to raw speedup so inefficiency cannot
//! hide (pitfall 12).
//!
//! ## Quick mode (`--quick`)
//!
//! [`BenchConfig::quick`] shrinks a campaign to a smoke test that exercises
//! the same [`run_campaign`] path: the smallest declared size per kernel, a
//! participant sweep of only {1, hardware}, and [`timing::K_SAMPLES_QUICK`]
//! samples per variant. It publishes no numbers — it proves the wiring runs.

pub mod kernels;
pub mod meta;
pub mod rust_twins;
pub mod timing;
pub mod triad;

use helix_engine::Value;
use helix_sema::ElemTy;
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
    /// Smoke-test mode: smallest size per kernel, a {1, hardware} thread
    /// sweep, and [`timing::K_SAMPLES_QUICK`] samples per variant. Same
    /// [`run_campaign`] path, no parallel code path — see the module docs.
    #[serde(default)]
    pub quick: bool,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            thread_sweep: vec![1, 2, 4, 8, 16, 24, 32],
            max_threads: 32,
            skip_triad: false,
            triad_n: triad::DEFAULT_TRIAD_N,
            only_kernels: Vec::new(),
            quick: false,
        }
    }
}

impl BenchConfig {
    /// Samples per variant this configuration asks the sampler for.
    #[must_use]
    pub fn samples(&self) -> usize {
        if self.quick {
            timing::K_SAMPLES_QUICK
        } else {
            timing::K_SAMPLES
        }
    }

    /// Participant counts swept for this configuration.
    ///
    /// A quick campaign measures 1 and the hardware width and nothing else:
    /// the scaling curve is what a full campaign is for, and a smoke test
    /// that takes twenty minutes is not a smoke test.
    #[must_use]
    pub fn effective_thread_sweep(&self) -> Vec<usize> {
        if !self.quick {
            return self.thread_sweep.clone();
        }
        let wide = available_parallelism().min(self.max_threads).max(1);
        if wide > 1 { vec![1, wide] } else { vec![1] }
    }
}

/// Logical CPUs visible to this process (1 when the OS will not say).
fn available_parallelism() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

/// Problem sizes [`run_campaign`] measures for one kernel.
///
/// Quick mode keeps only the SMALLEST declared size, so the campaign's point
/// count collapses to one per kernel without changing what is measured.
///
/// # Errors
/// In quick mode, a kernel that declares no size at all has no smaller size
/// to reduce to. Measuring nothing for it would look like "no data" when it
/// is really a registry mistake, so it is an error instead.
pub fn selected_sizes(kernel: &KernelDef, config: &BenchConfig) -> Result<Vec<i64>, String> {
    if !config.quick {
        return Ok(kernel.sizes.to_vec());
    }
    kernel.sizes.iter().min().copied().map_or_else(
        || {
            Err(format!(
                "kernel '{}' declares no sizes, so --quick has no smaller size to select",
                kernel.name
            ))
        },
        |smallest| Ok(vec![smallest]),
    )
}

/// Converts an analysis plan to the backend's seam type (M10: the fields map
/// 1:1 and reduction operators pass through so min/max regions combine with
/// the right monoid).
///
/// The CLI's JIT path shares this conversion rather than keeping a second
/// copy: a plan converted two different ways is a plan that can drift.
#[must_use]
pub fn to_backend_plan(p: &helix_analysis::ParallelPlan) -> helix_backend::ParallelPlan {
    use helix_backend::engine::helix_analysis_stub::ReductionOp as BOp;
    let mut out = helix_backend::ParallelPlan::default();
    for r in &p.regions {
        out.regions.push(helix_backend::RegionDesc {
            func_idx: r.func_idx,
            header: r.header,
            kind: match r.kind {
                helix_analysis::RegionKind::DoAll => helix_backend::RegionKind::DoAll,
                helix_analysis::RegionKind::Reduction(op) => {
                    helix_backend::RegionKind::Reduction(match op {
                        helix_analysis::ReductionOp::Add => BOp::Add,
                        helix_analysis::ReductionOp::Mul => BOp::Mul,
                        helix_analysis::ReductionOp::Min => BOp::Min,
                        helix_analysis::ReductionOp::Max => BOp::Max,
                    })
                }
            },
            body_fn_name: r.body_fn_name.clone(),
        });
    }
    out
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
    /// The checksum is the interpreter's FULL state witness — every printed
    /// line followed by the arrays `main` leaves bound — so a variant that
    /// prints the right values and corrupts memory cannot pass a comparison.
    ///
    /// # Errors
    /// Backend-specific failure text (parse/sema/runtime), rendered readably.
    fn run_once(&self) -> Result<RunOutputLike, String>;

    /// Milliseconds spent compiling this variant, measured once when it was
    /// built and reported as its own column so compile cost never contaminates
    /// steady-state execution numbers. `0.0` for variants with no compile
    /// step to report (the interpreter).
    fn compile_ms(&self) -> f64 {
        0.0
    }

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
/// Shared compile pipeline: parse → check → IR build → SSA → optimizer →
/// loop analysis → parallel plan, then a Cranelift compile with an EMPTY
/// backend plan — no regions — so the variant is genuinely sequential
/// whatever the analyzer approved. Each `run_once` executes the machine code
/// once (prints captured host-side).
///
/// # Errors
/// Backend unavailability or any pipeline-stage failure, as readable text.
pub fn native_variant(src: &str) -> Result<Box<dyn ExecVariant>, String> {
    match native_availability() {
        NativeAvailability::Ready => Ok(Box::new(NativeVariant::new(src, false)?)),
        NativeAvailability::Unavailable(why) => Err(why.to_string()),
    }
}

/// JIT-backed [`ExecVariant`] — sequential or parallel per the plan it was
/// compiled with.
pub struct NativeVariant {
    engine: helix_backend::JitEngine,
    /// "native-seq" or "native-par<P>" — par variants pin HELIX_NTHREADS
    /// around every execution (the runtime reads it at dispatch time).
    label: &'static str,
    /// When Some(p), HELIX_NTHREADS is set to p for each run/batch.
    threads: Option<usize>,
    /// Cranelift compile time of this variant, measured once at construction.
    compile_ms: f64,
    /// Element types of the arrays `main` leaves bound, in the interpreter's
    /// fold order — the array half of the full-state checksum.
    final_arrays: Vec<ElemTy>,
}

impl NativeVariant {
    /// Compiles `src` sequentially: the shared pipeline runs, and the engine
    /// is built from an **empty** backend plan.
    ///
    /// The analysis still runs (the verdict gate needs its verdicts), but its
    /// approved regions are deliberately NOT handed to the backend. Without
    /// that separation a "sequential baseline" would silently run parallel on
    /// any parallelizable kernel, which makes every speedup measured against
    /// it wrong.
    ///
    /// `unchecked` strips array bounds checks (`--unchecked`); division guards
    /// always remain.
    ///
    /// # Errors
    /// Any pipeline stage or backend compile failure, as text.
    pub fn new(src: &str, unchecked: bool) -> Result<Self, String> {
        let units = helix_artifact::prepare_pipeline(src)?;
        Self::from_units(&units, None, unchecked, "native-seq", None)
    }

    /// Same compilation, but the engine is built from the REAL analysis plan
    /// and executions pin `HELIX_NTHREADS=p` (the parallel variant). The plan
    /// is baked at compile time; the runtime honours the env override per
    /// dispatch, so ONE engine serves the whole sweep.
    ///
    /// # Errors
    /// Any pipeline stage or backend compile failure, as text.
    pub fn new_parallel(src: &str, p: usize) -> Result<Self, String> {
        let units = helix_artifact::prepare_pipeline(src)?;
        Self::from_units(&units, Some(&units.plan), false, "native-par", Some(p))
    }

    /// Builds the variant from an already-compiled [`CompiledUnits`].
    ///
    /// `plan = None` is the empty-plan (truly sequential) build; `plan = Some`
    /// hands the analyzer's approved regions to the backend. Taking the units
    /// as a parameter rather than the source keeps one pipeline run per
    /// variant instead of one per plan decision.
    fn from_units(
        units: &helix_artifact::CompiledUnits,
        plan: Option<&helix_analysis::ParallelPlan>,
        unchecked: bool,
        label: &'static str,
        threads: Option<usize>,
    ) -> Result<Self, String> {
        let final_arrays = units
            .funcs
            .iter()
            .find(|f| f.name == "main")
            .map(folded_array_elem_tys)
            .unwrap_or_default();
        let bplan = plan.map_or_else(helix_backend::ParallelPlan::default, to_backend_plan);
        // Compile time is its own cost, reported as its own column; folding it
        // into steady-state execution would corrupt every small-N number.
        let compile_start = std::time::Instant::now();
        let engine = helix_backend::JitEngine::compile(&units.funcs, &bplan, unchecked)?;
        let compile_ms = compile_start.elapsed().as_secs_f64() * 1_000.0;
        Ok(Self {
            engine,
            label,
            threads,
            compile_ms,
            final_arrays,
        })
    }

    /// Parallel regions the compiled engine can dispatch.
    ///
    /// `0` proves the variant is sequential no matter what its plan said:
    /// the sequential baseline is supposed to read `0` here, and a parallel
    /// variant is supposed to read at least `1`.
    #[must_use]
    pub fn region_count(&self) -> usize {
        self.engine.region_count()
    }

    /// Milliseconds this variant's Cranelift compile took, measured once.
    #[must_use]
    pub fn compile_ms(&self) -> f64 {
        self.compile_ms
    }

    /// Executes the machine code once with the print sink captured, leaving
    /// the JIT host heap intact so the caller can still snapshot final array
    /// state before [`helix_backend::reset_host_heap`] reclaims it.
    fn run_captured(&self) -> (Vec<String>, Result<(), String>) {
        helix_backend::engine::arm_trap_recorder();
        helix_backend::engine::capture_prints(|| self.engine.run_main())
    }

    /// One full native execution.
    ///
    /// The checksum is the interpreter's FULL state witness: every printed
    /// line (folded with the shared [`rust_twins::printed_checksum`]) followed
    /// by the arrays `main` leaves bound, encoded with the interpreter's own
    /// canonical byte encoder. A parallel kernel that prints the right values
    /// and corrupts memory therefore fails a parity comparison instead of
    /// passing it.
    ///
    /// # Errors
    /// Runtime trap, execution failure, or a program whose final-array shape
    /// the JIT host heap cannot witness — the last is reported rather than
    /// silently falling back to a printed-lines-only checksum.
    pub fn execute(&self) -> Result<RunOutputLike, String> {
        let (printed, result) = self.run_captured();
        let images = helix_backend::engine::live_array_images();
        helix_backend::reset_host_heap(); // reclaim this run's arrays
        result?;
        if let Some((code, a, b)) = helix_backend::engine::take_last_trap() {
            return Err(format!("runtime trap code={code} at ({a},{b})"));
        }
        let checksum = self.full_state_checksum(&printed, &images)?;
        Ok(RunOutputLike { printed, checksum })
    }

    /// Folds printed lines plus final arrays into the interpreter's FNV-1a
    /// stream.
    ///
    /// The pairing of host allocations to array bindings is by position: the
    /// JIT allocates `main`'s root-scope arrays in declaration order, and the
    /// interpreter folds them in ascending symbol order — the same order. A
    /// count or width mismatch means some other allocation (a callee's, one
    /// made inside a loop) is in the heap, so the stream could not be the
    /// interpreter's and the run reports that instead of guessing.
    ///
    /// # Errors
    /// A host allocation count or size that does not match what the IR says
    /// `main` leaves bound, which makes the interpreter's stream unreachable.
    fn full_state_checksum(
        &self,
        printed: &[String],
        images: &[(usize, Vec<u8>)],
    ) -> Result<u64, String> {
        if images.len() != self.final_arrays.len() {
            return Err(format!(
                "cannot witness final array state: main leaves {} array binding(s) bound but \
                 the JIT host heap holds {} live allocation(s) — the full-state checksum would \
                 not be the interpreter's, so no checksum is reported",
                self.final_arrays.len(),
                images.len()
            ));
        }
        let mut checksum = rust_twins::printed_checksum(printed);
        for (elem, (elements, bytes)) in self.final_arrays.iter().zip(images) {
            let width = elem_byte_width(*elem);
            if bytes.len() != width * *elements {
                return Err(format!(
                    "cannot witness final array state: an array of {elements} element(s) \
                     should occupy {} bytes but the JIT host heap holds {}",
                    width * *elements,
                    bytes.len()
                ));
            }
            for chunk in bytes.chunks_exact(width) {
                let value = match elem {
                    ElemTy::I32 => {
                        Value::I32(i32::from_le_bytes(chunk.try_into().unwrap_or([0; 4])))
                    }
                    ElemTy::I64 => {
                        Value::I64(i64::from_le_bytes(chunk.try_into().unwrap_or([0; 8])))
                    }
                    ElemTy::F32 => {
                        Value::F32(f32::from_le_bytes(chunk.try_into().unwrap_or([0; 4])))
                    }
                    ElemTy::F64 => {
                        Value::F64(f64::from_le_bytes(chunk.try_into().unwrap_or([0; 8])))
                    }
                    // The JIT stores a bool as a single byte, which is already
                    // the interpreter's canonical 0/1 encoding.
                    ElemTy::Bool => Value::Bool(chunk.first().is_some_and(|&b| b != 0)),
                };
                // The interpreter's own canonical encoder, so the byte stream
                // is identical by construction rather than by coincidence.
                value.hash_bits_into(&mut checksum);
            }
        }
        Ok(checksum)
    }
}

impl ExecVariant for NativeVariant {
    fn name(&self) -> &str {
        self.label
    }

    fn compile_ms(&self) -> f64 {
        self.compile_ms
    }

    fn run_once(&self) -> Result<RunOutputLike, String> {
        let _guard = self
            .threads
            .map(|p| env_guard("HELIX_NTHREADS", &p.to_string()));
        self.execute()
    }

    /// Timed batches skip the state checksum.
    ///
    /// Folding every final array into an FNV-1a stream costs as much as the
    /// kernel itself at campaign sizes, and it is a correctness obligation,
    /// not part of steady-state execution. The trait's default would call
    /// [`run_once`] and therefore measure the checksum; overriding it keeps
    /// the numbers honest. Correctness is enforced separately, once per kernel,
    /// by the campaign's parity gate.
    fn time_batch(&self, reps: u32) -> Result<std::time::Duration, String> {
        let _guard = self
            .threads
            .map(|p| env_guard("HELIX_NTHREADS", &p.to_string()));
        let start = std::time::Instant::now();
        for _ in 0..reps {
            let (printed, result) = self.run_captured();
            std::hint::black_box(&printed);
            helix_backend::reset_host_heap();
            result?;
            if let Some((code, a, b)) = helix_backend::engine::take_last_trap() {
                return Err(format!("runtime trap code={code} at ({a},{b})"));
            }
        }
        Ok(start.elapsed())
    }
}

/// Byte width the JIT host stores one element of `elem` in.
///
/// Mirrors `helix_backend::lower::elem_size`, the value the lowering passes to
/// `helix_zeros` when it allocates the buffer. Every width agrees with what
/// the interpreter folds for that type, so the raw bytes can be re-read as
/// values and re-encoded with the interpreter's own encoder.
fn elem_byte_width(elem: ElemTy) -> usize {
    match elem {
        ElemTy::I32 | ElemTy::F32 => 4,
        ElemTy::I64 | ElemTy::F64 => 8,
        ElemTy::Bool => 1,
    }
}

/// Element types of the arrays `main` leaves bound in its root scope, in
/// ascending symbol order — exactly the set, and the order, the interpreter
/// folds into the array half of its state checksum.
///
/// A `zeros` call qualifies when its block dominates `main`'s exit: that is
/// the static shape of "still bound when the run ends". Nested scopes are
/// popped by the interpreter before the checksum is taken, and a call inside
/// a loop body or an `if` arm does not dominate the exit, so both are excluded
/// — mirroring the rule the wasm emitter documents for the same stream.
fn folded_array_elem_tys(main: &helix_ir::FuncIr) -> Vec<ElemTy> {
    let Some(exit) = main
        .blocks
        .iter()
        .position(|b| matches!(b.term, helix_ir::Term::Return(_)))
    else {
        return Vec::new();
    };
    let doms = helix_ir::dominators(main);
    let exit = helix_ir::BlockId(exit as u32);
    let mut locals: Vec<helix_ir::LocalId> = main
        .blocks
        .iter()
        .enumerate()
        .filter(|(index, _)| doms.dominates(helix_ir::BlockId(*index as u32), exit))
        .flat_map(|(_, block)| {
            block
                .insts
                .iter()
                .filter_map(|inst| match inst {
                    helix_ir::Inst::Call(call) if call.callee == "zeros" => {
                        call.arr_refs.first().copied()
                    }
                    _ => None,
                })
                .collect::<Vec<_>>()
        })
        .collect();
    locals.sort_unstable_by_key(|local| local.0);
    locals.dedup();
    locals
        .iter()
        .filter_map(|local| main.types.elem_ty(*local))
        .collect()
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
    /// Cranelift compile time of this variant, milliseconds, measured once at
    /// construction. `0.0` for the interpreter, which has no compile step.
    /// Reported separately from `median_ms` so a one-off compile cost never
    /// contaminates steady-state execution numbers.
    #[serde(default)]
    pub compile_ms: f64,
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

/// Runs the shared compile pipeline on `src` and collects [`PlanView`].
///
/// The plan comes from the OPTIMIZED IR — the same IR `native-par` executes —
/// so the gate judges what actually runs rather than a pre-optimizer snapshot.
fn analyze_plan(src: &str) -> PlanView {
    let empty = PlanView {
        approved: Vec::new(),
        loops_analyzed: 0,
    };
    let Ok(units) = helix_artifact::prepare_pipeline(src) else {
        return empty;
    };
    let total: usize = units.loops.iter().map(|l| l.loops.len()).sum();
    let approved = units
        .plan
        .regions
        .iter()
        .filter_map(|r| {
            // RegionDesc carries the loop's HEADER BLOCK; LoopReport is keyed
            // by the loop FOREST id, so resolve header -> loop -> report.
            let lp = units
                .loops
                .get(r.func_idx)?
                .loops
                .iter()
                .find(|l| l.header == r.header)?;
            units
                .reports
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
            ExpectedVerdict::ReductionParallel => view
                .approved
                .iter()
                .any(|v| matches!(v, helix_analysis::Verdict::ReductionParallel(_))),
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
    let sizes = selected_sizes(kernel, config).unwrap_or_else(|e| {
        panic!("{}: {e}", kernel.name);
    });
    let samples = config.samples();
    let sweep = config.effective_thread_sweep();
    for size in sizes {
        let src = kernel.source_at_size(size);
        let mut variants: Vec<Box<dyn ExecVariant>> = Vec::new();

        // Interpreter only up to `interp_max_size`: interpreting millions of
        // iterations costs minutes per sample and adds no information (the
        // interp/native ratio is read from the largest shared size).
        if size <= kernel.interp_max_size {
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
        if has_regions && sweep.len() > 1 {
            let hw = available_parallelism();
            let p = sweep
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
        let measurements =
            timing::run_interleaved_n(&names, |vi, reps| variants[vi].time_batch(reps), samples)
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
            .zip(&variants)
            .map(|((m, name), variant)| VariantResult {
                name: (*name).to_string(),
                median_ms: m.median_ms,
                min_ms: m.min_ms,
                cv: m.cv,
                samples_ms: m.samples_ms.clone(),
                reps_per_sample: m.reps_per_sample,
                reran_for_cv: m.reran_for_cv,
                compile_ms: variant.compile_ms(),
            })
            .collect();

        // Thread sweep for parallel candidates: p=1 vs p via HELIX_NTHREADS.
        let efficiency = if kernel.is_parallel_candidate() {
            sweep_efficiency(kernel, &src, size, config)
        } else {
            Vec::new()
        };

        out.push(KernelPoint {
            kernel: kernel.name.to_string(),
            n: size,
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
///
/// Two comparisons are made per variant, and the second is the stronger one:
///
/// * **printed lines** must match the oracle lines under the kernel's
///   tolerance (epsilon for FP reductions, exact otherwise);
/// * **full-state checksum** (printed lines + the arrays `main` leaves bound)
///   must match the interpreter's checksum exactly for `interp` and
///   `native-seq`. A parallel variant is checksum-compared only when the
///   kernel declares no tolerance: a nonzero documented tolerance means
///   parallel partial sums legitimately reassociate FP addition, so its array
///   and printed values may differ in the last bits. The printed-line check
///   still guards every parallel variant's visible output in that case.
///
/// Without the checksum comparison a kernel that computes the right printed
/// values and corrupts memory would pass the gate on printed lines alone.
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
        Err(e) => panic!(
            "{}: interp parity variant failed to compile: {e}",
            kernel.name
        ),
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
    let sweep = config.effective_thread_sweep();
    let mut pins: Vec<usize> = Vec::new();
    if has_regions {
        let hw = available_parallelism();
        let cap = config.max_threads.min(hw.max(1));
        if sweep.len() > 1 {
            let mid = sweep
                .iter()
                .copied()
                .filter(|&x| x > 1)
                .find(|&x| x >= hw / 2)
                .unwrap_or(hw.min(config.max_threads).max(2));
            pins.push(mid);
        }
        for p in &sweep {
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

    // The interpreter IS the oracle for the checksum comparison: its run
    // already produced the full-state stream this gate compares against.
    let interp = shapes
        .iter()
        .find(|(label, _)| label == "interp")
        .map(|(_, v)| v.run_once())
        .expect("interp shape is always present")
        .unwrap_or_else(|e| {
            panic!(
                "{}: parity ABORT — variant [interp] failed at correctness size: {e}",
                kernel.name
            )
        });

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
        // Full-state parity: mandatory for the interpreter (it is the oracle)
        // and for the truly-sequential native build. A parallel variant only
        // has to agree when the kernel documents no reassociation slack.
        let is_oracle = label == "interp";
        let is_sequential = label == "native-seq";
        let exact_kernel = kernel.tolerance == Tolerance::Exact;
        if is_oracle || is_sequential || (label.starts_with("native-par") && exact_kernel) {
            assert_eq!(
                out.checksum, interp.checksum,
                "{}: parity ABORT — variant [{label}] full-state checksum \
                 {:016x} differs from the interpreter's {:016x} (printed lines \
                 matched, so the divergence is in main's final arrays)",
                kernel.name, out.checksum, interp.checksum
            );
        }
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
    let hw = available_parallelism();
    let ps: Vec<usize> = config
        .effective_thread_sweep()
        .iter()
        .copied()
        .filter(|p| *p >= 1 && *p <= config.max_threads.min(hw.max(1)))
        .collect();
    if ps.first() != Some(&1) {
        return Vec::new();
    }

    // Every row (INCLUDING p=1) gets its own variant whose executions pin
    // HELIX_NTHREADS=p. Pinning the baseline matters: without it the engine's
    // baked-in hint would run, and any env override that merely caps the hint
    // would tie every row above it at ~1.0x — exactly the flat-sweep bug this
    // replaces. The region hint is now deliberately unreachable, so the pin
    // is what a row measures.
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
    let samples = config.samples();
    let mut rows = Vec::with_capacity(ps.len());
    let mut base = f64::NAN;
    for &p in &ps {
        let variant = make_variant(p);
        let med = timing::measure_with_reps_n(|r| variant.time_batch(r), samples)
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
    campaign_main_with(out_dir, false)
}

/// [`campaign_main`] with an explicit quick-mode flag.
///
/// Quick mode runs the SAME [`run_campaign`] path with
/// [`BenchConfig::quick`](crate::BenchConfig::quick) set, so the only
/// difference is how much is measured — never how it is measured.
///
/// # Errors
/// Returns write failures; measurement itself never fails hard.
pub fn campaign_main_with(out_dir: &Path, quick: bool) -> Result<(), String> {
    let config = BenchConfig {
        quick,
        ..BenchConfig::default()
    };
    let kind = if quick { "quick " } else { "" };
    println!("HELIX {kind}benchmark campaign starting");
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
        "{:<22}{:>12}{:>12}{:>10}{:>11}",
        "kernel/N", "variant", "median ms", "CV %", "compile ms"
    );
    for p in &report.points {
        for v in &p.variants {
            println!(
                "{:<22}{:>12}{:>12.3}{:>10.2}{:>11.3}",
                format!("{}/{}", p.kernel, p.n),
                v.name,
                v.median_ms,
                v.cv * 100.0,
                v.compile_ms
            );
        }
    }
    Ok(())
}
