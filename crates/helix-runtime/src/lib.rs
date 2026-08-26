//! HELIX Lite parallel runtime.
//!
//! The host side of the auto-parallelization pipeline: JITed program code
//! (produced by helix-backend) calls into this crate to execute approved
//! parallel regions. Zero dependencies on other helix crates — the backend
//! talks to us purely through [`helix_parallel_for`],
//! [`helix_parallel_reduction`] and [`register_body`] / [`register_combine`].
//!
//! # Architecture
//!
//! * [`schedule`] — chunk planning: static equal chunks (libgomp formula,
//!   cache-line-aligned where cheap) plus dynamic/guided claiming through one
//!   padded atomic counter.
//! * [`pool`] — Stage B: persistent workers, short-spin-then-park, generation
//!   dispatch; reusable across regions at ~µs per fork/join.
//! * [`scope_stage`] — Stage A: `std::thread::scope` spawn-per-call reference
//!   stage whose fixed cost is the baseline the pool erases.
//! * [`exec`] — the scheduling kernel shared by both stages.
//! * [`reduction`] — 128-byte-strided per-participant accumulator cells with
//!   a serial post-join combine (never atomics in the hot loop).
//! * [`config`] — cost gate (`n < max(1024, GRAIN·P)`) and env overrides
//!   (`HELIX_NTHREADS`, `HELIX_SCHEDULE`, `HELIX_RUNTIME`).
//!
//! # Safety posture
//!
//! This is one of two crates allowed `unsafe` (with helix-backend). Every
//! unsafe block carries a justification; body/combine pointers come only from
//! our registry (populated by the backend from finalized JIT code — trusted
//! compiler output, never user-controlled values); no unwinds cross an
//! `extern "C"` boundary (caught at every call site); no `unwrap()` on
//! anything user-controllable (locks recover from poisoning, malformed env
//! vars are ignored, unknown ids produce errors not panics).

mod config;
mod exec;
mod pool;
mod reduction;
mod registry;
mod scope_stage;

pub mod schedule;

pub use pool::BodyFn;
pub use registry::CombineFn;

use std::hint;
use std::panic::{self, AssertUnwindSafe};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use exec::{RegionOutcome, RegionParams};
use pool::{RegionRequest, run_on_pool};
use schedule::SchedKind;

/// Byte stride between per-participant reduction accumulator cells.
pub const REDUCTION_CELL_STRIDE: usize = reduction::CELL_STRIDE;

/// Byte offset of the accumulator field INSIDE a participant cell (word 0
/// holds the shared-ctx pointer). Single source of truth: helix-backend's
/// emitter and dispatcher must import this constant, never restate it.
pub const REDUCTION_ACC_OFFSET: usize = reduction::ACC_OFFSET;

/// Iterations below this per participant are not worth a fork/join.
///
/// Public mirror of [`config::GRAIN`] so benches/docs can cite one number:
/// the gate runs serial when `n < max(1024, GRAIN * nthreads)`.
pub const GRAIN: i64 = config::GRAIN;

/// Which execution engine dispatches parallel regions (contract name).
///
/// * [`RuntimeStage::ScopeThreads`] — Stage A: spawn-per-call via
///   `std::thread::scope`.
/// * [`RuntimeStage::Pool`] — Stage B: persistent worker pool.
///
/// Selected programmatically via [`set_stage`] or per-process via the
/// `HELIX_RUNTIME=scope|pool` environment variable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeStage {
    /// Stage A: `std::thread::scope` + static chunks; pays CreateThread cost
    /// (~50–100 µs/thread on Windows) per region.
    ScopeThreads,
    /// Stage B: persistent pool + bounded spin/park idle policy; ~µs regions.
    Pool,
}

impl RuntimeStage {
    /// Internal view used by the config layer.
    fn choice(self) -> config::StageChoice {
        match self {
            RuntimeStage::ScopeThreads => config::StageChoice::ScopeThreads,
            RuntimeStage::Pool => config::StageChoice::Pool,
        }
    }

    /// Canonical lowercase name (`scope` / `pool`).
    pub fn name(self) -> &'static str {
        self.choice().name()
    }
}

impl From<config::StageChoice> for RuntimeStage {
    fn from(c: config::StageChoice) -> Self {
        match c {
            config::StageChoice::ScopeThreads => RuntimeStage::ScopeThreads,
            config::StageChoice::Pool => RuntimeStage::Pool,
        }
    }
}

static ACTIVE_STAGE: Mutex<Option<RuntimeStage>> = Mutex::new(None);

/// Selects the execution engine for subsequent parallel regions.
///
/// The default (no call, no env override) is [`RuntimeStage::Pool`]. An
/// explicit `HELIX_RUNTIME` environment variable wins over this setting so
/// lab scripts can flip stages without recompiling.
pub fn set_stage(stage: RuntimeStage) {
    let mut slot = match ACTIVE_STAGE.lock() {
        Ok(s) => s,
        Err(e) => e.into_inner(),
    };
    *slot = Some(stage);
}

/// Currently selected engine (default [`RuntimeStage::Pool`], then
/// `set_stage`, then the `HELIX_RUNTIME` override).
fn current_stage() -> RuntimeStage {
    let set = match ACTIVE_STAGE.lock() {
        Ok(s) => *s,
        Err(e) => *e.into_inner(),
    };
    let base = set.unwrap_or(RuntimeStage::Pool);
    if let Some(forced) = std::env::var("HELIX_RUNTIME")
        .ok()
        .and_then(|v| config::StageChoice::parse(&v))
    {
        return forced.into();
    }
    base
}

/// Snapshot of pool health/counters for the overhead microbench graphs
/// (contract name: `pool_stats`).
pub fn pool_stats() -> PoolStats {
    pool::pool_stats()
}

pub use pool::PoolStats;

/// Registers the JIT-emitted body `f` under `id` ([`registry::register_body`]).
pub fn register_body(id: i64, f: pool::BodyFn) {
    registry::register_body(id, f);
}

/// Registers the JIT-emitted combine fn `f` under `id`.
pub fn register_combine(id: i64, f: registry::CombineFn) {
    registry::register_combine(id, f);
}

/// Result of one dispatched region, used by tests/selftest and verbose traces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DispatchStatus {
    /// Executed serially because the cost gate fired.
    Serial,
    /// Executed on threads via Stage A.
    Scope,
    /// Executed on threads via Stage B (pool).
    Pool,
}

/// Outcome of the last dispatched region (verbose tracing / selftest).
#[derive(Clone, Copy, Debug)]
pub struct LastDispatch {
    pub status: DispatchStatus,
    pub participants: usize,
    pub sched: SchedKind,
    /// Wall-clock duration of the dispatch call.
    pub elapsed: Duration,
}

static LAST_DISPATCH: Mutex<Option<LastDispatch>> = Mutex::new(None);

fn record(d: LastDispatch) {
    let mut slot = match LAST_DISPATCH.lock() {
        Ok(s) => s,
        Err(e) => e.into_inner(),
    };
    *slot = Some(d);
}

/// Reads (and clears) the last dispatch record. Diagnostic aid for labs.
pub fn take_last_dispatch() -> Option<LastDispatch> {
    let mut slot = match LAST_DISPATCH.lock() {
        Ok(s) => s,
        Err(e) => e.into_inner(),
    };
    slot.take()
}

/// Hardware parallelism, tolerating sandboxes that report zero/none.
fn hw_threads() -> usize {
    hint::black_box(
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
    )
}

/// Core dispatcher shared by both entry points.
///
/// Order of decisions: resolve body -> size region -> cost gate -> pick stage
/// -> dispatch -> fold reduction (if any). Every failure surfaces as `Err`
/// (clean message), never as a panic across the JIT boundary.
fn dispatch(
    start: i64,
    end: i64,
    body_id: i64,
    nthreads_hint: i64,
    acc_cells: Option<(*mut u8, i64)>,
    combine_id: Option<i64>,
) -> Result<(), String> {
    // ---- Resolve the body pointer from the registry. ----------------------
    let Some(body) = registry::lookup_body(body_id) else {
        return Err(format!(
            "helix-runtime: no body registered for id {body_id}"
        ));
    };
    let n = end.saturating_sub(start).max(0);
    let hw = hw_threads();

    // ---- Plan participants/schedule/stage (env overrides apply). ----------
    let decision = config::plan_region(n, nthreads_hint, hw, current_stage().choice(), MIN_CHUNK);
    let started = Instant::now();
    let status = if decision.serial_gate || n == 0 {
        // Cost gate fired (or empty range): run inline, participant 0 only.
        // SAFETY: ctx_base points at the coordinator's own cell (or is null
        // for plain regions); validity is established below.
        let ctx = acc_cells.map_or(std::ptr::null_mut(), |(base, _stride)| base);
        let ctx = pool::CtxPtr(ctx);
        if !exec::drive(
            &RegionParams {
                start,
                end,
                participants: 1,
                sched: SchedKind::Static,
                min_chunk: MIN_CHUNK,
                body,
                ctx_base: ctx,
                ctx_stride: 0,
            },
            &schedule::ClaimCounter::default(),
            0,
            ctx,
        ) {
            return Err(BODY_PANIC_MSG.to_string());
        }
        DispatchStatus::Serial
    } else {
        // ---- Threaded path: validate reduction plumbing. ------------------
        let combine = match (acc_cells, combine_id) {
            (Some(_), Some(cid)) => {
                let Some(combine) = registry::lookup_combine(cid) else {
                    return Err(format!(
                        "helix-runtime: no combine fn registered for id {cid}"
                    ));
                };
                Some(combine)
            }
            (None, None) => None,
            (Some(_), None) => {
                return Err("helix-runtime: reduction cells passed without combine id".to_string());
            }
            (None, Some(_)) => {
                return Err("helix-runtime: combine id without reduction cells".to_string());
            }
        };

        match combine {
            Some(combine) => {
                // Reduction: use the CALLER's accumulator area (contract:
                // cell p lives at acc_cells + p * REDUCTION_CELL_STRIDE; cell
                // 0 is the coordinator's and receives the combined total).
                let (cells_base, _stride_tag) =
                    acc_cells.expect("checked above: acc_cells present with combine");
                let base = cells_base;
                // SAFETY: the caller (JIT program) owns `acc_cells` for at
                // least `participants * REDUCTION_CELL_STRIDE` bytes — that is
                // the FFI contract of `helix_parallel_reduction` — and it
                // outlives this call. Cells are disjoint by stride.
                let params = RegionParams {
                    start,
                    end,
                    participants: decision.participants,
                    sched: decision.sched,
                    min_chunk: decision.min_chunk,
                    body,
                    ctx_base: pool::CtxPtr(base),
                    ctx_stride: REDUCTION_CELL_STRIDE,
                };
                let outcome = thread_out(params, decision.stage);
                // SAFETY: base is the caller's 128-strided cell area with
                // `participants` slots; combine is backend-registered code.
                unsafe { reduction::fold(base, decision.participants, combine) };
                outcome_to_result(outcome)?;
            }
            None => {
                // Plain parallel-for: one shared (null) context.
                let params = RegionParams {
                    start,
                    end,
                    participants: decision.participants,
                    sched: decision.sched,
                    min_chunk: decision.min_chunk,
                    body,
                    ctx_base: pool::CtxPtr(std::ptr::null_mut()),
                    ctx_stride: 0,
                };
                outcome_to_result(thread_out(params, decision.stage))?;
            }
        }
        if decision.stage == config::StageChoice::Pool {
            DispatchStatus::Pool
        } else {
            DispatchStatus::Scope
        }
    };

    record(LastDispatch {
        status,
        participants: decision.participants,
        sched: decision.sched,
        elapsed: started.elapsed(),
    });
    Ok(())
}

const BODY_PANIC_MSG: &str = "helix-runtime: parallel body terminated abnormally (caught unwind)";

fn outcome_to_result(outcome: RegionOutcome) -> Result<(), String> {
    match outcome {
        RegionOutcome::Completed => Ok(()),
        RegionOutcome::BodyPanicked => Err(BODY_PANIC_MSG.to_string()),
    }
}

/// Runs a threaded region on the requested stage, catching any runtime-side
/// panic before it can reach JIT frames.
fn thread_out(params: RegionParams, stage: config::StageChoice) -> RegionOutcome {
    let req = |p: &RegionParams| RegionRequest {
        start: p.start,
        end: p.end,
        participants: p.participants,
        sched: p.sched,
        min_chunk: p.min_chunk,
        body: p.body,
        ctx_base: p.ctx_base,
        ctx_stride: p.ctx_stride,
    };
    let run = || match stage {
        config::StageChoice::Pool => run_on_pool(req(&params)),
        config::StageChoice::ScopeThreads => scope_stage::run_guarded(params),
    };
    // Belt-and-braces: nothing in either stage should unwind, but a bug there
    // must still not cross into JIT frames.
    match panic::catch_unwind(AssertUnwindSafe(run)) {
        Ok(o) => o,
        Err(_) => RegionOutcome::BodyPanicked,
    }
}

/// Executes `[start, end)` in parallel, calling the registered body once per
/// iteration: `body(iteration, ctx)`.
///
/// Called from JITed main functions (M10 backend) with all-pointer args as
/// i64s. Decides internally between serial execution (cost gate), Stage A
/// (spawn-per-call) and Stage B (persistent pool); schedule comes from
/// `HELIX_SCHEDULE` or defaults to static chunks.
///
/// # Safety (FFI contract)
/// `body_id` must have been registered via [`register_body`] by the backend
/// after JIT finalization. Panics inside the JIT body never escape: they are
/// caught and reported by process abort (see module docs in lib top matter)
/// — actually they are caught and this function exits the process cleanly
/// with a diagnostic, matching `helix_panic` semantics for runtime errors.
///
/// # Errors
/// Unknown body ids abort the process with a diagnostic on stderr (a JIT/host
//  contract violation is unrecoverable for the running program).
pub extern "C" fn helix_parallel_for(start: i64, end: i64, body_id: i64, nthreads_hint: i64) {
    let r = dispatch(start, end, body_id, nthreads_hint, None, None);
    ffi_finish(r);
}

/// Reduction variant: each participant accumulates into its own 128-byte cell
/// starting at `acc_cells`; after join the cells are folded serially into cell
/// 0 with the registered combine fn (`dst = combine(dst, src)`), and control
/// returns with cell 0 holding the total.
///
/// Cell `p` lives at `acc_cells + p * REDUCTION_CELL_STRIDE` bytes; cell 0 is
/// the coordinator's and receives the combined result. Bodies must initialize
/// their own cell (monoid identity) before accumulating — the runtime zeroes
/// the area, which matches identity for `+`; other monoids should overwrite.
///
/// Same FFI safety/error contract as [`helix_parallel_for`].
pub extern "C" fn helix_parallel_reduction(
    start: i64,
    end: i64,
    body_id: i64,
    nthreads: i64,
    acc_cells: *mut u8,
    combine_id: i64,
) {
    // Null cells with a valid combine would corrupt memory downstream; treat
    // as contract violation rather than UB.
    if acc_cells.is_null() {
        eprintln!("helix-runtime: null accumulator cells for reduction");
        std::process::abort();
    }
    let r = dispatch(
        start,
        end,
        body_id,
        nthreads,
        Some((acc_cells, REDUCTION_CELL_STRIDE as i64)),
        Some(combine_id),
    );
    ffi_finish(r);
}

/// Single exit funnel for the extern "C" entry points: no unwinds may pass,
/// and host-side errors abort cleanly with a diagnostic (the JIT program's
/// contract is that runtime failures terminate the process, like
/// `helix_panic`).
fn ffi_finish(r: Result<(), String>) {
    match r {
        Ok(()) => {}
        Err(msg) => {
            // Diagnostics first, then a clean abort — mirrors the spec's
            // "print `runtime error: ...` and exit" behaviour without ever
            // unwinding through JIT frames.
            eprintln!("runtime error: {msg}");
            std::process::abort();
        }
    }
}

const MIN_CHUNK: u64 = schedule::MIN_CHUNK_DEFAULT;

#[cfg(test)]
mod tests;
