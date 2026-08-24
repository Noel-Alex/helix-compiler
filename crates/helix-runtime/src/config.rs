//! Runtime configuration: stage selection, thread-count caps, schedule and
//! cost-gate knobs, sourced from defaults + environment overrides.
//!
//! Env vars (read once per call, cheap `std::env::var` lookups):
//!
//! | Variable         | Effect                                            |
//! |------------------|---------------------------------------------------|
//! | `HELIX_NTHREADS` | caps the effective participant count (1..=hw)     |
//! | `HELIX_SCHEDULE` | forces a schedule: `static` \| `dynamic` \| `guided` |
//! | `HELIX_RUNTIME`  | selects the stage: `scope` \| `pool`              |
//!
//! Malformed values never fail a run: they are ignored (the default wins) so
//! a typo in an env var cannot break a lab session. `RUST_LOG=helix_runtime=…`
//! style verbosity is out of scope; use [`set_verbose`] for decision tracing.

use crate::schedule::SchedKind;

/// Which execution engine dispatches parallel regions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StageChoice {
    /// Stage A: spawn-per-call via `std::thread::scope`. Trivially correct;
    /// pays Windows CreateThread costs per region (~50–100 µs per thread).
    ScopeThreads,
    /// Stage B: persistent worker pool with bounded spin/park idle policy.
    #[default]
    Pool,
}

impl StageChoice {
    /// Parses a stage name; accepts `scope` / `pool` (case-insensitive).
    /// `None` means "no usable override".
    pub fn parse(s: &str) -> Option<StageChoice> {
        match s.trim().to_ascii_lowercase().as_str() {
            "scope" => Some(StageChoice::ScopeThreads),
            "pool" => Some(StageChoice::Pool),
            _ => None,
        }
    }

    /// Canonical lowercase name (dumps, verbose traces, bench metadata).
    pub fn name(self) -> &'static str {
        match self {
            StageChoice::ScopeThreads => "scope",
            StageChoice::Pool => "pool",
        }
    }
}

/// Iterations below this per participant are not worth a fork/join
/// (`docs/research/parallel-runtime.md`, fact 11 / recommendation 6).
pub const GRAIN: i64 = 1024;

/// Absolute serial-execution floor: even one participant needs this many
/// iterations before threading machinery can pay for itself.
pub const SERIAL_FLOOR: i64 = 1024;

/// Cost gate: true when the region should run **serially**.
///
/// Mirrors OpenMP's `if(n > threshold)` clause with the community-standard
/// shape `n < max(SERIAL_FLOOR, GRAIN * nthreads)` — small regions lose to
/// their own fork/join overhead.
pub fn should_run_serial(n: i64, nthreads: usize) -> bool {
    let p = nthreads.max(1) as i64;
    n < SERIAL_FLOOR.max(GRAIN.saturating_mul(p))
}

/// Effective participant count for a region, after every clamp:
///
/// 1. start from the compiler's hint,
/// 2. apply the `HELIX_NTHREADS` cap when set,
/// 3. clamp to `[1, available_parallelism]`.
pub fn effective_nthreads(hint: i64, hw: usize, env_cap: Option<u64>) -> usize {
    let hint = if hint < 1 {
        1
    } else {
        hint.min(i64::from(u32::MAX)) as usize
    };
    let capped = match env_cap {
        Some(c) if c >= 1 => (hint as u64).min(c) as usize,
        _ => hint,
    };
    capped.clamp(1, hw.max(1))
}

/// Reads `HELIX_NTHREADS` if present and well-formed (> 0), else `None`.
fn env_nthreads() -> Option<u64> {
    std::env::var("HELIX_NTHREADS")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|&v| v >= 1)
}

/// Reads `HELIX_SCHEDULE` if present and recognized, else `None`.
fn env_schedule() -> Option<SchedKind> {
    std::env::var("HELIX_SCHEDULE")
        .ok()
        .and_then(|v| SchedKind::parse(&v))
}

/// Reads `HELIX_RUNTIME` if present and recognized, else `None`.
fn env_stage() -> Option<StageChoice> {
    std::env::var("HELIX_RUNTIME")
        .ok()
        .and_then(|v| StageChoice::parse(&v))
}

/// Everything [`crate::dispatch`] needs to know about one region.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PlanDecision {
    /// Participants actually used (1 = fully serial).
    pub(crate) participants: usize,
    pub(crate) sched: SchedKind,
    /// True when the gate fired: execute inline, no threads at all.
    pub(crate) serial_gate: bool,
    /// Which engine would have been used had the region threaded.
    pub(crate) stage: StageChoice,
    /// Lower clamp on dynamic/guided chunk sizes (elements), passed through.
    pub(crate) min_chunk: u64,
}

/// Computes the full decision for one region (pure function of inputs + env).
///
/// `hw` is injected (rather than read here) so tests are deterministic on any
/// machine; production passes `available_parallelism()`.
pub(crate) fn plan_region(
    n: i64,
    nthreads_hint: i64,
    hw: usize,
    default_stage: StageChoice,
    min_chunk: u64,
) -> PlanDecision {
    let env_cap = env_nthreads();
    let participants = effective_nthreads(nthreads_hint, hw, env_cap);
    let stage = env_stage().unwrap_or(default_stage);
    let sched = env_schedule().unwrap_or(SchedKind::Static);
    let serial_gate = should_run_serial(n, participants);
    PlanDecision {
        participants: if serial_gate { 1 } else { participants },
        sched,
        serial_gate,
        stage,
        min_chunk,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cost_gate_fires_below_thresholds() {
        // Below the absolute floor: always serial, any thread count.
        assert!(should_run_serial(0, 1));
        assert!(should_run_serial(1023, 1));
        // At exactly max(1024, 1024*1): boundary is inclusive-serial, so
        // n == threshold threads, n == threshold+1 runs threaded.
        assert!(!should_run_serial(1024, 1));
        // 8 threads need GRAIN*8 iterations.
        assert!(should_run_serial(GRAIN * 8 - 1, 8));
        assert!(!should_run_serial(GRAIN * 8, 8));
        assert!(should_run_serial(5_000_000.min(GRAIN * 33 - 1), 33));
        assert!(!should_run_serial(GRAIN * 33, 33));
        // Zero/negative iteration counts must be serial (and harmless).
        assert!(should_run_serial(-10, 4));
        assert!(should_run_serial(100, 0)); // degenerate thread count clamps to 1
    }

    #[test]
    fn nthread_caps_apply_in_order() {
        let hw = 8usize;
        assert_eq!(effective_nthreads(4, hw, None), 4);
        assert_eq!(effective_nthreads(0, hw, None), 1); // nonsense hint clamps
        assert_eq!(effective_nthreads(-5, hw, None), 1);
        assert_eq!(effective_nthreads(999, hw, None), 8); // hardware cap
        assert_eq!(effective_nthreads(16, 2, None), 2);
        // HELIX_NTHREADS caps but never raises above hw either.
        assert_eq!(effective_nthreads(8, hw, Some(4)), 4);
        assert_eq!(effective_nthreads(2, hw, Some(6)), 2);
        assert_eq!(effective_nthreads(99, hw, Some(99)), 8);
        assert_eq!(effective_nthreads(4, hw, Some(0)), 4); // invalid cap ignored
        assert_eq!(effective_nthreads(4, 0, None), 1); // no hardware reported
    }

    #[test]
    fn stage_parse_and_names() {
        assert_eq!(StageChoice::parse("scope"), Some(StageChoice::ScopeThreads));
        assert_eq!(StageChoice::parse("POOL"), Some(StageChoice::Pool));
        assert_eq!(StageChoice::parse("threads"), None);
        assert_eq!(StageChoice::default(), StageChoice::Pool);
        assert_eq!(StageChoice::ScopeThreads.name(), "scope");
    }
}
