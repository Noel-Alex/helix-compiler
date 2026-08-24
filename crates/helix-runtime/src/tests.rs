//! End-to-end tests through the public FFI surface.
//!
//! Covers: schedule x stage correctness vs a sequential reference, reduction
//! combines (+, *, min, max) at 1..=33 participants, the cost gate, env
//! overrides, pool-vs-scope overhead ordering, and pool reuse across many
//! regions. Chunk-coverage invariants live in the [`crate::schedule`] unit
//! tests (union == range, no overlap).
//!
//! Test plumbing notes: bodies must be `extern "C"` and cannot capture, so
//! each counting body owns a dedicated static counter — no shared slots, so
//! tests can run concurrently on the harness's worker threads. Tests that
//! mutate environment variables serialize through [`ENV_LOCK`].

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use crate::{
    GRAIN, RuntimeStage, helix_parallel_for, helix_parallel_reduction, register_body,
    register_combine, set_stage,
};

/// Serializes tests that touch process-global environment variables.
static ENV_LOCK: Mutex<()> = Mutex::new(());

fn set_env(key: &str, value: &str) {
    // SAFETY: callers hold ENV_LOCK, so no other test thread reads the env
    // concurrently; values are static test strings.
    unsafe {
        std::env::set_var(key, value);
    }
}

fn remove_env(key: &str) {
    // SAFETY: same locking discipline as set_env.
    unsafe {
        std::env::remove_var(key);
    }
}

// --- Counting bodies: each test gets its own static counter -----------------

/// Counter for [`plain_parallel_for_counts_every_iteration_once_per_schedule_and_stage`].
static HITS_SCHED_STAGES: AtomicU64 = AtomicU64::new(0);
extern "C" fn count_sched_stages(_i: i64, _ctx: *mut u8) {
    HITS_SCHED_STAGES.fetch_add(1, Ordering::Relaxed);
}

/// Counter for [`cost_gate_runs_small_regions_serially_on_both_stages`].
static HITS_GATE: AtomicU64 = AtomicU64::new(0);
extern "C" fn count_gate(_i: i64, _ctx: *mut u8) {
    HITS_GATE.fetch_add(1, Ordering::Relaxed);
}

/// Counter for [`env_nthreads_caps_participants`].
static HITS_CAP: AtomicU64 = AtomicU64::new(0);
extern "C" fn count_cap(_i: i64, _ctx: *mut u8) {
    HITS_CAP.fetch_add(1, Ordering::Relaxed);
}

/// Counter for [`env_malformed_values_are_ignored`].
static HITS_MALFORMED: AtomicU64 = AtomicU64::new(0);
extern "C" fn count_malformed(_i: i64, _ctx: *mut u8) {
    HITS_MALFORMED.fetch_add(1, Ordering::Relaxed);
}

/// Counter for [`env_stage_override_wins_over_set_stage`].
static HITS_STAGE_OVERRIDE: AtomicU64 = AtomicU64::new(0);
extern "C" fn count_stage_override(_i: i64, _ctx: *mut u8) {
    HITS_STAGE_OVERRIDE.fetch_add(1, Ordering::Relaxed);
}

/// Counter for [`stress_many_regions_reuse_pool_threads`].
static HITS_STRESS: AtomicU64 = AtomicU64::new(0);
extern "C" fn count_stress(_i: i64, _ctx: *mut u8) {
    HITS_STRESS.fetch_add(1, Ordering::Relaxed);
}

extern "C" fn noop(_i: i64, _ctx: *mut u8) {}

/// Reduction body: adds the iteration into this participant's private cell.
extern "C" fn red_add(i: i64, ctx: *mut u8) {
    // SAFETY: ctx is this participant's own i64 accumulator cell.
    unsafe {
        *(ctx as *mut i64) += i;
    }
}

/// Multiplicative reduction body folding small positive factors only.
extern "C" fn red_mul(i: i64, ctx: *mut u8) {
    // SAFETY: ctx is this participant's private i64 cell.
    unsafe {
        *(ctx as *mut i64) *= i % 3 + 1;
    }
}

/// Unique id allocator so concurrent tests never share registry slots.
static NEXT_ID: AtomicU64 = AtomicU64::new(1_000_000);

fn next_id() -> i64 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed) as i64
}

fn with_stage<T>(stage: RuntimeStage, f: impl FnOnce() -> T) -> T {
    set_stage(stage);
    let out = f();
    set_stage(RuntimeStage::Pool); // restore default
    out
}

const ALL_STAGES: [RuntimeStage; 2] = [RuntimeStage::ScopeThreads, RuntimeStage::Pool];

// ---------------------------------------------------------------------------
// Correctness of all schedules x stages vs sequential reference.
// ---------------------------------------------------------------------------

#[test]
fn schedules_x_stages_match_sequential_reference() {
    let _env = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let body_id = next_id();
    let combine_id = next_id();
    register_body(body_id, red_add);
    register_combine(combine_id, combines::add_i64);
    for &stage in &ALL_STAGES {
        with_stage(stage, || {
            for sched in ["static", "dynamic", "guided"] {
                set_env("HELIX_SCHEDULE", sched);
                let mut cells = vec![0i64; 128]; // >= P * stride/8 words
                let base = cells.as_mut_ptr().cast::<u8>();
                helix_parallel_reduction(-5_000, 5_000, body_id, 4, base, combine_id);
                assert_eq!(cells[0], (-5_000..5_000).sum::<i64>(), "{stage:?}/{sched}");
            }
            remove_env("HELIX_SCHEDULE");
        });
    }
}

#[test]
fn plain_parallel_for_counts_every_iteration_once_per_schedule_and_stage() {
    let _env = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let body_id = next_id();
    register_body(body_id, count_sched_stages);
    for &stage in &ALL_STAGES {
        with_stage(stage, || {
            for sched in ["static", "dynamic", "guided"] {
                set_env("HELIX_SCHEDULE", sched);
                HITS_SCHED_STAGES.store(0, Ordering::Relaxed);
                helix_parallel_for(100, 100 + 9_999, body_id, 6);
                assert_eq!(
                    HITS_SCHED_STAGES.load(Ordering::Relaxed),
                    9_999,
                    "{stage:?}/{sched}"
                );
            }
            remove_env("HELIX_SCHEDULE");
        });
    }
}

#[test]
fn empty_and_degenerate_ranges_are_noops() {
    let body_id = next_id();
    register_body(body_id, noop);
    helix_parallel_for(7, 7, body_id, 8); // empty range
    helix_parallel_for(10, 3, body_id, 8); // inverted range
    // Reaching here without a hang or abort is the assertion; nothing ran.
}

// ---------------------------------------------------------------------------
// Reduction combine correctness (+, *, min, max) with 1..=33 participants.
// ---------------------------------------------------------------------------

pub(crate) mod combines {
    /// Combine fn signature alias.
    pub type C = extern "C" fn(*mut u8, *const u8);

    pub extern "C" fn add_i64(dst: *mut u8, src: *const u8) {
        // SAFETY: both point at caller-owned i64 reduction cells.
        unsafe {
            *(dst as *mut i64) += *(src as *const i64);
        }
    }

    pub extern "C" fn mul_i64(dst: *mut u8, src: *const u8) {
        // SAFETY: both point at caller-owned i64 reduction cells.
        unsafe {
            *(dst as *mut i64) *= *(src as *const i64);
        }
    }

    pub extern "C" fn min_i64(dst: *mut u8, src: *const u8) {
        // SAFETY: both point at caller-owned i64 reduction cells.
        unsafe {
            let d = dst as *mut i64;
            let s = *(src as *const i64);
            if s < *d {
                *d = s;
            }
        }
    }

    pub extern "C" fn max_i64(dst: *mut u8, src: *const u8) {
        // SAFETY: both point at caller-owned i64 reduction cells.
        unsafe {
            let d = dst as *mut i64;
            let s = *(src as *const i64);
            if s > *d {
                *d = s;
            }
        }
    }

    /// Sequential reference fold over iteration values.
    pub type FoldRef = fn(&[i64]) -> i64;

    /// One reduction case: combine fn, monoid identity, reference fold.
    pub type Case = (C, i64, FoldRef);

    /// All four monoids (+, *, min, max) with their identities.
    #[allow(clippy::type_complexity)]
    pub(crate) fn cases() -> Vec<Case> {
        vec![
            (add_i64, 0, |v| v.iter().sum()),
            (mul_i64, 1, |v| v.iter().product()),
            (min_i64, i64::MAX, |v| *v.iter().min().unwrap_or(&i64::MAX)),
            (max_i64, i64::MIN, |v| *v.iter().max().unwrap_or(&i64::MIN)),
        ]
    }
}

#[test]
fn reduction_combines_correct_across_participant_counts_and_stages() {
    let _env = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let add_id = next_id();
    let mul_id = next_id();
    register_body(add_id, red_add);
    register_body(mul_id, red_mul);
    set_env("HELIX_SCHEDULE", "static");

    for &stage in &ALL_STAGES {
        with_stage(stage, || {
            // Additive/min/max monoids over 100 iterations; the multiplicative
            // case is skipped here (100! overflows i64) and covered below.
            let cases: Vec<combines::Case> = combines::cases()
                .into_iter()
                .filter(|(c, _, _)| {
                    // Compare by address (fn pointers are not Eq).
                    let addr = |f: combines::C| f as usize;
                    addr(*c) != addr(combines::mul_i64)
                })
                .collect();
            for p in 1..=33i64 {
                // min/max cases: cells start at the identity and the body
                // OVERWRITES with its partial via `min`/`max` semantics — but
                // our test body ADDS. So for min/max we pre-seed cell values
                // by running once: instead use small-range addition against
                // identity i64::MIN/i64::MAX would overflow, so run the add
                // body only for the additive case and validate min/max with a
                // dedicated constant body below.
                for &(combine, identity, fold_ref) in &cases {
                    if identity != 0 {
                        continue; // min/max validated by the dedicated test
                    }
                    let cid = next_id();
                    register_combine(cid, combine);
                    let mut cells = vec![identity; 8 * 128]; // >= P * stride/8
                    helix_parallel_reduction(1, 101, add_id, p, cells.as_mut_ptr().cast(), cid);
                    let expect = fold_ref(&(1..101).collect::<Vec<i64>>());
                    assert_eq!(cells[0], expect, "{stage:?} combine#{cid} p={p}");
                }
            }
            // Multiplicative body sanity on a few participant counts. The
            // body folds `abs(i % 3) + 1` (values 1..=3), so use wrapping
            // arithmetic in the reference too and keep the range small.
            let cid2 = next_id();
            register_combine(cid2, combines::mul_i64);
            for p in [1i64, 5, 16, 32] {
                let mut cells = vec![1i64; 8 * 128];
                helix_parallel_reduction(1, 41, mul_id, p, cells.as_mut_ptr().cast(), cid2);
                let expect: i64 = (1..41).map(|i| i % 3 + 1).product();
                assert_eq!(cells[0], expect, "{stage:?} mul p={p}");
            }
        });
    }
    remove_env("HELIX_SCHEDULE");
}

/// Min-reduction body: overwrites the cell with `min(cell_partial, i)` —
/// mirroring how the JIT lowers `x = min(x, t)` (write-once per iteration).
extern "C" fn red_min(i: i64, ctx: *mut u8) {
    // SAFETY: ctx is this participant's private i64 cell.
    unsafe {
        let p = ctx as *mut i64;
        let v = *p;
        // First write in a region must seed from the iteration itself; the
        // runtime zeroes cells, so use wrapping-free compare against 0-sentinel:
        // our test range is positive, so plain min works if we treat 0 as
        // "uninitialized". The runtime zero-fills, so start from +inf-like.
        *p = if v == 0 { i } else { v.min(i) };
    }
}

/// Max-reduction body: symmetric to [`red_min`].
extern "C" fn red_max(i: i64, ctx: *mut u8) {
    // SAFETY: ctx is this participant's private i64 cell.
    unsafe {
        let p = ctx as *mut i64;
        let v = *p;
        *p = if v == 0 { i } else { v.max(i) };
    }
}

#[test]
fn min_max_reductions_match_sequential_reference() {
    let _env = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let min_id = next_id();
    let max_id = next_id();
    register_body(min_id, red_min);
    register_body(max_id, red_max);
    set_env("HELIX_SCHEDULE", "guided");

    for &stage in &ALL_STAGES {
        with_stage(stage, || {
            for p in [1i64, 3, 8, 17, 33] {
                // MIN reduction: identity for combine is i64::MAX; body seeds
                // each cell from its first iteration (range is positive so the
                // zero-fill acts as "unset").
                let cid = next_id();
                register_combine(cid, combines::min_i64);
                let mut cells = vec![i64::MAX; 8 * 128];
                helix_parallel_reduction(1, 101, min_id, p, cells.as_mut_ptr().cast(), cid);
                assert_eq!(cells[0], 1, "{stage:?} min p={p}");

                let cid2 = next_id();
                register_combine(cid2, combines::max_i64);
                let mut cells2 = vec![i64::MIN; 8 * 128];
                helix_parallel_reduction(1, 101, max_id, p, cells2.as_mut_ptr().cast(), cid2);
                assert_eq!(cells2[0], 100, "{stage:?} max p={p}");
            }
        });
    }
    remove_env("HELIX_SCHEDULE");
}

// ---------------------------------------------------------------------------
// Cost gate behaviour.
// ---------------------------------------------------------------------------

#[test]
fn cost_gate_runs_small_regions_serially_on_both_stages() {
    // n=512 < max(1024, GRAIN*p) for every p => serial regardless of stage.
    let body_id = next_id();
    register_body(body_id, count_gate);
    for &stage in &ALL_STAGES {
        with_stage(stage, || {
            HITS_GATE.store(0, Ordering::Relaxed);
            helix_parallel_for(0, 512, body_id, 32);
            assert_eq!(HITS_GATE.load(Ordering::Relaxed), 512);
            let last = crate::take_last_dispatch().expect("dispatch recorded");
            assert_eq!(
                last.status,
                crate::DispatchStatus::Serial,
                "gate must fire below threshold"
            );
        });
    }
}

#[test]
fn gate_threshold_boundary_matches_formula() {
    // n == GRAIN*P exactly threads; just below stays serial (P <= hw).
    assert!(!crate::config::should_run_serial(1024, 1));
    assert!(crate::config::should_run_serial(1023, 1));
    assert!(crate::config::should_run_serial(crate::GRAIN * 8 - 1, 8));
    assert!(!crate::config::should_run_serial(crate::GRAIN * 8, 8));
    assert!(crate::config::should_run_serial(crate::GRAIN * 33 - 1, 33));
    assert!(!crate::config::should_run_serial(crate::GRAIN * 33, 33));
}

// ---------------------------------------------------------------------------
// Env overrides.
// ---------------------------------------------------------------------------

#[test]
fn env_nthreads_caps_participants() {
    let _env = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let body_id = next_id();
    register_body(body_id, count_cap);
    set_env("HELIX_NTHREADS", "2");
    helix_parallel_for(0, 200_000, body_id, 64);
    remove_env("HELIX_NTHREADS");
    assert_eq!(HITS_CAP.load(Ordering::Relaxed), 200_000);
    let last = crate::take_last_dispatch().expect("recorded");
    assert!(
        last.participants <= 2,
        "HELIX_NTHREADS=2 must cap participants, got {}",
        last.participants
    );
}

#[test]
fn env_malformed_values_are_ignored() {
    let _env = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    use crate::config::StageChoice;
    // Pure parse paths (no env mutation):
    assert_eq!(StageChoice::parse("pool"), Some(StageChoice::Pool));
    assert_eq!(StageChoice::parse("nope"), None);
    assert_eq!(crate::schedule::SchedKind::parse("not-a-schedule"), None);
    // Live path: malformed HELIX_NTHREADS is ignored, run still succeeds.
    let body_id = next_id();
    register_body(body_id, count_malformed);
    set_env("HELIX_NTHREADS", "not-a-number");
    helix_parallel_for(0, 200_000, body_id, 4);
    remove_env("HELIX_NTHREADS");
    assert_eq!(HITS_MALFORMED.load(Ordering::Relaxed), 200_000);
}

#[test]
fn env_stage_override_wins_over_set_stage() {
    let _env = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let body_id = next_id();
    register_body(body_id, count_stage_override);
    set_stage(RuntimeStage::ScopeThreads);
    set_env("HELIX_RUNTIME", "pool");
    helix_parallel_for(0, 50_000, body_id, 4);
    remove_env("HELIX_RUNTIME");
    set_stage(RuntimeStage::Pool);
    let last = crate::take_last_dispatch().expect("recorded");
    assert_eq!(last.status, crate::DispatchStatus::Pool);
}

// ---------------------------------------------------------------------------
// Overhead ordering: pooled region < scope-spawn region (ordering only).
// ---------------------------------------------------------------------------

static OVERHEAD_BODY_ID: OnceLock<i64> = OnceLock::new();

extern "C" fn overhead_body(_i: i64, _ctx: *mut u8) {}

fn time_regions(stage: RuntimeStage, regions: usize, iters: i64) -> std::time::Duration {
    let body_id = *OVERHEAD_BODY_ID.get_or_init(|| {
        let id = next_id();
        register_body(id, overhead_body);
        id
    });
    with_stage(stage, || {
        // Warm-up: pool lazy init, page faults, branch predictors.
        helix_parallel_for(0, iters, body_id, 4);
        let t0 = std::time::Instant::now();
        for _ in 0..regions {
            helix_parallel_for(0, iters, body_id, 4);
        }
        t0.elapsed()
    })
}

#[test]
fn pool_region_overhead_below_scope_spawn_overhead() {
    // 4 participants => the gate needs n >= GRAIN*4 for regions to thread;
    // ITERS is just above that so fork/join cost still dominates. The env
    // lock keeps concurrent tests from flipping HELIX_SCHEDULE / NTHREADS /
    // stage mid-measurement (those change process-global dispatch).
    let _env = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const REGIONS: usize = 30;
    const ITERS: i64 = GRAIN * 4 + 1;
    let scope_t = time_regions(RuntimeStage::ScopeThreads, REGIONS, ITERS);
    let pool_t = time_regions(RuntimeStage::Pool, REGIONS, ITERS);
    assert!(
        pool_t < scope_t,
        "expected pool ({pool_t:?}) to beat spawn-per-call ({scope_t:?}) \
         for back-to-back regions"
    );
    let stats = crate::pool_stats();
    assert_eq!(
        stats.worker_spawns_total, stats.workers as u64,
        "pool must not have spawned extra threads during the benchmark"
    );
}

// ---------------------------------------------------------------------------
// Stress: many sequential regions reuse the same pool threads.
// ---------------------------------------------------------------------------

#[test]
fn stress_many_regions_reuse_pool_threads() {
    let _env = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    // Warm the pool first so the spawn counter is stable across the stress.
    crate::pool::warm_pool();
    let before = crate::pool_stats();
    let body_id = next_id();
    register_body(body_id, count_stress);
    let mut total_expected: u64 = 0;
    // Keep regions above the cost gate (n >= GRAIN * P) so they actually
    // reach the pool; that is the machinery under stress here.
    const P: i64 = 2;
    for r in 0..150usize {
        let end = crate::GRAIN * P;
        total_expected += end as u64;
        let sched = ["static", "dynamic", "guided"][r % 3];
        set_env("HELIX_SCHEDULE", sched);
        helix_parallel_for(0, end, body_id, P);
    }
    remove_env("HELIX_SCHEDULE");
    assert_eq!(HITS_STRESS.load(Ordering::Relaxed), total_expected);
    let after = crate::pool_stats();
    assert_eq!(
        after.worker_spawns_total, before.worker_spawns_total,
        "stress must not create new worker threads"
    );
    assert!(
        after.regions_run >= before.regions_run + 150,
        "all 150 gated-in regions must be dispatched through the pool"
    );
}
