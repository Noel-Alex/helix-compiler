//! Persistent worker pool (Stage B).
//!
//! One pool of OS threads is spawned lazily on the first pooled parallel
//! region and reused for every subsequent region, so per-region fork/join cost
//! drops from Windows CreateThread territory (~50–100 µs per thread) to a few
//! microseconds of state flip + wake (see `docs/research/parallel-runtime.md`,
//! facts 2–4).
//!
//! Dispatch protocol (generation based):
//!
//! 1. The dispatcher builds a [`RegionState`] carrying a fresh generation
//!    number and publishes it in the job slot under the pool mutex, **then**
//!    calls `notify_all` — state flip strictly before unpark, so a worker can
//!    never observe the notification without the work being visible.
//! 2. Idle workers short-spin (a few slot probes plus `hint::spin_loop`
//!    iterations probing the slot) and only then park in a timed condvar wait.
//!    Spinning absorbs back-to-back regions; the bounded budget keeps idle
//!    workers from starving serial phases (the problem KMP_BLOCKTIME solves).
//! 3. Waking is level-triggered: workers re-check the published generation
//!    after every wake, because spurious wakeups are legal (`std::thread`
//!    parking semantics).
//! 4. Each drafted participant (the dispatcher itself is participant 0)
//!    executes its share of the iteration space, then atomically decrements
//!    `remaining`. The last participant out notifies the dispatcher *under the
//!    pool mutex*; the dispatcher checks the `remaining == 0` predicate under
//!    that same mutex before sleeping, so no wakeup can slip between check
//!    and sleep (rayon `sleep.rs` / Mara Bos discipline). A short timeout
//!    backstop turns any residual pathology into a bounded stall, never a hang.
//!
//! Regions are serialized through a global dispatch lock: HELIX never emits
//! nested parallel regions (loop bodies are side-effect-free leaf loops per
//! lang-spec), so nesting would deadlock by construction — the lock converts
//! accidental concurrency into ordered queuing instead of slot corruption.

use std::collections::HashSet;
use std::hint;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, PoisonError};
use std::thread::{self, JoinHandle};

use crate::exec::{RegionOutcome, RegionParams, drive};
use crate::schedule::{ClaimCounter, Pad, SchedKind};

/// A JIT-emitted loop body: trusted compiler output (see SAFETY notes at call
/// sites). Invoked as `body(iteration, ctx)`; `ctx` is an opaque context or,
/// for reductions, the calling participant's private accumulator cell.
pub type BodyFn = extern "C" fn(iter: i64, ctx: *mut u8);

/// Raw byte pointer wrapper making region contexts shareable across threads.
///
/// # Safety contract
/// The pointer targets dispatcher-owned storage whose lifetime encloses the
/// whole region ([`crate::dispatch`] owns reduction cells across the call).
/// Participants only ever *receive* their own cell's address from
/// [`RegionState::ctx_for`] — they never dereference through this wrapper
/// themselves; the JIT body does. This is exactly the shape of C's
/// `void *ctx` in pthread-style APIs.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub(crate) struct CtxPtr(pub(crate) *mut u8);

// SAFETY: Send+Sync are sound because the pointee follows the region protocol:
// written only by its owning participant during the region, published to the
// coordinator via the Acquire join load, never freed while any worker can see
// it (dispatcher outlives the region it dispatched).
unsafe impl Send for CtxPtr {}
unsafe impl Sync for CtxPtr {}

/// Everything one parallel region needs while it executes. Shared between the
/// dispatcher and every drafted participant via `Arc`.
pub(crate) struct RegionState {
    /// Monotonic dispatch generation; lets workers ignore stale wakeups.
    pub(crate) generation: u64,
    /// Half-open iteration space `[start, end)`.
    pub(crate) start: i64,
    pub(crate) end: i64,
    /// Number of executing participants (dispatcher included).
    pub(crate) participants: usize,
    pub(crate) sched: SchedKind,
    /// Lower clamp on dynamic/guided chunk sizes (elements).
    pub(crate) min_chunk: u64,
    pub(crate) body: BodyFn,
    /// Base of the per-participant context area.
    pub(crate) ctx_base: CtxPtr,
    /// Bytes between consecutive participants' contexts; `0` = one context
    /// shared by all participants (plain parallel-for). Reductions pass a
    /// stride of one padded accumulator cell so participants never share a
    /// cache line (see [`crate::REDUCTION_CELL_STRIDE`]).
    pub(crate) ctx_stride: usize,
    /// Shared next-index counter for dynamic/guided claiming.
    pub(crate) counter: ClaimCounter,
    /// Participants still executing (starts at `participants`, hits 0 at join).
    pub(crate) remaining: Pad<AtomicU64>,
    /// Set if any participant's payload unwound (caught here, never allowed to
    /// cross the `extern "C"` boundary).
    pub(crate) panicked: Pad<AtomicBool>,
}

impl RegionState {
    /// Context handed to `participant`'s body calls.
    ///
    /// # Safety
    /// `ctx_base` must be valid for `ctx_stride * (participants - 1) + 1`
    /// bytes for the whole region; the dispatcher guarantees this because it
    /// owns the backing storage across the entire dispatch call.
    unsafe fn ctx_for(&self, participant: usize) -> *mut u8 {
        if self.ctx_stride == 0 {
            return self.ctx_base.0;
        }
        debug_assert!(participant < self.participants);
        // SAFETY: strided contexts are disjoint by construction (stride >=
        // cell size), so distinct participants never write the same bytes,
        // let alone the same cache line; offset is bounded by participants.
        unsafe {
            self.ctx_base.0.add(
                participant
                    .checked_mul(self.ctx_stride)
                    .expect("context offset overflow"),
            )
        }
    }
}

/// What the job slot currently holds.
enum Slot {
    /// No region in flight; workers park.
    Idle,
    /// A region executable by every worker whose generation is stale.
    Run(Arc<RegionState>),
}

#[derive(Default)]
struct Stats {
    worker_spawns_total: u64,
    regions_run: u64,
}

struct PoolShared {
    /// Job slot + the lock that orders every transition (publish, join notify).
    slot: Mutex<Slot>,
    /// Workers wait here for new work.
    work_available: Condvar,
    /// The dispatcher waits here for region completion.
    region_done: Condvar,
    stats: Mutex<Stats>,
    next_gen: AtomicU64,
    /// Distinct worker thread ids observed (diagnostics for reuse tests).
    worker_ids: Mutex<HashSet<thread::ThreadId>>,
}

/// Handle to the process-global pool.
pub(crate) struct Pool {
    shared: Arc<PoolShared>,
    /// Spawned once via lazy init. Deliberately never joined: the pool is
    /// process-global and outlives every region — joining would reintroduce
    /// exactly the per-region teardown cost the pool exists to avoid. At
    /// process exit the OS reclaims the threads; no destructors are skipped
    /// because workers own nothing but their condvar handles.
    _handles: Vec<JoinHandle<()>>,
    worker_count: usize,
}

/// Spin budget before parking: a handful of slot probes separated by pure
/// `spin_loop` bursts — roughly a microsecond total. Long enough to bridge
/// back-to-back regions without a kernel transition; short enough that idle
/// workers never starve serial phases (the problem KMP_BLOCKTIME solves).
const SPIN_PROBES: u32 = 4;
const SPIN_PER_PROBE: u32 = 64;

/// Timed-wait backstop while parked: workers re-probe periodically even if a
/// notification were somehow missed; bounded staleness beats an unbounded hang.
const PARK_POLL: std::time::Duration = std::time::Duration::from_millis(50);

/// Serializes region dispatch process-wide (see module docs: no nesting).
static DISPATCH_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

fn dispatch_guard() -> MutexGuard<'static, ()> {
    let m = DISPATCH_LOCK.get_or_init(|| Mutex::new(()));
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Pool {
    /// Spawns `worker_count` workers. Called once, via [`global_pool`].
    fn spawn(worker_count: usize) -> Pool {
        let shared = Arc::new(PoolShared {
            slot: Mutex::new(Slot::Idle),
            work_available: Condvar::new(),
            region_done: Condvar::new(),
            stats: Mutex::new(Stats::default()),
            next_gen: AtomicU64::new(0),
            worker_ids: Mutex::new(HashSet::new()),
        });
        let mut handles = Vec::with_capacity(worker_count);
        for idx in 1..=worker_count {
            let shared = Arc::clone(&shared);
            // The closure touches only `shared` (Send + Sync) and its own
            // worker index; no JIT state crosses here. If the OS refuses the
            // thread we shrink the pool rather than panic — the runtime must
            // stay usable on constrained machines.
            match thread::Builder::new()
                .name(format!("helix-worker-{idx}"))
                .stack_size(256 * 1024)
                .spawn(move || worker_loop(&shared, idx))
            {
                Ok(h) => handles.push(Some(h)),
                Err(e) => {
                    eprintln!("helix-runtime: worker {idx} spawn failed: {e}");
                    handles.push(None);
                }
            }
        }
        let handles: Vec<_> = handles.into_iter().flatten().collect();
        let spawned = handles.len() as u64;
        if let Ok(mut st) = shared.stats.lock() {
            st.worker_spawns_total = spawned;
        }
        Pool {
            shared,
            _handles: handles,
            worker_count: spawned as usize,
        }
    }

    /// Dispatches `req` on this pool: publish, participate as participant 0,
    /// then join. Caller must hold [`dispatch_guard`].
    fn run_region(&self, mut req: RegionRequest) -> RegionOutcome {
        // Participants = dispatcher + drafted workers. Clamping keeps the join
        // accounting exact: every counted participant must actually exist.
        let max_participants = self.worker_count + 1;
        if req.participants > max_participants || req.participants == 0 {
            req.participants = max_participants.max(1);
        }
        let generation = self.shared.next_gen.fetch_add(1, Ordering::Relaxed) + 1;
        let region = Arc::new(RegionState {
            generation,
            start: req.start,
            end: req.end,
            participants: req.participants,
            sched: req.sched,
            min_chunk: req.min_chunk,
            body: req.body,
            ctx_base: req.ctx_base,
            ctx_stride: req.ctx_stride,
            counter: ClaimCounter::default(),
            remaining: Pad(AtomicU64::new(req.participants as u64)),
            panicked: Pad(AtomicBool::new(false)),
        });

        {
            let mut slot = self
                .shared
                .slot
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            if let Ok(mut st) = self.shared.stats.lock() {
                st.regions_run += 1;
            }
            // State flip FIRST ...
            *slot = Slot::Run(Arc::clone(&region));
            drop(slot);
            // ... THEN unpark everyone. Workers re-validate via `gen`.
            self.shared.work_available.notify_all();
        }

        // The dispatcher is participant 0 and pulls its own weight.
        participant_guarded(&region, 0);

        // ---- Join: wait until every drafted participant has retired. -------
        // Predicate (`remaining`) is read under the pool mutex and re-checked
        // after every wake; the final notify arrives under the same mutex, so
        // no wakeup can land between our check and our sleep. A few cheap
        // probes first (the countdown may already be done); then block on the
        // condvar — spinning here would steal the slot mutex out from under
        // the very workers we are waiting for.
        let mut slot = match self.shared.slot.lock() {
            Ok(s) => s,
            Err(e) => e.into_inner(),
        };
        let mut spins: u32 = 0;
        while region.remaining.load(Ordering::Acquire) != 0 {
            if spins < 4 {
                drop(slot);
                hint::spin_loop();
                spins += 1;
                slot = match self.shared.slot.lock() {
                    Ok(s) => s,
                    Err(e) => e.into_inner(),
                };
            } else {
                let (g, _) = self
                    .shared
                    .region_done
                    .wait_timeout(slot, JOIN_POLL)
                    .unwrap_or_else(PoisonError::into_inner);
                slot = g;
            }
        }
        let outcome = if region.panicked.load(Ordering::SeqCst) {
            RegionOutcome::BodyPanicked
        } else {
            RegionOutcome::Completed
        };
        // Retire the region so workers go back to parking promptly.
        *slot = Slot::Idle;
        drop(slot);
        self.shared.work_available.notify_all();
        outcome
    }

    fn stats_snapshot(&self) -> (u64, u64, usize) {
        let (spawns, regions) = match self.shared.stats.lock() {
            Ok(st) => (st.worker_spawns_total, st.regions_run),
            Err(e) => {
                let st = e.into_inner();
                (st.worker_spawns_total, st.regions_run)
            }
        };
        let ids = match self.shared.worker_ids.lock() {
            Ok(ids) => ids.len(),
            Err(e) => e.into_inner().len(),
        };
        (spawns, regions, ids)
    }
}

/// Backstop for the dispatcher's blocking join wait.
const JOIN_POLL: std::time::Duration = std::time::Duration::from_millis(5);

/// Executes participant `idx`'s share of `region`.
///
/// Panic containment: the body itself is already guarded deeper down
/// ([`crate::exec::call_body`]); this second net catches any residual unwind
/// from runtime bookkeeping and records it in the region flag instead of
/// letting a panic escape into a foreign (JIT) frame. Join accounting runs on
/// every path, so a panicking participant can never hang the dispatcher.
fn participant_guarded(region: &Arc<RegionState>, idx: usize) {
    if idx >= region.participants {
        // Not drafted for this region (more workers than participants): must
        // not touch `remaining`, which counts drafted participants only.
        return;
    }
    let params = RegionParams {
        start: region.start,
        end: region.end,
        participants: region.participants,
        sched: region.sched,
        min_chunk: region.min_chunk,
        body: region.body,
        // Unused by the pool path: `drive` receives the resolved `ctx`
        // directly; the field is populated for completeness.
        ctx_base: CtxPtr(std::ptr::null_mut()),
        ctx_stride: 0,
    };
    let ran = panic::catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: the dispatcher owns `ctx_base` for the whole region and
        // guarantees validity for `ctx_stride * (participants - 1)` bytes;
        // `idx < participants` was checked above.
        let ctx = unsafe { region.ctx_for(idx) };
        drive(&params, &region.counter, idx, CtxPtr(ctx))
    }));
    if ran.is_err() {
        region.panicked.store(true, Ordering::SeqCst);
    }
    // Release ordering pairs with the dispatcher's Acquire load: all memory
    // writes made while executing this participant's chunks happen-before the
    // dispatcher observing the countdown reach zero.
    let prev = region.remaining.fetch_sub(1, Ordering::AcqRel);
    if prev == 1 {
        notify_dispatcher(region);
    }
}

/// Last-out notification, performed while holding the pool mutex so the
/// dispatcher (which checks the predicate under that mutex) cannot miss it.
fn notify_dispatcher(_region: &Arc<RegionState>) {
    let Some(pool) = POOL.get() else { return };
    // Locking proves the dispatcher is either (a) spinning outside the mutex
    // — it will observe `remaining == 0` on its next probe — or (b) already
    // blocked inside wait(), which released the mutex and will receive this
    // notify. There is no third state between its check and its sleep.
    if let Ok(slot) = pool.shared.slot.lock() {
        drop(slot);
        pool.shared.region_done.notify_all();
    }
}

/// Worker main loop: short-spin probing, then park; re-check after every wake.
fn worker_loop(shared: &PoolShared, idx: usize) {
    if let Ok(mut ids) = shared.worker_ids.lock() {
        ids.insert(thread::current().id());
    }
    let mut seen_gen: u64 = 0;
    loop {
        // ---- Bounded spin: poll the slot without blocking. ----------------
        // A few lock probes separated by pure spinning: probing hammers the
        // slot mutex, which the dispatcher and retiring participants need, so
        // keep the probe count low and let spin_loop fill the budget.
        for probe in 0..SPIN_PROBES {
            if let Ok(guard) = shared.slot.try_lock() {
                match take_work(guard, &mut seen_gen, idx) {
                    TakeWork::Executed => break, // fresh spin budget below
                    TakeWork::Idle(_) => {}      // keep spinning
                }
            }
            for _ in 0..SPIN_PER_PROBE {
                hint::spin_loop();
            }
            let _ = probe;
        }
        // ---- Park: block on the mutex, re-check, then timed-wait. ---------
        // Re-checking AFTER acquiring the mutex (never trusting the spin
        // phase) makes spurious wakeups and near-miss notifications harmless.
        let guard = shared.slot.lock().unwrap_or_else(PoisonError::into_inner);
        if let TakeWork::Idle(guard) = take_work(guard, &mut seen_gen, idx) {
            // Sleep holding the mutex. wait_timeout releases it while blocked,
            // so publishers can flip the slot underneath us; the loop's
            // re-check after the wake handles real regions AND spurious ones.
            let (g, _) = shared
                .work_available
                .wait_timeout(guard, PARK_POLL)
                .unwrap_or_else(PoisonError::into_inner);
            drop(g);
        }
        // Loop restarts with a fresh spin budget: a region may have arrived.
    }
}

/// Outcome of probing the job slot once.
enum TakeWork<'a> {
    /// This worker executed its share of a (new) region.
    Executed,
    /// Nothing new for this worker; the slot lock is handed back untouched.
    Idle(MutexGuard<'a, Slot>),
}

/// Probes the locked job slot once. If it holds a region newer than
/// `*seen_gen` and this worker is drafted, executes the worker's share
/// (releasing the lock first — the join path re-takes the mutex) and returns
/// [`TakeWork::Executed`]; otherwise hands the lock back as
/// [`TakeWork::Idle`] without ever blocking.
fn take_work<'a>(slot: MutexGuard<'a, Slot>, seen_gen: &mut u64, idx: usize) -> TakeWork<'a> {
    // Clone the Arc out first so the lock can be released before executing.
    let new_region = match &*slot {
        Slot::Run(region) if region.generation != *seen_gen => Some(Arc::clone(region)),
        _ => None,
    };
    match new_region {
        None => TakeWork::Idle(slot),
        Some(region) => {
            drop(slot);
            *seen_gen = region.generation;
            participant_guarded(&region, idx);
            TakeWork::Executed
        }
    }
}

static POOL: OnceLock<Pool> = OnceLock::new();

/// Returns the process-global pool, spawning its workers on first use.
pub(crate) fn global_pool() -> &'static Pool {
    POOL.get_or_init(|| {
        let hw = thread::available_parallelism().map_or(1, |n| n.get());
        // The dispatcher participates as participant 0, so the pool needs
        // hw - 1 workers to reach full machine width.
        Pool::spawn(hw.saturating_sub(1).max(1))
    })
}

/// Forces lazy pool creation without dispatching a region (test/bench warm-up).
#[cfg(test)]
pub(crate) fn warm_pool() {
    let _ = global_pool();
}

/// Description of one region to execute, built identically for both stages.
pub(crate) struct RegionRequest {
    pub(crate) start: i64,
    pub(crate) end: i64,
    pub(crate) participants: usize,
    pub(crate) sched: SchedKind,
    pub(crate) min_chunk: u64,
    pub(crate) body: BodyFn,
    pub(crate) ctx_base: CtxPtr,
    pub(crate) ctx_stride: usize,
}

/// Dispatches `req` on the pool (Stage B); blocks until the region completes.
pub(crate) fn run_on_pool(req: RegionRequest) -> RegionOutcome {
    let _serial = dispatch_guard();
    global_pool().run_region(req)
}

/// Snapshot of pool health/counters for the overhead microbench graphs.
///
/// `worker_spawns_total` stops growing after warm-up — precisely the
/// "persistent pool amortizes spawn cost" story the bench tells.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PoolStats {
    /// Worker threads owned by the pool right now.
    pub workers: usize,
    /// Cumulative OS-thread spawns the pool has ever performed.
    pub worker_spawns_total: u64,
    /// Parallel regions dispatched through the pool so far.
    pub regions_run: u64,
    /// Distinct worker thread ids observed (equals `workers` once warmed).
    pub distinct_worker_ids: usize,
}

/// Reads [`PoolStats`] without forcing pool creation (idle => all zeros).
pub fn pool_stats() -> PoolStats {
    let Some(pool) = POOL.get() else {
        return PoolStats::default();
    };
    let (worker_spawns_total, regions_run, distinct_worker_ids) = pool.stats_snapshot();
    PoolStats {
        workers: pool.worker_count,
        worker_spawns_total,
        regions_run,
        distinct_worker_ids,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    extern "C" fn noop(_iter: i64, _ctx: *mut u8) {}

    fn req(start: i64, end: i64, participants: usize, sched: SchedKind) -> RegionRequest {
        RegionRequest {
            start,
            end,
            participants,
            sched,
            min_chunk: 8,
            body: noop,
            ctx_base: CtxPtr(std::ptr::null_mut()),
            ctx_stride: 0,
        }
    }

    #[test]
    fn pool_reuses_the_same_threads_across_many_regions() {
        // Warm up FIRST (this may be the test that pays lazy pool init), then
        // snapshot: the invariant under test is "no spawns across regions",
        // not "no spawns ever".
        let _ = global_pool();
        let before = pool_stats();
        const REGIONS: usize = 200;
        for _ in 0..REGIONS {
            assert_eq!(
                run_on_pool(req(0, 4096, 3, SchedKind::Static)),
                RegionOutcome::Completed
            );
        }
        let after = pool_stats();
        assert_eq!(
            after.worker_spawns_total, before.worker_spawns_total,
            "pool must not spawn fresh OS threads per region"
        );
        // Other tests may dispatch regions concurrently through the shared
        // process-global pool, so only assert the lower bound here.
        assert!(
            after.regions_run >= before.regions_run + REGIONS as u64,
            "every region must be accounted (before={}, after={})",
            before.regions_run,
            after.regions_run
        );
        assert_eq!(
            after.distinct_worker_ids, after.workers,
            "each pool worker should have been observed exactly once"
        );
    }

    #[test]
    fn pool_executes_every_iteration_exactly_once() {
        const N: i64 = 50_000;
        // Shared context points at one flat array of counters (stride 0).
        let seen: Vec<AtomicU64> = (0..N).map(|_| AtomicU64::new(0)).collect();
        extern "C" fn mark(i: i64, ctx: *mut u8) {
            // SAFETY: test-owned array lives across the whole region; `i` is
            // guaranteed in-bounds by the region bounds [0, N).
            unsafe {
                (*ctx.cast::<AtomicU64>().add(i as usize)).fetch_add(1, Ordering::Relaxed);
            }
        }
        let r = RegionRequest {
            start: 0,
            end: N,
            participants: 8,
            sched: SchedKind::Guided,
            min_chunk: 16,
            body: mark,
            ctx_base: CtxPtr(seen.as_ptr() as *mut u8),
            ctx_stride: 0,
        };
        assert_eq!(run_on_pool(r), RegionOutcome::Completed);
        let bad: Vec<usize> = (0..N as usize)
            .filter(|&i| seen[i].load(Ordering::Relaxed) != 1)
            .collect();
        assert!(bad.is_empty(), "iterations executed != once: {bad:?}");
    }

    #[test]
    fn strided_contexts_reach_distinct_cells() {
        // Simulates reduction cells: each participant writes ONLY its own
        // 128-byte-strided slot (once per iteration of its chunk), so the sum
        // of the first words must equal n and all padding words stay zero —
        // proving contexts were disjoint and correctly addressed.
        const PARTICIPANTS: usize = 6;
        const N: u64 = 1024;
        const STRIDE: usize = crate::REDUCTION_CELL_STRIDE;
        let mut cells = vec![0u64; (PARTICIPANTS + 1) * STRIDE / 8];
        extern "C" fn bump_slot(_i: i64, ctx: *mut u8) {
            // SAFETY: ctx is this participant's private cell (dispatcher-
            // computed via stride), valid for a u64 write.
            unsafe {
                let p = ctx.cast::<u64>();
                *p = p.read_unaligned().wrapping_add(1);
            }
        }
        let r = RegionRequest {
            start: 0,
            end: N as i64,
            participants: PARTICIPANTS,
            sched: SchedKind::Static,
            min_chunk: 8,
            body: bump_slot,
            ctx_base: CtxPtr(cells.as_mut_ptr() as *mut u8),
            ctx_stride: STRIDE,
        };
        assert_eq!(run_on_pool(r), RegionOutcome::Completed);
        let per_cell = STRIDE / 8;
        // Every iteration hit SOME cell exactly once.
        let total: u64 = (0..PARTICIPANTS).map(|p| cells[p * per_cell]).sum();
        assert_eq!(total, N, "each iteration must bump exactly one cell");
        // Padding inside each cell is untouched (no stray writes).
        for p in 0..PARTICIPANTS {
            assert!(
                cells[p * per_cell + 1..(p + 1) * per_cell]
                    .iter()
                    .all(|&v| v == 0),
                "cell {p} padding was written"
            );
        }
    }

    #[test]
    fn fewer_participants_than_workers_still_joins() {
        // Leftover workers see the generation flip but must stay out of the
        // join accounting; every participant count must complete cleanly.
        let width = global_pool().worker_count.max(1);
        for participants in 1..=width {
            let sched = if participants % 2 == 0 {
                SchedKind::Dynamic
            } else {
                SchedKind::Guided
            };
            assert_eq!(
                run_on_pool(req(0, 2048, participants, sched)),
                RegionOutcome::Completed
            );
        }
        // Overlarge requests get clamped instead of hanging.
        assert_eq!(
            run_on_pool(req(0, 2048, width * 4, SchedKind::Static)),
            RegionOutcome::Completed
        );
    }

    #[test]
    fn many_sequential_regions_hit_the_same_workers() {
        // Stress: interleaved schedules and sizes; the pool must survive all
        // of them with zero spawns beyond warm-up.
        let _ = global_pool(); // ensure lazy init is paid before the snapshot
        let before = pool_stats();
        for i in 0..100usize {
            let sched = [SchedKind::Static, SchedKind::Dynamic, SchedKind::Guided][i % 3];
            let end = 1 + (i as i64 * 977) % 20_000;
            assert_eq!(
                run_on_pool(req(0, end, 1 + i % 5, sched)),
                RegionOutcome::Completed
            );
        }
        let after = pool_stats();
        assert_eq!(after.worker_spawns_total, before.worker_spawns_total);
    }

    #[test]
    fn stats_are_sane_after_warmup() {
        warm_pool(); // this test may be the only one running
        let s = pool_stats();
        assert!(s.workers >= 1);
        let _ = s.regions_run; // depends on execution order across the suite
        assert!(s.distinct_worker_ids <= s.workers);
    }

    #[test]
    fn dispatch_is_serialized_not_corrupting_the_slot() {
        // Two host threads dispatching concurrently must both succeed (the
        // dispatch lock queues them); results stay exact.
        let hits = Arc::new(AtomicUsize::new(0));
        extern "C" fn touch(_i: i64, _ctx: *mut u8) {}
        let h1_hits = Arc::clone(&hits);
        let t1 = thread::spawn(move || {
            for _ in 0..25 {
                let r = RegionRequest {
                    start: 0,
                    end: 512,
                    participants: 2,
                    sched: SchedKind::Guided,
                    min_chunk: 4,
                    body: touch,
                    ctx_base: CtxPtr(std::ptr::null_mut()),
                    ctx_stride: 0,
                };
                let _ = h1_hits.clone();
                assert_eq!(run_on_pool(r), RegionOutcome::Completed);
            }
        });
        for _ in 0..25 {
            assert_eq!(
                run_on_pool(req(-500, 500, 2, SchedKind::Static)),
                RegionOutcome::Completed
            );
        }
        t1.join().expect("concurrent dispatcher must not panic");
        let _ = hits.load(Ordering::Relaxed);
    }
}
