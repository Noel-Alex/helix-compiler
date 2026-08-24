//! Stage A: `std::thread::scope` spawn-per-call dispatch.
//!
//! The reference stage: trivially correct, minimal unsafe beyond the raw body
//! call, and — crucially for the course — it makes fork/join cost
//! *measurable*. On Windows each spawn maps to CreateThread (~50–100 µs incl.
//! CRT/TLS init and stack commit), so a 16-thread region pays ~1–2 ms of fixed
//! cost; the pool stage exists precisely to erase that (research facts 1–3).

use std::panic::{self, AssertUnwindSafe};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::exec::{RegionOutcome, RegionParams, drive};
use crate::pool::CtxPtr;
use crate::schedule::ClaimCounter;

/// Dispatches by spawning `participants - 1` scoped workers, joining on scope
/// exit, and executing participant 0's share inline on the caller.
///
/// Guided/dynamic sharing note: scoped workers cannot reach a single shared
/// counter through the borrow here, so — like the pool path — each participant
/// receives the region counter. For the scope stage we keep the counter inside
/// an `Arc` created per region; claims stay disjoint exactly as in Stage B.
pub(crate) fn run_on_scope_threads(params: RegionParams) -> RegionOutcome {
    let participants = params.participants.max(1);
    let ctx_stride = params.ctx_stride;
    // Strip the raw pointer out of `params` before it is captured by the
    // spawned closures (a bare `*mut u8` is !Send; CtxPtr carries the region
    // safety contract instead). Each participant gets its resolved cell
    // address through a CtxPtr below.
    let ctx_base = params.ctx_base;
    let mut params = params;
    params.participants = participants;
    params.ctx_base = CtxPtr(std::ptr::null_mut());
    let failed = Arc::new(AtomicBool::new(false));
    let counter = Arc::new(ClaimCounter::default());

    // SAFETY (whole function): `ctx_base` is valid for
    // `ctx_stride * (participants - 1) + 1` bytes for this entire call — the
    // dispatcher owns the backing storage (e.g. reduction cells) across it —
    // and every scoped thread is joined before that storage can be released.
    std::thread::scope(|scope| {
        for participant in 1..params.participants {
            let failed = Arc::clone(&failed);
            let counter = Arc::clone(&counter);
            // Resolve the participant's cell address HERE (coordinator side)
            // so only the Send wrapper crosses into the spawned closure.
            // SAFETY: participant < participants keeps the stride offset
            // inside the dispatcher-owned context area.
            let ctx = CtxPtr(unsafe { context_at(ctx_base.0, ctx_stride, participant) });
            scope.spawn(move || {
                if !drive(&params, &counter, participant, ctx) {
                    failed.store(true, Ordering::SeqCst);
                }
            });
        }
        // Coordinator executes participant 0's share itself.
        if !drive(&params, &counter, 0, ctx_base) {
            failed.store(true, Ordering::SeqCst);
        }
    });

    // thread::scope has joined every worker here; per-worker body panics were
    // already caught inside drive(), so `failed` is the only signal needed.
    // (The outer catch_unwind in run_guarded covers runtime-side bugs only.)
    if failed.load(Ordering::SeqCst) {
        return RegionOutcome::BodyPanicked;
    }
    RegionOutcome::Completed
}

/// Context pointer for `participant`, honouring the reduction-cell stride.
///
/// # Safety
/// Same contract as [`crate::pool::RegionState`] contexts: caller guarantees
/// `ctx_base` valid for `ctx_stride * (participants - 1) + 1` bytes.
unsafe fn context_at(ctx_base: *mut u8, ctx_stride: usize, participant: usize) -> *mut u8 {
    if ctx_stride == 0 {
        return ctx_base;
    }
    // SAFETY: stays within the dispatcher-owned area per the contract above;
    // checked_mul rules out offset overflow.
    unsafe {
        ctx_base.add(
            participant
                .checked_mul(ctx_stride)
                .expect("context offset overflow"),
        )
    }
}

/// Panic containment wrapper used by [`crate::dispatch`]: `std::thread::scope`
/// re-raises a child's panic after joining; this converts that into an outcome
/// flag so no unwind ever crosses into JIT frames.
pub(crate) fn run_guarded(params: RegionParams) -> RegionOutcome {
    match panic::catch_unwind(AssertUnwindSafe(|| run_on_scope_threads(params))) {
        Ok(outcome) => outcome,
        Err(_) => RegionOutcome::BodyPanicked,
    }
}
