//! Region execution kernel shared by both runtime stages.
//!
//! [`drive`] hands one participant its share of `[start, end)`:
//!
//! * **Static** — the participant's precomputed chunk from
//!   [`crate::schedule::static_chunk_for`]; zero shared state.
//! * **Guided / dynamic** — a claim loop over the region's shared
//!   [`ClaimCounter`], exactly libgomp's guided self-scheduling.
//!
//! The body pointer is JIT-compiled code (trusted compiler output). Every
//! call is wrapped in `catch_unwind` so no panic unwinds through an
//! `extern "C"` frame (UB if it did); failures are reported by return value
//! and folded into the region outcome by the stage dispatchers.

use std::panic::{self, AssertUnwindSafe};

use crate::pool::{BodyFn, CtxPtr};
use crate::schedule::{ClaimCounter, SchedKind, static_chunk_for};

/// Scheduling-relevant description of one region, identical for both stages.
#[derive(Clone, Copy)]
pub(crate) struct RegionParams {
    /// Half-open iteration space `[start, end)`.
    pub(crate) start: i64,
    pub(crate) end: i64,
    /// Number of executing participants (dispatcher/coordinator included).
    pub(crate) participants: usize,
    pub(crate) sched: SchedKind,
    /// Lower clamp on dynamic/guided chunk sizes (elements).
    pub(crate) min_chunk: u64,
    pub(crate) body: BodyFn,
    /// Base of the per-participant context area (`null` for plain regions
    /// whose bodies ignore the context).
    pub(crate) ctx_base: CtxPtr,
    /// Bytes between consecutive participants' contexts; `0` = one context
    /// shared by all. Reductions pass [`crate::REDUCTION_CELL_STRIDE`].
    pub(crate) ctx_stride: usize,
}

/// How a dispatched parallel region ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RegionOutcome {
    /// All iterations executed with no unwinds.
    Completed,
    /// At least one body invocation unwound (caught at this boundary; the
    /// caller surfaces it as a clean runtime error, never a cross-ABI panic).
    BodyPanicked,
}

/// Executes one participant's share of the region.
///
/// * `counter` — the region's shared next-index (used only by guided/dynamic;
///   every participant must receive the SAME counter for claims to tile).
/// * `ctx` — context pointer handed to each body invocation.
///
/// Returns false if any body invocation unwound.
pub(crate) fn drive(
    params: &RegionParams,
    counter: &ClaimCounter,
    participant: usize,
    ctx: CtxPtr,
) -> bool {
    let ctx = ctx.0;
    match params.sched {
        SchedKind::Static => {
            let n = params.end.saturating_sub(params.start);
            let chunk = static_chunk_for(params.start, n, params.participants, participant);
            run_range(params.body, chunk.start, chunk.end, ctx)
        }
        sched => {
            let total = params.end.saturating_sub(params.start).max(0) as u64;
            let mut ok = true;
            while let Some((lo, hi)) =
                counter.claim(total, sched, params.min_chunk, params.participants as u64)
            {
                let from = params.start + (lo as i64);
                let to = params.start + (hi as i64);
                ok &= run_range(params.body, from, to, ctx);
            }
            ok
        }
    }
}

/// Runs `body(i, ctx)` for every `i` in `[from, to)`.
fn run_range(body: BodyFn, from: i64, to: i64, ctx: *mut u8) -> bool {
    let mut i = from;
    while i < to {
        // SAFETY: caller (drive/run_range chain) upholds the registry contract.
        if !(unsafe { call_body(body, i, ctx) }) {
            // Stop this participant's stream on first failure; join
            // accounting stays intact because stages decrement unconditionally.
            return false;
        }
        i += 1;
    }
    true
}

/// Calls the JIT body once, catching any unwind before it reaches foreign
/// (`extern "C"` / worker) frames.
///
/// # Safety
/// `body` must be valid to call with these arguments. The registry only ever
/// stores pointers captured from finalized JIT code by the backend — never
/// values derived from user-controllable input — so the contract holds by
/// construction; see [`crate::registry`].
#[inline]
unsafe fn call_body(body: BodyFn, iter: i64, ctx: *mut u8) -> bool {
    // AssertUnwindSafe: the raw fn pointer and byte context carry no Rust
    // state we could observe as poisoned; any unwind is contained wholesale.
    panic::catch_unwind(AssertUnwindSafe(|| body(iter, ctx))).is_ok()
}
