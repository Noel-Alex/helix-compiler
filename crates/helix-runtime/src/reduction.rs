//! Reduction support: per-thread accumulator cells combined serially.
//!
//! Layout follows the research digest (facts 8–10, recommendation 4): one
//! `#[repr(align(128))]` cell per participant so adjacent accumulators never
//! share a cache line — unpadded arrays measurably cost 2–10x to false
//! sharing — and a trivial O(P) serial fold after join. No atomics ever touch
//! the hot loop; FP combination order is unspecified exactly as OpenMP's
//! "order of combination is unspecified" clause documents.
//!
//! The cells hold raw bytes: the JIT body knows the element width and writes
//! its participant's slot directly; [`fold`] folds slots pairwise through the
//! registered combine fn (`dst = combine(dst, src)`).

use crate::registry::CombineFn;

/// Byte stride between per-participant reduction accumulator cells.
///
/// 128 bytes (not 64): Intel spatial prefetchers pull line pairs, so 64-byte
/// spacing still lets neighbours interfere; this matches crossbeam
/// `CachePadded`'s N on x86-64/aarch64.
pub const CELL_STRIDE: usize = 128;

/// Byte offset of the accumulator field INSIDE a participant cell (word 0 of
/// the cell holds the shared-context pointer). The backend's emitter bakes
/// the same constant (`CELL_ACC_OFF` in helix-backend/src/parallel.rs); the
/// two must stay in sync — a layout drift here would fold the wrong words.
pub(crate) const ACC_OFFSET: usize = 8;

/// A region of `participants` private accumulator cells.
///
/// Invariant: cell `p` occupies bytes `[p * CELL_STRIDE, (p+1) * CELL_STRIDE)`
/// of `storage`; distinct cells never share an address, let alone a line.
///
/// Production dispatch passes the JIT program's own accumulator area straight
/// through (see [`crate::helix_parallel_reduction`] FFI contract); this helper
/// remains for callers that want runtime-owned, guaranteed-128-aligned cells
/// and for the layout tests below.
#[cfg(test)]
pub(crate) struct CellArea {
    /// `(participants + 1) * CELL_STRIDE` bytes; the leading slack lets
    /// [`CellArea::base_ptr`] advance to a true 128-boundary.
    storage: Vec<u64>,
}

#[cfg(test)]
impl CellArea {
    /// Allocates zeroed cells for `participants` slots (>= 1).
    pub(crate) fn new(participants: usize) -> CellArea {
        let participants = participants.max(1);
        // u64 words so the base is naturally 8-aligned. Vec's allocator gives
        // alignment >= align_of(u64), so we over-allocate one extra cell and
        // advance to a true 128-boundary in [`CellArea::base_ptr`].
        let words = participants * (CELL_STRIDE / 8);
        CellArea {
            storage: vec![0u64; words + CELL_STRIDE / 8],
        }
    }

    /// Base pointer handed to the dispatcher (`ctx_base`).
    ///
    /// # Safety (callers)
    /// The returned pointer is valid for `CELL_STRIDE * participants` bytes
    /// while `self` is alive and unmutated.
    pub(crate) fn base_ptr(&mut self) -> *mut u8 {
        let head = self.storage.as_mut_ptr() as *mut u8;
        let addr = head as usize;
        let misalign = addr % CELL_STRIDE;
        if misalign == 0 {
            head
        } else {
            // SAFETY: we allocated one extra cell (128 bytes) of slack, so
            // advancing to the next 128-boundary stays within the allocation.
            unsafe { head.add(CELL_STRIDE - misalign) }
        }
    }
}

/// Folds every non-coordinator cell into cell 0 with `combine`, serially.
///
/// # Safety
/// `base` must be the pointer returned by [`CellArea::base_ptr`] for an area
/// of `participants` cells, and `combine` must accept two pointers valid for
/// the element's width — both hold when called from [`crate::dispatch`] with
/// backend-registered combine fns.
pub(crate) unsafe fn fold(base: *mut u8, participants: usize, combine: CombineFn) {
    for p in 1..participants.max(1) {
        // SAFETY: p < participants keeps each cell inside the area; combine is
        // trusted compiler output registered by the backend. Both pointers
        // target the ACCUMULATOR FIELD at +ACC_OFFSET — passing raw cell bases
        // would combine word 0 (the shared-ctx pointer) instead and corrupt it.
        combine(
            // SAFETY: cell 0's accumulator field, inside the area.
            unsafe { base.add(ACC_OFFSET) }, // dst: cell 0 acc
            // SAFETY: p < participants keeps the cell inside the area.
            unsafe {
                base.add(p.checked_mul(CELL_STRIDE).expect("cell offset overflow") + ACC_OFFSET)
            }, // src: cell p acc
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    extern "C" fn add_i64(dst: *mut u8, src: *const u8) {
        // SAFETY: test-owned i64 cells, both valid for the element width.
        unsafe {
            *(dst as *mut i64) += *(src as *const i64);
        }
    }

    #[test]
    fn cells_are_128_aligned_and_disjoint() {
        for participants in [1usize, 2, 7, 33] {
            let mut area = CellArea::new(participants);
            let base = area.base_ptr();
            assert_eq!(base as usize % 128, 0, "cell base must be 128-aligned");
            for p in 0..participants {
                let cell = unsafe { base.add(p * CELL_STRIDE) } as usize;
                assert_eq!(cell % 128, 0, "cell {p} misaligned");
            }
        }
    }

    #[test]
    fn serial_combine_sums_partials_in_cell_zero() {
        const P: usize = 9;
        let mut area = CellArea::new(P);
        let base = area.base_ptr();
        // Write each cell's partial into its ACCUMULATOR FIELD (offset 8 —
        // the same layout the backend's dispatcher and JIT body use).
        for p in 0..P {
            // SAFETY: p < P keeps the write inside the allocated area.
            unsafe {
                *(base.add(p * CELL_STRIDE + ACC_OFFSET) as *mut i64) = p as i64 + 1;
            }
        }
        // SAFETY: test-owned area + registered-safe combine.
        unsafe { fold(base, P, add_i64) };
        let total: i64 = (1..=P as i64).sum();
        // SAFETY: cell 0 accumulator readback.
        unsafe {
            assert_eq!(*(base.add(ACC_OFFSET) as *mut i64), total);
            // Word 0 (the shared-ctx slot) must be untouched by folding.
            assert_eq!(base.add(ACC_OFFSET - 8).cast::<u64>().read(), 0);
        }
        // Other cells are untouched (fold reads them, never writes).
        for p in 1..P {
            // SAFETY: in-bounds cell read.
            unsafe {
                assert_eq!(
                    *(base.add(p * CELL_STRIDE + ACC_OFFSET) as *mut i64),
                    p as i64 + 1
                );
            }
        }
    }

    #[test]
    fn single_participant_area_combines_to_itself() {
        let mut area = CellArea::new(1);
        let base = area.base_ptr();
        // SAFETY: single-cell write/read at the accumulator offset.
        unsafe {
            *(base.add(ACC_OFFSET) as *mut i64) = 5;
            fold(base, 1, add_i64);
            assert_eq!(*(base.add(ACC_OFFSET) as *mut i64), 5);
        }
    }

    #[test]
    fn cell_stride_constant_matches_docs() {
        assert_eq!(CELL_STRIDE, 128);
        assert_eq!(crate::REDUCTION_CELL_STRIDE, CELL_STRIDE);
    }
}
