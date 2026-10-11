//! The fixed ABI functions the module ships: the FNV folder, the bump
//! allocator, the run wrapper, `hx_checksum`, and `hx_heap_bytes`.
//!
//! They are emitted as literal byte sequences because they are small, fixed,
//! and read better as the operation they perform than as a chain of builder
//! calls. `hx_run` is the only part of the ABI that is not exported as itself:
//! it is a thin reset wrapper around the lowered `fn main()`, so a second run
//! of the same instance starts from the interpreter's starting state.

use super::codegen::{
    ELEMS_OFFSET, FN_FOLD, FOLD_TABLE_BASE, FOLD_TABLE_STRIDE, GLOBAL_BYTES, GLOBAL_CHECKSUM,
    GLOBAL_FOLDS, GLOBAL_HEAP, IMPORT_TRAP, TRAP_ALLOC, TRAP_NEGATIVE_ZEROS,
};
use super::encode::Body;

/// FNV-1a offset basis — the interpreter's starting state.
pub(super) const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

/// `(i32 ptr, i32 len) -> ()`: folds `len` bytes at `ptr` into the checksum.
///
/// Exported as `hx_fold`; the browser calls it once per printed line (after
/// rendering the line with the canonical formatter, which is why the stream
/// matches the interpreter's byte for byte), and `hx_checksum` uses it for the
/// recorded array payloads too.
pub(super) fn fold_bytes() -> Vec<u8> {
    let mut body = Body::new(2);
    body.declare_local(0x7F); // $i : i32
    let i = 2;

    body.block();
    body.loop_block();
    body.local_get(i);
    body.local_get(1);
    body.raw(&[0x4F]); // i32.ge_u — unsigned, so a stray length exits at once
    body.branch_if(1); // out of the block when i >=u len

    body.global_get(GLOBAL_CHECKSUM);
    body.local_get(0);
    body.local_get(i);
    body.raw(&[0x6A]); // i32.add
    body.raw(&[0x2D, 0x00, 0x00]); // i32.load8_u, align 0, offset 0
    body.raw(&[0xAD]); // i64.extend_i32_u
    body.raw(&[0x85]); // i64.xor
    body.global_set(GLOBAL_CHECKSUM);

    body.global_get(GLOBAL_CHECKSUM);
    body.i64_const(0x0000_0100_0000_01b3);
    body.raw(&[0x7E]); // i64.mul
    body.global_set(GLOBAL_CHECKSUM);

    body.local_get(i);
    body.i32_const(1);
    body.raw(&[0x6A]); // i32.add
    body.local_set(i);
    body.branch(0);
    body.end(); // loop
    body.end(); // block
    body.finish()
}

/// `(i64 count, i32 elem_size) -> i32`: the bump allocator. The callers
/// (`zeros` lowering) already hold the widened length, so it comes first.
///
/// Returns the base of a fresh header + elements region. A negative length and
/// an exhausted address space both go through the host trap hook, so the
/// browser renders the interpreter's message instead of a wasm fault.
pub(super) fn alloc() -> Vec<u8> {
    let mut body = Body::new(2);
    let base = body.declare_local(0x7F); // $base : i32
    let total = body.declare_local(0x7F); // $total : i32
    let total64 = body.declare_local(0x7E); // $total : i64, before narrowing

    // Negative length: the interpreter's `zeros(n)` trap.
    body.local_get(0);
    body.i64_const(0);
    body.raw(&[0x53]); // i64.lt_s
    body.if_block();
    body.i32_const(TRAP_NEGATIVE_ZEROS);
    body.local_get(0); // a = n
    body.i64_const(0);
    body.call(IMPORT_TRAP);
    body.end();

    // total = 8 + count * elem_size, kept in i64 so it cannot wrap; a value
    // past the 4 GiB address space is an allocation failure.
    body.local_get(0);
    body.local_get(1);
    body.raw(&[0xAD]); // i64.extend_i32_u
    body.raw(&[0x7E]); // i64.mul
    body.i64_const(i64::from(ELEMS_OFFSET));
    body.raw(&[0x7C]); // i64.add
    body.local_tee(total64);
    body.i64_const(i64::from(u32::MAX));
    body.raw(&[0x56]); // i64.gt_u
    body.if_block();
    body.i32_const(TRAP_ALLOC);
    body.i64_const(0);
    body.i64_const(0);
    body.call(IMPORT_TRAP);
    body.end();
    body.local_get(total64);
    body.wrap_i64();
    body.local_set(total);

    // Grow memory until `base + total` bytes fit. Fresh pages arrive zeroed,
    // but a reset run reuses memory that still holds the previous run's data.
    body.global_get(GLOBAL_HEAP);
    body.local_set(base);
    body.block();
    body.loop_block();
    body.local_get(base);
    body.local_get(total);
    body.raw(&[0x6A]); // i32.add
    body.raw(&[0x3F, 0x00]); // memory.size
    body.i32_const(16);
    body.raw(&[0x74]); // i32.shl
    body.raw(&[0x4D]); // i32.le_u
    body.branch_if(1);
    body.i32_const(1);
    body.raw(&[0x40, 0x00]); // memory.grow
    body.i32_const(-1);
    body.raw(&[0x46]); // i32.eq
    body.if_block();
    body.i32_const(TRAP_ALLOC);
    body.local_get(base);
    body.raw(&[0xAD]); // i64.extend_i32_u
    body.i64_const(0);
    body.call(IMPORT_TRAP);
    body.end();
    body.branch(0);
    body.end(); // loop
    body.end(); // block

    // Commit the bump pointer and the allocation counter.
    body.global_get(GLOBAL_BYTES);
    body.local_get(total);
    body.raw(&[0x6A]); // i32.add
    body.global_set(GLOBAL_BYTES);
    body.local_get(base);
    body.local_get(total);
    body.raw(&[0x6A]); // i32.add
    body.global_set(GLOBAL_HEAP);

    // Header: the element count.
    body.local_get(base);
    body.local_get(0);
    body.raw(&[0xA7]); // i32.wrap_i64
    body.raw(&[0x36, 0x02, 0x00]); // i32.store, align 2, offset 0

    // Elements: zeroed explicitly, because the region may be recycled from a
    // previous run of the same instance.
    body.local_get(base);
    body.i32_const(ELEMS_OFFSET);
    body.raw(&[0x6A]); // i32.add
    body.i32_const(0);
    body.local_get(total);
    body.i32_const(ELEMS_OFFSET);
    body.raw(&[0x6B]); // i32.sub
    body.raw(&[0xFC, 0x0B, 0x00]); // memory.fill

    body.local_get(base);
    body.finish()
}

/// `() -> ()`: resets every piece of run state, then runs `fn main()`.
///
/// The reset is what makes repeated runs of one instantiated module
/// comparable: the heap starts over, the checksum restarts from the FNV basis,
/// and the recorded final arrays are dropped. Exported as `hx_run`.
pub(super) fn run(main: u32) -> Vec<u8> {
    let mut body = Body::new(0);
    body.i32_const(HEAP_BASE);
    body.global_set(GLOBAL_HEAP);
    body.i64_const(FNV_OFFSET as i64);
    body.global_set(GLOBAL_CHECKSUM);
    body.i32_const(0);
    body.global_set(GLOBAL_BYTES);
    body.i32_const(0);
    body.global_set(GLOBAL_FOLDS);
    body.call(main);
    body.finish()
}

/// `() -> i64`: folds the final arrays `main` recorded, then returns the
/// checksum.
///
/// Recording at `main`'s exit and folding here is what pins the byte order:
/// the interpreter folds every printed line first (the host does that through
/// `hx_fold` as the lines arrive) and only then the final arrays.
pub(super) fn checksum() -> Vec<u8> {
    let mut body = Body::new(0);
    let i = body.declare_local(0x7F); // $i : i32
    let count = body.declare_local(0x7F); // $count : i32
    let entry = body.declare_local(0x7F); // $entry : i32
    let ptr = body.declare_local(0x7F); // $ptr : i32
    let len = body.declare_local(0x7F); // $len : i32

    body.global_get(GLOBAL_FOLDS);
    body.local_set(count);
    body.i32_const(0);
    body.local_set(i);

    body.block();
    body.loop_block();
    body.local_get(i);
    body.local_get(count);
    body.raw(&[0x4F]); // i32.ge_u
    body.branch_if(1);

    // entry = fold table base + i * stride
    body.i32_const(FOLD_TABLE_BASE);
    body.local_get(i);
    body.i32_const(FOLD_TABLE_STRIDE);
    body.raw(&[0x6C]); // i32.mul
    body.raw(&[0x6A]); // i32.add
    body.local_set(entry);

    body.local_get(entry);
    body.raw(&[0x28, 0x02, 0x00]); // i32.load — element base
    body.local_set(ptr);
    body.local_get(entry);
    body.i32_const(4);
    body.raw(&[0x6A]); // i32.add
    body.raw(&[0x28, 0x02, 0x00]); // i32.load — byte length
    body.local_set(len);

    body.local_get(ptr);
    body.local_get(len);
    body.call(FN_FOLD);

    body.local_get(i);
    body.i32_const(1);
    body.raw(&[0x6A]); // i32.add
    body.local_set(i);
    body.branch(0);
    body.end(); // loop
    body.end(); // block

    body.global_get(GLOBAL_CHECKSUM);
    body.finish()
}

/// `() -> i32`: bytes allocated by the current run.
pub(super) fn heap_bytes() -> Vec<u8> {
    let mut body = Body::new(0);
    body.global_get(GLOBAL_BYTES);
    body.finish()
}

/// First byte above the host's print scratch and the fold table: where the
/// bump allocator starts.
pub(super) const HEAP_BASE: i32 = 8192;
