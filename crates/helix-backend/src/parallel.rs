//! Parallel-region lowering: loop-body extraction, context packing, and the
//! host dispatch layer between JITed code and `helix-runtime`.
//!
//! ## What gets transformed
//!
//! HELIX's builder emits one exact CFG per source `for` (see
//! `helix_ir::build::Builder::for_stmt`): a *preheader* computing the induction
//! cell and the end bound, a *header* comparing `iv < end`, a *body*, a *latch*
//! running `iv = iv + 1`, and an *exit*. After SSA construction the header
//! hosts the φs of every loop-carried variable.
//!
//! The transform keeps the parent function structurally IDENTICAL and swaps
//! exactly one thing: when the header's terminator is translated, the lowering
//! emits instead
//!
//! ```text
//! 1. stash:  every captured scalar and array fat pointer into host tables,
//! 2. call:   handle = helix_dispatch(start, end, region_id, nthreads),
//! 3. readback (reductions): accumulator = helix_read_W(handle),
//! 4. jump:   straight to the loop exit (feeding its φs).
//! ```
//!
//! All remaining parent blocks that were only reachable THROUGH the loop are
//! skipped as orphans (terminated with a dummy return), so the emitted machine
//! code contains no dead loop.
//!
//! The loop *body* becomes its own tiny SSA [`FuncIr`] named after
//! `RegionDesc::body_fn_name`, lowered by the UNMODIFIED [`crate::lower`] into
//! an `extern "C"` function taking two I64 parameters `(iteration, ctx)`:
//!
//! * the induction variable is parameter 0;
//! * captured scalars and array fat pointers are loaded from fixed offsets of
//!   the packed context in the entry block (reusing the original SSA ids as
//!   load destinations, so almost nothing else needs remapping);
//! * the old latch terminates with `return`;
//! * for reductions the accumulator's per-iteration incoming value is loaded
//!   from the calling participant's PRIVATE cell and the chain's result is
//!   stored back there at the top of the latch — the chain itself remains
//!   native CLIF, preserving exact integer wraparound and IEEE semantics.
//!
//! Because the extracted IR is ordinary SSA `FuncIr`, every existing guarantee
//! carries over untouched: bounds/division guards, the shared panic block,
//! IEEE min/max selection, saturating casts.
//!
//! ## Context memory model
//!
//! One 8-byte-aligned host allocation per region execution:
//!
//! ```text
//! [ shared ctx: L bytes ][ cell 0 (128 B) ][ cell 1 (128 B) ] …
//! ```
//!
//! * shared ctx — word 0 reserved, word 1 the reduction readback slot, then
//!   two words per array slot `(data, len)`, then one word per captured
//!   scalar (integers sign-extended to i64, floats widened to f64 — both
//!   lossless).
//! * cell `p` — participant `p`'s private 128-byte-strided area
//!   ([`helix_runtime::REDUCTION_CELL_STRIDE`]): word 0 holds the shared-ctx
//!   pointer (written by the dispatcher before the fork), offset 8 the
//!   private accumulator (seeded by the dispatcher with the monoid seed).
//!   Bodies dereference word 0 to reach shared data, so false sharing is
//!   confined to the accumulator line.
//!
//! Every region is dispatched through
//! [`helix_runtime::helix_parallel_reduction`] — DoAll regions simply carry a
//! no-op combine and no readback. One code path through the runtime; the plain
//! `helix_parallel_for` entry (whose serial path passes NULL contexts) stays
//! reserved for future ctx-free regions. When the runtime's cost gate fires
//! (or `HELIX_NTHREADS=1`) everything executes inline on participant 0:
//! integer/min/max results stay bit-exact vs sequential by associativity, and
//! FP add/mul totals only reassociate under genuine multi-threaded dispatch.
//!
//! ## Sequential-fallback gate
//!
//! The analysis plan approves loops; [`prepare`] re-checks each region against
//! what this transform can express and demotes anything else to the plain
//! sequential lowering (the empty-plan path, untouched): prints, user calls,
//! `return` inside the loop, mirrored/non-`Lt` comparisons, second
//! loop-carried variables, bool-typed captures, edges leaving the loop, or a
//! nested region inside the candidate. Demotion makes "approved but
//! unsupported" a performance question, never a correctness one.

use std::collections::HashMap;
use std::collections::{HashMap as StdMap, HashSet};
use std::sync::LazyLock;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Mutex, MutexGuard};

use helix_analysis::{ReductionOp as AnOp, RegionDesc};
use helix_ir::{BinOp, BlockId, FuncIr, Inst, Load, LocalId, Phi, Term, ValueId};

/// Thread-count hint baked into every dispatch call.
///
/// The runtime clamps hints to available parallelism and honours the
/// `HELIX_NTHREADS` override, so a generous constant is safe.
pub(crate) const NTHREADS_HINT: i64 = 8;

/// Byte offset of the shared-ctx pointer inside a participant cell.
#[allow(dead_code)] // self-documenting counterpart of the accumulator offset
const CELL_CTX_OFF: i64 = 0;
/// Byte offset of the private accumulator inside a participant cell.
const CELL_ACC_OFF: i64 = 8;

// ---------------------------------------------------------------------------
// Host-symbol registration (engine side)
// ---------------------------------------------------------------------------

/// Every dispatch-layer symbol exposed to JITed code, with its host address.
pub(crate) fn host_symbols() -> Vec<(&'static str, *const u8)> {
    vec![
        (
            "helix_stash_i",
            helix_stash_i as extern "C" fn(i64, i64) as *const u8,
        ),
        (
            "helix_stash_f",
            helix_stash_f as extern "C" fn(i64, f64) as *const u8,
        ),
        (
            "helix_stash_arr",
            helix_stash_arr as extern "C" fn(i64, i64, i64) as *const u8,
        ),
        (
            "helix_dispatch",
            helix_dispatch as extern "C" fn(i64, i64, i64, i64) -> i64 as *const u8,
        ),
        (
            "helix_read_i64",
            helix_read_i64 as extern "C" fn(i64) -> i64 as *const u8,
        ),
        (
            "helix_read_i32",
            helix_read_i32 as extern "C" fn(i64) -> i32 as *const u8,
        ),
        (
            "helix_read_f64",
            helix_read_f64 as extern "C" fn(i64) -> f64 as *const u8,
        ),
        (
            "helix_read_f32",
            helix_read_f32 as extern "C" fn(i64) -> f32 as *const u8,
        ),
        // Body-context imports (called from extracted bodies).
        (
            "helix_ld_i64",
            helix_ld_i64 as extern "C" fn(i64, i64) -> i64 as *const u8,
        ),
        (
            "helix_ld_i32",
            helix_ld_i32 as extern "C" fn(i64, i64) -> i32 as *const u8,
        ),
        (
            "helix_ld_f64",
            helix_ld_f64 as extern "C" fn(i64, i64) -> f64 as *const u8,
        ),
        (
            "helix_ld_f32",
            helix_ld_f32 as extern "C" fn(i64, i64) -> f32 as *const u8,
        ),
        (
            "helix_acc_load_i64",
            helix_acc_load_i64 as extern "C" fn(i64) -> i64 as *const u8,
        ),
        (
            "helix_acc_load_i32",
            helix_acc_load_i32 as extern "C" fn(i64) -> i32 as *const u8,
        ),
        (
            "helix_acc_load_f64",
            helix_acc_load_f64 as extern "C" fn(i64) -> f64 as *const u8,
        ),
        (
            "helix_acc_load_f32",
            helix_acc_load_f32 as extern "C" fn(i64) -> f32 as *const u8,
        ),
        (
            "helix_acc_store_i64",
            helix_acc_store_i64 as extern "C" fn(i64, i64) as *const u8,
        ),
        (
            "helix_acc_store_i32",
            helix_acc_store_i32 as extern "C" fn(i64, i32) as *const u8,
        ),
        (
            "helix_acc_store_f64",
            helix_acc_store_f64 as extern "C" fn(i64, f64) as *const u8,
        ),
        (
            "helix_acc_store_f32",
            helix_acc_store_f32 as extern "C" fn(i64, f32) as *const u8,
        ),
    ]
}

/// Names of every dispatch-ABI symbol JITed code can import — the
/// parent-side stash/dispatch/readback set plus the body-context imports
/// emitted inside extracted region bodies.
pub const HOST_SYMBOL_NAMES: &[&str] = &[
    "helix_stash_i",
    "helix_stash_f",
    "helix_stash_arr",
    "helix_dispatch",
    "helix_read_i64",
    "helix_read_i32",
    "helix_read_f64",
    "helix_read_f32",
    "helix_ld_i64",
    "helix_ld_i32",
    "helix_ld_f64",
    "helix_ld_f32",
    "helix_acc_load_i64",
    "helix_acc_load_i32",
    "helix_acc_load_f64",
    "helix_acc_load_f32",
    "helix_acc_store_i64",
    "helix_acc_store_i32",
    "helix_acc_store_f64",
    "helix_acc_store_f32",
];
/// Byte offset of the reduction readback slot in the shared ctx.
const CTX_ACC_OFF: i64 = 8;
/// First byte of the array-slot area in the shared ctx (words 0–1 reserved).
const CTX_ARRAY_BASE: i64 = 16;

// ---------------------------------------------------------------------------
// Region metadata (shared by the extractor, the engine and the dispatcher)
// ---------------------------------------------------------------------------

/// Execution flavour of one region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RKind {
    /// Independent iterations; results live only in memory.
    DoAll,
    /// Per-thread partials combined after the join.
    Reduction(RSpec),
}

/// Everything the combine/readback path needs about one reduction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RSpec {
    /// Associative operator (analysis view).
    pub(crate) op: AnOp,
    /// Accumulator byte width (4 or 8).
    pub(crate) width: u64,
    /// True when the accumulator is floating point.
    pub(crate) float: bool,
}

/// CLIF signature of every extracted body: `extern "C" fn(i64 iter, i64 ctx)`.
///
/// Both parameters ride as I64 (the ctx pointer is carried as an integer, the
/// established convention in this backend), WindowsFastcall == Rust `extern "C"`.
#[must_use]
pub fn body_signature() -> cranelift::codegen::ir::Signature {
    use cranelift::codegen::ir::{AbiParam, Signature};
    let mut s = Signature::new(crate::lower::CALL_CONV);
    s.params
        .push(AbiParam::new(cranelift::codegen::ir::types::I64));
    s.params
        .push(AbiParam::new(cranelift::codegen::ir::types::I64));
    s
}

/// Byte layout of the shared ctx, computed once at extraction and consumed by
/// the emitter (offsets baked into CLIF), the dispatcher (packing) and the
/// readback imports.
#[derive(Clone, Debug, Default)]
pub(crate) struct CtxLayout {
    /// `(array local, element byte size, offset of the data pointer)` — the
    /// length half sits at `offset + 8`.
    pub(crate) arrays: Vec<(LocalId, i64, i64)>,
    /// `(SSA id, slot width, is_float, byte offset)` — one 8-byte word per
    /// captured scalar.
    pub(crate) scalars: Vec<(u32, u64, bool, i64)>,
}

impl CtxLayout {
    /// Total packed size in bytes (always a positive multiple of 8).
    pub(crate) fn len_bytes(&self) -> i64 {
        self.scalars
            .last()
            .map_or(CTX_ARRAY_BASE, |s| s.3 + 8)
            .max(CTX_ARRAY_BASE)
    }

    /// Word position of scalar `k` (relative to the start of the ctx).
    fn scalar_word(k: usize) -> usize {
        (CTX_ARRAY_BASE / 8) as usize + k
    }
}

/// One prepared region handed to the engine and the parent-function emitter.
pub(crate) struct PlannedRegion {
    /// Unique integer baked into the parent's dispatch call; also the
    /// registry key for the body and the combine fn.
    pub(crate) region_id: i64,
    /// Index of the owning function in the compiled slice.
    pub(crate) func_idx: usize,
    /// Header block of the replaced loop.
    pub(crate) header: BlockId,
    /// False-successor of the header: where control resumes after dispatch.
    pub(crate) exit: BlockId,
    /// Symbol name of the extracted body (`RegionDesc::body_fn_name`).
    pub(crate) body_fn_name: String,
    /// Extracted body IR; `None` = demoted, lower sequentially.
    pub(crate) body: Option<FuncIr>,
    /// Array fat-pointer prebindings for the body's lowering.
    pub(crate) array_prebind: Vec<(LocalId, ValueId, ValueId)>,
    /// Kind + operator (refined during extraction).
    pub(crate) kind: RKind,
    /// Packed-ctx layout.
    pub(crate) layout: CtxLayout,
    /// Parent-side SSA id of the start bound (`None` = constant below).
    pub(crate) start_ssa: Option<u32>,
    /// Parent-side SSA id of the end bound (`None` = constant below).
    pub(crate) end_ssa: Option<u32>,
    /// Constant start bound when known.
    pub(crate) start_const: Option<i64>,
    /// Constant end bound when known.
    pub(crate) end_const: Option<i64>,
    /// Parent SSA name overwritten with the combined total (reductions only).
    pub(crate) acc_dst: Option<u32>,
    /// Reduction seed: `(scalar word position, width, is_float)`.
    pub(crate) seed: Option<(usize, u64, bool)>,
    /// Blocks the region spans (nesting-demotion checks).
    pub(crate) covers: Vec<BlockId>,
}

impl PlannedRegion {
    /// Dispatcher metadata snapshot for [`register_spec`].
    pub(crate) fn spec_meta(&self) -> SpecMeta {
        SpecMeta {
            len_bytes: self.layout.len_bytes(),
            acc: match self.kind {
                RKind::DoAll => None,
                RKind::Reduction(s) => Some((s.width, s.float)),
            },
            seed: self.seed,
        }
    }
}

/// Emitter-side hook attached to one parent function's lowering.
#[derive(Clone)]
pub(crate) struct RtHook {
    /// Header block whose terminator becomes the dispatch sequence.
    pub(crate) header: BlockId,
    /// Where control resumes after the dispatch.
    pub(crate) exit: BlockId,
    /// Region id baked into the dispatch call.
    pub(crate) region_id: i64,
    /// Kind (drives the readback).
    pub(crate) kind: RKind,
    /// Array slots in layout order.
    pub(crate) arrays: Vec<(LocalId, i64, i64)>,
    /// Captured scalars in layout order.
    pub(crate) scalars: Vec<(u32, u64, bool, i64)>,
    /// Start bound: SSA id or constant.
    pub(crate) start_ssa: Option<u32>,
    pub(crate) start_const: i64,
    /// End bound: SSA id or constant.
    pub(crate) end_ssa: Option<u32>,
    pub(crate) end_const: i64,
    /// SSA name overwritten with the combined total.
    pub(crate) acc_dst: Option<u32>,
}

impl RtHook {
    pub(crate) fn of(r: &PlannedRegion) -> RtHook {
        RtHook {
            header: r.header,
            exit: r.exit,
            region_id: r.region_id,
            kind: r.kind,
            arrays: r.layout.arrays.clone(),
            scalars: r.layout.scalars.clone(),
            start_ssa: r.start_ssa,
            start_const: r.start_const.unwrap_or(0),
            end_ssa: r.end_ssa,
            end_const: r.end_const.unwrap_or(0),
            acc_dst: r.acc_dst,
        }
    }
}

// ---------------------------------------------------------------------------
// Plan preparation
// ---------------------------------------------------------------------------

/// Prepares the whole plan: extracts every region, demoting candidates whose
/// shape is unsupported or that contains another region (nested parallelism
/// would deadlock the runtime's dispatch lock).
pub(crate) fn prepare(plan: &crate::ParallelPlan, program: &[FuncIr]) -> Vec<PlannedRegion> {
    // The backend's seam type mirrors the analysis descriptor 1:1 (contract
    // addendum 2); lift it into the analysis view this module consumes.
    let lifted = helix_analysis::ParallelPlan {
        regions: plan
            .regions
            .iter()
            .map(|r| RegionDesc {
                func_idx: r.func_idx,
                header: r.header,
                kind: match r.kind {
                    crate::RegionKind::DoAll => helix_analysis::RegionKind::DoAll,
                    crate::RegionKind::Reduction(op) => {
                        helix_analysis::RegionKind::Reduction(to_an(op))
                    }
                },
                reduction: match r.kind {
                    crate::RegionKind::Reduction(op) => Some(to_an(op)),
                    crate::RegionKind::DoAll => None,
                },
                body_fn_name: r.body_fn_name.clone(),
                // The backend's seam descriptor predates the bound fields
                // (contract addendum 2); bounds are recovered from the IR in
                // `extract` instead.
                start_val: None,
                end_val: None,
            })
            .collect(),
    };
    let mut out: Vec<PlannedRegion> = Vec::new();
    for desc in &lifted.regions {
        let Some(ir) = program.get(desc.func_idx) else {
            continue;
        };
        let mut planned = PlannedRegion {
            region_id: next_region_id(),
            func_idx: desc.func_idx,
            header: desc.header,
            exit: desc.header,
            body_fn_name: desc.body_fn_name.clone(),
            body: None,
            array_prebind: Vec::new(),
            kind: match desc.kind {
                helix_analysis::RegionKind::DoAll => RKind::DoAll,
                helix_analysis::RegionKind::Reduction(op) => RKind::Reduction(RSpec {
                    op,
                    width: 0,
                    float: false,
                }),
            },
            layout: CtxLayout::default(),
            start_ssa: None,
            end_ssa: None,
            start_const: None,
            end_const: None,
            acc_dst: None,
            seed: None,
            covers: Vec::new(),
        };
        if extract(ir, desc, &mut planned).is_some() {
            out.push(planned);
        }
    }

    // Nested-region demotion: a region whose span contains another region's
    // header loses (the inner one keeps the parallelism).
    let demote: Vec<usize> = out
        .iter()
        .enumerate()
        .filter(|&(i, outer)| {
            outer.body.is_some()
                && out.iter().enumerate().any(|(j, inner)| {
                    i != j
                        && inner.body.is_some()
                        && inner.func_idx == outer.func_idx
                        && outer.covers.contains(&inner.header)
                })
        })
        .map(|(i, _)| i)
        .collect();
    for i in demote {
        out[i].body = None;
    }
    out.retain(|r| r.body.is_some());
    out
}

/// Maps the backend stub operator onto the analysis operator.
fn to_an(op: crate::engine::helix_analysis_stub::ReductionOp) -> AnOp {
    match op {
        crate::engine::helix_analysis_stub::ReductionOp::Add => AnOp::Add,
        crate::engine::helix_analysis_stub::ReductionOp::Mul => AnOp::Mul,
        crate::engine::helix_analysis_stub::ReductionOp::Min => AnOp::Min,
        crate::engine::helix_analysis_stub::ReductionOp::Max => AnOp::Max,
    }
}

/// Debug helper: logs a demotion gate (temporary).
fn gate<T>(_: T, line: u32) -> Option<T> {
    eprintln!("gate hit at line {line}");
    None
}

/// Extracts the body of `desc`'s loop from `ir`, filling `planned`'s metadata.
/// Returns `None` when the shape is unsupported (caller drops the region).
///
/// Shape discovered here (post-SSA canonical `for`): the header branches
/// `iv < end ? body : exit`; the latch is the single in-loop predecessor of
/// the header; the header hosts one φ per carried variable — the induction φ
/// plus (for reductions) exactly one accumulator φ.
pub(crate) fn extract(ir: &FuncIr, desc: &RegionDesc, planned: &mut PlannedRegion) -> Option<()> {
    let hdr = desc.header;
    let header = ir.block(hdr);

    // ---- skeleton -----------------------------------------------------------
    let Term::Branch {
        cond,
        t: body_b,
        f: exit_b,
    } = &header.term
    else {
        return gate((), line!());
    };
    let (body_b, exit_b) = (*body_b, *exit_b);

    // Latch: exactly one in-loop predecessor besides the header itself.
    let loop_set = natural_loop_of(ir, hdr)?;
    let mut latches: Vec<BlockId> = ir
        .preds(hdr)
        .iter()
        .copied()
        .filter(|p| *p != hdr && loop_set.contains(p))
        .collect();
    if latches.len() != 1 {
        return gate((), line!());
    }
    let latch = latches.pop().expect("checked len");
    if !loop_set.contains(&body_b) || exit_b == hdr {
        return gate((), line!());
    }
    // A `return` inside the loop would skip iterations in the extracted body.
    for b in &loop_set {
        if matches!(ir.block(*b).term, Term::Return(_)) {
            return gate((), line!());
        }
    }

    // ---- iv φ + comparison operands ------------------------------------------
    let Some(Inst::Bin {
        op: BinOp::Lt,
        a,
        b,
        ..
    }) = ir.inst_defining(*cond)
    else {
        return gate((), line!());
    };
    let mut iv_phi: Option<&Phi> = None;
    let mut end_candidates = 0usize;
    let mut end_val: Option<ValueId> = None;
    for cand in [*a, *b] {
        if let Some(p) = header
            .phis
            .iter()
            .find(|p| p.dst == cand && p.args.len() == 2)
        {
            if iv_phi.is_some() {
                return gate((), line!());
            }
            iv_phi = Some(p);
        } else {
            end_candidates += 1;
            end_val = Some(cand);
        }
    }
    if end_candidates != 1 || iv_phi.is_none() {
        return gate((), line!());
    }
    let iv_phi: &Phi = iv_phi.expect("checked above");
    let end_val: ValueId = end_val.expect("checked above");
    let iv_dst = iv_phi.dst;
    let start_val: ValueId = iv_phi
        .args
        .iter()
        .find(|(from, _)| *from != hdr && !loop_set.contains(from))
        .map(|(_, v)| *v)?;

    // ---- accumulator φ (reductions) -------------------------------------------
    let mut spec = match planned.kind {
        RKind::DoAll => None,
        RKind::Reduction(s) => Some(s),
    };
    let acc: Option<(ValueId, LocalId, ValueId)> = if spec.is_none() {
        // DoAll: no carried scalar other than the iv may survive.
        if header
            .phis
            .iter()
            .any(|p| p.dst != iv_dst && p.args.len() >= 2)
        {
            return gate((), line!());
        }
        None
    } else {
        let s = spec.expect("checked above");
        let cands: Vec<&Phi> = header
            .phis
            .iter()
            .filter(|p| p.dst != iv_dst && p.args.len() == 2)
            .collect();
        if cands.len() != 1 {
            return gate((), line!());
        }
        let phi = cands[0];
        let entry_arg: ValueId = phi
            .args
            .iter()
            .find(|(from, _)| *from != hdr && !loop_set.contains(from))
            .map(|(_, v)| *v)?;
        let back_arg: ValueId = phi
            .args
            .iter()
            .find(|(from, _)| loop_set.contains(from))
            .map(|(_, v)| *v)?;
        // The chain shape must match the planned operator.
        let chain = ir.inst_defining(back_arg)?;
        let ok = match (s.op, chain) {
            (
                AnOp::Add,
                Inst::Bin {
                    op: BinOp::Add | BinOp::Sub,
                    ..
                },
            )
            | (AnOp::Mul, Inst::Bin { op: BinOp::Mul, .. }) => true,
            (AnOp::Min, Inst::Call(c)) => c.callee == "min",
            (AnOp::Max, Inst::Call(c)) => c.callee == "max",
            _ => false,
        };
        if !ok {
            return gate((), line!());
        }
        let ty = ir.types.local_ty(phi.var)?;
        let (width, float) = match ty {
            helix_sema::Ty::I64 => (8u64, false),
            helix_sema::Ty::I32 => (4, false),
            helix_sema::Ty::F64 => (8, true),
            helix_sema::Ty::F32 => (4, true),
            _ => return None,
        };
        spec = Some(RSpec {
            op: s.op,
            width,
            float,
        });
        Some((phi.dst, phi.var, entry_arg))
    };

    // ---- arrays + scalar uses inside the loop ----------------------------------
    let mut arrs: Vec<LocalId> = Vec::new();
    let mut uses: Vec<ValueId> = Vec::new();
    for blk in &loop_set {
        let bd = ir.block(*blk);
        for inst in &bd.insts {
            match inst {
                Inst::Load(l) => arrs.push(l.arr),
                Inst::Store { arr, .. } => arrs.push(*arr),
                Inst::Call(c) => {
                    // Pure/host-side builtins travel; anything else (user
                    // calls, prints) may touch foreign state — demote.
                    match c.callee.as_str() {
                        "min" | "max" | "sqrt" | "abs" | "len" => {}
                        "zeros" if c.arr_refs.len() == 1 && c.dst.is_none() => {}
                        _ => return None,
                    }
                }
                _ => {}
            }
            uses.extend(inst.uses());
        }
        if let Term::Branch { cond, .. } = &bd.term {
            uses.push(*cond);
        }
        for p in &bd.phis {
            uses.extend(p.args.iter().map(|(_, v)| *v));
        }
    }
    let mut seen_arr = HashSet::new();
    arrs.retain(|l| seen_arr.insert(l.0));

    // ---- classify scalar uses: internal vs captured ----------------------------
    let acc_names: Vec<ValueId> = acc.as_ref().map(|&(d, _, _)| vec![d]).unwrap_or_default();
    let mut captures: Vec<ValueId> = Vec::new();
    let mut seen_cap = HashSet::new();
    for v in uses {
        if v == iv_dst || acc_names.contains(&v) || !seen_cap.insert(v) {
            continue;
        }
        if loop_set.contains(&ir.def_block(v)) {
            // Defined in-loop: travels with the cloned instructions — unless
            // it is another carried φ (a second loop-carried variable).
            if loop_set
                .iter()
                .any(|&blk| ir.block(blk).phis.iter().any(|p| p.dst == v))
            {
                return gate((), line!());
            }
            continue;
        }
        // Outside-defined → captured by value. Bools cannot ride the I64 ABI.
        if matches!(ir.val_ty(v), Some(helix_sema::Ty::Bool)) {
            return gate((), line!());
        }
        captures.push(v);
    }
    // Reduction seed: the accumulator's initial value rides along too.
    if let Some((_, _, entry_arg)) = acc
        && seen_cap.insert(entry_arg)
    {
        if matches!(ir.val_ty(entry_arg), Some(helix_sema::Ty::Bool)) {
            return gate((), line!());
        }
        captures.push(entry_arg);
    }
    captures.sort_unstable();

    // ---- ctx layout --------------------------------------------------------------
    let mut layout = CtxLayout::default();
    let mut next_off = CTX_ARRAY_BASE;
    for arr in &arrs {
        let esz = match ir.types.elem_ty(*arr) {
            Some(helix_sema::ElemTy::I32 | helix_sema::ElemTy::F32) => 4,
            Some(_) => 8,
            None => return None, // non-array local on the array path
        };
        layout.arrays.push((*arr, esz, next_off));
        next_off += 16;
    }
    for v in &captures {
        let ty = ir.val_ty(*v)?;
        let (w, f) = match ty {
            helix_sema::Ty::I64 => (8u64, false),
            helix_sema::Ty::I32 => (4, false),
            helix_sema::Ty::F64 => (8, true),
            helix_sema::Ty::F32 => (8, true), // widened; payload stays lossless
            _ => return None,
        };
        layout.scalars.push((v.0, w, f, next_off));
        next_off += 8;
    }

    // Reduction seed bookkeeping: which scalar word holds the seed.
    let seed = acc.and_then(|(_, _, entry_arg)| {
        layout
            .scalars
            .iter()
            .position(|(sid, _, _, _)| *sid == entry_arg.0)
            .map(|k| {
                let (_, w, f, _) = layout.scalars[k];
                (CtxLayout::scalar_word(k), w, f)
            })
    });

    // ---- bounds -----------------------------------------------------------------
    let bound_of = |v: ValueId| match ir.const_of(v) {
        Some(c) => (None, Some(c)),
        None => (Some(v.0), None),
    };
    let (start_ssa, start_const) = bound_of(start_val);
    let (end_ssa, end_const) = bound_of(end_val);

    // ---- build the body IR ---------------------------------------------------------
    let (body, prebind) = build_body_ir(ir, hdr, &loop_set, latch, iv_phi, &acc, spec, &layout)?;

    planned.exit = exit_b;
    planned.body = Some(body);
    planned.array_prebind = prebind;
    planned.layout = layout;
    planned.kind = match spec {
        Some(s) => RKind::Reduction(s),
        None => RKind::DoAll,
    };
    planned.start_ssa = start_ssa;
    planned.end_ssa = end_ssa;
    planned.start_const = start_const;
    planned.end_const = end_const;
    planned.acc_dst = acc.map(|(d, _, _)| d.0);
    planned.seed = seed;
    planned.covers = loop_set;
    Some(())
}

/// Maximum array slots supported per region (layout/id-space guard).
const MAX_ARRAY_SLOTS: usize = 1 << 20;

/// One compiled region body plus its array prebinding list, as handed from
/// extraction to the engine's definition pass.
pub(crate) struct BodyArtifact {
    /// Symbol name to declare/define the body under.
    pub(crate) name: String,
    /// The extracted body IR.
    pub(crate) ir: FuncIr,
    /// `(array local, ptr id, len id)` prebindings for the lowering.
    pub(crate) prebind: Vec<(LocalId, ValueId, ValueId)>,
}

/// Array prebinding triple used across extraction and lowering.
pub(crate) type Prebind = Vec<(LocalId, ValueId, ValueId)>;

/// Sentinel block id of the body's trailing exit-return block (appended after
/// all cloned loop blocks). Edges that would leave the loop jump here.
const EXIT_RET_BLOCK: BlockId = BlockId(u32::MAX);

/// Builds the standalone body function for the loop subgraph.
///
/// Id discipline: the iv φ's destination id and every captured scalar's id are
/// REUSED as the destinations of the entry ctx-loads, so the vast majority of
/// cloned operands need no remapping at all. Array fat-pointer halves get
/// fresh ids from the same extended space; remaining in-loop definitions get
/// fresh ids above `max(next_value, max_value_id + 1)`.
///
/// Returns the body IR plus its array prebinding list `(local, ptr id,
/// len id)` for the parent-side lowering.
#[allow(clippy::too_many_arguments)]
fn build_body_ir(
    ir: &FuncIr,
    hdr: BlockId,
    loop_set: &[BlockId],
    latch: BlockId,
    iv_phi: &Phi,
    acc: &Option<(ValueId, LocalId, ValueId)>,
    spec: Option<RSpec>,
    layout: &CtxLayout,
) -> Option<(FuncIr, Prebind)> {
    if layout.arrays.len() > MAX_ARRAY_SLOTS {
        return None; // too many array slots
    }

    // ---- fresh id space -------------------------------------------------------
    let mut nv = ir.next_value.max(ir.max_value_id() + 1);

    // Widened side tables (fresh rows default to I64; patched below).
    let mut val_tys = ir.types.val_tys.clone();
    val_tys.resize(nv as usize, helix_sema::Ty::I64);
    let mut local_tys = ir.types.local_tys.clone();
    let mut local_names = ir.types.local_names.clone();

    // Compiler-temporary local slot binding the ctx-pointer parameter.
    let ctx_local = LocalId(ir.n_locals as u32);
    local_tys.push(helix_sema::Ty::I64);
    local_names.push("$par_ctx".into());

    // The ctx pointer's SSA id (entry φ #2 → function parameter 1).
    let ctx_v = ValueId(nv);
    nv += 1;
    val_tys.resize(nv as usize, helix_sema::Ty::I64);

    // ---- entry preamble --------------------------------------------------------
    // Order: array fat pointers, captured scalars, accumulator load. Array
    // halves get fresh ids from the SAME extended space as everything else
    // (ids without val_tys rows would fail IR verification).
    let mut prebind: Vec<(LocalId, ValueId, ValueId)> = Vec::with_capacity(layout.arrays.len());
    let mut entry: Vec<Inst> = Vec::new();
    for (local, _, off) in &layout.arrays {
        // One shared offset constant per array slot pair.
        let off_v0 = ValueId(nv);
        nv += 1;
        val_tys.resize(nv as usize, helix_sema::Ty::I64);
        entry.push(Inst::Const {
            dst: off_v0,
            c: helix_ir::Constant::I64(*off),
        });
        let pv = ValueId(nv);
        nv += 1;
        let lv = ValueId(nv);
        nv += 1;
        val_tys.resize(nv as usize, helix_sema::Ty::I64);
        entry.push(ld_call(pv, ctx_v, off_v0));
        // The length half sits one word later; reuse the same constant plus a
        // second const for +8.
        let off_v8 = ValueId(nv);
        nv += 1;
        val_tys.resize(nv as usize, helix_sema::Ty::I64);
        entry.push(Inst::Const {
            dst: off_v8,
            c: helix_ir::Constant::I64(*off + 8),
        });
        entry.push(ld_call(lv, ctx_v, off_v8));
        prebind.push((*local, pv, lv));
    }
    for (sid, w, f, off) in &layout.scalars {
        // Offset constant then the typed load.
        let off_v = ValueId(nv);
        nv += 1;
        val_tys.resize(nv as usize, helix_sema::Ty::I64);
        entry.push(Inst::Const {
            dst: off_v,
            c: helix_ir::Constant::I64(*off),
        });
        entry.push(Inst::Call(helix_ir::Call {
            dst: Some(ValueId(*sid)),
            callee: ld_sym(*w, *f).into(),
            args: vec![ctx_v, off_v],
            arr_refs: Vec::new(),
        }));
    }
    // Accumulator incoming value ← this participant's private cell. The seed
    // itself was written by the HOST before the fork (the body entry runs once
    // per ITERATION, so seeding here would clobber the accumulator).
    if let (Some((acc_dst, _, _)), Some(s)) = (acc, spec) {
        entry.push(Inst::Call(helix_ir::Call {
            dst: Some(*acc_dst),
            callee: acc_sym(s.width, s.float).into(),
            args: vec![ctx_v],
            arr_refs: Vec::new(),
        }));
    }

    // Latch preamble: store the chain result back into the private cell.
    let mut latch_store: Vec<Inst> = Vec::new();
    if let (Some((_, _, back)), Some(s)) = (acc, spec.map(|s| (s.width, s.float))) {
        let sym = acc_store_sym(s.0, s.1);
        latch_store.push(Inst::Call(helix_ir::Call {
            dst: None,
            callee: sym.into(),
            args: vec![ctx_v, *back],
            arr_refs: Vec::new(),
        }));
    }

    // ---- assemble the FuncIR ----------------------------------------------------
    let mut body = FuncIr::new("__extracted__", helix_sema::Ty::Unit, ir.n_source_locals);
    body.next_value = nv;
    body.types.val_tys = val_tys;
    body.types.local_tys = local_tys;
    body.types.local_names = local_names;

    // Blocks in ascending original-id order; the header becomes the entry.
    // FuncIr::new pre-creates block 0 as the entry — REUSE it for the header
    // clone instead of appending a fresh block after it.
    let mut order: Vec<BlockId> = loop_set.to_vec();
    order.sort_unstable_by_key(|b| b.0);
    debug_assert_eq!(
        order[0], hdr,
        "header must be the lowest-id loop block (canonical `for` shape)"
    );
    let mut bmap: HashMap<BlockId, BlockId> = HashMap::new();
    for (k, b) in order.iter().enumerate() {
        let nb = if k == 0 {
            body.entry // pre-created entry block
        } else {
            body.new_block()
        };
        bmap.insert(*b, nb);
    }

    // Operand remapper: pre-bound ids map to themselves; everything else gets
    // a fresh id on first mention.
    let mut vmap: HashMap<ValueId, ValueId> = HashMap::new();
    for (sid, _, _, _) in &layout.scalars {
        vmap.insert(ValueId(*sid), ValueId(*sid));
    }
    if let Some((d, _, _)) = acc {
        vmap.insert(*d, *d);
    }
    vmap.insert(iv_phi.dst, iv_phi.dst);
    vmap.insert(ctx_v, ctx_v);
    for (_, pv, lv) in &prebind {
        vmap.insert(*pv, *pv);
        vmap.insert(*lv, *lv);
    }

    for b in &order {
        let src = ir.block(*b);
        let nb = bmap[b];

        let mut phis_out: Vec<Phi> = Vec::new();
        let mut insts_out: Vec<Inst> = Vec::new();

        if *b == hdr {
            // Entry bindings replace ALL header phis: iv → parameter 0, ctx
            // pointer → parameter 1 (zero-arg entry phis are the lowering's
            // parameter convention). Every other header φ was rejected above.
            phis_out.push(Phi {
                dst: iv_phi.dst,
                var: iv_phi.var,
                args: Vec::new(),
            });
            phis_out.push(Phi {
                dst: ctx_v,
                var: ctx_local,
                args: Vec::new(),
            });
            insts_out.append(&mut entry);
            // The header's own instructions (the iv/end comparison) travel.
            for inst in &src.insts {
                insts_out.push(clone_inst(inst, &mut vmap, &mut nv)?);
            }
            // The header φ's back-edge argument (accumulator chain result /
            // incremented iv) is NOT redefined here: the chain lives in the
            // body blocks, the iv is the iteration parameter.
        } else {
            // Inner phis join control flow INSIDE the loop: clone them.
            for p in &src.phis {
                let ndst = map_id(p.dst, &mut vmap, &mut nv);
                let mut args = Vec::with_capacity(p.args.len());
                for (from, v) in &p.args {
                    let nf = *bmap.get(from)?;
                    args.push((nf, map_id(*v, &mut vmap, &mut nv)));
                }
                phis_out.push(Phi {
                    dst: ndst,
                    var: p.var,
                    args,
                });
            }
            for inst in &src.insts {
                insts_out.push(clone_inst(inst, &mut vmap, &mut nv)?);
            }
            if *b == latch {
                // Top of the latch: publish this iteration's chain result.
                insts_out.append(&mut latch_store);
            }
        }

        // Terminator cloning. The runtime calls the body once per ITERATION,
        // so every edge that would LEAVE the loop becomes `return`: the
        // header's false edge (condition failed → this call does nothing),
        // and any other exit edge (none exists in HELIX v1 — no break — but
        // treat one defensively rather than mis-cloning it). Exit edges jump
        // to a dedicated trailing block whose terminator is a bare `return`.
        let term = if *b == latch {
            Term::Return(None)
        } else {
            match &src.term {
                Term::Jump(t, args) if bmap.contains_key(t) => Term::Jump(
                    bmap[t],
                    args.iter()
                        .map(|a| map_id(*a, &mut vmap, &mut nv))
                        .collect(),
                ),
                Term::Jump(t, _) if !bmap.contains_key(t) => Term::Jump(EXIT_RET_BLOCK, vec![]),
                Term::Branch { cond, t, f } => {
                    let c = map_id(*cond, &mut vmap, &mut nv);
                    match (bmap.get(t), bmap.get(f)) {
                        (Some(&nt), Some(&nf)) => Term::Branch {
                            cond: c,
                            t: nt,
                            f: nf,
                        },
                        (Some(&nt), None) => Term::Branch {
                            cond: c,
                            t: nt,
                            f: EXIT_RET_BLOCK,
                        },
                        (None, Some(&nf)) => Term::Branch {
                            cond: c,
                            t: EXIT_RET_BLOCK,
                            f: nf,
                        },
                        (None, None) => Term::Jump(EXIT_RET_BLOCK, vec![]),
                    }
                }
                Term::Return(v) => {
                    let rv = v.map(|x| map_id(x, &mut vmap, &mut nv));
                    Term::Return(rv)
                }
                _ => return None,
            }
        };

        let bd = body.block_mut(nb);
        bd.phis = phis_out;
        bd.insts = insts_out;
        bd.term = term;
    }

    // Trailing exit-return block: target of every edge leaving the loop. The
    // runtime's per-iteration model means "leave the loop" == "end this call".
    let exit_ret = body.new_block();
    body.block_mut(exit_ret).term = Term::Return(None);
    for b in &order {
        let nb = bmap[b];
        match &body.block(nb).term {
            Term::Branch { t, f, .. } if *t == EXIT_RET_BLOCK || *f == EXIT_RET_BLOCK => {
                let (t2, f2) = (retarget(*t, exit_ret), retarget(*f, exit_ret));
                if let Term::Branch { t, f, .. } = &mut body.block_mut(nb).term {
                    *t = t2;
                    *f = f2;
                }
            }
            Term::Jump(t, _) if *t == EXIT_RET_BLOCK => {
                body.block_mut(nb).term = Term::Jump(exit_ret, vec![]);
            }
            _ => {}
        }
    }

    // Patch fresh value-type rows for every id minted above (cloning advanced
    // `nv` past the value `body.next_value` was frozen at), THEN verify.
    // Types are re-derived from the SOURCE id through the remap so freshly
    // cloned definitions keep their real widths/boolness (a resize alone
    // would stamp I64 over comparisons' bool results).
    body.next_value = nv;
    body.types.val_tys.resize(nv as usize, helix_sema::Ty::I64);
    for (old, new) in &vmap {
        if let Some(ty) = ir.val_ty(*old) {
            body.types.val_tys[new.0 as usize] = ty;
        }
    }

    // Drop the temporary name; the engine declares bodies under the planned
    // symbol (`body_fn_name`), so the IR name is cosmetic.
    body.name = String::new();

    body.recompute_edges();
    if let Err(e) = helix_ir::verify(&body) {
        eprintln!("helix-backend: extracted-body verify failed: {e}");
        if std::env::var_os("HELIX_DUMP_BODY").is_some() {
            eprintln!("{}", helix_ir::print_ir(&body, true));
        }
        return None; // body IR failed verification
    }
    Some((body, prebind))
}

/// Maps `old` to a fresh id on first mention (identity for pre-bound ids).
fn map_id(old: ValueId, vmap: &mut HashMap<ValueId, ValueId>, nv: &mut u32) -> ValueId {
    if let Some(&n) = vmap.get(&old) {
        return n;
    }
    let n = ValueId(*nv);
    *nv += 1;
    vmap.insert(old, n);
    n
}

/// Rewrites the EXIT_RET_BLOCK sentinel to the real trailing block id.
fn retarget(b: BlockId, exit_ret: BlockId) -> BlockId {
    if b == EXIT_RET_BLOCK { exit_ret } else { b }
}

/// Deep-clones one instruction under the id map.
fn clone_inst(inst: &Inst, vmap: &mut HashMap<ValueId, ValueId>, nv: &mut u32) -> Option<Inst> {
    Some(match inst {
        Inst::Const { dst, c } => Inst::Const {
            dst: map_id(*dst, vmap, nv),
            c: *c,
        },
        Inst::Bin { op, dst, a, b } => Inst::Bin {
            op: *op,
            dst: map_id(*dst, vmap, nv),
            a: map_id(*a, vmap, nv),
            b: map_id(*b, vmap, nv),
        },
        Inst::Unary { op, dst, a } => Inst::Unary {
            op: *op,
            dst: map_id(*dst, vmap, nv),
            a: map_id(*a, vmap, nv),
        },
        Inst::Cast { dst, val, to } => Inst::Cast {
            dst: map_id(*dst, vmap, nv),
            val: map_id(*val, vmap, nv),
            to: *to,
        },
        Inst::Load(l) => Inst::Load(Load {
            dst: map_id(l.dst, vmap, nv),
            arr: l.arr,
            idx: map_id(l.idx, vmap, nv),
        }),
        Inst::Store { arr, idx, val } => Inst::Store {
            arr: *arr,
            idx: map_id(*idx, vmap, nv),
            val: map_id(*val, vmap, nv),
        },
        Inst::Call(c) => Inst::Call(helix_ir::Call {
            dst: c.dst.map(|d| map_id(d, vmap, nv)),
            callee: c.callee.clone(),
            args: c.args.iter().map(|&a| map_id(a, vmap, nv)).collect(),
            arr_refs: c.arr_refs.clone(),
        }),
    })
}

/// `dst = *(shared_ctx(ctx) + off)` expressed as an IR call.
fn ld_call(dst: ValueId, ctx_v: ValueId, off_v: ValueId) -> Inst {
    // The byte offset travels as an ordinary SSA operand (a Const def emitted
    // by the caller), so the lowered call marshals both arguments naturally.
    Inst::Call(helix_ir::Call {
        dst: Some(dst),
        callee: "helix_ld_i64".into(),
        args: vec![ctx_v, off_v],
        arr_refs: Vec::new(),
    })
}

/// Typed ctx-load symbol for a captured scalar of `(width, floatness)`.
fn ld_sym(w: u64, float: bool) -> &'static str {
    match (w, float) {
        (8, false) => "helix_ld_i64",
        (4, false) => "helix_ld_i32",
        (8, true) => "helix_ld_f64",
        _ => "helix_ld_f32",
    }
}

/// Private-cell load symbol for an accumulator of `(width, floatness)`.
fn acc_sym(w: u64, float: bool) -> &'static str {
    match (w, float) {
        (8, false) => "helix_acc_load_i64",
        (4, false) => "helix_acc_load_i32",
        (8, true) => "helix_acc_load_f64",
        _ => "helix_acc_load_f32",
    }
}

/// Private-cell store symbol for an accumulator of `(width, floatness)`.
fn acc_store_sym(w: u64, float: bool) -> &'static str {
    match (w, float) {
        (8, false) => "helix_acc_store_i64",
        (4, false) => "helix_acc_store_i32",
        (8, true) => "helix_acc_store_f64",
        _ => "helix_acc_store_f32",
    }
}

/// The natural-loop block set headed by `hdr` (recomputed locally so this
/// module stays independent of analysis-crate lifetimes).
fn natural_loop_of(ir: &FuncIr, hdr: BlockId) -> Option<Vec<BlockId>> {
    let doms = helix_ir::dominators(ir);
    helix_ir::natural_loops(ir, &doms)
        .into_iter()
        .find(|(h, _)| *h == hdr)
        .map(|(_, body)| body)
}

// ---------------------------------------------------------------------------
// Host-side dispatch layer
// ---------------------------------------------------------------------------

/// Global region-id counter (unique across engines in one process).
static NEXT_REGION_ID: AtomicI64 = AtomicI64::new(1);

/// Mints a fresh region id.
pub(crate) fn next_region_id() -> i64 {
    NEXT_REGION_ID.fetch_add(1, Ordering::Relaxed)
}

/// Dispatcher-side snapshot of one region's ctx shape (compile time).
#[derive(Clone, Copy, Debug)]
pub(crate) struct SpecMeta {
    /// Packed shared-ctx size in bytes.
    pub(crate) len_bytes: i64,
    /// Reduction readback `(byte width, is_float)`; `None` for DoAll.
    pub(crate) acc: Option<(u64, bool)>,
    /// Reduction seed `(scalar word position, width, is_float)`.
    pub(crate) seed: Option<(usize, u64, bool)>,
}

/// Live specs keyed by region id (written at compile time).
static SPECS: LazyLock<Mutex<StdMap<i64, SpecMeta>>> = LazyLock::new(|| Mutex::new(StdMap::new()));
/// Packed contexts awaiting readback, keyed by region id.
static LIVE: LazyLock<Mutex<StdMap<i64, Vec<u64>>>> = LazyLock::new(|| Mutex::new(StdMap::new()));
/// Capture stash keyed by packed word position (main thread only; a mutex for
/// defence in depth). Integers store sign-extended payloads, floats their bit
/// patterns — a word is exclusively one or the other, so keys never clash.
static STASH: LazyLock<Mutex<StdMap<usize, u64>>> = LazyLock::new(|| Mutex::new(StdMap::new()));
/// Array fat-pointer stash keyed by layout slot.
static STASH_ARR: LazyLock<Mutex<StdMap<usize, (i64, i64)>>> =
    LazyLock::new(|| Mutex::new(StdMap::new()));

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    match m.lock() {
        Ok(g) => g,
        Err(e) => e.into_inner(),
    }
}

fn lock_specs() -> MutexGuard<'static, StdMap<i64, SpecMeta>> {
    lock(&SPECS)
}

#[allow(dead_code)] // used by tests/reset paths
fn lock_live() -> MutexGuard<'static, StdMap<i64, Vec<u64>>> {
    lock(&LIVE)
}

#[allow(dead_code)] // used by tests/reset paths
fn lock_stash() -> MutexGuard<'static, StdMap<usize, u64>> {
    lock(&STASH)
}

#[allow(dead_code)] // used by tests/reset paths
fn lock_arrs() -> MutexGuard<'static, StdMap<usize, (i64, i64)>> {
    lock(&STASH_ARR)
}

/// Records one region's dispatcher metadata (compile time).
pub(crate) fn register_spec(region_id: i64, meta: SpecMeta) {
    lock_specs().insert(region_id, meta);
}

/// Clears every spec/live/stash record (test isolation helper).
#[allow(dead_code)] // test isolation helper
pub(crate) fn reset_tables() {
    lock_specs().clear();
    lock_live().clear();
    lock_stash().clear();
    lock_arrs().clear();
}

/// `helix_stash_i(word, v)`: records integer capture `word` (i32s arrive
/// sign-extended).
pub extern "C" fn helix_stash_i(word: i64, v: i64) {
    lock(&STASH).insert(word.max(0) as usize, v as u64);
}

/// `helix_stash_f(word, v)`: records float capture `word` (bit-exact f64).
pub extern "C" fn helix_stash_f(word: i64, v: f64) {
    lock(&STASH).insert(word.max(0) as usize, v.to_bits());
}

/// `helix_stash_arr(slot, ptr, len)`: records array slot `slot` (layout order).
pub extern "C" fn helix_stash_arr(slot: i64, ptr: i64, len: i64) {
    lock(&STASH_ARR).insert(slot.max(0) as usize, (ptr, len));
}

/// `helix_dispatch(start, end, region_id, nthreads) -> handle`.
///
/// Packs the shared ctx from the stashes, allocates and seeds the participant
/// cells, runs the region on `helix-runtime`, copies the folded total into the
/// readback slot, and parks the context under `region_id` for `helix_read_*`.
///
/// # Safety (FFI contract)
/// The stashes contain exactly what the parent stashed for this region; body
/// and combine pointers were registered from finalized JIT code. All raw-
/// pointer arithmetic stays inside the dispatcher-owned allocation, which
/// outlives the parallel call.
pub extern "C" fn helix_dispatch(start: i64, end: i64, region_id: i64, nthreads: i64) -> i64 {
    let caps = std::mem::take(&mut *lock(&STASH));
    let arrs = std::mem::take(&mut *lock(&STASH_ARR));
    let Some(&meta) = lock(&SPECS).get(&region_id) else {
        // Out-of-sync compiler/runtime tables: unrecoverable for the program.
        eprintln!("runtime error: helix-backend: no spec for region {region_id}");
        std::process::abort();
    };

    let ctx_words = (meta.len_bytes as usize).div_ceil(8);
    let cell_words = helix_runtime::REDUCTION_CELL_STRIDE / 8;
    let cells = cell_count();
    let total_words = ctx_words + cells * cell_words;
    // u64 backing keeps the whole area 8-byte aligned; cell pointer/acc slots
    // are accessed through naturally aligned word operations.
    let mut mem: Vec<u64> = vec![0; total_words];

    // ---- pack the shared ctx --------------------------------------------------
    // Scalar stash keys are ABSOLUTE packed-word positions (`off / 8`, baked
    // by the parent emitter) — the same convention `meta.seed` uses — so they
    // land directly at their layout slots.
    for (slot, &(ptr, len)) in &arrs {
        let off = CTX_ARRAY_BASE as usize + slot * 16;
        if off + 8 <= meta.len_bytes as usize {
            mem[off / 8] = ptr as u64;
            mem[off / 8 + 1] = len as u64;
        }
    }
    for (word, bits) in &caps {
        if *word < ctx_words {
            mem[*word] = *bits;
        }
    }

    // ---- participant cells: word 0 = shared-ctx pointer -----------------------
    let shared_addr = mem.as_mut_ptr() as usize;
    for p in 0..cells {
        mem[ctx_words + p * cell_words] = shared_addr as u64;
    }

    // ---- seed every private accumulator (reductions) ---------------------------
    if let Some((word, width, _float)) = meta.seed
        && let Some(bits) = caps.get(&word)
    {
        for p in 0..cells {
            let slot = ctx_words + p * cell_words;
            // SAFETY: in-allocation byte write at cell + CELL_ACC_OFF,
            // within the `width` <= 8 bytes of the accumulator field.
            unsafe {
                let dst = (mem.as_mut_ptr().cast::<u8>()).add(slot * 8 + CELL_ACC_OFF as usize);
                // Captures store widened payloads; f32/i32 seeds take the
                // low 4 bytes, everything else the full word.
                let bytes: [u8; 8] = bits.to_le_bytes();
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, width as usize);
            }
        }
    }

    // ---- dispatch ---------------------------------------------------------------
    let cells_ptr =
        // SAFETY: `ctx_words * 8 <= allocation length`; the cast keeps the
        // provenance of the same allocation the runtime reads cells from.
        unsafe { mem.as_mut_ptr().add(ctx_words).cast::<u8>() };
    helix_runtime::helix_parallel_reduction(
        start, end, region_id, nthreads, cells_ptr,
        region_id, // combines are registered under the region id
    );

    // ---- publish the folded total into the readback slot ----------------------
    if let Some((w, _float)) = meta.acc {
        let cell0 = cells_ptr as usize;
        // SAFETY: cell 0's accumulator field, written by the runtime's fold
        // (or by the lone serial participant) before the join completed.
        let total = unsafe {
            match w {
                4 => u64::from(((cell0 + CELL_ACC_OFF as usize) as *const u32).read_unaligned()),
                _ => ((cell0 + CELL_ACC_OFF as usize) as *const u64).read_unaligned(),
            }
        };
        mem[CTX_ACC_OFF as usize / 8] = total;
    }

    lock(&LIVE).insert(region_id, mem);
    region_id
}

/// Participant-cell budget: hardware parallelism clamped to a sane ceiling
/// (the runtime never drafts more participants than the machine has anyway).
fn cell_count() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .clamp(1, 256)
}

/// `helix_read_i64(handle)`: the combined integer total (readback slot).
///
/// # Safety (FFI contract)
/// `handle` must be a value returned by [`helix_dispatch`] whose context is
/// still parked in `LIVE` — true for handles the backend itself emits.
pub extern "C" fn helix_read_i64(handle: i64) -> i64 {
    read_slot(handle) as i64
}

/// `helix_read_i32(handle)`: narrow integer total.
///
/// # Safety (FFI contract)
/// See [`helix_read_i64`].
pub extern "C" fn helix_read_i32(handle: i64) -> i32 {
    read_slot(handle) as i32
}

/// `helix_read_f64(handle)`.
///
/// # Safety (FFI contract)
/// See [`helix_read_i64`].
pub extern "C" fn helix_read_f64(handle: i64) -> f64 {
    f64::from_bits(read_slot(handle))
}

/// `helix_read_f32(handle)` (total stored widened; demoted on read).
///
/// # Safety (FFI contract)
/// See [`helix_read_i64`].
pub extern "C" fn helix_read_f32(handle: i64) -> f32 {
    f64::from_bits(read_slot(handle)) as f32
}

/// Reads the readback slot of a parked context (zero when unknown).
fn read_slot(handle: i64) -> u64 {
    lock(&LIVE)
        .get(&handle)
        .and_then(|mem| mem.get(CTX_ACC_OFF as usize / 8).copied())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Body-context imports (called from JITed bodies)
// ---------------------------------------------------------------------------

/// Resolves the shared-ctx pointer for a participant cell handle.
///
/// # Safety
/// `cell` must be a dispatcher-provided participant context (valid for the
/// whole region) whose word 0 holds the shared-ctx pointer — the invariant
/// established in [`helix_dispatch`].
unsafe fn shared_ctx_of(cell: i64) -> usize {
    // SAFETY: word 0 of the dispatcher-owned cell.
    unsafe { ((cell as *const u64).read_unaligned()) as usize }
}

/// `helix_ld_i64(cell, off)`: loads the i64 at `shared_ctx + off`.
///
/// # Safety (FFI contract)
/// `cell` is the context the runtime handed this participant and `off` is a
/// layout offset the backend baked — together they address the packed ctx
/// allocation, which outlives the region.
pub extern "C" fn helix_ld_i64(cell: i64, off: i64) -> i64 {
    // SAFETY: contract above; both addresses 8-aligned by construction.
    unsafe { ((shared_ctx_of(cell) + off as usize) as *const u64).read_unaligned() as i64 }
}

/// `helix_ld_i32(cell, off)`: narrow integer capture (stored widened).
///
/// # Safety (FFI contract)
/// See [`helix_ld_i64`].
pub extern "C" fn helix_ld_i32(cell: i64, off: i64) -> i32 {
    // SAFETY: contract above.
    unsafe { ((shared_ctx_of(cell) + off as usize) as *const u64).read_unaligned() as i32 }
}

/// `helix_ld_f64(cell, off)`: f64 capture.
///
/// # Safety (FFI contract)
/// See [`helix_ld_i64`].
pub extern "C" fn helix_ld_f64(cell: i64, off: i64) -> f64 {
    // SAFETY: contract above.
    unsafe { f64::from_bits(((shared_ctx_of(cell) + off as usize) as *const u64).read_unaligned()) }
}

/// `helix_ld_f32(cell, off)`: f32 capture (stored widened, demoted here).
///
/// # Safety (FFI contract)
/// See [`helix_ld_i64`].
pub extern "C" fn helix_ld_f32(cell: i64, off: i64) -> f32 {
    // SAFETY: contract above.
    unsafe {
        f64::from_bits(((shared_ctx_of(cell) + off as usize) as *const u64).read_unaligned()) as f32
    }
}

/// `helix_acc_load_i64(cell)`: this participant's private accumulator.
///
/// # Safety (FFI contract)
/// `cell` must be a dispatcher-provided participant context.
pub extern "C" fn helix_acc_load_i64(cell: i64) -> i64 {
    // SAFETY: in-cell read at CELL_ACC_OFF (dispatcher-owned for the region).
    unsafe { ((cell as usize + CELL_ACC_OFF as usize) as *const u64).read_unaligned() as i64 }
}

/// `helix_acc_load_i32(cell)`.
///
/// # Safety (FFI contract)
/// See [`helix_acc_load_i64`].
pub extern "C" fn helix_acc_load_i32(cell: i64) -> i32 {
    // SAFETY: in-cell read at CELL_ACC_OFF.
    unsafe { ((cell as usize + CELL_ACC_OFF as usize) as *const u32).read_unaligned() as i32 }
}

/// `helix_acc_load_f64(cell)`.
///
/// # Safety (FFI contract)
/// See [`helix_acc_load_i64`].
pub extern "C" fn helix_acc_load_f64(cell: i64) -> f64 {
    // SAFETY: in-cell read at CELL_ACC_OFF.
    unsafe {
        f64::from_bits(((cell as usize + CELL_ACC_OFF as usize) as *const u64).read_unaligned())
    }
}

/// `helix_acc_load_f32(cell)`.
///
/// # Safety (FFI contract)
/// See [`helix_acc_load_i64`].
pub extern "C" fn helix_acc_load_f32(cell: i64) -> f32 {
    // SAFETY: in-cell read at CELL_ACC_OFF.
    unsafe {
        f32::from_bits(((cell as usize + CELL_ACC_OFF as usize) as *const u32).read_unaligned())
    }
}

/// `helix_acc_store_i64(cell, v)`: publishes the accumulated value to this
/// participant's private cell.
///
/// # Safety (FFI contract)
/// `cell` must be a dispatcher-provided participant context.
pub extern "C" fn helix_acc_store_i64(cell: i64, v: i64) {
    // SAFETY: in-cell write at CELL_ACC_OFF (8-byte field).
    unsafe { ((cell as usize + CELL_ACC_OFF as usize) as *mut u64).write_unaligned(v as u64) };
}

/// `helix_acc_store_i32(cell, v)`.
///
/// # Safety (FFI contract)
/// See [`helix_acc_store_i64`].
pub extern "C" fn helix_acc_store_i32(cell: i64, v: i32) {
    // SAFETY: in-cell write of the 4-byte accumulator field.
    unsafe { ((cell as usize + CELL_ACC_OFF as usize) as *mut u32).write_unaligned(v as u32) };
}

/// `helix_acc_store_f64(cell, v)`.
///
/// # Safety (FFI contract)
/// See [`helix_acc_store_i64`].
pub extern "C" fn helix_acc_store_f64(cell: i64, v: f64) {
    // SAFETY: in-cell write of the 8-byte accumulator field.
    unsafe { ((cell as usize + CELL_ACC_OFF as usize) as *mut u64).write_unaligned(v.to_bits()) };
}

/// `helix_acc_store_f32(cell, v)`.
///
/// # Safety (FFI contract)
/// See [`helix_acc_store_i64`].
pub extern "C" fn helix_acc_store_f32(cell: i64, v: f32) {
    // SAFETY: in-cell write of the 4-byte accumulator field.
    unsafe { ((cell as usize + CELL_ACC_OFF as usize) as *mut u32).write_unaligned(v.to_bits()) };
}

// ---------------------------------------------------------------------------
// Combine functions (registered per region after finalize)
// ---------------------------------------------------------------------------

/// No-op combine for DoAll regions (the threaded runtime path requires a
/// registered combine whenever cells are passed).
extern "C" fn combine_noop(_dst: *mut u8, _src: *const u8) {}

macro_rules! arith_combiner {
    ($name:ident, $ty:ty, |$a:ident, $b:ident| $body:expr) => {
        extern "C" fn $name(dst: *mut u8, src: *const u8) {
            // SAFETY: dispatcher-owned cells, wide enough for `$ty`.
            unsafe {
                let d = dst as *mut $ty;
                let s = src as *const $ty;
                let ($a, $b) = (*d, *s);
                *d = $body;
            }
        }
    };
}

arith_combiner!(combine_add_i64, i64, |a, b| a.wrapping_add(b));
arith_combiner!(combine_add_i32, i32, |a, b| a.wrapping_add(b));
arith_combiner!(combine_mul_i64, i64, |a, b| a.wrapping_mul(b));
arith_combiner!(combine_mul_i32, i32, |a, b| a.wrapping_mul(b));
arith_combiner!(combine_add_f64, f64, |a, b| a + b);
arith_combiner!(combine_add_f32, f32, |a, b| a + b);
arith_combiner!(combine_mul_f64, f64, |a, b| a * b);
arith_combiner!(combine_mul_f32, f32, |a, b| a * b);

/// Integer min/max combinators.
macro_rules! int_minmax_combiner {
    ($name:ident, $ty:ty, $is_min:expr) => {
        extern "C" fn $name(dst: *mut u8, src: *const u8) {
            // SAFETY: dispatcher-owned cells, wide enough for `$ty`.
            unsafe {
                let d = dst as *mut $ty;
                let s = src as *const $ty;
                let take_src = if $is_min { *s < *d } else { *s > *d };
                if take_src {
                    *d = *s;
                }
            }
        }
    };
}

int_minmax_combiner!(combine_min_i64, i64, true);
int_minmax_combiner!(combine_max_i64, i64, false);
int_minmax_combiner!(combine_min_i32, i32, true);
int_minmax_combiner!(combine_max_i32, i32, false);

/// Float min/max combinators reproducing the interpreter's ordered semantics:
/// a NaN operand loses against a real number (IEEE minNum/maxNum).
macro_rules! fp_minmax_combiner {
    ($name:ident, $ty:ty, $is_min:expr) => {
        extern "C" fn $name(dst: *mut u8, src: *const u8) {
            // SAFETY: dispatcher-owned cells, wide enough for `$ty`.
            unsafe {
                let d = dst as *mut $ty;
                let s = src as *const $ty;
                let (x, y) = (*d, *s);
                *d = if x.is_nan() {
                    y
                } else if y.is_nan() || ($is_min && x <= y) || (!$is_min && x >= y) {
                    x
                } else {
                    y
                };
            }
        }
    };
}

fp_minmax_combiner!(combine_min_f64, f64, true);
fp_minmax_combiner!(combine_max_f64, f64, false);
fp_minmax_combiner!(combine_min_f32, f32, true);

/// The registered combine fn for a region kind.
pub(crate) fn combine_for(kind: RKind) -> helix_runtime::CombineFn {
    let spec = match kind {
        RKind::DoAll => return combine_noop,
        RKind::Reduction(s) => s,
    };
    match (spec.op, spec.float, spec.width) {
        (AnOp::Add, false, 8) => combine_add_i64,
        (AnOp::Add, false, 4) => combine_add_i32,
        (AnOp::Mul, false, 8) => combine_mul_i64,
        (AnOp::Mul, false, 4) => combine_mul_i32,
        (AnOp::Add, true, 8) => combine_add_f64,
        (AnOp::Add, true, 4) => combine_add_f32,
        (AnOp::Mul, true, 8) => combine_mul_f64,
        (AnOp::Mul, true, 4) => combine_mul_f32,
        (AnOp::Min, false, 8) => combine_min_i64,
        (AnOp::Min, false, 4) => combine_min_i32,
        (AnOp::Max, false, 8) => combine_max_i64,
        (AnOp::Max, false, 4) => combine_max_i32,
        (AnOp::Min, true, 8) => combine_min_f64,
        (AnOp::Min, true, 4) => combine_min_f32,
        (AnOp::Max, true, 8) => combine_max_f64,
        // Widths other than 4/8 never leave `extract` (it demotes them), so
        // the wildcard is unreachable in practice; DoAll regions never reach
        // this function either (`combine_for` returns early above).
        _ => combine_noop,
    }
}
