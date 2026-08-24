//! SSA construction: semi-pruned φ placement + dominator-tree renaming.
//!
//! The algorithm is the classic Cytron/Briggs pipeline (see
//! `docs/research/ssa-design.md`):
//!
//! 1. **Strip unreachable blocks** — dominance and iterated-DF placement are
//!    only correct on reachable flowgraphs; an unreachable predecessor would
//!    contribute a bogus φ argument.
//! 2. **Classify global names** — a variable is *global* if some reachable
//!    block uses it before defining it locally (an upward-exposed use). Only
//!    globals need φ-nodes; purely local temporaries stay register-allocated
//!    inside their block.
//! 3. **φ placement** — for each global, insert phis at the iterated dominance
//!    frontier of its def blocks (semi-pruned: no liveness fixpoint; Briggs et
//!    al. measured phi counts within a few percent of pruned SSA at a fraction
//!    of the cost).
//! 4. **Renaming** — iterative preorder walk of the dominator tree with one
//!    stack per local: defs push, block exits pop, and successor edges read
//!    the current top-of-stack to fill jump argument lists and φ arguments.
//!
//! ## Naming scheme
//!
//! Fresh names are `stride*(local+1) + version` where `stride` sits strictly
//! above every pre-existing id, so the *pre-SSA cell spelling* of local `v`
//! (`ValueId(v.0)`) stays addressable after renaming — version 0 of every
//! variable is its own cell id. That is what keeps entry-block parameter
//! definitions (zero-argument entry phis) stable across the transformation.
//!
//! ## What does not become SSA
//!
//! Arrays: [`Inst::Load`] / [`Inst::Store`] keep addressing the array's local
//! slot directly and never receive phis — memory deliberately stays outside
//! SSA (LLVM precedent), which is exactly what affine dependence analysis
//! wants to see.

use std::collections::{HashMap, HashSet};

use helix_sema::Ty;

use crate::dom::dominators;
use crate::ir::{BlockId, FuncIr, Inst, LocalId, Phi, Term, ValueId};

// ---------------------------------------------------------------------------
// Step 1 + step 2 helpers
// ---------------------------------------------------------------------------

/// Remove blocks unreachable from entry, remapping terminators.
pub fn strip_unreachable(ir: &mut FuncIr) {
    let live = crate::dom::reachability(ir);
    if live.iter().all(|v| *v) {
        return;
    }
    ir.compact(&live);
}

/// Result of the upward-exposed-use analysis over blocks.
#[derive(Clone, Debug, Default)]
pub struct GlobalNames {
    /// Locals used-before-defined somewhere in the function.
    pub globals: Vec<LocalId>,
}

/// One linear pass classifying global names (semi-pruned SSA's only dataflow).
///
/// Compiler temporaries (`LocalId >= n_source_locals`: `$sc` short-circuit
/// results, `$ret` return accumulators) are classified like any other
/// variable — they ARE multi-path globals, and skipping them left them with
/// several un-merged definitions after renaming.
///
/// ARRAY locals are excluded: arrays stay outside SSA by design (their slot
/// names the fat pointer, never re-bound), and renaming their cell spelling
/// in call arguments would silently alias them onto an unrelated scalar's
/// SSA name.
#[must_use]
pub fn global_names(ir: &FuncIr) -> GlobalNames {
    let is_array_local = |l: u32| -> bool {
        matches!(
            ir.types.local_ty(LocalId(l)),
            Some(helix_sema::Ty::Array(_))
        )
    };
    let live = crate::dom::reachability(ir);
    let mut set: HashSet<LocalId> = HashSet::new();
    for (bi, block) in ir.blocks.iter().enumerate() {
        if !live[bi] {
            continue;
        }
        let mut defs: HashSet<u32> = HashSet::new();

        // Entry phis (no args) are parameter definitions. Array parameters
        // keep their zero-arg phi as the definition but are NOT renamed.
        for p in &block.phis {
            if p.args.is_empty() {
                defs.insert(p.var.0);
                continue;
            }
            // A phi itself uses values from other blocks; those uses belong to
            // the predecessors' exposure, not this block's prefix.
        }
        for inst in &block.insts {
            for u in inst.uses() {
                if (u.0 as usize) < ir.n_locals && !defs.contains(&u.0) && !is_array_local(u.0) {
                    set.insert(LocalId(u.0));
                }
            }
            if let Some(d) = inst.dst()
                && (d.0 as usize) < ir.n_locals
            {
                defs.insert(d.0);
            }
        }
        match &block.term {
            Term::Jump(_, args) => {
                for v in args {
                    if (v.0 as usize) < ir.n_locals && !defs.contains(&v.0) && !is_array_local(v.0)
                    {
                        set.insert(LocalId(v.0));
                    }
                }
            }
            Term::Branch { cond, .. } => {
                if (cond.0 as usize) < ir.n_locals && !defs.contains(&cond.0) {
                    set.insert(LocalId(cond.0));
                }
            }
            Term::Return(Some(v)) => {
                if (v.0 as usize) < ir.n_locals && !defs.contains(&v.0) {
                    set.insert(LocalId(v.0));
                }
            }
            Term::Return(None) => {}
        }
    }
    let mut globals: Vec<LocalId> = set.into_iter().collect();
    globals.sort_unstable();
    GlobalNames { globals }
}

// ---------------------------------------------------------------------------
// to_ssa
// ---------------------------------------------------------------------------

/// Convert a freshly built function into semi-pruned SSA form in place.
///
/// After this returns:
/// * every scalar local has a family of unique definitions,
/// * joins merge reaching definitions through [`Phi`] nodes whose argument
///   lists align with the block's predecessor order,
/// * unconditional jumps carry one argument per target φ, positionally
///   aligned (Cranelift block-parameter style),
/// * arrays remain explicit load/store operations against their slots.
pub fn to_ssa(ir: &mut FuncIr) {
    strip_unreachable(ir);

    let doms = dominators(ir);
    let df = crate::dom::dominance_frontiers(ir, &doms);
    let names = global_names(ir);

    place_phis(ir, &names, &df);
    rename(ir, &doms);
    ir.normalize_phis();
}

/// Semi-pruned φ insertion: iterated dominance frontiers of each global's
/// def blocks.
fn place_phis(ir: &mut FuncIr, names: &GlobalNames, df: &[Vec<BlockId>]) {
    let n = ir.blocks.len();
    for var in &names.globals {
        let mut has_phi: HashSet<BlockId> = HashSet::new();
        let mut queued: Vec<bool> = vec![false; n];
        let mut work: Vec<usize> = Vec::new();

        // Seed with every block that defines `var` (dst == var's cell id), or
        // that defines it as a parameter phi (zero args).
        for (bi, block) in ir.blocks.iter().enumerate() {
            let defines = block
                .insts
                .iter()
                .any(|i| i.dst().is_some_and(|d| d.0 == var.0))
                || block
                    .phis
                    .iter()
                    .any(|p| p.var == *var && p.args.is_empty());
            if defines {
                work.push(bi);
                queued[bi] = true;
            }
        }

        while let Some(xi) = work.pop() {
            queued[xi] = false;
            for y in df[xi].clone() {
                if has_phi.contains(&y) {
                    continue;
                }
                has_phi.insert(y);
                let yi = y.0 as usize;
                ir.blocks[yi].phis.push(Phi {
                    dst: ValueId(var.0),
                    var: *var,
                    args: Vec::new(),
                });
                if !queued[yi] {
                    queued[yi] = true;
                    work.push(yi);
                }
            }
        }
    }

    // Order phis by variable id so jump argument lists are deterministic.
    for b in &mut ir.blocks {
        b.phis.sort_by_key(|p| p.var.0);
    }
}

// ---------------------------------------------------------------------------
// Renaming
// ---------------------------------------------------------------------------

/// Overwrite the destination of `inst` (helper for the in-place def rename).
fn set_dst(inst: &mut Inst, new: ValueId) {
    match inst {
        Inst::Const { dst, .. }
        | Inst::Bin { dst, .. }
        | Inst::Unary { dst, .. }
        | Inst::Cast { dst, .. } => *dst = new,
        Inst::Load(l) => l.dst = new,
        Inst::Call(c) => {
            if c.dst.is_some() {
                c.dst = Some(new);
            }
        }
        Inst::Store { .. } => {}
    }
}

/// Dominator-tree preorder renaming with per-local stacks (iterative, so deep
/// dominator trees cannot overflow the native stack).
///
/// EVERY instruction destination participates — source-variable cells *and*
/// compiler temporaries (`$sc`, `$ret`). Temporaries were historically left
/// untouched on the grounds of being single-assignment, but short-circuit
/// results and return accumulators are defined on MULTIPLE control paths, so
/// they are ordinary globals and must be renamed and φ-merged like cells.
///
/// Uses are rewritten **interleaved** with definitions (classic Cytron):
/// an instruction's operands see the reaching name at its own program point,
/// not the block's final stack. This keeps `x = x + 1` self-consistent — the
/// read resolves to the incoming name, the write mints the next one — and
/// fixes same-block use-before-redefine sequences.
fn rename(ir: &mut FuncIr, doms: &crate::dom::Doms) {
    let cell_stride = ir.max_value_id() + 1;
    // Stacks are indexed by LOCAL SLOT id, but temporary VALUES (v3, v4, …)
    // spell their slot as their own id — the builder allocates temporaries
    // with LocalId == ValueId. Size the vector over both spaces so every
    // id in either space has a stack.
    let n_slots = ir.n_locals.max(cell_stride as usize).max(
        ir.blocks
            .iter()
            .flat_map(|b| b.phis.iter().map(|p| p.var.0 as usize))
            .max()
            .map_or(0, |m| m + 1),
    );
    let n_cells = ir.n_source_locals;

    // Current reaching name of each local (bottom of stack = cell id).
    // Temporaries get the same treatment as source variables: their id is
    // version 0, so pre-rename uses resolve and multi-path defs get φs.
    let mut stacks: Vec<Vec<u32>> = vec![Vec::new(); n_slots];
    for (idx, st) in stacks.iter_mut().enumerate() {
        st.push(idx as u32); // version 0 = own id
    }

    // Fresh ids start above every existing id (cell ids AND temporaries), so
    // they can collide with nothing.
    let mut next_fresh = cell_stride * (n_cells as u32 + 1);
    let mut fresh_ty: HashMap<u32, Ty> = HashMap::new();
    let mut def_map: HashMap<ValueId, ValueId> = HashMap::new();
    let mut undo_log: Vec<Vec<u32>> = vec![Vec::new(); ir.blocks.len()];
    let mut phi_dst_new: HashMap<(usize, u32), u32> = HashMap::new(); // (block idx, var idx) -> new id

    enum Step {
        Enter(BlockId),
        Exit(BlockId),
    }
    let mut steps: Vec<Step> = vec![Step::Enter(BlockId(0))];
    let mut pending_succ_args: Vec<HashMap<BlockId, Vec<ValueId>>> =
        vec![HashMap::new(); ir.blocks.len()];
    let dom_children = doms.tree_children();

    while let Some(step) = steps.pop() {
        match step {
            Step::Enter(b) => {
                let bi = b.0 as usize;

                // Phis define their variable first.
                for pi in 0..ir.blocks[bi].phis.len() {
                    let var = ir.blocks[bi].phis[pi].var;
                    let fresh = next_fresh;
                    next_fresh += 1;
                    let ty = ir.types.local_ty(var).unwrap_or(Ty::I64);
                    fresh_ty.insert(fresh, ty);
                    phi_dst_new.insert((bi, var.0), fresh);
                    if let Some(st) = stacks.get_mut(var.0 as usize) {
                        st.push(fresh);
                        undo_log[bi].push(var.0);
                    }
                }

                // Instructions: rewrite uses at their own program point,
                // THEN push this instruction's fresh def (Cytron order).
                // Array-typed values (call arguments naming an array slot)
                // are NEVER rewritten — arrays stay outside SSA by design,
                // and rewriting them would alias the array onto a scalar.
                for ii in 0..ir.blocks[bi].insts.len() {
                    // 1. Operands see the stack as of *before* this def.
                    {
                        let block = &mut ir.blocks[bi];
                        let inst = &mut block.insts[ii];
                        inst.rewrite_uses(&mut |v: ValueId| {
                            let vi = v.0 as usize;
                            if vi >= n_slots
                                || matches!(ir.types.val_tys.get(vi), Some(Ty::Array(_)))
                            {
                                return v;
                            }
                            stacks[vi].last().map_or(v, |top| ValueId(*top))
                        });
                    }
                    // 2. Mint + push the fresh name for this def.
                    let Some(d) = ir.blocks[bi].insts[ii].dst() else {
                        continue;
                    };
                    let li = d.0 as usize;
                    if matches!(ir.types.val_tys.get(li), Some(Ty::Array(_))) {
                        continue; // array slots keep their single cell name
                    }
                    let ty = ir.types.val_tys.get(li).copied().unwrap_or(Ty::I64);
                    let fresh = next_fresh;
                    next_fresh += 1;
                    fresh_ty.insert(fresh, ty);
                    if li < stacks.len() {
                        stacks[li].push(fresh);
                        undo_log[bi].push(d.0);
                        def_map.insert(d, ValueId(fresh));
                    }
                    set_dst(&mut ir.blocks[bi].insts[ii], ValueId(fresh));
                }

                // Terminator operands (condition / return value) see the
                // block-final stack.
                {
                    let block = &mut ir.blocks[bi];
                    match &mut block.term {
                        Term::Branch { cond, .. } => {
                            *cond = stacks
                                .get(cond.0 as usize)
                                .and_then(|st| st.last())
                                .map_or(*cond, |top| ValueId(*top));
                        }
                        Term::Return(v) => {
                            if let Some(x) = v {
                                *x = stacks
                                    .get(x.0 as usize)
                                    .and_then(|st| st.last())
                                    .map_or(*x, |top| ValueId(*top));
                            }
                        }
                        Term::Jump(..) => {} // args rebuilt in apply_renamings
                    }
                }

                // Patch φ destinations to their fresh names.
                for p in ir.blocks[bi].phis.iter_mut() {
                    if let Some(new) = phi_dst_new.get(&(bi, p.var.0)) {
                        p.dst = ValueId(*new);
                    }
                }

                // Record what our outgoing edges will pass to successor phis.
                let succs = ir.block(b).term.succs();
                for s in succs {
                    let si = s.0 as usize;
                    let mut args = Vec::with_capacity(ir.blocks[si].phis.len());
                    for p in &ir.blocks[si].phis {
                        let cur = stacks
                            .get(p.var.0 as usize)
                            .and_then(|st| st.last().copied())
                            .unwrap_or(p.var.0);
                        args.push(ValueId(cur));
                    }
                    pending_succ_args[bi].insert(s, args);
                }

                // Children after us; our Exit fires when all are done.
                steps.push(Step::Exit(b));
                for child in dom_children[bi].iter().rev() {
                    steps.push(Step::Enter(*child));
                }
            }
            Step::Exit(b) => {
                let bi = b.0 as usize;
                for l in undo_log[bi].drain(..).rev() {
                    if let Some(st) = stacks.get_mut(l as usize) {
                        st.pop();
                    }
                }
            }
        }
    }

    // ---- apply recorded renamings ------------------------------------------
    apply_renamings(
        ir,
        &phi_dst_new,
        &pending_succ_args,
        &fresh_ty,
        &mut def_map,
    );
}

/// Second phase: install renamed phi destinations, patch predecessor-supplied
/// edge arguments consistently, and register fresh value types.
fn apply_renamings(
    ir: &mut FuncIr,
    phi_dst_new: &HashMap<(usize, u32), u32>,
    succ_args: &[HashMap<BlockId, Vec<ValueId>>],
    fresh_ty: &HashMap<u32, Ty>,
    def_map: &mut HashMap<ValueId, ValueId>,
) {
    // 1. Phi destinations.
    for (bi, b) in ir.blocks.iter_mut().enumerate() {
        for p in &mut b.phis {
            if let Some(new) = phi_dst_new.get(&(bi, p.var.0)) {
                p.dst = ValueId(*new);
            } else {
                p.dst = ValueId(p.var.0); // parameter phi keeps the cell id
            }
        }
    }

    // 2. Edge arguments: gather each block's incoming (pred, value) columns
    //    from the per-pred maps recorded during the walk.
    for pi in 0..ir.blocks.len() {
        let arity = ir.blocks[pi].phis.len();
        if arity == 0 {
            continue;
        }
        let target = BlockId(pi as u32);
        let preds = ir.blocks[pi].preds.clone();
        let mut columns: Vec<Vec<(BlockId, ValueId)>> = vec![Vec::new(); arity];
        for p in preds {
            if let Some(args) = succ_args.get(p.0 as usize).and_then(|m| m.get(&target)) {
                for (i, v) in args.iter().enumerate() {
                    if i < arity {
                        columns[i].push((p, *v));
                    }
                }
            }
        }
        for (i, col) in columns.into_iter().enumerate() {
            ir.blocks[pi].phis[i].args = col;
            ir.blocks[pi].phis[i].args.sort_unstable_by_key(|(b, _)| *b);
        }
    }

    // 3. Jump argument lists mirror the phi argument rows they feed.
    for bi in 0..ir.blocks.len() {
        let t = match &ir.blocks[bi].term {
            Term::Jump(t, _) => *t,
            _ => continue,
        };
        let ti = t.0 as usize;
        let from = BlockId(bi as u32);
        let mut new_args = Vec::with_capacity(ir.blocks[ti].phis.len());
        for phi in &ir.blocks[ti].phis {
            let v = phi
                .args
                .iter()
                .find(|(f, _)| *f == from)
                .map(|(_, v)| *v)
                .unwrap_or(ValueId(phi.var.0));
            new_args.push(v);
        }
        if let Term::Jump(_, args) = &mut ir.blocks[bi].term {
            *args = new_args;
        }
    }

    // 4. Register types for fresh names and remember the final def map.
    let max_needed = fresh_ty.keys().copied().max().unwrap_or(0);
    if ir.types.val_tys.len() <= max_needed as usize {
        ir.types.val_tys.resize(max_needed as usize + 1, Ty::I64);
    }
    for (id, ty) in fresh_ty {
        ir.types.val_tys[*id as usize] = *ty;
    }
    for ((_bi, var), new) in phi_dst_new {
        def_map.insert(ValueId(*var), ValueId(*new));
    }
}

// ---------------------------------------------------------------------------
// Queries + verification hook
// ---------------------------------------------------------------------------

/// Argument that pred `p` contributes to the first φ of `b`, or `None`.
#[must_use]
pub fn phi_arg_for_pred(ir: &FuncIr, b: BlockId, p: BlockId) -> Option<ValueId> {
    ir.block(b)
        .phis
        .first()
        .and_then(|phi| phi.args.iter().find(|(f, _)| *f == p).map(|(_, v)| *v))
}

/// Is this function in SSA form? Derived, not stored: runs the verifier's
/// unique-def/dominance checks.
#[must_use]
pub fn is_ssa(ir: &FuncIr) -> bool {
    verify_ssa_unique_defs(ir).is_ok() && verify_ssa(ir).is_ok()
}

/// No-op retained for API symmetry; SSA-ness is derived by [`is_ssa`].
pub fn mark_ssa() {}

/// SSA-specific structural check folded into `verify`.
///
/// # Errors
/// Returns a precise message naming the offending block/value.
pub fn verify_ssa(ir: &FuncIr) -> Result<(), String> {
    crate::verify::verify(ir)?;
    verify_ssa_unique_defs(ir)
}

/// Every non-entry-block value must be defined exactly once; entry parameter
/// phis are exempt because their zero-arg form *is* the definition.
fn verify_ssa_unique_defs(ir: &FuncIr) -> Result<(), String> {
    let mut seen: HashSet<u32> = HashSet::new();
    for (bi, b) in ir.blocks.iter().enumerate() {
        for p in &b.phis {
            if !seen.insert(p.dst.0) {
                return Err(format!(
                    "ssa violation: value {} defined more than once (block bb{bi})",
                    p.dst.0
                ));
            }
        }
        for inst in &b.insts {
            if let Some(d) = inst.dst()
                && !seen.insert(d.0)
            {
                return Err(format!(
                    "ssa violation: value {} defined more than once (block bb{bi})",
                    d.0
                ));
            }
        }
    }
    Ok(())
}
