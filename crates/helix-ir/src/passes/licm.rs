//! Loop-invariant code motion over natural loops.
//!
//! Natural-loop discovery lives here (back edge = edge to a dominator
//! ancestor; body = header plus everything reaching the back-edge source
//! without passing through the header) because LICM is its only in-crate
//! consumer; `helix-analysis` builds its richer nest/depth model separately.
//!
//! What may move: **pure, non-trapping** instructions (`Bin`/`Unary`/`Cast`
//! — integer div/rem can trap, so they stay put) whose operands are all
//! defined outside the loop. Loads/stores/calls never hoist: loads may trap
//! and observe memory, stores/calls are effects (the task brief is explicit:
//! "licm only pure non-memory ops"). Loop-header φs are immovable by
//! construction (their inputs differ per back edge — that *is* the
//! loop-carried value).
//!
//! Hoisting appends the instruction to the loop's preheader (the unique
//! non-back-edge predecessor of the header). SSA makes the transform
//! trivially safe: the hoisted value has exactly one definition, so every use
//! in the loop still reaches it — now unconditionally before the loop.

use std::collections::HashSet;

use helix_syntax::ast::BinOp;

use crate::dom::{dominators, natural_loops};
use crate::ir::{BlockId, FuncIr, Inst, Term, ValueId};
use crate::passmod::ChangeFlag;

/// Hoist invariant pure instructions out of every natural loop.
pub fn licm(ir: &mut FuncIr) -> ChangeFlag {
    let mut flag = ChangeFlag::new();

    let doms = dominators(ir);
    for (header, body) in natural_loops(ir, &doms) {
        if body.len() <= 1 {
            continue; // self-loop with a single block: nothing to restructure
        }
        let body_set: HashSet<u32> = body.iter().map(|b| b.0).collect();
        let Some(preheader) = ensure_preheader(ir, header, &body_set, &mut flag) else {
            continue;
        };

        // "Is this operand defined outside the loop?" — decided against a
        // SNAPSHOT taken before any instruction moves, because the sweep
        // below drains blocks while it scans them.
        //
        // Constant defs get their own bucket (`loop_consts`): they are
        // trivially invariant, so they never BLOCK hoisting of their users —
        // instead they are dragged along whenever a hoisted op reads one
        // (moving a Const is always safe; it has no operands or effects).
        let mut loop_defs: HashSet<u32> = HashSet::new();
        let mut loop_consts: HashSet<u32> = HashSet::new();
        for b in &body {
            let bb = &ir.blocks[b.0 as usize];
            for p in &bb.phis {
                loop_defs.insert(p.dst.0);
            }
            for i in &bb.insts {
                match (i, i.dst()) {
                    (Inst::Const { .. }, Some(d)) => {
                        loop_consts.insert(d.0);
                    }
                    (_, Some(d)) => {
                        loop_defs.insert(d.0);
                    }
                    _ => {}
                }
            }
        }
        let def_outside = |v: ValueId| -> bool { !loop_defs.contains(&v.0) };

        // Collect hoistable instructions (one round per call; repeated calls
        // via the fixpoint driver hoist nested-invariant chains). A hoisted op
        // using an in-loop constant def drags that def along so no use dangles
        // after the move.
        let mut hoisted: Vec<Inst> = Vec::new();
        let mut dragged_consts: HashSet<u32> = HashSet::new();
        for b in &body {
            let bi = b.0 as usize;
            let old = std::mem::take(&mut ir.blocks[bi].insts);
            let mut kept: Vec<Inst> = Vec::with_capacity(old.len());
            for inst in old {
                let movable = match &inst {
                    Inst::Bin { op, a, b: b2, .. } => {
                        !matches!(op, BinOp::Div | BinOp::Rem)
                            && def_outside(*a)
                            && def_outside(*b2)
                    }
                    Inst::Unary { a, .. } | Inst::Cast { val: a, .. } => def_outside(*a),
                    Inst::Const { .. } => false, // nothing gained by moving these
                    _ => false,
                };
                if movable {
                    for u in inst.uses() {
                        if loop_consts.contains(&u.0) {
                            dragged_consts.insert(u.0);
                        }
                    }
                    hoisted.push(inst);
                    flag.changed = true;
                } else {
                    kept.push(inst);
                }
            }
            ir.blocks[bi].insts = kept;
        }
        // Pull the dragged constant defs out of the body ahead of their users
        // so the preheader keeps def-before-use order.
        if !dragged_consts.is_empty() {
            let mut consts: Vec<Inst> = Vec::new();
            for b in &body {
                let bi = b.0 as usize;
                let old = std::mem::take(&mut ir.blocks[bi].insts);
                let mut kept: Vec<Inst> = Vec::with_capacity(old.len());
                for inst in old {
                    match &inst {
                        Inst::Const { dst, .. } if dragged_consts.contains(&dst.0) => {
                            consts.push(inst);
                        }
                        _ => kept.push(inst),
                    }
                }
                ir.blocks[bi].insts = kept;
            }
            consts.extend(hoisted);
            hoisted = consts;
        }

        ir.blocks[preheader.0 as usize].insts.extend(hoisted);
    }

    flag
}

/// Find or create the preheader: the unique predecessor of `header` that is
/// not inside the loop. When several outside jump-preds exist they are
/// forwarded through one fresh block so there is exactly one insertion point.
fn ensure_preheader(
    ir: &mut FuncIr,
    header: BlockId,
    body: &HashSet<u32>,
    flag: &mut ChangeFlag,
) -> Option<BlockId> {
    let outside: Vec<BlockId> = ir
        .preds(header)
        .iter()
        .copied()
        .filter(|p| !body.contains(&p.0))
        .collect();
    match outside.as_slice() {
        [only] => return Some(*only),
        [] => return None, // unreachable header
        _ => {}
    }

    // Multiple outside preds: forward them all through one fresh block.
    let pre = ir.new_block();
    let mut forwarded: Vec<(BlockId, Vec<ValueId>)> = Vec::new();
    for p in outside {
        let pi = p.0 as usize;
        let term = std::mem::replace(&mut ir.blocks[pi].term, Term::Return(None));
        match term {
            Term::Jump(t, args) if t == header => {
                forwarded.push((p, args));
                ir.set_term(p, Term::Jump(pre, Vec::new()));
                flag.changed = true;
            }
            other => {
                // Branch preds keep their terminator untouched.
                ir.blocks[pi].term = other;
            }
        }
    }
    if forwarded.is_empty() {
        // Only branch preds: no safe single insertion point via forwarding.
        // Keep `pre` unused-but-jumping to header to preserve structure.
        ir.set_term(pre, Term::Jump(header, Vec::new()));
        return Some(pre);
    }

    // Forward phi values from each old edge into the new one. Every old edge
    // passed args positionally aligned with the header phis; merge columns by
    // picking the first available value (all preds must agree per phi after
    // SSA renaming — distinct values per pred would mean the preheader split
    // happened pre-renaming, which our pipeline never does post-SSA).
    let phis = ir.block(header).phis.clone();
    let mut new_args = Vec::with_capacity(phis.len());
    for k in 0..phis.len() {
        let v = forwarded
            .iter()
            .filter_map(|(_, args)| args.get(k).copied())
            .next()
            .unwrap_or(ValueId(phis[k].var.0));
        new_args.push(v);
        for (from, args) in &forwarded {
            if let Some(old_v) = args.get(k) {
                register_phi(ir, header, phis[k].dst, *from, *old_v);
            }
        }
    }
    ir.set_term(pre, Term::Jump(header, new_args));
    Some(pre)
}

fn register_phi(ir: &mut FuncIr, block: BlockId, dst: ValueId, from: BlockId, v: ValueId) {
    for p in ir.blocks[block.0 as usize].phis.iter_mut() {
        if p.dst == dst && !p.args.iter().any(|(b, _)| *b == from) {
            p.args.push((from, v));
            p.args.sort_unstable_by_key(|(b, _)| *b);
        }
    }
}
