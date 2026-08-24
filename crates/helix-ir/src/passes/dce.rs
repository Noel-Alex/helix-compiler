//! Dead-code elimination by mark-and-sweep over effect reachability.
//!
//! Roots are everything observable: `Store` (callee writes escape), `Call`
//! (arbitrary effects — even `zeros` allocates), `Load` (may trap on a bad
//! index, so deleting one would erase a mandated runtime error). Terminators
//! are structural roots by definition. A backward worklist marks every pure
//! instruction whose value feeds a marked instruction; the sweep then deletes
//! unmarked pure instructions and unmarked non-parameter phis.
//!
//! The mark phase is a transitive closure over operand edges. Post-SSA names
//! have unique defs, so "who produces this operand?" is one hash lookup into
//! the def index built up front. Pre-SSA cell spellings are skipped (they
//! have no unique def site; `to_ssa` cleans them up).

use std::collections::HashMap;

use crate::ir::{FuncIr, Inst, ValueId};
use crate::passmod::ChangeFlag;

/// One pending node of the backward mark walk.
enum Node {
    /// Instruction at `(block, index)`.
    Inst(u32, usize),
    /// Phi defining `dst`.
    Phi(ValueId),
}

/// Sweep dead pure instructions and phis; returns whether anything was
/// removed.
pub fn dce(ir: &mut FuncIr) -> ChangeFlag {
    let mut flag = ChangeFlag::new();

    // ---- def index: value id -> def site ---------------------------------------
    #[derive(Clone, Copy)]
    enum Site {
        Inst(u32, usize),
        Phi(ValueId),
    }
    let mut def_site: HashMap<u32, Site> = HashMap::new();
    for bi in 0..ir.blocks.len() {
        for p in &ir.blocks[bi].phis {
            def_site.entry(p.dst.0).or_insert(Site::Phi(p.dst));
        }
        for (ii, inst) in ir.blocks[bi].insts.iter().enumerate() {
            if let Some(d) = inst.dst() {
                def_site.entry(d.0).or_insert(Site::Inst(bi as u32, ii));
            }
        }
    }

    // ---- mark --------------------------------------------------------------------
    let mut live_insts: Vec<Vec<bool>> = ir
        .blocks
        .iter()
        .map(|b| vec![false; b.insts.len()])
        .collect();
    let mut live_phis: Vec<Vec<bool>> = ir
        .blocks
        .iter()
        .map(|b| vec![false; b.phis.len()])
        .collect();

    // Locate a phi by its dst value (block idx, phi idx).
    let phi_pos = |ir: &FuncIr, dst: ValueId| -> Option<(usize, usize)> {
        for (bi, b) in ir.blocks.iter().enumerate() {
            if let Some(pi) = b.phis.iter().position(|p| p.dst == dst) {
                return Some((bi, pi));
            }
        }
        None
    };

    let mut work: Vec<Node> = Vec::new();

    // Seed with effecting instructions AND every value read by a terminator
    // (branch conditions, jump arguments, return values are all observable).
    for bi in 0..ir.blocks.len() {
        for (ii, inst) in ir.blocks[bi].insts.iter().enumerate() {
            if matches!(inst, Inst::Load(_) | Inst::Store { .. } | Inst::Call { .. }) {
                live_insts[bi][ii] = true;
                work.push(Node::Inst(bi as u32, ii));
            }
        }
        // Mark the defs of terminator operands live by walking them as if
        // they were used by a virtual root instruction.
        let term_uses: Vec<ValueId> = match &ir.blocks[bi].term {
            crate::ir::Term::Jump(_, args) => args.clone(),
            crate::ir::Term::Branch { cond, .. } => vec![*cond],
            crate::ir::Term::Return(v) => v.iter().copied().collect(),
        };
        for u in term_uses {
            if ir.is_slot_value(u) {
                continue;
            }
            match def_site.get(&u.0) {
                Some(Site::Inst(dbi, dii)) => {
                    if !live_insts[*dbi as usize][*dii] {
                        live_insts[*dbi as usize][*dii] = true;
                        work.push(Node::Inst(*dbi, *dii));
                    }
                }
                Some(Site::Phi(dst)) => {
                    if let Some((pbi, pii)) = phi_pos(ir, *dst)
                        && !live_phis[pbi][pii]
                    {
                        live_phis[pbi][pii] = true;
                        work.push(Node::Phi(*dst));
                    }
                }
                None => {}
            }
        }
    }

    while let Some(node) = work.pop() {
        let uses: Vec<ValueId> = match node {
            Node::Inst(bi, ii) => ir.blocks[bi as usize].insts[ii].uses(),
            Node::Phi(dst) => match phi_pos(ir, dst) {
                Some((bi, pi)) => ir.blocks[bi].phis[pi]
                    .args
                    .iter()
                    .map(|(_, v)| *v)
                    .collect(),
                None => continue,
            },
        };
        for u in uses {
            if ir.is_slot_value(u) {
                continue;
            }
            match def_site.get(&u.0) {
                Some(Site::Inst(bi, ii)) => {
                    if !live_insts[*bi as usize][*ii] {
                        live_insts[*bi as usize][*ii] = true;
                        work.push(Node::Inst(*bi, *ii));
                    }
                }
                Some(Site::Phi(dst)) => {
                    if let Some((bi, pi)) = phi_pos(ir, *dst)
                        && !live_phis[bi][pi]
                    {
                        live_phis[bi][pi] = true;
                        work.push(Node::Phi(*dst));
                    }
                }
                None => {}
            }
        }
    }

    // How many definitions does each value id have? Post-SSA this is 1 for
    // every defined id; functions that reach DCE with a multiply-defined id
    // (e.g. fib's `$ret` accumulator, defined once per return arm) need it:
    // every def of such an id is a root, because any of them may feed the
    // jump/phi arguments that reference the id on their own edge.
    let mut def_count: HashMap<u32, usize> = HashMap::new();
    for b in &ir.blocks {
        for p in &b.phis {
            *def_count.entry(p.dst.0).or_insert(0) += 1;
        }
        for i in &b.insts {
            if let Some(d) = i.dst() {
                *def_count.entry(d.0).or_insert(0) += 1;
            }
        }
    }
    for (bi, block) in ir.blocks.iter().enumerate() {
        for (ii, inst) in block.insts.iter().enumerate() {
            if let Some(d) = inst.dst()
                && def_count[&d.0] > 1
                && !live_insts[bi][ii]
            {
                live_insts[bi][ii] = true;
                work.push(Node::Inst(bi as u32, ii));
            }
        }
    }
    while let Some(node) = work.pop() {
        let uses: Vec<ValueId> = match node {
            Node::Inst(bi, ii) => ir.blocks[bi as usize].insts[ii].uses(),
            Node::Phi(dst) => match phi_pos(ir, dst) {
                Some((bi, pi)) => ir.blocks[bi].phis[pi]
                    .args
                    .iter()
                    .map(|(_, v)| *v)
                    .collect(),
                None => continue,
            },
        };
        for u in uses {
            if ir.is_slot_value(u) {
                continue;
            }
            match def_site.get(&u.0) {
                Some(Site::Inst(bi, ii)) => {
                    if !live_insts[*bi as usize][*ii] {
                        live_insts[*bi as usize][*ii] = true;
                        work.push(Node::Inst(*bi, *ii));
                    }
                }
                Some(Site::Phi(dst)) => {
                    if let Some((bi, pi)) = phi_pos(ir, *dst)
                        && !live_phis[bi][pi]
                    {
                        live_phis[bi][pi] = true;
                        work.push(Node::Phi(*dst));
                    }
                }
                None => {}
            }
        }
    }

    // ---- sweep ---------------------------------------------------------------------
    // Track which phi positions survive per block so predecessor jump
    // argument lists can be filtered to exactly the surviving columns.
    let mut phi_keep: Vec<Vec<bool>> = ir
        .blocks
        .iter()
        .map(|b| Vec::with_capacity(b.phis.len()))
        .collect();
    for bi in 0..ir.blocks.len() {
        let old = std::mem::take(&mut ir.blocks[bi].insts);
        let mut kept: Vec<Inst> = Vec::with_capacity(old.len());
        for (ii, inst) in old.into_iter().enumerate() {
            let pure = matches!(
                inst,
                Inst::Const { .. } | Inst::Bin { .. } | Inst::Unary { .. } | Inst::Cast { .. }
            );
            // Two exemptions keep the sweep safe on multi-def (non-renamed)
            // ids:
            //
            // * **Non-live defs with surviving references.** The mark walk
            //   resolves every use through `def_site`, whose entries are the
            //   FIRST def of each id (`or_insert`). When a function reaches
            //   DCE carrying a multiply-defined value — e.g. fib's `$ret`
            //   accumulator, defined by one `v = …` per return arm — marking
            //   `v`'s uses keeps only the first-encountered def alive. The
            //   other same-id defs are "not live" but deleting them would
            //   strand the jump/phi arguments that legitimately name that id
            //   on their own edge. So: delete an unmarked pure def only when
            //   NO other definition of the same id exists anywhere.
            // * **Cell-range ids** (`dst < n_source_locals`) are pre-SSA cell
            //   spellings that phi/jump args may reference without any unique
            //   def site at all; they always stay.
            let d = inst.dst();
            let is_cell_def = d.is_some_and(|x| (x.0 as usize) < ir.n_source_locals);
            let multi_def = d.is_some_and(|x| def_count[&x.0] > 1);
            if live_insts[bi][ii] || !pure || is_cell_def || multi_def {
                kept.push(inst);
            } else {
                flag.changed = true;
            }
        }
        ir.blocks[bi].insts = kept;

        let old_phis = std::mem::take(&mut ir.blocks[bi].phis);
        let mut kept_phis = Vec::with_capacity(old_phis.len());
        let mut keep_mask = Vec::with_capacity(old_phis.len());
        for (pi, p) in old_phis.into_iter().enumerate() {
            // Zero-arg phis are function parameters: always kept.
            if p.args.is_empty() || live_phis[bi][pi] {
                keep_mask.push(true);
                kept_phis.push(p);
            } else {
                keep_mask.push(false);
                flag.changed = true;
            }
        }
        ir.blocks[bi].phis = kept_phis;
        phi_keep[bi] = keep_mask;
    }

    // Filter predecessor jump argument lists to the surviving phi columns.
    for (bi, keep) in phi_keep.iter().enumerate() {
        if !keep.iter().any(|k| !k) {
            continue; // nothing deleted here
        }
        let target = crate::ir::BlockId(bi as u32);
        let preds: Vec<crate::ir::BlockId> = ir.blocks[bi].preds.clone();
        for p in preds {
            if let crate::ir::Term::Jump(t, args) = &mut ir.blocks[p.0 as usize].term
                && *t == target
            {
                let filtered: Vec<crate::ir::ValueId> = args
                    .iter()
                    .zip(keep.iter())
                    .filter(|(_, k)| **k)
                    .map(|(v, _)| *v)
                    .collect();
                *args = filtered;
            }
        }
    }

    flag
}
