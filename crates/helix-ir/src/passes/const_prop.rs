//! Constant propagation (post-SSA) + branch folding.
//!
//! Post-SSA every value has exactly one definition, so "which constant does
//! this use see?" is a direct lookup: if a use's operand names a value whose
//! unique def is `Inst::Const`, the operand is rewritten to that def's id —
//! which is already what it names, so step 2 below actually normalizes
//! *duplicated* constant defs to one canonical def (CSE does this too; doing
//! it here keeps branch folding simple). The pass then folds the
//! now-constant operations via [`super::const_fold`] and collapses
//! `Branch(cond = const bool)` into an unconditional jump, after which
//! [`super::simplify_cfg`] cleans up stranded blocks.
//!
//! Pre-SSA this transformation would be wrong: a cell may have several defs,
//! and only straight-line reaching analysis could tell which one applies. The
//! pass therefore refuses to run on non-SSA functions (it becomes a no-op),
//! keeping it total over any input.

use crate::ir::{BlockId, FuncIr, Inst, Term, ValueId};
use crate::passmod::ChangeFlag;
use crate::ssa;

/// Propagate constants through SSA names; fold branches with constant conds.
pub fn const_prop(ir: &mut FuncIr) -> ChangeFlag {
    let mut flag = ChangeFlag::new();
    if !ssa::is_ssa(ir) {
        return flag; // unsafe pre-SSA by design; documented above
    }

    // 1. Canonicalize duplicated constant defs: identical payloads collapse to
    //    their first definition.
    let mut canonical: std::collections::HashMap<String, ValueId> =
        std::collections::HashMap::new();
    let mut replacements: Vec<(ValueId, ValueId)> = Vec::new();
    for b in &ir.blocks {
        for inst in &b.insts {
            if let Inst::Const { dst, c } = inst {
                let key = format!("{c:?}");
                match canonical.get(&key) {
                    Some(leader) if leader != dst => replacements.push((*dst, *leader)),
                    Some(_) => {}
                    None => {
                        canonical.insert(key, *dst);
                    }
                }
            }
        }
    }
    for (dup, leader) in replacements {
        ir.replace_all_uses(dup, leader);
        for b in ir.blocks.iter_mut() {
            b.insts.retain(|i| i.dst() != Some(dup));
        }
        flag.changed = true;
    }

    // 2. Fold newly-constant operations.
    if super::const_fold::const_fold(ir).changed {
        flag.changed = true;
    }

    // 3. Branch folding: cond defined by Const(bool) -> Jump.
    for bi in 0..ir.blocks.len() {
        let cond_const = match &ir.blocks[bi].term {
            Term::Branch { cond, .. } => find_bool_def(ir, *cond),
            _ => None,
        };
        if let Some(b) = cond_const {
            let term = std::mem::replace(&mut ir.blocks[bi].term, Term::Return(None));
            if let Term::Branch { t, f, .. } = term {
                let target = if b { t } else { f };
                let args = phi_args_for(ir, BlockId(bi as u32), target);
                ir.set_term(BlockId(bi as u32), Term::Jump(target, args));
                flag.changed = true;
            }
        }
    }

    if flag.changed {
        // Dropped edges may strand blocks or leave empty chains behind.
        super::simplify_cfg::simplify_cfg(ir);
    }
    flag
}

use crate::ir::Constant;

fn find_bool_def(ir: &FuncIr, v: ValueId) -> Option<bool> {
    for b in &ir.blocks {
        for inst in &b.insts {
            if let Inst::Const { dst, c } = inst
                && *dst == v
                && let Constant::Bool(bv) = c
            {
                return Some(*bv);
            }
        }
    }
    None
}

/// Values to forward from `from` to `target`'s phis (identity mapping post-
/// SSA: each edge keeps whatever it currently passes).
fn phi_args_for(ir: &FuncIr, from: BlockId, target: BlockId) -> Vec<ValueId> {
    ir.block(target)
        .phis
        .iter()
        .map(|p| {
            p.args
                .iter()
                .find(|(f, _)| *f == from)
                .map(|(_, v)| *v)
                .unwrap_or(ValueId(p.var.0))
        })
        .collect()
}
