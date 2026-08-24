//! CFG simplification: reachability cleanup, block merging, empty-block
//! removal.
//!
//! Three rewrites, iterated to a local fixpoint:
//!
//! 1. **Unreachable blocks** are dropped (forward DFS from entry) and the
//!    rest renumbered densely — unreachable code corrupts every dominance
//!    computation downstream.
//! 2. **Chain merging**: a block ending in `Jump(t)` whose target has exactly
//!    one predecessor folds into its successor when that successor has no
//!    φ-nodes (merging with phis would require renaming their arguments into
//!    the merged body — possible but not worth the bookkeeping here).
//! 3. **Empty-block threading**: a block containing only a jump forwards its
//!    own predecessors' edges to its target (again skipping phi-carrying
//!    targets, for the same reason).
//!
//! All edge surgery goes through [`FuncIr::set_term`] / [`FuncIr::compact`]
//! so predecessor lists and φ argument lists stay symmetric.

use std::collections::HashSet;

use crate::dom::reachability;
use crate::ir::{BlockId, FuncIr, Term};
use crate::passmod::ChangeFlag;

/// Clean up the function's control-flow graph.
pub fn simplify_cfg(ir: &mut FuncIr) -> ChangeFlag {
    let mut flag = ChangeFlag::new();

    // ---- 1. strip unreachable blocks ----------------------------------------
    let live = reachability(ir);
    if live.iter().any(|v| !v) {
        ir.compact(&live);
        flag.changed = true;
    }

    // ---- iterate merges until stable (small functions; bounded rounds) -------
    loop {
        let mut changed_this_round = false;

        // 2. chain merge: b ends in Jump(t), t has one pred (b) and no phis,
        //    and no phi ANYWHERE takes an argument along an edge into t (such
        //    an argument list could not be migrated mechanically here).
        let phi_mentions = |ir: &FuncIr, t: BlockId| -> bool {
            ir.blocks
                .iter()
                .any(|b| b.phis.iter().any(|p| p.args.iter().any(|(f, _)| *f == t)))
        };
        'outer: for bi in 0..ir.blocks.len() {
            let t = match &ir.blocks[bi].term {
                Term::Jump(t, _) => *t,
                _ => continue,
            };
            if t == BlockId(bi as u32) {
                continue; // self-loop: leave alone
            }
            let ti = t.0 as usize;
            if ir.blocks[ti].preds.len() != 1
                || !ir.blocks[ti].phis.is_empty()
                || ti == 0
                || phi_mentions(ir, t)
            {
                continue;
            }
            // Splice t's contents after bi's instructions. Because t has
            // exactly one predecessor (bi) and no phis — and no third-party φ
            // takes a value along an edge into t (checked above) — moving its
            // terminator into bi preserves every edge semantics. Phis in
            // t's SUCCESSORS may list t as their predecessor; retarget those
            // argument entries to bi, which now owns the edge.
            {
                let t_succs: Vec<BlockId> = ir.blocks[ti].term.succs();
                for s in t_succs {
                    for p in ir.blocks[s.0 as usize].phis.iter_mut() {
                        for entry in p.args.iter_mut() {
                            if entry.0 == t {
                                entry.0 = BlockId(bi as u32);
                            }
                        }
                        p.args.sort_unstable_by_key(|(b, _)| *b);
                    }
                }
            }
            let tail = std::mem::take(&mut ir.blocks[ti].insts);
            let term = std::mem::replace(&mut ir.blocks[ti].term, Term::Return(None));
            ir.blocks[bi].insts.extend(tail);
            ir.set_term(BlockId(bi as u32), term);
            // Kill t: no successors and unreachable ⇒ compact() removes it.
            ir.blocks[ti].term = Term::Return(None);
            changed_this_round = true;
            flag.changed = true;
            break 'outer;
        }
        if changed_this_round {
            let live = reachability(ir);
            ir.compact(&live);
            continue;
        }

        // 3. empty-block threading: b = [jump t] only, t has phis? skip. Else
        // retarget preds of b straight to t.
        'thread: for bi in 0..ir.blocks.len() {
            let b = &ir.blocks[bi];
            if !b.insts.is_empty() || !b.phis.is_empty() {
                continue;
            }
            let (t, args) = match &b.term {
                Term::Jump(t, args) => (*t, args.clone()),
                _ => continue,
            };
            if t == BlockId(bi as u32) {
                continue;
            }
            if !args.is_empty() || !ir.block(t).phis.is_empty() {
                continue; // argument forwarding through phis not supported here
            }
            let preds = ir.block(BlockId(bi as u32)).preds.clone();
            if preds.is_empty() {
                continue;
            }
            for p in preds {
                let term = std::mem::replace(&mut ir.blocks[p.0 as usize].term, Term::Return(None));
                let new_term = match term {
                    Term::Jump(x, args) if x == BlockId(bi as u32) => Term::Jump(t, args),
                    Term::Branch { cond, t: tt, f } => Term::Branch {
                        cond,
                        t: if tt == BlockId(bi as u32) { t } else { tt },
                        f: if f == BlockId(bi as u32) { t } else { f },
                    },
                    other => other,
                };
                ir.set_term(p, new_term);
            }
            changed_this_round = true;
            flag.changed = true;
            break 'thread;
        }

        if changed_this_round {
            let live = reachability(ir);
            ir.compact(&live);
        } else {
            break;
        }
    }

    // Final hygiene: normalize phi alignment after all the surgery.
    ir.normalize_phis();
    flag
}

/// Exposed helper: which of these blocks are reachable? (used by tests)
#[must_use]
pub fn reachable_set(ir: &FuncIr) -> HashSet<BlockId> {
    reachability(ir)
        .into_iter()
        .enumerate()
        .filter(|(_, v)| *v)
        .map(|(i, _)| BlockId(i as u32))
        .collect()
}
