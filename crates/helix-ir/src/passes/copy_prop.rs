//! Copy propagation (post-SSA).
//!
//! A "copy" here is the degenerate SSA shape `v2 = v1`: either a φ with all
//! arguments equal (`v2 = φ(v1; v1)`) or — after const_prop — an operation
//! whose result equals one operand. Where `v1` dominates every use of `v2`,
//! all uses of `v2` can be rewritten to `v1` and the copy deleted.
//!
//! The dominance guard matters: rewriting uses that `v1` does *not* dominate
//! would fabricate values read before their def. For phi-copies we check
//! domination per use site; for the common case (both defined in the same
//! block, copy immediately after source) this holds trivially.

use crate::dom::{dominators, reachability};
use crate::ir::{FuncIr, Inst, Term};
use crate::passmod::ChangeFlag;

/// Rewrite dominated uses through dominating copies; returns whether anything
/// changed.
pub fn copy_prop(ir: &mut FuncIr) -> ChangeFlag {
    let mut flag = ChangeFlag::new();

    // Collect candidate copies first: dst -> src for phis whose args are all
    // identical (and non-self), where the shared source has a SINGLE def.
    // The single-def requirement is what makes the rewrite sound: with one
    // def we can check it dominates every use site; a source id with several
    // defs (a pre-renaming cell spelling reaching the pass unrenamed) has no
    // single dominance relation and rewriting would fabricate values.
    // Identity casts are NOT copies (they may change representation at the
    // backend boundary).
    let mut copies: Vec<(crate::ir::ValueId, crate::ir::ValueId)> = Vec::new();
    let mut copy_sites: Vec<(u32, bool)> = Vec::new(); // (block idx, is_phi)
    let live = reachability(ir);

    for bi in 0..ir.blocks.len() {
        let b = &ir.blocks[bi];
        for p in &b.phis {
            if p.args.is_empty() {
                continue;
            }
            let mut args = p.args.iter().map(|(_, v)| *v);
            let first = args.next().unwrap_or(p.dst);
            if !args.any(|v| v != first) && first != p.dst {
                // Single-def check on `first`.
                let def_count = ir
                    .blocks
                    .iter()
                    .map(|bb| {
                        bb.phis.iter().filter(|q| q.dst == first).count()
                            + bb.insts.iter().filter(|i| i.dst() == Some(first)).count()
                    })
                    .sum::<usize>();
                if def_count == 1 {
                    copies.push((p.dst, first));
                    copy_sites.push((bi as u32, true));
                }
            }
        }
        let _ = live;
    }

    if copies.is_empty() {
        return flag;
    }

    let doms = dominators(ir);
    let def_block_of = |ir: &FuncIr, v: crate::ir::ValueId| -> Option<u32> {
        for bi in 0..ir.blocks.len() {
            if ir.blocks[bi].phis.iter().any(|p| p.dst == v)
                || ir.blocks[bi].insts.iter().any(|i| i.dst() == Some(v))
            {
                return Some(bi as u32);
            }
        }
        None
    };

    // Rewrite uses where src dominates the use site.
    let n_blocks = ir.blocks.len();
    for (dst, src) in &copies {
        let Some(sdb) = def_block_of(ir, *src) else {
            continue;
        };
        for bi in 0..n_blocks {
            for inst in ir.blocks[bi].insts.iter_mut() {
                let site_ok =
                    doms.dominates(crate::ir::BlockId(sdb), crate::ir::BlockId(bi as u32));
                if !site_ok {
                    continue;
                }
                inst.rewrite_uses(&mut |v| {
                    if v == *dst {
                        flag.changed = true;
                        *src
                    } else {
                        v
                    }
                });
            }
            match &mut ir.blocks[bi].term {
                Term::Jump(_, args) => {
                    for a in args.iter_mut() {
                        if *a == *dst {
                            *a = *src;
                            flag.changed = true;
                        }
                    }
                }
                Term::Branch { cond, .. } => {
                    if *cond == *dst {
                        *cond = *src;
                        flag.changed = true;
                    }
                }
                Term::Return(v) => {
                    if *v == Some(*dst) {
                        *v = Some(*src);
                        flag.changed = true;
                    }
                }
            }
        }

        // Phi arguments too (they read on edges; conservative: rewrite when
        // the copy's block dominates the predecessor block).
        for bi in 0..n_blocks {
            for p in ir.blocks[bi].phis.iter_mut() {
                for (from, v) in p.args.iter_mut() {
                    if *v == *dst
                        && doms.dominates(crate::ir::BlockId(sdb), crate::ir::BlockId(from.0))
                    {
                        *v = *src;
                        flag.changed = true;
                    }
                }
            }
        }
    }

    // Delete now-unused copies. NOTE: `p.dst == v` is deliberately NOT a use
    // — a definition site never counts as a reference, otherwise every
    // phi-copy would keep itself alive.
    let used = |v: crate::ir::ValueId, ir: &FuncIr| -> bool {
        ir.blocks.iter().any(|b| {
            b.phis.iter().any(|p| p.args.iter().any(|(_, x)| *x == v))
                || b.insts.iter().any(|i| i.uses().contains(&v))
                || matches!(&b.term, Term::Jump(_, args) if args.contains(&v))
                || matches!(&b.term, Term::Branch { cond, .. } if *cond == v)
                || matches!(&b.term, Term::Return(Some(x)) if *x == v)
        })
    };

    for (k, (dst, _)) in copies.iter().enumerate() {
        let (bi, is_phi) = copy_sites[k];
        if used(*dst, ir) {
            continue;
        }
        if is_phi {
            let before = ir.blocks[bi as usize].phis.len();
            ir.blocks[bi as usize].phis.retain(|p| p.dst != *dst);
            if ir.blocks[bi as usize].phis.len() != before {
                flag.changed = true;
                // The phi list shrank: every predecessor's jump argument list
                // must shrink by the same entry or the arity contract breaks.
                for p in ir.blocks[bi as usize].preds.clone() {
                    if let Term::Jump(t, args) = &mut ir.blocks[p.0 as usize].term
                        && t.0 == bi
                    {
                        args.pop();
                    }
                }
            }
        }
    }

    flag
}

/// Silence the unused-import lint for `Inst` kept for future cast handling.
const _: Option<Inst> = None;
