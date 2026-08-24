//! Canonical loop recovery: find the induction-variable φ, the latch increment,
//! and the header comparison — turning the IR shape of a `for` back into
//! (iv, start, end) for analysis and reporting.

use crate::Bound;
use crate::loops::Loop;
use helix_ir::{BinOp, BlockId, Constant, FuncIr, Inst, LocalId, Term, ValueId};

/// A recognized canonical counting loop.
#[derive(Clone, Debug)]
pub struct CanonicalLoop {
    /// Induction variable's local slot (source-level `i`).
    pub iv: LocalId,
    /// Value fed from outside the loop (start bound).
    pub start: Bound,
    /// Value the header condition compares against (end bound).
    pub end: Bound,
    /// Latch step (HELIX v1 always 1).
    pub step: i64,
    pub header: BlockId,
    /// SSA name of the induction value inside the loop.
    pub iv_value_in_loop: ValueId,
}

/// Attempt canonicalization; None for non-counting shapes.
pub fn canon(func: &FuncIr, lp: &Loop) -> Option<CanonicalLoop> {
    let header = func.block(lp.header);

    // Find a header φ whose back-edge arg is defined by `<phi> + const`.
    for phi in &header.phis {
        if phi.args.len() != 2 {
            continue;
        }
        // Split incoming args into from-outside vs from-inside-the-loop.
        let mut pre_val: Option<ValueId> = None;
        let mut back_val: Option<ValueId> = None;
        for (pb, pv) in &phi.args {
            if *pb == lp.header {
                continue;
            }
            if lp.blocks.contains(pb) {
                back_val = Some(*pv);
            } else {
                pre_val = Some(*pv);
            }
        }
        let (Some(pre), Some(back)) = (pre_val, back_val) else {
            continue;
        };

        // Back-edge def must be phi.dst + const (or const + phi.dst).
        let step = match find_def(func, back) {
            Some(Inst::Bin {
                op: BinOp::Add,
                a,
                b,
                ..
            }) => {
                if *a == phi.dst {
                    const_i64(func, *b)
                } else if *b == phi.dst {
                    const_i64(func, *a)
                } else {
                    None
                }
            }
            _ => None,
        };
        let Some(step) = step else { continue };
        if step != 1 {
            continue; // HELIX v1 for-loops always step by 1
        }

        // Header terminator must branch on a comparison involving phi.dst.
        let Term::Branch { cond, .. } = &header.term else {
            continue;
        };
        let Some(Inst::Bin { op, a, b, .. }) = find_def(func, *cond) else {
            continue;
        };
        if !matches!(op, BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge) {
            continue;
        }
        let (iv_side, other) = if *a == phi.dst {
            (*a, *b)
        } else if *b == phi.dst {
            (*b, *a)
        } else {
            continue;
        };

        return Some(CanonicalLoop {
            iv: phi.var,
            start: bound_of(func, pre),
            end: bound_of(func, other),
            step,
            header: lp.header,
            iv_value_in_loop: iv_side,
        });
    }
    None
}

/// Locate the instruction defining `v` anywhere in the function.
pub fn find_def(func: &FuncIr, v: ValueId) -> Option<&Inst> {
    for bd in &func.blocks {
        if let Some(i) = bd.insts.iter().find(|i| i.dst() == Some(v)) {
            return Some(i);
        }
    }
    None
}

fn const_i64(func: &FuncIr, v: ValueId) -> Option<i64> {
    match find_def(func, v) {
        Some(Inst::Const { c, .. }) => match c {
            Constant::I64(x) => Some(*x),
            Constant::I32(x) => Some(i64::from(*x)),
            _ => None,
        },
        _ => None,
    }
}

fn bound_of(func: &FuncIr, v: ValueId) -> Bound {
    const_i64(func, v).map_or(Bound::Sym(v.0), Bound::Const)
}
