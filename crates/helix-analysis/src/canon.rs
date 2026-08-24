//! Canonical loop recovery: identify the induction variable phi, the latch
//! increment, and the header comparison — turning the IR shape of a `for` loop
//! back into (iv, start, end) for analysis and reporting.

use crate::loops::Loop;
use crate::Bound;
use helix_ir::{BinOp, BlockId, FuncIr, Inst, LocalId, Term, ValueId};

/// A recognized canonical single-induction loop.
#[derive(Clone, Debug)]
pub struct CanonicalLoop {
    /// The induction variable local (source-level name index).
    pub iv: LocalId,
    /// The header phi's incoming value from the preheader (start bound).
    pub start: Bound,
    /// Value compared against in the header condition (end bound), if symbolic.
    pub end: Bound,
    /// Latch increment (usually 1).
    pub step: i64,
    /// Header block that holds iv = phi(start, iv+step) and the branch.
    pub header: BlockId,
    /// The exit block taken when the condition fails.
    pub exit: Option<BlockId>,
    /// SSA name of iv inside the loop (header param / phi result).
    pub iv_value_in_loop: ValueId,
}

/// Attempt canonicalization of `loop_`. Returns None when the shape doesn't
/// match a simple counting loop (e.g. while-style or data-dependent exit).
pub fn canon(func: &FuncIr, loop_: &Loop) -> Option<CanonicalLoop> {
    let header = &func.blocks[loop_.header.0 as usize];

    // Find the phi whose two args are (outside-value, inside-value) and whose
    // inside value is defined by add 1 in a latch block that jumps back to header.
    for phi in &header.phis {
        if phi.args.len() != 2 {
            continue;
        }
        let (in_from_pre, in_from_back) = {
            // Identify which arg comes around the back edge: the pred whose block
            // is inside the loop body and is not the header.
            let mut pre = None;
            let mut back = None;
            for (pb, pv) in &phi.args {
                if *pb == loop_.header {
                    continue;
                }
                if loop_.blocks.contains(pb) {
                    back = Some((pb, pv));
                } else {
                    pre = Some((pb, pv));
                }
            }
            match (pre, back) {
                (Some(p), Some(b)) => (p.1, b.1),
                _ => continue,
            }
        };

        // Back-edge value must be iv_value + const-step.
        let step = match func.inst_defining(in_from_back) {
            Some(Inst::Bin { op: BinOp::Add, a, b, .. }) => {
                if func.const_of(*b) == Some(1) && *a == phi.dst {
                    1
                } else if func.const_of(*a) == Some(1) && *b == phi.dst {
                    1
                } else if let Some(s) = func.const_of(*b) {
                    if *a == phi.dst {
                        s
                    } else {
                        continue;
                    }
                } else {
                    continue;
                }
            }
            _ => continue,
        };
        let _ = step;

        // Header must end in Branch on a comparison involving the phi.
        let (end_bound, exit, cmp_ok) = match &header.term {
            Term::Branch { cond, t, f } => {
                match func.inst_defining(*cond) {
                    Some(Inst::Bin { op, a, b, .. })
                        if matches!(op, BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge) =>
                    {
                        let (iv_side, other) = if *a == phi.dst { (*a, *b) } else if *b == phi.dst { (*b, *a) } else {
                            continue;
                        };
                        let bound = match func.const_of(other) {
                            Some(c) => Bound::Const(c),
                            None => Bound::Sym(other.0),
                        };
                        // Exit is the successor NOT inside the loop.
                        let exit_blk = if !loop_.blocks.contains(f) { Some(*f) } else { Some(*t) };
                        (bound, exit_blk, iv_side == phi.dst)
                    }
                    _ => continue,
                }
            }
            _ => continue,
        };
        let _ = cmp_ok;

        return Some(CanonicalLoop {
            iv: phi.var,
            start: bound_from_value(func, in_from_pre),
            end: end_bound,
            step: step.max(1),
            header: loop_.header,
            exit,
            iv_value_in_loop: phi.dst,
        });
    }
    None
}

fn bound_from_value(func: &FuncIr, v: ValueId) -> Bound {
    match func.const_of(v) {
        Some(c) => Bound::Const(c),
        None => Bound::Sym(v.0),
    }
}
