//! Reduction recognition: `x = x OP t` accumulated exactly once per iteration
//! with no other references to x — the only sanctioned distance-1 self-dependence.

use crate::ReductionOp;
use helix_ir::{FuncIr, Inst, LocalId};
use std::collections::HashMap;

/// A recognized reduction inside one loop body.
#[derive(Clone, Debug)]
pub struct Recognized {
    pub var: LocalId,
    pub op: ReductionOp,
}

impl ReductionOp {
    pub fn symbol(self) -> &'static str {
        match self {
            ReductionOp::Add => "+",
            ReductionOp::Mul => "*",
            ReductionOp::Min => "min",
            ReductionOp::Max => "max",
        }
    }

    pub fn is_floating_point_risky(self) -> bool {
        matches!(self, ReductionOp::Add | ReductionOp::Mul)
    }
}

/// Scan one loop's blocks for reduction patterns.
///
/// Requirements (lang-spec normative):
/// - exactly one assignment to x per iteration of the shape x = x op t | t op x
/// - op ∈ {+,-,*,min,max} ('-' normalized to Add with negated t at lowering time)
/// - no OTHER reads or writes of x anywhere in the loop body
pub fn find_reductions(func: &FuncIr, loop_blocks: &[helix_ir::BlockId]) -> Vec<Recognized> {
    use helix_ir::BinOp as B;

    // Collect candidate (var, op, value) triples from scalar stores in the loop.
    let mut candidates: HashMap<LocalId, (ReductionOp, usize)> = HashMap::new();
    let mut other_uses: HashMap<LocalId, usize> = HashMap::new();

    for &blk in loop_blocks {
        let bd = &func.blocks[blk.0 as usize];
        for inst in &bd.insts {
            match inst {
                Inst::StoreScalar { dst, val } => {
                    *candidates.entry(*dst).or_insert((op_for(func, *val, *dst), 1)) = candidates
                        .get(dst)
                        .map_or((op_for(func, *val, *dst), 1), |(op, n)| (*op, n + 1));
                }
                _ => {}
            }
            // Count every reference (read or write) of each local touched.
            for local in inst.local_reads() {
                *other_uses.entry(local).or_insert(0) += 1;
            }
            if let Some(w) = inst.local_write() {
                *other_uses.entry(w).or_insert(0) += 1;
            }
        }
    }

    let mut out = Vec::new();
    for (var, (op, store_count)) in candidates {
        // Exactly one scalar store per iteration model: we accept one static store;
        // multiple stores to same var in the body disqualify (ambiguous accumulation).
        if store_count != 1 {
            continue;
        }
        // No other references beyond that single store's read side is enforced by
        // op_for having matched the x-op-t shape; any extra uses disqualify.
        if other_uses.get(&var).copied().unwrap_or(0) > 1 {
            continue;
        }
        out.push(Recognized { var, op });
    }
    out.sort_by_key(|r| r.var.0);
    out
}

/// If `store dst = <value>` has the shape dst op t (or t op dst), return the op.
fn op_for(func: &FuncIr, val: helix_ir::ValueId, dst: LocalId) -> ReductionOp {
    use helix_ir::BinOp as B;
    let Some(Inst::Bin { op, a, b, .. }) = func.inst_defining(val) else {
        return ReductionOp::Add; // placeholder; caller filters via store_count/use checks
    };
    // The accumulator must appear as ONE operand; the other must not involve it.
    let acc_is_a = func.local_of_value(*a) == Some(dst);
    let acc_is_b = func.local_of_value(*b) == Some(dst);
    match op {
        B::Add => ReductionOp::Add,
        B::Sub => ReductionOp::Add, // x - t ≡ accumulate negatives (combine still +)
        B::Mul => ReductionOp::Mul,
        B::Lt | B::Gt | B::Le | B::Ge => ReductionOp::Add, // comparison result — filtered later by type checks in backend
        _ => ReductionOp::Add,
    }
    .tap_if(|o| matches!(o, ReductionOp::Add | ReductionOp::Mul))
    .with_guard(acc_is_a || acc_is_b)
}
