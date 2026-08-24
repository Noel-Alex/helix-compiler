//! Reduction recognition: `x = x OP t` accumulated once per iteration with no
//! other references to x — the only sanctioned distance-1 self-dependence.
//!
//! Detection runs on SSA form, where the accumulator's loop-carried state is
//! exactly a header φ whose back-edge operand flows through one associative
//! binop. The checklist (lang-spec normative):
//!
//! 1. header φ for local x,
//! 2. back-edge operand = φ-result OP invariant (or symmetric), OP ∈ {+,-,*}
//!    plus min/max builtins when they appear as Call shapes,
//! 3. exactly ONE store/definition site per iteration (the latch chain),
//! 4. no other USE of any version of x inside the body besides the chain.

use crate::ReductionOp;
use helix_ir::{BinOp, BlockId, FuncIr, Inst, LocalId, ValueId};
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

    /// FP +/* are not associative: parallel combination order changes rounding.
    pub fn is_floating_point_risky(self) -> bool {
        matches!(self, ReductionOp::Add | ReductionOp::Mul)
    }
}

/// Scan an SSA-form loop body for reductions. Returns at most one per local.
pub fn find_reductions(func: &FuncIr, loop_blocks: &[BlockId]) -> Vec<Recognized> {
    let mut out: Vec<Recognized> = Vec::new();

    // Header phis are the candidate accumulators.
    let Some(header) = loop_blocks
        .iter()
        .copied()
        .find(|b| !func.block(*b).phis.is_empty())
    else {
        return out;
    };
    let hb = func.block(header);

    'phi_loop: for phi in &hb.phis {
        if phi.args.len() != 2 {
            continue;
        }
        // Back-edge arg: from a pred inside the loop (≠ header itself).
        let Some((_, back_val)) = phi
            .args
            .iter()
            .find(|(pb, _)| *pb != header && loop_blocks.contains(pb))
        else {
            continue;
        };

        // Chain: back_val must be defined by an associative binop consuming the
        // phi result. Allow one intervening copy shape later; v1 keeps it strict.
        let Some(Inst::Bin { op, a, b, .. }) = def_inst(func, *back_val) else {
            continue;
        };
        let red_op = match op {
            BinOp::Add => ReductionOp::Add,
            BinOp::Sub => ReductionOp::Add, // x -= t ≡ sum of negated terms
            BinOp::Mul => ReductionOp::Mul,
            _ => continue,
        };
        if !((*a == phi.dst) ^ (*b == phi.dst)) {
            continue; // both/neither operands are the accumulator
        }
        // The non-accumulator operand must be loop-invariant OR at least not
        // depend on the accumulator (it may read arrays — fine).
        let other = if *a == phi.dst { *b } else { *a };
        if depends_on(func, other, phi.dst, loop_blocks) {
            continue;
        }

        // No OTHER uses of the phi result anywhere except its consumption by
        // the chain instruction.
        let mut uses_of_phi = 0u32;
        for &blk in loop_blocks {
            let bd = func.block(blk);
            for inst in &bd.insts {
                if inst.uses().contains(&phi.dst) {
                    uses_of_phi += 1;
                }
            }
            for term_arg in bd.term.forwarded_args() {
                if *term_arg == phi.dst {
                    uses_of_phi += 1;
                }
            }
            for p in &bd.phis {
                if p.args.iter().any(|(_, v)| *v == phi.dst) && p.dst != phi.dst {
                    uses_of_phi += 1;
                }
            }
        }
        if uses_of_phi != 1 {
            continue 'phi_loop;
        }

        // One accumulation site per iteration: the single defining binop.
        out.push(Recognized {
            var: phi.var,
            op: red_op,
        });
    }
    out.sort_by_key(|r| r.var.0);
    out
}

fn def_inst(func: &FuncIr, v: ValueId) -> Option<&Inst> {
    for bd in &func.blocks {
        if let Some(i) = bd.insts.iter().find(|i| i.dst() == Some(v)) {
            return Some(i);
        }
    }
    None
}

/// Does `v`'s computation (within the loop) transitively involve `target`?
fn depends_on(func: &FuncIr, v: ValueId, target: ValueId, loop_blocks: &[BlockId]) -> bool {
    let mut seen = HashMap::new();
    let _ = loop_blocks;
    dep_walk(func, v, target, &mut seen, 0)
}

fn dep_walk(
    func: &FuncIr,
    v: ValueId,
    target: ValueId,
    seen: &mut HashMap<ValueId, bool>,
    depth: u32,
) -> bool {
    if depth > 64 {
        return true; // conservative
    }
    if v == target {
        return true;
    }
    if let Some(hit) = seen.get(&v) {
        return *hit;
    }
    seen.insert(v, false);
    let result = match def_inst(func, v) {
        Some(inst) => inst
            .uses()
            .iter()
            .any(|&u| dep_walk(func, u, target, seen, depth + 1)),
        None => false, // defined outside the loop or a slot cell — no dependency path here
    };
    seen.insert(v, result);
    result
}
