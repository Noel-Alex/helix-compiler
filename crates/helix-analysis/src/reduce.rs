//! Reduction recognition: `x = x OP t` accumulated once per iteration with no
//! other references to x — the only sanctioned distance-1 self-dependence.
//!
//! Detection runs on SSA form, where the accumulator's loop-carried state is
//! exactly a header φ whose back-edge operand flows through one associative
//! computation. The checklist (lang-spec normative):
//!
//! 1. a loop φ merging `var` from outside and from the back edge,
//! 2. the back-edge operand defined by an associative shape:
//!    * `binop(acc, t)` with OP ∈ {+, -, *} — `x -= t` folds into the Add
//!      family as a sum of negated terms,
//!    * `call min(...)` / `call max(...)` with the accumulator among the
//!      arguments,
//! 3. exactly one accumulation site per iteration (the chain above),
//! 4. **zero** other uses of any SSA name of the accumulator inside the loop
//!    besides the chain itself (`extra_reads` — lang-spec: "referenced nowhere
//!    else in the body").
//!
//! ## The two spellings of "consumes the accumulator"
//!
//! The IR renamer rewrites every use in a block against the block's final
//! definition stack, so the accumulation chain `dot = dot + p` (read and write
//! of `dot` in one statement) surfaces as a *self-referential* definition:
//! the chain's destination id also appears as its own operand. Recognition
//! therefore accepts an operand equal to the φ result **or** equal to the
//! chain's own destination; both mean "the previous iteration's value".
//! Anything else in that operand position (both operands accumulators, or
//! neither) disqualifies the shape.
//!
//! ## What does NOT count as an extra use
//!
//! The latch's terminator forwards the accumulated value into the header φ;
//! that edge argument *is* the reduction flow itself, not a stray read, so it
//! is exempted positionally. Likewise the candidate φ's own input column and
//! the chain instruction's operand slots are part of the sanctioned shape.
//! Everything else — a second statement touching `x`, a branch condition, a
//! φ of another variable reading `x` — disqualifies the recognition.

use crate::ReductionOp;
use helix_ir::{BinOp, BlockId, FuncIr, Inst, LocalId, Term, ValueId};
use std::collections::HashMap;

/// A recognized reduction inside one loop body.
#[derive(Clone, Debug)]
pub struct Recognized {
    /// Source variable being accumulated.
    pub var: LocalId,
    /// The associative operator.
    pub op: ReductionOp,
}

impl ReductionOp {
    /// Printable operator symbol for report lines (`REDUCTION(+)`).
    #[must_use]
    pub fn symbol(self) -> &'static str {
        match self {
            ReductionOp::Add => "+",
            ReductionOp::Mul => "*",
            ReductionOp::Min => "min",
            ReductionOp::Max => "max",
        }
    }

    /// FP `+`/`*` are not associative: parallel combination order changes
    /// rounding (documented nondeterminism, OpenMP-style).
    #[must_use]
    pub fn is_floating_point_risky(self) -> bool {
        matches!(self, ReductionOp::Add | ReductionOp::Mul)
    }
}

/// Scan an SSA-form loop body for reductions.
///
/// `exclude` lists locals that must never be reported — the induction
/// variable's φ has exactly the additive shape of a reduction and is ruled
/// out by passing the canonicalized iv here. Returns at most one recognition
/// per local, sorted by local id for stable reports.
pub fn find_reductions(
    func: &FuncIr,
    loop_blocks: &[BlockId],
    exclude: &[LocalId],
) -> Vec<Recognized> {
    let mut out: Vec<Recognized> = Vec::new();

    // Candidate accumulators: every 2-argument φ hosted inside the loop whose
    // variable is neither excluded nor an array slot (arrays carry no phis).
    for &blk in loop_blocks {
        for phi in &func.block(blk).phis {
            if phi.args.len() != 2 || exclude.contains(&phi.var) {
                continue;
            }
            // Split incoming args into from-outside vs from-inside-the-loop.
            let mut entry_val: Option<ValueId> = None;
            let mut back_val: Option<ValueId> = None;
            for (pb, pv) in &phi.args {
                if pb == &blk {
                    continue; // degenerate self-reference; not our shape
                }
                if loop_blocks.contains(pb) {
                    back_val = Some(*pv);
                } else {
                    entry_val = Some(*pv);
                }
            }
            let (Some(_entry), Some(back)) = (entry_val, back_val) else {
                continue;
            };

            let Some((op, other)) = match_chain(func, phi.dst, back) else {
                continue;
            };
            // The feeding computation must not transitively read the
            // accumulator through any other path (that would be a second use).
            if depends_on_any(func, other, &[phi.dst, back]) {
                continue;
            }
            // No OTHER uses of either accumulator name anywhere in the loop
            // except the defining chain itself.
            if extra_reads(func, loop_blocks, phi.dst, back) > 0 {
                continue;
            }
            if !out.iter().any(|r| r.var == phi.var) {
                out.push(Recognized { var: phi.var, op });
            }
        }
    }
    out.sort_by_key(|r| r.var.0);
    out
}

/// Match the accumulation chain hanging off a header φ.
///
/// Returns `(operator, other_operand)` when `back` is defined by an
/// associative shape consuming the accumulator exactly once.
fn match_chain(func: &FuncIr, phi_dst: ValueId, back: ValueId) -> Option<(ReductionOp, ValueId)> {
    match def_inst(func, back)? {
        Inst::Bin { op, a, b, .. } => {
            let acc_a = *a == phi_dst || *a == back;
            let acc_b = *b == phi_dst || *b == back;
            if acc_a == acc_b {
                return None; // both or neither operands are the accumulator
            }
            let red_op = match op {
                BinOp::Add => ReductionOp::Add,
                BinOp::Sub => ReductionOp::Add, // x -= t ≡ sum of negated terms
                BinOp::Mul => ReductionOp::Mul,
                _ => return None,
            };
            Some((red_op, if acc_a { *b } else { *a }))
        }
        Inst::Call(c) => {
            if c.dst != Some(back) {
                return None;
            }
            let op = match c.callee.as_str() {
                "min" => ReductionOp::Min,
                "max" => ReductionOp::Max,
                _ => return None,
            };
            // Exactly one accumulator argument; the other operand (if any
            // beyond it) is the folded term.
            let acc_positions = c
                .args
                .iter()
                .filter(|&&a| a == phi_dst || a == back)
                .count();
            if acc_positions != 1 {
                return None;
            }
            let other = *c
                .args
                .iter()
                .find(|&&a| a != phi_dst && a != back)
                .unwrap_or(&ValueId(u32::MAX));
            Some((op, other))
        }
        _ => None,
    }
}

/// Locate the instruction defining `v` anywhere in the function.
fn def_inst(func: &FuncIr, v: ValueId) -> Option<&Inst> {
    func.inst_defining(v)
}

/// Does `v`'s computation transitively involve any of `targets`?
///
/// Walks pure instruction operands only; loads, calls and values with no
/// in-function instruction definition terminate the walk (they cannot embed
/// the accumulator's SSA name).
fn depends_on_any(func: &FuncIr, v: ValueId, targets: &[ValueId]) -> bool {
    if targets.contains(&v) {
        return true;
    }
    let mut seen: HashMap<ValueId, bool> = HashMap::new();
    dep_walk(func, v, targets, &mut seen, 0)
}

fn dep_walk(
    func: &FuncIr,
    v: ValueId,
    targets: &[ValueId],
    seen: &mut HashMap<ValueId, bool>,
    depth: u32,
) -> bool {
    if targets.contains(&v) {
        return true;
    }
    if depth > 64 {
        return true; // conservative: assume dependency on unfathomable shapes
    }
    if let Some(hit) = seen.get(&v) {
        return *hit;
    }
    seen.insert(v, false);
    let result = match def_inst(func, v) {
        Some(inst) if inst.is_pure() => inst
            .uses()
            .iter()
            .any(|&u| dep_walk(func, u, targets, seen, depth + 1)),
        _ => false, // loads/calls/externals: no transparent operand path
    };
    seen.insert(v, result);
    result
}

/// Count uses of either SSA name of the accumulator inside the loop beyond
/// the accumulation chain itself (lang-spec: "referenced nowhere else").
///
/// Exempt positions — all part of the sanctioned reduction shape:
/// * the chain instruction (unique definition of `back`) and its operand
///   slots,
/// * the latch terminator argument forwarding `back` into the candidate φ
///   (`phi_dst`) on the header block — that edge IS the reduction flow,
/// * the candidate φ's own input column.
///
/// Every other occurrence counts: a second statement touching the variable,
/// a branch condition, a φ of another variable reading the accumulator.
fn extra_reads(func: &FuncIr, loop_blocks: &[BlockId], phi_dst: ValueId, back: ValueId) -> u32 {
    let mut uses = 0u32;
    for &blk in loop_blocks {
        let bd = func.block(blk);
        for inst in &bd.insts {
            if inst.dst() == Some(back) {
                continue; // the chain itself
            }
            uses += inst
                .uses()
                .iter()
                .filter(|u| **u == phi_dst || **u == back)
                .count() as u32;
        }
        // Terminator traffic: only the jump feeding the candidate φ is part
        // of the shape; any other forwarded use (a different consumer) is an
        // extra read.
        if let Term::Jump(t, args) = &bd.term {
            let feeds_candidate = *t == header_of(func, loop_blocks, phi_dst);
            for a in args {
                if (*a == phi_dst || *a == back) && !(feeds_candidate && *a == back) {
                    uses += 1;
                }
            }
        } else {
            for term_arg in bd.term.forwarded_args() {
                if *term_arg == phi_dst || *term_arg == back {
                    uses += 1;
                }
            }
            if let Term::Branch { cond, .. } = &bd.term
                && (*cond == phi_dst || *cond == back)
            {
                uses += 1;
            }
        }
        for p in &bd.phis {
            if p.dst == phi_dst {
                continue; // the candidate φ's own input column
            }
            if p.args.iter().any(|(_, v)| *v == phi_dst || *v == back) {
                uses += 1;
            }
        }
    }
    uses
}

/// The block hosting the candidate φ with destination `dst` (its header).
fn header_of(func: &FuncIr, loop_blocks: &[BlockId], dst: ValueId) -> BlockId {
    loop_blocks
        .iter()
        .copied()
        .find(|&b| func.block(b).phis.iter().any(|p| p.dst == dst))
        .unwrap_or(BlockId(u32::MAX))
}
