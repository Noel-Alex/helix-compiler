//! Affine subscript extraction: express each Load/Store index as `a*i + b`
//! relative to a loop's induction variable, walking the SSA def graph.
//!
//! The classifier answers one question per array-index operand: *as the
//! induction variable advances by 1, by how much does this subscript move?*
//! A subscript is [`Affine`] `{a, b}` when the answer is deterministic;
//! anything else (loads, calls, iteration-varying phis) returns `None` and the
//! dependence battery falls back to a conservative "dependence exists".
//!
//! ## Loop invariants and the symbolic-constant collapse
//!
//! Real kernels index arrays through values computed *outside* the analyzed
//! loop (`ibase = i * N` feeding an inner `k` loop). Such values are constant
//! across the loop's iterations, so for dependence purposes they behave like
//! symbolic constants. Rather than inventing a symbol type, every invariant
//! sub-expression collapses to the constant `0`: two occurrences of the *same*
//! `ValueId` on the source and sink sides still compare equal (which is what
//! distance-0 reasoning needs), while two different invariants spuriously
//! compare equal too. The latter can only ever *manufacture* a suspected
//! dependence, never hide a real one — the conservative direction. Constant
//! literals keep their true values, so `a[i-1]`-style offsets stay exact.

use crate::deps::Affine;
use crate::loops::Loop;
use helix_ir::{Constant, FuncIr, Inst, ValueId};
use std::collections::HashMap;

/// One memory access inside a loop body.
#[derive(Clone, Debug)]
pub struct Access {
    /// The array's local slot.
    pub arr: helix_ir::LocalId,
    /// True for `Inst::Store` (writes escape; reads are side-effect free).
    pub is_write: bool,
    /// `(loop-block ordinal, instruction ordinal)` — program order within the
    /// loop body, used to order dependence pairs and label RAW vs WAR.
    pub site: (u32, u32),
    /// Affine form w.r.t. the induction value, when extractable.
    pub affine: Option<Affine>,
}

/// Collect and classify all array accesses in `loop_`, in program order.
///
/// `iv_value` is the SSA name of the induction variable inside the loop
/// (the header φ result). Indices defined outside the loop are invariant
/// (see the module docs); only their constant-ness matters for the battery.
pub fn collect(func: &FuncIr, loop_: &Loop, iv_value: ValueId) -> Vec<Access> {
    let mut out = Vec::new();
    let mut memo: HashMap<ValueId, Option<Affine>> = HashMap::new();
    for (bi, &blk) in loop_.blocks.iter().enumerate() {
        let bd = func.block(blk);
        for (ii, inst) in bd.insts.iter().enumerate() {
            let (arr, idx, is_write) = match inst {
                Inst::Load(l) => (l.arr, l.idx, false),
                Inst::Store { arr, idx, .. } => (*arr, *idx, true),
                _ => continue,
            };
            let affine = classify(func, loop_, iv_value, idx, &mut memo, 0);
            out.push(Access {
                arr,
                is_write,
                site: (bi as u32, ii as u32),
                affine,
            });
        }
    }
    out.sort_by_key(|a| a.site);
    out
}

/// Try to express `v` as `a*iv + b`.
///
/// Resolution order per value:
/// 1. the induction value itself → `{1, 0}`;
/// 2. an instruction inside the loop → structural recursion
///    (constants/arithmetic only; loads and calls are runtime values);
/// 3. a φ inside the loop → affine only when every incoming argument folds to
///    the *same* coefficient-free form (an if-join of invariants);
/// 4. anything defined outside the loop → invariant, collapsed to `{0, 0}`;
/// 5. no def at all (function parameter) → invariant, `{0, 0}`.
///
/// The recursion depth cap breaks cycles through self-referential φ shapes
/// (the latch increment feeds the header φ it reads).
fn classify(
    func: &FuncIr,
    loop_: &Loop,
    iv: ValueId,
    v: ValueId,
    memo: &mut HashMap<ValueId, Option<Affine>>,
    depth: u32,
) -> Option<Affine> {
    if depth > 64 {
        return None;
    }
    if v == iv {
        return Some(Affine { a: 1, b: 0 });
    }
    if let Some(hit) = memo.get(&v) {
        return *hit;
    }
    // Cycle guard: seed a pessimistic entry BEFORE recursing. The renamer
    // spells `x = x + 1` self-referentially (dst appears as its own operand),
    // so a naive recursion re-enters the same value forever without ever
    // filling the memo; the placeholder turns that cycle into a clean
    // "non-affine" answer instead of a depth-cap failure.
    memo.insert(v, None);

    // Locate the definition, remembering whether it lives inside the loop.
    let mut inside_inst: Option<&Inst> = None;
    let mut inside_phi_args: Option<&[(helix_ir::BlockId, ValueId)]> = None;
    let mut found_inside = false;
    'search: for &blk in &loop_.blocks {
        let bd = func.block(blk);
        if let Some(p) = bd.phis.iter().find(|p| p.dst == v) {
            inside_phi_args = Some(p.args.as_slice());
            found_inside = true;
            break 'search;
        }
        if let Some(inst) = bd.insts.iter().find(|i| i.dst() == Some(v)) {
            inside_inst = Some(inst);
            found_inside = true;
            break 'search;
        }
    }
    if !found_inside {
        // Defined outside the loop (or not at all): loop-invariant by
        // definition. A *literal constant* keeps its value — offsets like
        // `i - 1` and multipliers like `SIZE` must stay numerically exact for
        // the SIV battery. A genuinely symbolic invariant (a load, a call
        // result, a parameter) has no usable numeric identity, so it
        // collapses to the symbolic constant 0: two occurrences of the same
        // ValueId still compare equal, and the collapse can only ever
        // manufacture a suspected dependence, never hide one (module docs).
        let inv = match inside_inst_anywhere(func, v) {
            Some(Inst::Const { c, .. }) => {
                const_i128(c).map_or(Affine { a: 0, b: 0 }, |b| Affine { a: 0, b })
            }
            _ => Affine { a: 0, b: 0 },
        };
        memo.insert(v, Some(inv));
        return Some(inv);
    }

    let result = if let Some(args) = inside_phi_args {
        // A loop-carried φ — some back-edge arm is defined by a computation
        // reading this φ's destination (the renamer's spelling of
        // `x = x + step`, or even a direct self-argument) — carries iteration
        // state of another variable: not affine here. Detect that shape
        // before walking arms so the cycle guard below stays a safety net.
        let self_carried = args.iter().any(|&(_, arm)| arm == v)
            || args.iter().any(|&(_, arm)| {
                matches!(
                    func.inst_defining(arm),
                    Some(Inst::Bin { a, b, .. })
                        if (*a == v || *b == v) && (*a == arm || *b == arm)
                )
            });
        if self_carried {
            return None;
        }
        // Otherwise: affine only when every arm agrees on one invariant form
        // (an if-join of invariants).
        let mut agreed: Option<Affine> = None;
        for (_, arm) in args {
            let a = classify(func, loop_, iv, *arm, memo, depth + 1)?;
            match agreed {
                None => agreed = Some(a),
                Some(f) if f == a => {}
                _ => return None,
            }
        }
        agreed.filter(|f| f.a == 0)
    } else {
        let cx = Ctxt { func, loop_, iv };
        match inside_inst.expect("inside def found above") {
            Inst::Const { c, .. } => const_i128(c).map(|x| Affine { a: 0, b: x }),
            Inst::Bin { op, a, b, .. } => bin_affine(&cx, *op, *a, *b, memo, depth),
            Inst::Unary { op, a, .. } => {
                let inner = cx.classify(*a, memo, depth + 1)?;
                match op {
                    helix_ir::UnOp::Neg => Some(Affine {
                        a: -inner.a,
                        b: -inner.b,
                    }),
                    helix_ir::UnOp::Not => None,
                }
            }
            // Loads, casts and calls produce runtime values with no affine
            // relation to the induction variable.
            _ => None,
        }
    };

    memo.insert(v, result);
    result
}

/// Combine two classified operands through one binary operator.
///
/// The operands arrive pre-classified via [`Ctxt`], keeping the arity lint
/// happy while preserving the memo table across sibling sub-expressions.
struct Ctxt<'f, 'l> {
    func: &'f FuncIr,
    loop_: &'l Loop,
    iv: ValueId,
}

impl Ctxt<'_, '_> {
    fn classify(
        &self,
        v: ValueId,
        memo: &mut HashMap<ValueId, Option<Affine>>,
        depth: u32,
    ) -> Option<Affine> {
        classify(self.func, self.loop_, self.iv, v, memo, depth)
    }
}

fn bin_affine(
    cx: &Ctxt<'_, '_>,
    op: helix_ir::BinOp,
    a: ValueId,
    b: ValueId,
    memo: &mut HashMap<ValueId, Option<Affine>>,
    depth: u32,
) -> Option<Affine> {
    use helix_ir::BinOp as B;
    let l = cx.classify(a, memo, depth + 1)?;
    let r = cx.classify(b, memo, depth + 1)?;
    match (op, l, r) {
        (B::Add, x, y) => Some(Affine {
            a: x.a + y.a,
            b: x.b + y.b,
        }),
        (B::Sub, x, y) => Some(Affine {
            a: x.a - y.a,
            b: x.b - y.b,
        }),
        // Multiplication is affine only when at most one factor varies with
        // the induction variable and that factor's coefficient is constant;
        // checked arithmetic guards pathological coefficient products.
        (B::Mul, x, y) if x.a == 0 && y.a == 0 => Some(Affine { a: 0, b: x.b * y.b }),
        (B::Mul, x, y) if x.a == 0 && x.b.checked_mul(y.a).is_some() => Some(Affine {
            a: x.b * y.a,
            b: x.b * y.b,
        }),
        (B::Mul, x, y) if y.a == 0 && y.b.checked_mul(x.a).is_some() => Some(Affine {
            a: y.b * x.a,
            b: y.b * x.b,
        }),
        _ => None,
    }
}

fn const_i128(c: &Constant) -> Option<i128> {
    match c {
        Constant::I64(x) => Some(i128::from(*x)),
        Constant::I32(x) => Some(i128::from(*x)),
        Constant::F32(_) | Constant::F64(_) | Constant::Bool(_) => None,
    }
}

/// Whole-function definition lookup for values outside the analyzed loop —
/// the fallback that keeps *constants* numerically exact under the invariant
/// collapse (see `classify`).
fn inside_inst_anywhere(func: &FuncIr, v: ValueId) -> Option<&Inst> {
    func.inst_defining(v)
}
