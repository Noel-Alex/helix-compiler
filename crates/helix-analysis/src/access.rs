//! Affine subscript extraction: express each Load/Store index as `a*i + b`
//! relative to a loop's induction variable, walking the SSA def graph.
//!
//! The classifier answers one question per array-index operand: *as the
//! induction variable advances by 1, by how much does this subscript move?*
//! A subscript is [`Affine`] `{a, b}` when the answer is deterministic;
//! anything else (loads, calls, iteration-varying phis) returns `None` and the
//! dependence battery falls back to a conservative "dependence exists".
//!
//! ## Loop invariants
//!
//! Integer literals defined outside the loop keep their exact values. Other
//! outside definitions and parameters have coefficient zero but unknown offset.
//! This can prove a sole write injective, not a read/write dependence distance.
//! ponytail: symbolic offsets stay unknown until symbolic affine forms exist.

use crate::Bound;
use crate::canon;
use crate::deps::{Affine, IterRange};
use crate::loops::Loop;
use helix_ir::{Constant, FuncIr, Inst, ValueId};
use helix_sema::Ty;
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
    /// Affine form with an exact offset, for dependence-distance testing.
    pub affine: Option<Affine>,
    /// IV coefficient only when injectivity is proven: no-wrap arithmetic,
    /// or an odd coefficient in a single machine-width modular affine form.
    /// The latter covers translations by unknown invariant offsets.
    pub iv_coefficient: Option<i128>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Form {
    a: i128,
    b: Option<i128>,
    /// Every intermediate operation agrees with mathematical integer arithmetic.
    exact: bool,
}

/// Collect and classify all array accesses in `loop_`, in program order.
///
/// `iv_value` is the SSA name of the induction variable inside the loop
/// (the header φ result). Outside-loop integer literals stay exact; other
/// invariant values remain unknown (see the module docs).
pub fn collect(func: &FuncIr, loop_: &Loop, iv_value: ValueId) -> Vec<Access> {
    let mut out = Vec::new();
    let mut memo: HashMap<ValueId, Option<Form>> = HashMap::new();
    let range = iv_range(func, loop_, iv_value);
    for (bi, &blk) in loop_.blocks.iter().enumerate() {
        let bd = func.block(blk);
        for (ii, inst) in bd.insts.iter().enumerate() {
            let (arr, idx, is_write) = match inst {
                Inst::Load(l) => (l.arr, l.idx, false),
                Inst::Store { arr, idx, .. } => (*arr, *idx, true),
                _ => continue,
            };
            let cx = Ctxt {
                func,
                loop_,
                iv: iv_value,
                range,
            };
            let form = cx.classify(idx, &mut memo, 0);
            out.push(Access {
                arr,
                is_write,
                site: (bi as u32, ii as u32),
                affine: form
                    .filter(|f| f.exact)
                    .and_then(|f| f.b.map(|b| Affine { a: f.a, b })),
                // Odd multiplication is a permutation modulo 2^width. Even
                // strides need a no-wrap proof; a nonzero integer coefficient
                // alone says nothing about collisions in machine arithmetic.
                iv_coefficient: form.filter(|f| f.exact || f.a % 2 != 0).map(|f| f.a),
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
/// 4. outside-loop integer literal → `{0, value}`;
/// 5. any other outside definition or function parameter → `{0, ?}`.
///
/// The recursion depth cap breaks cycles through self-referential φ shapes
/// (the latch increment feeds the header φ it reads).
fn classify(
    cx: &Ctxt<'_, '_>,
    v: ValueId,
    memo: &mut HashMap<ValueId, Option<Form>>,
    depth: u32,
) -> Option<Form> {
    let Ctxt {
        func, loop_, iv, ..
    } = *cx;
    if depth > 64 {
        return None;
    }
    if v == iv {
        return Some(Form {
            a: 1,
            b: Some(0),
            exact: true,
        });
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
        // Invariance proves coefficient zero, but only literals give an
        // exact offset. Never feed an unknown offset to distance testing.
        let inv = Some(Form {
            a: 0,
            exact: true,
            b: match inside_inst_anywhere(func, v) {
                Some(Inst::Const { c, .. }) => const_i128(c),
                _ => None,
            },
        });
        memo.insert(v, inv);
        return inv;
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
        let mut agreed: Option<Form> = None;
        for (_, arm) in args {
            let a = cx.classify(*arm, memo, depth + 1)?;
            match agreed {
                None => agreed = Some(a),
                Some(f) if f == a => {}
                _ => return None,
            }
        }
        agreed.filter(|f| f.a == 0 && f.b.is_some() && f.exact)
    } else {
        match inside_inst.expect("inside def found above") {
            Inst::Const { c, .. } => const_i128(c).map(|x| Form {
                a: 0,
                b: Some(x),
                exact: true,
            }),
            Inst::Bin { op, a, b, .. }
                if func.val_ty(v) == func.val_ty(*a) && func.val_ty(*a) == func.val_ty(*b) =>
            {
                bin_affine(cx, *op, *a, *b, memo, depth)
            }
            Inst::Unary { op, a, .. } => {
                let inner = cx.classify(*a, memo, depth + 1)?;
                // Do not carry a modular form through a width change.
                if func.val_ty(v) != func.val_ty(*a) {
                    return None;
                }
                match op {
                    helix_ir::UnOp::Neg => Some(Form {
                        a: inner.a.checked_neg()?,
                        b: inner.b.and_then(i128::checked_neg),
                        exact: inner.exact,
                    }),
                    helix_ir::UnOp::Not => None,
                }
            }
            // A same-type cast is the copy identity build.rs emits for plain
            // variable copies (`let j = i`); it is transparent. Other casts
            // and calls/loads produce runtime values with no affine relation
            // to the induction variable.
            Inst::Cast { val, to, .. } if Some(*to) == func.types.val_ty(*val) => {
                cx.classify(*val, memo, depth + 1)
            }
            _ => None,
        }
    };

    let result = result.and_then(|mut f| {
        let limits = integer_range(func.val_ty(v)?)?;
        // The same-width modular form is useful for sole-write injectivity,
        // but NEVER for mathematical distance/divisibility/range proofs.
        f.exact &= f.b.is_some_and(|b| {
            [cx.range.lo, cx.range.hi].into_iter().all(|i| {
                f.a.checked_mul(i)
                    .and_then(|x| x.checked_add(b))
                    .is_some_and(|x| limits.lo <= x && x <= limits.hi)
            })
        });
        Some(f)
    });
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
    range: IterRange,
}

impl Ctxt<'_, '_> {
    fn classify(
        &self,
        v: ValueId,
        memo: &mut HashMap<ValueId, Option<Form>>,
        depth: u32,
    ) -> Option<Form> {
        classify(self, v, memo, depth)
    }
}

fn bin_affine(
    cx: &Ctxt<'_, '_>,
    op: helix_ir::BinOp,
    a: ValueId,
    b: ValueId,
    memo: &mut HashMap<ValueId, Option<Form>>,
    depth: u32,
) -> Option<Form> {
    use helix_ir::BinOp as B;
    let l = cx.classify(a, memo, depth + 1)?;
    let r = cx.classify(b, memo, depth + 1)?;
    let exact = l.exact && r.exact;
    match (op, l, r) {
        (B::Add, x, y) => Some(Form {
            exact,
            a: x.a.checked_add(y.a)?,
            b: x.b.zip(y.b).and_then(|(x, y)| x.checked_add(y)),
        }),
        (B::Sub, x, y) => Some(Form {
            exact,
            a: x.a.checked_sub(y.a)?,
            b: x.b.zip(y.b).and_then(|(x, y)| x.checked_sub(y)),
        }),
        (B::Mul, x, y) if x.a == 0 && y.a == 0 => Some(Form {
            exact,
            a: 0,
            b: x.b.zip(y.b).and_then(|(x, y)| x.checked_mul(y)),
        }),
        // A symbolic multiplier could be zero: require an exact constant
        // factor when the other operand varies with the induction variable.
        (B::Mul, x, y) if x.a == 0 => Some(Form {
            exact,
            a: x.b?.checked_mul(y.a)?,
            b: x.b.zip(y.b).and_then(|(x, y)| x.checked_mul(y)),
        }),
        (B::Mul, x, y) if y.a == 0 => Some(Form {
            exact,
            a: y.b?.checked_mul(x.a)?,
            b: x.b.zip(y.b).and_then(|(x, y)| x.checked_mul(y)),
        }),
        _ => None,
    }
}

fn integer_range(ty: Ty) -> Option<IterRange> {
    let (lo, hi) = match ty {
        Ty::I64 => (i128::from(i64::MIN), i128::from(i64::MAX)),
        Ty::I32 => (i128::from(i32::MIN), i128::from(i32::MAX)),
        _ => return None,
    };
    Some(IterRange { lo, hi })
}

/// Overapproximate all executed IV values, using each known bound separately.
/// Unlike dependence refutation, no-wrap checking needs only an enclosure.
fn iv_range(func: &FuncIr, lp: &Loop, iv: ValueId) -> IterRange {
    let mut range = integer_range(func.val_ty(iv).unwrap_or(Ty::I64)).unwrap_or(IterRange {
        lo: i128::from(i64::MIN),
        hi: i128::from(i64::MAX),
    });
    if let Some(c) = canon::canon(func, lp)
        && c.iv_value_in_loop == iv
    {
        // Canonical for loops execute only while iv < end. Even a symbolic
        // end cannot exceed the signed type maximum.
        range.hi -= 1;
        let start_lo = match c.start {
            Bound::Const(lo) => i128::from(lo),
            Bound::Sym(v) => bound_range(func, ValueId(v), 0).lo,
        };
        range.lo = range.lo.max(start_lo);
        if let Bound::Const(hi) = c.end {
            range.hi = range.hi.min(i128::from(hi) - 1);
        }
    }
    range
}

/// Casts stay symbolic in canonical bounds. A widening cast can still give
/// no-wrap checking a safe enclosure without inventing an exact trip window.
fn bound_range(func: &FuncIr, v: ValueId, depth: u32) -> IterRange {
    let limits = integer_range(func.val_ty(v).unwrap_or(Ty::I64)).unwrap_or(IterRange {
        lo: i128::from(i64::MIN),
        hi: i128::from(i64::MAX),
    });
    if depth < 64
        && let Some(Inst::Cast { val, to, .. }) = func.inst_defining(v)
        && let (Some(from), Some(to)) = (
            func.val_ty(*val).and_then(integer_range),
            integer_range(*to),
        )
        && to.lo <= from.lo
        && from.hi <= to.hi
    {
        return bound_range(func, *val, depth + 1);
    }
    limits
}

fn const_i128(c: &Constant) -> Option<i128> {
    match c {
        Constant::I64(x) => Some(i128::from(*x)),
        Constant::I32(x) => Some(i128::from(*x)),
        Constant::F32(_) | Constant::F64(_) | Constant::Bool(_) => None,
    }
}

/// Whole-function definition lookup for integer literals outside the loop.
fn inside_inst_anywhere(func: &FuncIr, v: ValueId) -> Option<&Inst> {
    func.inst_defining(v)
}
