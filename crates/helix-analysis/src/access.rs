//! Affine subscript extraction: express each Load/Store index as `a*i + b`
//! relative to a loop's induction variable, walking the SSA def graph.

use crate::deps::Affine;
use crate::loops::Loop;
use helix_ir::{Constant, FuncIr, Inst, ValueId};
use std::collections::HashMap;

/// One memory access inside a loop body.
#[derive(Clone, Debug)]
pub struct Access {
    /// The array's local slot.
    pub arr: helix_ir::LocalId,
    pub is_write: bool,
    /// Block + instruction position (stable reporting order).
    pub site: (u32, u32),
    /// Affine form w.r.t. the induction value, when extractable.
    pub affine: Option<Affine>,
}

/// Collect and classify all array accesses in `loop_`.
///
/// `iv_value` is the SSA name of the induction variable inside the loop
/// (the header φ result). Indices defined outside the loop are invariant
/// (affine with a=0); only their constant-ness matters for the battery.
pub fn collect(func: &FuncIr, loop_: &Loop, iv_value: ValueId) -> Vec<Access> {
    let mut out = Vec::new();
    for (bi, &blk) in loop_.blocks.iter().enumerate() {
        let bd = func.block(blk);
        for (ii, inst) in bd.insts.iter().enumerate() {
            let (arr, idx, is_write) = match inst {
                Inst::Load(l) => (l.arr, l.idx, false),
                Inst::Store { arr, idx, .. } => (*arr, *idx, true),
                _ => continue,
            };
            let mut memo = HashMap::new();
            let affine = classify_rec(func, loop_, iv_value, idx, &mut memo, 0);
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

/// Try to express `v` as `a*iv + b`. Returns None when non-affine
/// (multiplications by non-constants, unknown ops, values from enclosing loops).
fn classify_rec(
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
    if let Some(hit) = memo.get(&v) {
        return *hit;
    }

    let result = if v == iv {
        Some(Affine { a: 1, b: 0 })
    } else {
        // Find the definition site of v.
        let mut def = None;
        for &blk in &loop_.blocks {
            let bd = func.block(blk);
            if let Some(p) = bd.phis.iter().find(|p| p.dst == v) {
                // A phi whose back-edge arg equals its entry arg is still affine;
                // otherwise it carries iteration state — not affine.
                def = phi_affine(func, p.args.as_slice(), loop_);
                break;
            }
            if let Some(inst) = bd.insts.iter().find(|i| i.dst() == Some(v)) {
                def = match inst {
                    Inst::Const { c, .. } => const_i128(c).map(|x| Affine { a: 0, b: x }),
                    Inst::Bin { op, a, b, .. } => bin_affine(func, loop_, iv, *op, *a, *b, memo, depth),
                    Inst::Unary { op, a, .. } => {
                        let inner = classify_rec(func, loop_, iv, *a, memo, depth + 1)?;
                        match op {
                            helix_ir::UnOp::Neg => Some(Affine { a: -inner.a, b: -inner.b }),
                            helix_ir::UnOp::Not => None,
                        }
                    }
                    // Loads/calls/casts produce runtime values — not affine.
                    _ => None,
                };
                break;
            }
        }
        def
    };

    memo.insert(v, result);
    result
}

/// A φ is affine when both incoming args classify to the SAME affine form
/// (typical for phis merging identical computations along two paths).
fn phi_affine(
    func: &FuncIr,
    args: &[(helix_ir::BlockId, ValueId)],
    loop_: &Loop,
) -> Option<Affine> {
    let mut first: Option<Affine> = None;
    for (_, v) in args {
        let mut memo = HashMap::new();
        let a = classify_rec(func, loop_, ValueId(u32::MAX), *v, &mut memo, 0);
        match (&first, a) {
            (None, x) => first = x.filter(|f| f.a == 0), // no iv dependency through phis
            (Some(f), Some(x)) if f.a == x.a && f.b == x.b => {}
            _ => return None,
        }
    }
    first
}

fn bin_affine(
    func: &FuncIr,
    loop_: &Loop,
    iv: ValueId,
    op: helix_ir::BinOp,
    a: ValueId,
    b: ValueId,
    memo: &mut HashMap<ValueId, Option<Affine>>,
    depth: u32,
) -> Option<Affine> {
    use helix_ir::BinOp as B;
    let l = classify_rec(func, loop_, iv, a, memo, depth + 1);
    let r = classify_rec(func, loop_, iv, b, memo, depth + 1);
    match (op, l?, r?) {
        (B::Add, x, y) => Some(Affine { a: x.a + y.a, b: x.b + y.b }),
        (B::Sub, x, y) => Some(Affine { a: x.a - y.a, b: x.b - y.b }),
        (B::Mul, x, y) if x.a == 0 && y.a == 0 => Some(Affine { a: 0, b: x.b * y.b }),
        (B::Mul, x, y) if x.a == 0 && y.a != 0 => Some(Affine { a: x.b * y.a, b: x.b * y.b }),
        (B::Mul, x, y) if x.a != 0 && y.a == 0 => Some(Affine { a: x.a * y.b, b: x.b * y.b }),
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
