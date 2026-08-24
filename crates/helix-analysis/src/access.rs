//! Affine subscript extraction: recover `a*i + b` (per loop level) from the SSA
//! value graph feeding each Load/Store address index.

use crate::deps::Affine;
use crate::loops::Loop;
use helix_ir::{FuncIr, Inst, ValueId};
use std::collections::HashMap;

/// What we know about one memory access inside a loop.
#[derive(Clone, Debug)]
pub struct Access {
    /// The array local being accessed.
    pub arr: helix_ir::LocalId,
    /// Read or write.
    pub is_write: bool,
    /// The instruction's position id for stable reporting.
    pub site: u32,
    /// Affine form of the index w.r.t. the analyzed loop, when extractable.
    pub affine: Option<Affine>,
    /// Raw index expression text for the report (e.g. "base + j").
    pub raw_index: String,
}

/// Try to express `v` as a*iv + b within `loop_`, where iv is the induction
/// variable's SSA name in the loop header.
pub fn classify(
    func: &FuncIr,
    loop_: &Loop,
    iv_name: ValueId,
    v: ValueId,
) -> (Option<Affine>, String) {
    let mut memo: HashMap<ValueId, Option<Affine>> = HashMap::new();
    let aff = classify_rec(func, loop_, iv_name, v, &mut memo, 0);
    (aff, render_raw(func, v))
}

fn classify_rec(
    func: &FuncIr,
    loop_: &Loop,
    iv: ValueId,
    v: ValueId,
    memo: &mut HashMap<ValueId, Option<Affine>>,
    depth: u32,
) -> Option<Affine> {
    if depth > 32 {
        return None; // pathological chains
    }
    if let Some(hit) = memo.get(&v) {
        return *hit;
    }

    // The induction variable itself.
    if v == iv {
        memo.insert(v, Some(Affine { a: 1, b: 0 }));
        return Some(Affine { a: 1, b: 0 });
    }

    let def_site = func.def_block(v);
    // Anything defined outside the loop is loop-invariant => constant offset a=0.
    let outside = !loop_.blocks.contains(&def_site);

    let result = if outside {
        match func.const_of(v) {
            Some(c) => Some(Affine { a: 0, b: c as i128 }),
            None => None, // symbolic invariant — not affine-constant, but still invariant
        }
    } else {
        match func.inst_defining(v) {
            Some(Inst::Const { c, .. }) => const_as_i64(c).map(|x| Affine { a: 0, b: x as i128 }),
            Some(Inst::Bin { op, a, b, .. }) => {
                use helix_ir::BinOp as B;
                let (l, r) = (
                    classify_rec(func, loop_, iv, *a, memo, depth + 1),
                    classify_rec(func, loop_, iv, *b, memo, depth + 1),
                );
                match (op, l, r) {
                    (B::Add, Some(x), Some(y)) => Some(Affine { a: x.a + y.a, b: x.b + y.b }),
                    (B::Sub, Some(x), Some(y)) => Some(Affine { a: x.a - y.a, b: x.b - y.b }),
                    // c*i where i has a==0 (i.e. scalar * invariant) or invariant * linear:
                    (B::Mul, Some(x), Some(y)) if x.a == 0 && y.a == 0 => {
                        Some(Affine { a: 0, b: x.b * y.b })
                    }
                    (B::Mul, Some(x), Some(y)) if x.a == 0 && y.a != 0 => {
                        Some(Affine { a: x.b * y.a, b: x.b * y.b })
                    }
                    (B::Mul, Some(x), Some(y)) if x.a != 0 && y.a == 0 => {
                        Some(Affine { a: x.a * y.b, b: x.b * y.b })
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    };

    memo.insert(v, result.clone());
    result
}

fn const_as_i64(c: &helix_ir::Constant) -> Option<i64> {
    match c {
        helix_ir::Constant::I64(x) | helix_ir::Constant::I32(x) => Some(*x as i64),
        _ => None,
    }
}

fn render_raw(_func: &FuncIr, _v: ValueId) -> String {
    // The IR printer renders full functions; per-value rendering is added there
    // via FuncIr::render_value. Placeholder keeps signature stable until wired.
    String::new()
}
