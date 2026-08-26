//! Constant folding: evaluate `Bin`/`Unary`/`Cast` ops whose operands are all
//! `Inst::Const` defs.
//!
//! Rust's operator semantics are deliberately reused as the HELIX semantics
//! (lang-spec: "use Rust semantics == interpreter semantics"):
//!
//! * integer `+ - *`: **wrapping** two's-complement (documented in the spec),
//!   so the folder uses `wrapping_add` etc.; a debug-build panic on the same
//!   input that release builds silently wrap would be a semantic divergence;
//! * `/ %`: folded only when the divisor is non-zero and not `i64::MIN / -1` —
//!   those trap at runtime and folding a trap away would change behavior;
//! * `%` is the truncated remainder with sign of dividend (`-7 % 2 == -1`),
//!   which coincides with Rust's `%`;
//! * float arithmetic follows IEEE-754 as Rust does (`/ 0.0` ⇒ Inf/NaN,
//!   never traps);
//! * casts follow the frozen rules: float→int **saturates** (NaN→0, clamp to
//!   MIN/MAX), int→int truncating reinterpretation of two's complement,
//!   int↔float rounding toward zero.
//!
//! The pass scans every block to a fixpoint so constant *chains* (`1 + 2 *
//! 3`) collapse in a single call.

use helix_sema::Ty;
use helix_syntax::ast::{BinOp, UnOp};

use crate::ir::{Constant, FuncIr, Inst};
use crate::passmod::ChangeFlag;

/// Fold every foldable instruction in place; returns whether anything changed.
pub fn const_fold(ir: &mut FuncIr) -> ChangeFlag {
    let mut flag = ChangeFlag::new();

    // Snapshot the constant table ONCE, before any block is drained — the
    // per-block loop below takes instruction lists out of the function while
    // it works, so lookups must not depend on live block contents. Unique
    // defs post-SSA make one id -> payload entry unambiguous.
    let mut konsts: std::collections::HashMap<u32, Constant> = std::collections::HashMap::new();
    for b in &ir.blocks {
        for inst in &b.insts {
            if let Inst::Const { dst, c } = inst {
                konsts.entry(dst.0).or_insert(*c);
            }
        }
    }

    for bi in 0..ir.blocks.len() {
        loop {
            let mut changed = false;
            let drained = std::mem::take(&mut ir.blocks[bi].insts);
            let mut keep: Vec<Inst> = Vec::with_capacity(drained.len());
            for inst in drained.iter() {
                match try_fold(&konsts, inst) {
                    Some((dst, c)) => {
                        konsts.insert(dst.0, c); // folded values feed later rounds
                        keep.push(Inst::Const { dst, c });
                        changed = true;
                    }
                    None => keep.push(inst.clone()),
                }
            }
            ir.blocks[bi].insts = keep;
            if !changed {
                break;
            }
            flag.changed = true;
        }
    }
    flag
}

/// If `inst` folds, produce `(dst, value)` using the precomputed table.
fn try_fold(
    konsts: &std::collections::HashMap<u32, Constant>,
    inst: &Inst,
) -> Option<(crate::ir::ValueId, Constant)> {
    let konst = |v: crate::ir::ValueId| -> Option<Constant> { konsts.get(&v.0).copied() };

    match inst {
        Inst::Bin { op, dst, a, b } => {
            let (x, y) = (konst(*a)?, konst(*b)?);
            fold_bin(*op, x, y).map(|c| (*dst, c))
        }
        Inst::Unary { op, dst, a } => {
            let x = konst(*a)?;
            Some((*dst, fold_unary(*op, x)?))
        }
        Inst::Cast { dst, val, to } => {
            let x = konst(*val)?;
            Some((*dst, fold_cast(x, *to)?))
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Operator evaluation
// ---------------------------------------------------------------------------

/// Fold one binary op over two constants.
#[must_use]
pub fn fold_bin(op: BinOp, x: Constant, y: Constant) -> Option<Constant> {
    use BinOp::*;
    // Bool ops first.
    if let (Constant::Bool(a), Constant::Bool(b)) = (&x, &y) {
        return match op {
            Eq => Some(Constant::Bool(a == b)),
            Ne => Some(Constant::Bool(a != b)),
            _ => None, // && / || never reach here as plain Bins
        };
    }

    match (x.as_num()?, y.as_num()?) {
        (crate::ir::Num::I(a), crate::ir::Num::I(b)) => fold_bin_int(op, a, b),
        (crate::ir::Num::F32(a), crate::ir::Num::F32(b)) => fold_bin_f32(op, a, b),
        (crate::ir::Num::F64(a), crate::ir::Num::F64(b)) => fold_bin_f64(op, a, b),
        _ => None, // mixed int/float requires an explicit cast in HELIX
    }
}

fn fold_bin_int(op: BinOp, a: i64, b: i64) -> Option<Constant> {
    use BinOp::*;
    let r = match op {
        Add => Constant::I64(a.wrapping_add(b)),
        Sub => Constant::I64(a.wrapping_sub(b)),
        Mul => Constant::I64(a.wrapping_mul(b)),
        Div => {
            if b == 0 || (a == i64::MIN && b == -1) {
                return None; // runtime trap stays a runtime trap
            }
            Constant::I64(a / b)
        }
        Rem => {
            if b == 0 || (a == i64::MIN && b == -1) {
                return None;
            }
            Constant::I64(a % b)
        }
        Lt => Constant::Bool(a < b),
        Gt => Constant::Bool(a > b),
        Le => Constant::Bool(a <= b),
        Ge => Constant::Bool(a >= b),
        Eq => Constant::Bool(a == b),
        Ne => Constant::Bool(a != b),
        And | Or => return None,
    };
    Some(r)
}

fn fold_bin_f64(op: BinOp, a: f64, b: f64) -> Option<Constant> {
    use BinOp::*;
    let r = match op {
        Add => Constant::F64(a + b),
        Sub => Constant::F64(a - b),
        Mul => Constant::F64(a * b),
        Div => Constant::F64(a / b), // IEEE: no traps
        Rem => Constant::F64(a % b),
        Lt => Constant::Bool(a < b),
        Gt => Constant::Bool(a > b),
        Le => Constant::Bool(a <= b),
        Ge => Constant::Bool(a >= b),
        Eq => Constant::Bool(a == b),
        Ne => Constant::Bool(a != b),
        And | Or => return None,
    };
    Some(r)
}

fn fold_bin_f32(op: BinOp, a: f32, b: f32) -> Option<Constant> {
    use BinOp::*;
    let r = match op {
        Add => Constant::F32(a + b),
        Sub => Constant::F32(a - b),
        Mul => Constant::F32(a * b),
        Div => Constant::F32(a / b), // IEEE: no traps
        Rem => Constant::F32(a % b),
        Lt => Constant::Bool(a < b),
        Gt => Constant::Bool(a > b),
        Le => Constant::Bool(a <= b),
        Ge => Constant::Bool(a >= b),
        Eq => Constant::Bool(a == b),
        Ne => Constant::Bool(a != b),
        And | Or => return None,
    };
    Some(r)
}

/// Fold one unary op.
#[must_use]
pub fn fold_unary(op: UnOp, x: Constant) -> Option<Constant> {
    match (op, &x) {
        (UnOp::Not, Constant::Bool(b)) => Some(Constant::Bool(!b)),
        (UnOp::Neg, Constant::I64(v)) => Some(Constant::I64(v.wrapping_neg())),
        (UnOp::Neg, Constant::I32(v)) => Some(Constant::I32(v.wrapping_neg())),
        (UnOp::Neg, Constant::F32(v)) => Some(Constant::F32(-v)),
        (UnOp::Neg, Constant::F64(v)) => Some(Constant::F64(-v)),
        _ => None,
    }
}

/// Fold one cast under the frozen conversion rules.
#[must_use]
pub fn fold_cast(x: Constant, to: Ty) -> Option<Constant> {
    Some(match (x, to) {
        (c, Ty::I64) => Constant::I64(match c {
            Constant::I64(v) => v,
            Constant::I32(v) => v as i64,
            Constant::F64(f) => saturate_i64(f),
            Constant::F32(f) => saturate_i64(f as f64),
            Constant::Bool(_) => return None,
        }),
        (c, Ty::I32) => Constant::I32(match c {
            Constant::I64(v) => v as i32,
            Constant::I32(v) => v,
            Constant::F64(f) => saturate_i32(f),
            Constant::F32(f) => saturate_i32(f as f64),
            Constant::Bool(_) => return None,
        }),
        (c, Ty::F64) => Constant::F64(match c {
            Constant::I64(v) => v as f64,
            Constant::I32(v) => v as f64,
            Constant::F64(f) => f,
            Constant::F32(f) => f as f64,
            Constant::Bool(_) => return None,
        }),
        (c, Ty::F32) => Constant::F32(match c {
            Constant::I64(v) => v as f32,
            Constant::I32(v) => v as f32,
            Constant::F64(f) => f as f32,
            Constant::F32(f) => f,
            Constant::Bool(_) => return None,
        }),
        _ => return None,
    })
}

/// float → i64 saturation (NaN→0).
#[must_use]
pub fn saturate_i64(f: f64) -> i64 {
    if f.is_nan() {
        0
    } else if f >= i64::MAX as f64 {
        i64::MAX
    } else if f <= i64::MIN as f64 {
        i64::MIN
    } else {
        f as i64
    }
}

/// float → i32 saturation (NaN→0).
#[must_use]
pub fn saturate_i32(f: f64) -> i32 {
    if f.is_nan() {
        0
    } else if f >= i32::MAX as f64 {
        i32::MAX
    } else if f <= i32::MIN as f64 {
        i32::MIN
    } else {
        f as i32
    }
}
