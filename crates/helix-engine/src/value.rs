//! Dynamic values manipulated by the reference interpreter.
//!
//! HELIX is statically typed — [`helix_sema`] has already proven that every
//! slot holds exactly one type — so the interpreter does not need a tagged
//! union with coercion logic. It needs something simpler: a *faithful
//! container*. [`Value`] mirrors [`helix_sema::Ty`] one-to-one, and every
//! operation in [`crate::interp`] dispatches on the variant that the static
//! type guarantees, treating any mismatch as an internal bug rather than a
//! coercible situation.
//!
//! Two shapes deserve explanation:
//!
//! * **Arrays are shared handles** ([`Value::Array`] =
//!   `Rc<RefCell<Vec<Value>>>`). The spec says arrays are fat pointers and
//!   assignment/passing is *by reference* — callee writes must escape. `Rc`
//!   gives the sharing (caller and callee see the same buffer),
//!   `RefCell` gives interior mutation without `unsafe`, keeping the crate
//!   `#![forbid(unsafe_code)]`-clean. Elements are scalars only, so cloning a
//!   `Value` out of an array is always cheap.
//! * **`Unit`** is the value of procedures and of `print`. It exists so
//!   statement execution has a uniform `Result<Value, _>` shape; it can never
//!   be stored into a variable (sema rejects unit bindings).

use std::cell::RefCell;
use std::rc::Rc;

use helix_sema::{ConstLit, ElemTy, Ty, fmt_bool, fmt_f32, fmt_f64, fmt_i64};

/// The shared, mutable backing buffer of every HELIX array.
///
/// One `Rc` per `zeros` allocation; cloned freely into callees so writes are
/// visible everywhere the array was passed (spec: "assignment/passing is BY
/// REFERENCE; callee writes ESCAPE").
pub type ArrayHandle = Rc<RefCell<Vec<Value>>>;

/// A HELIX runtime value: one variant per static [`Ty`].
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// The sole inhabitant of `()`. Produced by procedures and `print`.
    Unit,
    /// A 32-bit signed integer (storage type).
    I32(i32),
    /// A 64-bit signed integer — the arithmetic integer of the language.
    I64(i64),
    /// Single-precision float.
    F32(f32),
    /// Double-precision float.
    F64(f64),
    /// Boolean; no truthiness conversions ever apply.
    Bool(bool),
    /// Shared array buffer. Never nested (element types are scalars only).
    Array(ArrayHandle),
}

impl Value {
    /// The zero element of an array of element type `elem` (`zeros(n)` fills
    /// with this). `0.0` is positive zero, matching IEEE and the JIT backend.
    #[must_use]
    pub fn zero(elem: ElemTy) -> Value {
        match elem {
            ElemTy::I32 => Value::I32(0),
            ElemTy::I64 => Value::I64(0),
            ElemTy::F32 => Value::F32(0.0),
            ElemTy::F64 => Value::F64(0.0),
            ElemTy::Bool => Value::Bool(false),
        }
    }

    /// Materialises a top-level constant. `ConstLit::Int` stores the raw
    /// literal; the declared type decides whether it is an `i32` or `i64`
    /// (sema has already range-checked `i32` literals).
    #[must_use]
    pub fn from_const(lit: &ConstLit, ty: Ty) -> Value {
        match lit {
            ConstLit::Int(v) => {
                if ty == Ty::I32 {
                    Value::I32(*v as i32)
                } else {
                    Value::I64(*v)
                }
            }
            ConstLit::Float(v) => {
                if ty == Ty::F32 {
                    Value::F32(*v as f32)
                } else {
                    Value::F64(*v)
                }
            }
            ConstLit::Bool(b) => Value::Bool(*b),
        }
    }

    /// Static type name of this value's variant, for internal error messages.
    #[must_use]
    pub fn ty_name(&self) -> &'static str {
        match self {
            Value::Unit => "()",
            Value::I32(_) => "i32",
            Value::I64(_) => "i64",
            Value::F32(_) => "f32",
            Value::F64(_) => "f64",
            Value::Bool(_) => "bool",
            Value::Array(_) => "[T]",
        }
    }

    /// Widening view of any integer value (`i32` widens, `i64` is itself).
    ///
    /// Used for array indices and loop/range bounds — exactly the two places
    /// the spec lets an `i32` value flow into an `i64` position.
    #[must_use]
    pub fn as_i64_widen(&self) -> Option<i64> {
        match self {
            Value::I32(v) => Some(i64::from(*v)),
            Value::I64(v) => Some(*v),
            _ => None,
        }
    }

    /// Renders the value exactly as `print` would, using the canonical
    /// formatter from [`helix_sema::fmt`] so the interpreter and the JIT emit
    /// byte-identical output (differential-testing requirement).
    ///
    /// Integer printing widens to `i64` first; `f32` prints *as* `f32`
    /// (never widened, or `0.1f32` would grow spurious digits).
    #[must_use]
    pub fn render(&self) -> String {
        match self {
            Value::I32(v) => helix_sema::fmt_i64(i64::from(*v)),
            Value::I64(v) => fmt_i64(*v),
            Value::F32(v) => fmt_f32(*v),
            Value::F64(v) => fmt_f64(*v),
            Value::Bool(b) => fmt_bool(*b),
            Value::Unit | Value::Array(_) => {
                format!("<internal: printing a {} value>", self.ty_name())
            }
        }
    }

    /// Feeds this value's canonical byte encoding into an FNV-1a state.
    ///
    /// Encoding choices, made for cross-run (and eventually cross-backend)
    /// checksum stability:
    /// * integers: little-endian two's complement of the fixed width;
    /// * floats: `to_bits()` little-endian, so every NaN payload collapses to
    ///   one canonical bit pattern;
    /// * bool: a single `0`/`1` byte.
    pub fn hash_bits_into(&self, h: &mut u64) {
        let feed = |h: &mut u64, bytes: &[u8]| {
            for &b in bytes {
                *h ^= u64::from(b);
                *h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        };
        match self {
            Value::Unit => {}
            Value::I32(v) => feed(h, &v.to_le_bytes()),
            Value::I64(v) => feed(h, &v.to_le_bytes()),
            Value::F32(v) => feed(h, &v.to_bits().to_le_bytes()),
            Value::F64(v) => feed(h, &v.to_bits().to_le_bytes()),
            Value::Bool(b) => feed(h, &[*b as u8]),
            Value::Array(a) => {
                for v in a.borrow().iter() {
                    v.hash_bits_into(h);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_matches_element_type() {
        assert_eq!(Value::zero(ElemTy::I32), Value::I32(0));
        assert_eq!(Value::zero(ElemTy::I64), Value::I64(0));
        assert_eq!(Value::zero(ElemTy::F32), Value::F32(0.0));
        assert_eq!(Value::zero(ElemTy::F64), Value::F64(0.0));
        assert_eq!(Value::zero(ElemTy::Bool), Value::Bool(false));
    }

    #[test]
    fn consts_respect_declared_width() {
        assert_eq!(Value::from_const(&ConstLit::Int(7), Ty::I32), Value::I32(7));
        assert_eq!(Value::from_const(&ConstLit::Int(7), Ty::I64), Value::I64(7));
        assert_eq!(
            Value::from_const(&ConstLit::Float(0.5), Ty::F32),
            Value::F32(0.5)
        );
        assert_eq!(
            Value::from_const(&ConstLit::Float(0.5), Ty::F64),
            Value::F64(0.5)
        );
        assert_eq!(
            Value::from_const(&ConstLit::Bool(true), Ty::Bool),
            Value::Bool(true)
        );
    }

    #[test]
    fn render_uses_canonical_formatter() {
        assert_eq!(Value::I64(-12).render(), "-12");
        // i32 prints widened, indistinguishable from i64 (spec: print any scalar).
        assert_eq!(Value::I32(-12).render(), "-12");
        assert_eq!(Value::Bool(false).render(), "false");
        assert_eq!(Value::F64(1.0).render(), "1.0");
        // f32 must NOT be widened before formatting.
        assert_eq!(Value::F32(0.1).render(), "0.1");
        assert_eq!(Value::F64(0.1).render(), "0.1");
        assert_eq!(Value::F64(f64::NAN).render(), "NaN");
        assert_eq!(Value::F64(f64::INFINITY).render(), "inf");
        assert_eq!(Value::F32(f32::NEG_INFINITY).render(), "-inf");
    }

    #[test]
    fn widening_only_accepts_integers() {
        assert_eq!(Value::I32(-3).as_i64_widen(), Some(-3));
        assert_eq!(Value::I64(9).as_i64_widen(), Some(9));
        assert_eq!(Value::F64(1.0).as_i64_widen(), None);
        assert_eq!(Value::Bool(true).as_i64_widen(), None);
    }
}
