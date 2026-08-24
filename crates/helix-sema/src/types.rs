//! HELIX's small type universe. See lang-spec.md: zero implicit coercions except
//! array-index i32→i64 widening; i64 is the arithmetic integer; i32/f32 are storage types.

use serde::{Deserialize, Serialize};

/// Element type of an array — scalars only, no nesting.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ElemTy {
    I32,
    I64,
    F32,
    F64,
    Bool,
}

/// The complete set of HELIX types.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Ty {
    I32,
    I64,
    F32,
    F64,
    Bool,
    Array(ElemTy),
    Unit,
}

impl Ty {
    pub fn is_numeric(&self) -> bool {
        matches!(self, Ty::I32 | Ty::I64 | Ty::F32 | Ty::F64)
    }

    pub fn is_integral(&self) -> bool {
        matches!(self, Ty::I32 | Ty::I64)
    }

    pub fn is_float(&self) -> bool {
        matches!(self, Ty::F32 | Ty::F64)
    }

    pub fn is_scalar(&self) -> bool {
        !matches!(self, Ty::Array(_) | Ty::Unit)
    }

    pub fn is_unit(&self) -> bool {
        matches!(self, Ty::Unit)
    }

    pub fn is_array(&self) -> bool {
        matches!(self, Ty::Array(_))
    }

    /// Human-readable name matching the source syntax.
    pub fn name(&self) -> &'static str {
        match self {
            Ty::I32 => "i32",
            Ty::I64 => "i64",
            Ty::F32 => "f32",
            Ty::F64 => "f64",
            Ty::Bool => "bool",
            Ty::Array(ElemTy::I32) => "[i32]",
            Ty::Array(ElemTy::I64) => "[i64]",
            Ty::Array(ElemTy::F32) => "[f32]",
            Ty::Array(ElemTy::F64) => "[f64]",
            Ty::Array(ElemTy::Bool) => "[bool]",
            Ty::Unit => "()",
        }
    }

    /// The scalar element type of an array, or None for non-arrays.
    pub fn elem(self) -> Option<ElemTy> {
        match self {
            Ty::Array(e) => Some(e),
            _ => None,
        }
    }
}
