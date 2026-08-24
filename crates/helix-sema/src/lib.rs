//! Semantic analysis for HELIX: scopes, symbol tables, type checking, and the static
//! rules from the frozen language spec (`docs/notes/lang-spec.md`).
//!
//! Produces a [`TypedProgram`] mirroring the syntax tree with resolved [`Ty`]s and
//! stable [`SymId`]s (the IR builder reuses them as local slots).

pub mod check;
pub mod fmt;
pub mod types;

pub use check::{
    Builtin, CallTarget, ConstLit, ElseArm, SemDiag, SymId, SymKind, Symbol, TypedBlock,
    TypedConstDef, TypedExpr, TypedExprKind, TypedFnDef, TypedFor, TypedIf, TypedLValue,
    TypedProgram, TypedStmt, check,
};
pub use fmt::{fmt_bool, fmt_f32, fmt_f64, fmt_i64};
pub use types::{ElemTy, Ty};
