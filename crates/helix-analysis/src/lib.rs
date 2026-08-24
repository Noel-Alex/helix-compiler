//! Loop detection, canonicalization, the affine dependence battery, reduction
//! recognition, and parallelization planning — HELIX's headline analysis.
//!
//! Layered design (each layer unit-testable):
//!
//! 1. [`loops`]   — natural loops via back edges, nest forest, depth.
//! 2. [`canon`]   — induction variable + bounds recovery from the IR shape.
//! 3. [`access`]  — affine subscript extraction from loads/stores.
//! 4. [`deps`]    — the dependence battery (ZIV → SIV family → gcd/box).
//! 5. [`reduce`]  — reduction recognition (`x = x op t` shapes).
//! 6. [`plan`]    — verdicts + per-loop reports feeding the backend and the UI.

pub mod access;
pub mod canon;
pub mod deps;
pub mod loops;
pub mod plan;
pub mod reduce;

pub use loops::{Loop, LoopInfo};
pub use plan::{DepEdge, LoopReport, Reduction, ReductionOp, Verdict, analyze};

use serde::{Deserialize, Serialize};

/// A loop bound: compile-time constant or symbolic (SSA value id).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Bound {
    Const(i64),
    Sym(u32),
}
