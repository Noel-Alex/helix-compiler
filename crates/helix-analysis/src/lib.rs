//! Loop detection, canonicalization, the affine dependence battery, reduction
//! recognition, and parallelization planning — HELIX's headline analysis.
//!
//! Layered design (each layer unit-testable):
//!
//! 1. [`loops`]   — natural loops via back edges, nest forest, depth.
//! 2. [`canon`]   — induction variable + bounds recovery from the IR shape.
//! 3. [`access`]  — affine subscript extraction from loads/stores.
//! 4. [`deps`]    — the dependence battery (ZIV → SIV family → gcd/box → Banerjee).
//! 5. [`reduce`]  — reduction recognition (`x = x op t` shapes).
//! 6. [`plan`]    — verdicts + per-loop reports feeding the backend and the UI.

pub mod access;
pub mod canon;
pub mod deps;
pub mod loops;
pub mod plan;
pub mod reduce;

pub use deps::{DepEdge, DirVec};
pub use plan::{analyze, LoopReport, Reduction, ReductionOp, Verdict};
pub use loops::{Loop, LoopInfo};

use helix_ir::FuncIr;
use serde::{Deserialize, Serialize};

/// A bound expression for a canonical loop: either a compile-time constant or an
/// SSA value defined outside the loop.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Bound {
    Const(i64),
    /// ValueId index into the function's value table.
    Sym(u32),
}
