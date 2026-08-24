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
//!
//! ## Pipeline position
//!
//! ```text
//! helix_ir::build ──to_ssa──▶ FuncIr (SSA)
//!                                 │
//!            find_loops(&func)    │ analyze(&func, &loops) per function
//!                 ▼               ▼
//!             Vec<LoopInfo> ─▶ Vec<Vec<LoopReport>> ──build_plan──▶ ParallelPlan
//! ```
//!
//! The plan is what the backend consumes; the reports are what humans (and
//! the Observatory) read.

pub mod access;
pub mod canon;
pub mod deps;
pub mod loops;
pub mod plan;
pub mod reduce;

use serde::{Deserialize, Serialize};

/// A loop bound: compile-time constant or symbolic (SSA value id).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Bound {
    Const(i64),
    Sym(u32),
}

// -- Contract surface --------------------------------------------------------

pub use loops::{Loop, LoopInfo, find_loops};
pub use plan::{
    DepEdge, LoopReport, ParallelPlan, Reduction, ReductionOp, RegionDesc, RegionKind, Verdict,
    analyze, build_plan,
};
pub use reduce::{Recognized, find_reductions};
