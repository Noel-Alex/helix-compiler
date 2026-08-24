//! # helix-ir — CFG intermediate representation, SSA and optimization passes
//!
//! Stage 3 of the HELIX pipeline: consumes the typed tree from `helix-sema`
//! and produces optimized, SSA-form control-flow graphs for the analysis,
//! backend and Observatory crates.
//!
//! ```text
//! TypedProgram ──build──▶ Vec<FuncIr> ──to_ssa──▶ SSA FuncIr ──passes──▶ …
//! ```
//!
//! ## Module map
//!
//! * [`ir`] — the contract data types ([`FuncIr`], [`BlockData`], [`Phi`],
//!   [`Inst`], [`Term`], [`Constant`]) plus the structural edge/φ helpers.
//! * [`build`] — typed-tree → CFG lowering (diamonds, loops, short-circuit).
//! * [`dom`] — reachability, Cooper–Harvey–Kennedy dominators, dominance
//!   frontiers, natural-loop discovery.
//! * [`ssa`] — semi-pruned SSA construction (global-name classification,
//!   iterated-dominance-frontier φ placement, dominator-tree renaming) and
//!   post-SSA utilities.
//! * [`passmod`] — the optimizer pass driver; individual passes live in
//!   [`passes`].
//! * [`verify`] — a total well-formedness checker (dominance, φ arity, types,
//!   terminator sanity), designed to run after every pass.
//! * [`print`] — stable `bb0`-style textual form for dumps and golden tests.
//!
//! ## Design notes (course-report material)
//!
//! **Two-stage IR.** The IR leaves the builder deliberately *not* in SSA
//! form: source variables are modelled by stable cell ids so that straight-
//! line lowering is trivial and reassignment is a no-op on ids. `to_ssa`
//! performs classic Cytron-style construction afterwards — this mirrors how
//! production compilers keep a "low" IR distinct from the optimizing form and
//! makes every pass's input unambiguous.
//!
//! **Arrays stay out of SSA.** Only scalars get value identities. Array reads
//! and writes are explicit [`Inst::Load`]/[`Inst::Store`] against the array's
//! local slot, following LLVM's mem2reg precedent; affine dependence analysis
//! can then read address arithmetic directly instead of reasoning about heap
//! SSA (`docs/research/ssa-design.md`, fact 13).
//!
//! **Passes return a change flag**, not unit: the Observatory snapshots IR
//! text between passes and reports `changed` per pass, and the pass driver
//! uses it to skip fixpoint re-runs when nothing moved.

pub mod build;
pub mod dom;
pub mod ir;
pub mod passmod;
pub mod passes;
pub mod print;
pub mod ssa;
pub mod verify;

// -- Contract surface --------------------------------------------------------

pub use build::build;
pub use dom::{Doms, dominance_frontiers, dominators, natural_loops, reachability};
pub use ir::{
    BlockData, BlockId, Call, Constant, FuncIr, Inst, Load, LocalId, Phi, SideTable, Term, ValueId,
};
pub use passmod::{
    ChangeFlag, PassId, run_optimization_pipeline, run_pass, run_pass_by_id, run_passes_to_fixpoint,
};
pub use print::print_ir;
pub use ssa::{
    GlobalNames, global_names, is_ssa, phi_arg_for_pred, strip_unreachable, to_ssa, verify_ssa,
};
pub use verify::verify;
