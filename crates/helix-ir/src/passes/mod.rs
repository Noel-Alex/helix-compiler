//! Optimization passes over SSA-form [`FuncIr`].
//!
//! Each submodule is one pass with signature `fn(&mut FuncIr) ->
//! ChangeFlag`, documented and unit-tested in place. Shared conventions:
//!
//! * **Purity triage.** `Const`/`Bin`/`Unary`/`Cast` are pure; `Load` may
//!   trap (bounds); `Store` and `Call` are effects. DCE roots, CSE keys and
//!   LICM candidates all derive from this single classification.
//! * **Arrays are memory.** No pass ever moves, deletes or duplicates a
//!   `Load`/`Store`/`Call`; only the dependence analysis (another crate) may
//!   reorder them, after proving independence.
//! * **CFG surgery is centralized.** Edge edits go through
//!   [`FuncIr::set_term`] / [`FuncIr::compact`] so predecessor lists and φ
//!   argument lists stay aligned — the classic corruption bug is a one-sided
//!   edge update.

pub mod const_fold;
pub mod const_prop;
pub mod copy_prop;
pub mod cse;
pub mod dce;
pub mod licm;
pub mod simplify_cfg;

pub use const_fold::{const_fold, fold_bin, fold_cast, fold_unary};
pub use copy_prop::copy_prop;
pub use cse::cse;
pub use dce::dce;
pub use licm::licm;
pub use simplify_cfg::simplify_cfg;

// const_prop re-exported by path to avoid a name clash with the module.
