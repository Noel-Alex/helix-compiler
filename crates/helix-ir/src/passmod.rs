//! Optimization-pass infrastructure: the change flag, the pass registry, and
//! the pipeline driver.
//!
//! Every pass has signature `fn(&mut FuncIr) -> ChangeFlag` so the driver can
//! (a) snapshot IR text between passes for the Observatory OPT view and
//! (b) skip re-runs when a pass reports no movement. The driver verifies after
//! each pass — a buggy rewrite is caught at its own doorstep
//! (`docs/research/ssa-design.md`, recommendation 7).

use crate::ir::FuncIr;
use crate::passes::{const_fold, copy_prop, cse, dce, licm, simplify_cfg};
use crate::print::print_ir;
use crate::verify;

/// Did a pass modify anything?
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChangeFlag {
    /// True when the pass rewrote at least one instruction, φ or edge.
    pub changed: bool,
}

impl ChangeFlag {
    /// "Nothing changed yet".
    #[must_use]
    pub fn new() -> Self {
        Self { changed: false }
    }

    /// Merge another flag into this one (for fixpoint loops).
    pub fn merge(&mut self, other: &ChangeFlag) {
        self.changed |= other.changed;
    }
}

impl Default for ChangeFlag {
    fn default() -> Self {
        Self::new()
    }
}

/// Identity of a pass in the registry (stable across versions — Observatory
/// artifacts key pass dumps by these names).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PassId {
    /// Constant folding ([`crate::passes::const_fold`]).
    ConstFold,
    /// Constant propagation + branch folding ([`crate::passes::const_prop`]).
    ConstProp,
    /// Copy propagation ([`crate::passes::copy_prop`]).
    CopyProp,
    /// Dead-code elimination ([`crate::passes::dce`]).
    Dce,
    /// Common-subexpression elimination ([`crate::passes::cse`]).
    Cse,
    /// Loop-invariant code motion ([`crate::passes::licm`]).
    Licm,
    /// CFG cleanup ([`crate::passes::simplify_cfg`]).
    SimplifyCfg,
}

impl PassId {
    /// Registry name used in artifacts and logs.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            PassId::ConstFold => "const_fold",
            PassId::ConstProp => "const_prop",
            PassId::CopyProp => "copy_prop",
            PassId::Dce => "dce",
            PassId::Cse => "cse",
            PassId::Licm => "licm",
            PassId::SimplifyCfg => "simplify_cfg",
        }
    }

    /// All passes in canonical pipeline order.
    #[must_use]
    pub fn pipeline() -> &'static [PassId] {
        const ORDER: &[PassId] = &[
            PassId::SimplifyCfg,
            PassId::ConstFold,
            PassId::ConstProp,
            PassId::CopyProp,
            PassId::Cse,
            PassId::Dce,
            PassId::Licm,
            PassId::SimplifyCfg,
        ];
        ORDER
    }
}

/// Apply one pass by identity.
pub fn run_pass_by_id(id: PassId, ir: &mut FuncIr) -> ChangeFlag {
    match id {
        PassId::ConstFold => const_fold(ir),
        PassId::ConstProp => crate::passes::const_prop::const_prop(ir),
        PassId::CopyProp => copy_prop(ir),
        PassId::Dce => dce(ir),
        PassId::Cse => cse(ir),
        PassId::Licm => licm(ir),
        PassId::SimplifyCfg => simplify_cfg(ir),
    }
}

/// Run a named pass and verify afterwards.
///
/// # Panics / Errors
/// Returns the verifier message when the pass corrupted the IR.
pub fn run_pass(name: &str, ir: &mut FuncIr) -> Result<ChangeFlag, String> {
    let flag = match name {
        "const_fold" => const_fold(ir),
        "const_prop" => crate::passes::const_prop::const_prop(ir),
        "copy_prop" => copy_prop(ir),
        "dce" => dce(ir),
        "cse" => cse(ir),
        "licm" => licm(ir),
        "simplify_cfg" => simplify_cfg(ir),
        other => return Err(format!("unknown pass '{other}'")),
    };
    verify(ir).map_err(|e| format!("after pass '{name}': {e}"))?;
    Ok(flag)
}

/// Outcome of one pipeline stage, for the Observatory `passes[]` array.
#[derive(Clone, Debug)]
pub struct StageReport {
    /// Pass that produced this state.
    pub pass: PassId,
    /// Whether it changed the IR.
    pub changed: bool,
    /// Full IR text after the pass (`print_ir(ssa=true)`).
    pub after: String,
    /// Instruction count before/after for cheap diff stats.
    pub insts_before: usize,
    pub insts_after: usize,
}

fn count_insts(ir: &FuncIr) -> usize {
    ir.blocks.iter().map(|b| b.insts.len() + b.phis.len()).sum()
}

/// Run the full optimization pipeline over an SSA-form function, verifying
/// after every pass and recording per-stage text snapshots.
pub fn run_optimization_pipeline(ir: &mut FuncIr) -> Vec<StageReport> {
    let mut reports = Vec::new();
    for pass in PassId::pipeline() {
        let before = count_insts(ir);
        let flag = run_pass_by_id(*pass, ir);
        let ok = verify(ir);
        if let Err(e) = ok {
            // A pass bug must be loud; course-scale compiler, no silent
            // recovery. Panic keeps the failure adjacent to its cause.
            panic!("pass {} broke IR: {e}", pass.name());
        }
        reports.push(StageReport {
            pass: *pass,
            changed: flag.changed,
            after: print_ir(ir, true),
            insts_before: before,
            insts_after: count_insts(ir),
        });
    }
    reports
}

/// Run passes until none of them reports a change (max 10 rounds).
pub fn run_passes_to_fixpoint(ir: &mut FuncIr) -> Vec<StageReport> {
    let mut all = Vec::new();
    for _round in 0..10 {
        let mut any_changed = false;
        for pass in PassId::pipeline() {
            let before = count_insts(ir);
            let flag = run_pass_by_id(*pass, ir);
            if let Err(e) = verify(ir) {
                panic!("pass {} broke IR: {e}", pass.name());
            }
            any_changed |= flag.changed;
            all.push(StageReport {
                pass: *pass,
                changed: flag.changed,
                after: print_ir(ir, true),
                insts_before: before,
                insts_after: count_insts(ir),
            });
        }
        if !any_changed {
            break;
        }
    }
    all
}
