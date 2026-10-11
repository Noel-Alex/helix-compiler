//! The one compile sequence every consumer shares.
//!
//! Everybody that needs to *execute* a HELIX program — the CLI's JIT backend,
//! the benchmark campaign's native variants, the differential selftest — must
//! run the identical stage order, because a divergence between consumers is
//! invisible until it shows up as "the optimizer's verdicts do not match what
//! the campaign published". This module is that single sequence:
//!
//! ```text
//! source ─▶ parse ─▶ check ─▶ IR build ─▶ to_ssa
//!         ─▶ run_optimization_pipeline ─▶ find_loops ─▶ analyze ─▶ build_plan
//! ```
//!
//! Everything downstream of [`prepare_pipeline`] therefore sees **post-SSA,
//! post-optimizer** IR and the loop analysis/plan derived from it — the same
//! IR the Observatory's artifact pipeline snapshots and the browser lab
//! compiles. The optimizer is not an optional extra layered on top later;
//! it is part of what "compiled" means here.
//!
//! Unlike [`crate::pipeline`], this module produces no human-readable
//! snapshots and never fails soft: a stage that cannot run returns `Err` with
//! readable text, because the caller is about to execute the result. It also
//! touches no host facility — no filesystem, clock, or thread APIs — so it
//! compiles unchanged for `wasm32-unknown-unknown`.

use helix_analysis::{LoopInfo, LoopReport, ParallelPlan};
use helix_ir::FuncIr;

/// Everything a consumer needs to lower and execute one program.
///
/// The four views are consistent with each other by construction: `loops` and
/// `reports` describe `funcs`, and `plan` was built from those same three
/// inputs. A consumer that re-runs any stage on its own would risk deriving a
/// plan from a *different* IR than the one it executes, which is precisely the
/// class of bug this type exists to close.
#[derive(Clone, Debug)]
pub struct CompiledUnits {
    /// Post-SSA, post-optimizer control-flow graphs.
    pub funcs: Vec<FuncIr>,
    /// Loop forest of every function, parallel to `funcs`.
    pub loops: Vec<LoopInfo>,
    /// Dependence verdicts per function, parallel to `funcs` and `loops`.
    pub reports: Vec<Vec<LoopReport>>,
    /// Parallel regions the analyzer approved for `funcs`.
    pub plan: ParallelPlan,
}

/// Runs the shared compile sequence on `source`.
///
/// # Errors
///
/// A parse failure, a semantic error, or an optimizer stage that rejects its
/// input, each rendered as readable text. These are the same diagnostics the
/// frontend would show, flattened to one string so every consumer reports the
/// failure the same way.
pub fn prepare_pipeline(source: &str) -> Result<CompiledUnits, String> {
    let program = helix_syntax::parse_str(source).map_err(|e| format!("syntax error: {e}"))?;
    let typed = helix_sema::check(&program).map_err(|diags| format!("{diags:#?}"))?;
    prepare_pipeline_checked(&typed)
}

/// [`prepare_pipeline`] for a caller that already holds a checked program.
///
/// Same sequence from the IR build onward. Consumers that ran their own
/// frontend for diagnostics (the CLI does, so it can render carets) use this
/// instead of paying for a second parse and check.
///
/// # Errors
///
/// An optimizer stage that rejects its input, rendered as readable text.
pub fn prepare_pipeline_checked(
    program: &helix_sema::TypedProgram,
) -> Result<CompiledUnits, String> {
    // ---- IR build + SSA ------------------------------------------------------
    let mut funcs = helix_ir::build(program);
    for f in &mut funcs {
        helix_ir::to_ssa(f);
    }

    // ---- optimizer ------------------------------------------------------------
    // Runs here, not per-consumer: a JIT path that skips it measures and
    // parallelizes IR the Observatory never showed the user.
    for f in &mut funcs {
        helix_ir::run_optimization_pipeline(f);
    }

    // ---- analysis + plan ------------------------------------------------------
    let loops: Vec<LoopInfo> = funcs.iter().map(helix_analysis::find_loops).collect();
    let reports: Vec<Vec<LoopReport>> = funcs
        .iter()
        .zip(&loops)
        .map(|(f, l)| helix_analysis::analyze(f, l))
        .collect();
    let plan = helix_analysis::build_plan(&funcs, &loops, &reports);

    Ok(CompiledUnits {
        funcs,
        loops,
        reports,
        plan,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const DOALL: &str = r#"
fn main() {
    let n = 64;
    let a: [f64] = zeros(n);
    let out: [f64] = zeros(n);
    for i in 0..n {
        a[i] = (i + 1) as f64;
    }
    for i in 0..n {
        out[i] = a[i] * 2.0;
    }
    print(out[63]);
}
"#;

    const REJECTED: &str = r#"
fn main() {
    let n = 32;
    let a: [i64] = zeros(n);
    a[0] = 1;
    for i in 1..n {
        a[i] = a[i - 1] + 1;
    }
    print(a[n - 1]);
}
"#;

    /// The pipeline reaches the analysis stage with all four views aligned.
    #[test]
    fn prepares_funcs_loops_reports_and_plan_together() {
        let units = prepare_pipeline(DOALL).expect("compile");
        assert!(!units.funcs.is_empty());
        assert_eq!(units.loops.len(), units.funcs.len());
        assert_eq!(units.reports.len(), units.funcs.len());
        assert_eq!(
            units.reports.iter().map(Vec::len).sum::<usize>(),
            units.loops.iter().map(|l| l.loops.len()).sum::<usize>(),
            "one report per loop"
        );
        // Both DOALL loops are approved (the init loop is independent too),
        // and every plan region names a real function index.
        assert_eq!(units.plan.regions.len(), 2);
        for region in &units.plan.regions {
            assert!(region.func_idx < units.funcs.len());
        }
    }

    /// The distance-1 recurrence must be rejected, not silently approved.
    #[test]
    fn rejected_loop_produces_an_empty_plan() {
        let units = prepare_pipeline(REJECTED).expect("compile");
        assert!(units.plan.regions.is_empty());
        assert!(
            units
                .reports
                .iter()
                .flatten()
                .any(|r| matches!(r.verdict, helix_analysis::Verdict::Sequential(_))),
            "the recurrence must be reported as sequential"
        );
    }

    /// A parse failure is a readable error, not a panic.
    #[test]
    fn syntax_error_is_reported_as_text() {
        let err = prepare_pipeline("fn main() { let = }").expect_err("must fail");
        assert!(err.contains("syntax error"), "got: {err}");
    }

    /// A semantic failure is a readable error, not a panic.
    #[test]
    fn semantic_error_is_reported_as_text() {
        let err = prepare_pipeline("fn main() { let x: i64 = \"s\"; }").expect_err("must fail");
        assert!(!err.is_empty());
    }

    /// Both entry points agree: the source-level one is the checked one plus
    /// a parse/check, so they must describe the same program.
    #[test]
    fn source_and_checked_entry_points_agree() {
        let program = helix_syntax::parse_str(DOALL).expect("parse");
        let typed = helix_sema::check(&program).expect("check");
        let from_source = prepare_pipeline(DOALL).expect("from source");
        let from_typed = prepare_pipeline_checked(&typed).expect("from typed");
        assert_eq!(
            from_source.plan.regions.len(),
            from_typed.plan.regions.len()
        );
        assert_eq!(from_source.funcs.len(), from_typed.funcs.len());
    }

    /// The optimizer actually ran: the IR handed downstream is the optimizer's
    /// output, so a foldable computation is already folded when it arrives.
    #[test]
    fn funcs_come_back_post_optimization() {
        // `1 + 2` is a constant expression; the IR that reaches the backend
        // must not still contain the addition.
        let units = prepare_pipeline(
            "fn main() { let x = 1 + 2; let a: [i64] = zeros(1); a[0] = x; print(a[0]); }",
        )
        .expect("compile");
        let text = units
            .funcs
            .iter()
            .map(|f| helix_ir::print_ir(f, true))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            !text.contains('+'),
            "constant addition survived the optimizer: {text}"
        );
    }
}
