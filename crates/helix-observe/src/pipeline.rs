//! The compile pipeline that produces a [`CompileArtifact`].
//!
//! Runs the real compiler front-to-back, snapshotting every human-readable
//! stage on the way and stopping at the first failure:
//!
//! ```text
//! lex+parse ─▶ tokens, ast
//!      │ error → diags_lex, stop (tokens stay)
//! sema check ─▶ diags_sem ([] = ok), stop when non-empty
//! ir build ─▶ ir_pre_ssa
//! to_ssa ─▶ ir_ssa
//! passes ─▶ passes[] (text + changed flag + inst counts per pass)
//! find_loops + analyze ─▶ loops[], domtree, cfg layout
//! interpret ─▶ exec { printed, checksum }
//! ```
//!
//! Nothing here is allowed to panic on user input: a stage that misbehaves
//! degrades to "that stage is absent" rather than taking the server down.

use helix_analysis::{LoopReport, Verdict};
use helix_ir::FuncIr;
use helix_syntax::Span;

use crate::artifact::{
    self, BoundPair, CfgStage, CompileArtifact, DepEdgeView, DiagView, DomTreeMap, ExecView,
    IrText, LoopView, PassReport, PlanHint, ReductionView, SpanView, TokenView, VerdictLabel,
};
use crate::layout;

/// Upper bound for POSTed sources (64 KiB — the server enforces it too).
pub const MAX_SOURCE_BYTES: usize = 64 * 1024;

/// Builds the full artifact for `source` in one call, executing the program
/// on the reference interpreter as the final stage.
///
/// Equivalent to [`build_artifact_with_opts`] with default options. See that
/// function for why execution is skippable.
#[must_use]
pub fn build_artifact(example: &str, source: &str) -> CompileArtifact {
    build_artifact_with_opts(example, source, BuildOpts::default())
}

/// Knobs controlling how far [`build_artifact_with_opts`] walks the pipeline.
#[derive(Debug, Clone, Copy)]
pub struct BuildOpts {
    /// Run the reference interpreter at the end (default: yes).
    ///
    /// The examples ship at *benchmark* sizes (saxpy allocates 2 × 32 Mi
    /// f64), so interpreting them costs 30–130 s each — fine for a live demo,
    /// pointless inside tests. Callers that only need the analysis stages set
    /// this to `false` and get an artifact whose `exec` is `None`.
    pub execute: bool,
}

impl Default for BuildOpts {
    fn default() -> Self {
        Self { execute: true }
    }
}

impl BuildOpts {
    /// Every stage except execution (used by batch/test callers).
    #[must_use]
    pub fn without_execution() -> Self {
        Self { execute: false }
    }
}

/// Builds the full artifact for `source`, honouring `opts`.
///
/// `example` names the artifact (`"<adhoc>"` for editor submissions). Every
/// pipeline stage fills its field; the first failing stage stops the walk.
#[must_use]
pub fn build_artifact_with_opts(example: &str, source: &str, opts: BuildOpts) -> CompileArtifact {
    let mut art = CompileArtifact::new(example, source);

    // ---- 1. lex + parse -----------------------------------------------------
    let program = match helix_syntax::parse_str(source) {
        Ok(p) => p,
        Err(e) => {
            let (span, msg) = syntax_error_parts(&e);
            art.diags_lex = Some(vec![DiagView {
                span: SpanView::from(span),
                msg,
            }]);
            // Tokens are still shown up to the failure point when we have them.
            if let Ok(toks) = helix_syntax::lex(source) {
                art.tokens = Some(
                    toks.iter()
                        .map(|t| TokenView::of(t, source))
                        .collect::<Vec<_>>(),
                );
            }
            return art;
        }
    };

    art.tokens = Some(
        helix_syntax::lex(source)
            .map(|toks| toks.iter().map(|t| TokenView::of(t, source)).collect())
            .unwrap_or_default(),
    );

    let ast_json = serde_json::to_value(&program).unwrap_or(serde_json::Value::Null);
    art.ast = Some(ast_json.clone());

    // ---- 2. semantic analysis -------------------------------------------------
    let typed = match helix_sema::check(&program) {
        Ok(t) => t,
        Err(diags) => {
            art.diags_sem = diags
                .iter()
                .map(|d| DiagView {
                    span: SpanView::from(d.span),
                    msg: d.msg.clone(),
                })
                .collect();
            return art;
        }
    };
    art.diags_sem = Vec::new();

    // ---- 3. IR build + SSA ------------------------------------------------------
    let mut funcs = helix_ir::build(&typed);
    art.ir_pre_ssa = Some(ir_stage(&funcs));

    for f in &mut funcs {
        helix_ir::to_ssa(f);
    }
    art.ir_ssa = Some(ir_stage(&funcs));

    // ---- 4. optimization passes ---------------------------------------------------
    let mut all_passes: Vec<Vec<helix_ir::passmod::StageReport>> = Vec::new();
    for f in &mut funcs {
        all_passes.push(helix_ir::run_optimization_pipeline(f));
    }
    art.passes = Some(
        all_passes
            .into_iter()
            .flat_map(|stages| stages.into_iter().map(Into::into))
            .collect(),
    );

    // ---- 5. analysis: loops + dominators + CFG layout ------------------------------
    let loop_infos: Vec<helix_analysis::LoopInfo> = funcs
        .iter()
        .map(helix_analysis::loops::find_loops)
        .collect();
    let reports_per_fn: Vec<Vec<LoopReport>> = funcs
        .iter()
        .zip(&loop_infos)
        .map(|(f, li)| helix_analysis::analyze(f, li))
        .collect();

    art.loops = Some(loops_view(&funcs, &reports_per_fn));

    let mut cfg_fns = Vec::with_capacity(funcs.len());
    let mut domtrees = std::collections::BTreeMap::new();
    for (fi, f) in funcs.iter().enumerate() {
        let li = loop_infos.get(fi).cloned().unwrap_or_default();
        cfg_fns.push(layout::cfg_layout(&f.name, f, &li));
        domtrees.insert(f.name.clone(), domtree_of(f));
    }
    art.cfg = Some(CfgStage { functions: cfg_fns });
    art.domtree = Some(domtrees);

    // ---- 6. execute (reference interpreter) ------------------------------------------
    if opts.execute && estimated_work(source) <= INTERACTIVE_WORK_BUDGET {
        art.exec = run_interpreter(source, &typed);
    } else if opts.execute {
        // Benchmark-sized program: interpreting it would block the server for
        // minutes. Leave `exec` absent — the UI greys the stage out and the
        // bench campaign (helix-bench, JIT-backed) owns these numbers anyway.
        art.exec = Some(ExecView {
            backend_used: "interp".to_string(),
            printed: vec![format!(
                "skipped: estimated work exceeds the interactive budget of \
                 {INTERACTIVE_WORK_BUDGET} loop iterations (run `helix bench` for timed results)"
            )],
            checksum: String::new(),
            timings_ms: None,
        });
    }

    art
}

/// Rough static work estimate: the largest literal bound in a `let`/`for`
/// line times a nesting multiplier per additional loop in the file.
///
/// This is deliberately crude — it exists only to separate "runs in
/// milliseconds" from "runs in minutes", not to predict runtime.
#[must_use]
pub fn estimated_work(source: &str) -> u64 {
    let mut max_bound: i64 = 0;
    let mut for_count: u32 = 0;
    for line in source.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("for ") {
            for_count += 1;
        }
        // A line like `let n = 33554432;` or `const N: i64 = 512;` feeds the
        // bound estimate through its numeric literals.
        let biggest = line
            .split(|c: char| !c.is_ascii_digit())
            .filter(|t| !t.is_empty())
            .filter_map(|t| t.parse::<i64>().ok())
            .max()
            .unwrap_or(0);
        max_bound = max_bound.max(biggest);
    }
    let base = max_bound.max(1_000) as u64;
    base.saturating_mul(1u64 << for_count.min(6))
}

/// Above this estimated iteration count the interpreter step is skipped.
pub const INTERACTIVE_WORK_BUDGET: u64 = 20_000_000;

/// Unwraps a [`helix_syntax::SyntaxError`] into its span and message.
fn syntax_error_parts(e: &helix_syntax::SyntaxError) -> (Span, String) {
    match e {
        helix_syntax::SyntaxError::Lex(le) => (le.span, le.to_string()),
        helix_syntax::SyntaxError::Parse(pe) => (pe.span, pe.to_string()),
    }
}

/// Prints every function of one stage.
fn ir_stage(funcs: &[FuncIr]) -> artifact::IrStage {
    artifact::IrStage {
        functions: funcs
            .iter()
            .map(|f| IrText {
                name: f.name.clone(),
                text: helix_ir::print_ir(f, true),
            })
            .collect(),
    }
}

/// Dominator tree of one function as parent-id → child-ids.
fn domtree_of(f: &FuncIr) -> DomTreeMap {
    let doms = helix_ir::dominators(f);
    let kids = doms.tree_children();
    let mut map = DomTreeMap::new();
    for (bi, children) in kids.iter().enumerate() {
        map.insert(
            format!("bb{bi}"),
            children.iter().map(|c| format!("bb{}", c.0)).collect(),
        );
    }
    map
}

/// Maps analysis reports onto the UI's loop objects, adding the thread hint.
fn loops_view(funcs: &[FuncIr], reports: &[Vec<LoopReport>]) -> Vec<LoopView> {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let mut out = Vec::new();
    for (fi, reps) in reports.iter().enumerate() {
        let _fname = funcs.get(fi).map_or("", |f| f.name.as_str());
        for r in reps {
            let (verdict, reason, reduction) = verdict_parts(&r.verdict);
            out.push(LoopView {
                id: r.loop_id,
                depth: r.depth,
                header: r.header.clone(),
                blocks: r.blocks.clone(),
                iv: r.iv.clone(),
                bounds: r.bounds.as_ref().map(|(s, e)| BoundPair {
                    start: s.clone(),
                    end: e.clone(),
                }),
                accesses: r.accesses.clone(),
                raw: r.raw_deps.iter().map(dep_edge).collect(),
                war: r.war_deps.iter().map(dep_edge).collect(),
                waw: r.waw_deps.iter().map(dep_edge).collect(),
                reduction,
                verdict,
                reason,
                plan: plan_for(&verdict, threads),
            });
        }
    }
    out.sort_by_key(|l| l.id);
    out
}

/// Splits the analysis verdict enum into the wire triple.
fn verdict_parts(v: &Verdict) -> (VerdictLabel, String, Option<ReductionView>) {
    match v {
        Verdict::SafeParallel => (
            VerdictLabel::Safe,
            "no loop-carried dependences — iterations may run concurrently".to_string(),
            None,
        ),
        Verdict::ReductionParallel(red) => (
            VerdictLabel::Reduction,
            format!(
                "associative {}-reduction on '{}' — private accumulators combined at exit",
                red.op.symbol(),
                red.var
            ),
            Some(ReductionView {
                op: red.op.symbol().to_string(),
                var: red.var.clone(),
            }),
        ),
        Verdict::Sequential(reason) => (VerdictLabel::Sequential, reason.clone(), None),
    }
}

/// Plan hint exactly when the loop will actually be parallelized.
fn plan_for(verdict: &VerdictLabel, threads: usize) -> Option<PlanHint> {
    matches!(verdict, VerdictLabel::Safe | VerdictLabel::Reduction).then_some(PlanHint { threads })
}

/// Projects an analysis dependence edge onto the wire type.
fn dep_edge(e: &helix_analysis::DepEdge) -> DepEdgeView {
    DepEdgeView {
        kind_label: e.kind_label.clone(),
        array: e.array.clone(),
        distance: e.distance,
        level: e.level,
        direction: e.direction.clone(),
        explain: e.explain.clone(),
    }
}

/// Runs the reference interpreter; any runtime error becomes a printed-line
/// diagnostic instead of a failed request.
#[allow(clippy::option_if_let_else)]
fn run_interpreter(source: &str, typed: &helix_sema::TypedProgram) -> Option<ExecView> {
    match helix_engine::run_with_source(source, typed) {
        Ok(out) => Some(ExecView {
            backend_used: "interp".to_string(),
            printed: out.printed,
            checksum: format!("0x{:016x}", out.checksum),
            timings_ms: None,
        }),
        Err(err) => Some(ExecView {
            backend_used: "interp".to_string(),
            printed: vec![format!("runtime error: {err}")],
            checksum: String::new(),
            timings_ms: None,
        }),
    }
}

// ---------------------------------------------------------------------------
// From<StageReport> for the pass list
// ---------------------------------------------------------------------------

impl From<helix_ir::passmod::StageReport> for PassReport {
    fn from(s: helix_ir::passmod::StageReport) -> Self {
        Self {
            name: s.pass.name().to_string(),
            changed: s.changed,
            after: s.after,
            diff_stats: crate::artifact::DiffStats {
                insts_before: s.insts_before,
                insts_after: s.insts_after,
            },
        }
    }
}
