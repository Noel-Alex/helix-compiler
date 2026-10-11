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
//!
//! Two stages need host facilities the wasm32 target does not expose, so both
//! degrade there behind one small helper each: the parallel plan hint reports
//! a single thread (the browser reports real core counts to its worker) and
//! interpreter timings read back as `0.0` (the JS host times wall-clock
//! itself). Everything else — IR, analysis, layout, execution — is identical
//! on both targets.

use helix_analysis::{LoopReport, Verdict};
use helix_ir::FuncIr;
use helix_syntax::Span;

use crate::artifact::{
    self, BenchVariant, BenchView, BoundPair, CfgStage, CompileArtifact, DepEdgeView, DiagView,
    DomTreeMap, ExecView, IrText, LoopView, PassReport, PlanHint, ReductionView, SpanView,
    TokenView, VerdictLabel,
};
use crate::cfg_layout;

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
        cfg_fns.push(cfg_layout(&f.name, f, &li));
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

/// Rough AST work estimate: add sequential work and multiply only nested
/// loop bodies. Comments and numeric fragments are not executable work.
/// Unknown bounds and overflowing arithmetic conservatively skip execution.
#[must_use]
pub fn estimated_work(source: &str) -> u64 {
    use helix_syntax::ast::{BinOp, ElsePart, Expr, Literal, Stmt, UnOp};
    use std::collections::{BTreeMap, BTreeSet};

    fn integer(expr: &Expr, bindings: &BTreeMap<String, i64>) -> Option<i64> {
        match expr {
            Expr::IntLit(n, _) => Some(*n),
            Expr::Var(name) => bindings.get(name.as_str()).copied(),
            Expr::Unary(UnOp::Neg, value, _) => integer(value, bindings)?.checked_neg(),
            Expr::Bin(op, lhs, rhs, _) => {
                let (lhs, rhs) = (integer(lhs, bindings)?, integer(rhs, bindings)?);
                match op {
                    BinOp::Add => lhs.checked_add(rhs),
                    BinOp::Sub => lhs.checked_sub(rhs),
                    BinOp::Mul => lhs.checked_mul(rhs),
                    BinOp::Div => lhs.checked_div(rhs),
                    BinOp::Rem => lhs.checked_rem(rhs),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn children(stmt: &Stmt) -> Vec<&Stmt> {
        match stmt {
            Stmt::For { body, .. } | Stmt::Block(body) => body.stmts.iter().collect(),
            Stmt::If {
                then_blk,
                else_part,
                ..
            } => {
                let mut stmts: Vec<_> = then_blk.stmts.iter().collect();
                match else_part.as_deref() {
                    Some(ElsePart::Block(block)) => stmts.extend(&block.stmts),
                    Some(ElsePart::If(stmt)) => stmts.push(stmt),
                    None => {}
                }
                stmts
            }
            _ => Vec::new(),
        }
    }

    fn assignments(stmt: &Stmt, mutated: &mut BTreeSet<String>) {
        if let Stmt::Assign { target, .. } = stmt
            && target.index.is_none()
        {
            mutated.insert(target.base.name.clone());
        }
        for child in children(stmt) {
            assignments(child, mutated);
        }
    }

    fn work(
        stmts: &[&Stmt],
        mut bindings: BTreeMap<String, i64>,
        mutated: &BTreeSet<String>,
    ) -> u64 {
        let mut total = 0_u64;
        for stmt in stmts {
            let cost = match stmt {
                Stmt::Let { name, init, .. } => {
                    // A zeros() allocation initializes every element: charge
                    // its resolved length, not a flat 1, so allocation-only
                    // programs cannot hide their true cost from the budget.
                    let value = integer(init, &bindings);
                    let cost = match init {
                        Expr::Call { callee, args, .. }
                            if callee.name == "zeros" && args.len() == 1 =>
                        {
                            // Unknown length fails closed like unknown loop
                            // bounds: the runtime allocation length is real
                            // work and memory, so it must not be guessed low.
                            integer(&args[0], &bindings).map_or(u64::MAX, |n| n.max(0) as u64)
                        }
                        _ => 1,
                    };
                    bindings.remove(name.as_str());
                    if !mutated.contains(name.as_str())
                        && let Some(value) = value
                    {
                        bindings.insert(name.name.clone(), value);
                    }
                    cost
                }
                Stmt::For { iv, start, end, .. } => {
                    let trips = integer(start, &bindings)
                        .zip(integer(end, &bindings))
                        .and_then(|(start, end)| end.checked_sub(start))
                        .map_or(u64::MAX, |n| n.max(0) as u64);
                    let mut nested = bindings.clone();
                    nested.remove(iv.as_str());
                    trips.saturating_mul(work(&children(stmt), nested, mutated).max(1))
                }
                Stmt::If {
                    then_blk,
                    else_part,
                    ..
                } => {
                    let then_work = work(
                        &then_blk.stmts.iter().collect::<Vec<_>>(),
                        bindings.clone(),
                        mutated,
                    );
                    let otherwise = match else_part.as_deref() {
                        Some(ElsePart::Block(block)) => work(
                            &block.stmts.iter().collect::<Vec<_>>(),
                            bindings.clone(),
                            mutated,
                        ),
                        Some(ElsePart::If(stmt)) => work(&[stmt], bindings.clone(), mutated),
                        None => 0,
                    };
                    then_work.max(otherwise).saturating_add(1)
                }
                Stmt::Block(_) => work(&children(stmt), bindings.clone(), mutated),
                _ => 1,
            };
            total = total.saturating_add(cost);
        }
        total
    }

    let Ok(program) = helix_syntax::parse_str(source) else {
        return u64::MAX;
    };
    let constants: BTreeMap<_, _> = program
        .consts()
        .filter_map(|c| match c.value {
            Literal::Int(n) => Some((c.name.name.clone(), n)),
            _ => None,
        })
        .collect();
    // ponytail: syntactic loop work, not a call-graph runtime bound. Unknown
    // ranges fail closed; instruction fuel would bound calls/recursion too.
    program.fns().fold(0_u64, |total, function| {
        let mut mutated = BTreeSet::new();
        for stmt in &function.body.stmts {
            assignments(stmt, &mut mutated);
        }
        let mut bindings = constants.clone();
        for param in &function.params {
            bindings.remove(param.name.as_str());
        }
        total.saturating_add(work(
            &function.body.stmts.iter().collect::<Vec<_>>(),
            bindings,
            &mutated,
        ))
    })
}

/// Above this estimated iteration count the interpreter step is skipped.
///
/// Eight million keeps the observer responsive for every example in the
/// repository on an ordinary laptop; larger benchmark-sized programs still
/// expose their complete analysis artifact and an explicit skipped-execution
/// note for `helix bench`.
pub const INTERACTIVE_WORK_BUDGET: u64 = 8_000_000;

/// Maximum estimated work accepted by the interactive custom-source
/// measurement. The benchmark button is deliberately more conservative than
/// the one-shot observer run: it executes several samples, so an arbitrary
/// source must not be able to turn a browser click into a minutes-long server
/// request.
pub const CUSTOM_BENCH_WORK_BUDGET: u64 = 1_000_000;

/// Number of measured interpreter samples for a custom-source benchmark.
pub const CUSTOM_BENCH_SAMPLE_COUNT: usize = 7;

/// Measures a checked custom program on the reference interpreter.
///
/// This is intentionally a single-variant measurement. The native/JIT
/// campaign only exists for the named kernel registry, so the result is
/// labelled `interp-local` and carries no invented speedup or efficiency
/// claims. The timed operation follows the same public `run_with_source`
/// path used by the benchmark harness, including its source-aware execution
/// contract.
///
/// # Errors
/// Returns a readable parse, semantic, runtime, nondeterminism, or budget
/// error. Callers should display this as an unavailable measurement rather
/// than treating it as a compiler failure.
pub fn measure_interpreter(example: &str, source: &str) -> Result<BenchView, String> {
    let estimated = estimated_work(source);
    if estimated > CUSTOM_BENCH_WORK_BUDGET {
        return Err(format!(
            "estimated work {estimated} exceeds the custom benchmark budget of \
             {CUSTOM_BENCH_WORK_BUDGET}; use a smaller input or the recorded kernel campaign"
        ));
    }

    let program = helix_syntax::parse_str(source).map_err(|e| format!("syntax error: {e}"))?;
    let typed = helix_sema::check(&program).map_err(|diags| format!("{diags:#?}"))?;

    // A warmup absorbs the first interpreter allocation path; it is not
    // reported as evidence.
    let baseline = helix_engine::run_with_source(source, &typed).map_err(|e| e.render(source))?;
    let mut samples = Vec::with_capacity(CUSTOM_BENCH_SAMPLE_COUNT);
    for _ in 0..CUSTOM_BENCH_SAMPLE_COUNT {
        let (result, elapsed_ms) = timed_ms(|| {
            helix_engine::run_with_source(source, &typed).map_err(|e| e.render(source))
        });
        let output = result?;
        if output.printed != baseline.printed || output.checksum != baseline.checksum {
            return Err(
                "custom benchmark produced non-deterministic output across samples".to_string(),
            );
        }
        std::hint::black_box(&output);
        samples.push(elapsed_ms);
    }

    let mut ordered = samples.clone();
    ordered.sort_by(f64::total_cmp);
    let median_ms = ordered[ordered.len() / 2];

    Ok(BenchView {
        kernel: example.to_string(),
        // Arbitrary source has no registry-defined problem-size field. The UI
        // omits `n` for this local measurement rather than displaying a fake
        // input size.
        n: 0,
        variants: vec![BenchVariant {
            name: "interp-local".to_string(),
            median_ms,
            samples,
        }],
        efficiency: Vec::new(),
    })
}

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

/// Thread-count hint for the runtime plan attached to approved loops.
///
/// `available_parallelism` is a host syscall with no wasm32 implementation,
/// so the hint degrades to a single thread there: the browser lab runs one
/// Web Worker per call and re-measures wall-clock itself, so the artifact
/// hint is a floor, not a claim.
fn plan_threads() -> usize {
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::thread::available_parallelism().map_or(1, |n| n.get())
    }
    #[cfg(target_arch = "wasm32")]
    {
        1
    }
}

/// Milliseconds spent inside `f`.
///
/// `std::time::Instant` has no monotonic clock on wasm32 (calling it aborts),
/// so the elapsed value reads back as `0.0` there — the browser host measures
/// wall-clock separately and reports it as its own evidence.
fn timed_ms<T>(f: impl FnOnce() -> T) -> (T, f64) {
    #[cfg(not(target_arch = "wasm32"))]
    let start = std::time::Instant::now();
    let out = f();
    #[cfg(not(target_arch = "wasm32"))]
    let elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0;
    #[cfg(target_arch = "wasm32")]
    let elapsed_ms = 0.0;
    (out, elapsed_ms)
}

/// Maps analysis reports onto the UI's loop objects, adding the thread hint.
fn loops_view(funcs: &[FuncIr], reports: &[Vec<LoopReport>]) -> Vec<LoopView> {
    let threads = plan_threads();

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
    // One timed execution: the budget gate already skipped benchmark-sized
    // programs, so this sample is interactive-sized by construction. Samples
    // are reference-only (timer precision is OS-dependent) — the bench
    // campaign owns authoritative numbers.
    let (result, elapsed_ms) = timed_ms(|| helix_engine::run_with_source(source, typed));
    match result {
        Ok(out) => Some(ExecView {
            backend_used: "interp".to_string(),
            printed: out.printed,
            checksum: format!("0x{:016x}", out.checksum),
            timings_ms: Some(vec![elapsed_ms]),
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The native branch of the gate is the host core count it always was, so
    /// `plan.threads` in the artifact keeps its meaning off wasm32.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn plan_threads_reports_the_host_core_count() {
        assert_eq!(
            plan_threads(),
            std::thread::available_parallelism().map_or(1, |n| n.get())
        );
    }

    /// Wrapping the timed region must not change what it returns.
    #[test]
    fn timed_ms_reports_the_closure_result() {
        let (value, elapsed_ms) = timed_ms(|| 7_u32);
        assert_eq!(value, 7);
        assert!(elapsed_ms.is_finite());
        assert!(elapsed_ms >= 0.0);
    }

    /// The native branch of the gate actually measures wall-clock.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn timed_ms_measures_a_real_region() {
        let (slept, elapsed_ms) =
            timed_ms(|| std::thread::sleep(std::time::Duration::from_millis(5)));
        assert_eq!(slept, ());
        assert!(elapsed_ms >= 1.0, "5 ms sleep measured {elapsed_ms} ms");
    }
}
