//! Pipeline integration: every shipped example must produce a well-formed
//! artifact — no panics, no NaN coordinates, stages present exactly as the
//! language semantics dictate (valid programs reach `exec`; broken ones stop
//! at diagnostics).
//!
//! Tests build with [`helix_observe::BuildOpts::without_execution`]: the
//! examples ship at benchmark sizes and the reference interpreter needs
//! 30–130 s per big kernel. Execution itself is covered by helix-engine's
//! own suite plus one small end-to-end run here.

mod common;

use common::{all_example_names, example_source};
use helix_observe::artifact::VerdictLabel;
use helix_observe::{BuildOpts, build_artifact, build_artifact_with_opts};

/// Builds without running the interpreter (fast — no benchmark-size loops).
fn fast(name: &str) -> String {
    let src = example_source(name);
    let art = build_artifact_with_opts(name, &src, BuildOpts::without_execution());
    assert!(
        art.diags_lex.is_none() && art.diags_sem.is_empty(),
        "{name} should be valid"
    );
    serde_json::to_string(&art).expect("serializes")
}

/// Sources small enough that full interpretation stays in milliseconds.
const CHEAP_FULL_RUN: &[&str] = &[
    "ssa_demo",
    "casts_demo",
    "div_guard",
    "recurrence_reject",
    "small_n",
    "gcd_box_test",
    "type_errors",
];

#[test]
fn every_example_builds_without_panic() {
    for name in all_example_names() {
        let src = example_source(&name);
        // Full pipeline only where interpretation is cheap (small_n = 1000
        // iterations); the benchmark-sized kernels run analysis-only here.
        let art = if CHEAP_FULL_RUN.contains(&name.as_str()) {
            build_artifact(&name, &src)
        } else {
            build_artifact_with_opts(&name, &src, BuildOpts::without_execution())
        };
        assert_eq!(art.schema, 1);
        assert_eq!(art.example, name);
        assert_eq!(art.source, src);
    }
}

#[test]
fn small_valid_examples_reach_exec_stage() {
    // Small programs: full pipeline including interpretation.
    for name in ["ssa_demo", "div_guard", "casts_demo", "const_globals"] {
        let art = build_artifact(name, &example_source(name));
        assert!(art.diags_sem.is_empty(), "{name}: unexpected sema diags");
        let exec = art
            .exec
            .as_ref()
            .unwrap_or_else(|| panic!("{name} missing exec"));
        assert_eq!(exec.backend_used, "interp");
        assert!(exec.checksum.starts_with("0x"), "{name}: checksum format");
        assert!(!exec.printed.is_empty(), "{name}: these demos all print");
    }
}

#[test]
fn type_errors_stops_at_sema_with_underline_spans() {
    let src = example_source("type_errors");
    let art = build_artifact("type_errors", &src);
    assert!(!art.diags_sem.is_empty(), "expected semantic errors");
    assert!(art.tokens.is_some(), "tokens survive a sema failure");
    assert!(art.ast.is_some(), "ast survives a sema failure");
    // Later stages absent.
    assert!(art.ir_pre_ssa.is_none());
    assert!(art.ir_ssa.is_none());
    assert!(art.cfg.is_none());
    assert!(art.loops.is_none());
    assert!(art.exec.is_none());
    // Every diagnostic span points inside the source at real characters.
    for d in &art.diags_sem {
        assert!((d.span.end as usize) <= src.len(), "span out of range");
        assert!(d.span.end > d.span.start, "empty span");
        assert!(
            !src[d.span.start as usize..d.span.end as usize]
                .trim()
                .is_empty(),
            "diag span covers only whitespace"
        );
    }
}

#[test]
fn syntax_error_artifact_carries_diags_lex_and_tokens() {
    let art = build_artifact("<adhoc>", "fn main( { let x = ; }");
    assert!(art.diags_lex.is_some(), "expected a syntax diagnostic");
    assert!(art.tokens.is_some());
    assert!(art.ast.is_none());
    assert!(art.exec.is_none());
}

#[test]
fn saxpy_produces_full_stack_with_safe_verdict() {
    let json = fast("saxpy");
    let art: helix_observe::CompileArtifact = serde_json::from_str(&json).expect("round-trips");

    let ir_pre = art.ir_pre_ssa.as_ref().expect("pre-SSA");
    let ir_ssa = art.ir_ssa.as_ref().expect("SSA");
    assert!(!ir_pre.functions.is_empty());
    assert_eq!(ir_pre.functions.len(), ir_ssa.functions.len());
    assert!(ir_ssa.functions[0].text.contains("bb"), "IR text shape");

    let passes = art.passes.as_ref().expect("passes");
    assert!(passes.len() >= 6, "pipeline has several passes");
    assert!(passes.iter().all(|p| !p.after.is_empty()));
    // diff_stats are monotone per pass (never invent instructions).
    assert!(
        passes
            .iter()
            .all(|p| p.diff_stats.insts_after <= p.diff_stats.insts_before)
    );

    let loops = art.loops.as_ref().expect("loops");
    assert!(!loops.is_empty(), "saxpy has one loop to classify");
    let main_loop = &loops[0];
    assert_eq!(
        main_loop.verdict,
        VerdictLabel::Safe,
        "saxpy is embarrassingly parallel"
    );
    assert!(main_loop.raw.is_empty() && main_loop.war.is_empty() && main_loop.waw.is_empty());
    let plan = main_loop.plan.expect("SAFE ⇒ plan hint present");
    assert!(plan.threads >= 1, "plan names a thread count");
}

#[test]
fn reductions_and_rejections_are_classified_correctly() {
    // dot product → +-reduction.
    let dot: helix_observe::CompileArtifact =
        serde_json::from_str(&fast("dot_reduction")).expect("ok");
    let dloops = dot.loops.clone().expect("dot loops");
    assert!(
        dloops.iter().any(|l| l.verdict == VerdictLabel::Reduction
            && l.reduction.as_ref().is_some_and(|r| r.op == "+")),
        "dot product is a +-reduction: {dloops:?}"
    );

    // minmax → min/max reductions.
    let mm: helix_observe::CompileArtifact =
        serde_json::from_str(&fast("minmax_reduction")).expect("ok");
    let mloops = mm.loops.clone().expect("minmax loops");
    assert!(
        mloops.iter().any(|l| l.verdict == VerdictLabel::Reduction),
        "min/max loop recognized: {mloops:?}"
    );

    // distance-1 RAW recurrence MUST stay sequential.
    let rec: helix_observe::CompileArtifact =
        serde_json::from_str(&fast("recurrence_reject")).expect("ok");
    let rloops = rec.loops.clone().expect("recurrence loops");
    assert!(
        rloops.iter().any(|l| l.verdict == VerdictLabel::Sequential
            && l.reason.to_lowercase().contains("distance")),
        "RAW distance-1 rejected with a reason: {rloops:?}"
    );

    // matmul mixes DOALL rows with an inner k-reduction.
    let mmul: helix_observe::CompileArtifact = serde_json::from_str(&fast("matmul")).expect("ok");
    let muloops = mmul.loops.clone().expect("matmul loops");
    assert!(muloops.iter().any(|l| l.verdict == VerdictLabel::Safe));
    assert!(muloops.iter().any(|l| l.verdict == VerdictLabel::Reduction));
}

#[test]
fn artifact_json_round_trips_with_serde_ast_shape() {
    let json = fast("ssa_demo");
    let back: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
    assert_eq!(back["schema"], 1);
    assert!(back["tokens"].as_array().expect("tokens").len() > 5);
    assert!(
        back["ast"]["items"].as_array().is_some(),
        "AST keeps serde shape"
    );
    assert!(back["cfg"]["functions"].as_array().is_some(), "cfg present");
    assert!(back["domtree"].is_object(), "domtree present");
}
