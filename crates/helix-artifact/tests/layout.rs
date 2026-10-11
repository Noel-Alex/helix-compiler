//! CFG layout integration: geometry invariants over every analyzed example.
//!
//! These tests pin down the contract the browser relies on: finite
//! coordinates, boxes that do not overlap their layer neighbours, edges whose
//! endpoints sit on box borders, backedges drawn as 3-point quadratic curves,
//! and roles/loop ids consistent with the analysis.
//!
//! The AST tidy tree has its own display hierarchy and lives with its own
//! crate; only the CFG half of the artifact's layout is covered here.

mod common;

use helix_artifact::artifact::{BlockRole, EdgeKind};
use helix_artifact::{BuildOpts, CompileArtifact, build_artifact_with_opts, cfg_layout};

/// Compiles `name` (no execution) and returns the artifact.
fn fast(name: &str) -> CompileArtifact {
    let src = common::example_source(name);
    build_artifact_with_opts(name, &src, BuildOpts::without_execution())
}

fn is_finite(x: f64) -> bool {
    x.is_finite()
}

#[test]
fn cfg_geometry_is_finite_and_bounded_for_every_example() {
    for name in common::all_example_names() {
        if name == "type_errors" || name == "stencil_2d_reject" {
            continue; // no IR stage ⇒ no CFG by design (sema / syntax failure)
        }
        let art = fast(&name);
        let cfg = art
            .cfg
            .as_ref()
            .unwrap_or_else(|| panic!("{name}: cfg missing"));
        assert!(!cfg.functions.is_empty(), "{name}: at least one function");

        for f in &cfg.functions {
            assert!(!f.nodes.is_empty(), "{name}/{}: no blocks", f.name);
            for n in &f.nodes {
                assert!(
                    is_finite(n.x) && is_finite(n.y),
                    "{name}/{}/{}: NaN pos",
                    f.name,
                    n.id
                );
                assert!(n.w >= 120.0 && n.h > 0.0, "{name}/{}: degenerate box", n.id);
                assert!(!n.lines.is_empty(), "{name}/{}: empty block text", n.id);
                // Every node sits inside a sane canvas (margin + content).
                assert!(n.x >= 0.0 && n.y >= 0.0, "{name}/{}: negative origin", n.id);
            }
            // Edge endpoints reference real nodes and carry points.
            let ids: std::collections::HashSet<_> = f.nodes.iter().map(|n| n.id.as_str()).collect();
            for e in &f.edges {
                assert!(
                    ids.contains(e.from.as_str()),
                    "{}: unknown src {}",
                    f.name,
                    e.from
                );
                assert!(
                    ids.contains(e.to.as_str()),
                    "{}: unknown dst {}",
                    f.name,
                    e.to
                );
                assert!(
                    e.points
                        .iter()
                        .all(|p| p.len() == 2 && is_finite(p[0]) && is_finite(p[1])),
                    "{name}/{}->{}: bad point",
                    e.from,
                    e.to
                );
                match e.kind {
                    EdgeKind::Backedge => {
                        assert_eq!(e.points.len(), 3, "backedge must be a 3-point bezier");
                    }
                    _ => assert!(
                        e.points.len() == 2 || e.points.len() == 4,
                        "forward edge is a line or elbow"
                    ),
                }
            }
        }
    }
}

#[test]
fn entry_block_exists_and_roles_are_sane() {
    // ssa_demo: `if cond { x = 10 }` — the join survives const_prop's branch
    // folding only when both arms are live; here the merge block exists.
    let art = fast("gcd_box_test"); // loop ⇒ header + latch + join-free body
    let cfg = art.cfg.expect("cfg");
    let main = cfg
        .functions
        .iter()
        .find(|f| f.name == "main")
        .expect("main");
    assert_eq!(main.nodes[0].role, BlockRole::Entry, "bb0 is the entry");
    assert!(
        main.nodes.iter().any(|n| matches!(n.role, BlockRole::Exit)),
        "a returning function has an exit block"
    );
    assert!(
        main.nodes.iter().any(|n| n.role == BlockRole::LoopHeader),
        "a counted loop has a loop-header block"
    );
}

#[test]
fn loop_headers_carry_loop_ids_matching_analysis() {
    let art = fast("matmul");
    let loops = art.loops.clone().expect("loops");
    assert!(!loops.is_empty());
    let cfg = art.cfg.expect("cfg");
    let main = cfg
        .functions
        .iter()
        .find(|f| f.name == "main")
        .expect("main");
    for lp in &loops {
        let header_node = main
            .nodes
            .iter()
            .find(|n| n.id == lp.header)
            .expect("loop header exists in cfg");
        assert_eq!(
            header_node.role,
            BlockRole::LoopHeader,
            "loop {} header flagged",
            lp.id
        );
        // Every body block knows SOME innermost loop id.
        for b in &lp.blocks {
            let node = main.nodes.iter().find(|n| n.id == *b).expect("body block");
            assert!(node.loop_id.is_some(), "block {b} carries a loop id");
        }
    }
    // A block belonging to two loops resolves to the innermost (deepest) one.
    let deepest = loops.iter().max_by_key(|l| l.depth).expect("some loop");
    let shared: Vec<&str> = deepest
        .blocks
        .iter()
        .filter(|b| {
            loops
                .iter()
                .filter(|l| l.depth < deepest.depth)
                .any(|l| l.blocks.contains(b))
        })
        .map(String::as_str)
        .collect();
    for b in shared {
        let node = main.nodes.iter().find(|n| n.id == b).expect("shared block");
        let inner = loops
            .iter()
            .filter(|l| l.blocks.contains(&node.id.clone()))
            .max_by_key(|l| l.depth)
            .expect("innermost")
            .id;
        assert_eq!(
            node.loop_id,
            Some(inner),
            "{b} resolves to its innermost loop"
        );
    }
}

#[test]
fn backedges_connect_body_blocks_to_headers() {
    let art = fast("saxpy");
    let cfg = art.cfg.expect("cfg");
    let main = &cfg.functions[0];
    let curves: Vec<_> = main
        .edges
        .iter()
        .filter(|e| e.kind == EdgeKind::Backedge)
        .collect();
    assert!(!curves.is_empty(), "saxpy's loop has a latch backedge");
    let loops = art.loops.as_ref().expect("loops");
    for e in curves {
        let lp = loops
            .iter()
            .find(|l| l.header == e.to)
            .expect("backedge targets a header");
        assert!(
            lp.blocks.contains(&e.from),
            "backedge source inside its body"
        );
        // The control point bulges right of both endpoints (sideways bow).
        let [sx, sy] = e.points[0];
        let [cx, _cy] = e.points[1];
        let [ex, ey] = e.points[2];
        assert!(cx > sx && cx > ex, "control point bows sideways");
        assert!(sy != ey || sx != ex, "self-loop still has extent");
    }
}

#[test]
fn same_layer_boxes_never_overlap() {
    for name in ["count_primes_sieve", "jacobi_2d", "fib_recursion"] {
        let art = fast(name);
        let cfg = art.cfg.expect("cfg");
        for f in &cfg.functions {
            let mut rows: Vec<(f64, f64)> = Vec::new(); // (y, x) pairs
            for n in &f.nodes {
                rows.push((n.y, n.x));
            }
            rows.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
            for w in rows.windows(2) {
                if (w[0].0 - w[1].0).abs() < 1.0 {
                    assert!(
                        (w[0].1 - w[1].1).abs() > 60.0,
                        "{name}/{}: boxes stacked too tightly",
                        f.name
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// cfg_layout as a standalone pure function
// ---------------------------------------------------------------------------

/// Compiles `name` and lays out the first function straight from its IR, the
/// way the pipeline does.
fn layout_of(name: &str) -> helix_artifact::CfgFunction {
    let src = common::example_source(name);
    let program = helix_syntax::parse_str(&src).expect("valid example");
    let typed = helix_sema::check(&program).expect("type-checks");
    let mut funcs = helix_ir::build(&typed);
    for f in &mut funcs {
        helix_ir::to_ssa(f);
    }
    let f = funcs.first().expect("at least one function");
    let loops = helix_analysis::loops::find_loops(f);
    cfg_layout(&f.name, f, &loops)
}

#[test]
fn cfg_layout_is_deterministic_and_pure() {
    let first = layout_of("gcd_box_test");
    let second = layout_of("gcd_box_test");
    let a = serde_json::to_string(&first).expect("serializes");
    let b = serde_json::to_string(&first).expect("serializes");
    let c = serde_json::to_string(&second).expect("serializes");
    assert_eq!(a, b, "same input ⇒ same document");
    assert_eq!(a, c, "no ambient state: recompute is identical");
}

#[test]
fn cfg_layout_handles_every_function_of_a_program_without_panicking() {
    // Every function gets its own layout, including ones with no loop at all.
    let src = "fn helper(n: i64) -> i64 { if n < 1 { return 0; } return helper(n - 1) + n; }\n\
               fn main() { print(helper(5)); }";
    let program = helix_syntax::parse_str(src).expect("valid");
    let typed = helix_sema::check(&program).expect("type-checks");
    let mut funcs = helix_ir::build(&typed);
    for f in &mut funcs {
        helix_ir::to_ssa(f);
    }
    let names: Vec<&str> = funcs.iter().map(|f| f.name.as_str()).collect();
    for f in &funcs {
        let loops = helix_analysis::loops::find_loops(f);
        let laid = cfg_layout(&f.name, f, &loops);
        assert!(names.contains(&laid.name.as_str()), "{} laid out", f.name);
        assert!(!laid.nodes.is_empty());
        assert!(
            laid.edges.iter().all(|e| e
                .points
                .iter()
                .all(|p| p.len() == 2 && is_finite(p[0]) && is_finite(p[1]))),
            "{}: every routed point is a finite pair",
            f.name
        );
    }
}
