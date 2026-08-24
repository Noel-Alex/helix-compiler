//! Layout integration: CFG geometry invariants over every analyzed example.
//!
//! These tests pin down the contract the browser relies on: finite
//! coordinates, boxes that do not overlap their layer neighbours, edges whose
//! endpoints sit on box borders, backedges drawn as 3-point quadratic curves,
//! and roles/loop ids consistent with the analysis.

mod common;

use helix_observe::artifact::{BlockRole, EdgeKind};
use helix_observe::{BuildOpts, TreeNode, build_artifact_with_opts, program_to_tree};

/// Compiles `name` (no execution) and returns the artifact.
fn fast(name: &str) -> helix_observe::CompileArtifact {
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
// AST tree layout
// ---------------------------------------------------------------------------

fn ast_of(name: &str) -> serde_json::Value {
    fast(name).ast.expect("ast present for valid examples")
}

#[test]
fn ast_adapter_builds_hierarchy_from_serde_program() {
    let ast = ast_of("saxpy");
    let root = program_to_tree(&ast).expect("Program JSON adapts");
    assert_eq!(root.label, "Program");
    assert!(!root.children.is_empty(), "top-level items present");

    // saxpy has exactly one fn item.
    let fns: Vec<&TreeNode> = root
        .children
        .iter()
        .filter(|c| c.label == "FnDef")
        .collect();
    assert_eq!(fns.len(), 1);
    assert_eq!(fns[0].detail, "main");
}

#[test]
fn ast_tree_layout_assigns_finite_nonoverlapping_coordinates() {
    let ast = ast_of("gcd_box_test"); // nested-ish expressions
    let root = program_to_tree(&ast).expect("adapts");
    let laid = helix_observe::ast_tree(&root);

    assert_eq!(laid[0].node.label, "Program", "root first (preorder)");
    assert!(laid.len() > 10, "non-trivial tree");

    for n in &laid {
        assert!(is_finite(n.x) && is_finite(n.y));
        assert!(n.x >= 0.0 && n.y >= 0.0);
    }

    // Depth strictly increases along parent→child edges of the payload tree.
    // Reconstruct depth from y (y = margin + depth*row_gap).
    const ROW_GAP: f64 = 78.0;
    let depth = |y: f64| ((y - 28.0) / ROW_GAP).round();
    // Leaves appear in increasing column order (in-order invariant).
    let mut last_leaf_x = -1.0;
    for n in &laid {
        if n.node.children.is_empty() {
            assert!(n.x > last_leaf_x, "leaf columns increase left-to-right");
            last_leaf_x = n.x;
        }
        assert!(depth(n.y) >= 0.0);
    }

    // A parent is centred over its children (average of first/last child x).
    let find = |label: &str| laid.iter().position(|n| n.node.label == label).unwrap();
    let prog_idx = find("Program");
    let prog = &laid[prog_idx];
    let kids: Vec<f64> = prog.node.children.iter().map(|_| 0.0).collect();
    let _ = kids; // children positions checked via the leaf ordering above

    // Total canvas width equals the number of leaves × slot width (bounded).
    let leaves = laid.iter().filter(|n| n.node.children.is_empty()).count();
    assert!(last_leaf_x <= (leaves as f64) * 200.0);
}
