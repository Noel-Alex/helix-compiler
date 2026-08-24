//! Dominator / frontier / natural-loop unit tests, including the shapes the
//! research digest flags as dangerous: irreducible-ish diamonds, self-loops
//! and nested loops.

use helix_ir::{
    BlockId, FuncIr, Term, dominance_frontiers, dominators, natural_loops, reachability,
};

/// Build a raw CFG: `edges` are (from, to), entry is block 0.
fn cfg(n: usize, edges: &[(u32, u32)]) -> FuncIr {
    let mut ir = FuncIr::new("t", helix_sema::Ty::Unit, 0);
    while ir.blocks.len() < n {
        ir.new_block();
    }
    let mut succs_of = vec![Vec::new(); n];
    for (a, b) in edges {
        succs_of[*a as usize].push(BlockId(*b));
    }
    for (i, mut ss) in succs_of.into_iter().enumerate() {
        ss.sort_unstable();
        ss.dedup();
        match ss.as_slice() {
            [] => ir.set_term(BlockId(i as u32), Term::Return(None)),
            [one] => ir.set_term(BlockId(i as u32), Term::Jump(*one, Vec::new())),
            _ => ir.set_term(
                BlockId(i as u32),
                Term::Branch {
                    cond: helix_ir::ValueId(0),
                    t: ss[0],
                    f: ss[1],
                },
            ),
        }
    }
    ir.recompute_edges();
    ir
}

fn ids(v: &[BlockId]) -> Vec<u32> {
    v.iter().map(|b| b.0).collect()
}

#[test]
fn straight_line() {
    let ir = cfg(3, &[(0, 1), (1, 2)]);
    assert_eq!(ids(&ir.block(BlockId(2)).preds.clone()), vec![1]);
    let d = dominators(&ir);
    assert!(d.dominates(BlockId(0), BlockId(2)));
    assert!(!d.dominates(BlockId(1), BlockId(0)));
    // No joins => no frontiers.
    assert!(dominance_frontiers(&ir, &d).iter().all(Vec::is_empty));
}

#[test]
fn diamond() {
    //      0
    //     / \
    //    1   2
    //     \ /
    //      3
    let ir = cfg(4, &[(0, 1), (0, 2), (1, 3), (2, 3)]);
    let d = dominators(&ir);
    for b in 0..4u32 {
        assert!(d.dominates(BlockId(0), BlockId(b)), "bb{b}");
    }
    assert!(!d.dominates(BlockId(1), BlockId(2)));
    let df = dominance_frontiers(&ir, &d);
    // Both arms have bb3 on their frontier.
    assert_eq!(df[1], vec![BlockId(3)]);
    assert_eq!(df[2], vec![BlockId(3)]);
    assert!(df[3].is_empty());
}

#[test]
fn nested_loops() {
    // Classic nested-while shape:
    //   0 -> 1 (outer header)
    //   1 -> 2 | 5 (exit)
    //   2 -> 3 (inner header) | ... simplified linear inner
    //   3 -> 4
    //   4 -> 1  (outer back edge; inner collapsed)
    let ir = cfg(6, &[(0, 1), (1, 2), (1, 5), (2, 3), (3, 4), (4, 1), (4, 5)]);
    let d = dominators(&ir);
    // Every reachable node dominated by entry.
    for b in 0..6u32 {
        assert!(d.dominates(BlockId(0), BlockId(b)));
    }
    // Back edge 4 -> 1 (1 dominates 4).
    let loops = natural_loops(&ir, &d);
    assert!(
        loops
            .iter()
            .any(|(h, body)| h.0 == 1 && ids(body).contains(&4))
    );
}

#[test]
fn self_loop() {
    let ir = cfg(2, &[(0, 1), (1, 1)]);
    let d = dominators(&ir);
    let loops = natural_loops(&ir, &d);
    assert!(
        loops.iter().any(|(h, body)| h.0 == 1 && body.len() == 1),
        "self-loop must be its own one-block body"
    );
}

#[test]
fn irreducible_ish_two_entry_region() {
    // Two entries into a shared join without a single dominating header —
    // CHK must still terminate and produce sane (if conservative) results.
    //   0 -> 1, 0 -> 2
    //   1 -> 3, 2 -> 3
    //   3 -> 2   (edge back to 2, but 2 is not dominated by 3 => NOT a back edge)
    let ir = cfg(4, &[(0, 1), (0, 2), (1, 3), (2, 3), (3, 2)]);
    let d = dominators(&ir);
    // All blocks reachable.
    assert!(reachability(&ir).iter().all(|v| *v));
    // No back edges => no natural loops (region is irreducible).
    assert!(natural_loops(&ir, &d).is_empty());
}

#[test]
fn deep_chain_no_stack_overflow() {
    // 20k-block chain: explicit-stack DFS and iterative renaming must cope.
    let n = 20_000;
    let mut edges = Vec::with_capacity(n);
    for i in 0..n - 1 {
        edges.push((i as u32, (i + 1) as u32));
    }
    let ir = cfg(n, &edges);
    let d = dominators(&ir);
    assert!(d.dominates(BlockId(0), BlockId((n - 1) as u32)));
    // Dominator tree is a path: idom(bb k) == bb k-1.
    for k in 1..n {
        assert_eq!(d.idom_of(BlockId(k as u32)), Some(BlockId((k - 1) as u32)));
    }
}
