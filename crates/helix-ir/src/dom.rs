//! Control-flow analysis primitives: reachability, immediate dominators
//! (Cooper–Harvey–Kennedy), and dominance frontiers.
//!
//! Everything is iterative — the DFS uses an explicit work stack, so deep
//! CFGs (e.g. a 10 000-statement chain) cannot blow the native stack. The
//! algorithms follow `docs/research/ssa-design.md`:
//!
//! * **Reachability**: forward DFS from entry over `succs`; anything unvisited
//!   after the walk is unreachable and must be stripped before any dominance
//!   computation (unreachable predecessors fabricate bogus φ arguments).
//! * **CHK idoms** ('A Simple, Fast Dominance Algorithm', Rice): process
//!   blocks in reverse postorder; repeatedly intersect processed predecessors
//!   by walking postorder-number fingers upward until they meet. Near-linear
//!   in practice and dramatically simpler than Lengauer–Tarjan.
//! * **Dominance frontiers**: for every join with ≥ 2 preds run a "runner"
//!   from each pred up the idom chain to the join's idom, inserting the join
//!   into `DF[runner]` along the way (Cooper/Torczon runner form).

use std::collections::VecDeque;

use crate::ir::{BlockId, FuncIr};

/// Sentinel meaning "idom not yet computed".
pub(crate) const UNDEF: u32 = u32::MAX;

/// Dominator information for one function. Valid only for reachable blocks —
/// query [`Doms::reachable`] first.
#[derive(Clone, Debug)]
pub struct Doms {
    /// Reverse-postorder numbering per block (`UNDEF` when unreachable).
    pub rpo: Vec<u32>,
    /// Blocks in reverse postorder (reachable ones only).
    pub order: Vec<BlockId>,
    /// Immediate dominator per block (`BlockId` of itself for the entry).
    /// Unreachable blocks map to [`UNDEF`]-wrapped.
    pub idom: Vec<Option<BlockId>>,
    /// Postorder number of a block (used by `intersect`). Entry has the
    /// largest number.
    pub po_num: Vec<u32>,
}

impl Doms {
    /// Does `a` dominate `b`? Walks the idom chain from `b`. Both must be
    /// reachable.
    #[must_use]
    pub fn dominates(&self, a: BlockId, b: BlockId) -> bool {
        let mut x = b;
        loop {
            if x == a {
                return true;
            }
            match self.idom[x.0 as usize] {
                Some(p) if p != x => x = p,
                _ => return false,
            }
        }
    }

    /// Is the block reachable from entry?
    #[must_use]
    pub fn reachable(&self, b: BlockId) -> bool {
        self.idom[b.0 as usize].is_some()
    }

    /// Immediate dominator (`None` only for the entry block).
    #[must_use]
    pub fn idom_of(&self, b: BlockId) -> Option<BlockId> {
        self.idom[b.0 as usize].filter(|p| *p != b)
    }

    /// Children of each node in the dominator tree, indexed by block id.
    #[must_use]
    pub fn tree_children(&self) -> Vec<Vec<BlockId>> {
        let mut kids: Vec<Vec<BlockId>> = vec![Vec::new(); self.idom.len()];
        for (i, d) in self.idom.iter().enumerate() {
            if let Some(p) = d {
                let pi = p.0 as usize;
                if pi != i {
                    kids[pi].push(BlockId(i as u32));
                }
            }
        }
        kids
    }

    /// Preorder walk of the dominator tree starting at the entry, invoking
    /// `enter(b)` on the way down and `exit_(b)` on the way up. This is the
    /// traversal SSA renaming uses; it is iterative for stack safety.
    pub fn preorder(&self, enter: &mut impl FnMut(BlockId), exit_: &mut impl FnMut(BlockId)) {
        let n = self.idom.len();
        if n == 0 {
            return;
        }
        let kids = self.tree_children();
        // Stack of (block, next-child-index).
        let mut stack: Vec<(BlockId, usize)> = vec![(BlockId(0), 0)];
        enter(BlockId(0));
        while let Some(top) = stack.last().copied() {
            let (b, ci) = top;
            if ci < kids[b.0 as usize].len() {
                let c = kids[b.0 as usize][ci];
                stack.last_mut().expect("nonempty").1 += 1;
                stack.push((c, 0));
                enter(c);
            } else {
                stack.pop();
                exit_(b);
            }
        }
    }
}

/// Forward reachability from the entry block.
///
/// Returns a `visited` flag vector; blocks with `false` are unreachable from
/// entry and must be stripped before dominance-dependent passes.
#[must_use]
pub fn reachability(ir: &FuncIr) -> Vec<bool> {
    let mut seen = vec![false; ir.blocks.len()];
    let mut stack = vec![ir.entry];
    seen[ir.entry.0 as usize] = true;
    while let Some(b) = stack.pop() {
        for s in &ir.block(b).succs {
            let i = s.0 as usize;
            if !seen[i] {
                seen[i] = true;
                stack.push(*s);
            }
        }
    }
    seen
}

/// Compute reverse postorder, CHK immediate dominators, and expose the
/// postorder numbers used by `intersect`.
///
/// # Panics
/// Panics in debug builds if the entry block is out of range.
#[must_use]
pub fn dominators(ir: &FuncIr) -> Doms {
    let n = ir.blocks.len();
    let mut doms = Doms {
        rpo: vec![UNDEF; n],
        order: Vec::new(),
        idom: vec![None; n],
        po_num: vec![UNDEF; n],
    };
    let live = reachability(ir);

    // ---- explicit-stack DFS producing a postorder list --------------------
    // state: 0 = entering, 1 = all children pushed
    let mut state = vec![0u8; n];
    let mut postorder: Vec<BlockId> = Vec::new();
    let mut stack: Vec<(BlockId, usize)> = vec![(ir.entry, 0)];
    state[ir.entry.0 as usize] = 1;
    while let Some(&mut (b, ref mut ci)) = stack.last_mut() {
        let succs = ir.block(b).succs.clone();
        if *ci < succs.len() {
            let s = succs[*ci];
            *ci += 1;
            let si = s.0 as usize;
            if live[si] && state[si] == 0 {
                state[si] = 1;
                stack.push((s, 0));
            }
        } else {
            stack.pop();
            postorder.push(b);
        }
    }

    // postorder -> reverse postorder numbering
    let mut po_num = vec![UNDEF; n];
    for (idx, b) in postorder.iter().enumerate() {
        po_num[b.0 as usize] = idx as u32;
    }
    let mut order = postorder.clone();
    order.reverse(); // RPO
    for (r, b) in order.iter().enumerate() {
        doms.rpo[b.0 as usize] = r as u32;
    }
    doms.po_num = po_num.clone();
    doms.order = order;

    // ---- CHK iteration -----------------------------------------------------
    let entry_i = ir.entry.0 as usize;
    if !live[entry_i] {
        return doms; // degenerate: nothing reachable
    }
    doms.idom[entry_i] = Some(ir.entry);
    let mut changed = true;
    while changed {
        changed = false;
        for i in 1..doms.order.len() {
            let b = doms.order[i];
            let bi = b.0 as usize;
            let mut new_idom: Option<BlockId> = None;
            for p in &ir.block(b).preds {
                let pi = p.0 as usize;
                if !live[pi] || doms.idom[pi].is_none() {
                    continue;
                }
                new_idom = Some(match new_idom {
                    None => *p,
                    Some(cur) => intersect(&po_num, &doms.idom, cur, *p),
                });
            }
            if new_idom != doms.idom[bi] && new_idom.is_some() {
                doms.idom[bi] = new_idom;
                changed = true;
            }
        }
    }
    doms
}

/// CHK `intersect`: two fingers climb toward the root; whichever finger holds
/// the *smaller* postorder number is strictly deeper, so it climbs.
fn intersect(
    po_num: &[u32],
    idom: &[Option<BlockId>],
    mut f1: BlockId,
    mut f2: BlockId,
) -> BlockId {
    while f1 != f2 {
        let n1 = po_num[f1.0 as usize];
        let n2 = po_num[f2.0 as usize];
        if n1 < n2 {
            f1 = idom[f1.0 as usize].expect("intersect above reachable region");
        } else if n2 < n1 {
            f2 = idom[f2.0 as usize].expect("intersect above reachable region");
        } else {
            // Equal numbers but different blocks cannot happen: postorder
            // numbers are unique among visited nodes.
            unreachable!("distinct blocks with equal postorder numbers");
        }
    }
    f1
}

/// Dominance frontiers: `DF[x]` = set of joins where control from a region
/// dominated by `x` may arrive without passing through `x`.
#[must_use]
pub fn dominance_frontiers(ir: &FuncIr, doms: &Doms) -> Vec<Vec<BlockId>> {
    let n = ir.blocks.len();
    let mut df: Vec<Vec<BlockId>> = vec![Vec::new(); n];
    let live = reachability(ir);
    for b in 0..n {
        if !live[b] {
            continue;
        }
        let preds: Vec<BlockId> = ir
            .blocks[b]
            .preds
            .iter()
            .copied()
            .filter(|p| live[p.0 as usize])
            .collect();
        if preds.len() < 2 {
            continue; // runners need a genuine join
        }
        let bj = BlockId(b as u32);
        for p in preds {
            let mut runner = p;
            let stop = doms.idom[b];
            while Some(runner) != stop {
                push_unique(&mut df[runner.0 as usize], bj);
                match doms.idom[runner.0 as usize] {
                    Some(next) => runner = next,
                    None => break, // safety net; should not happen on valid CFGs
                }
            }
        }
    }
    df
}

fn push_unique(v: &mut Vec<BlockId>, x: BlockId) {
    if !v.contains(&x) {
        v.push(x);
    }
}

/// Natural loops discovered from back edges: an edge `t -> h` is a back edge
/// iff `h` dominates `t`; the loop is `h` plus everything that reaches `t`
/// without passing through `h`.
///
/// Returns `(header, body_blocks)` pairs sorted by header id. Nested loops are
/// reported separately (inner first by construction of the backward walk).
#[must_use]
pub fn natural_loops(ir: &FuncIr, doms: &Doms) -> Vec<(BlockId, Vec<BlockId>)> {
    let live = reachability(ir);
    let mut loops: Vec<(BlockId, Vec<BlockId>)> = Vec::new();
    for t in 0..ir.blocks.len() {
        if !live[t] {
            continue;
        }
        for h in ir.blocks[t].succs.clone() {
            if !live[h.0 as usize] || !doms.dominates(h, BlockId(t as u32)) {
                continue;
            }
            // Collect the natural loop body: header + everything that reaches
            // the back-edge source without passing through the header.
            // The walk must NOT traverse the header itself (its predecessors
            // outside the loop, like the preheader, are not part of the body).
            let mut body = vec![h];
            if BlockId(t as u32) != h {
                body.push(BlockId(t as u32));
                let mut work = VecDeque::new();
                work.push_back(BlockId(t as u32));
                while let Some(b) = work.pop_front() {
                    for p in &ir.blocks[b.0 as usize].preds {
                        if *p != h && !body.contains(p) {
                            body.push(*p);
                            work.push_back(*p);
                        }
                    }
                }
            }
            body.sort_unstable();
            loops.push((h, body));
        }
    }
    loops.sort_unstable_by_key(|(h, _)| *h);
    loops
}
