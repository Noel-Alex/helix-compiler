//! Natural-loop discovery: back edges (edges whose target dominates their source)
//! and the backward-reachable block sets they define. Produces a nest forest.

use helix_ir::{BlockId, FuncIr};
use serde::{Deserialize, Serialize};

/// One natural loop. `blocks` includes the header; iteration order unspecified.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Loop {
    pub id: usize,
    pub header: BlockId,
    pub blocks: Vec<BlockId>,
    pub depth: u32,
    /// Index into LoopInfo::loops of the immediately enclosing loop, if any.
    pub parent: Option<usize>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LoopInfo {
    pub loops: Vec<Loop>,
}

impl LoopInfo {
    pub fn for_block(&self, b: BlockId) -> Option<&Loop> {
        // Innermost loop containing b = one with smallest depth among matches.
        self.loops
            .iter()
            .filter(|l| l.blocks.contains(&b))
            .min_by_key(|l| l.depth)
    }
}

/// Discover all natural loops of `func` using its dominator tree.
///
/// A back edge is u→h where h dominates u; the natural loop of that edge is
/// {h} ∪ {nodes that can reach u without passing through h}.
pub fn find_loops(func: &FuncIr) -> LoopInfo {
    let dom = helix_ir::dom::Dominators::compute(func);
    let mut loops: Vec<(BlockId, Vec<BlockId>)> = Vec::new();

    for (u_idx, &bd) in func.blocks.iter().enumerate() {
        let u = BlockId(u_idx as u32);
        if !dom.reachable[u_idx] {
            continue;
        }
        for &s in &bd.succs {
            if dom.dominates(s, u) {
                // Back edge u -> s. Collect the natural loop.
                let mut body = vec![s];
                let mut stack = vec![u];
                if u != s {
                    body.push(u);
                }
                while let Some(x) = stack.pop() {
                    for &p in &func.blocks[x.0 as usize].preds {
                        if !body.contains(&p) && dom.reachable[p.0 as usize] {
                            body.push(p);
                            stack.push(p);
                        }
                    }
                }
                // Merge loops sharing a header (multiple back edges to same header).
                if let Some(entry) = loops.iter_mut().find(|(h, _)| *h == s) {
                    for b in body {
                        if !entry.1.contains(&b) {
                            entry.1.push(b);
                        }
                    }
                } else {
                    loops.push((s, body));
                }
            }
        }
    }

    // Sort by size so parents come before children; derive depth/parent from containment.
    loops.sort_by_key(|(_, body)| body.len());
    let mut out: Vec<Loop> = Vec::new();
    for (i, (header, body)) in loops.iter().enumerate() {
        // Parent = innermost already-added loop strictly containing this one's header+body.
        let parent = out
            .iter()
            .enumerate()
            .filter(|(_, cand)| body.iter().all(|b| cand.blocks.contains(b)))
            .map(|(i, _)| i)
            .max();
        let depth = parent.map_or(1, |p| out[p].depth + 1);
        out.push(Loop {
            id: i,
            header: *header,
            blocks: body.clone(),
            depth,
            parent,
        });
    }
    LoopInfo { loops: out }
}

#[cfg(test)]
mod tests {
    use super::*;
    use helix_ir::testutil;

    #[test]
    fn single_loop_shape() {
        let ir = testutil::counting_loop(); // bb0 -> bb1 -> (bb2 -> bb1)* -> bb3
        let li = find_loops(&ir);
        assert_eq!(li.loops.len(), 1);
        let l = &li.loops[0];
        assert_eq!(l.header, BlockId(1));
        assert_eq!(l.blocks.len(), 3); // header, body, latch
        assert_eq!(l.depth, 1);
    }

    #[test]
    fn nested_loops_get_depths() {
        let ir = testutil::nested_loops();
        let li = find_loops(&ir);
        assert_eq!(li.loops.len(), 2);
        assert!(li.loops.iter().any(|l| l.depth == 1));
        assert!(li.loops.iter().any(|l| l.depth == 2));
        // Outer contains inner.
        let outer = li.loops.iter().find(|l| l.depth == 1).unwrap();
        let inner = li.loops.iter().find(|l| l.depth == 2).unwrap();
        assert!(outer.blocks.contains(&inner.header));
    }
}
