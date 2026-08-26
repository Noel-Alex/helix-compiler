//! Loop-nest construction on top of helix-ir's natural-loop discovery.
//!
//! helix-ir::dom gives raw back-edge loops (header + body sets); this module
//! orders them into a forest (outer before inner), assigns depths, and links
//! each loop to its immediate parent.

use helix_ir::{BlockId, FuncIr};
use serde::{Deserialize, Serialize};

/// One loop of the nest forest. `blocks` includes the header.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Loop {
    pub id: usize,
    pub header: BlockId,
    pub blocks: Vec<BlockId>,
    pub depth: u32,
    /// Index into [`LoopInfo::loops`] of the immediately enclosing loop, if any.
    pub parent: Option<usize>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LoopInfo {
    pub loops: Vec<Loop>,
}

impl LoopInfo {
    /// The innermost loop containing `b` (smallest depth among matches).
    pub fn innermost_containing(&self, b: BlockId) -> Option<&Loop> {
        self.loops
            .iter()
            .filter(|l| l.blocks.contains(&b))
            .min_by_key(|l| l.depth)
    }

    /// Loops at exactly the given nesting depth (1 = outermost).
    pub fn at_depth(&self, d: u32) -> impl Iterator<Item = &Loop> {
        self.loops.iter().filter(move |l| l.depth == d)
    }
}

/// Build the nest forest from `func`.
///
/// Multiple back edges sharing a header are merged by the underlying discovery;
/// containment is decided by body-set inclusion (a loop is a child when its
/// whole body sits inside the candidate's).
pub fn find_loops(func: &FuncIr) -> LoopInfo {
    let doms = helix_ir::dom::dominators(func);
    let mut raw = helix_ir::dom::natural_loops(func, &doms);

    // Outer first: only a strictly larger body can contain ours, so every
    // potential parent is already placed when a loop is processed. (Sorting
    // ascending here flattened the forest — every nested loop came out at
    // depth 1 because the containment search only ever saw smaller loops.)
    raw.sort_by_key(|(_, body)| std::cmp::Reverse(body.len()));

    let mut out: Vec<Loop> = Vec::new();
    for (i, (header, body)) in raw.into_iter().enumerate() {
        // Parent = the innermost already-placed loop whose body strictly
        // contains ours and whose header differs. `out` runs outer→inner,
        // so the LAST hit is the smallest container: the immediate parent.
        let parent = out.iter().rposition(|cand| {
            cand.header != header && body.iter().all(|b| cand.blocks.contains(b))
        });
        let depth = parent.map_or(1, |p| out[p].depth + 1);
        out.push(Loop {
            id: i,
            header,
            blocks: body,
            depth,
            parent,
        });
    }
    LoopInfo { loops: out }
}
