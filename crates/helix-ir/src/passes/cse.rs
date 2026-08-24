//! Common-subexpression elimination via dominator-scoped value numbering.
//!
//! Post-SSA, two pure instructions are interchangeable when they compute the
//! same operation on the same operands *and* one dominates the other. The
//! pass walks the dominator tree in preorder carrying a hash map from
//! operation signature `(opcode, operand ids)` to the first defining value;
//! because the walk is a dominator-tree preorder, any later occurrence found
//! while the leader is still in the table is genuinely dominated by it — the
//! table *is* the dominance scope. Only `Bin`/`Unary`/`Cast` participate (plus
//! constant deduplication); memory operations never do — loads may trap,
//! stores/calls have effects.

use std::collections::HashMap;

use crate::dom::dominators;
use crate::ir::{BlockId, FuncIr, Inst, ValueId};
use crate::passmod::ChangeFlag;

/// Eliminate dominated duplicate computations.
///
/// Constants are handled by the same dominator-scoped walk as everything
/// else (a `Const` has a signature), which keeps the transform sound: a
/// duplicate is only replaced when the dominator-tree preorder guarantees
/// its leader dominates it. A naive "first def wins globally" pass would be
/// WRONG — two identical constants in sibling subtrees have no dominance
/// relation, and rewriting one to the other fabricates a value.
pub fn cse(ir: &mut FuncIr) -> ChangeFlag {
    let mut flag = ChangeFlag::new();

    // ---- Dominator-scoped value numbering over Const/Bin/Unary/Cast --------
    let doms = dominators(ir);
    type Table = HashMap<String, ValueId>;

    // Iterative preorder with explicit scope save/restore: each frame
    // snapshots the table on entry and restores it after its subtree finishes.
    let mut table: Table = HashMap::new();
    let mut dead: Vec<(u32, usize)> = Vec::new();

    struct Frame {
        block: BlockId,
        saved: Table,
    }
    let mut stack: Vec<Frame> = vec![Frame {
        block: BlockId(0),
        saved: table.clone(),
    }];

    while let Some(frame) = stack.pop() {
        // Enter: restore the parent's scope (siblings don't leak into us),
        // then process this block.
        table = frame.saved;
        let bi = frame.block.0 as usize;

        let mut block_dead: Vec<(usize, ValueId)> = Vec::new();
        for (ii, inst) in ir.blocks[bi].insts.iter().enumerate() {
            if let Some(sig) = signature(inst) {
                match table.get(&sig.0) {
                    // Leader exists and differs: the dominator-tree preorder
                    // guarantees it dominates us → this occurrence is dead.
                    Some(leader) if *leader != sig.1 => {
                        block_dead.push((ii, *leader));
                    }
                    _ => {
                        table.insert(sig.0, sig.1);
                    }
                }
            }
        }
        if !block_dead.is_empty() {
            flag.changed = true;
        }

        // Record children BEFORE mutating anything they might read.
        let kids = doms.tree_children()[bi].clone();

        for (ii, leader) in block_dead {
            if let Some(dst) = ir.blocks[bi].insts[ii].dst() {
                ir.replace_all_uses(dst, leader);
                dead.push((bi as u32, ii));
            }
        }

        for child in kids {
            stack.push(Frame {
                block: child,
                saved: table.clone(),
            });
        }
    }

    // Delete the now-unused duplicates (indices recorded per block).
    let mut per_block: HashMap<u32, Vec<usize>> = HashMap::new();
    for (bi, ii) in dead {
        per_block.entry(bi).or_default().push(ii);
    }
    for (bi, idxs) in per_block {
        let old = std::mem::take(&mut ir.blocks[bi as usize].insts);
        ir.blocks[bi as usize].insts = old
            .into_iter()
            .enumerate()
            .filter(|(k, _)| !idxs.contains(k))
            .map(|(_, i)| i)
            .collect();
    }

    flag
}

/// Operation signature of a pure instruction: stable string + defining value.
fn signature(inst: &Inst) -> Option<(String, ValueId)> {
    match inst {
        Inst::Const { dst, c } => Some((format!("const {c:?}"), *dst)),
        Inst::Bin { op, dst, a, b } => Some((format!("bin {op:?} {} {}", a.0, b.0), *dst)),
        Inst::Unary { op, dst, a } => Some((format!("unary {op:?} {}", a.0), *dst)),
        Inst::Cast { dst, val, to } => Some((format!("cast {} {:?}", val.0, to), *dst)),
        _ => None,
    }
}
