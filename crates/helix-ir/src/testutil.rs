//! Hand-built CFG fixtures for downstream crates' unit tests.
//!
//! These construct canonical control shapes directly (no parser/sema round
//! trip) so that `helix-analysis` and the backend can test their analyses
//! against known topologies: a single counting loop and a two-level nest.
//! Every block gets a real terminator and the structural edge lists are
//! repaired with [`FuncIr::recompute_edges`], so the fixtures pass
//! [`crate::verify`] unchanged.

use helix_syntax::ast::BinOp;

use crate::ir::{BlockId, Constant, FuncIr, Inst, Term, ValueId};

/// A classic counting loop:
///
/// ```text
/// bb0 (entry) ──▶ bb1 (header) ──▶ bb2 (body) ──▶ bb3 (latch) ──▶ bb1 …
///                     │
///                     └──▶ bb4 (exit, return)
/// ```
///
/// The header compares the iv cell against a constant bound; the latch
/// increments it. `iv` lives in local slot 0, spelled as cell `ValueId(0)`
/// pre-renaming — callers that need SSA form can run [`crate::to_ssa`].
#[must_use]
pub fn counting_loop() -> FuncIr {
    let mut ir = FuncIr::new("counting_loop", helix_sema::Ty::Unit, 1);
    ir.declare_local(crate::ir::LocalId(0), helix_sema::Ty::I64, "i");
    while ir.blocks.len() < 5 {
        ir.new_block();
    }
    let iv = ValueId(0);

    // bb0: entry — i = 0; jump header.
    let zero = ir.new_value(helix_sema::Ty::I64);
    ir.block_mut(BlockId(0)).insts.push(Inst::Const {
        dst: zero,
        c: Constant::I64(0),
    });
    ir.set_term(BlockId(0), Term::Jump(BlockId(1), Vec::new()));

    // bb1: header — cond = i < n; branch(body, exit).
    let bound = ir.new_value(helix_sema::Ty::I64);
    ir.block_mut(BlockId(1)).insts.push(Inst::Const {
        dst: bound,
        c: Constant::I64(8),
    });
    let cond = ir.new_value(helix_sema::Ty::Bool);
    ir.block_mut(BlockId(1)).insts.push(Inst::Bin {
        op: BinOp::Lt,
        dst: cond,
        a: iv,
        b: bound,
    });
    ir.set_term(
        BlockId(1),
        Term::Branch {
            cond,
            t: BlockId(2),
            f: BlockId(4),
        },
    );

    // bb2: body — store-like effect placeholder (a call keeps DCE honest).
    ir.block_mut(BlockId(2))
        .insts
        .push(Inst::Call(crate::ir::Call {
            dst: None,
            callee: "print".into(),
            args: vec![iv],
            arr_refs: Vec::new(),
        }));
    ir.set_term(BlockId(2), Term::Jump(BlockId(3), Vec::new()));

    // bb3: latch — i = i + 1; jump header.
    let one = ir.new_value(helix_sema::Ty::I64);
    ir.block_mut(BlockId(3)).insts.push(Inst::Const {
        dst: one,
        c: Constant::I64(1),
    });
    let next = ir.new_value(helix_sema::Ty::I64);
    ir.block_mut(BlockId(3)).insts.push(Inst::Bin {
        op: BinOp::Add,
        dst: next,
        a: iv,
        b: one,
    });
    ir.set_term(BlockId(3), Term::Jump(BlockId(1), Vec::new()));

    // bb4: exit.
    ir.set_term(BlockId(4), Term::Return(None));

    ir.recompute_edges();
    ir
}

/// A two-level nested counting loop sharing the same shape rules as
/// [`counting_loop`]:
///
/// ```text
/// bb0 ─▶ bb1 (outer hdr) ─▶ bb2 (inner pre) ─▶ bb3 (inner hdr) ─▶ bb4 (body)
///            │                    │                  │              │
///            │                    │                  │              ▼
///            │                    │               bb6 (latch) ──▶ bb3
///            │                    ▼
///            │                 bb5 (outer body tail) ──▶ bb7 (outer latch) ─▶ bb1
///            ▼
///         bb8 (exit)
/// ```
#[must_use]
pub fn nested_loops() -> FuncIr {
    let mut ir = FuncIr::new("nested_loops", helix_sema::Ty::Unit, 2);
    ir.declare_local(crate::ir::LocalId(0), helix_sema::Ty::I64, "i");
    ir.declare_local(crate::ir::LocalId(1), helix_sema::Ty::I64, "j");
    while ir.blocks.len() < 9 {
        ir.new_block();
    }
    let iv = ValueId(0); // outer i
    let jv = ValueId(1); // inner j

    // bb0: entry.
    let z0 = ir.new_value(helix_sema::Ty::I64);
    ir.block_mut(BlockId(0)).insts.push(Inst::Const {
        dst: z0,
        c: Constant::I64(0),
    });
    ir.set_term(BlockId(0), Term::Jump(BlockId(1), Vec::new()));

    // bb1: outer header.
    let b1 = ir.new_value(helix_sema::Ty::I64);
    ir.block_mut(BlockId(1)).insts.push(Inst::Const {
        dst: b1,
        c: Constant::I64(8),
    });
    let c1 = ir.new_value(helix_sema::Ty::Bool);
    ir.block_mut(BlockId(1)).insts.push(Inst::Bin {
        op: BinOp::Lt,
        dst: c1,
        a: iv,
        b: b1,
    });
    ir.set_term(
        BlockId(1),
        Term::Branch {
            cond: c1,
            t: BlockId(2),
            f: BlockId(8),
        },
    );

    // bb2: inner preheader — j = 0.
    let zj = ir.new_value(helix_sema::Ty::I64);
    ir.block_mut(BlockId(2)).insts.push(Inst::Const {
        dst: zj,
        c: Constant::I64(0),
    });
    ir.set_term(BlockId(2), Term::Jump(BlockId(3), Vec::new()));

    // bb3: inner header.
    let bj = ir.new_value(helix_sema::Ty::I64);
    ir.block_mut(BlockId(3)).insts.push(Inst::Const {
        dst: bj,
        c: Constant::I64(8),
    });
    let cj = ir.new_value(helix_sema::Ty::Bool);
    ir.block_mut(BlockId(3)).insts.push(Inst::Bin {
        op: BinOp::Lt,
        dst: cj,
        a: jv,
        b: bj,
    });
    ir.set_term(
        BlockId(3),
        Term::Branch {
            cond: cj,
            t: BlockId(4),
            f: BlockId(5),
        },
    );

    // bb4: inner body — print(i).
    ir.block_mut(BlockId(4))
        .insts
        .push(Inst::Call(crate::ir::Call {
            dst: None,
            callee: "print".into(),
            args: vec![iv],
            arr_refs: Vec::new(),
        }));
    ir.set_term(BlockId(4), Term::Jump(BlockId(6), Vec::new()));

    // bb5: outer body tail (after inner exit).
    ir.set_term(BlockId(5), Term::Jump(BlockId(7), Vec::new()));

    // bb6: inner latch — j += 1.
    let onej = ir.new_value(helix_sema::Ty::I64);
    ir.block_mut(BlockId(6)).insts.push(Inst::Const {
        dst: onej,
        c: Constant::I64(1),
    });
    let nj = ir.new_value(helix_sema::Ty::I64);
    ir.block_mut(BlockId(6)).insts.push(Inst::Bin {
        op: BinOp::Add,
        dst: nj,
        a: jv,
        b: onej,
    });
    ir.set_term(BlockId(6), Term::Jump(BlockId(3), Vec::new()));

    // bb7: outer latch — i += 1.
    let onei = ir.new_value(helix_sema::Ty::I64);
    ir.block_mut(BlockId(7)).insts.push(Inst::Const {
        dst: onei,
        c: Constant::I64(1),
    });
    let ni = ir.new_value(helix_sema::Ty::I64);
    ir.block_mut(BlockId(7)).insts.push(Inst::Bin {
        op: BinOp::Add,
        dst: ni,
        a: iv,
        b: onei,
    });
    ir.set_term(BlockId(7), Term::Jump(BlockId(1), Vec::new()));

    // bb8: exit.
    ir.set_term(BlockId(8), Term::Return(None));

    ir.recompute_edges();
    ir
}
