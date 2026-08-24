//! Unit tests for reduction recognition over hand-built SSA fixtures.
//!
//! `helix_ir::testutil` is only compiled into helix-ir's own test builds
//! (`#[cfg(test)]`), so these tests construct minimal loop CFGs directly.
//! Each fixture mirrors the exact shape the real builder emits post-SSA:
//! a separate induction-variable φ feeding the header condition, plus the
//! candidate accumulator φ whose back-edge arm flows through a chain that
//! the renamer spells self-referentially.

#![forbid(unsafe_code)]

use helix_analysis::ReductionOp;
use helix_analysis::reduce::find_reductions;
use helix_ir::{BinOp, BlockId, Constant, FuncIr, Inst, LocalId, Phi, Term, ValueId};

const IV: LocalId = LocalId(0);
const ACC: LocalId = LocalId(1);

/// Canonical single-loop fixture with two header phis (iv + accumulator):
///
/// ```text
/// bb0: entry  → jump bb1 [iv0, acc_cell]
/// bb1: iv  = φ(bb0: cell0, bb4: inc)     ← reads nothing but its own arms
///      acc = φ(bb0: cell1, bb4: chain)
///      cond = iv < N ; branch bb2 / bb3
/// bb2: body   → jump bb4                 ← body instructions pushed by tests
/// bb4: latch  inc = inc + 1 ; jump bb1 [inc, chain]
/// bb3: exit   return
/// ```
struct Fixture {
    ir: FuncIr,
    /// The chain value id (accumulator's back-edge arm).
    chain: ValueId,
    /// The accumulator φ result id.
    acc_phi: ValueId,
    /// Next fresh id for test bodies.
    next: u32,
}

impl Fixture {
    fn new(name: &str) -> Self {
        let mut ir = FuncIr::new(name, helix_sema::Ty::Unit, 2);
        ir.declare_local(IV, helix_sema::Ty::I64, "i");
        ir.declare_local(ACC, helix_sema::Ty::I64, "acc");
        while ir.blocks.len() < 5 {
            ir.new_block();
        }
        // Ids are assigned deterministically so tests can reason about them.
        let chain = ValueId(10);
        let acc_phi = ValueId(11);
        let one = ValueId(12);
        let inc = ValueId(13);

        ir.set_term(
            BlockId(0),
            Term::Jump(BlockId(1), vec![ValueId(0), ValueId(1)]),
        );
        ir.block_mut(BlockId(1)).phis.push(Phi {
            dst: ValueId(0),
            var: IV,
            args: vec![(BlockId(0), ValueId(0)), (BlockId(4), inc)],
        });
        ir.block_mut(BlockId(1)).phis.push(Phi {
            dst: acc_phi,
            var: ACC,
            args: vec![(BlockId(0), ValueId(1)), (BlockId(4), chain)],
        });
        // Header condition reads ONLY the iv phi (as in real IR).
        let bound = ValueId(14);
        ir.block_mut(BlockId(1)).insts.push(Inst::Const {
            dst: bound,
            c: Constant::I64(16),
        });
        let cond = ValueId(15);
        ir.block_mut(BlockId(1)).insts.push(Inst::Bin {
            op: BinOp::Lt,
            dst: cond,
            a: ValueId(0),
            b: bound,
        });
        ir.set_term(
            BlockId(1),
            Term::Branch {
                cond,
                t: BlockId(2),
                f: BlockId(3),
            },
        );
        // Latch: inc = inc + 1; forward [inc, chain] into the header phis.
        ir.block_mut(BlockId(4)).insts.push(Inst::Const {
            dst: one,
            c: Constant::I64(1),
        });
        ir.block_mut(BlockId(4)).insts.push(Inst::Bin {
            op: BinOp::Add,
            dst: inc,
            a: inc,
            b: one,
        });
        ir.set_term(BlockId(4), Term::Jump(BlockId(1), vec![inc, chain]));
        ir.set_term(BlockId(3), Term::Return(None));
        ir.recompute_edges();

        Fixture {
            ir,
            chain,
            acc_phi,
            next: 16,
        }
    }

    fn body(&mut self) -> &mut Vec<Inst> {
        &mut self.ir.block_mut(BlockId(2)).insts
    }

    /// Emit `t = const k` and return `t`.
    fn konst(&mut self, k: i64) -> ValueId {
        let t = ValueId(self.next);
        self.next += 1;
        self.body().push(Inst::Const {
            dst: t,
            c: Constant::I64(k),
        });
        t
    }

    /// The canonical accumulation chain over the ACCUMULATOR CELL spelling:
    /// `chain = chain OP t` (self-referential, as the renamer leaves it when
    /// read and write share the source statement).
    fn chain_op(&mut self, op: BinOp, t: ValueId) {
        let chain = self.chain;
        self.body().push(Inst::Bin {
            op,
            dst: chain,
            a: chain,
            b: t,
        });
    }

    fn build(self) -> FuncIr {
        self.ir
    }
}

const LOOP_BLOCKS: [BlockId; 3] = [BlockId(1), BlockId(2), BlockId(4)];

fn recognize(ir: &FuncIr) -> Vec<(u32, ReductionOp)> {
    find_reductions(ir, &LOOP_BLOCKS, &[IV])
        .iter()
        .map(|r| (r.var.0, r.op))
        .collect()
}

// ---------------------------------------------------------------------------
// Positives
// ---------------------------------------------------------------------------

#[test]
fn add_chain_self_referential_spelling_is_recognized() {
    let mut fx = Fixture::new("add_chain");
    let t = fx.konst(5);
    fx.chain_op(BinOp::Add, t);
    let ir = fx.build();
    assert_eq!(recognize(&ir), vec![(ACC.0, ReductionOp::Add)]);
}

#[test]
fn sub_chain_folds_into_add() {
    let mut fx = Fixture::new("sub_chain");
    let t = fx.konst(5);
    fx.chain_op(BinOp::Sub, t);
    let ir = fx.build();
    assert_eq!(recognize(&ir), vec![(ACC.0, ReductionOp::Add)]);
}

#[test]
fn mul_chain_commutative_spelling_is_recognized() {
    // t * acc instead of acc * t — order must not matter.
    let mut fx = Fixture::new("mul_chain");
    let t = fx.konst(3);
    let chain = fx.chain;
    fx.body().push(Inst::Bin {
        op: BinOp::Mul,
        dst: chain,
        a: t,
        b: chain,
    });
    let ir = fx.build();
    assert_eq!(recognize(&ir), vec![(ACC.0, ReductionOp::Mul)]);
}

#[test]
fn min_call_with_accumulator_arg_is_recognized() {
    let mut fx = Fixture::new("min_call");
    let t = fx.konst(7);
    let (chain, acc_phi) = (fx.chain, fx.acc_phi);
    fx.body().push(Inst::Call(helix_ir::Call {
        dst: Some(chain),
        callee: "min".into(),
        args: vec![acc_phi, t],
        arr_refs: Vec::new(),
    }));
    let ir = fx.build();
    assert_eq!(recognize(&ir), vec![(ACC.0, ReductionOp::Min)]);
}

#[test]
fn max_call_with_accumulator_arg_is_recognized() {
    let mut fx = Fixture::new("max_call");
    let t = fx.konst(7);
    let (chain, acc_phi) = (fx.chain, fx.acc_phi);
    fx.body().push(Inst::Call(helix_ir::Call {
        dst: Some(chain),
        callee: "max".into(),
        args: vec![t, acc_phi],
        arr_refs: Vec::new(),
    }));
    let ir = fx.build();
    assert_eq!(recognize(&ir), vec![(ACC.0, ReductionOp::Max)]);
}

// ---------------------------------------------------------------------------
// Negatives
// ---------------------------------------------------------------------------

#[test]
fn stray_read_of_accumulator_vetoes() {
    let mut fx = Fixture::new("stray_read");
    let t = fx.konst(5);
    fx.chain_op(BinOp::Add, t);
    let stray = ValueId(fx.next);
    fx.next += 1;
    let acc_phi = fx.acc_phi;
    fx.body().push(Inst::Bin {
        op: BinOp::Mul,
        dst: stray,
        a: acc_phi, // second consumer of the accumulator
        b: t,
    });
    let ir = fx.build();
    assert!(recognize(&ir).is_empty(), "stray read must veto");
}

#[test]
fn both_operands_accumulator_vetoes() {
    // x = x * x has no independent term to combine.
    let mut fx = Fixture::new("square");
    let chain = fx.chain;
    fx.body().push(Inst::Bin {
        op: BinOp::Mul,
        dst: chain,
        a: chain,
        b: chain,
    });
    let ir = fx.build();
    assert!(recognize(&ir).is_empty());
}

#[test]
fn non_associative_division_vetoes() {
    let mut fx = Fixture::new("div_chain");
    let t = fx.konst(2);
    fx.chain_op(BinOp::Div, t);
    let ir = fx.build();
    assert!(recognize(&ir).is_empty());
}

#[test]
fn call_without_accumulator_arg_vetoes() {
    let mut fx = Fixture::new("call_no_acc");
    let t = ValueId(fx.next);
    fx.next += 1;
    let u = ValueId(fx.next);
    fx.next += 1;
    fx.body().push(Inst::Const {
        dst: t,
        c: Constant::I64(1),
    });
    fx.body().push(Inst::Const {
        dst: u,
        c: Constant::I64(2),
    });
    let chain = fx.chain;
    fx.body().push(Inst::Call(helix_ir::Call {
        dst: Some(chain),
        callee: "min".into(),
        args: vec![t, u], // no accumulator anywhere
        arr_refs: Vec::new(),
    }));
    let ir = fx.build();
    assert!(recognize(&ir).is_empty());
}

#[test]
fn non_builtin_callee_vetoes() {
    let mut fx = Fixture::new("call_other");
    let _t = fx.konst(1);
    let (chain, acc_phi) = (fx.chain, fx.acc_phi);
    fx.body().push(Inst::Call(helix_ir::Call {
        dst: Some(chain),
        callee: "sqrt".into(), // associative? no; also unary
        args: vec![acc_phi],
        arr_refs: Vec::new(),
    }));
    let ir = fx.build();
    assert!(recognize(&ir).is_empty());
}

#[test]
fn excluded_iv_never_reported_even_when_additively_shaped() {
    // The iv φ is additively shaped by construction (its back arm is the
    // `inc = inc + 1` chain); in this fixture its only remaining use is the
    // header condition, so without exclusion it is rejected by extra_reads,
    // not by exclusion itself. The meaningful assertion is the negative one:
    // with `IV` excluded, no result ever names l0.
    fn build_fixture() -> FuncIr {
        Fixture::new("iv_excluded").build()
    }
    let all = find_reductions(&build_fixture(), &LOOP_BLOCKS, &[]);
    assert!(
        all.iter().all(|r| r.op == ReductionOp::Add || r.var != IV),
        "only additive shapes exist in this fixture"
    );
    let none = find_reductions(&build_fixture(), &LOOP_BLOCKS, &[IV]);
    assert!(none.iter().all(|r| r.var != IV));
}
