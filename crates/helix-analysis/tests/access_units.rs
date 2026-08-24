//! Unit tests for affine subscript extraction ([`helix_analysis::access`]).
//!
//! Uses the same hand-built loop fixture style as `reduce_units.rs`: a
//! canonical loop whose body loads/stores arrays through index expressions of
//! increasing complexity. Each test pins what the classifier must derive —
//! and, equally important, what it must refuse.

#![forbid(unsafe_code)]

use helix_analysis::access::collect;
use helix_ir::{BinOp, BlockId, Constant, FuncIr, Inst, LocalId, Phi, Term, ValueId};

const IV: LocalId = LocalId(0);
/// The induction value's SSA id inside the fixture (deterministic layout).
const IV_VALUE: ValueId = ValueId(11);

/// Canonical single-loop fixture with a pluggable body:
///
/// ```text
/// bb0: jump bb1 [iv0]
/// bb1: iv = φ(bb0: cell0, bb4: inc); cond = iv < N; branch bb2 / bb3
/// bb2: body (pushed by tests) → jump bb4
/// bb4: inc = inc + 1; jump bb1 [inc]
/// ```
struct Fixture {
    ir: FuncIr,
    next: u32,
}

impl Fixture {
    fn new(name: &str) -> Self {
        let mut ir = FuncIr::new(name, helix_sema::Ty::Unit, 3);
        ir.declare_local(IV, helix_sema::Ty::I64, "i");
        ir.declare_local(
            LocalId(1),
            helix_sema::Ty::Array(helix_sema::ElemTy::I64),
            "a",
        );
        ir.declare_local(
            LocalId(2),
            helix_sema::Ty::Array(helix_sema::ElemTy::I64),
            "b",
        );
        while ir.blocks.len() < 5 {
            ir.new_block();
        }
        let one = ValueId(12);
        let inc = ValueId(13);
        let bound = ValueId(14);
        let cond = ValueId(15);

        ir.set_term(BlockId(0), Term::Jump(BlockId(1), vec![ValueId(0)]));
        ir.block_mut(BlockId(1)).phis.push(Phi {
            dst: ValueId(0),
            var: IV,
            args: vec![(BlockId(0), ValueId(0)), (BlockId(4), inc)],
        });
        ir.block_mut(BlockId(1)).insts.push(Inst::Const {
            dst: bound,
            c: Constant::I64(16),
        });
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
        // The body block MUST get its terminator or the back edge (and hence
        // the loop) disappears; tests push instructions into bb2 before the
        // edge lists are recomputed in `build`.
        ir.set_term(BlockId(2), Term::Jump(BlockId(4), vec![]));
        ir.set_term(BlockId(4), Term::Jump(BlockId(1), vec![inc]));
        ir.set_term(BlockId(3), Term::Return(None));
        ir.recompute_edges();
        Fixture { ir, next: 16 }
    }

    fn body(&mut self) -> &mut Vec<Inst> {
        &mut self.ir.block_mut(BlockId(2)).insts
    }

    /// `t = k`; returns `t`.
    fn konst(&mut self, k: i64) -> ValueId {
        let v = ValueId(self.next);
        self.next += 1;
        self.body().push(Inst::Const {
            dst: v,
            c: Constant::I64(k),
        });
        v
    }

    /// `v = x OP y`; returns `v`.
    fn bin(&mut self, op: BinOp, x: ValueId, y: ValueId) -> ValueId {
        let v = ValueId(self.next);
        self.next += 1;
        self.body().push(Inst::Bin {
            op,
            dst: v,
            a: x,
            b: y,
        });
        v
    }

    /// `dst = arr[idx]`; returns `dst`.
    fn load(&mut self, arr: u32, idx: ValueId) -> ValueId {
        let v = ValueId(self.next);
        self.next += 1;
        self.body().push(Inst::Load(helix_ir::Load {
            dst: v,
            arr: LocalId(arr),
            idx,
        }));
        v
    }

    /// `arr[idx] = val`
    fn store(&mut self, arr: u32, idx: ValueId, val: ValueId) {
        self.body().push(Inst::Store {
            arr: LocalId(arr),
            idx,
            val,
        });
    }

    /// Recompute edge lists after body instructions landed, then finish.
    fn build(mut self) -> FuncIr {
        self.ir.recompute_edges();
        self.ir
    }
}

/// One classified access: `(is_write, (a, b)?, array_label)`.
type Classified = (bool, Option<(i128, i128)>, String);

/// Run the classifier over the fixture's only loop; results in program order.
fn classify_all(ir: &FuncIr) -> Vec<Classified> {
    let li = helix_analysis::find_loops(ir);
    assert_eq!(li.loops.len(), 1, "fixture has exactly one loop");
    let lp = &li.loops[0];
    collect(ir, lp, IV_VALUE)
        .into_iter()
        .map(|a| {
            (
                a.is_write,
                a.affine.map(|f| (f.a, f.b)),
                format!("l{}", a.arr.0),
            )
        })
        .collect()
}

#[test]
fn direct_index_is_unit_affine() {
    let mut fx = Fixture::new("direct");
    let v = fx.konst(9);
    fx.store(1, IV_VALUE, v); // a[i] = 9
    let res = classify_all(&fx.build());
    assert_eq!(res.len(), 1);
    assert!(res[0].0, "is write");
    assert_eq!(res[0].1, Some((1, 0)));
}

#[test]
fn constant_offset_index() {
    let mut fx = Fixture::new("offset");
    let one = fx.konst(1);
    let idx = fx.bin(BinOp::Sub, IV_VALUE, one); // i - 1
    let val = fx.load(1, idx);
    fx.store(2, IV_VALUE, val);
    let res = classify_all(&fx.build());
    assert_eq!(res[0].1, Some((1, -1)), "read a[i-1]");
    assert_eq!(res[1].1, Some((1, 0)), "write b[i]");
}

#[test]
fn stride_two_coefficient() {
    let mut fx = Fixture::new("stride");
    let two = fx.konst(2);
    let idx = fx.bin(BinOp::Mul, two, IV_VALUE); // 2*i — commuted spelling
    let v = fx.konst(1);
    fx.store(1, idx, v);
    let res = classify_all(&fx.build());
    assert_eq!(res[0].1, Some((2, 0)));
}

#[test]
fn invariant_literal_index_keeps_value() {
    let mut fx = Fixture::new("invariant_lit");
    let c = fx.konst(7);
    let v = fx.load(1, c); // a[7]
    fx.store(2, IV_VALUE, v);
    let res = classify_all(&fx.build());
    assert_eq!(res[0].1, Some((0, 7)), "literal invariant stays exact");
}

#[test]
fn invariant_arithmetic_folds_exactly() {
    let mut fx = Fixture::new("invariant_fold");
    let five = fx.konst(5);
    let idx = fx.bin(BinOp::Add, five, five); // 10, loop-invariant
    let v = fx.load(1, idx);
    fx.store(2, IV_VALUE, v);
    let res = classify_all(&fx.build());
    assert_eq!(res[0].1, Some((0, 10)));
}

#[test]
fn load_result_index_refused() {
    // a[i] read first (affine), then b[a[i]] — the SECOND access's index is
    // a load result and must be refused. The first access is affine {1,0}.
    let mut fx = Fixture::new("load_idx");
    let j = fx.load(2, IV_VALUE);
    let _v = fx.load(1, j);
    let res = classify_all(&fx.build());
    assert_eq!(res[0].1, Some((1, 0)), "b[i] itself classifies");
    assert_eq!(res[1].1, None, "a[b[i]]: index from a load must be None");
}

#[test]
fn quadratic_index_refused() {
    // i * i — the product of two varying factors is not affine.
    let mut fx = Fixture::new("quad");
    let idx = fx.bin(BinOp::Mul, IV_VALUE, IV_VALUE);
    let v = fx.konst(1);
    fx.store(1, idx, v);
    let res = classify_all(&fx.build());
    assert_eq!(res[0].1, None);
}

#[test]
fn negated_index_flips_sign() {
    // -i spelled as 0 - i.
    let mut fx = Fixture::new("neg");
    let zero = fx.konst(0);
    let neg = fx.bin(BinOp::Sub, zero, IV_VALUE);
    let _v = fx.load(1, neg);
    let res = classify_all(&fx.build());
    assert_eq!(res[0].1, Some((-1, 0)));
}

#[test]
fn mixed_index_plus_invariant_offset() {
    // i + K where K is an invariant literal: affine with shifted base.
    let mut fx = Fixture::new("shifted");
    let k = fx.konst(40);
    let idx = fx.bin(BinOp::Add, IV_VALUE, k);
    let _v = fx.load(1, idx);
    let res = classify_all(&fx.build());
    assert_eq!(res[0].1, Some((1, 40)));
}

#[test]
fn accesses_reported_in_program_order() {
    let mut fx = Fixture::new("order");
    let one = fx.konst(1);
    let idx = fx.bin(BinOp::Add, IV_VALUE, one);
    let v1 = fx.load(1, IV_VALUE);
    let v2 = fx.load(1, idx);
    let sum = fx.bin(BinOp::Add, v1, v2);
    fx.store(2, IV_VALUE, sum);
    let res = classify_all(&fx.build());
    assert_eq!(res.len(), 3);
    assert_eq!(res[0].2, "l1");
    assert_eq!(res[1].2, "l1");
    assert_eq!(res[2].2, "l2");
    assert!(!res[0].0 && !res[1].0 && res[2].0, "reads precede write");
}
