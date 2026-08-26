//! Verifier regression tests:
//!
//! 1. Duplicate definitions must fail `verify_ssa` (the SSA seams), while
//!    plain `verify` keeps tolerating them pre-SSA — and dominance checks
//!    really run once the IR *is* singly-defined.
//! 2. Builtin calls get conservative arity/type checks (`min`/`max` binary,
//!    `abs`/`sqrt`/`len` unary, scalar arguments), and unknown-callee args
//!    must still carry a side-table type row.
//! 3. Real lowering (including `len`'s array-by-reference argument and
//!    `zeros`) still verifies after `to_ssa`.

use helix_ir::{BinOp, BlockId, Call, Constant, FuncIr, Inst, Term, ValueId, verify, verify_ssa};
use helix_sema::{ElemTy, Ty};

fn compile(src: &str) -> Vec<helix_ir::FuncIr> {
    let ast = helix_syntax::parse_str(src).expect("parse");
    let typed = helix_sema::check(&ast).expect("sema");
    helix_ir::build(&typed)
}

// ---------------------------------------------------------------------------
// 1. duplicate defs and the dominance gate
// ---------------------------------------------------------------------------

#[test]
fn duplicate_def_fails_verify_ssa_but_plain_verify_tolerates_it() {
    // Pre-SSA cell spellings redefine their id legally; the SSA seams must
    // reject the same shape.
    let mut ir = FuncIr::new("dup_def", Ty::Unit, 0);
    let v = ir.new_value(Ty::I64);
    ir.block_mut(BlockId(0)).insts.push(Inst::Const {
        dst: v,
        c: Constant::I64(1),
    });
    ir.block_mut(BlockId(0)).insts.push(Inst::Const {
        dst: v,
        c: Constant::I64(2),
    });
    ir.set_term(BlockId(0), Term::Return(None));

    verify(&ir).unwrap_or_else(|e| panic!("pre-SSA redefinition must pass plain verify: {e}"));
    let err = verify_ssa(&ir).expect_err("duplicate def must fail verify_ssa");
    assert!(err.contains("defined more than once"), "unexpected: {err}");
}

#[test]
fn undominated_use_fails_verify_once_defs_are_unique() {
    // SSA-shaped (every id defined once) => the dominance gate opens and must
    // catch a use in bb0 of a value defined in bb1.
    let mut ir = FuncIr::new("late_def", Ty::Unit, 0);
    ir.new_block();
    let v = ir.new_value(Ty::I64);
    let w = ir.new_value(Ty::I64);
    ir.block_mut(BlockId(0)).insts.push(Inst::Bin {
        op: BinOp::Add,
        dst: w,
        a: v,
        b: v,
    });
    ir.set_term(BlockId(0), Term::Jump(BlockId(1), Vec::new()));
    ir.block_mut(BlockId(1)).insts.push(Inst::Const {
        dst: v,
        c: Constant::I64(0),
    });
    ir.set_term(BlockId(1), Term::Return(None));

    let err = verify(&ir).expect_err("use before a dominating def must fail");
    assert!(err.contains("does not dominate"), "unexpected: {err}");
}

#[test]
fn dominated_use_passes_verify_on_ssa_shaped_ir() {
    // Same shape with the def hoisted into the entry block: dominance is
    // satisfied and verification succeeds.
    let mut ir = FuncIr::new("early_def", Ty::Unit, 0);
    ir.new_block();
    let v = ir.new_value(Ty::I64);
    let w = ir.new_value(Ty::I64);
    ir.block_mut(BlockId(0)).insts.push(Inst::Const {
        dst: v,
        c: Constant::I64(0),
    });
    ir.block_mut(BlockId(0)).insts.push(Inst::Bin {
        op: BinOp::Add,
        dst: w,
        a: v,
        b: v,
    });
    ir.set_term(BlockId(0), Term::Jump(BlockId(1), Vec::new()));
    ir.set_term(BlockId(1), Term::Return(None));

    verify(&ir).unwrap_or_else(|e| panic!("well-formed IR rejected: {e}"));
}

// ---------------------------------------------------------------------------
// 2. builtin call checks
// ---------------------------------------------------------------------------

/// Single-block fixture: constants of `arg_tys`, one call, unit return.
fn call_fixture(callee: &str, arg_tys: &[Ty]) -> FuncIr {
    let mut ir = FuncIr::new("calls", Ty::Unit, 0);
    let args: Vec<ValueId> = arg_tys.iter().map(|t| ir.new_value(*t)).collect();
    for (k, v) in args.iter().enumerate() {
        ir.block_mut(BlockId(0)).insts.push(Inst::Const {
            dst: *v,
            c: Constant::I64(k as i64),
        });
    }
    let dst = ir.new_value(Ty::I64);
    ir.block_mut(BlockId(0)).insts.push(Inst::Call(Call {
        dst: Some(dst),
        callee: callee.to_string(),
        args,
        arr_refs: Vec::new(),
    }));
    ir.set_term(BlockId(0), Term::Return(None));
    ir
}

#[test]
fn min_with_two_scalar_args_verifies() {
    let ir = call_fixture("min", &[Ty::I64, Ty::F64]);
    verify(&ir).unwrap_or_else(|e| panic!("well-formed min call rejected: {e}"));
}

#[test]
fn max_with_wrong_arity_fails() {
    let ir = call_fixture("max", &[Ty::I64]);
    let err = verify(&ir).expect_err("max with 1 arg must fail");
    assert!(
        err.contains("call to max takes 2 arg(s)"),
        "imprecise message: {err}"
    );
}

#[test]
fn abs_rejects_non_scalar_argument() {
    let ir = call_fixture("abs", &[Ty::Array(ElemTy::I64)]);
    let err = verify(&ir).expect_err("abs over an array must fail");
    assert!(err.contains("takes scalar args"), "imprecise: {err}");
}

#[test]
fn sqrt_with_wrong_arity_fails() {
    let ir = call_fixture("sqrt", &[]);
    let err = verify(&ir).expect_err("sqrt with 0 args must fail");
    assert!(
        err.contains("call to sqrt takes 1 arg(s)"),
        "imprecise message: {err}"
    );
}

#[test]
fn unknown_callee_skips_arity_but_requires_type_rows() {
    // A user-function call: arity is unknowable here, yet every argument must
    // still have a side-table row (checked upstream by the type-consistency
    // walk, which this pins down end to end).
    let mut ir = FuncIr::new("user_call", Ty::Unit, 0);
    let x = ValueId(9);
    ir.block_mut(BlockId(0)).insts.push(Inst::Const {
        dst: x,
        c: Constant::I64(1),
    });
    ir.block_mut(BlockId(0)).insts.push(Inst::Call(Call {
        dst: None,
        callee: "user_fn".into(),
        args: vec![x],
        arr_refs: Vec::new(),
    }));
    ir.set_term(BlockId(0), Term::Return(None));

    let err = verify(&ir).expect_err("argument with no type row must fail");
    assert!(err.contains("no recorded type"), "unexpected: {err}");
}

// ---------------------------------------------------------------------------
// 3. real lowering must not trip the new checks
// ---------------------------------------------------------------------------

#[test]
fn real_lowering_with_array_builtins_verifies_after_to_ssa() {
    // `zeros` passes its length through args and `len` receives the array BY
    // REFERENCE as a cell value — both legal shapes the conservative checks
    // must keep accepting.
    let src = r#"
        fn main() {
            let a: [f64] = zeros(4);
            let n = len(a);
            print(min(n, 3));
            print(max(n, 1));
            print(sqrt(abs(-2.25)));
        }
    "#;
    for mut f in compile(src) {
        helix_ir::to_ssa(&mut f);
        verify_ssa(&f).unwrap_or_else(|e| panic!("{} failed SSA verification: {e}", f.name));
    }
}
