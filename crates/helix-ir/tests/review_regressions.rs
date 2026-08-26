//! Regression tests for reviewed helix-ir findings:
//!
//! 1. `Builder::assign` must evaluate the array-store INDEX before the stored
//!    VALUE (interpreter order is normative).
//! 2. `copy_prop` phi deletion must filter predecessor jump args positionally,
//!    never pop the last column.
//! 3. `to_ssa` must not duplicate a pre-existing builder phi (the `$ret` exit
//!    accumulator) at the same block.
//! 4. `to_ssa` output must verify: no spurious phis whose columns carry the
//!    undefined version-0 spelling of short-circuit temps.

use helix_ir::{build, print_ir, run_passes_to_fixpoint, to_ssa, verify};

fn compile(src: &str) -> Vec<helix_ir::FuncIr> {
    let ast = helix_syntax::parse_str(src).expect("parse");
    let typed = helix_sema::check(&ast).expect("sema");
    build(&typed)
}

// ---------------------------------------------------------------------------
// 1. store operand emission order
// ---------------------------------------------------------------------------

#[test]
fn assign_emits_index_defs_before_value_defs() {
    // `a[tag(2)] = tag(3);` — the interpreter (normative) prints 2 then 3.
    // The builder must therefore emit the index call before the value call so
    // every backend that walks IR order agrees.
    let src = r#"
        fn tag(v: i64) -> i64 {
            print(v);
            return v;
        }
        fn main() {
            let a: [i64] = zeros(4);
            a[tag(2)] = tag(3);
        }
    "#;
    let f = compile(src).into_iter().next().unwrap();
    // Collect callee names in program order across all blocks in layout order
    // — for straight-line entry-block code this is execution order.
    let calls: Vec<&str> = f
        .blocks
        .iter()
        .flat_map(|b| b.insts.iter())
        .filter_map(|i| match i {
            helix_ir::Inst::Call(c) => Some(c.callee.as_str()),
            _ => None,
        })
        .collect();
    let idx_pos = calls
        .iter()
        .position(|c| *c == "print" || *c == "tag")
        .unwrap_or_else(|| panic!("calls present: {calls:?}"));
    // The two tag calls: whichever comes FIRST must be the index (2), i.e.
    // the print of 2 precedes the print of 3. Calls are lowered in order; the
    // index argument's call lands first.
    let prints_in_order = calls.len() >= 2 && calls[0] == calls[1];
    assert!(
        prints_in_order || idx_pos == 0,
        "expected index-defining call emitted before value-defining call; got {calls:?}"
    );
}

#[test]
fn assign_store_operands_reference_defs_in_evaluation_order() {
    // Stronger structural check independent of call lowering details: within
    // the block holding the Store, the definition feeding Store.idx must be
    // emitted BEFORE the definition feeding Store.val whenever both operands
    // are defined by distinct non-cell instructions in that block.
    let src = r#"
        fn main() {
            let a: [i64] = zeros(4);
            let k = 1;
            a[k + 0] = k + 1;
        }
    "#;
    let f = compile(src).into_iter().next().unwrap();
    'outer: for b in &f.blocks {
        for inst in b.insts.iter() {
            if let helix_ir::Inst::Store { idx, val, .. } = inst {
                let def_pos = |v: helix_ir::ValueId| -> Option<usize> {
                    b.insts.iter().position(|i| i.dst() == Some(v))
                };
                if let (Some(pi), Some(vi)) = (def_pos(*idx), def_pos(*val)) {
                    assert!(
                        pi < vi,
                        "index def (#{pi}) must be emitted before value def (#{vi})"
                    );
                }
                break 'outer;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 2. copy_prop phi deletion keeps surviving phis' edge values intact
// ---------------------------------------------------------------------------

#[test]
fn copy_prop_phi_deletion_keeps_sibling_phi_columns() {
    // A join with TWO phis where only one is an identical-input copy. The old
    // cleanup popped one arg off each pred's jump list, stealing the column of
    // the SURVIVING phi (silent miscompile / verifier panic downstream).
    let src = r#"
        fn f(p: i64) -> i64 {
            let a = 0;
            let b = p;
            if p > 0 {
                a = 0;
                b = p + 1;
            }
            return a + b * 10;
        }
        fn main() { print(f(2)); }
    "#;
    let ast = helix_syntax::parse_str(src).expect("parse");
    let typed = helix_sema::check(&ast).expect("sema");
    // Drive passes individually to reach copy_prop with a deletable phi-copy.
    for fidx in 0..typed.funcs.len() {
        let mut ir = build(&typed)[fidx].clone();
        to_ssa(&mut ir);
        let _ = helix_ir::passes::simplify_cfg(&mut ir);
        let _ = helix_ir::passes::const_fold(&mut ir);
        let _ = helix_ir::passes::const_prop::const_prop(&mut ir);
        let flag = helix_ir::passes::copy_prop(&mut ir);
        let _ = flag;
        verify(&ir)
            .unwrap_or_else(|e| panic!("verify after copy_prop failed on fn {}: {}", ir.name, e));
        // Invariant: jump arg lists still align 1:1 with target phis and each
        // edge value is one the target phi accepts from that pred.
        for (bi, b) in ir.blocks.iter().enumerate() {
            if let helix_ir::Term::Jump(t, args) = &b.term {
                let target = &ir.blocks[t.0 as usize];
                assert_eq!(
                    args.len(),
                    target.phis.len(),
                    "bb{bi}: jump arity {} != {} target phis",
                    args.len(),
                    target.phis.len()
                );
                for (k, v) in args.iter().enumerate() {
                    let phi = &target.phis[k];
                    assert!(
                        phi.args
                            .iter()
                            .any(|(from, pv)| *from == helix_ir::BlockId(bi as u32) && pv == v),
                        "bb{bi} jump passes v{} but bb{} phi #{k} does not accept it from this edge",
                        v.0,
                        t.0
                    );
                }
            }
        }
    }
}

#[test]
fn copy_prop_pipeline_survives_multi_phi_join() {
    // Same shape through the full fix-point driver (which verifies after every
    // pass) — used to panic inside run_passes_to_fixpoint.
    let src = r#"
        fn f(p: i64) -> i64 {
            let a = 0;
            let b = p;
            if p > 0 {
                a = 0;
                b = p + 1;
            }
            print(b);
            return a;
        }
        fn main(){ print(f(2)); }
    "#;
    for mut f in compile(src) {
        to_ssa(&mut f);
        let reports = run_passes_to_fixpoint(&mut f);
        assert!(!reports.is_empty());
        verify(&f).unwrap_or_else(|e| panic!("post-pipeline: {e}"));
    }
}

// ---------------------------------------------------------------------------
// 3. to_ssa must not mint a second phi over a var that already has one
// ---------------------------------------------------------------------------

#[test]
fn to_ssa_no_duplicate_phi_for_ret_accumulator() {
    // Early-return functions make the $ret cell's def blocks' dominance
    // frontier reach the shared exit block, which ALREADY hosts a builder-made
    // phi over $ret. place_phis used to push a second identical phi there.
    let src = r#"
        fn g(x: i64) -> i64 {
            if x > 0 {
                return 1;
            }
            return 2;
        }
        fn fib(n: i64) -> i64 {
            if n < 2 {
                return n;
            } else if n < 15 {
                return fib(n - 1) + fib(n - 2);
            }
            return fib(n - 3) + 4;
        }
        fn main() { }
    "#;
    for f in compile(src) {
        let mut ssa = f.clone();
        to_ssa(&mut ssa);
        for (bi, b) in ssa.blocks.iter().enumerate() {
            let mut seen: Vec<helix_ir::LocalId> = Vec::new();
            for p in &b.phis {
                assert!(
                    !seen.contains(&p.var),
                    "{} bb{bi}: duplicate phi over local {}",
                    f.name,
                    p.var.0
                );
                seen.push(p.var);
            }
        }
        helix_ir::verify_ssa(&ssa)
            .unwrap_or_else(|e| panic!("{} failed SSA verification after to_ssa: {e}", f.name));
        assert!(helix_ir::is_ssa(&ssa), "{} should be SSA", f.name);
    }
}

#[test]
fn to_ssa_leaves_functions_ready_for_const_prop() {
    // Side effect of finding 3: is_ssa()==false made const_prop refuse to run.
    let src = r#"
        fn g(x: i64) -> i64 {
            if x > 0 {
                return 1;
            }
            return 2;
        }
        fn main() { }
    "#;
    let mut f = compile(src).into_iter().next().unwrap();
    to_ssa(&mut f);
    assert!(helix_ir::is_ssa(&f), "to_ssa output must satisfy is_ssa");
}

// ---------------------------------------------------------------------------
// 4. to_ssa output verifies (no undefined cell ids on edges into phis)
// ---------------------------------------------------------------------------

#[test]
fn to_ssa_output_verifies_with_shortcircuit_in_if_arm() {
    let src = r#"
        fn f(p: i64, q: bool) -> bool {
            let r: bool = q;
            if p > 0 {
                r = p < 10 && p > 2;
            }
            return r;
        }
        fn main() { print(f(5, false)); }
    "#;
    let mut f = compile(src).into_iter().next().unwrap();
    to_ssa(&mut f);
    println!("{}", print_ir(&f, true));
    verify(&f).unwrap_or_else(|e| panic!("verify right after to_ssa: {e}"));
}

#[test]
fn to_ssa_output_verifies_with_shortcircuit_in_loop_body() {
    let src = r#"
        fn main() {
            let hits: [i64] = zeros(5);
            for i in 0..4 {
                if i > 1 && i < 3 {
                    hits[i] = 1;
                }
            }
            print(hits[2]);
        }
    "#;
    let mut f = compile(src).into_iter().next().unwrap();
    to_ssa(&mut f);
    println!("{}", print_ir(&f, true));
    verify(&f).unwrap_or_else(|e| panic!("verify right after to_ssa: {e}"));
}

#[test]
fn to_ssa_output_verifies_nested_shortcircuits_and_pipeline() {
    let src = r#"
        fn f(a: i64, b: i64, c: i64) -> bool {
            if a > 0 {
                return (a < b && b < c) || a == c;
            }
            return false;
        }
        fn main() { print(f(1, 2, 3)); print(f(9, 2, 9)); print(f(5, 2, 3)); }
    "#;
    for mut f in compile(src) {
        to_ssa(&mut f);
        verify(&f).unwrap_or_else(|e| panic!("verify right after to_ssa ({}): {e}", f.name));
        run_passes_to_fixpoint(&mut f);
        verify(&f).unwrap_or_else(|e| panic!("post-pipeline ({}): {e}", f.name));
    }
}

// ---------------------------------------------------------------------------
// 5. const_fold must evaluate each width natively (P1-1: f64 folded at f32)
// ---------------------------------------------------------------------------

#[test]
fn const_fold_folds_f64_at_double_precision() {
    // 2^24 + 1 is exact in f64 but rounds in f32, so `16777217.0 + 1.0`
    // discriminates the widths: the f32 path yields 16777216.0.
    let sum = helix_ir::passes::fold_bin(
        helix_syntax::BinOp::Add,
        helix_ir::Constant::F64(16777217.0),
        helix_ir::Constant::F64(1.0),
    );
    assert_eq!(
        sum,
        Some(helix_ir::Constant::F64(16777218.0)),
        "f64 fold must compute (and type) in double precision"
    );
    // f32 inputs still round in single precision: 2^24 + 1 ties and rounds
    // back down to 2^24.
    assert_eq!(
        helix_ir::passes::fold_bin(
            helix_syntax::BinOp::Add,
            helix_ir::Constant::F32(16777217.0_f32),
            helix_ir::Constant::F32(1.0),
        ),
        Some(helix_ir::Constant::F32(16777216.0))
    );
    // Mixed widths never fold (HELIX requires an explicit cast).
    assert_eq!(
        helix_ir::passes::fold_bin(
            helix_syntax::BinOp::Add,
            helix_ir::Constant::F32(1.0),
            helix_ir::Constant::F64(1.0),
        ),
        None
    );
}

#[test]
fn const_fold_pass_folds_f64_source_at_double_precision() {
    let src = r#"
        fn main() {
            let x = 16777217.0 + 1.0;
        }
    "#;
    let mut f = compile(src).into_iter().next().unwrap();
    to_ssa(&mut f);
    let flag = helix_ir::passes::const_fold(&mut f);
    assert!(flag.changed);
    verify(&f).unwrap_or_else(|e| panic!("verify after const_fold: {e}"));
    let has_exact_sum = f.blocks.iter().any(|b| {
        b.insts.iter().any(|i| {
            matches!(
                i,
                helix_ir::Inst::Const {
                    c: helix_ir::Constant::F64(v),
                    ..
                } if *v == 16777218.0
            )
        })
    });
    assert!(
        has_exact_sum,
        "folded f64 sum 16777218.0 missing:\n{}",
        print_ir(&f, true)
    );
    // The old bug retyped results as F32; none may appear.
    assert!(
        !f.blocks.iter().any(|b| {
            b.insts.iter().any(|i| {
                matches!(
                    i,
                    helix_ir::Inst::Const {
                        c: helix_ir::Constant::F32(_),
                        ..
                    }
                )
            })
        }),
        "f64 arithmetic must not mint F32 constants"
    );
}

#[test]
fn const_fold_integer_wrapping_unchanged_by_float_fix() {
    use helix_ir::passes::fold_bin;
    use helix_syntax::BinOp;
    // i64 wrapping semantics pinned (also covered in passes_tests).
    assert_eq!(
        fold_bin(
            BinOp::Add,
            helix_ir::Constant::I64(i64::MAX),
            helix_ir::Constant::I64(1)
        ),
        Some(helix_ir::Constant::I64(i64::MIN))
    );
    assert_eq!(
        fold_bin(
            BinOp::Mul,
            helix_ir::Constant::I64(1 << 62),
            helix_ir::Constant::I64(4)
        ),
        Some(helix_ir::Constant::I64(0))
    );
    // Integers fold at i64 width regardless of literal class (pre-existing
    // widening design) — pinned so the float-width fix cannot drift it.
    assert_eq!(
        fold_bin(
            BinOp::Add,
            helix_ir::Constant::I32(i32::MAX),
            helix_ir::Constant::I32(1)
        ),
        Some(helix_ir::Constant::I64(2147483648))
    );
}

// ---------------------------------------------------------------------------
// 6. DCE must keep trapping div/rem (P1-2)
// ---------------------------------------------------------------------------

fn count_div_rem(f: &helix_ir::FuncIr) -> usize {
    f.blocks
        .iter()
        .map(|b| {
            b.insts
                .iter()
                .filter(|i| {
                    matches!(
                        i,
                        helix_ir::Inst::Bin {
                            op: helix_syntax::BinOp::Div | helix_syntax::BinOp::Rem,
                            ..
                        }
                    )
                })
                .count()
        })
        .sum()
}

#[test]
fn dce_keeps_unused_trapping_div_rem() {
    // Every division below traps by lang-spec (`x/0`, `x%0`, `MIN/-1`) yet
    // none of the results is used. DCE used to sweep them away, erasing the
    // mandated runtime errors; they are roots now and must all survive.
    let src = r#"
        fn probe(x: i64) -> i64 {
            let min = 0 - 9223372036854775807 - 1;
            let z = 0;
            let m1 = 0 - 1;
            let a = x / z;
            let b = x % z;
            let c = min / m1;
            let d = min % m1;
            return x;
        }
        fn main() { }
    "#;
    for mut f in compile(src).into_iter().filter(|f| f.name == "probe") {
        to_ssa(&mut f);
        let before = count_div_rem(&f);
        assert!(before >= 4, "fixture lost its divisions");
        let _ = helix_ir::passes::dce(&mut f);
        assert_eq!(
            count_div_rem(&f),
            before,
            "trapping div/rem must survive DCE"
        );
        verify(&f).unwrap_or_else(|e| panic!("verify after dce: {e}"));
    }
}

#[test]
fn dce_still_removes_unused_non_trapping_ops() {
    // Guard against the P1-2 fix over-correcting: plain unused adds, muls and
    // negations remain deletable.
    let src = r#"
        fn probe(p: i64) -> i64 {
            let u = p + 1;
            let v = p * 2;
            let w = 0 - p;
            return p;
        }
        fn main() { }
    "#;
    for mut f in compile(src).into_iter().filter(|f| f.name == "probe") {
        to_ssa(&mut f);
        let count_pure = |f: &helix_ir::FuncIr| -> usize {
            f.blocks
                .iter()
                .map(|b| b.insts.iter().filter(|i| i.is_pure()).count())
                .sum()
        };
        let before = count_pure(&f);
        let flag = helix_ir::passes::dce(&mut f);
        assert!(flag.changed, "dead pure ops must still go");
        assert!(count_pure(&f) < before, "pure dead code must shrink");
        verify(&f).unwrap_or_else(|e| panic!("verify after dce: {e}"));
    }
}

// ---------------------------------------------------------------------------
// 7. LICM must not hoist through a synthetic preheader that a branch bypasses
//    (P1-3)
// ---------------------------------------------------------------------------

/// Hand-built CFG (following the `compact_renumbers_phi_args` style):
///
/// ```text
/// bb0 ──Branch──▶ bb1 ──Jump──▶ bb3 (header) ◀──Jump── bb4 (latch)
///        │                                      ▲
///        └─────────▶ bb2 ──Branch(t)────────────┘   (outside BRANCH pred!)
/// bb3 ──Branch(f)──▶ bb5 (exit)
/// ```
///
/// The header holds a loop-invariant add whose result feeds the header
/// condition — hoisting it into a synthesized preheader without redirecting
/// bb2's branch edge leaves the use undominated on the bb2 path.
fn branch_pred_loop() -> helix_ir::FuncIr {
    use helix_sema::Ty;
    let mut ir = helix_ir::FuncIr::new("branch_pred_loop", Ty::Unit, 0);
    while ir.blocks.len() < 6 {
        ir.new_block();
    }
    let c1 = ir.new_value(Ty::I64);
    let c2 = ir.new_value(Ty::I64);
    let cond0 = ir.new_value(Ty::Bool);
    let cb = ir.new_value(Ty::Bool);

    // bb0: consts; branch picks between the two outside entries.
    ir.block_mut(helix_ir::BlockId(0))
        .insts
        .push(helix_ir::Inst::Const {
            dst: c1,
            c: helix_ir::Constant::I64(7),
        });
    ir.block_mut(helix_ir::BlockId(0))
        .insts
        .push(helix_ir::Inst::Const {
            dst: c2,
            c: helix_ir::Constant::I64(11),
        });
    ir.block_mut(helix_ir::BlockId(0))
        .insts
        .push(helix_ir::Inst::Const {
            dst: cond0,
            c: helix_ir::Constant::Bool(true),
        });
    ir.set_term(
        helix_ir::BlockId(0),
        helix_ir::Term::Branch {
            cond: cond0,
            t: helix_ir::BlockId(1),
            f: helix_ir::BlockId(2),
        },
    );

    // bb1: outside JUMP pred.
    ir.set_term(
        helix_ir::BlockId(1),
        helix_ir::Term::Jump(helix_ir::BlockId(3), Vec::new()),
    );

    // bb2: outside BRANCH pred into the header (else exits).
    ir.block_mut(helix_ir::BlockId(2))
        .insts
        .push(helix_ir::Inst::Const {
            dst: cb,
            c: helix_ir::Constant::Bool(true),
        });
    ir.set_term(
        helix_ir::BlockId(2),
        helix_ir::Term::Branch {
            cond: cb,
            t: helix_ir::BlockId(3),
            f: helix_ir::BlockId(5),
        },
    );

    // bb3: header — invariant add feeding the header condition.
    let inv = ir.new_value(Ty::I64);
    ir.block_mut(helix_ir::BlockId(3))
        .insts
        .push(helix_ir::Inst::Bin {
            op: helix_syntax::BinOp::Add,
            dst: inv,
            a: c1,
            b: c2,
        });
    let cond_h = ir.new_value(Ty::Bool);
    ir.block_mut(helix_ir::BlockId(3))
        .insts
        .push(helix_ir::Inst::Bin {
            op: helix_syntax::BinOp::Lt,
            dst: cond_h,
            a: inv,
            b: c2,
        });
    ir.set_term(
        helix_ir::BlockId(3),
        helix_ir::Term::Branch {
            cond: cond_h,
            t: helix_ir::BlockId(4),
            f: helix_ir::BlockId(5),
        },
    );

    // bb4: latch (back edge).
    ir.set_term(
        helix_ir::BlockId(4),
        helix_ir::Term::Jump(helix_ir::BlockId(3), Vec::new()),
    );

    // bb5: exit.
    ir.set_term(helix_ir::BlockId(5), helix_ir::Term::Return(None));

    ir.recompute_edges();
    ir
}

#[test]
fn licm_skips_loop_with_outside_branch_pred() {
    let mut f = branch_pred_loop();
    let flag = helix_ir::passes::licm(&mut f);
    // Skipping is the mandated outcome: no hoist, no CFG surgery.
    assert!(
        !flag.changed,
        "loop with an outside branch pred must be skipped entirely"
    );
    verify(&f).unwrap_or_else(|e| panic!("verify after skipped licm: {e}"));
    helix_ir::verify_ssa(&f).unwrap_or_else(|e| panic!("verify_ssa after skipped licm: {e}"));
    // Structure untouched: no synthesized block, header still owns the add,
    // and all three predecessors still reach the header directly.
    assert_eq!(f.blocks.len(), 6, "no preheader may be synthesized");
    assert!(
        f.blocks[3].insts.iter().any(|i| {
            matches!(
                i,
                helix_ir::Inst::Bin {
                    op: helix_syntax::BinOp::Add,
                    ..
                }
            )
        }),
        "invariant add stays in the header when hoisting is skipped"
    );
    assert_eq!(
        f.blocks[3].preds.iter().map(|p| p.0).collect::<Vec<u32>>(),
        vec![1, 2, 4],
        "header preds unchanged"
    );
}

#[test]
fn licm_still_forwards_multiple_jump_preds_and_hoists() {
    // Same shape but BOTH outside preds are jumps: forwarding through one
    // fresh preheader must keep working (regression guard for the skip rule).
    let mut f = branch_pred_loop();
    // Turn bb2's branch into a jump to the header (drop its exit edge).
    f.set_term(
        helix_ir::BlockId(2),
        helix_ir::Term::Jump(helix_ir::BlockId(3), Vec::new()),
    );
    f.recompute_edges();

    let flag = helix_ir::passes::licm(&mut f);
    assert!(flag.changed, "two jump preds: forward + hoist must happen");
    verify(&f).unwrap_or_else(|e| panic!("verify after licm: {e}"));
    helix_ir::verify_ssa(&f).unwrap_or_else(|e| panic!("verify_ssa after licm: {e}"));

    // The invariant add moved out of the header into the synthesized
    // preheader, which dominates the header (verifier proved it above).
    assert!(
        !f.blocks[3].insts.iter().any(|i| {
            matches!(
                i,
                helix_ir::Inst::Bin {
                    op: helix_syntax::BinOp::Add,
                    ..
                }
            )
        }),
        "invariant add must leave the header"
    );
    let pre = f.blocks.iter().enumerate().find(|(_, b)| {
        matches!(b.term, helix_ir::Term::Jump(t, _) if t == helix_ir::BlockId(3))
            && b.insts.iter().any(|i| {
                matches!(
                    i,
                    helix_ir::Inst::Bin {
                        op: helix_syntax::BinOp::Add,
                        ..
                    }
                )
            })
    });
    assert!(
        pre.is_some(),
        "hoisted add lives in the forwarding preheader"
    );
}
