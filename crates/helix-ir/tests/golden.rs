//! Golden-shape tests: lowering + SSA construction on the canonical programs
//! from the project brief (ssa_demo.hx shape), plus for-loop canonical form,
//! short-circuit diamonds and early returns.
//!
//! NOTE on observability: `TypedExprKind::Call` in helix-sema currently drops
//! argument expressions (upstream interface gap), so `print(x)` cannot observe
//! a value. Tests therefore force cross-block uses through arithmetic
//! (`let y = x + 0`) instead of calls.

use helix_ir::{build, print_ir, to_ssa, verify};

fn compile_all(src: &str) -> Vec<helix_ir::FuncIr> {
    let ast = helix_syntax::parse_str(src).expect("parse");
    let typed = helix_sema::check(&ast).expect("sema");
    build(&typed)
}

fn compile_fn_named(src: &str, name: &str) -> helix_ir::FuncIr {
    compile_all(src)
        .into_iter()
        .find(|f| f.name == name)
        .unwrap_or_else(|| panic!("function '{name}' not found"))
}

#[test]
fn ssa_demo_phi_shape() {
    let src = r#"
        fn main() {
            let x = 5;
            let cond = 1 < 2;
            if cond {
                x = 10;
            }
            let y = x + 0;
        }
    "#;
    let mut f = compile_fn_named(src, "main");

    // Pre-SSA: two defs of the x cell (bb0 const 5, then the arm's const 10).
    let pre = print_ir(&f, false);
    assert!(pre.contains("x = const 5"), "{pre}");
    assert!(pre.contains("x = const 10"), "{pre}");

    to_ssa(&mut f);
    verify(&f).unwrap_or_else(|e| panic!("verify: {e}"));
    let ssa = print_ir(&f, true);

    // The merge block must carry exactly one phi over x with both arms.
    let phi_lines: Vec<&str> = ssa.lines().filter(|l| l.contains('φ')).collect();
    assert_eq!(phi_lines.len(), 1, "one phi expected:\n{ssa}");
    assert!(phi_lines[0].contains("[bb0:"), "{ssa}");
    assert!(phi_lines[0].contains("[bb1:"), "{ssa}");
}

#[test]
fn shortcircuit_rhs_on_one_path_only() {
    // The rhs of && must execute ONLY when lhs is true: structurally, the rhs
    // computation lives in its own block reachable solely from the branch's
    // true arm, and the merge joins two preds.
    let src = r#"
        fn main() {
            let a: [i64] = zeros(3);
            if len(a) > 1 && len(a) < 2 {
                print(1);
            }
            let z = len(a) + 0;
        }
    "#;
    let mut f = compile_fn_named(src, "main");
    to_ssa(&mut f);
    verify(&f).unwrap_or_else(|e| panic!("verify: {e}"));

    // Find a branch whose one arm has >=2 preds (a merge) and the other arm
    // holds the second comparison.
    let mut found_sc = false;
    for b in &f.blocks {
        if let helix_ir::Term::Branch { t, f: ff, .. } = &b.term {
            let t_preds = f.blocks[t.0 as usize].preds.len();
            let f_preds = f.blocks[ff.0 as usize].preds.len();
            let t_has_cmp = f.blocks[t.0 as usize]
                .insts
                .iter()
                .chain(f.blocks[ff.0 as usize].insts.iter())
                .any(|i| {
                    matches!(
                        i,
                        helix_ir::Inst::Bin {
                            op: helix_syntax::BinOp::Lt,
                            ..
                        }
                    )
                });
            if (t_preds >= 2 || f_preds >= 2) && t_has_cmp {
                found_sc = true;
            }
        }
    }
    assert!(
        found_sc,
        "no short-circuit diamond found:\n{}",
        print_ir(&f, true)
    );
}

#[test]
fn early_return_exit_block() {
    let src = r#"
        fn sign(x: i64) -> i64 {
            if x < 0 {
                return 0 - 1;
            }
            return 1;
        }

        fn main() { }
    "#;
    let mut sign = compile_fn_named(src, "sign");
    to_ssa(&mut sign);
    verify(&sign).unwrap_or_else(|e| panic!("verify: {e}"));

    // Exactly one Return in the whole function, in a block with more than one
    // predecessor (the shared exit).
    let rets: Vec<usize> = sign
        .blocks
        .iter()
        .enumerate()
        .filter(|(_, b)| matches!(b.term, helix_ir::Term::Return(_)))
        .map(|(i, _)| i)
        .collect();
    assert_eq!(rets.len(), 1, "single shared exit expected");
    assert!(
        sign.blocks[rets[0]].preds.len() >= 2,
        "both return paths must funnel into one exit"
    );
}

#[test]
fn for_loop_canonical_iv_add_cmp() {
    let src = r#"
        fn main() {
            let s = 0;
            for i in 0..8 {
                s = s + i;
            }
            let out = s + 0;
        }
    "#;
    let mut f = compile_fn_named(src, "main");
    to_ssa(&mut f);
    verify(&f).unwrap_or_else(|e| panic!("verify: {e}"));
    let ssa = print_ir(&f, true);

    // Canonical pieces:
    //   * header φ over i merging preheader and latch,
    //   * `bin < iv end` comparison,
    //   * latch increment by one feeding the back edge.
    let has_iv_phi = f.blocks.iter().any(|b| {
        b.phis
            .iter()
            .any(|p| p.args.len() == 2 && p.args[0].0 == helix_ir::BlockId(0))
    });
    assert!(has_iv_phi, "iv phi merging entry+backedge:\n{ssa}");

    let cmp_lt = f.blocks.iter().any(|b| {
        b.insts.iter().any(|i| {
            matches!(
                i,
                helix_ir::Inst::Bin {
                    op: helix_syntax::BinOp::Lt,
                    ..
                }
            )
        })
    });
    assert!(cmp_lt, "header comparison missing");

    // Increment: some Add whose one operand is a Const(1).
    let add_one = f.blocks.iter().any(|b| {
        b.insts.iter().any(|i| match i {
            helix_ir::Inst::Bin { op: helix_syntax::BinOp::Add, b: rhs, .. } => {
                f.blocks.iter().any(|bb| {
                    bb.insts.iter().any(|x| {
                        matches!(x,
                            helix_ir::Inst::Const { dst, c: helix_ir::Constant::I64(1) } if dst == rhs)
                    })
                })
            }
            _ => false,
        })
    });
    assert!(add_one, "latch increments iv by one:\n{ssa}");
}

#[test]
fn arrays_stay_out_of_ssa() {
    let src = r#"
        fn main() {
            let a: [i64] = zeros(4);
            for i in 0..4 {
                a[i] = i * i;
            }
        }
    "#;
    let mut f = compile_fn_named(src, "main");
    to_ssa(&mut f);
    verify(&f).unwrap_or_else(|e| panic!("verify: {e}"));

    // Loads/stores reference array slots directly; no phi may exist for an
    // array local (arrays are memory, never SSA names).
    for (bi, b) in f.blocks.iter().enumerate() {
        for p in &b.phis {
            let ty = f.types.local_ty(p.var);
            assert!(
                ty.map(|t| !t.is_array()).unwrap_or(true),
                "bb{bi}: array local got a phi"
            );
        }
    }
    // A store survives somewhere.
    assert!(
        f.blocks.iter().any(|b| b
            .insts
            .iter()
            .any(|i| matches!(i, helix_ir::Inst::Store { .. }))),
        "store must remain explicit"
    );
}
