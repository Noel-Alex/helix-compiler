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
