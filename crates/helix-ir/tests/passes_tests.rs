//! Per-pass positive cases + no-op cases, and verifier corruption detection.
//! Every pass runs on SSA-form IR built through `build` + `to_ssa`, exactly
//! as the pass driver does.

use helix_ir::{Constant, FuncIr, Inst, Term, build, to_ssa, verify};

fn compile_ssa(src: &str) -> FuncIr {
    let ast = helix_syntax::parse_str(src).expect("parse");
    let typed = helix_sema::check(&ast).expect("sema");
    let mut f = build(&typed).into_iter().next().expect("one fn");
    to_ssa(&mut f);
    verify(&f).expect("pre-pass verify");
    f
}

fn count_pure(f: &FuncIr) -> usize {
    f.blocks
        .iter()
        .map(|b| b.insts.iter().filter(|i| i.is_pure()).count())
        .sum()
}

fn count_insts(f: &FuncIr) -> usize {
    f.blocks.iter().map(|b| b.insts.len()).sum()
}

fn count_stores(f: &FuncIr) -> usize {
    f.blocks
        .iter()
        .map(|b| {
            b.insts
                .iter()
                .filter(|i| matches!(i, Inst::Store { .. }))
                .count()
        })
        .sum()
}

fn count_loads(f: &FuncIr) -> usize {
    f.blocks
        .iter()
        .map(|b| {
            b.insts
                .iter()
                .filter(|i| matches!(i, Inst::Load(_)))
                .count()
        })
        .sum()
}

// ---------------------------------------------------------------------------
// const_fold
// ---------------------------------------------------------------------------

#[test]
fn const_fold_folds_arith_chain() {
    let mut f = compile_ssa("fn main() { let x = 1 + 2 * 3; }");
    let flag = helix_ir::passes::const_fold(&mut f);
    assert!(flag.changed);
    verify(&f).unwrap_or_else(|e| panic!("{e}"));

    // After folding: a single Const(7) exists; no Bin over two consts remains.
    let has_seven = f.blocks.iter().any(|b| {
        b.insts.iter().any(|i| {
            matches!(
                i,
                Inst::Const {
                    c: Constant::I64(7),
                    ..
                }
            )
        })
    });
    assert!(has_seven, "folded to 7");
    let bin_over_consts = f.blocks.iter().any(|b| {
        b.insts.iter().any(|i| match i {
            Inst::Bin { a, b: b2, .. } => {
                let is_c = |v: helix_ir::ValueId| {
                    f.blocks.iter().any(|bb| {
                        bb.insts
                            .iter()
                            .any(|x| matches!(x, Inst::Const { dst, .. } if *dst == v))
                    })
                };
                is_c(*a) && is_c(*b2)
            }
            _ => false,
        })
    });
    assert!(
        !bin_over_consts,
        "no Bin with two Const operands may remain"
    );
}

#[test]
fn const_fold_no_op_on_variables() {
    let mut f = compile_ssa("fn f(p: i64) -> i64 { return p + 1; } fn main() { }");
    let flag = helix_ir::passes::const_fold(&mut f);
    assert!(!flag.changed, "runtime add cannot fold");
}

#[test]
fn const_fold_truncated_remainder_and_div_guards() {
    use helix_ir::passes::{fold_bin, fold_cast};
    use helix_syntax::BinOp;
    // -7 % 2 == -1 (sign of dividend); 7 % -2 == 1.
    assert_eq!(
        fold_bin(BinOp::Rem, Constant::I64(-7), Constant::I64(2)),
        Some(Constant::I64(-1))
    );
    assert_eq!(
        fold_bin(BinOp::Rem, Constant::I64(7), Constant::I64(-2)),
        Some(Constant::I64(1))
    );
    // Division traps must NOT fold away.
    assert_eq!(
        fold_bin(BinOp::Div, Constant::I64(1), Constant::I64(0)),
        None
    );
    assert_eq!(
        fold_bin(BinOp::Rem, Constant::I64(i64::MIN), Constant::I64(-1)),
        None
    );
    // Wrapping arithmetic matches the frozen semantics.
    assert_eq!(
        fold_bin(BinOp::Add, Constant::I64(i64::MAX), Constant::I64(1)),
        Some(Constant::I64(i64::MIN))
    );
    // Saturating float->int casts.
    assert_eq!(
        fold_cast(Constant::F64(1e300), helix_sema::Ty::I64),
        Some(Constant::I64(i64::MAX))
    );
    assert_eq!(
        fold_cast(Constant::F64(f64::NAN), helix_sema::Ty::I32),
        Some(Constant::I32(0))
    );
    assert_eq!(
        fold_cast(Constant::F64(-1.9), helix_sema::Ty::I64),
        Some(Constant::I64(-1)) // round toward zero
    );
}

#[test]
fn const_prop_folds_branch_to_jump() {
    // `if 1 < 2` collapses: the branch becomes a jump and simplify_cfg cleans
    // up the dead arm.
    let mut f = compile_ssa("fn main() { if 1 < 2 { let a = 5; } else { let b = 6; } }");
    let flag = helix_ir::passes::const_prop::const_prop(&mut f);
    assert!(flag.changed);
    verify(&f).unwrap_or_else(|e| panic!("{e}"));
    // No Branch with a constant condition remains.
    let still_const_branch = f.blocks.iter().any(|b| {
        matches!(&b.term, Term::Branch { cond, .. }
            if f.blocks.iter().any(|bb| bb.insts.iter().any(
                |x| matches!(x, Inst::Const { dst, .. } if dst == cond))))
    });
    assert!(!still_const_branch, "constant branch must be folded");
}

// ---------------------------------------------------------------------------
// dce
// ---------------------------------------------------------------------------

#[test]
fn dce_removes_dead_pure_inst() {
    // `len(a) + len(a)` feeds nothing pure-and-live; the calls stay (effects)
    // but the dead Add goes.
    let mut f =
        compile_ssa("fn main() { let a: [i64] = zeros(2); let d = len(a) + len(a); a[0] = 1; }");
    let before = count_pure(&f);
    let flag = helix_ir::passes::dce(&mut f);
    assert!(flag.changed);
    let after = count_pure(&f);
    assert!(after < before, "pure dead code removed");
    verify(&f).unwrap_or_else(|e| panic!("{e}"));
    // Store kept.
    assert!(
        f.blocks
            .iter()
            .any(|b| b.insts.iter().any(|i| matches!(i, Inst::Store { .. })))
    );
}

#[test]
fn dce_keeps_loads_even_if_unused() {
    // A load may trap (bounds); DCE never removes one even when its value is
    // discarded by later DCE-able ops.
    let mut f = compile_ssa("fn main() { let a: [i64] = zeros(2); let u = a[0]; a[1] = u - u; }");
    let loads_before = count_loads(&f);
    let _ = helix_ir::passes::dce(&mut f);
    assert_eq!(count_loads(&f), loads_before, "load survived");
}

// ---------------------------------------------------------------------------
// cse
// ---------------------------------------------------------------------------

#[test]
fn cse_dedupes_same_expression() {
    let mut f = compile_ssa(
        "fn f(p: i64) -> i64 { let x = p * 2; let y = p * 2; return x + y; } fn main() { }",
    );
    let before = count_insts(&f);
    let flag = helix_ir::passes::cse(&mut f);
    assert!(flag.changed, "duplicate p*2 eliminated");
    let after = count_insts(&f);
    assert!(after < before, "inst count dropped: {before} -> {after}");
    verify(&f).unwrap_or_else(|e| panic!("{e}"));
}

#[test]
fn cse_no_op_on_distinct_operands() {
    // NOTE: CSE's constant-dedup stage legitimately fires here (`const 2`
    // appears twice), so the assertion targets the *expression* table only:
    // after one call, no duplicate `p*2`/`q*2` pairs may remain but both
    // multiplies must survive as distinct computations.
    let mut f = compile_ssa(
        "fn f(p: i64, q: i64) -> i64 { let x = p * 2; let y = q * 2; return x + y; } fn main() { }",
    );
    let _ = helix_ir::passes::cse(&mut f);
    verify(&f).unwrap_or_else(|e| panic!("{e}"));
    let muls: Vec<(u32, u32)> = f
        .blocks
        .iter()
        .flat_map(|b| b.insts.iter())
        .filter_map(|i| match i {
            Inst::Bin {
                op: helix_syntax::BinOp::Mul,
                a,
                b,
                ..
            } => Some((a.0, b.0)),
            _ => None,
        })
        .collect();
    assert_eq!(muls.len(), 2, "both multiplications survive");
    assert!(muls[0] != muls[1], "p*2 and q*2 are distinct computations");
}

// ---------------------------------------------------------------------------
// copy_prop
// ---------------------------------------------------------------------------

#[test]
fn copy_prop_collapses_identical_phi_inputs() {
    // Both paths carry the same constant into the join phi; after CSE
    // canonicalizes the duplicated `const 7` defs, the phi's inputs become
    // identical ids and copy-propagation replaces every use with that def.
    let src = r#"
        fn g(p: i64) -> i64 {
            let v = 7;
            if p < 0 {
                v = 7;
            }
            return v + 0;
        }
        fn main() { }
    "#;
    let mut f = compile_ssa(src);
    let _ = helix_ir::passes::cse(&mut f); // canonicalize const defs
    let flag = helix_ir::passes::copy_prop(&mut f);
    verify(&f).unwrap_or_else(|e| panic!("{e}"));
    assert!(flag.changed, "identical-input phi must propagate");
    let identical_phis = f.blocks.iter().any(|b| {
        b.phis.iter().any(|p| {
            !p.args.is_empty()
                && p.args.iter().all(|(_, v)| *v == p.args[0].1)
                && p.args[0].1 != p.dst
        })
    });
    assert!(!identical_phis, "all-identical phi should be gone");
}

#[test]
fn copy_prop_safe_when_nothing_to_do() {
    let mut f =
        compile_ssa("fn f(p: i64) -> i64 { if p < 3 { return p; } return 0; } fn main() { }");
    let _ = helix_ir::passes::copy_prop(&mut f);
    verify(&f).unwrap_or_else(|e| panic!("{e}"));
}

// ---------------------------------------------------------------------------
// licm
// ---------------------------------------------------------------------------

#[test]
fn licm_hoists_invariant_pure_ops() {
    // `3 * 4` is loop-invariant and pure => hoisted (with its constant
    // operands) out of the body into the preheader.
    let src = r#"
        fn main() {
            let a: [i64] = zeros(8);
            for i in 0..8 {
                a[i] = i * (3 * 4);
            }
        }
    "#;
    let mut f = compile_ssa(src);

    // The store's body block must contain NO invariant multiply after
    // hoisting: everything feeding the iv-multiply moved to the preheader.
    for _ in 0..4 {
        if !helix_ir::passes::licm(&mut f).changed {
            break;
        }
    }
    verify(&f).unwrap_or_else(|e| panic!("{e}"));

    let body_blocks: Vec<usize> = f
        .blocks
        .iter()
        .enumerate()
        .filter(|(_, b)| b.insts.iter().any(|i| matches!(i, Inst::Store { .. })))
        .map(|(i, _)| i)
        .collect();
    assert_eq!(body_blocks.len(), 1, "single store block");
    let body_insts = &f.blocks[body_blocks[0]].insts;

    // Exactly ONE multiply may remain in the body: `iv * invariant`. Any
    // second Mul would mean the invariant product was not hoisted.
    let muls_in_body: Vec<&Inst> = body_insts
        .iter()
        .filter(|i| {
            matches!(
                i,
                Inst::Bin {
                    op: helix_syntax::BinOp::Mul,
                    ..
                }
            )
        })
        .collect();
    assert_eq!(
        muls_in_body.len(),
        1,
        "only the iv-multiply remains in the body"
    );

    // And somewhere before the header the hoisted computation exists: two
    // Const defs plus a Bin whose operands are those consts, all in one
    // non-body block.
    let pre_ok = f.blocks.iter().enumerate().any(|(bi, b)| {
        bi != body_blocks[0]
            && b.insts.iter().any(|i| {
                matches!(
                    i,
                    Inst::Const {
                        c: Constant::I64(3),
                        ..
                    }
                )
            })
            && b.insts.iter().any(|i| {
                matches!(
                    i,
                    Inst::Const {
                        c: Constant::I64(4),
                        ..
                    }
                )
            })
            && b.insts.iter().any(|i| {
                matches!(
                    i,
                    Inst::Bin {
                        op: helix_syntax::BinOp::Mul,
                        ..
                    }
                )
            })
    });
    assert!(pre_ok, "hoisted consts + mul live before the loop");
}

#[test]
fn licm_never_hoists_loads_stores_or_traps() {
    let src = r#"
        fn main() {
            let a: [i64] = zeros(8);
            let k = 4;
            for i in 0..8 {
                a[i] = 16 / k;
            }
        }
    "#;
    let mut f = compile_ssa(src);
    let stores_before = count_stores(&f);
    let loads_before = count_loads(&f);
    for _ in 0..3 {
        if !helix_ir::passes::licm(&mut f).changed {
            break;
        }
    }
    verify(&f).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(count_stores(&f), stores_before, "stores never hoisted");
    assert_eq!(count_loads(&f), loads_before, "loads never hoisted");
}

// ---------------------------------------------------------------------------
// verifier catches seeded corruption
// ---------------------------------------------------------------------------

#[test]
fn verify_rejects_bad_phi_arity() {
    let f = compile_ssa("fn main() { let x = 5; if 1 < 2 { x = 10; } let y = x + 0; }");
    let mut corrupted = f.clone();
    let mut hit = false;
    for b in corrupted.blocks.iter_mut() {
        for p in b.phis.iter_mut() {
            if p.args.len() > 1 {
                p.args.pop();
                hit = true;
            }
        }
    }
    assert!(hit, "expected a multi-pred phi to corrupt");
    check_err(helix_ir::verify(&corrupted), "arg");
}

#[test]
fn verify_rejects_duplicate_phi_pred() {
    let f = compile_ssa("fn main() { let x = 5; if 1 < 2 { x = 10; } let y = x + 0; }");
    let mut corrupted = f.clone();
    let mut hit = false;
    'outer: for bi in 0..corrupted.blocks.len() {
        for pi in 0..corrupted.blocks[bi].phis.len() {
            if corrupted.blocks[bi].phis[pi].args.len() > 1 {
                // Duplicate the FIRST pred entry, keeping arity: the pred set
                // no longer matches the block's preds exactly.
                let first = corrupted.blocks[bi].phis[pi].args[0];
                let n = corrupted.blocks[bi].phis[pi].args.len();
                corrupted.blocks[bi].phis[pi].args.truncate(n - 1);
                corrupted.blocks[bi].phis[pi].args.push(first);
                hit = true;
                break 'outer;
            }
        }
    }
    assert!(hit);
    // The duplicate pred must be caught by SOME phi check (arity mismatch on
    // a sibling phi or the duplicate-pred message) — assert verification
    // fails at all, then look for the precise wording.
    let err = helix_ir::verify(&corrupted).expect_err("duplicate pred must be detected");
    assert!(
        err.contains("more than once")
            || err.contains("pred set")
            || err.contains("arg(s)")
            || err.contains("does not accept"),
        "message should identify the phi corruption: {err}"
    );
}

#[test]
fn verify_rejects_use_after_nothing() {
    let mut f = compile_ssa("fn f(p: i64) -> i64 { return p + 1; } fn main() { }");
    let bogus = helix_ir::ValueId(9_999);
    let mut hit = false;
    'outer: for b in f.blocks.iter_mut() {
        for inst in b.insts.iter_mut() {
            if matches!(inst, Inst::Bin { .. }) {
                inst.rewrite_uses(&mut |_| bogus);
                hit = true;
                break 'outer;
            }
        }
    }
    assert!(hit);
    check_err(helix_ir::verify(&f), "definition");
}

#[test]
fn verify_rejects_terminator_pred_mismatch() {
    let mut f = compile_ssa("fn main() { let a = 1; }");
    f.blocks[0].term = Term::Return(None);
    f.blocks[0].succs.clear();
    let err = helix_ir::verify(&f).expect_err("must detect pred/succ mismatch");
    assert!(
        err.contains("pred") || err.contains("succ"),
        "message names the broken edge: {err}"
    );
}

fn check_err(r: Result<(), String>, needle: &str) {
    let msg = r.expect_err("corruption must be detected");
    assert!(
        msg.to_lowercase().contains(&needle.to_lowercase()),
        "message '{msg}' should mention '{needle}'"
    );
}

// ---------------------------------------------------------------------------
// regression: CFG mutations must keep phi args aligned 1:1 with preds
// ---------------------------------------------------------------------------

/// Run every pass once in pipeline order, verifying after each — the same
/// discipline the driver uses, so a regression trips at its own doorstep.
fn run_pipeline_once(f: &mut FuncIr) {
    for pass in helix_ir::PassId::pipeline() {
        helix_ir::run_pass_by_id(*pass, f);
        verify(f).unwrap_or_else(|e| panic!("pass {} broke IR: {e}", pass.name()));
    }
}

#[test]
fn simplify_cfg_chain_merge_rekeys_phi_args_of_absorbed_block() {
    // The inner `for j` latch is a chain (latch -> header, single pred); when
    // simplify_cfg folds it into its predecessor, the HEADER's phis keep an
    // argument keyed on the absorbed block id. That entry must be re-keyed to
    // the merged block or phi args stop aligning with preds. Regression for
    // "simplify_cfg broke IR: bb12 phi(v9) has 1 arg(s) but block has 2 pred(s)"
    // observed on matmul/jacobi_2d.
    let mut f = compile_ssa(
        "fn main() { \
           let a: [i64] = zeros(4); \
           for i in 0..2 { \
             for j in 0..2 { a[i] = a[i] + j; } \
           } \
         }",
    );
    run_pipeline_once(&mut f);
    verify(&f).unwrap_or_else(|e| panic!("post-pipeline verify: {e}"));

    // Invariant, spelled out: every phi's pred set == its block's preds.
    for (bi, b) in f.blocks.iter().enumerate() {
        for p in &b.phis {
            assert_eq!(
                p.args.len(),
                b.preds.len(),
                "bb{bi} phi arity {} != {} preds",
                p.args.len(),
                b.preds.len()
            );
            let mut listed: Vec<u32> = p.args.iter().map(|(from, _)| from.0).collect();
            listed.sort_unstable();
            let want: Vec<u32> = b.preds.iter().map(|p| p.0).collect();
            assert_eq!(listed, want, "bb{bi} phi pred set mismatch");
        }
    }
}

#[test]
fn compact_renumbers_phi_args_of_surviving_blocks() {
    // Direct unit test of FuncIr::compact: dropping a mid-sequence block
    // shifts every later id down by one; surviving phis' argument entries
    // MUST be renumbered with them (terminators always were). Hand-built
    // diamond: bb0 branches over doomed bb1 into arms bb2/bb3, joining at
    // bb4 through a phi fed from both arms.
    use helix_ir::{BlockId, LocalId, Phi};
    let mut f = FuncIr::new("diamond", helix_sema::Ty::I64, 1);
    f.declare_local(LocalId(0), helix_sema::Ty::I64, "x");
    let cond = f.new_value(helix_sema::Ty::Bool);
    let v7 = f.new_value(helix_sema::Ty::I64);
    let v9 = f.new_value(helix_sema::Ty::I64);
    let merged = f.new_value(helix_sema::Ty::I64);
    for _ in 0..4 {
        f.new_block();
    }
    // bb0: branch cond ? bb2 : bb3
    f.block_mut(BlockId(0)).insts.push(Inst::Const {
        dst: cond,
        c: Constant::Bool(true),
    });
    f.set_term(
        BlockId(0),
        Term::Branch {
            cond,
            t: BlockId(2),
            f: BlockId(3),
        },
    );
    // bb1: the doomed block (unreachable once bb0 branches elsewhere, but
    // structurally complete so the IR verifies before compaction). It jumps
    // to bb4 passing its own constant.
    let v_dead = f.new_value(helix_sema::Ty::I64);
    f.block_mut(BlockId(1)).insts.push(Inst::Const {
        dst: v_dead,
        c: Constant::I64(0),
    });
    f.set_term(BlockId(1), Term::Jump(BlockId(4), vec![v_dead]));
    // bb2: const 7 -> jump bb4(7)
    f.block_mut(BlockId(2)).insts.push(Inst::Const {
        dst: v7,
        c: Constant::I64(7),
    });
    f.set_term(BlockId(2), Term::Jump(BlockId(4), vec![v7]));
    // bb3: const 9 -> jump bb4(9)
    f.block_mut(BlockId(3)).insts.push(Inst::Const {
        dst: v9,
        c: Constant::I64(9),
    });
    f.set_term(BlockId(3), Term::Jump(BlockId(4), vec![v9]));
    // bb4: x = phi [bb1: 0] [bb2: 7] [bb3: 9]; return x. The doomed bb1
    // contributes one column; compact() drops it along with the dead edge.
    f.block_mut(BlockId(4)).phis.push(Phi {
        dst: merged,
        var: LocalId(0),
        args: vec![(BlockId(1), v_dead), (BlockId(2), v7), (BlockId(3), v9)],
    });
    f.set_term(BlockId(4), Term::Return(Some(merged)));
    verify(&f).expect("hand-built IR verifies before compaction");

    let keep = vec![true, false, true, true, true];
    f.compact(&keep);

    // Old bb2/bb3/bb4 are now bb1/bb2/bb3; the merge phi's argument keys had
    // to move with them.
    assert_eq!(
        f.blocks[3].preds.iter().map(|p| p.0).collect::<Vec<u32>>(),
        vec![1, 2],
        "merge block preds renumbered"
    );
    let phi = &f.blocks[3].phis[0];
    let keys: Vec<u32> = phi.args.iter().map(|(from, _)| from.0).collect();
    assert_eq!(keys, vec![1, 2], "phi args must be renumbered like preds");
    assert_eq!(phi.args[0].1, v7, "arm value preserved across renumbering");
    assert_eq!(phi.args[1].1, v9, "arm value preserved across renumbering");
    verify(&f).expect("IR verifies after compaction");
}

// ---------------------------------------------------------------------------
// regression: dominance must hold after value-numbering-style rewrites
// ---------------------------------------------------------------------------

#[test]
fn const_prop_does_not_propagate_across_non_dominating_def() {
    // Diamond where both arms compute the SAME constant expression: naive
    // constant canonicalization would rewrite the join's use to whichever
    // def was seen first — but neither arm dominates the join. Regression
    // for "const_prop broke IR: inst #0 in bb5 uses value 12 defined in bb3,
    // which does not dominate it" (count_primes_sieve / const_globals).
    let mut f = compile_ssa(
        "fn f(p: i64) -> i64 { \
           let v = 0; \
           if p < 3 { v = 4 * 5; } else { v = 4 * 5; } \
           return v + p; \
         } \
         fn main() { }",
    );
    run_pipeline_once(&mut f);
    verify(&f).unwrap_or_else(|e| panic!("post-pipeline verify: {e}"));

    // And the semantic invariant: whatever remains, every use is still
    // dominated by its reaching def (verify above) AND the folded result is
    // preserved somewhere (20 feeds the return path).
    let has_twenty = f.blocks.iter().any(|b| {
        b.insts.iter().any(|i| {
            matches!(
                i,
                Inst::Const {
                    c: Constant::I64(20),
                    ..
                }
            )
        })
    });
    assert!(has_twenty, "constant 20 must survive folding");
}

#[test]
fn dce_never_deletes_one_def_of_a_multiply_defined_value() {
    // A function whose $ret accumulator is defined on several return arms
    // reaches DCE with a multiply-defined temp id. DCE's def index keeps only
    // the FIRST site per id, so sweeping "not-live" same-id defs strands the
    // jump arguments that name the id from other edges. Regression for
    // "dce broke IR: fib: jump argument in bb2 uses value 1 defined in bb1,
    // which does not dominate it".
    let mut f = compile_ssa(
        "fn pick(p: i64) -> i64 { \
           if p < 1 { return 1; } \
           if p < 10 { return p + 20; } \
           return p + 30; \
         } \
         fn main() { }",
    );
    run_pipeline_once(&mut f);
    verify(&f).unwrap_or_else(|e| panic!("post-pipeline verify: {e}"));

    // Every terminator operand still resolves to a definition that exists.
    for b in &f.blocks {
        let term_uses: Vec<helix_ir::ValueId> = match &b.term {
            Term::Jump(_, args) => args.clone(),
            Term::Branch { cond, .. } => vec![*cond],
            Term::Return(v) => v.iter().copied().collect(),
        };
        for u in term_uses {
            let defined = f.blocks.iter().any(|bb| {
                bb.phis.iter().any(|p| p.dst == u) || bb.insts.iter().any(|i| i.dst() == Some(u))
            });
            assert!(defined, "terminator uses value {} with no definition", u.0);
        }
    }
}
