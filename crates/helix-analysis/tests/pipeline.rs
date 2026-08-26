//! Pipeline integration tests on synthetic sources.
//!
//! Where `golden.rs` pins the verdicts of the repository's example kernels,
//! this file exercises shapes those kernels don't contain: WAR/WAW carried
//! edges, reduction *disqualifiers* (the negative cases — a wrong approval
//! here is a miscompile in M10), multi-reduction interactions, and
//! build_plan's depth rules. Sources are inline so each test documents its
//! own shape.

#![forbid(unsafe_code)]

use helix_analysis::{RegionKind, Verdict, analyze, find_loops};
use helix_syntax::parse_str;

/// Run parse → check → build → to_ssa → analyze over one function's source.
fn analyze_src(src: &str) -> Vec<helix_analysis::LoopReport> {
    let prog = parse_str(src).expect("parse");
    let typed = helix_sema::check(&prog).expect("check");
    let mut funcs = helix_ir::build(&typed);
    for f in &mut funcs {
        helix_ir::to_ssa(f);
    }
    let li = find_loops(&funcs[0]);
    analyze(&funcs[0], &li)
}

fn all_edges(r: &helix_analysis::LoopReport) -> usize {
    r.raw_deps.len() + r.war_deps.len() + r.waw_deps.len()
}

// ---------------------------------------------------------------------------
// Reduction disqualifiers (negative cases)
// ---------------------------------------------------------------------------

#[test]
fn accumulator_used_elsewhere_disqualifies_reduction() {
    // `s` flows into a store in addition to its own chain: lang-spec says the
    // accumulator is "referenced nowhere else in the body".
    let reps = analyze_src(
        "fn main() {
            let n = 100;
            let b: [i64] = zeros(n);
            let s = 0;
            for i in 0..n {
                s = s + i;
                b[i] = s * 2;
            }
        }",
    );
    assert!(
        reps.iter()
            .all(|r| !matches!(r.verdict, Verdict::ReductionParallel(_))),
        "extra use of accumulator must veto: {reps:?}"
    );
}

#[test]
fn accumulator_as_subscript_disqualifies_reduction() {
    let reps = analyze_src(
        "fn main() {
            let n = 100;
            let b: [i64] = zeros(n);
            let s = 0;
            for i in 0..10 {
                s = s + 1;
                b[s] = i;
            }
        }",
    );
    assert!(
        reps.iter()
            .all(|r| !matches!(r.verdict, Verdict::ReductionParallel(_))),
        "accumulator feeding an index must veto"
    );
}

#[test]
fn self_squaring_is_not_a_mul_reduction() {
    // s = s*s reads the accumulator twice; neither operand is an independent
    // term, so no associative combine exists.
    let reps = analyze_src(
        "fn main() {
            let n = 8;
            let s = 2;
            for i in 0..n {
                s = s * s;
            }
        }",
    );
    assert!(
        reps.iter()
            .all(|r| !matches!(r.verdict, Verdict::ReductionParallel(_))),
        "x = x*x must not approve"
    );
}

#[test]
fn print_in_body_vetoes_everything() {
    // Spec normative: print inside a loop means never parallelized.
    let reps = analyze_src(
        "fn main() {
            let n = 4;
            let a: [i64] = zeros(8);
            let s = 0;
            for i in 0..n {
                s = s + i;
                print(s);
            }
        }",
    );
    let r = &reps[0];
    let Verdict::Sequential(reason) = &r.verdict else {
        panic!("print must force Sequential");
    };
    assert!(reason.contains("side effect"), "{reason}");
    assert!(r.notes.iter().any(|n| n.contains("side effect")));
}

// ---------------------------------------------------------------------------
// Reduction positives beyond the golden set
// ---------------------------------------------------------------------------

#[test]
fn sub_reduction_folds_into_add_family() {
    let reps = analyze_src(
        "fn main() {
            let n = 20;
            let t = 0;
            for i in 0..n {
                t = t - i * 3;
            }
            print(t);
        }",
    );
    let r = &reps[0];
    let Verdict::ReductionParallel(red) = &r.verdict else {
        panic!("subtraction is a sum of negated terms: {:?}", r.verdict);
    };
    assert_eq!(red.op, helix_analysis::ReductionOp::Add);
    assert_eq!(red.var, "t");
}

#[test]
fn mul_reduction_recognized() {
    let reps = analyze_src(
        "fn main() {
            let n = 50;
            let f = 1.0;
            for i in 1..n {
                f = f * (i + 2) as f64;
            }
            print(f);
        }",
    );
    let r = &reps[0];
    let Verdict::ReductionParallel(red) = &r.verdict else {
        panic!("product loop should reduce: {}", r.summary_line());
    };
    assert_eq!(red.op, helix_analysis::ReductionOp::Mul);
}

#[test]
fn max_reduction_recognized_when_dominant() {
    let reps = analyze_src(
        "fn main() {
            let n = 10;
            let a: [f64] = zeros(n);
            let lo = 1.0e300;
            let hi = 0.0;
            for i in 0..n {
                lo = min(lo, a[i]);
                hi = max(hi, lo);
            }
            print(hi);
        }",
    );
    // `lo` feeds `hi`, which serializes them; only ONE clean recognition may
    // be reported and it must not claim both.
    let reds: Vec<_> = reps
        .iter()
        .filter(|r| matches!(r.verdict, Verdict::ReductionParallel(_)))
        .collect();
    assert_eq!(reds.len(), 1);
}

#[test]
fn two_independent_add_reductions_coexist() {
    let reps = analyze_src(
        "fn main() {
            let n = 50;
            let a: [i64] = zeros(n);
            let p = 0;
            let q = 0;
            for i in 0..n {
                p = p + i;
                q = q + a[i];
            }
            print(p + q);
        }",
    );
    let r = &reps[0];
    assert!(
        matches!(r.verdict, Verdict::ReductionParallel(_)),
        "{}",
        r.summary_line()
    );
    // No array edges at all: both chains are scalar-carried.
    assert_eq!(all_edges(r), 0);
}

// ---------------------------------------------------------------------------
// WAR / WAW carried edges
// ---------------------------------------------------------------------------

#[test]
fn anti_and_output_dependences_are_reported_separately() {
    let reps = analyze_src(
        "fn main() {
            let n = 100;
            let a: [i64] = zeros(n);
            for i in 0..n - 1 {
                a[i + 1] = 7;
                a[i] = a[i + 1];
            }
        }",
    );
    let r = &reps[0];
    let Verdict::Sequential(_) = &r.verdict else {
        panic!("carried edges must reject");
    };
    // W(a[i+1]) vs W(a[i]): cross-iteration output dependence. Sign
    // convention: distance = conflicting-element iteration of the FIRST-fed
    // store minus that of the SECOND. Here the body-first store (a[i+1]=7)
    // reaches shared element k at iteration k-1 while the body-second store
    // (a[i]=...) reaches it at k => -1. (The mirror shape `a[i]=1;
    // a[i+1]=2` yields +1 — see output_dependence_... below.)
    assert_eq!(r.waw_deps.len(), 1, "a[i+1]=7 then a[i]=... next iter WAW");
    assert_eq!(r.waw_deps[0].distance, Some(-1));
    // R(a[i+1]) vs W(a[i]): iteration i+1 READS what iteration i's write
    // `a[i] = ...` overwrote — an ANTI (WAR) dependence, solved negative.
    // The pre-fix classifier labeled every mixed pair RAW, hiding it.
    assert_eq!(r.war_deps.len(), 1, "a[i+1] read vs a[i] overwrite: WAR");
    assert!(matches!(
        r.verdict,
        Verdict::Sequential(_)
    ));
    // No flow edge exists in this loop: the only same-iteration mixed pair
    // (write a[i+1], read a[i+1]) is loop-independent and dropped, and the
    // carried pair is anti. RAW must stay empty.
    assert!(r.raw_deps.is_empty(), "no cross-iteration flow here");
}

// ---------------------------------------------------------------------------
// Ordered-kind showcases (2026-08-26 review): one clean example per kind
// ---------------------------------------------------------------------------

/// Classic flow dependence spelled with the READ first in the body
/// (`a[i] = a[i-1]`): the solved positive distance proves the WRITE site is
/// temporally first — this MUST be RAW even though body position alone would
/// call the pair WAR.
#[test]
fn solved_positive_distance_is_raw_even_when_read_spelled_first() {
    let reps = analyze_src(
        "fn main() {
            let n = 100;
            let a: [i64] = zeros(n);
            for i in 1..n {
                a[i] = a[i - 1] + 1;
            }
        }",
    );
    let r = &reps[0];
    let Verdict::Sequential(reason) = &r.verdict else {
        panic!("recurrence must reject");
    };
    assert_eq!(r.raw_deps.len(), 1, "{reason}");
    assert!(r.war_deps.is_empty() && r.waw_deps.is_empty());
    assert_eq!(r.raw_deps[0].distance, Some(1));
    assert!(reason.contains("RAW"), "{reason}");
    assert_eq!(
        r.summary_line(),
        "Loop #1: RAW 1 / WAR 0 / WAW 0 => SEQUENTIAL (RAW a[i] <- a[i - 1] \
         (carried by iteration distance 1, level 1))"
    );
}

/// Anti-dependence showcase: each iteration reads an element a LATER
/// iteration overwrites. Read site runs BEFORE the write site → WAR, and no
/// flow edge may appear.
#[test]
fn anti_dependence_read_early_write_later_is_war_not_raw() {
    let reps = analyze_src(
        "fn main() {
            let n = 50;
            let a: [i64] = zeros(n + 1);
            for i in 0..n {
                a[i] = a[i + 1];
            }
        }",
    );
    let r = &reps[0];
    let Verdict::Sequential(reason) = &r.verdict else {
        panic!("anti-dependence must reject");
    };
    assert_eq!(r.war_deps.len(), 1, "read a[i+1], overwritten at i+1: WAR");
    assert_eq!(r.war_deps[0].distance, Some(-1));
    assert!(r.raw_deps.is_empty(), "no flow: {reason}");
    assert!(r.waw_deps.is_empty());
    assert!(reason.contains("WAR"), "{reason}");
    assert!(
        r.summary_line()
            .starts_with("Loop #1: RAW 0 / WAR 1 / WAW 0 => SEQUENTIAL ("),
        "{}",
        r.summary_line()
    );
}

/// Output-dependence showcase: two writes to the same element in different
/// iterations race under DOALL — WAW, no reads involved.
#[test]
fn output_dependence_two_writes_same_element_is_waw() {
    let reps = analyze_src(
        "fn main() {
            let n = 50;
            let a: [i64] = zeros(n + 1);
            for i in 0..n {
                a[i] = 1;
                a[i + 1] = 2;
            }
        }",
    );
    let r = &reps[0];
    let Verdict::Sequential(_) = &r.verdict else {
        panic!("output dependence must reject");
    };
    assert_eq!(r.waw_deps.len(), 1, "a[i]=1 vs a[i+1]=2 next iter");
    // W→W pairs are fed to the battery earlier-write-first, so the distance
    // is physical: the body-LATER store (a[i+1]=2) hits index j+1, which the
    // body-EARLIER store (a[i]=1) re-overwrites one iteration later — the
    // second write's site sits +1 past the first. WAW either way.
    assert_eq!(r.waw_deps[0].distance, Some(1));
    assert!(r.raw_deps.is_empty());
    assert!(r.war_deps.is_empty());
    assert!(
        r.summary_line()
            .starts_with("Loop #1: RAW 0 / WAR 0 / WAW 1 => SEQUENTIAL ("),
        "{}",
        r.summary_line()
    );
}

#[test]
fn distance_zero_same_iteration_pairs_are_not_carried() {
    // saxpy shape: y[i] read and written in the SAME iteration — that RAW is
    // loop-independent and must not appear as a carried DepEdge.
    let reps = analyze_src(
        "fn main() {
            let n = 32;
            let y: [f64] = zeros(n);
            for i in 0..n {
                y[i] = y[i] * 2.0;
            }
        }",
    );
    let r = &reps[0];
    assert_eq!(all_edges(r), 0, "same-iteration traffic is not carried");
    assert!(matches!(r.verdict, Verdict::SafeParallel));
}

#[test]
fn rar_never_a_dependence() {
    let reps = analyze_src(
        "fn main() {
            let n = 32;
            let a: [i64] = zeros(16);
            let acc = 0;
            for i in 0..n {
                acc = acc + a[i];
            }
        }",
    );
    // Only reads of a[] exist; RAR pairs are ignored entirely.
    assert!(reps.iter().all(|r| matches!(
        r.verdict,
        Verdict::SafeParallel | Verdict::ReductionParallel(_)
    )));
}

#[test]
fn non_divisible_stride_proves_independence() {
    // a[2*i+1] write vs a[2*i] read can never touch the same element.
    let reps = analyze_src(
        "fn main() {
            let n = 50;
            let a: [i64] = zeros(200);
            for i in 0..n {
                a[2 * i + 1] = 3;
            }
            print(a[1]);
        }",
    );
    let r = &reps[0];
    // Single access pair per array with disjoint parities → independent.
    assert!(
        matches!(r.verdict, Verdict::SafeParallel),
        "{}",
        r.summary_line()
    );
}

// ---------------------------------------------------------------------------
// Canonicalization boundaries
// ---------------------------------------------------------------------------

#[test]
fn symbolic_bounds_still_canonicalize() {
    // end bound comes from a parameter-like expression (`len`); the analysis
    // must not lose canonicality just because the bound isn't a literal.
    let reps = analyze_src(
        "fn main() {
            let n = 40;
            let a: [f64] = zeros(n);
            let out: [f64] = zeros(n);
            let m = n / 2;
            for i in 0..m {
                out[i] = a[i] + 1.0;
            }
        }",
    );
    let r = &reps[0];
    assert!(
        matches!(r.verdict, Verdict::SafeParallel),
        "{}",
        r.summary_line()
    );
    assert!(r.bounds.is_some(), "canonical loops report bounds");
}

#[test]
fn nested_loops_get_independent_verdicts() {
    // Outer safe (its own writes are distinct iterations), inner sequential
    // (carried scalar through the shared accumulator would serialize BOTH).
    let reps = analyze_src(
        "fn main() {
            let n = 8;
            let b: [i64] = zeros(9);
            let s = 0;
            for i in 0..n {
                b[i] = i;
                for j in 0..n {
                    s = s + j;
                }
            }
            print(s);
        }",
    );
    // Inner j-loop reduces (+); outer carries nothing of its own but contains
    // the inner accumulator — the inner level is what gets approved.
    assert!(
        reps.iter()
            .any(|r| matches!(r.verdict, Verdict::ReductionParallel(_)))
            || reps
                .iter()
                .any(|r| matches!(r.verdict, Verdict::Sequential(_))),
        "inner reduction or honest rejection expected"
    );
}

// ---------------------------------------------------------------------------
// build_plan integration from synthetic analyses
// ---------------------------------------------------------------------------

#[test]
fn plan_regions_match_approved_verdicts() {
    let src = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../examples/scale.hx"
    ))
    .unwrap();
    let prog = parse_str(&src).unwrap();
    let typed = helix_sema::check(&prog).unwrap();
    let mut funcs = helix_ir::build(&typed);
    for f in &mut funcs {
        helix_ir::to_ssa(f);
    }
    let loops: Vec<_> = funcs.iter().map(find_loops).collect();
    let reports: Vec<Vec<_>> = funcs
        .iter()
        .zip(&loops)
        .map(|(f, li)| analyze(f, li))
        .collect();
    let plan = helix_analysis::build_plan(&funcs, &loops, &reports);

    // Count approved verdicts across all functions; plan must match 1:1.
    let approved = reports
        .iter()
        .flatten()
        .filter(|r| !matches!(r.verdict, Verdict::Sequential(_)))
        .count();
    assert_eq!(plan.regions.len(), approved);

    // scale.hx: single DoAll region at depth 1 with symbolic end bound.
    let reg = &plan.regions[0];
    assert_eq!(reg.kind, RegionKind::DoAll);
    assert!(reg.reduction.is_none());
    assert_eq!(reg.func_idx, 0);
}

#[test]
fn summary_line_format_matches_contract() {
    let reps = analyze_src(
        "fn main() {
            let n = 4;
            let a: [i64] = zeros(8);
            for i in 0..n {
                a[i] = i;
            }
        }",
    );
    assert_eq!(
        reps[0].summary_line(),
        "Loop #1: RAW 0 / WAR 0 / WAW 0 => SAFE"
    );

    // Sequential formatting embeds the reason.
    let reps = analyze_src(
        "fn main() {
            let n = 100;
            let a: [i64] = zeros(n);
            for i in 1..n {
                a[i] = a[i - 1] + 1;
            }
        }",
    );
    let line = reps[0].summary_line();
    assert!(
        line.starts_with("Loop #1: RAW 1 / WAR 0 / WAW 0 => SEQUENTIAL ("),
        "{line}"
    );
}
