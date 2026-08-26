//! Golden end-to-end verdicts over the repository's example kernels.
//!
//! These tests run the full analysis pipeline — `parse_str` → `check` →
//! `build` → `to_ssa` → `find_loops` → `analyze` → `build_plan` — on real
//! HELIX sources and assert the exact verdict the dependence engine must
//! produce. They are the acceptance gate for M8: every SAFE/REDUCTION/
//! SEQUENTIAL label shown in the demo slides traces back to one test here.

#![forbid(unsafe_code)]

use helix_analysis::{ReductionOp, Verdict, analyze, build_plan, find_loops};
use helix_syntax::parse_str;

/// Source path helper: examples live in the workspace root.
const EXAMPLES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../examples");

struct LoopVerdicts {
    reports: Vec<helix_analysis::LoopReport>,
    funcs: Vec<helix_ir::FuncIr>,
    loops: Vec<helix_analysis::LoopInfo>,
}

impl LoopVerdicts {
    /// First report whose loop id matches (loops are discovered in a stable
    /// order: smaller bodies first, so inner loops come before their parents).
    fn of(&self, idx: usize) -> &helix_analysis::LoopReport {
        &self.reports[idx]
    }
}

fn analyze_example(name: &str) -> LoopVerdicts {
    let src = std::fs::read_to_string(format!("{EXAMPLES}/{name}.hx"))
        .unwrap_or_else(|e| panic!("read {name}.hx: {e}"));
    let prog = parse_str(&src).unwrap_or_else(|e| panic!("parse {name}: {e}"));
    let typed =
        helix_sema::check(&prog).unwrap_or_else(|ds| panic!("check {name}: {:?}", ds.first()));
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
    assert_eq!(reports.len(), loops.len(), "one report list per function");
    LoopVerdicts {
        reports: reports.into_iter().flatten().collect(),
        funcs,
        loops,
    }
}

fn summary(v: &LoopVerdicts, idx: usize) -> String {
    v.of(idx).summary_line()
}

// ---------------------------------------------------------------------------
// Streaming kernels — plain DOALL
// ---------------------------------------------------------------------------

#[test]
fn scale_is_safe() {
    let v = analyze_example("scale");
    assert_eq!(v.reports.len(), 1, "one loop");
    let r = v.of(0);
    assert!(
        matches!(r.verdict, Verdict::SafeParallel),
        "{}",
        r.summary_line()
    );
    assert!(r.raw_deps.is_empty() && r.war_deps.is_empty() && r.waw_deps.is_empty());
    // Access card: read + write on distinct arrays.
    assert_eq!(r.accesses.len(), 2);
    assert!(r.accesses[0].starts_with("READ a[i]"));
    assert!(r.accesses[1].starts_with("WRITE out[i]"));
}

#[test]
fn saxpy_is_safe_despite_read_write_pair_on_y() {
    let v = analyze_example("saxpy");
    let r = v.of(0);
    assert!(
        matches!(r.verdict, Verdict::SafeParallel),
        "{}",
        r.summary_line()
    );
    // x[i], y[i] reads and y[i] write: same-iteration RAW only (not carried),
    // which is why the verdict survives.
    assert_eq!(r.accesses.len(), 3);
}

#[test]
fn small_n_still_safe_verdict_even_though_runtime_gates() {
    let v = analyze_example("small_n");
    let r = v.of(0);
    // Verdicts are about DEPENDENCE; trip-count profitability is the
    // runtime's grain-size decision, deliberately not folded in here.
    assert!(matches!(r.verdict, Verdict::SafeParallel));
}

// ---------------------------------------------------------------------------
// Recurrence — the canonical rejection with an exact explanation
// ---------------------------------------------------------------------------

#[test]
fn recurrence_is_sequential_with_distance_one_raw() {
    let v = analyze_example("recurrence_reject");
    let r = v.of(0);
    let Verdict::Sequential(reason) = &r.verdict else {
        panic!("expected Sequential, got {:?}", r.verdict);
    };
    assert!(reason.contains("RAW"), "reason: {reason}");
    assert!(reason.contains("distance 1"), "reason: {reason}");
    assert_eq!(r.raw_deps.len(), 1);
    let edge = &r.raw_deps[0];
    assert_eq!(edge.distance, Some(1));
    assert_eq!(edge.direction, ">"); // sink after source
    assert_eq!(edge.level, r.depth);
    assert!(
        edge.explain.contains('a'),
        "names the array: {}",
        edge.explain
    );
    assert!(r.war_deps.is_empty() && r.waw_deps.is_empty());
}

// ---------------------------------------------------------------------------
// Reductions — the sanctioned self-dependence
// ---------------------------------------------------------------------------

#[test]
fn dot_reduction_is_reduction_add() {
    let v = analyze_example("dot_reduction");
    let r = v.of(0);
    let Verdict::ReductionParallel(red) = &r.verdict else {
        panic!("expected ReductionParallel, got {:?}", r.verdict);
    };
    assert_eq!(red.op, ReductionOp::Add);
    assert_eq!(red.var, "dot");
    // The accumulator's own chain produced no ARRAY edges at all.
    assert!(r.raw_deps.is_empty() && r.war_deps.is_empty() && r.waw_deps.is_empty());
    assert!(r.notes.is_empty());
    assert_eq!(
        summary(&v, 0),
        "Loop #1: RAW 0 / WAR 0 / WAW 0 => REDUCTION(+)"
    );
}

#[test]
fn minmax_recognizes_min_and_max() {
    let v = analyze_example("minmax_reduction");
    let r = v.of(0);
    let Verdict::ReductionParallel(red) = &r.verdict else {
        panic!("expected ReductionParallel, got {:?}", r.verdict);
    };
    assert!(
        red.op == ReductionOp::Min || red.op == ReductionOp::Max,
        "min or max expected, got {:?}",
        red.op
    );
    assert!(
        red.var == "lo" || red.var == "hi",
        "accumulator name: {}",
        red.var
    );
}

// ---------------------------------------------------------------------------
// GCD/box — inconclusive Diophantine must reject, not approve
// ---------------------------------------------------------------------------

#[test]
fn gcd_box_test_is_sequential() {
    let v = analyze_example("gcd_box_test");
    let r = v.of(0);
    let Verdict::Sequential(reason) = &r.verdict else {
        panic!("expected Sequential, got {:?}", r.verdict);
    };
    assert!(reason.contains("RAW"), "reason: {reason}");
    assert_eq!(r.raw_deps.len(), 1);
    // a[2i] vs a[i]: distance not a single constant => None; the gcd/box
    // test leaves all crossing directions open (`<=>`).
    assert_eq!(r.raw_deps[0].distance, None);
    assert_eq!(r.raw_deps[0].direction, "<=>");
}

// ---------------------------------------------------------------------------
// Nested kernels — per-level verdicts inside one nest
// ---------------------------------------------------------------------------

#[test]
fn matmul_inner_k_loop_is_add_reduction() {
    let v = analyze_example("matmul");
    assert!(v.reports.len() >= 4, "4 loops: init/i/j/k");
    // Discovery order sorts by body size: k-loop first, then j, i, init.
    let reds: Vec<_> = v
        .reports
        .iter()
        .filter_map(|r| match &r.verdict {
            Verdict::ReductionParallel(red) => Some((r.loop_id, red.op)),
            _ => None,
        })
        .collect();
    assert_eq!(reds.len(), 1, "exactly the k-loop reduces: {reds:?}");
    assert_eq!(reds[0].1, ReductionOp::Add);
    // Every other loop is plain-safe here.
    for r in &v.reports {
        if r.loop_id != reds[0].0 {
            assert!(
                matches!(r.verdict, Verdict::SafeParallel),
                "{}",
                r.summary_line()
            );
        }
    }
}

#[test]
fn matmul_plan_contains_reduction_region() {
    let v = analyze_example("matmul");
    let mut per_fn = vec![Vec::new(); v.funcs.len()];
    for r in &v.reports {
        let fi = v
            .loops
            .iter()
            .position(|li| li.loops.iter().any(|l| l.id == r.loop_id))
            .expect("loop belongs to a function");
        per_fn[fi].push(r.clone());
    }
    let plan = build_plan(&v.funcs, &v.loops, &per_fn);
    let red_regions: Vec<_> = plan
        .regions
        .iter()
        .filter(|r| matches!(r.kind, helix_analysis::RegionKind::Reduction(_)))
        .collect();
    assert_eq!(red_regions.len(), 1);
    assert_eq!(red_regions[0].reduction, Some(ReductionOp::Add));
    assert_eq!(red_regions[0].func_idx, 0);
    assert!(red_regions[0].body_fn_name.starts_with("main.loop"));
    assert!(red_regions[0].body_fn_name.ends_with(".body"));
}

#[test]
fn jacobi_innermost_copy_and_stencil_levels() {
    let v = analyze_example("jacobi_2d");
    assert_eq!(v.reports.len(), 4, "k / i / j / copy loops");
    // Flattened-2D stencil subscripts mix two indices; single-level affine
    // analysis honestly reports them as unanalyzable and stays sequential.
    let stencil_seq = v
        .reports
        .iter()
        .any(|r| matches!(r.verdict, Verdict::Sequential(_)));
    assert!(stencil_seq, "stencil level must not be claimed as parallel");
    // The swap/copy loop (`cur[i] = next[i]`) IS safe: distinct arrays,
    // distance-0 pairs only.
    assert!(
        v.reports
            .iter()
            .any(|r| matches!(r.verdict, Verdict::SafeParallel)),
        "copy/init levels remain safe"
    );
}

// ---------------------------------------------------------------------------
// build_plan shape checks across all golden programs
// ---------------------------------------------------------------------------

#[test]
fn plan_skips_sequential_loops_everywhere() {
    for name in [
        "scale",
        "saxpy",
        "recurrence_reject",
        "dot_reduction",
        "minmax_reduction",
        "gcd_box_test",
        "matmul",
        "jacobi_2d",
        "small_n",
        "count_primes_sieve",
    ] {
        let v = analyze_example(name);
        let mut per_fn = vec![Vec::new(); v.funcs.len()];
        for r in &v.reports {
            let fi = v
                .loops
                .iter()
                .position(|li| li.loops.iter().any(|l| l.id == r.loop_id))
                .expect("report belongs to a function");
            per_fn[fi].push(r.clone());
        }
        let plan = build_plan(&v.funcs, &v.loops, &per_fn);
        // Every planned region's header must belong to a loop whose verdict
        // was NOT Sequential.
        for reg in &plan.regions {
            let rep = v
                .reports
                .iter()
                .find(|r| {
                    v.loops[reg.func_idx]
                        .loops
                        .iter()
                        .any(|l| l.id == r.loop_id && l.header == reg.header)
                })
                .unwrap_or_else(|| panic!("{name}: plan region without report"));
            assert!(
                !matches!(rep.verdict, Verdict::Sequential(_)),
                "{name}: sequential loop entered the plan ({})",
                rep.summary_line()
            );
            assert!(
                matches!(reg.kind, helix_analysis::RegionKind::DoAll) == reg.reduction.is_none(),
                "{name}: RegionKind/Reduction coherence"
            );
        }
        // recurrence_reject and gcd_box_test approve nothing at all.
        if matches!(name, "recurrence_reject" | "gcd_box_test") {
            assert!(plan.regions.is_empty(), "{name} approved something!");
        }
    }
}

#[test]
fn plan_body_names_are_unique_and_wellformed() {
    let v = analyze_example("matmul");
    let mut per_fn = vec![Vec::new(); v.funcs.len()];
    for r in &v.reports {
        let fi = v
            .loops
            .iter()
            .position(|li| li.loops.iter().any(|l| l.id == r.loop_id))
            .unwrap();
        per_fn[fi].push(r.clone());
    }
    let plan = build_plan(&v.funcs, &v.loops, &per_fn);
    let names: Vec<_> = plan.regions.iter().map(|r| &r.body_fn_name).collect();
    let unique = names.len() == std::collections::HashSet::<_>::from_iter(names.iter()).len();
    assert!(unique, "duplicate body names: {names:?}");
}

// ---------------------------------------------------------------------------
// Report formatting contract (Observatory depends on these strings)
// ---------------------------------------------------------------------------

#[test]
fn report_fields_render_for_the_ui() {
    let v = analyze_example("scale");
    let r = v.of(0);
    assert_eq!(r.header, "bb1");
    assert!(r.blocks.iter().all(|b| b.starts_with("bb")));
    assert_eq!(r.iv.as_deref(), Some("i"));
    assert_eq!(r.bounds.as_ref().map(|b| b.0.as_str()), Some("0"));
}

// ---------------------------------------------------------------------------
// 2026-08-25 review wave 2: user-call side-effect gate
// ---------------------------------------------------------------------------

/// A loop calling a USER function that prints must be SEQUENTIAL. The old
/// gate recognized only a literal inline `print` call, so `a[i] = tag(i)`
/// (tag prints) got an unsound SAFE verdict — parallelizing iterations that
/// each print is wrong regardless of memory independence.
#[test]
fn user_call_with_side_effect_vetoes_parallelization() {
    let src = r#"
        fn tag(x: i64) -> i64 {
            print(x);
            return x;
        }
        fn main() {
            let a: [i64] = zeros(8);
            for i in 0..8 {
                a[i] = tag(i);
            }
            print(a[3]);
        }
    "#;
    let prog = parse_str(src).unwrap_or_else(|e| panic!("parse: {e}"));
    let typed = helix_sema::check(&prog).unwrap_or_else(|ds| panic!("check: {:?}", ds.first()));
    let mut funcs = helix_ir::build(&typed);
    for f in &mut funcs {
        helix_ir::to_ssa(f);
    }
    let li = find_loops(&funcs[typed.main_idx()]);
    assert!(!li.loops.is_empty(), "test bug: no loop found");
    for rep in analyze(&funcs[typed.main_idx()], &li) {
        assert!(
            matches!(rep.verdict, Verdict::Sequential(_)),
            "user-call loop must be SEQUENTIAL, got {:?}",
            rep.verdict
        );
        assert!(
            rep.notes.iter().any(|n| n.contains("non-pure")),
            "expected a non-pure-call note, notes: {:?}",
            rep.notes
        );
    }
}

/// Pure builtins (min/max/sqrt/abs/len/zeros) stay exempt from the gate.
#[test]
fn pure_builtin_calls_do_not_veto() {
    let src = r#"
        fn main() {
            let n = 64;
            let a: [f64] = zeros(n);
            let out: [f64] = zeros(n);
            for i in 0..n {
                out[i] = sqrt(abs(a[i])) + min(1.0, 2.0);
            }
            print(out[7]);
        }
    "#;
    let prog = parse_str(src).unwrap_or_else(|e| panic!("parse: {e}"));
    let typed = helix_sema::check(&prog).unwrap_or_else(|ds| panic!("check: {:?}", ds.first()));
    let mut funcs = helix_ir::build(&typed);
    for f in &mut funcs {
        helix_ir::to_ssa(f);
    }
    let li = find_loops(&funcs[typed.main_idx()]);
    let reps = analyze(&funcs[typed.main_idx()], &li);
    assert!(
        reps.iter().any(|r| matches!(r.verdict, Verdict::SafeParallel)),
        "pure-builtin loop should remain SAFE, got {:?}",
        reps.iter().map(|r| r.verdict.clone()).collect::<Vec<_>>()
    );
}

// ---------------------------------------------------------------------------
// Loop-nest forest — regression for the find_loops depth/parent fix
// ---------------------------------------------------------------------------

/// find_loops sorted bodies ASCENDING while the parent search scanned
/// already-placed loops, so a nested loop could never find its (larger)
/// container: every loop of matmul came out depth=1, parent=None, and
/// build_plan approved mid-nest parallelization. Depths must now reflect
/// the true nesting, and only innermost loops enter the plan.
#[test]
fn matmul_loop_forest_has_real_depths() {
    let v = analyze_example("matmul");
    assert!(v.reports.len() >= 4, "4 loops in main: init/i/j/k");
    let depths: Vec<(usize, u32)> = v
        .reports
        .iter()
        .map(|r| (r.loop_id, r.depth))
        .collect();
    // Exactly one outermost nest level-1 chain: init at depth 1, i at 1,
    // j inside i at 2, k inside j at 3.
    let mut d1 = 0;
    let mut has_d2 = false;
    let mut has_d3 = false;
    for (_, d) in &depths {
        match *d {
            1 => d1 += 1,
            2 => has_d2 = true,
            3 => has_d3 = true,
            other => panic!("unexpected depth {other}: {depths:?}"),
        }
    }
    assert_eq!(d1, 2, "init + i live at depth 1: {depths:?}");
    assert!(has_d2, "j must sit at depth 2: {depths:?}");
    assert!(has_d3, "k must sit at depth 3: {depths:?}");
}

/// The plan approves ONLY innermost (or top-level) loops: the mid-nest j
/// loop of matmul used to slip through when depths were flat.
#[test]
fn matmul_plan_never_approves_midnest_loops() {
    let v = analyze_example("matmul");
    let mut per_fn = vec![Vec::new(); v.funcs.len()];
    for r in &v.reports {
        let fi = v
            .loops
            .iter()
            .position(|li| li.loops.iter().any(|l| l.id == r.loop_id))
            .expect("loop belongs to a function");
        per_fn[fi].push(r.clone());
    }
    let plan = build_plan(&v.funcs, &v.loops, &per_fn);
    // Every planned region's header must belong to an innermost or depth-1
    // loop; the depth-2 j loop is forbidden.
    for reg in &plan.regions {
        let lp = v.loops[reg.func_idx]
            .loops
            .iter()
            .find(|l| l.header == reg.header)
            .unwrap_or_else(|| panic!("region header bb{} not a known loop", reg.header.0));
        let rep = v
            .reports
            .iter()
            .find(|r| r.loop_id == lp.id)
            .expect("report for planned loop");
        assert!(
            lp.depth == 1 || lp.depth == rep.depth && is_innermost(&v, reg.func_idx, lp.id),
            "planned region at depth {} (loop {}) — mid-nest parallelization",
            lp.depth,
            lp.id
        );
    }
}

fn is_innermost(v: &LoopVerdicts, fi: usize, id: usize) -> bool {
    let info = &v.loops[fi];
    !info.loops.iter().any(|c| {
        c.id != id
            && c.blocks.len() < info.loops.iter().find(|l| l.id == id).expect("id").blocks.len()
            && c.blocks.iter().all(|b| {
                info.loops.iter().find(|l| l.id == id).expect("id").blocks.contains(b)
            })
    })
}
