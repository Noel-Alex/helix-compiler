//! Seeded-reduction exactness across worker counts.
//!
//! A reduction whose seed is NOT the operator's identity must fold exactly:
//! integer `+` and `*` are associative and commutative modulo 2^64, so every
//! worker split — `HELIX_NTHREADS` = 1, 2, 4, 8 — has to print the same bits,
//! and those bits have to equal the interpreter's sequential order. A
//! dropped, doubled, or width-mismatched seed (the historical bugs: the seed
//! read back from a widened stash word, or counted once per participant
//! instead of once per loop) surfaces as a value mismatch at some swept
//! count, never as a slowdown.
//!
//! Both kernels run at n = 200_000, far above the cost gate's serial floor
//! (`SERIAL_FLOOR` = `GRAIN` × nthreads ≤ 8192 at the swept counts), so on
//! any multi-core host the fork/join combine — not the inline fallback — is
//! what each run exercises. Every run goes through the analysis-derived
//! `ParallelPlan`, so the region under test is the one the compiler
//! approved.

use helix_backend::JitEngine;
use helix_ir::{to_ssa, verify};

/// Full frontend: parse + check + build + SSA (the backend's input contract).
fn compile_ir(src: &str) -> Vec<helix_ir::FuncIr> {
    let ast = helix_syntax::parse_str(src).expect("parse");
    let typed = helix_sema::check(&ast).expect("sema");
    let mut irs = helix_ir::build(&typed);
    for f in &mut irs {
        verify(f).unwrap_or_else(|e| panic!("pre-SSA verify of '{}': {e}", f.name));
        to_ssa(f);
        verify(f).unwrap_or_else(|e| panic!("SSA verify of '{}': {e}", f.name));
    }
    irs
}

/// Analysis pipeline: loops → reports → ParallelPlan.
fn analyze_plan(irs: &[helix_ir::FuncIr]) -> helix_analysis::ParallelPlan {
    let li: Vec<_> = irs.iter().map(helix_analysis::find_loops).collect();
    let reps: Vec<_> = irs
        .iter()
        .zip(&li)
        .map(|(f, l)| helix_analysis::analyze(f, l))
        .collect();
    helix_analysis::build_plan(irs, &li, &reps)
}

/// Maps an analysis plan onto the backend's seam type (same 1:1 shape the
/// CLI uses).
fn to_backend_plan(p: &helix_analysis::ParallelPlan) -> helix_backend::ParallelPlan {
    use helix_backend::engine::helix_analysis_stub::ReductionOp as BOp;
    let mut out = helix_backend::ParallelPlan::default();
    for r in &p.regions {
        out.regions.push(helix_backend::RegionDesc {
            func_idx: r.func_idx,
            header: r.header,
            kind: match r.kind {
                helix_analysis::RegionKind::DoAll => helix_backend::RegionKind::DoAll,
                helix_analysis::RegionKind::Reduction(op) => {
                    helix_backend::RegionKind::Reduction(match op {
                        helix_analysis::ReductionOp::Add => BOp::Add,
                        helix_analysis::ReductionOp::Mul => BOp::Mul,
                        helix_analysis::ReductionOp::Min => BOp::Min,
                        helix_analysis::ReductionOp::Max => BOp::Max,
                    })
                }
            },
            body_fn_name: r.body_fn_name.clone(),
        });
    }
    out
}

/// Sets `name=value` for the guard's lifetime, restoring the previous value
/// on drop. Mirrors helix-bench's campaign guard: the runtime reads
/// `HELIX_NTHREADS` at dispatch, so the override must be live exactly around
/// one `run_main`.
struct EnvGuard {
    name: &'static str,
    prev: Option<String>,
}

impl EnvGuard {
    fn set(name: &'static str, value: &str) -> Self {
        let prev = std::env::var(name).ok();
        // SAFETY: every run happens under the engine's serial lock, so no
        // other thread reads or writes the process environment concurrently
        // (the same invariant helix-bench's guard relies on).
        unsafe { std::env::set_var(name, value) };
        Self { name, prev }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        // SAFETY: same serialized critical section as `set`.
        match &self.prev {
            Some(v) => unsafe { std::env::set_var(self.name, v) },
            None => unsafe { std::env::remove_var(self.name) },
        }
    }
}

/// Worker counts swept by every test below.
const THREADS: [usize; 4] = [1, 2, 4, 8];

/// Compiles `src` with its analysis-derived plan and runs it with
/// `HELIX_NTHREADS=threads`, returning `(printed lines, planned regions)`.
///
/// Holds the engine serial lock across compile and run: prints are captured
/// into process-global state and the thread override is process-global too.
fn planned_lines_at(src: &str, threads: usize) -> (Vec<String>, usize) {
    let _serial = helix_backend::testutil::serial_lock();
    let _env = EnvGuard::set("HELIX_NTHREADS", &threads.to_string());
    let irs = compile_ir(src);
    let plan = analyze_plan(&irs);
    let regions = plan.regions.len();
    let engine =
        JitEngine::compile(&irs, &to_backend_plan(&plan), false).expect("JIT compile (plan)");
    let (lines, result) = helix_backend::testutil::capture_prints(|| engine.run_main());
    result.expect("JIT run (plan)");
    (lines, regions)
}

/// The reference interpreter's printed lines for the same source.
fn interp_lines(src: &str) -> Vec<String> {
    let ast = helix_syntax::parse_str(src).expect("parse");
    let typed = helix_sema::check(&ast).expect("sema");
    helix_engine::run_with_source(src, &typed)
        .expect("interpreter run")
        .printed
}

/// `let s = 17; s = s + a[i]` — a non-identity seed through the additive
/// combine. The seed must enter the total exactly once whatever the worker
/// split, and the integer sum is bit-identical to the sequential order.
#[test]
fn seeded_add_reduction_is_exact_across_worker_counts() {
    let src = r#"
        fn main() {
            let n = 200000;
            let a: [i64] = zeros(n);
            for i in 0..n {
                a[i] = i % 97;
            }
            let s = 17;
            for i in 0..n {
                s = s + a[i];
            }
            print(s);
        }
    "#;
    // Independent expected value: the sequential order, computed here.
    let mut want = 17i64;
    for i in 0..200000i64 {
        want += i % 97;
    }
    let (one, regions) = planned_lines_at(src, 1);
    assert_eq!(regions, 2, "init loop + reduction region");
    assert_eq!(one, vec![want.to_string()], "seed counted once, exact sum");
    for p in THREADS {
        let (lines, _) = planned_lines_at(src, p);
        assert_eq!(lines, one, "HELIX_NTHREADS={p} changed the sum");
    }
    assert_eq!(one, interp_lines(src), "planned JIT vs interpreter");
}

/// `let p = 3; p = p * a[i]` — the same property through the multiply
/// combine. The product overflows i64 many times over, which is the point:
/// wrapping i64 multiplication is still associative modulo 2^64, so any
/// chunked-and-recombined grouping must reproduce the interpreter's
/// sequential product bit for bit. The elements are all odd, so the wrapped
/// product never collapses to zero — a dropped, doubled, or mis-narrowed
/// seed cannot hide behind a zero total.
#[test]
fn seeded_mul_reduction_is_exact_across_worker_counts() {
    let src = r#"
        fn main() {
            let n = 200000;
            let a: [i64] = zeros(n);
            for i in 0..n {
                a[i] = 1 + 2 * (i % 3);
            }
            let p = 3;
            for i in 0..n {
                p = p * a[i];
            }
            print(p);
        }
    "#;
    // Independent expected value: the wrapped sequential product.
    let mut want = 3i64;
    for i in 0..200000i64 {
        want = want.wrapping_mul(1 + 2 * (i % 3));
    }
    let (one, regions) = planned_lines_at(src, 1);
    assert_eq!(regions, 2, "init loop + reduction region");
    assert_eq!(
        one,
        vec![want.to_string()],
        "seed counted once, exact product"
    );
    for p in THREADS {
        let (lines, _) = planned_lines_at(src, p);
        assert_eq!(lines, one, "HELIX_NTHREADS={p} changed the product");
    }
    assert_eq!(one, interp_lines(src), "planned JIT vs interpreter");
}
