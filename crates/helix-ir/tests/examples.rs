//! Whole-pipeline gauntlet over every example program shipped with the repo:
//! build → to_ssa → verify → full pass fixpoint → verify. Any example that
//! breaks any stage fails here — this is the cheapest end-to-end regression
//! net for the IR crate.

use helix_ir::{build, print_ir, run_passes_to_fixpoint, to_ssa, verify};

const EXAMPLES: &[(&str, &str)] = &[
    ("ssa_demo", include_str!("../../../examples/ssa_demo.hx")),
    ("shortcircuit", include_str!("../../../examples/shortcircuit.hx")),
    
    ("matmul", include_str!("../../../examples/matmul.hx")),
    ("dot_reduction", include_str!("../../../examples/dot_reduction.hx")),
    ("fib_recursion", include_str!("../../../examples/fib_recursion.hx")),
    ("gcd_box_test", include_str!("../../../examples/gcd_box_test.hx")),
    ("jacobi_2d", include_str!("../../../examples/jacobi_2d.hx")),
    ("count_primes_sieve", include_str!("../../../examples/count_primes_sieve.hx")),
    ("minmax_reduction", include_str!("../../../examples/minmax_reduction.hx")),
    ("scale", include_str!("../../../examples/scale.hx")),
    ("div_guard", include_str!("../../../examples/div_guard.hx")),
    ("casts_demo", include_str!("../../../examples/casts_demo.hx")),
    ("const_globals", include_str!("../../../examples/const_globals.hx")),
    ("small_n", include_str!("../../../examples/small_n.hx")),
];

#[test]
fn all_examples_survive_the_whole_pipeline() {
    let mut failures = Vec::new();
    for (name, src) in EXAMPLES {
        let ast = match helix_syntax::parse_str(src) {
            Ok(a) => a,
            Err(e) => {
                failures.push(format!("{name}: parse: {e}"));
                continue;
            }
        };
        let typed = match helix_sema::check(&ast) {
            Ok(t) => t,
            Err(diags) => {
                failures.push(format!("{name}: sema: {} diags", diags.len()));
                continue;
            }
        };
        let irs = build(&typed);
        if irs.is_empty() {
            failures.push(format!("{name}: no functions built"));
            continue;
        }
        for mut f in irs {
            let fname = f.name.clone();
            if let Err(e) = verify(&f) {
                failures.push(format!("{name}/{fname}: pre-SSA verify: {e}"));
                continue;
            }
            to_ssa(&mut f);
            if let Err(e) = verify(&f) {
                failures.push(format!("{name}/{fname}: SSA verify: {e}\n{}", print_ir(&f, true)));
                continue;
            }
            // Full pass pipeline with verification inside the driver.
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run_passes_to_fixpoint(&mut f);
            }))
            .unwrap_or_else(|_| failures.push(format!("{name}/{fname}: pass pipeline panicked")));
            if let Err(e) = verify(&f) {
                failures.push(format!("{name}/{fname}: post-pass verify: {e}"));
            }
            if is_ssa(&f) {
                if let Err(e) = helix_ir::verify_ssa(&f) {
                    failures.push(format!("{name}/{fname}: strict SSA verify: {e}"));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{} example(s) failed:\n{}", failures.len(), failures.join("\n\n"));
}

fn is_ssa(f: &helix_ir::FuncIr) -> bool {
    helix_ir::is_ssa(f)
}
