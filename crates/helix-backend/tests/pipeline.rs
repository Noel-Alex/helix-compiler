//! Whole-pipeline smoke tests: HELIX source → parse → check → build →
//! `to_ssa` → [`JitEngine`] → native execution.
//!
//! These complement `jit_pipeline.rs` (differential against the interpreter)
//! by pinning a handful of outputs to **hardcoded** spec values, so a shared
//! misreading between the two backends cannot pass silently.

use helix_backend::JitEngine;
use helix_ir::{build, to_ssa};
use helix_sema::check;
use helix_syntax::parse_str;

/// Full frontend: parse + check + build + SSA (the backend's input contract).
fn compile_ir(src: &str) -> Vec<helix_ir::FuncIr> {
    let ast = parse_str(src).expect("parse");
    let typed = check(&ast).expect("sema");
    let mut irs = build(&typed);
    for f in &mut irs {
        to_ssa(f);
    }
    irs
}

/// Compiles `src` and runs it under a print capture, returning the lines.
///
/// Holds the serial lock for the whole run: the trap recorder is
/// process-global, and an UNLOCKED run racing a `trapped()` test could hit a
/// guard in the disarmed window — where `helix_panic` exits the process and
/// kills the entire test binary (observed as flaky "test exited abnormally").
fn jit_prints(src: &str) -> Vec<String> {
    let _guard = helix_backend::testutil::serial_lock();
    let irs = compile_ir(src);
    let plan = helix_backend::ParallelPlan::default();
    let engine = JitEngine::compile(&irs, &plan, false).expect("JIT compile");
    let (lines, result) = helix_backend::testutil::capture_prints(|| engine.run_main());
    result.expect("JIT run");
    lines
}

// ---------------------------------------------------------------------------
// Arithmetic and control flow
// ---------------------------------------------------------------------------

#[test]
fn consts_and_iadd_run_through_jit() {
    assert_eq!(
        jit_prints("fn main() {\n    print(2 + 3);\n    print(-7 * 4);\n    print(100 - 1);\n}\n"),
        vec!["5", "-28", "99"]
    );
}

#[test]
fn division_and_remainder_sign_semantics() {
    // Spec: `/` truncates toward zero; `%` takes the DIVIDEND's sign.
    assert_eq!(
        jit_prints(
            "fn main() {\n    let a = -7;\n    let b = 2;\n    print(a % b);\n    print(7 % -2);\n    print(a / b);\n}\n"
        ),
        vec!["-1", "1", "-3"]
    );
}

#[test]
fn float_arithmetic_and_bool_printing() {
    assert_eq!(
        jit_prints(
            "fn main() {\n    print(2.5 * 2.0);\n    print(1.0 / 4.0);\n    print(true);\n    print(1.5 < 2.5);\n}\n"
        ),
        vec!["5.0", "0.25", "true", "true"]
    );
}

#[test]
fn branches_loops_and_phi_join() {
    // The loop accumulator exercises φ placement → Cranelift block params on
    // both loop edges; the if/else diamond exercises branch edge arguments.
    assert_eq!(
        jit_prints(
            "fn main() {\n    let n = 10;\n    let acc = 0;\n    for i in 0..n {\n        acc = acc + i;\n    }\n    print(acc);\n\n    let x = 3;\n    if x > 2 {\n        print(100);\n    } else {\n        print(200);\n    }\n}\n"
        ),
        vec!["45", "100"]
    );
}

#[test]
fn arrays_zeros_len_and_stores() {
    assert_eq!(
        jit_prints(
            "fn main() {\n    let a: [i64] = zeros(4);\n    a[0] = 42;\n    a[3] = a[0] + 8;\n    print(a[3]);\n    print(len(a));\n}\n"
        ),
        vec!["50", "4"]
    );
}

#[test]
fn user_calls_and_recursion() {
    assert_eq!(
        jit_prints(
            "fn add(a: i64, b: i64) -> i64 {\n    return a + b;\n}\nfn fib(n: i64) -> i64 {\n    if n < 2 {\n        return n;\n    }\n    return fib(n - 1) + fib(n - 2);\n}\nfn main() {\n    print(add(20, 22));\n    print(fib(15));\n}\n"
        ),
        vec!["42", "610"]
    );
}

// ---------------------------------------------------------------------------
// Runtime-error guards (trap recorder instead of process exit)
// ---------------------------------------------------------------------------

/// Runs `src` with the trap recorder armed, returning `(code, aux_a, aux_b)`.
/// `(None, ..)` means the program completed without trapping — itself a
/// useful assertion when paired with expected-output checks.
fn trapped(src: &str) -> Option<(i64, i64, i64)> {
    let _guard = helix_backend::testutil::serial_lock();
    let irs = compile_ir(src);
    let plan = helix_backend::ParallelPlan::default();
    let engine = JitEngine::compile(&irs, &plan, false).expect("JIT compile");
    helix_backend::testutil::arm_trap_recorder();
    let (_, result) = helix_backend::testutil::capture_prints(|| engine.run_main());
    let trap = helix_backend::testutil::take_last_trap();
    helix_backend::testutil::disarm_trap_recorder();
    // Either the run completed cleanly (no trap) or it halted at the trap's
    // defined fallback return (run_main still reports Ok).
    let _ = result;
    trap
}

#[test]
fn out_of_bounds_load_traps_with_bounds_code() {
    let src = "fn main() {\n    let a: [i64] = zeros(3);\n    let i = 7;\n    print(a[i]);\n}\n";
    let (code, idx, len) = trapped(src).expect("must trap");
    assert_eq!(code, helix_backend::testutil::codes::BOUNDS);
    assert_eq!((idx, len), (7, 3));
}

#[test]
fn negative_store_index_traps_too() {
    let src = "fn main() {\n    let a: [i64] = zeros(3);\n    let i = 0 - 2;\n    a[i] = 5;\n}\n";
    let (code, _, _) = trapped(src).expect("must trap");
    assert_eq!(code, helix_backend::testutil::codes::BOUNDS);
}

#[test]
fn division_by_zero_traps() {
    let src = "fn main() {\n    let z = 0;\n    print(10 / z);\n}\n";
    let (code, _, _) = trapped(src).expect("must trap");
    assert_eq!(code, helix_backend::testutil::codes::DIV_BY_ZERO);
}

#[test]
fn remainder_by_zero_traps() {
    let src = "fn main() {\n    let z = 0;\n    print(10 % z);\n}\n";
    let (code, _, _) = trapped(src).expect("must trap");
    assert_eq!(code, helix_backend::testutil::codes::DIV_BY_ZERO);
}

#[test]
fn min_div_minus_one_overflow_edge_traps() {
    // Written as `-MAX - 1` because sema range-checks bare literals.
    let src = "fn main() {\n    let m = -9223372036854775807 - 1;\n    print(m / -1);\n}\n";
    let (code, _, _) = trapped(src).expect("must trap");
    assert_eq!(code, helix_backend::testutil::codes::DIV_OVERFLOW);
}

#[test]
fn healthy_program_records_no_trap() {
    let src = "fn main() {\n    let a: [i64] = zeros(2);\n    a[1] = 6;\n    print(a[1]);\n    print(8 / 2);\n}\n";
    let _guard = helix_backend::testutil::serial_lock();
    assert_eq!(jit_prints(src), vec!["6", "4"]);
    assert_eq!(
        helix_backend::testutil::take_last_trap(),
        None,
        "guards must stay silent on in-range accesses and safe divisions"
    );
}

#[test]
fn unchecked_strips_bounds_but_keeps_division_guards() {
    let _guard = helix_backend::testutil::serial_lock();
    let irs = compile_ir("fn main() {\n    let z = 0;\n    print(5 / z);\n}\n");
    let engine = JitEngine::compile(&irs, &helix_backend::ParallelPlan::default(), true)
        .expect("unchecked compile");
    helix_backend::testutil::arm_trap_recorder();
    let (_, r) = helix_backend::testutil::capture_prints(|| engine.run_main());
    let trap = helix_backend::testutil::take_last_trap();
    helix_backend::testutil::disarm_trap_recorder();
    let _ = r;
    assert_eq!(
        trap.map(|(c, _, _)| c),
        Some(helix_backend::testutil::codes::DIV_BY_ZERO),
        "--unchecked removes bounds guards only; division traps stay"
    );
}
