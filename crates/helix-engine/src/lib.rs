//! # helix-engine — the reference tree-walking interpreter
//!
//! The interpreter is the project's *semantic oracle*: a direct, readable
//! execution of `docs/notes/lang-spec.md` against which the JIT backend is
//! differentially tested. Correctness and clarity outrank speed here.
//!
//! ```text
//! source ─┬─ helix_syntax::parse_str ─▶ Program ──┐
//!         │                                       ├─ adapter ─▶ EIR ─▶ Interp ─▶ RunOutput
//!         └─ helix_sema::check ───────▶ TypedProgram ─┘
//! ```
//!
//! ## Why an engine-local evaluation tree?
//!
//! `helix-sema` v1's `TypedExprKind::Call` resolves callees but does not
//! store argument sub-expressions (the checker type-checks them for
//! diagnostics, then discards). Rather than mutate another crate, the
//! [`adapter`] walks the original AST against the typed program — replaying
//! sema's sequential symbol allocation so every [`SymId`] matches exactly —
//! and emits [`EExpr`] nodes whose calls carry arguments. The typed program
//! remains the semantic authority: its symbol arenas decide names/types/kinds,
//! its const table pre-binds globals, its function table resolves callees.
//! See [`adapter`] for why this join is exact for programs sema accepted.
//!
//! ## Public surface (per interface-contracts.md)
//!
//! * [`run`] / [`run_with_source`] → [`RunOutput`] / [`RunError`]
//! * [`RunOutput::printed`] + [`RunOutput::checksum`] (FNV-1a)
//! * [`Interp`] is re-exported as a named type for documentation purposes;
//!   construction stays internal ([`execute`]).
//!
//! ## Checksum design
//!
//! The FNV-1a stream hashes, in order:
//!
//! 1. every printed line plus a `\n` terminator each (so "1","23" can never
//!    equal "12","3");
//! 2. every array still visible in `main`'s final environment, in ascending
//!    symbol order, element by element, as canonical bytes (`to_le_bytes`,
//!    float bit patterns, bools as 0/1).
//!
//! Two runs of one program therefore produce identical checksums — the
//! property the benchmark harness uses to compare backends.

#![forbid(unsafe_code)]

pub mod adapter;
pub mod error;
pub mod interp;
pub mod value;

use helix_sema::TypedProgram;
use helix_syntax::Span;

pub use crate::error::{LineMap, RunError, RunErrorKind};
pub use crate::interp::Interp;
pub use crate::value::Value;

/// Output of a successful run.
///
/// Contract type (interface-contracts.md): the lines `print` produced and a
/// content checksum stable across identical runs.
#[derive(Debug, Clone, PartialEq)]
pub struct RunOutput {
    /// One string per `print` call, in call order, without newlines.
    pub printed: Vec<String>,
    /// FNV-1a over printed lines (+`\n` each) then final array bytes.
    pub checksum: u64,
}

/// Runs a checked program from `fn main()`.
///
/// This is the contracted entry point. Because runtime errors must quote
/// source lines while the typed tree only stores byte offsets, line numbers
/// are only available when source text is supplied; prefer
/// [`run_with_source`] whenever the caller has it (the CLI always does).
///
/// # Errors
///
/// Returns the first runtime error (bounds violation, division traps,
/// negative `zeros`, stack exhaustion) with its span.
pub fn run(program: &TypedProgram) -> Result<RunOutput, RunError> {
    run_inner(program, None)
}

/// Like [`run`], but also receives the source text the program was parsed
/// from, enabling `runtime error: … at line N` messages.
///
/// This is the primary entry point used by tests and the CLI.
///
/// # Errors
///
/// Same as [`run`]; additionally, a mismatch between `src` and `program`
/// (the typed tree was not checked from this text) yields a single
/// internal-error result rather than wrong output.
pub fn run_with_source(src: &str, program: &TypedProgram) -> Result<RunOutput, RunError> {
    run_inner(program, Some(src))
}

fn run_inner(program: &TypedProgram, src: Option<&str>) -> Result<RunOutput, RunError> {
    // Re-parse to obtain AST structure sema v1 dropped (call arguments).
    let ast = match src {
        Some(text) => match helix_syntax::parse_str(text) {
            Ok(ast) => Some(ast),
            Err(e) => {
                return Err(RunError {
                    kind: RunErrorKind::Internal(format!(
                        "source no longer parses ({e}); cannot build evaluation tree"
                    )),
                    span: Span { start: 0, end: 0 },
                    printed_so_far: Vec::new(),
                });
            }
        },
        None => None,
    };

    // Join AST + typed program when both are available.
    let adapted_owned;
    let adapted: &crate::adapter::AdaptedProgram = match (&ast, src.is_some()) {
        (Some(ast), true) => match crate::adapter::adapt_program(ast, program) {
            Some(adapted) => {
                adapted_owned = adapted;
                &adapted_owned
            }
            None => {
                return Err(RunError {
                    kind: RunErrorKind::Internal(
                        "source and typed program do not correspond".to_string(),
                    ),
                    span: Span { start: 0, end: 0 },
                    printed_so_far: Vec::new(),
                });
            }
        },
        _ => {
            return Err(RunError {
                kind: RunErrorKind::Internal(
                    "no source text available: programs containing calls cannot be executed \
                     because sema v1 omits call arguments from the typed tree"
                        .to_string(),
                ),
                span: Span { start: 0, end: 0 },
                printed_so_far: Vec::new(),
            });
        }
    };

    // On failure the error carries `printed_so_far` (attached by the worker),
    // so callers can emit buffered stdout before the rendered message.
    interp::execute(adapted).map(|outcome| RunOutput {
        printed: outcome.printed,
        checksum: outcome.checksum,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interp::FNV_OFFSET;
    use helix_sema::check;

    /// Parses, checks and runs HELIX source in one step. Runtime errors are
    /// rendered with full source context (so they quote line numbers).
    pub(crate) fn run_src(src: &str) -> Result<RunOutput, String> {
        let ast = helix_syntax::parse_str(src).map_err(|e| e.to_string())?;
        let tp = check(&ast).map_err(|ds| format!("{ds:#?}"))?;
        match run_with_source(src, &tp) {
            Ok(out) => Ok(out),
            Err(e) => Err(e.render(src)),
        }
    }

    #[test]
    fn prints_and_checksum_are_stable_across_runs() {
        let src = r#"
            fn main() {
                let a: [i64] = zeros(4);
                a[0] = 5;
                a[3] = -2;
                print(a[0]);
                print(a[1]);
                print(a[2]);
                print(a[3]);
            }
        "#;
        let first = run_src(src).unwrap();
        let second = run_src(src).unwrap();
        assert_eq!(first.printed, vec!["5", "0", "0", "-2"]);
        assert_eq!(first.checksum, second.checksum);
        assert_ne!(first.checksum, FNV_OFFSET);
    }

    #[test]
    fn callee_writes_escape_to_caller_arrays() {
        let src = r#"
            fn poke(a: [i64]) {
                a[1] = 42;
            }
            fn main() {
                let xs: [i64] = zeros(3);
                poke(xs);
                print(xs[1]);
            }
        "#;
        assert_eq!(run_src(src).unwrap().printed, vec!["42"]);
    }

    #[test]
    fn recursion_fib_example() {
        // examples/fib_recursion.hx is deliberately NOT textbook Fibonacci:
        // for n >= 15 it uses fib(n) = fib(n-3) + 2*fib(n-2) - fib(n-4) + 4
        // (exercising multi-branch recursion, else-if chains, and argument
        // evaluation). Hand-computed value for n=24: 20001. The point of the
        // test is that deep mutual recursion runs and terminates.
        let out = run_src(include_str!("../../../examples/fib_recursion.hx")).unwrap();
        assert_eq!(out.printed, vec!["20001"]);
        // Cross-check the base cases, which ARE standard Fibonacci.
        for (n, want) in [(0i64, "0"), (1, "1"), (10, "55"), (14, "377")] {
            let src = format!(
                "fn fib(n: i64) -> i64 {{\n if n < 2 {{ return n; }}\n return fib(n-1)+fib(n-2);\n}}\nfn main() {{ print(fib({n})); }}\n"
            );
            assert_eq!(run_src(&src).unwrap().printed, vec![want], "fib({n})");
        }
    }

    #[test]
    fn div_guard_exact_outputs() {
        let out = run_src(include_str!("../../../examples/div_guard.hx")).unwrap();
        assert_eq!(out.printed, vec!["-1", "1", "-3"]);
    }

    #[test]
    fn shortcircuit_hit_counts() {
        // side(v) returns `v == 3`, so:
        //   side(1) && side(3): side(1) is false  -> short-circuit, side(3) NEVER runs
        //     => prints false, hits[1]=1, hits[3]=0
        //   side(2) || side(4): side(2) is false  -> side(4) DOES run (also false)
        //     => prints false, hits[2]=1, hits[4]=1
        let out = run_src(include_str!("../../../examples/shortcircuit.hx")).unwrap();
        assert_eq!(out.printed, vec!["false", "1", "0", "false", "1", "1"]);
    }

    #[test]
    fn ssa_demo_prints_10() {
        let out = run_src(include_str!("../../../examples/ssa_demo.hx")).unwrap();
        assert_eq!(out.printed, vec!["10"]);
    }

    #[test]
    fn casts_saturate_like_rust_as() {
        let out = run_src(include_str!("../../../examples/casts_demo.hx")).unwrap();
        // 300.7 -> 300; NaN -> 0.
        // `-1.0e300 as i32` parses as -(1.0e300 as i32) because `as` binds
        // TIGHTER than unary minus: the inner cast saturates to +i32::MAX,
        // then negates. A parenthesized `(-1.0e300) as i32` would saturate
        // to i32::MIN instead — pinned by casts_saturating_values below.
        assert_eq!(out.printed, vec!["300", "-2147483647", "0"]);
    }

    #[test]
    fn casts_saturating_values() {
        let src = r#"
            fn main() {
                print((-1.0e300) as i32);   // saturates to i32::MIN
                print(1.0e300 as i32);      // saturates to i32::MAX
                print((-1.0e300) as i64);   // saturates to i64::MIN
                print((0.0 / 0.0) as i64);   // NaN -> 0
                print(2147483647.6 as i32); // rounds toward zero then clamps
                print(3.9 as i64);          // truncates toward zero
            }
        "#;
        let out = run_src(src).unwrap();
        assert_eq!(
            out.printed,
            vec![
                "-2147483648",          // f64 -1e300 saturates at i32::MIN
                "2147483647",           // f64 +1e300 saturates at i32::MAX
                "-9223372036854775808", // f64 -1e300 saturates at i64::MIN
                "0",                    // NaN -> 0
                "2147483647",           // 2147483647.6 truncates out of range -> clamp MAX
                "3",                    // plain truncate
            ]
        );
    }

    #[test]
    fn oob_read_reports_line() {
        let err = run_src("fn main() {\n    let a: [i64] = zeros(2);\n    print(a[7]);\n}\n")
            .expect_err("must trap");
        assert!(
            err.contains("index 7 out of bounds"),
            "wrong message: {err}"
        );
        assert!(err.contains("at line 3"), "wrong message: {err}");
    }

    #[test]
    fn trap_preserves_printed_lines_for_jit_parity() {
        // The JIT streams prints as they happen; the interpreter must hand
        // back everything printed before the trap so drivers emit identical
        // stdout for both backends.
        let src = "fn main() {\n    print(1);\n    print(2);\n    let q = 10 / 0;\n}\n";
        let ast = helix_syntax::parse_str(src).unwrap();
        let tp = check(&ast).unwrap();
        let err = run_with_source(src, &tp).expect_err("must trap");
        assert_eq!(err.printed_so_far, vec!["1", "2"]);
        assert_eq!(
            err.render(src),
            "runtime error: integer division by zero at line 4"
        );
    }

    #[test]
    fn div_by_zero_reports_line() {
        let err = run_src("fn main() {\n    let q = 10 / 0;\n}\n").expect_err("must trap");
        assert_eq!(err, "runtime error: integer division by zero at line 2");
    }

    #[test]
    fn i64_min_over_minus_one_traps() {
        // Note: `-9223372036854775808` is NOT a valid literal (the number
        // itself exceeds i64::MAX; unary minus applies afterwards). Build
        // i64::MIN arithmetically: 0 - 2^63, using wrapping subtraction.
        let src = r#"
            fn main() {
                let half = 4611686018427387904; // 2^62
                let m = 0 - half - half;        // wrapping to i64::MIN
                let q = m / -1;
                print(q);
            }
        "#;
        let err = run_src(src).expect_err("must trap");
        let lines = LineMap::new(src);
        let want = format!(
            "runtime error: integer division overflow (i64::MIN / -1) at line {}",
            lines.line_of(src.find("m / -1").unwrap() as u32)
        );
        assert_eq!(err, want);
    }

    #[test]
    fn negative_zeros_traps() {
        let err = run_src("fn main() {\n    let n = -3;\n    let a: [f64] = zeros(n);\n}\n")
            .expect_err("must trap");
        assert!(
            err.contains("zeros(-3)") && err.contains("at line 3"),
            "wrong message: {err}"
        );
    }
}
