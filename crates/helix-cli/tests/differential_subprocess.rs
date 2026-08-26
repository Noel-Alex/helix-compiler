//! Subprocess differential harness: run canned HELIX programs through BOTH
//! backends as real `helix run` processes and compare observable behavior.
//!
//! In-process comparisons (selftest, engine tests) share one process's
//! globals; the reviewer's red-team item #10 asks for the harder guarantee —
//! the CLI driver itself must behave identically whichever backend you pick:
//!
//! * valid programs: byte-identical stdout, exit 0, empty stderr;
//! * trapping programs: identical stdout prefix (lines printed before the
//!   trap), exit code 1 on both, stderr naming a runtime error on both.
//!
//! Each case spawns two subprocesses (`--backend interp` / `--backend jit`)
//! against a temp `.hx` file. Fast (<2 s total): every program is tiny.

use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

/// Path to the freshly built `helix` binary (cargo injects the env var for
/// integration tests of this crate).
const HELIX: &str = env!("CARGO_BIN_EXE_helix");

/// Repo root (crate dir is `crates/helix-cli`), so relative paths in error
/// messages and future example-based cases resolve consistently.
fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repo root exists")
}

/// One canned case.
struct Case {
    /// Short name used in assertion messages and the temp file stem.
    name: &'static str,
    /// Program source.
    src: &'static str,
    /// Expected stdout for VALID programs (exact match, both backends).
    expected_stdout: &'static str,
    /// Trapping program? Traps assert prefix/exit-1/stderr instead of exact
    /// stdout + exit 0.
    traps: bool,
}

/// The matrix: valid programs plus one trap per spec runtime-error class.
const CASES: &[Case] = &[
    Case {
        name: "arith_and_print",
        src: r"
fn main() {
    print(1 + 2);
    print(10 - 3);
    print(4 * 5);
    print(7 / 2);   // truncating
    print(-7 / 2);  // sign follows dividend
    print(2.5 * 4.0);
}
",
        expected_stdout: "3\n7\n20\n3\n-3\n10.0\n", // floats print with a decimal
        traps: false,
    },
    Case {
        name: "loop_and_array",
        src: r"
fn main() {
    let n = 8;
    let a: [i64] = zeros(n);
    for i in 0..n {
        a[i] = i * i;
    }
    let sum = 0;
    for i in 0..n {
        sum = sum + a[i];
    }
    print(sum);
    print(a[5]);
}
",
        expected_stdout: "140\n25\n", // 0+1+4+9+16+25+36+49 = 140
        traps: false,
    },
    Case {
        name: "if_else_chain",
        src: r"
fn main() {
    let x = 17;
    if x < 0 { print(-1); } else if x == 0 { print(0); } else { print(1); }
    if x > 100 { print(100); } else { print(x); }
}
",
        expected_stdout: "1\n17\n",
        traps: false,
    },
    Case {
        name: "returning_else_if_chain",
        // Regression (always_returns P0): a value-returning function whose
        // every path returns inside a COMPLETE 3-arm else-if chain used to be
        // mis-verified as falling off the end — sema accepted, then the JIT
        // aborted at IR-build time ("sema guaranteed all paths return").
        // Exercises both the sema verdict and the builder's dead-merge path.
        src: r"
fn classify(x: i64) -> i64 {
    if x < 0 { return -1; } else if x == 0 { return 0; } else { return 1; }
}
fn main() {
    print(classify(-5));
    print(classify(0));
    print(classify(17));
}
",
        expected_stdout: "-1\n0\n1\n",
        traps: false,
    },
    Case {
        name: "bounds_read_trap",
        // Prints one line, then reads past the end. Both backends must show
        // that line before dying with status 1 and a `runtime error`.
        src: r"
fn main() {
    let a: [i64] = zeros(4);
    print(a[2]);
    print(a[4]);
    print(999);
}
",
        expected_stdout: "",
        traps: true,
    },
    Case {
        name: "bounds_negative_index_trap",
        src: r"
fn main() {
    let a: [i64] = zeros(4);
    let i = -1;
    print(42);
    print(a[i]);
    print(999);
}
",
        expected_stdout: "",
        traps: true,
    },
    Case {
        name: "div_zero_trap",
        src: r"
fn main() {
    print(1);
    let z = 0;
    print(10 / z);
    print(3);
}
",
        expected_stdout: "",
        traps: true,
    },
    Case {
        name: "rem_zero_trap",
        src: r"
fn main() {
    print(6);
    print(10 % 0);
    print(3);
}
",
        expected_stdout: "",
        traps: true,
    },
];

/// Writes `src` to a temp `.hx` file and returns its path.
fn write_temp_program(name: &str, src: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "helix-diff-{name}-{}-{}.hx",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos()
    ));
    let mut f = std::fs::File::create(&path).expect("create temp .hx");
    f.write_all(src.as_bytes()).expect("write temp .hx");
    path
}

/// Runs `helix run <file> --backend <backend>` as a subprocess from the repo
/// root, capturing stdout/stderr/exit code.
fn run_backend(file: &std::path::Path, backend: &str) -> (i32, String, String) {
    let out = Command::new(HELIX)
        .args([
            "run",
            file.to_str().expect("utf-8 temp path"),
            "--backend",
            backend,
        ])
        .current_dir(repo_root())
        .output()
        .expect("spawn helix");
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

#[test]
fn backends_agree_as_subprocesses() {
    for case in CASES {
        let file = write_temp_program(case.name, case.src);
        let (interp_code, interp_out, interp_err) = run_backend(&file, "interp");
        let (jit_code, jit_out, jit_err) = run_backend(&file, "jit");

        let ctx = format!("case '{}'", case.name);

        if !case.traps {
            assert_eq!(
                interp_code, 0,
                "{ctx}: interp exit code; stderr: {interp_err}"
            );
            assert_eq!(jit_code, 0, "{ctx}: jit exit code; stderr: {jit_err}");
            assert_eq!(
                interp_out, jit_out,
                "{ctx}: stdout must be identical across backends"
            );
            assert_eq!(interp_out, case.expected_stdout, "{ctx}: stdout content");
            assert_eq!(interp_err, "", "{ctx}: interp stderr should be empty");
            assert_eq!(jit_err, "", "{ctx}: jit stderr should be empty");
        } else {
            // Identical stdout prefix (lines printed before the trap) and no
            // extra stdout after it — both backends stop at the same line.
            assert_eq!(interp_out, jit_out, "{ctx}: trapped stdout must match");
            assert!(
                !interp_out.contains("999"),
                "{ctx}: code after trap ran (interp)"
            );
            assert!(!jit_out.contains("999"), "{ctx}: code after trap ran (jit)");
            assert!(
                !interp_out.is_empty(),
                "{ctx}: expected at least one printed line before the trap"
            );
            assert_eq!(
                interp_code, 1,
                "{ctx}: interp exit code on trap; stderr: {interp_err}"
            );
            assert_eq!(
                jit_code, 1,
                "{ctx}: jit exit code on trap; stderr: {jit_err}"
            );
            assert!(
                interp_err.contains("runtime error"),
                "{ctx}: interp stderr must name the runtime error, got: {interp_err}"
            );
            assert!(
                jit_err.contains("runtime error"),
                "{ctx}: jit stderr must name the runtime error, got: {jit_err}"
            );
        }

        drop(std::fs::remove_file(&file));
    }
}
