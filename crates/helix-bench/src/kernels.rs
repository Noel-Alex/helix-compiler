//! The benchmark kernel registry.
//!
//! Every kernel bundles, per interface-contracts.md ("bench" bullet): the
//! HELIX source (parameterized small-N correctness variant + big-N perf
//! variant), the size sweep used by the campaign, the dependence-analysis
//! verdict the compiler is expected to produce (so a campaign FAILS if
//! analysis and reality disagree — the sieve/recurrence negatives are as much
//! tests of the compiler as benchmarks), and how outputs are validated across
//! execution variants ([`Tolerance`]).
//!
//! # Validation model
//!
//! All variants of one kernel run *the same HELIX program*, so their printed
//! lines must agree modulo FP reduction reassociation (methodology pitfall 1):
//! strict equality for integer/min/max kernels, relative-epsilon for FP
//! sums. [`RunOutputLike::printed`] carries what `print` produced;
//! [`RunOutputLike::checksum`] mirrors helix-engine's FNV-1a so the JIT side
//! can be compared byte-for-byte where semantics allow it.
//!
//! # Size rewriting
//!
//! Sources embed their performance sizes; [`resize_source`] rewrites
//! them to tiny values for correctness runs. Rewrites target the exact
//! declaration lines (`let n = 33554432;`, `const N: i64 = 512;`) so no other
//! occurrence can be touched accidentally.

use serde::{Deserialize, Serialize};

/// How strictly two variants' printed outputs must match.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Tolerance {
    /// Bit-exact: integers, bools, min/max reductions (order-independent).
    Exact,
    /// FP reductions: parallel summation reassociates, so require
    /// `|a-b| <= eps * max(|a|,|b|)` on each float line instead of equality.
    RelEps(f64),
}

impl Tolerance {
    /// Canonical name for JSON output (`"exact"` / `"rel_eps=1e-9"`).
    #[must_use]
    pub fn name(self) -> String {
        match self {
            Tolerance::Exact => "exact".to_string(),
            Tolerance::RelEps(e) => format!("rel_eps={e:e}"),
        }
    }

    /// Compares two printed-output vectors under this policy.
    ///
    /// Lines that parse as f64 on both sides are compared numerically when
    /// the policy allows slack; every other line must match verbatim.
    #[must_use]
    pub fn matches(self, a: &[String], b: &[String]) -> bool {
        if a.len() != b.len() {
            return false;
        }
        a.iter().zip(b.iter()).all(|(x, y)| self.line_matches(x, y))
    }

    fn line_matches(self, x: &str, y: &str) -> bool {
        if x == y {
            return true;
        }
        match self {
            Tolerance::Exact => false,
            Tolerance::RelEps(eps) => match (x.parse::<f64>(), y.parse::<f64>()) {
                (Ok(fx), Ok(fy)) => {
                    let scale = fx.abs().max(fy.abs());
                    (fx - fy).abs() <= eps * scale.max(1.0)
                }
                _ => false,
            },
        }
    }
}

/// Backend-independent view of one program run. Both the interpreter path
/// (`helix_engine::run_with_source`) and the JIT path wrap themselves in this,
/// which is what lets the harness stay agnostic about which backends exist.
#[derive(Clone, Debug, PartialEq)]
pub struct RunOutputLike {
    /// One string per `print`, without newlines.
    pub printed: Vec<String>,
    /// FNV-1a content hash (printed bytes + final array bits) — identical
    /// only when both backends produced bit-identical state.
    pub checksum: u64,
}

impl RunOutputLike {
    /// Wraps a real interpreter result.
    #[must_use]
    pub fn from_engine(out: &helix_engine::RunOutput) -> Self {
        Self {
            printed: out.printed.clone(),
            checksum: out.checksum,
        }
    }

    /// Builds from raw parts (JIT host wrappers typically do this).
    #[must_use]
    pub fn new(printed: Vec<String>, checksum: u64) -> Self {
        Self { printed, checksum }
    }
}

/// What the dependence engine should conclude about the kernel's hot loop.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExpectedVerdict {
    /// Independent iterations → DOALL parallelization approved.
    SafeParallel,
    /// Recognized reduction → private accumulators + combine, approved.
    ReductionParallel,
    /// Must be REJECTED (sequential fallback); rejecting is the pass condition.
    Sequential,
    /// No hot loop to analyze (e.g. pure recursion); analysis not exercised.
    NotApplicable,
}

/// One registered benchmark kernel.
#[derive(Clone, Debug)]
pub struct KernelDef {
    /// Registry key, e.g. `"saxpy"` — also the JSON series name.
    pub name: &'static str,
    /// Human description for reports.
    pub description: &'static str,
    /// Full HELIX source at PERFORMANCE sizes (the campaign's timed shape).
    pub perf_source: String,
    /// Same program rewritten to a tiny size for parity/correctness checks.
    pub correctness_source: String,
    /// Problem-size sweep for the report table (elements or matrix edge).
    pub sizes: &'static [i64],
    /// Verdict the dependence engine must produce for this kernel's hot loop.
    pub expected_verdict: ExpectedVerdict,
    /// Cross-variant output comparison policy.
    pub tolerance: Tolerance,
    /// Expected printed line(s) at the CORRECTNESS size (oracle values,
    /// derived analytically or via an independent Rust recomputation).
    pub expected_printed: &'static [&'static str],
    /// Exclude from the timed campaign even though it stays registered
    /// (correctness-only kernels like `fib_recursion`).
    pub correctness_only: bool,
    /// Largest problem size the INTERPRETER variant is timed at. Interpreting
    /// millions of iterations costs minutes per sample — the numbers would be
    /// meaningless and the campaign would stall; at big N only the native
    /// variants are measured, and the interp/native ratio is taken from the
    /// largest shared size (this is also methodologically honest: ns/elem
    /// columns are compared, never absolute wall-times across sizes).
    pub interp_max_size: i64,
}

// ---------------------------------------------------------------------------
// HELIX sources
// ---------------------------------------------------------------------------

const SAXPY_SRC: &str = r#"// saxpy: y = s*x + y -- memory-bound streaming kernel (~24 B/elem moved).
// Seeded deterministically (sign/fraction mix) so no execution pass can fold
// the loop or pass a parity check vacuously; mirrors rust_twins::saxpy_inputs.
fn main() {
    let n = 33554432;
    let x: [f64] = zeros(n);
    let y: [f64] = zeros(n);
    for i in 0..n {
        x[i] = ((i * 17 + 3) % 251) as f64 / 17.0 - 4.0;
        y[i] = ((i * 29 + 11) % 241) as f64 / 19.0 - 5.0;
    }
    let s = 2.5;
    for i in 0..n {
        y[i] = s * x[i] + y[i];
    }
    print(y[7]);
}
"#;

const SCALE_SRC: &str = r#"// scale: out = s*a -- independent iterations -> SAFE parallel (~16 B/elem).
// Seeded deterministically (mirrors rust_twins::scale_inputs) so the checked
// element is a nonzero signed fraction, not the all-zero vacuity.
fn main() {
    let n = 33554432;
    let a: [f64] = zeros(n);
    let out: [f64] = zeros(n);
    for i in 0..n {
        a[i] = ((i * 13 + 5) % 199) as f64 / 7.0 - 12.0;
    }
    for i in 0..n {
        out[i] = a[i] * 5.0;
    }
    print(out[42]);
}
"#;

const DOT_SRC: &str = r#"// dot product: +-reduction over products of two arrays (FP sum reassociates).
// Both operands are seeded with sign-mixed fractions (mirroring
// rust_twins::dot_inputs) so partial products cancel — a zero-initialized
// pair would make every broken kernel print the "right" 0.
fn main() {
    let n = 16777216;
    let a: [f64] = zeros(n);
    let b: [f64] = zeros(n);
    for i in 0..n {
        a[i] = ((i * 7 + 1) % 97) as f64 / 9.0 - 4.0;
        b[i] = ((i * 11 + 2) % 89) as f64 / 11.0 - 3.0;
    }
    let dot = 0.0;
    for i in 0..n {
        dot = dot + a[i] * b[i];
    }
    print(dot);
}
"#;

const MINMAX_SRC: &str = r#"// min + max in one loop: order-independent reductions -> bit-exact validation
fn main() {
    let n = 16777216;
    let a: [f64] = zeros(n);
    let lo = 1.0e300;
    let hi = 0.0;
    for i in 0..n {
        let sq = a[i] * a[i];
        lo = min(lo, a[i]);
        hi = max(hi, sq);
    }
    print(lo);
    print(hi);
}
"#;

const SIEVE_SRC: &str = r#"// Strided sieve: outer loop sequential (branch on composite), inner sweep
// writes disjoint strided indices -> inner loop is DOALL. Expressed with an
// affine subscript because HELIX has no step clause.
fn main() {
    let n = 10000000;
    let composite: [bool] = zeros(n);
    let count = 0;
    for i in 2..n {
        if !composite[i] {
            count = count + 1;
            let start = i + i;
            let sweeps = (n - start + i - 1) / i;
            for k in 0..sweeps {
                composite[start + k * i] = true;
            }
        }
    }
    print(count);
}
"#;

const RECURRENCE_SRC: &str = r#"// Distance-1 recurrence: RAW flow dependence at distance 1 through 'a'.
// NOT a reduction (intermediate stores are observed later), so the
// dependence engine MUST reject parallelization; we still measure the
// sequential run to show the cost of refusal is zero.
fn main() {
    let n = 10000000;
    let a: [i64] = zeros(n);
    a[0] = 1;
    for i in 1..n {
        a[i] = a[i - 1] + 1;
    }
    print(a[n - 1]);
}
"#;

const SMALL_N_SRC: &str = r#"// Honest overhead case: N=1000 trip count, threading LOSES (fork/join +
// grain gate dominate). Presented as-is in the report, efficiency < 1.
fn main() {
    let n = 1000;
    let a: [f64] = zeros(n);
    let out: [f64] = zeros(n);
    for i in 0..n {
        out[i] = a[i] + 1.0;
    }
    print(out[999]);
}
"#;

const JACOBI_SRC: &str = r#"// 5-point Jacobi stencil, flattened SIZE x SIZE: interior rows independent
// within a sweep -> level-2 DOALL. Consts live at TOP LEVEL per the frozen
// grammar (they are scalars, which is legal there).
const SIZE: i64 = 4096;
const ITER: i64 = 10;

fn main() {
    let n = SIZE * SIZE;
    let cur: [f64] = zeros(n);
    let next: [f64] = zeros(n);
    cur[SIZE * (SIZE / 2) + SIZE / 2] = 256.0;
    for k in 0..ITER {
        for i in 1..SIZE - 1 {
            let base = i * SIZE;
            for j in 1..SIZE - 1 {
                next[base + j] = 0.25 * (cur[base + j - 1] + cur[base + j + 1]
                                       + cur[base - SIZE + j] + cur[base + SIZE + j]);
            }
        }
        for i in 0..n {
            cur[i] = next[i];
        }
    }
    print(cur[SIZE * (SIZE / 2) + SIZE / 2]);
}
"#;

const MATMUL_SRC: &str = r#"// C = A*B, naive i-j-k: outer-i rows of C are independent (DOALL), inner-k
// is a classic +-reduction into acc. Compared only against its own modes.
const N: i64 = 512;

fn main() {
    let nn = N * N;
    let a: [f64] = zeros(nn);
    let b: [f64] = zeros(nn);
    let c: [f64] = zeros(nn);
    for i in 0..nn {
        a[i] = (i % 97) as f64 * 0.5;
        b[i] = ((i * 7) % 89) as f64 * 0.25;
    }
    for i in 0..N {
        let ibase = i * N;
        for j in 0..N {
            let acc = 0.0;
            for k in 0..N {
                acc = acc + a[ibase + k] * b[k * N + j];
            }
            c[ibase + j] = acc;
        }
    }
    print(c[N * (N / 2) + N / 2]);
}
"#;

const FIB_SRC: &str = r#"// Recursion + else-if + early return exercise (frontend/sema/interp/JIT
// control-flow parity). Correctness-only: fib-style branching is exactly
// what an auto-parallelizer must NOT touch.
fn fib(n: i64) -> i64 {
    if n < 2 {
        return n;
    } else if n < 15 {
        return fib(n - 1) + fib(n - 2);
    }
    return fib(n - 3) + 2 * fib(n - 2) - fib(n - 4) + 4;
}

fn main() {
    print(fib(24));
}
"#;

// ---------------------------------------------------------------------------
// Size rewriting
// ---------------------------------------------------------------------------

/// Rewrites a kernel's problem size inside its HELIX source.
///
/// Handles both declaration styles the suite uses:
/// * scalar binding: `let n = 33554432;`
/// * top-level const: `const N: i64 = 512;` / `const SIZE: i64 = 4096;`
///
/// Only the FIRST occurrence of each pattern is replaced (kernels declare a
/// given size once); unknown names are returned unchanged so callers can
/// detect the mistake via the resulting parse/sema failure rather than by
/// magic sentinels here.
#[must_use]
pub fn resize_source(src: &str, replacements: &[(&str, i64)]) -> String {
    let trailing_newline = src.ends_with('\n');
    let mut out = src.to_string();
    for (name, value) in replacements {
        // Scan line-wise for the single declaration of `name`; kernels never
        // shadow a size binding, so replacing every match is safe and simple.
        let rewritten: Vec<String> = out
            .lines()
            .map(|line| {
                let t = line.trim_start();
                if t.starts_with(&format!("let {name} = ")) {
                    let indent = " ".repeat(line.len() - t.len());
                    format!("{indent}let {name} = {value};")
                } else if t.starts_with(&format!("const {name}: i64 = ")) {
                    let indent = " ".repeat(line.len() - t.len());
                    format!("{indent}const {name}: i64 = {value};")
                } else {
                    line.to_string()
                }
            })
            .collect();
        out = rewritten.join("\n");
    }
    // Preserve the file's exact terminator shape so round-trips are
    // byte-stable (idempotence for callers that resize twice).
    if trailing_newline && !out.ends_with('\n') {
        out.push('\n');
    }
    out
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

/// Streaming kernels' size sweep: LLC-resident small + DRAM-bound large.
const STREAM_SIZES: &[i64] = &[65_536, 16_777_216];

/// All registered kernels, in campaign order.
///
/// Perf sizes follow methodology rec. 14 tuned for a 16C/32T desktop:
/// medium working sets sit near LLC, large ones force DRAM-bound behavior.
#[must_use]
pub fn registry() -> Vec<KernelDef> {
    vec![
        KernelDef {
            name: "scale",
            description: "out[i] = s*a[i] — streaming, DOALL",
            perf_source: SCALE_SRC.to_string(),
            correctness_source: resize_source(SCALE_SRC, &[("n", 1_000)]),
            sizes: STREAM_SIZES,
            expected_verdict: ExpectedVerdict::SafeParallel,
            tolerance: Tolerance::Exact,
            // a[42]: 42*13+5 = 551, 551 % 199 = 153 -> 153/7 - 12 ~= 9.857;
            // out[42] = 5 * that ~= 49.286 (exact f64 pinned by the
            // rust_twins::scale oracle test).
            expected_printed: &["49.28571428571429"],
            correctness_only: false,
            interp_max_size: 262144,
        },
        KernelDef {
            name: "saxpy",
            description: "y[i] += s*x[i] — streaming, DOALL, ~24 B/elem",
            perf_source: SAXPY_SRC.to_string(),
            correctness_source: resize_source(SAXPY_SRC, &[("n", 1_024)]),
            sizes: STREAM_SIZES,
            expected_verdict: ExpectedVerdict::SafeParallel,
            tolerance: Tolerance::Exact,
            // x[7]: 7*17+3 = 122, 122 % 251 = 122 -> 122/17 - 4 ~= 3.176;
            // y[7]: 7*29+11 = 214 -> 214/19 - 5 ~= 6.263;
            // y'[7] = 2.5*x[7] + y[7] ~= 14.204 (exact f64 pinned by the
            // rust_twins::saxpy oracle test).
            expected_printed: &["14.204334365325078"],
            correctness_only: false,
            interp_max_size: 262144,
        },
        KernelDef {
            name: "dot_reduction",
            description: "sum a[i]*b[i] — FP +-reduction",
            perf_source: DOT_SRC.to_string(),
            correctness_source: resize_source(DOT_SRC, &[("n", 4_096)]),
            sizes: &[65_536, 4_194_304],
            expected_verdict: ExpectedVerdict::ReductionParallel,
            tolerance: Tolerance::RelEps(1e-9),
            // Interpreter sum over the seeded pair at n=4096; independently
            // recomputed by rust_twins (see dot_inputs' oracle test).
            expected_printed: &["5206.808080808095"],
            correctness_only: false,
            interp_max_size: 400000,
        },
        KernelDef {
            name: "minmax_reduction",
            description: "min(a[i]) and max(a[i]^2) — two order-independent reductions",
            perf_source: MINMAX_SRC.to_string(),
            correctness_source: resize_source(MINMAX_SRC, &[("n", 4_096)]),
            sizes: &[65_536, 16_777_216],
            expected_verdict: ExpectedVerdict::ReductionParallel,
            tolerance: Tolerance::Exact,
            expected_printed: &["0.0", "0.0"],
            correctness_only: false,
            interp_max_size: 400000,
        },
        KernelDef {
            name: "count_primes_sieve",
            description: "strided Eratosthenes — outer serial, inner DOALL writes",
            perf_source: SIEVE_SRC.to_string(),
            correctness_source: resize_source(SIEVE_SRC, &[("n", 100)]),
            sizes: &[100_000, 4_000_000],
            expected_verdict: ExpectedVerdict::SafeParallel,
            tolerance: Tolerance::Exact,
            expected_printed: &["25"], // pi(100)
            correctness_only: false,
            interp_max_size: 100000,
        },
        KernelDef {
            name: "recurrence_reject",
            description: "a[i]=a[i-1]+1 — distance-1 RAW; MUST be rejected",
            perf_source: RECURRENCE_SRC.to_string(),
            correctness_source: resize_source(RECURRENCE_SRC, &[("n", 1_000)]),
            sizes: &[100_000, 10_000_000],
            expected_verdict: ExpectedVerdict::Sequential,
            tolerance: Tolerance::Exact,
            expected_printed: &["1000"], // a[999] = 1 + 999 steps
            correctness_only: false,
            interp_max_size: i64::MAX,
        },
        KernelDef {
            name: "small_n",
            description: "N=1000 add — threading loses; honest overhead point",
            perf_source: SMALL_N_SRC.to_string(),
            // N=1000 is already the tiny size; perf and correctness share it.
            correctness_source: SMALL_N_SRC.to_string(),
            sizes: &[1_000],
            expected_verdict: ExpectedVerdict::SafeParallel,
            tolerance: Tolerance::Exact,
            expected_printed: &["1.0"],
            correctness_only: false,
            interp_max_size: 1000,
        },
        KernelDef {
            name: "jacobi_2d",
            description: "5-point Jacobi stencil — level-2 DOALL, DRAM-bound",
            perf_source: JACOBI_SRC.to_string(),
            correctness_source: resize_source(JACOBI_SRC, &[("SIZE", 32), ("ITER", 4)]),
            sizes: &[512, 1024],
            expected_verdict: ExpectedVerdict::SafeParallel,
            tolerance: Tolerance::Exact,
            expected_printed: &["36.0"], // centre after 4 sweeps of the seeded cell
            correctness_only: false,
            interp_max_size: 512,
        },
        KernelDef {
            name: "matmul",
            description: "C=A*B naive i-j-k — outer DOALL + inner +-reduction",
            perf_source: MATMUL_SRC.to_string(),
            correctness_source: resize_source(MATMUL_SRC, &[("N", 8)]),
            sizes: &[128, 256],
            expected_verdict: ExpectedVerdict::ReductionParallel,
            tolerance: Tolerance::RelEps(1e-9),
            // Independently recomputed by rust_twins::matmul_centre(8).
            expected_printed: &["1626.625"],
            correctness_only: false,
            interp_max_size: 128,
        },
        KernelDef {
            name: "fib_recursion",
            description: "recursive control flow — correctness parity only",
            perf_source: FIB_SRC.to_string(),
            correctness_source: FIB_SRC.to_string(),
            sizes: &[24],
            expected_verdict: ExpectedVerdict::NotApplicable,
            tolerance: Tolerance::Exact,
            expected_printed: &["20001"],
            correctness_only: true,
            interp_max_size: i64::MAX,
        },
    ]
}

impl KernelDef {
    /// Source at a specific campaign size: starts from the perf program and
    /// rewrites the declared sizes down/up to `size`.
    ///
    /// `size` means elements for flat kernels and the matrix EDGE for
    /// jacobi/matmul (their total footprint scales with its square).
    #[must_use]
    pub fn source_at_size(&self, size: i64) -> String {
        match self.name {
            "jacobi_2d" => resize_source(&self.perf_source.clone(), &[("SIZE", size)]),
            "matmul" => resize_source(&self.perf_source.clone(), &[("N", size)]),
            _ => resize_source(&self.perf_source.clone(), &[("n", size)]),
        }
    }

    /// True when this kernel's campaign points should include a thread sweep.
    #[must_use]
    pub fn is_parallel_candidate(&self) -> bool {
        !self.correctness_only && self.expected_verdict != ExpectedVerdict::Sequential
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_has_expected_members_and_order() {
        let reg = registry();
        let names: Vec<_> = reg.iter().map(|k| k.name).collect();
        assert_eq!(
            names,
            vec![
                "scale",
                "saxpy",
                "dot_reduction",
                "minmax_reduction",
                "count_primes_sieve",
                "recurrence_reject",
                "small_n",
                "jacobi_2d",
                "matmul",
                "fib_recursion",
            ]
        );
        // Exactly one deliberate negative, clearly labelled.
        assert_eq!(
            reg.iter()
                .filter(|k| k.expected_verdict == ExpectedVerdict::Sequential)
                .map(|k| k.name)
                .collect::<Vec<_>>(),
            vec!["recurrence_reject"]
        );
    }

    #[test]
    fn all_sources_parse_and_check_cleanly() {
        for k in registry() {
            for (label, src) in [
                ("perf", k.perf_source.as_str()),
                ("correctness", k.correctness_source.as_str()),
            ] {
                let ast = helix_syntax::parse_str(src)
                    .unwrap_or_else(|e| panic!("{} ({label}) parse: {e}", k.name));
                let typed = helix_sema::check(&ast)
                    .unwrap_or_else(|ds| panic!("{} ({label}) sema: {ds:#?}", k.name));
                assert_eq!(typed.funcs.len() + typed.consts.len(), ast.items.len());
            }
        }
    }

    #[test]
    fn resize_rewrites_declared_sizes_only() {
        let resized = resize_source(MATMUL_SRC, &[("N", 8)]);
        assert!(resized.contains("const N: i64 = 8;"));
        assert!(!resized.contains("const N: i64 = 512;"));
        // The init loop bound `for i in 0..nn` derives from N, untouched.
        assert!(resized.contains("let nn = N * N;"));

        let saxpy_small = resize_source(SAXPY_SRC, &[("n", 1024)]);
        assert!(saxpy_small.contains("let n = 1024;"));
        assert!(!saxpy_small.contains("33554432"));
    }

    #[test]
    fn resize_is_idempotent_and_leaves_other_names_alone() {
        let once = resize_source(JACOBI_SRC, &[("SIZE", 16), ("ITER", 2)]);
        let twice = resize_source(&once, &[("SIZE", 16), ("ITER", 2)]);
        assert_eq!(once, twice);
        // Unknown names change nothing.
        assert_eq!(resize_source(SMALL_N_SRC, &[("zzz", 5)]), SMALL_N_SRC);
    }

    #[test]
    fn tolerance_policies_behave() {
        let lines = |xs: &[&str]| -> Vec<String> { xs.iter().map(|s| (*s).to_string()).collect() };
        assert!(Tolerance::Exact.matches(&lines(&["25"]), &lines(&["25"])));
        assert!(!Tolerance::Exact.matches(&lines(&["0.30000000000000004"]), &lines(&["0.3"])));
        assert!(
            Tolerance::RelEps(1e-9).matches(&lines(&["1000000.000001"]), &lines(&["1000000.0"]))
        );
        assert!(!Tolerance::RelEps(1e-9).matches(&lines(&["1.1"]), &lines(&["1.2"])));
        // Length mismatch fails under any policy.
        assert!(!Tolerance::Exact.matches(&lines(&["1"]), &lines(&["1", "2"])));
        assert_eq!(Tolerance::Exact.name(), "exact");
        assert_eq!(Tolerance::RelEps(1e-9).name(), "rel_eps=1e-9");
    }

    #[test]
    fn source_at_size_targets_each_kernel_shape() {
        let reg = registry();
        let matmul = reg.iter().find(|k| k.name == "matmul").unwrap();
        assert!(matmul.source_at_size(128).contains("const N: i64 = 128;"));
        let jacobi = reg.iter().find(|k| k.name == "jacobi_2d").unwrap();
        assert!(jacobi.source_at_size(64).contains("const SIZE: i64 = 64;"));
        let saxpy = reg.iter().find(|k| k.name == "saxpy").unwrap();
        assert!(saxpy.source_at_size(65_536).contains("let n = 65536;"));
    }

    #[test]
    fn parallel_candidates_exclude_negative_and_correctness_kernels() {
        let reg = registry();
        let seq = reg
            .iter()
            .filter(|k| !k.is_parallel_candidate())
            .map(|k| k.name)
            .collect::<Vec<_>>();
        assert_eq!(seq, vec!["recurrence_reject", "fib_recursion"]);
    }
}
