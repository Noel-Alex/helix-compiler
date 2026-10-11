//! Deterministic soundness fuzzer for the approval gate.
//!
//! `golden.rs` pins the verdicts of the repository's example kernels and
//! `pipeline.rs` pins adversarial shapes; this file asks the one property
//! that decides whether DOALL parallelization is *correct* rather than
//! merely interesting, and answers it by brute force:
//!
//! > if [`helix_analysis::analyze`] approves a loop, no two of its
//! > iterations may write (or write past a read of) the same element.
//!
//! The corpus is generated here, never in the compiler: one or two array
//! accesses per loop, each spelled `A*i + B` with small literals and trip
//! counts, covering write/write, write/read, read/read, same-array and
//! different-array pairs plus the single-access self-site obligation. Every
//! generated index is kept inside the array for all iterations, so no
//! generated program can trap — a trapping program proves nothing about
//! parallel execution.
//!
//! The oracle is a dumb enumeration over the same model: for every ordered
//! pair of iterations `i != j` and every access pair on a shared array, do
//! the two touched elements coincide with at least one write among them?
//! Read/read pairs never conflict and same-iteration pairs are
//! loop-independent by definition, so both are excluded.
//!
//! HELIX is allowed to be conservative, so a `Sequential` verdict never
//! fails this test. The summary line reports how often the rejection was
//! justified by a real conflict, and the mix assertion guards the other
//! failure mode: a corpus that approves nothing would pass vacuously.

#![forbid(unsafe_code)]

use helix_analysis::{Verdict, analyze, find_loops};
use helix_syntax::parse_str;

/// Parse → check → build → `to_ssa` → `run_optimization_pipeline` →
/// `find_loops` → `analyze`, returning the reports of one named function.
///
/// The ordering is the production one — the Observatory and the wasm
/// artifact builder analyze post-optimization IR — so the verdicts under
/// test are the shipped ones.
fn reports_of(src: &str, func: &str) -> Vec<helix_analysis::LoopReport> {
    let prog = parse_str(src).unwrap_or_else(|e| panic!("parse: {e}"));
    let typed = helix_sema::check(&prog).unwrap_or_else(|ds| panic!("check: {ds:?}"));
    let mut funcs = helix_ir::build(&typed);
    for f in &mut funcs {
        helix_ir::to_ssa(f);
        helix_ir::run_optimization_pipeline(f);
    }
    let idx = funcs
        .iter()
        .position(|f| f.name == func)
        .unwrap_or_else(|| panic!("no function '{func}' in:\n{src}"));
    let li = find_loops(&funcs[idx]);
    analyze(&funcs[idx], &li)
}

// ---------------------------------------------------------------------------
// The oracle's model
// ---------------------------------------------------------------------------

/// One generated access: `<arr>[stride * i + offset]`, written or read.
#[derive(Clone, Copy)]
struct Access {
    /// 0 = `a`, 1 = `b` — the two arrays every generated program declares.
    arr: u8,
    write: bool,
    stride: i64,
    offset: i64,
}

impl Access {
    /// The element touched at iteration `i`; the model's whole semantics.
    fn touch(&self, i: i64) -> i64 {
        self.stride * i + self.offset
    }

    fn name(&self) -> &'static str {
        if self.arr == 0 { "a" } else { "b" }
    }
}

/// Brute-force semantic oracle: do two DISTINCT iterations of this access
/// set write, or write past a read of, the same element?
///
/// Pairs from one iteration are excluded because DOALL preserves the body's
/// internal order; read/read pairs never conflict at all. `None` means the
/// loop is conflict-free, so approving it is sound.
fn oracle_conflict(n: i64, accesses: &[Access]) -> Option<(usize, usize, i64, i64)> {
    for (p, src) in accesses.iter().enumerate() {
        for (q, dst) in accesses.iter().enumerate() {
            if src.arr != dst.arr || !(src.write || dst.write) {
                continue;
            }
            for i in 0..n {
                for j in 0..n {
                    if i != j && src.touch(i) == dst.touch(j) {
                        return Some((p, q, i, j));
                    }
                }
            }
        }
    }
    None
}

/// Renders `stride*i + offset` the way `plan::render_affine` does, so the
/// fuzzer can demand that the analysis saw exactly the access the source
/// wrote. A divergence would mean the oracle is checking a program the
/// compiler never approved.
fn render_access(acc: &Access) -> String {
    let sub = match (acc.stride, acc.offset) {
        (0, b) => b.to_string(),
        (1, 0) => "i".to_string(),
        (1, c) if c > 0 => format!("i + {c}"),
        (1, c) => format!("i - {}", -c),
        (-1, 0) => "-i".to_string(),
        (-1, c) if c > 0 => format!("{c} - i"),
        (-1, c) => format!("-i - {}", -c),
        (k, 0) => format!("{k}*i"),
        (k, c) if c > 0 => format!("{k}*i + {c}"),
        (k, c) => format!("{k}*i - {}", -c),
    };
    format!(
        "{} {}[{sub}]",
        if acc.write { "WRITE" } else { "READ" },
        acc.name()
    )
}

/// Spells the same subscript in HELIX source; the classifier re-derives the
/// affine form from the IR the builder emits for it.
fn subscript(acc: &Access) -> String {
    let (k, c) = (acc.stride, acc.offset);
    match (k, c) {
        (0, c) => c.to_string(),
        (k, 0) => format!("{k} * i"),
        (k, c) if c > 0 => format!("{k} * i + {c}"),
        (k, c) => format!("{k} * i - {}", -c),
    }
}

// ---------------------------------------------------------------------------
// Generator (deterministic; no external rng crate)
// ---------------------------------------------------------------------------

/// A plain 64-bit LCG: the corpus must be reproducible run to run.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    /// Inclusive draw from `[lo, hi]`.
    fn between(&mut self, lo: i64, hi: i64) -> i64 {
        lo + self.below((hi - lo + 1) as u64) as i64
    }
}

/// Draws one access whose touched range clears zero for every iteration.
///
/// One draw in three is a clearly injective translation (a positive stride
/// over a non-negative offset), so the corpus always contains approvals
/// instead of degenerating into pure rejection.
fn draw_access(rng: &mut Lcg, n: i64) -> Option<Access> {
    for _ in 0..8 {
        let (stride, offset) = if rng.below(3) == 0 {
            match rng.below(3) {
                0 => (1, 0),
                1 => (2, 1),
                _ => (rng.between(1, 4), rng.between(0, 6)),
            }
        } else {
            (rng.between(-4, 4), rng.between(-6, 6))
        };
        // A trapping program proves nothing: the low end of the touched
        // range (at i = 0 for a rising stride, at i = n-1 otherwise) must
        // stay inside the array. The high end just grows the slack.
        if (stride * (n - 1) + offset).min(offset) >= 0 {
            return Some(Access {
                arr: rng.below(2) as u8,
                write: rng.below(2) == 0,
                stride,
                offset,
            });
        }
    }
    None
}

/// One generated program plus the model the oracle checks it against.
struct Case {
    src: String,
    n: i64,
    accesses: Vec<Access>,
}

fn draw_case(rng: &mut Lcg) -> Option<Case> {
    let n = rng.between(1, 10);
    // Half the corpus carries a SINGLE access: the self-site obligation,
    // where the only possible cross-iteration race is a write colliding
    // with its own other iterations.
    let count = 1 + usize::try_from(rng.below(2)).expect("0 or 1");
    let mut accesses = Vec::with_capacity(count);
    for _ in 0..count {
        accesses.push(draw_access(rng, n)?);
    }
    // Each array is at least as long as the trip count, and long enough to
    // hold every element its own accesses can touch.
    let mut len = [n.max(2), n.max(2)];
    for acc in &accesses {
        let hi = (acc.stride * (n - 1) + acc.offset).max(acc.offset) + 1;
        len[usize::from(acc.arr)] = len[usize::from(acc.arr)].max(hi);
    }
    let mut body = String::new();
    let mut reads = 0;
    for acc in &accesses {
        if acc.write {
            body.push_str(&format!(
                "        {}[{}] = i;\n",
                acc.name(),
                subscript(acc)
            ));
        } else {
            let t = if reads == 0 { "t" } else { "u" };
            reads += 1;
            body.push_str(&format!(
                "        let {t} = {}[{}];\n",
                acc.name(),
                subscript(acc)
            ));
        }
    }
    // The trailing prints keep both arrays live, so nothing downstream of
    // the loop can consider the body's traffic dead.
    let src = format!(
        "fn main() {{\n    let n = {n};\n    let a: [i64] = zeros({});\n    \
         let b: [i64] = zeros({});\n    for i in 0..n {{\n{body}    }}\n    \
         print(a[0]);\n    print(b[1]);\n}}\n",
        len[0], len[1]
    );
    Some(Case { src, n, accesses })
}

// ---------------------------------------------------------------------------
// The adversarial table: shapes a reviewer must be able to point at, each
// with the verdict the compiler owes. HELIX may reject more of these; it may
// never approve a conflicting one.
// ---------------------------------------------------------------------------

/// `a[0] = i`: one element written every iteration. The sole write races
/// with itself, so the constant subscript must reject even though it is a
/// trivially affine (degenerate) form — and the oracle agrees the race is
/// real.
#[test]
fn self_site_constant_write_is_sequential() {
    let reps = reports_of(
        r#"
        fn main() {
            let n = 8;
            let a: [i64] = zeros(8);
            for i in 0..n {
                a[0] = i;
            }
        }
        "#,
        "main",
    );
    assert_eq!(reps.len(), 1, "one loop");
    let r = &reps[0];
    let Verdict::Sequential(reason) = &r.verdict else {
        panic!("a[0] self-site must reject: {}", r.summary_line());
    };
    assert!(reason.contains("overlap"), "reason: {reason}");
    let model = [Access {
        arr: 0,
        write: true,
        stride: 0,
        offset: 0,
    }];
    assert!(
        oracle_conflict(8, &model).is_some(),
        "oracle must see the self-site race"
    );
}

/// `a[2*i + 1] = i` over literal bounds: a stride of 2 with an odd offset is
/// an injective translation, so the sole write cannot overlap itself and
/// must be approved.
#[test]
fn injective_literal_stride_write_is_safe() {
    let reps = reports_of(
        r#"
        fn main() {
            let a: [i64] = zeros(64);
            for i in 0..10 {
                a[2 * i + 1] = i;
            }
            print(a[1]);
        }
        "#,
        "main",
    );
    let r = &reps[0];
    assert!(
        matches!(r.verdict, Verdict::SafeParallel),
        "{}",
        r.summary_line()
    );
    let model = [Access {
        arr: 0,
        write: true,
        stride: 2,
        offset: 1,
    }];
    assert!(
        oracle_conflict(10, &model).is_none(),
        "2*i+1 is injective: no cross-iteration conflict exists"
    );
}

/// `a[i % 2] = i`: a non-affine subscript cannot be reasoned about, so the
/// write must be treated as potentially self-overlapping and rejected.
#[test]
fn non_affine_subscript_write_is_sequential() {
    let reps = reports_of(
        r#"
        fn main() {
            let n = 8;
            let a: [i64] = zeros(2);
            for i in 0..n {
                a[i % 2] = i;
            }
        }
        "#,
        "main",
    );
    let r = &reps[0];
    assert!(
        matches!(r.verdict, Verdict::Sequential(_)),
        "non-affine subscript must reject: {}",
        r.summary_line()
    );
    assert!(
        r.accesses.iter().any(|a| a == "WRITE a[?]"),
        "subscript must render unanalyzable: {:?}",
        r.accesses
    );
    assert!(
        r.notes.iter().any(|n| n.contains("overlap")),
        "notes: {:?}",
        r.notes
    );
}

/// The unknown-invariant-offset row of the adversarial table: a sole write
/// `a[i + k]` with parameter `k`.
///
/// This row matches the repo's pinned contract and is SOUND: translating every
/// touch by the same loop-invariant `k` is injective, so no two iterations can
/// ever write the same element — there is no WAW obligation to prove.
/// `plan.rs`'s sole-write injectivity rule and the root AGENTS.md contract
/// ("an invariant offset like `a[i+k]` qualifies") pin SafeParallel for this
/// shape, and `jit_pipeline.rs::invariant_offset_sole_write_still_parallelizes`
/// pins it end to end. The soundness danger the review worried about is the
/// MIXED pair (write `a[i+k]` alongside a read of `a[i]`), which is pinned
/// separately by `unknown_invariant_offset_read_pair_is_sequential` below: an
/// exact no-wrap affine form is required there, and an unknown offset never
/// provides one.
#[test]
fn unknown_invariant_offset_sole_write_is_parallel() {
    let reps = reports_of(
        r#"
        fn f(a: [i64], n: i64, k: i64) {
            for i in 0..n {
                a[i + k] = i;
            }
        }
        fn main() {
            let buf: [i64] = zeros(32);
            f(buf, 10, 3);
            print(buf[0]);
        }
        "#,
        "f",
    );
    let r = &reps[0];
    assert!(
        matches!(r.verdict, Verdict::SafeParallel),
        "injective invariant translation is provably parallel: {}",
        r.summary_line()
    );
    // Oracle check for this exact shape: a[i+k] touches a distinct element per
    // iteration for every k, so no cross-iteration pair can conflict.
    for n in 1..=12i64 {
        for k in -8i64..=8 {
            let mut seen = std::collections::HashSet::new();
            for i in 0..n {
                assert!(
                    seen.insert(i + k),
                    "translation by k={k} collided at i={i} with n={n}"
                );
            }
        }
    }
}

/// The read/write spelling of the same unknown offset: `a[i] = a[i + k]`.
/// Translation by an unknown `k` blocks the read's dependence distance, so
/// the mixed pair must be rejected as unanalyzable rather than guessed
/// independent.
#[test]
fn unknown_invariant_offset_read_pair_is_sequential() {
    let reps = reports_of(
        r#"
        fn f(a: [i64], n: i64, k: i64) {
            for i in 0..n {
                a[i] = a[i + k];
            }
        }
        fn main() {
            let buf: [i64] = zeros(32);
            f(buf, 10, 3);
            print(buf[0]);
        }
        "#,
        "f",
    );
    let r = &reps[0];
    assert!(
        matches!(r.verdict, Verdict::Sequential(_)),
        "read at an unknown offset must reject: {}",
        r.summary_line()
    );
    assert!(
        r.notes.iter().any(|n| n.contains("unanalyzable")),
        "notes: {:?}",
        r.notes
    );
    assert_eq!(r.war_deps.len(), 1, "{:?}", r.war_deps);
    assert_eq!(r.war_deps[0].distance, None, "distance must stay unknown");
}

/// `x = a[i] - x` with a seeded `x`: a genuine recurrence, and the
/// subtraction-on-the-right shape has no additive term to reassociate. It
/// must never be approved as a reduction — the loop-carried scalar veto has
/// to catch it instead.
#[test]
fn subtraction_on_the_right_is_not_a_reduction() {
    let reps = reports_of(
        r#"
        fn main() {
            let n = 32;
            let a: [i64] = zeros(n);
            let x = 5;
            for i in 0..n {
                x = a[i] - x;
            }
            print(x);
        }
        "#,
        "main",
    );
    let r = &reps[0];
    assert!(
        !matches!(r.verdict, Verdict::ReductionParallel(_)),
        "right-hand subtraction is not a sum of terms: {}",
        r.summary_line()
    );
    assert!(
        matches!(r.verdict, Verdict::Sequential(_)),
        "carried scalar must veto: {}",
        r.summary_line()
    );
    assert!(
        r.notes.iter().any(|n| n.contains("loop-carried scalar")),
        "notes: {:?}",
        r.notes
    );
}

/// `x = x - a[i]`: the same subtraction with the accumulator on the left
/// folds into the additive family (a sum of negated terms) and is approved
/// as a reduction.
#[test]
fn subtraction_on_the_left_folds_into_an_add_reduction() {
    let reps = reports_of(
        r#"
        fn main() {
            let n = 32;
            let a: [i64] = zeros(n);
            let x = 7;
            for i in 0..n {
                x = x - a[i];
            }
            print(x);
        }
        "#,
        "main",
    );
    let r = &reps[0];
    let Verdict::ReductionParallel(red) = &r.verdict else {
        panic!("left-hand subtraction must reduce: {}", r.summary_line());
    };
    assert_eq!(red.op, helix_analysis::ReductionOp::Add);
    assert_eq!(red.var, "x");
}

// ---------------------------------------------------------------------------
// The fuzz run
// ---------------------------------------------------------------------------

/// Programs per corpus run. Small enough to stay instant, large enough that
/// the stride/offset/trip-count combinations are well covered.
const CORPUS: usize = 600;

#[test]
fn approved_loops_never_conflict_under_the_oracle() {
    let mut rng = Lcg(20_261_010);
    let mut approved = 0usize;
    let mut rejected = 0usize;
    let mut conservative = 0usize;
    let mut unmodellable = 0usize;
    let mut checked = 0usize;
    let mut guard = 0usize;
    while checked < CORPUS {
        guard += 1;
        assert!(
            guard < CORPUS * 100,
            "generator starved after {checked}/{CORPUS} programs"
        );
        let Some(case) = draw_case(&mut rng) else {
            continue;
        };
        checked += 1;
        let reps = reports_of(&case.src, "main");
        if reps.len() != 1 {
            unmodellable += 1;
            continue;
        }
        let rep = &reps[0];
        let rendered: Vec<String> = case.accesses.iter().map(render_access).collect();
        if rep.accesses != rendered {
            // The pipeline rewrote the loop's accesses: the oracle's model no
            // longer describes what the analysis saw, so checking it would
            // prove nothing either way.
            unmodellable += 1;
            continue;
        }
        match &rep.verdict {
            Verdict::SafeParallel => {
                approved += 1;
                if let Some((p, q, i, j)) = oracle_conflict(case.n, &case.accesses) {
                    panic!(
                        "UNSOUND APPROVAL: iterations {i} and {j} collide on \
                         accesses #{p} vs #{q}\n{}\n{}",
                        case.src,
                        rep.summary_line(),
                    );
                }
            }
            // The corpus is array-only, so no accumulator can exist; seeing
            // one would mean the analysis approved a shape it misread.
            Verdict::ReductionParallel(red) => {
                panic!(
                    "corpus bug: array-only program reduced ({red:?}):\n{}",
                    case.src
                );
            }
            Verdict::Sequential(_) => {
                rejected += 1;
                if oracle_conflict(case.n, &case.accesses).is_none() {
                    conservative += 1;
                }
            }
        }
    }
    println!(
        "soundness fuzz: {approved} approved / {rejected} rejected of {CORPUS} \
         programs ({conservative} conservative, {unmodellable} unmodellable)"
    );
    assert_eq!(
        unmodellable, 0,
        "the analysis's access list diverged from the generated source"
    );
    assert!(
        approved * 100 >= CORPUS * 15,
        "corpus is vacuous: only {approved}/{CORPUS} programs approved"
    );
}
