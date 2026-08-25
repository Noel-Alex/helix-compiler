# Devlog — 2026-08-25: The adversarial review wave

Four independent reviewers swept the finished compiler for soundness holes. 16 findings;
9 critical/major, all fixed with regression tests. The best ones for the report:

## 1. Bare blocks were compiled away (critical)
`{ let j = i * 2; a[k] = f(j); }` — sema checked the statements *for diagnostics* and
then threw them away. The interpreter (which re-walks source) executed them; the JIT
(which reads the typed tree) never saw them → **silently wrong values**, lost prints,
even lost `return`s. Lesson: poison sentinels (`TypedExprKind::Error`) are where
information goes to die; if a grammar production is legal, it must survive to codegen.

## 2. Evaluation-order divergence between backends (critical)
The builder lowered `a[i] = v` value-first; the interpreter evaluates index-first.
With printing calls as operands (`a[tag(2)] = tag(3)`) the two backends printed
different orders. Neither order is "wrong" per the spec — but the differential-testing
contract requires ONE order, and the interpreter's is normative.

## 3. copy_prop corrupted sibling phis (major)
Deleting an unused phi popped the LAST jump argument instead of filtering positionally
by keep-mask — so a surviving φ received another variable's value on that edge. Caught
by the verifier in-tree (panic), but would have been a silent miscompile without it.
**The verifier paid for itself exactly as designed.**

## 4. Semi-pruned SSA's known wart, reproduced (major)
φ placement at iterated dominance frontiers can insert phis on paths where the variable
was never defined (short-circuit temps inside loops/arms). The renamer filled those
columns with version-0 cell ids → verify failure. Fixed by pruning provably-dead spurious
phis. Great report material: this is precisely the pruned-vs-semi-pruned trade-off from
Briggs et al., met in the wild.

## 5. Process-global state vs concurrent campaigns (critical)
Two campaigns in one process: A's timed run has live arrays; B's `reset_host_heap()`
frees them mid-loop → heap corruption (reproduced 0xc0000374). Plus unsynchronized
env-var writes racing the runtime's reads. Fix: whole-campaign mutex + feature-gate
honor. Lesson for the report: FFI-adjacent global state must be documented as a
concurrency contract, not a comment.

## 6. Const assignment accepted (major)
`const N = 1; N = 2;` passed sema; interpreter printed 2, JIT printed 1. The immutability
guard only covered loop variables. One-line fix, classic checker-completeness bug.

## Minor findings recorded (not fixed, by design)
- `fn main() -> ()` rejected by the main-signature check (works for other fns).
- const_fold folds i32 adds at i64 width (latent; pipeline output never reaches backend).
- Production reduction cells are 8- not 128-aligned (stride disjointness still holds).
- Observatory work-budget estimator can be bypassed by recursion-heavy tiny-literal code.
- Campaign parity gate checks interp-only when JIT absent (integration tests cover both).
- selftest's "both reject" arm unreachable under armed trap recorder.

Each is documented for future work; none affects correctness of shipped paths.
