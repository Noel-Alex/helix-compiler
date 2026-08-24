# Devlog — 2026-08-24: IR fixes, dependence engine wired, Observatory UI, GitHub live

## Shipped

- **GitHub repo live**: https://github.com/Noel-Alex/helix-compiler (public). Clean history,
  MIT license, LF-normalized, secret-scanned, 1 MB `.git`.
- **helix-engine complete** (agent): tree-walking interpreter, 56 tests. Exact frozen
  semantics incl. srem `%`, saturating casts, checked div/bounds with line numbers,
  short-circuit observability, FNV checksums for differential testing.
- **helix-ir optimizer bugfixes** (agent, adversarial pass):
  1. `compact()` renumbered terminator BlockIds but NOT φ-argument BlockIds — after any
     block deletion every surviving φ silently named a different block. Root cause of the
     matmul/jacobi failures.
  2. `const_prop` canonicalized duplicate constant defs globally; a def in one branch
     doesn't dominate sibling uses. Now only entry-block defs lead (entry dominates all).
  3. `dce` kept only the first def-site per multiply-defined id (multi-return `$ret`
     accumulator), sweeping live adds and stranding jump arguments.
  Plus 4 regression tests each reproducing one signature.
- **Observatory frontend complete** (agent): all 8 phases (SOURCE→TOKENS→AST→CFG→SSA→OPT→
  LOOP ANALYSIS→BENCH), keyboard nav, hazard-striped rejection cards, redundant verdict
  encoding (color+dash+glyph) for colorblind/grayscale safety, offline d3, dev fixtures +
  node dev server. Verified via DOM/a11y assertions, zero console errors.
- **helix-analysis reconciled to the real IR API** (by me): natural loops via
  `dom::natural_loops`, nest forest with depths, affine access extraction over the
  slot-value model, ZIV/StrongSIV/WeakZero/WeakCrossing/gcd-box battery in i128,
  reduction recognition on header φs, polished LoopReport strings.
- Course notes: dependence-theory.md, ssa-notes.md, optimization-passes.md,
  parallel-runtime-notes.md, benchmarks/methodology.md, artifact-schema.md.

## Honest moments

- One of my own deps-battery tests had a wrong expectation: `a[i-100]` vs `a[i]` over
  `[0,99)` has NO feasible iteration pair — `Independent` was the *correct* answer. The
  bounds-awareness caught me, not the code. Fixed the test to use 200 trips.
- The IR builder evolved its API past my drafted analysis skeleton (struct-variant
  CallTarget carrying typed arg subtrees; slot-value locals; `Doms` not `Dominators`).
  Reconciliation took a full pass — the interface-contracts addendum helped but drift
  still happened. Lesson recorded: contracts need the SAME review cadence as code.

## State

243/243 tests workspace-wide, clippy `-D warnings` clean, fmt clean.

## Next

M9 backend (build+verify workflow running), M10 parallel lowering, M11 benchmarks,
M12 Observatory server wiring, M13 polish/release.
