# HELIX Report Outline (map from lab notebook to final write-up)

Every section below can be assembled almost verbatim from existing `docs/` material.

## 1. Introduction
- Motivation: automatic parallelization as the intersection of every compiler course topic.
- The pitch: sequential source in; *proven* multicore execution or a precise refusal, out.
- Contributions list (language + full pipeline + dependence engine + Observatory).
- Source: README.md intro.

## 2. Language design (HELIX v1)
- Grammar & typing rules → docs/notes/lang-spec.md (frozen spec, EBNF, type table).
- Design decisions worth defending:
  - zero implicit coercions (and why that protects reduction soundness),
  - braces mandatory / assignment-as-statement (dangling-else & typo classes eliminated),
  - arrays as fat references, aliasing rejected statically (`f(a,a)`) — deletes alias analysis,
  - checked semantics by default (div/bounds), saturating casts.
- Source: lang-spec.md + research/language-spec-review.md.

## 3. Compiler architecture
- Pipeline diagram (README) + crate map (10 crates, one concern each).
- Interface-contract-driven development: how parallel agents built disjoint crates against
  frozen contracts; deviations tracked. Source: notes/interface-contracts.md + devlog.

## 4. Frontend
- Lexer/Pratt parser details; error recovery posture. Source: syntax crate docs + tests.
- Semantic analysis: bidirectional checking, symbol arenas, definite assignment dataflow,
  all-paths-return. Source: sema crate module docs + tests/spec_tests.rs.

## 5. Reference interpreter
- Tree-walking design; exact semantics conformance (srem %, saturating casts, IEEE min/max);
  checksum definition for cross-backend comparison. Source: engine crate docs.

## 6. IR and SSA  ⭐ course-core chapter
- CFG construction: diamonds, short-circuit lowering, early-return exit blocks.
- CHK dominators, dominance frontiers, semi-pruned SSA, renaming.
- Why arrays stay out of SSA; φ→Cranelift block params (no out-of-SSA!).
- Verifier discipline after every pass.
- Sources: notes/ssa-notes.md + ir crate docs + golden tests.

## 7. Optimization passes
- Six passes with before/after evidence. Sources: notes/optimization-passes.md,
  passmod snapshots (Observatory OPT view screenshots).

## 8. Loop dependence analysis  ⭐ the centerpiece chapter
- Theory: distance/direction vectors, levels, DOALL theorem. → notes/dependence-theory.md.
- The battery: ZIV → SIV family → gcd+box Diophantine; worked examples per test.
- Reduction recognition rules + FP honesty clause.
- Golden verdicts table for all examples (analysis crate tests).

## 9. Code generation
- CLIF mapping table, WindowsFastcall ABI notes, guarded traps vs SEH.
- Parallel lowering: body extraction, ctx packing, registry-after-finalize trick.
- Sources: notes/cranelift-backend-notes.md + backend crate docs.

## 10. Runtime
- Pool vs spawn-per-call overhead graph; schedules; false-sharing padding; cost gate.
- Source: notes/parallel-runtime-notes.md + runtime crate tests.

## 11. Evaluation  ⭐
- Methodology → benchmarks/methodology.md (interleaving, CV gating, anti-cheat checklist).
- Headline results tables + figures (campaign.json + tools/plot_bench.py outputs).
- Correctness: selftest gauntlet 16/16 interp-vs-JIT parity.
- Honest cases: small-N losses, jacobi conservative verdict, sieve stride fix history.

## 12. Related work
- Allen & Kennedy, Goff/Kennedy/Tseng, Cytron SSA, Briggs semi-pruned, CHK dominators,
  LLVM DependenceAnalysis/Polly, GCC Graphite/parloops, OpenMP semantics.

## 13. Conclusions & future work
- Two-level affine extension (jacobi), multi-reduction loops (minmax demotion),
  guided scheduling demos, AOT object emission via cranelift-object.

## Appendices
- A: full language spec · B: artifact JSON schema · C: benchmark meta/environment ·
  D: devlog excerpts showing the engineering process.
