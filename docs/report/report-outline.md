# Report Outline — HELIX Lite: An Automatic Parallelizing Compiler

*Everything in `docs/` maps to a report section. This file is the assembly guide.*

## Suggested structure (typical course-report shape)

### 1. Introduction (≈2 pages)
- The problem: sequential source, multicore machines, the gap.
- The claim: a compiler can *prove* which loops parallelize — and show its proof.
- Contributions list (language + pipeline + dependence engine + runtime + Observatory).
- Source: README.md intro, proposal brief.

### 2. The HELIX Language (≈3 pages)
- Design goals: small enough to fully specify, rich enough to need real analyses.
- Grammar appendix pointer: docs/notes/lang-spec.md (EBNF verbatim).
- Semantic choices worth defending: zero implicit coercions; i64 arithmetic core;
  braces mandatory; aliasing rejected statically (`f(a,a)`); checked-by-default
  bounds/division. Each choice DELETES a compiler subsystem — explain the engineering.
- Semantics table: srem `%`, saturating casts, IEEE floats, wrapping ints.

### 3. Compiler Architecture (≈4 pages)
- Pipeline figure (README diagram), crate map (10 crates, contracts in
  interface-contracts.md).
- Frontend: hand-written lexer + Pratt parser (precedence ladder table);
  recursive-descent statements; error recovery philosophy.
- Sema: bidirectional checking, symbol arena as stable indices, definite assignment
  dataflow, all-paths-return.

### 4. IR, SSA, and Optimization (≈5 pages)
- CFG design: basic blocks, phis with per-pred argument lists (Cranelift-style block
  params from day one).
- SSA construction: CHK dominators → dominance frontiers → semi-pruned φ placement →
  dominator-tree renaming. Why scalar-only SSA (LLVM precedent) — docs/notes/ssa-notes.md.
- The six passes, each with before/after IR from the Observatory OPT view:
  const-fold / const-prop / copy-prop / DCE / CSE / LICM (+simplify_cfg).
- The verifier: dominance, φ arity, reaching-defs after EVERY pass; regression tests
  named for each soundness bug it caught.

### 5. Dependence Analysis & Automatic Parallelization (≈6 pages — THE chapter)
- Theory: distance/direction vectors, levels, DOALL theorem (docs/notes/dependence-theory.md).
- The battery per dimension: ZIV → Strong/Weak-Zero/Weak-Crossing SIV → gcd+bounded-box;
  why every test is EXACT on HELIX's subscript grammar (stronger than production claims).
- Worked examples with REAL compiler output:
  - scale.hx → SAFE (show `helix loops` output)
  - recurrence_reject.hx → SEQUENTIAL (RAW dist 1) — the money screenshot
  - dot_reduction.hx → REDUCTION(+) with the checklist applied
  - gcd_box_test.hx → where GCD is inconclusive and the box test decides
  - stencil_2d_reject.hx → level-2 carried analysis discussion
- Reduction transform: private aligned accumulators, monoid seeds, post-join combine,
  FP reassociation honesty clause.

### 6. Code Generation (≈3 pages)
- Cranelift rationale vs LLVM (compile-time ms, block params ≙ our φs 1:1).
- Lowering rules table (docs/notes/cranelift-backend-notes.md).
- Checked semantics as generated guards (not OS faults); panic containment at the host
  boundary; no unwinding through JIT frames.
- Verified-API war stories: BlockArg sum type, MemFlags builder redesign (2026),
  WindowsFastcall ≡ extern "C" — cite docs/research/cranelift-api.md.

### 7. The Parallel Runtime (≈3 pages)
- Stage A/B runtimes + measured overhead delta graph.
- Static/dynamic/guided scheduling; libgomp chunk formula; cost gate.
- False sharing: 128-byte accumulator cells (docs/notes/parallel-runtime-notes.md).

### 8. Evaluation (≈4 pages)
- Methodology summary (docs/benchmarks/methodology.md): interleaved adaptive sampling,
  CV gating, checksummed parity gates, triad ceiling.
- Results tables + speedup figures (docs/benchmarks/results.md, figs/*.svg):
  saxpy 4.13×@8T bandwidth-bound story; dot 4.7×@24T; interp→native 20–270×;
  matmul@128 251× end-to-end.
- The honest sections: minmax demoted (two accumulators), small-N overhead,
  jacobi conservative verdict on flattened subscripts.
- Correctness: 450+ tests, selftest gauntlet interp≡JIT across all examples.

### 9. The Observatory (≈2 pages)
- Screenshots: pipeline stepper, CFG with amber backedges, loop cards (green/hazard/blue).
- Server-side layout rationale (offline demos; layout = graph algorithms = syllabus).

### 10. Related Work (≈1 page)
- Allen & Kennedy; Goff/Kennedy/Tseng; Cytron SSA; Briggs semi-pruned; CHK; Braun et al.;
  Cranelift/regalloc2; OpenMP reduction semantics; LLVM DependenceAnalysis; Polly/Graphite
  contrast (we stay exact-and-small instead of polyhedral-general).

### 11. Conclusions & Future Work (≈1 page)
- What multi-level affine analysis would unlock (jacobi rows); two-accumulator regions;
  while-loops; AOT object emission via cranelift-object (spike documented).

## Asset checklist
- [x] EBNF + type rules (notes/lang-spec.md)
- [x] Pipeline figure (README)
- [x] Loop-verdict screenshots (Observatory live; recurrence card verified)
- [x] Speedup SVGs ×15 (benchmarks/figs/)
- [x] Campaign JSON + meta (benchmarks/data/)
- [x] Demo script (docs/demo-script.md)
- [ ] CFG screenshot of jacobi (multi-loop nest) — capture during demo prep
- [ ] Stage-A/B overhead microbench chart (runtime tests have data; plot if time)
