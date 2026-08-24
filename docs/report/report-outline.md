# HELIX Report Outline

*Every section maps to material that already exists in this repository. Writing the
final report is assembly, not archaeology.*

## 1. Introduction (≈2 pages)
- Problem: sequential numerical code underuses multicore hardware; manual parallelization
  is error-prone; existing auto-parallelizers are black boxes.
- Contribution: a complete compiler that *proves* loop safety and *shows its work*
  (Observatory), from source to x86-64 machine code.
- Demo teaser: the scale vs recurrence pair (parallelized ×N vs rejected RAW-distance-1).
- Sources: README.md, docs/notes/lang-spec.md intro.

## 2. The HELIX Language (≈3 pages)
- Design goals: small enough to specify completely, rich enough to demo.
- Grammar (EBNF) + type rules + semantics table → docs/notes/lang-spec.md verbatim.
- Deliberate restrictions and what they buy:
  - zero implicit coercions → reduction associativity arguments stay honest;
  - `f(a,a)` rejected → dependence analysis never guesses about aliasing;
  - flat arrays → affine subscripts stay exact;
  - braces mandatory → no dangling-else.
- Sources: lang-spec.md, examples/*.hx.

## 3. Frontend (≈4 pages)
### 3.1 Lexing
- hand-written scanner, maximal munch, span tracking, comment nesting rules.
### 3.2 Parsing
- recursive descent + Pratt expression parsing; precedence ladder; why assignment is a
  statement; else-if chains as nested spines.
- Error handling philosophy: precise spans, "expected X found Y", first-error reporting
  with carets (screenshot: `helix check examples/type_errors.hx`).
### 3.3 Semantic analysis
- two-pass signatures-then-bodies (recursion for free);
- bidirectional checking with literal adaptation but NO coercions;
- definite-assignment reality: mandatory initializers make use-before-init structurally
  impossible (interesting design finding!);
- all-paths-return structural check.
- Sources: crates/helix-syntax (86 tests), helix-sema (12 conformance tests).

## 4. Intermediate Representation & SSA (≈5 pages)
- CFG construction: diamonds, short-circuit desugaring, dedicated exit blocks.
- Dominators (CHK) + dominance frontiers + semi-pruned φ placement + renaming.
- Arrays out of SSA (LLVM precedent); slot-value pre-SSA model → SSA renaming.
- No out-of-SSA translation: φ → Cranelift block params 1:1 (lost-copy/swap problems
  delegated to regalloc2).
- The verifier: dominance, φ arity/pred-match; runs after EVERY pass.
- Show `helix dump ssa examples/ssa_demo.hx` output (the textbook example).
- Sources: ssa-notes.md, crates/helix-ir/{dom,ssa,verify}.rs.

## 5. Optimizations (≈4 pages)
- Six passes: const-fold, const-prop(+branch fold+CFG simplify), copy-prop, DCE, CSE, LICM.
- For each: what it proves, SSA's role, before/after IR snippet from Observatory OPT view.
- Engineering war stories (great report material):
  - compact() must renumber φ args too;
  - branch folding must clone-before-set_term or pred lists corrupt;
  - multiply-defined ids in DCE.
- Sources: optimization-passes.md, passmod.rs snapshots, devlog 2026-08-24.

## 6. Dependence Analysis ⭐ (≈6 pages)
- Theory: distance/direction vectors, levels, the parallelization theorem.
- The battery: ZIV → Strong SIV → Weak-Zero → Weak-Crossing → gcd+box; exactness claim
  for HELIX's grammar; i128 analyzer arithmetic.
- Reduction recognition checklist + monoid identities + FP honesty clause.
- Worked examples with real verdicts:
  - scale.hx → SAFE (table of accesses);
  - recurrence_reject.hx → RAW dist 1 REJECTED;
  - dot_reduction.hx → REDUCTION(+);
  - gcd_box_test.hx → box-intersection decides where GCD alone can't;
  - stencil_2d_reject.hx → level-2 carried, inner loop still parallel.
- Loop #N summary-line format (`RAW 0 / WAR 0 / WAW 0 => SAFE`).
- Sources: dependence-theory.md, deps.rs tests, plan.rs reports.

## 7. Code Generation (≈4 pages)
- Why Cranelift: JIT speed, safe Rust, clean SSA-with-block-params match.
- Windows x64 ABI facts learned by experiment (WindowsFastcall ≡ extern "C",
  BlockArg wrapping, MemFlags redesign, finalize flow) — cite jit_spike.rs as evidence.
- Checked semantics as guarded branches (div-by-zero, bounds) instead of SEH traps.
- Builtin lowering choices (sqrt instruction; min/max via select+fcmp for NaN rules).
- Parity testing vs interpreter (checksums, bit-exactness policy).
- Sources: docs/research/cranelift-api.md, jit_spike.rs, backend tests.

## 8. Parallel Runtime (≈4 pages)
- Two stages measured (scope-spawn vs pool); schedules static/dynamic/guided;
  libgomp chunk formula; cost gate.
- False sharing: aligned accumulator cells; measured impact.
- Reduction combine protocol.
- Honest expectations: bandwidth ceilings, Amdahl, small-N losses.
- Env knobs for labs.
- Sources: parallel-runtime-notes.md, helix-runtime tests (41).

## 9. Evaluation (≈5 pages)
- Methodology summary (interleaving, CV gating, anti-cheat checksums) → methodology.md.
- Results tables per kernel × tier × threads; efficiency columns; STREAM-triad context.
- The three headline narratives:
  1. interpreter → native: orders of magnitude (with Rust-twin validation);
  2. unoptimized → optimized native: the passes' measurable value;
  3. sequential → parallel: scaling curves + the rejected-loop control case.
- Threats to validity, honestly listed.
- Sources: docs/benchmarks/data/*.json, figures via tools/plot_bench.py.

## 10. The Observatory (≈2 pages)
- Architecture: artifact JSON contract, server-side layout, offline-first.
- Phase tour with screenshots; redundant verdict encoding rationale.
- Sources: artifact-schema.md, web/, dev fixtures.

## 11. Related Work (≈1 page)
- Allen&Kennedy, Goff/Kennedy/Tseng, Cytron et al., Briggs et al., CHK, LLVM/GCC
  (-ftree-parallelize-loops, Polly/Graphite), OpenMP semantics; Cranelift/wasmtime.
- Position: teaching-complete pipeline with exact analysis on a restricted grammar.

## 12. Conclusions & Future Work (≈1 page)
- What worked (SSA-native lowering, contracts-first parallel development, verifier spine).
- Future: privatization/array expansion, while-loops, strided ranges, SIMD,
  cranelift-object AOT path (already researched).

## Appendices
- A: full language spec (lang-spec.md).
- B: benchmark raw data + regeneration instructions.
- C: artifact JSON schema (artifact-schema.md).
- D: build/test reproduction commands (`cargo test --workspace`, `cargo run -p helix-cli`).
