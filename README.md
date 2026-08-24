# HELIX Lite — an automatic parallelizing compiler

Write ordinary sequential numerical code. HELIX proves which loops are safe to run on all
your cores — and *shows you the proof*.

```helix
fn main() {
    let n = 50000000;
    let a: [f64] = zeros(n);
    let out: [f64] = zeros(n);
    for i in 0..n {
        out[i] = a[i] * 5.0;
    }
}
```

```
Loop #1  ── READ a[i], WRITE out[i] ── loop-carried deps: none
✓ SAFE → parallelized across N threads

for i in 1..n { a[i] = a[i-1] + 1; }

Loop #1  ✗ REJECTED — RAW dependence: a[i] ← a[i-1] (distance 1)
```

## The pipeline

source → lexer → Pratt parser → AST → type checker → CFG IR → SSA →
constant folding / propagation / copy-prop / DCE / CSE / LICM →
loop detection → **dependence analysis (GCD/SIV/Banerjee tests)** →
parallelization (DOALL + reduction recognition) → Cranelift → x86-64 machine code

## Status

Under construction. Milestones tracked in `docs/devlog/`, design decisions in
`docs/decisions/`, verified implementation research in `docs/research/`.

## Layout

| Path | What |
|---|---|
| crates/helix-syntax | lexer + parser + AST |
| crates/helix-sema | types, scopes, static checks |
| crates/helix-ir | CFG IR, SSA, optimization passes |
| crates/helix-analysis | loops, dependence battery, parallelization verdicts |
| crates/helix-backend | Cranelift JIT lowering |
| crates/helix-runtime | parallel_for runtime (pool, schedules, reductions) |
| crates/helix-engine | reference interpreter |
| crates/helix-bench | benchmark harness + kernels |
| crates/helix-observe | Observatory web UI server |
| examples/*.hx | demo programs incl. deliberately rejected loops |
| docs/ | lab notebook for report writing |

License: MIT
