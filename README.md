# HELIX Lite — an automatic parallelizing compiler

Write ordinary sequential numerical code. HELIX proves which loops are safe to run on
all your cores — and *shows you the proof*.

```helix
fn main() {
    let n = 50000000;
    let a: [f64] = zeros(n);
    let out: [f64] = zeros(n);
    for i in 0..n {
        out[i] = a[i] * 5.0 + 1.0;
    }
}
```

```
$ helix loops examples/scale.hx
Loop #1: RAW 0 / WAR 0 / WAW 0 => SAFE          ✓ parallelized × N threads

$ helix loops examples/recurrence_reject.hx
Loop #1: RAW 1 / WAR 0 / WAW 0 => SEQUENTIAL
    RAW a[i] <- a[i - 1] (carried by iteration distance 1, level 1)
```

## Measured results (laptop, 2026-08-25)

| Claim | Number |
|---|---|
| saxpy @16.8M f64, parallel speedup | **4.13× @ 8 threads** (bandwidth-bound) |
| dot-product reduction | **4.7× @ 24 threads** |
| matmul 128², interpreter → native JIT | **251×** |
| recurrence loop | **refused** — RAW distance-1 proven, runs sequential |
| cross-backend correctness | every example bit-identical interp ≡ JIT |

Full tables + figures: [docs/benchmarks/results.md](docs/benchmarks/results.md).

## The pipeline

source → lexer → Pratt parser → AST → type checker → CFG IR → SSA (semi-pruned,
CHK dominators) → constant folding/propagation · copy-prop · DCE · CSE · LICM →
loop detection → **dependence battery (ZIV · SIV family · gcd+bounded-box)** →
parallelization (DOALL + reduction recognition) → Cranelift → x86-64 machine code

## Try it

```bash
cargo run --release -p helix-cli -- observe     # the Observatory web UI
helix run examples/saxpy.hx                     # interpreter
helix run --backend jit examples/saxpy.hx       # JIT (identical output)
helix selftest                                  # differential gauntlet
```

## Layout

| Path | What |
|---|---|
| crates/helix-syntax | lexer + parser + AST |
| crates/helix-sema | types, scopes, static checks |
| crates/helix-ir | CFG IR, SSA, optimization passes + verifier |
| crates/helix-analysis | loops, dependence battery, parallelization plans |
| crates/helix-backend | Cranelift lowering + JIT engine + parallel regions |
| crates/helix-runtime | worker pool, schedules, reductions |
| crates/helix-engine | reference interpreter |
| crates/helix-bench | benchmark harness + kernel suite |
| crates/helix-observe | Observatory artifact builder + server |
| docs/ | lab notebook: research digests, decisions, notes, benchmarks, report outline |
| examples/*.hx | demo programs incl. deliberately rejected loops |

License: MIT
