# Benchmark Campaign Results (2026-10-11, laptop run — post-hardening)

> **Post-hardening campaign.** This run was recorded after the soundness and
> validity pass: the optimizer now runs inside the one shared compile pipeline
> every executing consumer uses (`helix_artifact::prepare_pipeline`), so
> `native-seq` is Cranelift native sequential compiled from an EMPTY parallel
> plan (genuinely sequential), sweep rows really run the thread count they are
> labelled with (the compiler hint no longer caps participants at 8), the parity
> gate compares the interpreter's FULL-state checksum (printed lines + the
> arrays `main` leaves bound), and each variant row carries its one-time
> `compile_ms`. The pre-hardening campaign (2026-08-26, commit `bb07e61`) is
> preserved verbatim in `data/pre-hardening/`; its `native-seq` rows were
> compiled with the parallel plan attached and its >8-thread sweep rows were
> capped at 8 participants, so the two campaigns are not comparable row by row.

Machine: 13th Gen Intel Core i9-13900HX (32 logical / 24 physical cores) ·
Windows 11 Home · rustc 1.99.0 · Cranelift 0.135 · commit `ce5f69f`. Measured
STREAM-triad reference: **27.3 GiB/s @1T / 52.0 GiB/s @32 threads** (both rows
in `triad_ceilings`). Raw JSON: [data/campaign.json](data/campaign.json) ·
figures: [figs/](figs/) · protocol: [methodology.md](methodology.md) ·
pre-hardening history: [data/pre-hardening/](data/pre-hardening/).

## The three-tier story

| Kernel | N | Interpreter | Native seq | Native par | Interp→seq | par vs seq |
|---|---|---|---|---|---|---|
| scale | 65K | 15.8 ms | 0.24 ms | 0.50 ms | **65.9×** | 0.48× |
| saxpy | 65K | 49.7 ms | 0.67 ms | 1.47 ms | **74.4×** | 0.45× |
| dot_reduction | 65K | 44.6 ms | 0.62 ms | 1.18 ms | **71.9×** | 0.53× |
| minmax_reduction | 65K | 41.3 ms | 0.30 ms | 0.30 ms | **138×** | 1.01× |
| count_primes_sieve | 100K | 83.9 ms | 0.62 ms | — refused | **134.6×** | — |
| recurrence_reject | 10M | 1997.5 ms | 30.3 ms | — refused | **66×** | — |
| small_n | 1K | 0.42 ms | 0.003 ms | 0.02 ms | **140×** | 0.15× |
| jacobi_2d | 512² | 3097.2 ms | 17.2 ms | 96.9 ms | **179.8×** | 0.18× |
| matmul | 128² | 1199.1 ms | 5.9 ms | 117.8 ms | **203.7×** | 0.05× |

Largest measured size per kernel (interpreter omitted above its
`interp_max_size`):

| Kernel | N | Native seq | Native par | par vs seq |
|---|---|---|---|---|
| scale | 16M f64 | 131.8 ms | 71.1 ms | **1.86× faster** |
| saxpy | 16M f64 | 202.0 ms | 92.0 ms | **2.20× faster** |
| dot_reduction | 4M f64 | 45.9 ms | 35.7 ms | **1.28× faster** |
| minmax_reduction | 16M f64 | 85.0 ms | 92.3 ms | 0.92× (par slower) |
| jacobi_2d | 1024² | 69.2 ms | 311.3 ms | 0.22× (par slower) |
| matmul | 256² | 46.7 ms | 658.4 ms | 0.07× (par slower) |
| count_primes_sieve | 4M bool | 29.1 ms | — refused | — |
| recurrence_reject | 10M i64 | 30.3 ms | — refused | — |

## Parallel scaling (sweep rows are real participant counts)

- **saxpy @16M f64**: peaks at **7.84× @ 32T** (8T: 5.79×; 16T: 7.40×). One
  pass streams 24 B/elem, so the 8-thread point moves ≈34 GiB/s against the
  52 GiB/s full-width triad reference.
- **scale @16M f64**: peaks at **7.18× @ 24T** (8T: 5.52×), same bandwidth
  story.
- **dot_reduction @4M**: peaks at **5.25× @ 24T** (8T: 3.08×).
- **Multi-array kernels do not scale here.** jacobi_2d, matmul and the
  mid-size points sit at ≈1.0× across the whole sweep, and their parallel
  rows are SLOWER than the optimized sequential row (matmul/256 par is 14×
  slower than seq). Cause, measured not guessed: a parallel region's body
  reaches every captured array and scalar through `helix_ld_*` host calls the
  JIT cannot inline, so a parallel body pays a large constant cost per memory
  access. The optimized sequential code pays nothing comparable. Threading
  only pays where the loop is long and simple enough for that overhead to
  amortize (saxpy/scale/dot at 4M–16M). Inlining cached accesses inside the
  extracted body is the obvious future work and is deliberately NOT claimed as
  done anywhere in this repo.
- **small_n (N=1000)**: approved SAFE, but the runtime cost gate
  (`max(1024, 1024×participants)` iterations) keeps it serial — the honest
  small-loss case.
- **count_primes_sieve**: the analysis REFUSES the inner sweep. The stride
  multiplier `i` is loop-invariant and cannot be proven non-zero, so the sole
  write's injectivity is unprovable — a sound refusal (if the stride were ever
  zero, every iteration would alias). Recovering it needs range reasoning over
  the outer induction variable, which the analysis does not yet do; the
  pre-hardening campaign approved this loop on an unproven assumption.

## Correctness gates active during this campaign

Every timed point passed, in order: analyzer-verdict assertion (expected vs
actual, on the OPTIMIZED IR the pipeline now analyzes), oracle parity at the
correctness size across ALL variants present, full-state checksum comparison
(exact for `interp` and `native-seq`; `native-par` checksum-compared only when
the kernel declares no tolerance, because a documented ε means its sums
legitimately reassociate), error-propagating timing (a failed repetition
invalidates the point), and deterministic input seeding. `native-seq` compiles
from an empty parallel plan, so its row cannot secretly contain threaded
execution. Interpreter rows charge the tree-walker's real allocations;
`hx_heap_bytes`/`alloc_bytes`-style counters keep the same accounting on the
other side of the pipeline.

## Reproduce

```bash
cargo install --path crates/helix-cli   # or use cargo run --release -p helix-cli --
helix bench --quick --out docs/benchmarks/data          # smoke: smallest size, {1, hw} sweep, 5 samples
helix bench --out docs/benchmarks/data                  # full campaign
node tools/plot_bench.mjs docs/benchmarks/data/campaign.json -o docs/benchmarks/figs
```

Native variants are always available (helix-backend is an unconditional
dependency). Triad ceilings are measured at 1 thread and full hardware width;
`triad_ceilings` in the JSON carries both rows. The pre-hardening campaign is
kept at `data/pre-hardening/` for development history.
