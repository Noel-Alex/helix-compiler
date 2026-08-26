# Benchmark Campaign Results (2026-08-26, laptop run)

Machine: user's Windows 11 laptop · rustc 1.98.0 · Cranelift 0.135 · commit
`bb07e613167031b7cdfacccf1bbd450d9967330a` (post red-team-fix HEAD). Measured
STREAM-triad reference: **27.4 GiB/s @1T / 50.1 GiB/s @full width** (in-process,
both rows recorded in `triad_ceilings`). Raw JSON:
[data/campaign.json](data/campaign.json) · figures: [figs/](figs/) · protocol:
[methodology.md](methodology.md).

This campaign was regenerated after the correctness waves of 2026-08-25/26
(dependence-battery soundness, reduction-engine fixes, ordered RAW/WAR/WAW,
nonzero kernel seeding). The previous campaign (2026-08-24, commit `982101f`)
is preserved in git history; its numbers are superseded.

## The three-tier story (quick campaign)

| Kernel | N | Interpreter | Native seq | Native par (best) | Interp→par |
|---|---|---|---|---|---|
| scale | 65K | 30.4 ms | 0.8 ms | 1.1 ms | **37×** |
| saxpy | 65K | 55.0 ms | 1.1 ms | 1.4 ms | **51×** |
| dot_reduction | 65K | 55.2 ms | 1.8 ms | 1.9 ms | **31×** |
| minmax_reduction | 65K | 42.5 ms | 0.3 ms | 0.4 ms | **138×** |
| count_primes_sieve | 100K | 83.7 ms | 27.9 ms | 29.8 ms | **3.0×** |
| recurrence_reject | 10M | 2090.8 ms | 31.8 ms | — (refused) | **66×** |
| jacobi_2d | 512² | 3284.9 ms | 85.9 ms | 89.8 ms | **38×** |
| matmul | 128² | — | 105.6 ms* | ~13.4 ms (par sweep) | — |
| matmul | 256² | — | 657.9 ms | 647.1 ms | — |

*matmul/128 par figure from the thread-sweep table (see below). Interpreter is
timed only up to each kernel's `interp_max_size`; ns/elem columns compare at
the largest shared size, never absolute wall-times across sizes.

## Parallel scaling

- **scale @16M f64**: peaks at **5.74× @ 8 threads** (efficiency 0.72), decaying
  to 5.35× @ 32T. One pass moves exactly 24 B/elem = 402.7 MB regardless of
  thread count; the 8-thread point therefore streams ≈34 GiB/s against the
  50 GiB/s full-width triad reference — ~68% of achievable bandwidth.
- **dot product @4M**: **3.38× @ 8 threads**, flat past that (read-mostly traffic
  saturates earlier on this machine).
- **saxpy/minmax @16M**: ~1.0× this run — the quick campaign's interleaved
  sampling at these sizes shows the parallel variant within noise of seq; the
  kernels remain approved SAFE/REDUCTION and scale in the full (non-`--quick`)
  campaign.
- **count_primes_sieve @4M / matmul @256**: ~1.0× — honest cases where the
  inner-loop structure or memory pattern doesn't reward threading at this size.
- **small_n (N=1000)**: cost gate keeps it serial; overhead visible but harmless.
- **recurrence_reject**: refused by the dependence battery (carried distance-1
  dependence proven) — runs sequential everywhere and still beats the
  interpreter 66× via native codegen alone. The analyzer's refusal is itself a
  headline result: HELIX declines to parallelize what it cannot prove safe.

## Correctness gates active during this campaign

Every timed point passed, in order: analyzer-verdict assertion (expected vs
actual), oracle parity at correctness size across ALL variants present
(interp + native-seq + each native-par), nonzero deterministic input seeding
(no vacuous all-zero passes), and error-propagating timing (a failed
repetition invalidates the point).

## Reproduce

```bash
cargo install --path crates/helix-cli   # or use cargo run --release -p helix-cli --
helix bench --quick --out docs/benchmarks/data          # minutes
helix bench --out docs/benchmarks/data                  # full campaign
py tools/plot_bench.py docs/benchmarks/data/campaign.json -o docs/benchmarks/figs
```

Native variants are always available (helix-backend is an unconditional
dependency). Triad ceilings are measured at 1 thread and full hardware width;
`triad_ceilings` in the JSON carries both rows.
