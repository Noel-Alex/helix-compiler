# Benchmark Methodology

*How HELIX measures itself honestly. Every number in the report comes from this protocol.*

## Harness design (hand-rolled, ~200 lines, no criterion)

1. **Warmup**: 3 untimed rounds per variant. Round 1 pays JIT compilation — timed
   *separately* and reported as its own column (compile time is a real cost but must not
   contaminate steady-state execution numbers).
2. **Adaptive sampling**: a pilot run chooses inner repetitions R so each sample lasts
   100–250 ms; then k = 15 samples are taken.
3. **Statistics**: median (robust to outliers), minimum (the "true cost" lower bound),
   and coefficient of variation CV%. If CV > 5%, the harness re-runs that variant rather
   than reporting noise.
4. **Interleaving**: variants alternate round-robin (interp → seq → par₁ → par₈ → interp →
   …) instead of finishing one before starting the next — defeats thermal drift and
   background-load drift across the campaign window.
5. **Priority**: the harness raises its own process priority (HIGH) for the run.

## Anti-cheat checklist (baked into the harness)

- JITed kernels are invoked through raw extern fn pointers — the host compiler cannot
  hoist or DCE them.
- After every timed run, every output array is checksummed (FNV-1a over bytes) and
  **all execution tiers must agree** — bit-exact for integer/min/max kernels,
  relative-ε for FP `+`/`×` reductions (documented nondeterminism).
- Buffers are pre-allocated and *touched* once before timing (page faults excluded).
- Values kept in 1..100 magnitude — no denormal stalls.
- The thread-pool barrier is inside the timed region (honest end-to-end numbers).

## Presentation rules (Hoefler-style)

- State the baseline explicitly on every table: speedup vs interpreter AND vs native-seq.
- Always show absolute performance alongside speedups (ns/elem and achieved GB/s columns).
- Parallel efficiency E_p = S_p / p next to each speedup, against both logical (32) and
  physical core counts.
- Bandwidth context: memory-bound kernels get their measured GB/s compared against an
  in-process STREAM-triad ceiling, so "we got 5.2×" becomes "we reach 82% of the machine's
  usable bandwidth".
- Include cases where threading loses (small N) — a compiler that refuses unsound or
  pointless parallelization is the *point* of the project.
- A hand-written Rust twin of one kernel shows HELIX-native lands within ~1.2–2× of it,
  validating the whole pipeline and making interpreter-vs-native ratios credible.

## Environment capture (reproducibility bundle)

`meta.json` per campaign: CPU model/cores (sysinfo), RAM, OS build, rustc version string,
cranelift version from Cargo.lock, git rev, date/power-plan note, RNG seed for input
generation. Per-kernel result JSONs keep **raw samples**, so medians can be recomputed and
figures regenerated offline via [tools/plot_bench.py](../../tools/plot_bench.py).

## Kernel suite

| Kernel | Shape | N | Expected verdict |
|---|---|---|---|
| scale | out[i] = a[i]*5 | 2²⁵ f64 | SAFE |
| saxpy | y[i] += s*x[i] | 2²⁴–2²⁵ f64 | SAFE (memory-bound) |
| dot | dot += a[i]*b[i] | 2²⁶ f64 | REDUCTION(+) |
| minmax | min(a), max(a²) | 2²⁴ f64 | REDUCTION(min,max) |
| sieve | Eratosthenes | 10⁷ bool | inner loop SAFE |
| jacobi | 5-point stencil | 4096² f64 | row loops SAFE |
| matmul | C=A·B, k-reduction | 512²,768² f64 | i-loop SAFE + REDUCTION(k) |
| recurrence | a[i]=a[i−1]+1 | 10⁵ i64 | **REJECTED** (RAW dist 1) |
| small_n | tiny copy loop | 10³ f64 | gate → serial (honest loss case) |

## What would invalidate our claims (pre-registration of failure modes)

- Timer granularity artifacts → Instant is QPC-backed (~100 ns); sample durations ≥100 ms.
- Background load → interleaving + CV gating + priority raise.
- Optimizer differences between tiers → all native tiers share one compiled binary;
  only the runtime schedule differs.
- Cherry-picking → every kernel's full raw sample set is committed, figures regenerate
  from JSONs.
