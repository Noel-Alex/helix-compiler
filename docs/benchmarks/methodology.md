# Benchmark Methodology

*How HELIX measures itself honestly. Every number in the report comes from this protocol.*

## Harness design (hand-rolled, ~200 lines, no criterion)

1. **Warmup**: 3 untimed rounds per variant. JIT compilation itself happens
   once per variant, before the pilot, and is never folded into a timed sample —
   compile time is a real cost, so it stays out of steady-state execution
   numbers (the CLI's `helix run --backend jit` reports it separately).
   Every variant row also carries `compile_ms`, the Cranelift compile time
   measured once at construction (`0.0` for the interpreter).
2. **Adaptive sampling**: a pilot run chooses inner repetitions R so each sample lasts
   100–250 ms; then k = 15 samples are taken.
3. **Statistics**: median (robust to outliers), minimum (the "true cost" lower bound),
   and coefficient of variation CV%. If CV > 5%, the harness re-runs that variant rather
   than reporting noise.
4. **Interleaving**: variants alternate round-robin (interp → seq → parₘᵢd, where
   the parallel variant pins the sweep's mid-range thread count) instead of
   finishing one before starting the next — defeats thermal drift and
   background-load drift across the campaign window.
5. **No priority games**: the thread-pool barrier is inside the timed region
   (honest end-to-end numbers) and the harness never changes process priority —
   it runs at whatever priority the caller gave it.

## Anti-cheat checklist (baked into the harness)

- JITed kernels are invoked through raw extern fn pointers — the host compiler cannot
  hoist or DCE them.
- **Parity gate, once per kernel**: every variant shape the campaign will time —
  interpreter, native-seq, and each pinned native-par<P> — runs once at the
  kernel's correctness size and must match the registry's oracle printed lines.
  Integer and min/max kernels must match exactly; the FP `+`/`×` reductions (`dot`,
  `matmul`) are compared under a relative tolerance (ε = 1e-9) because parallel
  partial sums reorder — documented nondeterminism, not a bug.
- **Full-state checksum in the same gate**: `interp` and `native-seq` must also
  produce the interpreter's exact full-state checksum (printed lines + the
  arrays `main` leaves bound). `native-par` is checksum-compared only when the
  kernel declares no tolerance; a documented ε means its sums legitimately
  reassociate. Matching printed lines alone would let a kernel that computes the
  right visible values and corrupts memory pass the gate.
- **Checksum definition**: the content witness is the interpreter's FNV-1a stream
  — every printed line with its newline terminator, then the elements of every
  array still visible at the end of `main` (float bit patterns, bools as 0/1). A
  program that computes the right values and prints none is still caught by the
  array half of that stream.
- Allocation stays inside the measured program (HELIX programs allocate with
  `zeros(n)`; nothing pre-sized is handed in), so allocation cost is included
  rather than hidden. The harness only reclaims the JIT host heap between timed
  repetitions so a campaign cannot leak.
- Values stay in the ±50 band (the suite's seeded inputs span roughly −12..+48)
  — no denormal stalls, and no all-zero pass can fold a loop vacuously.
  (`minmax_reduction` is the deliberate exception: its input is zeros and its
  oracle is `0.0`.)

## Presentation rules (Hoefler-style)

- State the baseline explicitly on every table: speedup vs interpreter AND vs
  native-seq. (`native-seq` is Cranelift native sequential — an empty parallel
  plan — which is what makes it a fair baseline. Rows recorded before that
  variant was pinned sequential are labelled as such in the campaign notes.)
- Always show absolute performance alongside speedups (ns/elem and achieved GB/s columns).
- Parallel efficiency E_p = S_p / p is published for every swept participant
  count alongside each speedup: the JSON carries one efficiency row per p and the
  figures plot the curve against the swept thread counts. The machine's logical
  and physical core counts live in `meta.json`; they are recorded, not drawn on
  the axis.
- Bandwidth context: memory-bound kernels get their measured GB/s compared against an
  in-process STREAM-triad ceiling, so "scale peaks at 5.74× @ 8 threads" becomes
  "≈34 GiB/s, about 68% of the 50.1 GiB/s full-width reference".
- Include cases where threading loses (small N) — a compiler that refuses unsound or
  pointless parallelization is the *point* of the project.
- Hand-written Rust twins of the kernels exist in `rust_twins` as independent
  oracle recomputations, and are the intended speed yardstick. The campaign does
  **not** currently publish a twin-vs-HELIX-native timing row, so no such ratio is
  claimed anywhere in the report.
- No native number is ever quoted for a browser run: a wasm run only reports what
  the module itself did on the reader's machine.

## Environment capture (reproducibility bundle)

`meta.json` per campaign: CPU model/cores (sysinfo), RAM, OS build, rustc version string,
cranelift version from Cargo.lock, git rev, date/power-plan note, RNG seed for input
generation. Per-kernel result JSONs keep **raw samples**, so medians can be recomputed and
figures regenerated offline via [tools/plot_bench.py](../../tools/plot_bench.py).

## Kernel suite

| Kernel | Shape | N (measured sweep) | Expected verdict |
|---|---|---|---|
| scale | out[i] = a[i]*5 | 2¹⁶, 2²⁴ f64 | SAFE |
| saxpy | y[i] += s*x[i] | 2¹⁶, 2²⁴ f64 | SAFE (memory-bound) |
| dot | dot += a[i]*b[i] | 2¹⁶, 2²² f64 | REDUCTION(+) |
| minmax | min(a), max(a²) | 2¹⁶, 2²⁴ f64 | REDUCTION(min,max) |
| sieve | Eratosthenes | 10⁵, 4·10⁶ bool | **REFUSED** (stride not provably non-zero) |
| jacobi | 5-point stencil | 512², 1024² f64 | row loops SAFE |
| matmul | C=A·B, k-reduction | 128², 256² f64 | i-loop SAFE + REDUCTION(k) |
| recurrence | a[i]=a[i−1]+1 | 10⁵, 10⁷ i64 | **REJECTED** (RAW dist 1) |
| small_n | tiny copy loop | 10³ f64 | gate → serial (honest loss case) |

A region needs at least `max(1024, 1024 × participants)` iterations before the
runtime will thread it; `small_n` at 10³ sits under that floor at every thread
count, which is why it is the campaign's honest serial row. Participant counts
never exceed `min(HELIX_NTHREADS, available_parallelism)`: the region's emitted
thread hint is a ceiling no host can reach, so a row labelled `p` really runs
`p` participants, clamped only by hardware.

`--quick` is a smoke run, not a short campaign: the smallest declared size per
kernel, a {1, hardware} participant sweep, and 5 samples per variant, driven
through the same `run_campaign` path. It proves the wiring runs; it is not a
source of publishable numbers.

## What would invalidate our claims (pre-registration of failure modes)

- Timer granularity artifacts → Instant is QPC-backed (~100 ns); sample durations ≥100 ms.
- Background load → interleaving + CV gating. (There is no priority raise to fall back on:
  the harness runs at the caller's priority, so a noisy host shows up as CV, not as a
  silent number.)
- Optimizer differences between tiers → all native tiers share one compiled binary;
  only the runtime schedule differs.
- Cherry-picking → every kernel's full raw sample set is committed, figures regenerate
  from JSONs.
- Thread counts nobody can reach → a sweep row above the participants the runtime can
  actually field is not a scaling claim. `HELIX_NTHREADS` caps participants and
  hardware caps them further, so the widest labelled row is
  `min(HELIX_NTHREADS, available_parallelism)`.
- A different pipeline per consumer → every executing path (CLI JIT, campaign,
  selftest) runs the one shared sequence in `helix-artifact`
  (`parse → check → IR → SSA → optimizer → analysis → plan`), so a verdict the
  Observatory shows is the verdict the JIT executed.
