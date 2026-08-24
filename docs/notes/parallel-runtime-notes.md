# The HELIX Parallel Runtime (course notes)

*How `helix_parallel_for` turns a proven-safe loop into multicore execution — and why
the naive versions are slow.*

## The contract

```text
helix_parallel_for(start, end, body_id, nthreads_hint)
```

The JITed main loop calls this instead of looping serially. `body_id` indexes a host-side
registry mapping ids → compiled body-function pointers, registered **after** Cranelift's
`finalize_definitions()` (code pointers don't exist until then — embedding them earlier is
impossible). Each worker invokes `body(iter)` for its share of iterations.

## Two stages, measured

| Stage | Mechanism | Per-region overhead | Why it exists |
|---|---|---|---|
| A | `std::thread::scope`, spawn per region | ~100 µs | trivially correct; makes fork/join cost *measurable* |
| B | persistent pool: workers spin briefly then park on a generation counter | ~2 µs | the real runtime |

The Stage-A → Stage-B overhead delta is one of the project's most instructive graphs.

## Scheduling: who gets which iterations?

- **static** — equal contiguous chunks via the libgomp formula: with n iterations and P
  threads, q = n/P, the first n%P threads get q+1. Zero coordination cost. Best when every
  iteration costs the same (our streaming kernels). Chunk boundaries are snapped to 64-byte
  element multiples so adjacent threads never straddle a cache line in the output array.
- **dynamic** — one padded atomic counter; each worker does `fetch_add(chunk)` to claim its
  next block, `chunk = max(min_chunk, remaining/P)`. ~15 lines, mirrors libgomp. Best when
  iteration cost varies (e.g. sieve inner loops).
- **guided** — same counter but chunk shrinks as remaining shrinks: self-tuning tail.
- **cost gate** — below `max(K_MIN, GRAIN·P)` iterations (~GRAIN=1024), run serially:
  fork/join overhead would dominate anyway. This is why [small_n.hx](../../examples/small_n.hx)
  honestly *loses* nothing by staying serial — and the benchmark suite includes such cases.

Env knobs for lab experiments: `HELIX_NTHREADS`, `HELIX_SCHEDULE=static|dynamic|guided`,
`HELIX_RUNTIME=scope|pool`.

## False sharing: the invisible tax

Cache lines are 64 bytes. If two threads write *different* variables that share a line,
the line ping-pongs between cores at full cache-coherence traffic — each "share" costs
hundreds of cycles. Classic case: per-thread reduction accumulators packed side by side.

HELIX pads every per-thread accumulator into its own `#[repr(align(128))]` cell (128 to be
safe across prefetcher pairs). Measured impact in the literature: 1.5–3× slowdowns from
unpadded counters on tight loops — the benchmark suite reproduces this as a lab exercise.

## Reductions

Recognized reductions lower to:
1. P aligned partial cells initialized to the monoid identity (+ : 0, × : 1,
   min : +MAX/+inf, max : −MIN/−inf),
2. each thread accumulates into its own cell (pure local writes — no atomics in the hot
   loop; atomic RMW would serialize the very loop we just parallelized),
3. after join, the coordinator folds P partials serially (O(P) ≤ 32 ops, unmeasurable).

Float +/× combination order is unspecified (documented, OpenMP-style); integer/min/max
are exact.

## What speedup should you *expect*? (honesty section)

- **Memory-bound** kernels (saxpy touches 24 MB/iteration-set at N=2²⁴ f64): bounded by
  DRAM bandwidth. One core gets ~X GB/s; all cores together top out near the machine's
  STREAM-triad ceiling (~1.5–2× single-core read+write throughput). Expect **4–6× at
  8–16 threads**, not 16×.
- **Compute-bound** kernels (register-resident arithmetic): scale with core count until
  power/thermal limits bite: expect **12–15×** on a 16-core/32-thread desktop.
- **Amdahl's law**: total speedup ≤ 1/f_serial where f_serial = setup, final combine,
  barriers. HELIX measures and reports this honestly, including cases where threading
  *loses* (small N).
- Hybrid CPUs (P/E cores): E-core stragglers cap speedup at the slowest thread;
  HELIX_NTHREADS sweeps make this visible.

## Windows specifics

`std::thread::park/unpark` semantics (spurious wakeups are legal → re-check state);
workers flip-the-generation-then-unpark ordering to avoid lost wakeups; the pool warms
itself with one dummy region before any timing (first region pays thread creation).
