# HELIX Lite — User Guide

Everything you need to go from *zero* to *watching your loops parallelize*, in
about five minutes.

## 1. Build it

```bash
cargo build --release -p helix-cli
```

One binary, `helix`, does everything. The rest of this page assumes it is on
your `PATH` (or substitute `cargo run --release -p helix-cli --`).

## 2. Write a program

HELIX programs look like ordinary sequential numerical code:

```rust,ignore
// saxpy.hx
fn main() {
    let n = 1000000;
    let x: [f64] = zeros(n);
    let y: [f64] = zeros(n);
    let s = 2.5;
    for i in 0..n {
        y[i] = s * x[i] + y[i];
    }
    print(y[7]);
}
```

The language: `i32/i64/f32/f64/bool`, single-level arrays, `for i in a..b`
loops, `if`/`else`, functions with by-reference array arguments, and seven
builtins (`print zeros len abs sqrt min max`). No implicit coercions; checked
bounds and division by default. Full spec: [notes/lang-spec.md](notes/lang-spec.md).

## 3. Run it

```bash
helix run saxpy.hx                 # reference interpreter (slow, always right)
helix run --backend jit saxpy.hx   # Cranelift JIT (native speed, same output)
```

Both backends print **byte-identical** output on the shipped examples —
`helix selftest` proves this by comparing printed lines. Floating-point
*reductions* are the documented exception: parallel partial sums reorder, so the
benchmark campaign compares those kernels under a relative tolerance
(ε = 1e-9) instead of bit-equality.

## 4. See the proof

```bash
helix loops saxpy.hx
```

```
==== main loop analysis ====
Loop #1: RAW 0 / WAR 0 / WAW 0 => SAFE
    READ x[i]
    READ y[i]
    WRITE y[i]
```

* **SAFE** — iterations are independent; the analysis proposes the loop for
  parallel execution (see §6 for what the backend and runtime then decide).
* **REDUCTION(+)** — an associative accumulation; private per-thread partials
  are combined after the join.
* **SEQUENTIAL** — a dependence was proven (or the body has side effects); the
  reason line tells you exactly which access pair carried it.

## 5. Control the runtime

| Control | Effect |
|---|---|
| `--threads <n>` | cap parallel loops at `n` threads |
| `HELIX_SCHEDULE=static\|dynamic\|guided` | pick the work-distribution policy |
| `HELIX_NTHREADS=<n>` | same as `--threads`, for scripts |
| `HELIX_RUNTIME=scope\|pool` | execution stage (pool = fast default) |

No knob can raise the participant count above what the compiled region asks
for, and the runtime clamps that to `min(HELIX_NTHREADS, available_parallelism)`.
A region below `max(1024, 1024 × threads)` iterations runs serially no matter
which knobs are set — the cost gate, not the analyzer, has the last word on small
loops.

## 6. Explore visually

```bash
helix observe            # opens http://127.0.0.1:8931
```

The Observatory walks any program through every compiler stage — tokens, AST,
control-flow graph, SSA, optimization passes, loop verdicts, benchmarks — in
the browser, with your own source editable in the left rail.

### Three decisions, three owners

A green verdict is an **analysis** result: the dependence battery proved the
iterations independent (or recognized a reduction). It is a proposal, not a
promise that threads will spin up. The **backend** decides whether it can lower
the approved region into a dispatched parallel region at all, and falls back to
sequential code when it cannot. The **runtime** then decides whether threading
pays: regions below the cost gate stay serial, and participants never exceed
`min(HELIX_NTHREADS, hardware)` (the emitted region's own hint caps them too).
The loop panel labels the plan width as a proposal for exactly this reason, and
the Benchmark phase is the only place that shows measured participants and
speedups — everything the analysis phase shows is about what *may* run.

### Browser lab: static Pages vs local Observatory

- **Local Observatory** (`helix observe`): the Rust server compiles every
  example, draws the pipeline, and attaches recorded benchmark evidence. This
  is the canonical artifact source.
- **GitHub Pages (static mode)**: the deployed site compiles in your browser.
  The shipped `helix-lab.wasm` bridge runs the real front end and the
  reference interpreter, and the optimized SSA IR is emitted to a WebAssembly
  module executed in a worker beside it. Every run asserts both paths print
  the same lines and fold to the same checksum (FNV-1a over the printed lines
  plus the arrays still visible at the end of `main`), and reports the module's
  heap bytes. No server is involved — the examples arrive as plain sources under
  `/data/examples/`.

Campaign cards wherever they appear still show the recorded, provenance-
stamped numbers from the host campaign; a browser run only measures what
actually ran on your machine.

## 7. Benchmark honestly

```bash
helix bench --quick      # smoke: smallest size per kernel, {1, hw} sweep, 5 samples
helix bench              # full campaign, writes JSON + figures inputs
```

Every timed kernel first passes a parity gate (all variants must produce the
oracle output at a small size) and the campaign asserts each kernel's expected
parallelization verdict before trusting its numbers.

## Troubleshooting

| Symptom | Meaning |
|---|---|
| `runtime error: index N out of bounds…` | bounds check fired — fix the index (or `--unchecked` on the JIT) |
| `=> SEQUENTIAL (...)` | not a bug: the analyzer refused the loop and says why |
| interp/JIT mismatch | please file it — this is precisely what selftest exists to catch |
