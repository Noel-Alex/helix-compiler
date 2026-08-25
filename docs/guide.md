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

Both backends must print **byte-identical** output — `helix selftest` proves
this across every example.

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

* **SAFE** — iterations are independent; the JIT runs them on all cores.
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

## 6. Explore visually

```bash
helix observe            # opens http://127.0.0.1:8931
```

The Observatory walks any program through every compiler stage — tokens, AST,
control-flow graph, SSA, optimization passes, loop verdicts, benchmarks — in
the browser, with your own source editable in the left rail.

## 7. Benchmark honestly

```bash
helix bench --quick      # minutes: small sizes
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
