# HELIX Demo Script (10-minute presentation flow)

## Setup (before audience)

```bash
cargo build --release -p helix-cli
```

Two terminals:
1. `target/release/helix observe --port 8931` — Observatory
2. spare terminal for CLI demos

Browser on the Observatory. Have [examples/](../examples/) open in an editor.

## Beat 1 — The hook (2 min)

> *"I wrote a compiler that reads ordinary sequential code and decides — by proving
> theorems about arrays — which loops can run on all 32 of these cores."*

In the Observatory, open **scale**. Walk the pipeline chips left→right:

- **SOURCE**: trivial loop, nothing special.
- **TOKENS/AST**: "the compiler builds a tree…"
- **CFG**: basic blocks; point at the back edge (amber curve) — "that's the loop".
- **SSA**: φ-functions at joins.
- **LOOP ANALYSIS**: green card — `✓ PARALLELIZED × N THREADS`.

## Beat 2 — The rejection (the money shot, 3 min)

Open **recurrence_reject**:

```
for i in 1..n { a[i] = a[i - 1] + 1; }
```

LOOP ANALYSIS now shows the red dashed card:

> ✗ SEQUENTIAL — RAW a[i] ← a[i−1] (distance 1)

> *"Iteration i+1 needs the value iteration i just wrote. No amount of cores helps —
> this is a data dependence, and the compiler PROVED it. This is the same analysis
> production compilers do — GCD test, Banerjee inequalities, SIV subscripts."*

Show `gcd_box_test` too: "here the cheap tests are inconclusive, so my analyzer solves
the Diophantine equation and intersects solution family with the trip-count box."

## Beat 3 — Reductions (2 min)

Open **dot_reduction**: blue card, `Σ+ REDUCTION`.

> *"`dot += a[i]*b[i]` looks sequential but isn't: plus is associative, so each thread
> sums a private partial and we combine at the end. My compiler recognizes this shape
> and lowers it to per-thread accumulators — false-sharing-padded to 128 bytes."*

## Beat 4 — Proof of correctness + speed (2 min)

Terminal:

```bash
helix selftest
```

> *"Every example runs through BOTH the interpreter I wrote as a reference and the JIT
> — outputs must match bit-for-bit."*

Then run the campaign numbers (BENCH phase in the UI): interpreter vs native vs parallel,
with efficiency columns and the honest small-N case where threading loses.

## Beat 5 — Architecture tour for questions (1 min)

- `docs/research/` — verified Aug-2026 research digests (Cranelift APIs, dependence theory)
- `docs/decisions/` — every design decision recorded
- `docs/notes/` — course notes per topic (SSA, dependence theory, passes, runtime)
- 393+ tests across 10 crates

## Anticipated questions

- **"Why not LLVM?"** — Cranelift gives fast JIT compilation (ms not seconds), tiny deps,
  block params = our phis 1:1. LLVM would dominate codegen effort over analysis effort.
- **"Is this real SSA?"** — semi-pruned SSA (Briggs et al.) built via CHK dominators +
  iterated dominance frontiers; verified after every pass.
- **"What can't it parallelize?"** — flattened 2D stencil subscripts are non-affine in one
  level (honestly reported); arbitrary pointers don't exist in HELIX by design.
- **"FP reductions deterministic?"** — no, documented OpenMP-style; integers/min/max exact.
