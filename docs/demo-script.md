# HELIX Live-Demo Script

*A 10-minute presentation flow, ordered for maximum impact. Rehearse once with real
numbers; fill the blanks from the latest benchmark campaign.*

## Setup (before audience)

```bash
cargo build --release
# Terminal 1: Observatory
./target/release/helix observe --port 8931
# Browser: http://127.0.0.1:8931  (works offline)
```

## Beat 1 — The hook (90 s)

Open **scale.hx** in the Observatory SOURCE phase:

```helix
for i in 0..n { out[i] = a[i] * 5.0; }
```

> "Ordinary sequential code. Watch what the compiler does with it."

Click compile → pipeline animation runs → land on **LOOP ANALYSIS**:
green card, `✓ PARALLELIZED × N THREADS`, accesses listed, `RAW 0 / WAR 0 / WAW 0`.

## Beat 2 — The rejection (90 s) ← the money shot

Switch to **recurrence_reject.hx**:

```helix
for i in 1..n { a[i] = a[i - 1] + 1; }
```

Red dashed card, hazard stripes: `✗ REJECTED — RAW a[i] ← a[i-1], distance 1`.

> "Iteration 42 writes what iteration 43 reads. There is no way around the physics —
> this loop is *provably* serial. The compiler didn't guess; it solved the equation
> `i' − i = 1` and checked feasibility."

## Beat 3 — The proof machinery (2 min)

Walk the stepper backwards: **SSA** phase — point at φ nodes.
**CFG** phase — show the loop header/latch/exit and the back-edge arc.
**OPT** phase — show a pass that shrank the IR (`insts 42 → 37`).

> "Everything on screen is computed by the same compiler you're reading the source of."

## Beat 4 — Reductions (90 s)

**dot_reduction.hx**: blue card `Σ+ REDUCTION — private accumulator per thread`.
Explain: looks like distance-1 RAW on `dot`, but `+` is associative → split per thread,
combine at join. FP honesty clause in one sentence.

## Beat 5 — Numbers (2 min)

BENCH phase bars + terminal table from `helix bench`:

| tier | median |
|---|---|
| interpreter | ___ |
| native seq | ___ |
| native par ×N | ___ |

Headline: "___× vs interpreter, ___× parallel speedup at N threads — and we reach
___% of this machine's measured memory bandwidth." Show the small-N honest case
where threading loses.

## Beat 6 — Safety net (60 s)

Terminal:

```bash
helix run examples/div_guard.hx     # -7 % 2 == -1 …
helix check examples/type_errors.hx # caret diagnostics
```

> "Checked semantics everywhere: bounds, division, saturating casts — identical in the
> interpreter and the JIT because both share one spec."

## Beat 7 — Close (30 s)

> "A complete compiler — lexer to machine code — whose headline feature is proving
> which loops are safe for all your cores, and showing you the proof. 243 tests,
> every stage inspectable. Thank you."

## Q&A ammunition

- *"How do you know it's safe?"* → dependence battery is exact for our grammar;
  anything unproven stays serial (conservative by construction).
- *"What about aliasing?"* → rejected statically: `f(a,a)` won't type-check.
- *"Why Cranelift?"* → SSA block params ≈ our φs 1:1; safe Rust; JIT latency fine.
- *"FP reductions deterministic?"* → integer/min/max bit-exact; FP +/* documented
  order-unspecified (same as OpenMP).
- *"Test coverage?"* → 243 tests incl. differential interpreter-vs-JIT and adversarial
  verifier passes.
