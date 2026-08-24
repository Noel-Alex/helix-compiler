# Loop-Carried Dependence Theory (course notes)

*The theory behind HELIX's parallelization decisions, with worked examples from `examples/`.*

## What a dependence is

Two memory accesses in a loop form a **dependence** if they touch the same memory location
and at least one is a write. The kinds:

| Kind | Name | Earlier → Later | Blocks parallelization? |
|---|---|---|---|
| RAW | flow / true | write → read | YES — the reader needs the value |
| WAR | anti | read → write | Only for storage reuse (fixable by renaming) |
| WAW | output | write → write | Only for storage reuse |
| RAR | input | read → read | NEVER |

HELIX's SSA form eliminates **all scalar** WAR/WAW dependences by construction — renaming
is what SSA *is*. Only array-level false dependences could ever force serialization, and
the language forbids the aliasing that would hide them (`f(a, a)` is rejected statically).

## Distance and direction vectors

For a loop nest with induction variables (i, j, ...), a dependence's **distance vector**
records the iteration difference (sink − source) at each level. Example
([recurrence_reject.hx](../../examples/recurrence_reject.hx)):

```helix
for i in 1..n { a[i] = a[i - 1] + 1; }
```

The write `a[i]` in iteration i feeds the read `a[i-1]` in iteration i+1: a **RAW with
distance ⟨1⟩**. Iteration i+1 *cannot start* before iteration i finishes → strictly serial.

A **direction vector** abstracts distances to symbols: `<` (d<0), `=` (d=0), `>` (d>0), `*`
(unknown). The **level** of a dependence is the position of its first non-`=` component.

**The parallelization theorem** (Allen & Kennedy ch.4): loop level k can execute in parallel
(“DOALL”) iff no dependence is *carried at level k* — i.e. no dependence whose first
non-`=` component is at position k. In [stencil_2d_reject.hx](../../examples/stencil_2d_reject.hx),
the RAW on `a` is carried at level 1 (outer i), so the **inner** j loop is still parallel.

## The dependence test battery

For each pair of accesses to the same array, per subscript dimension, HELIX runs tests
cheapest-first. Any test that *succeeds* proves independence (kills the dependence);
failures refine what we know. All arithmetic is i128 — the analyzer must not overflow
even when the program's i64 math does (LLVM widens for the same reason).

### 1. ZIV — zero index variables
`a[3] = a[5]` — constants both sides: dependence iff 3 == 5. Exact.

### 2. Strong SIV — same coefficient
Subscripts `a*i + p` and `a*i' + q`: the equation `a*i' + q = a*i + p` has the integer
solution `i' − i = d = (q − p)/a` **iff a divides (q−p)**. Non-divisible ⇒ **independent,
proven**. Divisible ⇒ dependence at exactly distance d (if a feasible iteration pair
exists within the trip count). This is the test that catches `a[i]` vs `a[i-1]`
(d = (0−(−1))/1 = 1) and proves `a[2i+1]` never meets `a[2i]`.

### 3. Weak-Zero SIV — one side constant
`a[i] = a[7]` touches one location: the single point `i = (7 − p)/a`. Integral and inside
the trip range ⇒ dependence exists; otherwise ⇒ independent. Exact.

### 4. Weak-Crossing SIV — opposite coefficients
`a[i]` vs `a[c − i]`: the two subscripts meet at the crossing point `i = (c − p)/(2a)`.
Integral and in-range ⇒ both `<` and `>` directions possible (reversal); out of range ⇒
independent. Exact.

### 5. GCD test + bounded box — general two-variable case
For `a[2i]` vs `a[i]` ([gcd_box_test.hx](../../examples/gcd_box_test.hx)): solve
`2i − j = 0`. The Diophantine equation `a·i − b·j = k` has integer solutions iff
`gcd(a,b) | k` — the **GCD test**. It's only a *necessary* condition (ignores bounds), so
HELIX then intersects the parametric solution family `i = i0 + (b/g)t, j = j0 + (a/g)t`
with the iteration box `[lo,hi]²`. Empty box ⇒ independent. Here the box is non-empty
(e.g. i=j=2), so a dependence exists and the loop stays serial — exactly the case where
the GCD test alone would have been inconclusive.

### 6. Banerjee (inequality) test
Bounds each subscript *difference* over the iteration ranges; independence for a direction
is proven iff 0 lies outside the achievable interval. HELIX implements the exact closed
forms above for its subscript grammar; the Banerjee fallback covers symbolic-bound cases
conservatively. (Production compilers stop here too — LLVM's DependenceAnalysis is
explicitly an incomplete implementation of the same Goff/Kennedy/Tseng scheme.)

## Reductions: the sanctioned self-dependence

```helix
let dot = 0.0;
for i in 0..n { dot = dot + a[i] * b[i]; }
```

This *looks* like a distance-1 RAW on `dot`. But `+` is associative, so the accumulation
can be split: each thread sums a private partial, and partials combine at the end
([dot_reduction.hx](../../examples/dot_reduction.hx)). HELIX recognizes the shape
`x = x op t` (or `t op x`) for op ∈ {+,−,*,min,max} when x is written exactly once per
iteration and referenced nowhere else. Private accumulators start at the monoid identity
(0, 0, 1, +MAX, −MIN).

**Honesty clause**: floating-point `+`/`*` are *not* associative — parallel combination
order changes rounding. HELIX documents this (OpenMP does the same: “order of combination
is unspecified”) and the test suite compares FP reductions with a relative tolerance,
while integer/min/max reductions must match bit-for-bit.

## Why this is exact for HELIX's grammar

HELIX's subscripts are affine in at most one index per dimension (`i`, `i+c`, `c−i`,
`c·i`, and 2D `i,j`), bounds are integers, and arrays are flat. Every test above is
**exact** on this grammar — a stronger claim than production compilers can usually make,
which is worth stating in the report.

## References

- Allen & Kennedy, *Optimizing Compilers for Modern Architectures*, ch.2 (dependence
  vectors/levels), ch.4 (parallel loop transformations), ch.8 (dependence testing).
- Goff, Kennedy & Tseng, “Practical Dependence Testing”, PLDI 1991 (the SIV battery).
- Cooper & Torczon, *Engineering a Compiler*, ch.14 (2nd ed.) — loop-carried dependence.
- LLVM `DependenceAnalysis.cpp` — production implementation of the same scheme.
