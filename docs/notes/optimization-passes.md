# The Optimization Passes (course notes)

*Six classic scalar optimizations, what each proves, and how HELIX keeps them honest.*

All passes run on SSA-form IR, each followed by the verifier (dominance, φ arity,
dangling uses). Every pass reports a `ChangeFlag` so the driver can snapshot
before/after text for the Observatory's OPT view.

## 1. Constant folding

Evaluate at compile time what only involves constants: `3 * 4 → 12`, `-7 % 2 → -1`
(sign follows dividend — srem semantics), saturating casts (`300.7 as i32 → 300`,
`NaN as i64 → 0`). The folder must use **exactly** the language semantics — it shares the
same arithmetic helpers as the interpreter and JIT backend. A folder that disagrees with
the hardware is a miscompiler.

## 2. Constant propagation (+ branch folding + CFG cleanup)

Replace uses of values whose unique reaching def is a constant with that constant,
re-enabling folding. On SSA this is a walk of def→use chains — no dataflow equations
needed (this is SSA paying rent). Consequences cascade:
`if true {A} else {B}` folds to `jump A`; empty blocks get merged; unreachable blocks
vanish. This trio is where most visible shrinking happens on demo programs.

## 3. Copy propagation

After SSA renaming, `x1 = y0; use x1` becomes `use y0` when y0 dominates the use.
Reduces name chains left by source-level staging variables and by other passes.

## 4. Dead code elimination

Mark side-effecting instructions (stores, calls, branches, returns) as roots; transitively
mark their operand defs; sweep everything unmarked. On SSA, DCE *is* reachability on the
def-use graph — no kill/gen sets. Dead phis are swept too (a φ nobody reads is dead even
though syntactically "used" by nothing).

## 5. Common subexpression elimination

Dominator-scoped value numbering: walking the dominator tree, hash pure ops by
(opcode, operands); a hit in a dominating scope reuses the earlier value. Dominator scoping
is what makes it correct with control flow — an expression computed in a dominating block
must have executed before any dominated use.

## 6. Loop-invariant code motion

Hoist loop-invariant *pure* computations to the preheader: operands defined outside the
loop (or already hoisted), instruction not a memory op (loads/stores/calls never hoist —
that would change memory behavior), not a header φ (those encode the loop-carried values).
Safety requires dominance reasoning from the same domtree SSA construction built earlier.

## What HELIX deliberately does NOT do (and why that's fine)

- **No vectorization**: Cranelift may auto-use SIMD within basic blocks, but loop-level
  vectorization is out of scope (multicore is the headline, SIMD would be a second one).
- **No e-graphs/equality saturation**: elegant, but orthogonal to dependence analysis and
  a project-eater. Cited as related work.
- **No alias analysis**: the language forbids the aliasing that would require it
  (`f(a,a)` rejected statically; arrays can't be value-copied). A design decision that
  deletes a whole compiler subsystem — worth a report paragraph.

## Measuring the passes

The Observatory shows insts_before/insts_after per pass plus full IR text after each —
optimization becomes something you can *see*. The benchmark tier "interpreter vs native
parallel" quantifies the pipeline's end-to-end runtime value (the passes run on the
JIT's input IR; there is deliberately no separate optimized/unoptimized native pair).
