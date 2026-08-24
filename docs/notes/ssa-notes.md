# SSA in HELIX (course notes)

*Why Static Single Assignment, how HELIX builds it, and why it never leaves it.*

## The problem SSA solves

Before SSA, `x` is one name that means different storage cells at different times. Dataflow
questions — “which write does this read see?” — require reaching-definitions analysis.
After SSA ([Cytron et al. 1991]), every *value* gets exactly one definition site:

```
x0 = 5
if c
  x1 = 10        (then branch)
  x2 = φ(x0, x1) (join: picks whichever flowed in)
print x2
```

The φ-function is not a machine operation; it's a *notation* for “the value depends on which
predecessor edge we arrived on”. Optimizations become trivial to reason about: a use points
at its unique def; DCE is graph reachability.

See [examples/ssa_demo.hx](../../examples/ssa_demo.hx) — HELIX prints exactly this shape.

## HELIX's construction pipeline

1. **Reachability** — forward DFS from entry; unreachable blocks are stripped before
   anything else (they corrupt dominators and φ placement).
2. **Dominators** — Cooper-Harvey-Kennedy's iterative algorithm: compute reverse-postorder,
   initialize `idom(entry) = entry`, then iterate over RPO picking the intersect of processed
   predecessors until fixpoint. ~80 lines, near-linear in practice, competitive with the
   classic Lengauer-Tarjan at course scale. The same idom tree feeds dominance-frontier
   computation, CSE scoping, LICM safety checks, and the verifier — one module, many users.
3. **Dominance frontiers** — the runner-walk rule: for every join b with ≥2 preds, run each
   pred p up the idom chain until `idom[b]`, inserting b into DF(runner). DF(X) = the set of
   joins where X's values *meet* other paths — precisely where φs are needed.
4. **φ placement (semi-pruned)** — full minimal SSA puts φs for *every* variable at every
   iterated-DF point; pruned adds liveness to skip dead ones. Semi-pruned (Briggs et al.)
   is the sweet spot: one linear pass finds "global" names (those whose value escapes their
   defining block via upward-exposed uses); only globals get iterated-DF φ placement.
   Local temporaries never cross blocks → zero useless phis at near-zero cost.
5. **Renaming** — preorder walk of the dominator tree carrying per-variable stacks of SSA
   names. Push on defs, pop on leaving, fill φ arguments when visiting successors.

## Arrays stay OUT of SSA (deliberately)

Scalars get SSA names; array elements remain explicit Load/Store ops on a fat pointer.
This is the LLVM/GCC model (mem2reg promotes only scalar allocas) and it's load-bearing:
- Array element *values* are unbounded and dynamic — naming them individually is impossible.
- Affine dependence analysis reads address arithmetic directly off the Load/Store indices,
  which is exactly what it wants.
- Memory effects keep CSE/DCE honest by default (loads/stores/calls are side-effecting).

## Why HELIX never destroys its SSA (no out-of-SSA translation)

Classic compilers translate out of SSA into parallel copies before register allocation, and
must solve the lost-copy problem and swap cycles (Sreedhar et al., Budimlić et al.) — real
research-grade machinery. **HELIX skips all of it**: Cranelift IR *is* SSA with block
parameters — semantically φs whose arguments are supplied per-edge by each predecessor's
terminator (`jump bb3(x1)` / `jump bb3(x0)`). HELIX's φ lowers 1:1:

| HELIX IR | Cranelift |
|---|---|
| `bb3: x2 = φ(bb1: x1, bb2: x0)` | block `bb3` with param x2 |
| terminator of bb1 | `jump bb3(x1)` |
| terminator of bb2 | `jump bb3(x0)` |

Move coalescing across block params becomes Cranelift's regalloc2's job — where it's
already solved well.

## Verifying SSA after every pass

A debug verifier runs after every optimization pass and checks:
- every use is dominated by its def (block-level + intra-block order),
- every φ has exactly one argument per predecessor, listed once, type-matched,
- no dangling references after CFG edits (the #1 source of silent corruption in hand-written
  SSA compilers: deleting an edge without deleting its φ argument).

## References

- Cytron, Ferrante, Rosen, Wegman, Zadeck, “Efficiently Computing Static Single Assignment
  Form and the Control Dependence Graph”, TOPLAS 1991.
- Briggs, Cooper, Harvey, Simpson, “Practical Improvements to the Construction and
  Destruction of Static Single Assignment Form”, SP&E 1998 (semi-pruned SSA).
- Cooper, Harvey, Kennedy, “A Simple, Fast Dominance Algorithm” (CHK).
- Braun et al., “Simple and Efficient Construction of SSA Form”, CC 2013 (what
  cranelift-frontend does instead — cited as related work).
