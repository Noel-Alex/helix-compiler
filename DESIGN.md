# HELIX Observatory — TempleOS compiler poster

## Design thesis

HELIX is a purpose-built 16-colour compiler computer. It should feel like a machine an engineer boots to inspect the compiler's proof, not a web dashboard decorated with terminal colors. Source selection, the pipeline keyboard, and the active compiler artifact are fixed machine regions with explicit jobs.

## Visual world

- Ground: bright cyan poster field (`#58F4EF`) with a dark ink chassis (`#13242B`).
- Active screen: warm paper (`#FFF9DF`) with yellow registration (`#FFE45B`).
- Evidence: green for accepted, ochre for measurement, coral for refusal or diagnostics.
- Type: bundled Silkscreen 400/700 for machine UI and code. The font files live in `web/fonts/` and keep the static surface offline-capable.
- Material: stepped borders, inset chassis rules, restrained scanlines, square controls, yellow bitmap lettering, and the authored sword-and-temple mark.

No soft cards, glass, serif editorial styling, or rounded SaaS chrome belongs in this system.

## Composition and hierarchy

The machine is organized into four regions:

1. System bar: HELIX identity, backend/checksum state, recompile, and export.
2. Source bay: bundled examples, custom source buffer, and the primary run action.
3. Pipeline keyboard: source → tokens → AST → CFG → SSA → optimization → loop analysis → benchmark.
4. Artifact display: one large phase surface for code, graphs, proof details, and benchmark evidence.

The pipeline keyboard is always the hinge between input and proof. The selected phase inverts to cyan and gains a yellow registration edge; completed phases remain visibly available. Paper surfaces and larger body type keep dense compiler evidence readable in a classroom demonstration.

## Signature interaction

A phase change performs a short, stepped screen-page wipe. It is deliberately quantized rather than eased, as if the machine is swapping a display page. Reduced-motion mode removes the wipe without hiding state.

The compiler guide at `/about` uses the same cyan/cream/yellow language as a readable poster, but gives the explanatory copy its own paper columns. Its hero plate is a contained original pixel illustration of the observatory, dependency nodes, and loop arrow; it never crosses the headline or body copy. The guide also includes a bounded browser lab: the real syntax, semantic checker, and reference interpreter are compiled to `wasm32`, and the optimized SSA IR is emitted to a genuine WebAssembly module (`helix-wasm-backend`) executed in a worker beside the interpreter with printed-line and checksum parity. What remains in the Observatory and is not misrepresented as browser execution is the Cranelift native JIT and the multicore parallel runtime.

## Components and states

- Example selection uses a yellow cursor block, group labels, and bitmap registration—not a decorative side stripe. All 19 repository examples are exposed in Kernels, Language/IR, and Refusal/diagnostics groups.
- Example selection is request-ordered: a rapid tour cannot let a slower earlier response repaint the currently selected example. The Rust observer is the canonical artifact source; the Node development port proxies to it when available, and its offline fallback shows each real example source with an explicit source-only fixture label rather than cloning one compiled sample.
- Benchmark evidence is provenance-bearing. The observer attaches the matching point from the committed `helix-campaign-v1` JSON to the nine measured kernel names, while the `/api/benchmarks` index exposes its date, commit, machine, sizes, point count, and the eight dated interpreter probes for bundled language/IR examples. The remaining refusal/diagnostic examples explain why numeric execution is unavailable rather than showing an empty chart.
- Every valid short bundled example can show its recorded `interp-local` value. Custom source can request a bounded seven-sample `interp-wasm-local` measurement from the browser's shipped WASM bridge; it deliberately reports no native, parallel, speedup, or efficiency value: those claims belong to the named kernel campaign.
- Compiler code and data tables use the same bundled pixel voice and explicit syntax colors.
- Safe, sequential, diagnostic, unavailable, and compiling states always retain words and redundant marks.
- Graphs, SSA panes, pass lists, loop verdicts, and benchmark plots inherit the machine palette but keep their existing behavior.
- Focus is a three-pixel coral outline. Selection is yellow on ink. Scrollbars are square aqua machine controls. Fixture responses are labelled in the header; the Rust observer remains canonical.

## Responsive behavior

At tablet widths, the source bay becomes a two-column region above the artifact display. At phone widths it becomes one column; the pipeline keyboard remains horizontally scrollable and the artifact remains full-size rather than shrinking code into illegibility. Integer-like spacing, hard borders, and state contrast survive every width.

## Product constraints

HELIX must continue to show why a loop was parallelized or refused. It must never invent speedups, dependencies, diagnostics, timing, or completion state. A plan width in the loop panel is a proposal from the analysis, never a dispatch count: the backend may demote a region it cannot lower and the runtime cost-gates small loops, so the label must say so and the Benchmark phase must be the only surface that states measured participants. The custom source buffer still posts to `/api/run`, and its optional local WASM measurement is explicitly interpreter-only; example artifacts, execution output/checksums, graph controls, export, keyboard behavior, campaign provenance, and offline operation remain functional.

## Direction record

Impeccable replacement-world seed `49d1dabc`, bolder reroll. The chosen direction is a professional TempleOS-era poster homage: bright cyan field, yellow/black bitmap lettering, a sword-and-temple compiler mark, readable paper work surfaces, and stepped page changes without parody language or theological framing.
