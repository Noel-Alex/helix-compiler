# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Primary users are compiler-course learners, reviewers, and systems engineers who need to inspect how a numerical program is compiled and why a loop is or is not safe to parallelize. This audience is inferred from the project's README, guide, and Observatory documentation.

## Product Purpose

HELIX Lite accepts ordinary sequential numerical code, compiles it through an inspectable pipeline, and exposes the proof behind parallelization decisions. Success means a visitor can move from source to dependence evidence to a measured result without treating a green badge as a black box.

## Positioning

HELIX is an automatic parallelizing compiler that shows the proof: it can parallelize a proven-independent loop and refuses a carried dependence with the exact dependence evidence behind the refusal.

## Operating Context

- The primary workflow begins in the local `helix` CLI and opens the local Observatory web surface; the same surface also deploys to GitHub Pages as a static site that compiles and runs HELIX entirely in the browser with zero server.
- Visitors examine source, tokens, AST, CFG, SSA, optimization passes, loop analysis, and benchmark output.
- Examples include deliberately accepted and deliberately rejected loops; benchmark claims come from committed artifacts and methodology documents.
- The web surface must work without a remote CDN or npm runtime dependency during a local demonstration.

## Capabilities and Constraints

- The Observatory renders compiler artifacts and server-provided graph/layout data, including multiple functions where present.
- Source diagnostics, token-to-source highlighting, graph panning/zooming, SSA comparison, optimization details, loop dependence evidence, and benchmark views are real behaviors to preserve.
- Three independent decisions stay separately labelled: the analysis verdict (SAFE/REDUCTION/SEQUENTIAL with its proof), whether the backend can lower the approved region, and whether the runtime finds the loop large enough to thread. Only the Benchmark phase reports measured participants and speedups.
- The implementation is a Rust workspace with a vanilla web client and local vendored assets; the redesign must not change compiler semantics or artifact contracts.
- Parallelization verdicts must remain understandable without relying on color alone.

## Brand Commitments

- The product name is HELIX Lite.
- The interactive visualizer is called the Observatory.
- The product promise is “shows you the proof,” not merely “makes code faster.”

## Evidence on Hand

- Compiler examples under `examples/`.
- Artifact fixtures under `web/dev-fixtures/`.
- Pipeline and benchmark documentation under `docs/`.
- The UI is allowed to label illustrative or fixture values as such; it must not invent benchmark results.

## Product Principles

- Proof before performance theatre.
- Refusal is a useful, inspectable outcome.
- Every compiler stage should be traceable to the next.
- Offline demonstrations should be as complete as connected ones.
- Measurements and claims must retain their source context.
