# ADR-0002: Observatory = axum + embedded vanilla JS/SVG + vendored d3

Date: 2026-08-23 · Status: accepted

## Context

The demo UI must run offline on college machines, impress on a projector, and add
no npm/node toolchain.

## Decision

- axum 0.8 server embedded in `helix observe`; ≤6 static assets via include_bytes!.
- One vendored dependency: d3.v7.min.js committed under web/vendor/.
- Syntax highlighting from HELIX's own tokens (spans → spans), not highlight.js.
- AST tidy-tree + CFG layered layout computed SERVER-SIDE in Rust; browser paints SVG.
- Dark palette #0d1117/#161b22/#30363d/#e6edf3; Okabito-derived block colors;
  rejected loops get dashed vermillion border + hazard stripes (redundant encoding).

## Consequences

- Zero network dependence at demo time.
- Layout code counts as project work (graph algorithms), not "just UI".
