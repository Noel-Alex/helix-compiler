# ADR-0001: Pin all cranelift-* crates to =0.135.0

Date: 2026-08-23 · Status: accepted · Verified by M0 spike

## Context

wasmtime/Cranelift ships multiple release trains concurrently (0.111.x, 0.123.x,
0.133.x, 0.134.x, 0.135.0 all live on crates.io in Aug 2026). The family shares
`ir::*` types; mixed minors fail with confusing type errors.

## Decision

Pin every member (`cranelift`, `cranelift-codegen`, `cranelift-frontend`,
`cranelift-module`, `cranelift-jit`, `cranelift-native`) to exactly `=0.135.0`.

## Consequences

- Reproducible builds all semester; documented 2026 API quirks stay true.
- Upgrades are deliberate, never accidental.
