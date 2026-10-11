# DOX — helix-wasm

## Purpose

`helix-wasm` is the raw-WASM bridge the browser lab links. It ships the real
syntax/sema/engine path to `wasm32-unknown-unknown` with no generated binding
runtime, and it also exposes the rest of the compiler pipeline (artifact,
WebAssembly emission, print formatting) through one JSON entry so the browser
can reach more than the interpreter.

## Ownership

- This crate owns the raw ABI (`helix_alloc`, `helix_free`, `helix_run`,
  `helix_apply`, `helix_output_ptr`, `helix_output_len`), the shared result
  buffer, and the request/response shapes of the `helix_apply` op table.
- The worker/browser scripts under `web/wasm/` are the only callers; the
  Observatory serves the built asset, it does not link this crate.
- Compiler behaviour stays in its owning crates: this crate only wires stages
  together (`helix-artifact` for artifacts and bench evidence,
  `helix-wasm-backend` for emission, `helix-sema` for print formatters).

## Local Contracts

- `web/wasm/helix-lab.wasm` is the shipped asset, produced by
  `cargo build --release --target wasm32-unknown-unknown -p helix-wasm` from
  `lib.rs`'s exports; the Observatory route `/wasm/helix-lab.wasm` serves it.
- `helix_run`'s result JSON is a frozen wire shape (the node tests in
  `web/` assert on it): `ok`, `status`, `printed`, `checksum`, `message`,
  `span`, `engine`. Changes there need the JS consumers updated in the same
  change.
- `helix_apply(ptr, len) -> i32` answers one JSON op
  (`build`/`emit`/`format_line`/`bench_index`/`interp_run`; see `apply.rs` for
  the table) into the shared result buffer. A readable answer — including an
  embedded `"ok":false` payload — returns 1; only an unreadable request
  (empty, invalid UTF-8, oversized) returns 0.
- Source inputs stay capped at `MAX_SOURCE_BYTES`, matching
  `helix-artifact`'s cap; `elapsed_ms` is 0 because wasm32 has no clock and
  the worker measures wall-clock itself.

## Work Guidance

- Keep comment style: doc comments on public items, prose that explains
  *why*, no changelog entries.
- No new dependencies: hand-rolled base64 lives in `b64.rs`, and requests
  parse with the workspace serde/serde_json.
- Prefer adding an op over adding an export; every entry point is frozen
  surface for the shipped worker.

## Verification

- `cargo test -p helix-wasm`
- `cargo clippy -p helix-wasm --all-targets -- -D warnings`
- `cargo check -p helix-wasm --target wasm32-unknown-unknown`
- Rebuild `web/wasm/helix-lab.wasm` after ABI changes; the Observatory
  embeds it with `include_bytes!`.

## Child DOX Index

- None.
