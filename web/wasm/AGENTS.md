# DOX — web/wasm

## Purpose

`web/wasm/` is the browser-side execution surface of the HELIX Observatory: the
worker that runs the shipped wasm compiler, and the bridge the app renders
through. On GitHub Pages this directory is the whole backend — there is no
server, so everything here must stay honest about what a browser run can and
cannot claim.

## Ownership

- This folder owns `helix-worker.js` (worker protocol + host-contract
  implementation), `helix-browser.js` (bridge API + worker lifecycle + the
  five-second deadline), and their node tests. The app (`web/app.js`) is the
  only consumer.
- `crates/helix-wasm` owns the raw ABI the worker calls (`helix_alloc`,
  `helix_free`, `helix_run`, `helix_apply`, `helix_output_ptr`,
  `helix_output_len`); changes there ship through the rebuilt
  `helix-lab.wasm` asset in this directory.
- `crates/helix-wasm-backend` owns the emitted module ABI the worker
  instantiates (`env` imports `hx_print_*`/`hx_trap`; exports `memory`,
  `hx_run`, `hx_fold`, `hx_checksum`, `hx_heap_bytes`). The worker implements
  that host contract; neither side may drift from the other's doc.

## Local Contracts

- One fresh worker per request, terminated on completion, error, or the
  five-second deadline. Execution never falls back to the main thread.
- `runHelixInBrowser` keeps its frozen result shape (`ok`, `status`,
  `printed`, `checksum`, `message`, `span`, `engine`, `elapsedMs`,
  `alloc_bytes`); node tests assert on it.
- Print parity: a compiled module's printed lines and FNV checksum must equal
  the interpreter's for the same source. The host formats every line through
  the wasm bridge's `format_line` (canonical `helix_sema` renderers) and folds
  it through the module's `hx_fold`, so the module's `hx_checksum` is the
  interpreter's number. A mismatch is a compiler bug, not a rounding note.
- No native, Cranelift, or parallel numbers may ever be synthesized for
  browser runs: the browser compiles to WebAssembly and runs it single
  threaded. Recorded campaign evidence keeps its provenance stamps and stays
  visibly separate from local measurements.
- Static mode (`web/app.js`): when `/api/*` is unreachable the app fetches
  `data/examples/<name>.hx` relative to the site directory and compiles through `buildArtifact`; the Pages
  deploy ships that directory from the repo's `examples/`.

## Work Guidance

- Keep the worker dependency-free (own base64 decoder, `performance.now` for
  timing, `WebAssembly` directly).
- Timer discipline: `elapsedMs`/`elapsed_ms` measure bridge and execution work
  inside the worker, never worker startup or message transport; measurement
  samples exclude instantiation.
- The editor draft is newer intent than any in-flight request: preserve it
  across superseded compiles and measurements.
- The browser work filter resolves literal scalar sizes, charges allocations and loops, and refuses unknown allocation lengths. It is a UX estimate; the worker deadline remains the execution bound.

## Verification

- `node web/wasm/helix-browser.test.mjs` (worker lifecycle, ABI shapes,
  interpreter/compiled parity, trap parity, measurement sampling)
- `node web/app.test.mjs` (Observatory client state, including static mode)
- Rebuild the shipped asset after any Rust-side ABI change:
  `cargo build --release --target wasm32-unknown-unknown -p helix-wasm`, then
  copy `target/wasm32-unknown-unknown/release/helix_wasm.wasm` over
  `helix-lab.wasm`.

## Child DOX Index

- None.
