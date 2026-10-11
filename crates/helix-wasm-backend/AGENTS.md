# DOX — helix-wasm-backend

## Purpose

`helix-wasm-backend` is the WebAssembly emitter for HELIX. It consumes
post-SSA, optimized IR (`helix_ir::build` → `to_ssa` →
`run_optimization_pipeline`) and produces one real WebAssembly module — the
binary format is written directly — so a browser can run the program with no
host runtime of its own.

## Ownership

- This crate owns the emitted module ABI, the binary encoder (`encode.rs`),
  the trap-code table, and `EmitError`.
- `helix-wasm` owns the browser entry (`helix_apply`) and calls `emit_wasm`;
  it does not re-implement emission.
- Compiler behaviour stays in its owning crates: this crate lowers `helix-ir`
  IR and formats through `helix_sema`, and changes there flow in as inputs.

## Local Contracts

- Imports, module `env`: `hx_print_i32` `(i32) -> ()`, `hx_print_i64`
  `(i64) -> ()`, `hx_print_f32` `(f32) -> ()`, `hx_print_f64` `(f64) -> ()`,
  `hx_print_bool` `(i32) -> ()`, and `hx_trap` `(i32 code, i64 a, i64 b) -> ()`.
- Exports: `memory` (32 pages minimum), `hx_run` `() -> ()` (resets state,
  then runs `fn main()`), `hx_fold` `(i32 ptr, i32 len) -> ()`, `hx_checksum`
  `() -> i64`, `hx_heap_bytes` `() -> i32`. The host supplies the print
  scratch; the module keeps arrays and state in its own memory.
- Trap codes: 0 division by zero, 1 division overflow (`i64::MIN / -1`),
  2 bounds violation (`a` = length, `b` = index), 3 negative `zeros(n)`,
  4 call depth exceeded, 5 allocation failure. The host renders the
  interpreter's message for the code it is handed.
- Linear memory: print scratch `[0, 4096)`, array fold table
  `[4096, 8192)` (256 entries × 8 bytes: element base, then byte length),
  bump heap above both. More final arrays than the table holds is refused at
  emission time, never silently folded.
- The checksum is the interpreter's FNV-1a stream: every printed line plus
  its newline (folded by `hx_fold` from the host's canonical formatting),
  then every array still bound in `main`'s root scope in ascending symbol
  order, element by element, as little-endian bytes (folded by
  `hx_checksum`). A browser run therefore must produce the interpreter's
  number.
- Structured-emission fallback: anything the emitter cannot prove —
  unknown callee, missing type, malformed phi edge, over-nested dispatch
  (`MAX_DISPATCH_BLOCKS`) — is an honest `EmitError`, never an emitted module
  whose behaviour could differ from the interpreter and never invalid wasm.

## Work Guidance

- Match interpreter semantics exactly, including trap payloads and the
  order lines and arrays enter the checksum.
- The parity tests in `tests/parity.rs` run every repository example under
  the dev-only `wasmi` runtime with the browser's host contract; they must
  stay green, and a failure there is a compiler bug.
- Emit only from post-SSA optimized IR, the same input the pipeline hands
  over; do not compensate for an upstream stage's work here.
- Keep the hand-rolled encoder: the shipped asset is the whole surface the
  browser sees, and the dependency list stays `helix-ir` + `helix-sema`
  (`wasmi` is a dev-dependency only).
- Keep comment style: doc comments on public items, prose that explains
  *why*, no changelog entries.

## Verification

- `cargo test -p helix-wasm-backend`
- `cargo clippy -p helix-wasm-backend --all-targets -- -D warnings`
- `cargo check -p helix-wasm-backend --target wasm32-unknown-unknown`

## Child DOX Index

- None.
