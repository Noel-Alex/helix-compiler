# DOX — helix-artifact

## Purpose

`helix-artifact` builds the `CompileArtifact` document: it runs the compiler
front to back, snapshots every stage, lays the CFG out, and attaches recorded
measurement evidence. The Observatory HTTP server and the wasm32 browser lab
both link this crate, so there is exactly one artifact pipeline. It also owns
`prepare_pipeline`, the shared compile sequence every executing consumer
(the CLI's JIT path, the benchmark campaign, the differential selftest) runs.

## Ownership

- This crate owns the artifact wire contract, the compile pipeline, the CFG
  layout, the recorded campaign/probe evidence, and the shared
  `prepare_pipeline` compile sequence.
- `helix-observe` owns the HTTP surface only: routes, static assets, and the
  standalone export. It calls this crate instead of rebuilding anything.
- `crates/helix-artifact/src/recorded-sources.json` is the frozen source
  identity of the measured catalogs; it moves only with a new recording.

## Local Contracts

- `artifact.rs` field names and serde attributes are a frozen wire schema
  (`docs/notes/artifact-schema.md` is normative). Never rename, retype, or drop
  a field; additive fields need a UI consumer on both sides.
- `prepare_pipeline` is the ONE compile sequence:
  parse → sema → IR build → to_ssa → `run_optimization_pipeline` → find_loops
  → analyze → build_plan. Consumers never re-run a stage; they run this and
  execute the IR it returns. `prepare_pipeline_checked` is the same sequence
  from the IR build onward, for callers that already hold a checked program.
- Recorded benchmark attachment requires executable-source identity against the
  frozen catalog revision. Campaign and probe provenance stay separate, and
  neither is a claim about the current build's performance.
- The crate is pure: no filesystem, network, clock, or thread APIs in the
  library path. Evidence is embedded with `include_bytes!`/`include_str!`.
  `prepare.rs` is host-free by construction (no gated calls at all);
  `pipeline.rs` still degrades its two host facilities (`plan_threads`,
  `timed_ms`) behind their wasm32 gates.
- `cargo check -p helix-artifact --target wasm32-unknown-unknown` must keep
  passing.

## Work Guidance

- Prefer moving the pipeline forward over adding configuration: one builder, a
  `BuildOpts` switch, and graceful degradation.
- Keep comment style: doc comments on public items, `file:line`-free prose that
  explains *why*, no changelog entries.

## Verification

- `cargo test -p helix-artifact`
- `cargo clippy -p helix-artifact --all-targets -- -D warnings`
- `cargo check -p helix-artifact --target wasm32-unknown-unknown`

## Child DOX Index

- None.
