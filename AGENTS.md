# DOX framework

- DOX is highly performant AGENTS.md hierarchy installed here
- Agent must follow DOX instructions across any edits

## Core Contract

- AGENTS.md files are binding work contracts for their subtrees
- Work products, source materials, instructions, records, assets, and durable docs must stay understandable from the nearest applicable AGENTS.md plus every parent AGENTS.md above it

## Read Before Editing

1. Read the root AGENTS.md
2. Identify every file or folder you expect to touch
3. Walk from the repository root to each target path
4. Read every AGENTS.md found along each route
5. If a parent AGENTS.md lists a child AGENTS.md whose scope contains the path, read that child and continue from there
6. Use the nearest AGENTS.md as the local contract and parent docs for repo-wide rules
7. If docs conflict, the closer doc controls local work details, but no child doc may weaken DOX

Do not rely on memory. Re-read the applicable DOX chain in the current session before editing.

## Update After Editing

Every meaningful change requires a DOX pass before the task is done.

Update the closest owning AGENTS.md when a change affects:

- purpose, scope, ownership, or responsibilities
- durable structure, contracts, workflows, or operating rules
- required inputs, outputs, permissions, constraints, side effects, or artifacts
- user preferences about behavior, communication, process, organization, or quality
- AGENTS.md creation, deletion, move, rename, or index contents

Update parent docs when parent-level structure, ownership, workflow, or child index changes. Update child docs when parent changes alter local rules. Remove stale or contradictory text immediately. Small edits that do not change behavior or contracts may leave docs unchanged, but the DOX pass still must happen.

## Hierarchy

- Root AGENTS.md is the DOX rail: project-wide instructions, global preferences, durable workflow rules, and the top-level Child DOX Index
- Child AGENTS.md files own domain-specific instructions and their own Child DOX Index
- Each parent explains what its direct children cover and what stays owned by the parent
- The closer a doc is to the work, the more specific and practical it must be

## Child Doc Shape

- Create a child AGENTS.md when a folder becomes a durable boundary with its own purpose, rules, responsibilities, workflow, materials, or quality standards
- Work Guidance must reflect the current standards of the project or user instructions; if there are no specific standards or instructions yet, leave it empty
- Verification must reflect an existing check; if no verification framework exists yet, leave it empty and update it when one exists

Default section order:
- Purpose
- Ownership
- Local Contracts
- Work Guidance
- Verification
- Child DOX Index

## Style

- Keep docs concise, current, and operational
- Document stable contracts, not diary entries
- Put broad rules in parent docs and concrete details in child docs
- Prefer direct bullets with explicit names
- Do not duplicate rules across many files unless each scope needs a local version
- Delete stale notes instead of explaining history
- Trim obvious statements, repeated rules, misplaced detail, and warnings for risks that no longer exist

## Closeout

1. Re-check changed paths against the DOX chain
2. Update nearest owning docs and any affected parents or children
3. Refresh every affected Child DOX Index
4. Remove stale or contradictory text
5. Run existing verification when relevant
6. Report any docs intentionally left unchanged and why

## User Preferences

When the user requests a durable behavior change, record it here or in the relevant child AGENTS.md

## Compiler correctness contract

- Parallelization requires proof: non-affine or zero-stride writes stay sequential. A sole write is injective only with a no-wrap proof over the actual integer width, or a same-width odd-coefficient modular permutation (an invariant offset like `a[i+k]` qualifies); a nonzero mathematical coefficient alone never proves injectivity because runtime `i64`/`i32` arithmetic wraps. Paired read/write proofs additionally require exact no-wrap affine forms. Read dependence distances stay conservative for unknown nonliteral offsets; narrowing-cast loop bounds stay symbolic. Per-iteration allocation (e.g. `zeros` inside a loop) is vetoed up front.
- Loop-carried scalar header phis must be induction variables, recognized reductions, or provably unused anywhere. Subtraction reductions require the accumulator on the left; equal constant addresses carry unknown distance, not a zero-distance independence proof.
- Observer work estimates charge direct `let` allocations by resolved length; unknown allocation lengths and loop bounds skip execution. This syntactic estimate is not a call-graph runtime bound.
- Observatory diagnostics include lexer/parser and semantic errors; show their messages, keep SOURCE accessible without tokens, and never report compilation success for an error artifact.
- Recorded benchmark attachment requires executable-source identity against the frozen catalog revision. Probe provenance is separate from campaign provenance; neither is a claim about the current build's performance.
- Every consumer that EXECUTES a HELIX program runs the one shared sequence,
  `helix_artifact::prepare_pipeline` (parse → sema → IR build → to_ssa →
  optimizer → find_loops → analyze → build_plan). A per-consumer copy of the
  stage order is a bug: it lets the CLI, the campaign and the Observatory
  disagree about what "compiled" means.
- `native-seq` in the campaign is compiled from an EMPTY backend plan — no
  parallel regions — so the sequential baseline is genuinely sequential. The
  parallel variant compiles the analyzer's real plan and pins `HELIX_NTHREADS`.
- The region's emitted thread hint is a ceiling no host can reach; effective
  participants are `min(HELIX_NTHREADS, available_parallelism)`, so a sweep row
  labelled `p` really runs `p`.
- The campaign's parity gate compares printed lines AND the interpreter's
  full-state checksum (printed lines + main's final arrays) for `interp` and
  `native-seq`; a parallel variant is checksum-compared only when the kernel
  documents no tolerance, because nonzero ε means FP sums legitimately
  reassociate.

## Verification

- Run `cargo test --workspace` for compiler/runtime changes; on Windows launch Cargo through PowerShell so Git Bash's `link.exe` does not shadow MSVC.
- `cargo test -p helix-analysis --test soundness_fuzz` and `cargo test -p helix-backend --test reduction_seeds` are the soundness gates: the fuzzer asserts every approved loop is conflict-free under a brute-force oracle (one-way property), and the seed tests pin non-identity reductions exact across worker counts.
- Regenerate the recorded campaign after compiler changes with `helix bench --out docs/benchmarks/data` plus `node tools/plot_bench.mjs docs/benchmarks/data/campaign.json -o docs/benchmarks/figs`, then update `campaign_git_rev` in `crates/helix-artifact/src/recorded-sources.json` to the recorded revision so evidence still attaches.
- Build the browser lab with `cargo build --release --target wasm32-unknown-unknown -p helix-wasm` and copy the result to `web/wasm/helix-lab.wasm`; CI builds it without overwriting the committed asset, and the `Deploy Pages` workflow ships the freshly built one to GitHub Pages.
- Run `node web/app.test.mjs` for Observatory client state changes and `node web/wasm/helix-browser.test.mjs` for worker lifecycle changes. Browser WASM execution stays off the UI thread with a five-second deadline; benchmark samples exclude worker startup and transport. The execution readout paints one evidence source per render (a failed browser run never mixes its empty output with observer output), and measurement/selection commits preserve in-flight editor drafts. Standalone exports inline every asset (style, scripts, fonts as data URIs) so the saved file boots with no server. Rebuild the observer (web assets are embedded) and verify changed interactions in the browser, including mobile when layout changes.

## Child DOX Index

- `crates/helix-artifact/AGENTS.md` owns `crates/helix-artifact/`: the pure compile-artifact builder that both the Observatory server and the wasm32 browser lab link, and the owner of `prepare_pipeline` — the one compile sequence every executing consumer shares.
- `crates/helix-wasm-backend/AGENTS.md` owns `crates/helix-wasm-backend/`: the IR→WebAssembly emitter plus the frozen emitted-module ABI the browser executes.
- `crates/helix-wasm/AGENTS.md` owns `crates/helix-wasm/`: the wasm bridge and its `helix_apply` JSON entry; the shipped `web/wasm/helix-lab.wasm` is built from it.
- `web/` owns the Observatory static surface: the `app.js` renderer, the wasm worker/bridge under `web/wasm/`, and the Pages data contract `/data/examples/<name>.hx`.
  - `web/wasm/AGENTS.md` owns the browser execution surface: the frozen worker protocol, the bridge API, and the five-second deadline.
  - The browser lab runs 100% client-side on GitHub Pages with no server: the wasm bridge does all compilation, the compiled path is the `helix-wasm-backend` module executed in a worker with a five-second deadline, interpreter/module parity (printed lines + FNV checksum) is asserted, and no native or parallel numbers may ever be synthesized for browser runs.
- Root-owned files: `README.md`, `LICENSE`, `Cargo.toml`, `Cargo.lock`, `.gitattributes`, `.gitignore`, and root-level project documentation.
