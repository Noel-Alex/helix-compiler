//! # helix-observe — the Observatory HTTP server
//!
//! Stage 7 of the HELIX pipeline: serves the browser UI plus the JSON API the
//! UI already speaks. Every artifact document it returns is built by
//! [`helix_artifact`], so the compile pipeline itself has no server
//! dependency and can also run inside wasm32 for the browser lab.
//!
//! ```text
//! source ──helix-artifact──▶ CompileArtifact ──axum──▶ http://127.0.0.1:8931/
//!                                     │
//!                                     └──dot──▶ Graphviz .dot (publication figures)
//! ```
//!
//! ## Module map
//!
//! * [`layout`] — browser-side tidy AST layout. The browser paints
//!   coordinates, never computes them: a Reingold–Tilford-flavoured layered
//!   tree over the display hierarchy produced from a serde `Program`. The CFG
//!   layout lives in `helix-artifact` because the artifact needs it.
//! * [`dot`] — Graphviz escape hatch mirroring the built-in layout.
//! * [`server`] — axum router: static UI assets (embedded via
//!   `include_bytes!`), `/api/examples`, `/api/artifact?example=…`,
//!   `/api/run` (POST), `/api/dot?example=…&fn=…` and the standalone
//!   offline export route.
//!
//! ## Design notes (course-report material)
//!
//! **Graceful degradation.** Compilation may stop at any stage, so every
//! post-parse stage is optional and the UI greys absent ones out. A lex error
//! yields tokens + `diags_lex`; a type error yields everything through the
//! AST plus underlined diagnostics; success yields the full stack including
//! interpreter output. The artifact is always valid JSON for its prefix.
//!
//! **No panics across the boundary.** Every stage that could fail returns
//! rather than unwraps; the server maps unexpected failures to a 500 JSON
//! body instead of dropping the connection.

#![forbid(unsafe_code)]

pub mod dot;
pub mod layout;
pub mod server;

// -- Contract surface --------------------------------------------------------

// The artifact pipeline is shared with the wasm32 browser lab; re-exported so
// `helix_observe::build_artifact` and friends keep working for API consumers.
pub use dot::cfg_to_dot;
pub use helix_artifact::{
    BuildOpts, CUSTOM_BENCH_SAMPLE_COUNT, CUSTOM_BENCH_WORK_BUDGET, CompileArtifact,
    INTERACTIVE_WORK_BUDGET, MAX_SOURCE_BYTES, artifact_for_example, bench_index, build_artifact,
    build_artifact_with_opts, cfg_layout, estimated_work, measure_interpreter,
};
pub use layout::{LaidOutNode, TreeNode, ast_tree, program_to_tree};
pub use server::{ServeConfig, serve};
