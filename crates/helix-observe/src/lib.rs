//! # helix-observe — the Observatory artifact builder and HTTP server
//!
//! Stage 7 of the HELIX pipeline: turns one compile into a single
//! self-contained [`CompileArtifact`] JSON document (tokens, AST, IR dumps,
//! per-pass snapshots, laid-out CFGs, dominator trees, loop verdicts,
//! execution output) and serves it — plus the web UI — over HTTP.
//!
//! ```text
//! source ──pipeline──▶ CompileArtifact ──axum──▶ http://127.0.0.1:8931/
//!                          │
//!                          └──dot──▶ Graphviz .dot (publication figures)
//! ```
//!
//! ## Module map
//!
//! * [`artifact`] — the wire data contract (`docs/notes/artifact-schema.md`
//!   is normative; field names there are frozen). Everything derives serde so
//!   the whole artifact serializes with one call.
//! * [`layout`] — server-side tidy layout. The browser paints coordinates,
//!   never computes them: an AST Reingold–Tilford tree plus per-function CFG
//!   layering/routing live here.
//! * [`pipeline`] — `build_artifact(example, source)`: runs the real compiler
//!   front to back, snapshotting each stage and stopping at the first failure.
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
//! **Layout is a pure function.** `cfg_layout` consumes only the IR and loop
//! info, so identical compiles produce byte-identical artifacts — golden-test
//! friendly and cacheable without invalidation subtleties.
//!
//! **No panics across the boundary.** Every stage that could fail returns
//! rather than unwraps; the server maps unexpected failures to a 500 JSON
//! body instead of dropping the connection.

#![forbid(unsafe_code)]

pub mod artifact;
pub mod dot;
pub mod layout;
pub mod pipeline;
pub mod server;

// -- Contract surface --------------------------------------------------------

pub use artifact::CompileArtifact;
pub use dot::cfg_to_dot;
pub use layout::{LaidOutNode, TreeNode, ast_tree, cfg_layout, program_to_tree};
pub use pipeline::{BuildOpts, MAX_SOURCE_BYTES, build_artifact, build_artifact_with_opts};
pub use server::{ServeConfig, serve};
