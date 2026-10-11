//! Shared helpers for the helix-observe integration tests.
//!
//! The artifact-side tests (every example, all CFG geometry) need the full
//! example listing and live in `helix-artifact`; here only the AST-tree tests
//! read a single example.

/// Absolute path of the repository's `examples/` directory.
#[must_use]
pub fn examples_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples")
        .canonicalize()
        .expect("examples dir exists next to the crate")
}

/// Reads `examples/<name>.hx`.
#[must_use]
pub fn example_source(name: &str) -> String {
    std::fs::read_to_string(examples_dir().join(format!("{name}.hx")))
        .unwrap_or_else(|e| panic!("read example {name}.hx: {e}"))
}
