//! Shared helpers for the helix-observe integration tests.

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

/// Every example stem, sorted.
#[must_use]
pub fn all_example_names() -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(examples_dir())
        .expect("examples dir")
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "hx"))
        .filter_map(|e| {
            e.path()
                .file_stem()
                .and_then(|s| s.to_str().map(String::from))
        })
        .collect();
    names.sort();
    names
}
