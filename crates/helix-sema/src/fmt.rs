//! Canonical scalar formatting — the single source of truth for what `print(x)` outputs,
//! shared verbatim by the interpreter and the JIT runtime so both backends render
//! identical bytes (a differential-testing requirement).
//!
//! Choices (documented in lang-spec.md):
//! - integers: plain decimal
//! - bools: `true` / `false`
//! - floats: Rust `Debug` formatting (`{:?}`) per width — always shows a decimal point
//!   (`1.0`), shortest round-trip digits, `NaN`/`inf`/`-inf` for specials.

/// Format an i64 (`print` of any integer width widens to i64 first).
#[must_use]
pub fn fmt_i64(v: i64) -> String {
    v.to_string()
}

/// Format a bool.
#[must_use]
pub fn fmt_bool(v: bool) -> String {
    if v { "true".into() } else { "false".into() }
}

/// Format an f64.
#[must_use]
pub fn fmt_f64(v: f64) -> String {
    format!("{v:?}")
}

/// Format an f32 **as f32** — never widen first, or `0.1f32` would print as
/// `0.10000000149011612`.
#[must_use]
pub fn fmt_f32(v: f32) -> String {
    format!("{v:?}")
}
