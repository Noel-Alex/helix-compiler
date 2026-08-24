//! Runtime errors and their source locations.
//!
//! The spec fixes the observable shape of a HELIX crash: the message
//! `runtime error: <message> at line N`, then exit status 1. This module owns
//! everything needed to produce those bytes.
//!
//! [`RunError`] carries a byte span (copied from the failing AST node), not a
//! line number. Lines are computed lazily by [`LineMap`], which scans the
//! source once and answers "which 1-based line does this offset fall on" in
//! O(log n). That keeps the interpreter core free of source-text knowledge —
//! it works purely in spans, like every other crate.

use std::fmt;

use helix_syntax::Span;

/// What went wrong at run time. Spec §Runtime errors names exactly these
/// situations: bounds violations, integer division/remainder by zero, and the
/// i64 overflow edge (`i64::MIN / -1`). The remaining variants cover
/// situations the spec leaves open but a reference implementation must still
/// handle deterministically.
#[derive(Clone, Debug)]
pub enum RunErrorKind {
    /// `a[i]` or `a[i] = v` where `i` is negative or `>= len(a)`.
    Bounds {
        /// Array length at the moment of the access.
        len: usize,
        /// Offending index (already widened to `i64`).
        idx: i64,
    },
    /// Integer `/` or `%` with divisor 0 — traps even under `--unchecked`.
    DivByZero,
    /// `i64::MIN / -1` or `i64::MIN % -1`: quotient overflows; hardware would trap.
    IdivOverflow,
    /// `zeros(n)` with `n < 0`: no sensible buffer exists, so this is a
    /// trapped runtime error (the task contract pins this behavior).
    NegativeZeros {
        /// The offending argument.
        n: i64,
    },
    /// Recursion exceeded the interpreter's call-depth limit. Not part of the
    /// language; purely a safety net so a runaway program fails cleanly
    /// instead of exhausting the host thread's stack.
    StackExhausted,
    /// A structural invariant broke between frontend and engine (bad span
    /// join, missing symbol). Unreachable on programs sema accepted; reported
    /// instead of panicking because the interpreter is the reference oracle.
    Internal(String),
}

impl RunErrorKind {
    /// The fixed `<message>` text of this error, per spec wording where the
    /// spec defines one.
    #[must_use]
    pub fn message(&self) -> String {
        match self {
            RunErrorKind::Bounds { len, idx } => {
                format!("index {idx} out of bounds for array of length {len}")
            }
            RunErrorKind::DivByZero => "integer division by zero".to_string(),
            RunErrorKind::IdivOverflow => "integer division overflow (i64::MIN / -1)".to_string(),
            RunErrorKind::NegativeZeros { n } => {
                format!("zeros({n}): array length must be non-negative")
            }
            RunErrorKind::StackExhausted => {
                "call stack exhausted (recursion deeper than the interpreter limit)".to_string()
            }
            RunErrorKind::Internal(m) => format!("internal interpreter inconsistency: {m}"),
        }
    }
}

/// A runtime failure: what happened and where (byte span into the source).
#[derive(Clone, Debug)]
pub struct RunError {
    /// Which class of failure occurred.
    pub kind: RunErrorKind,
    /// Span of the offending node (the index expression for bounds errors,
    /// the whole binary expression for division errors).
    pub span: Span,
}

impl RunError {
    /// Builds an error at `span`.
    #[must_use]
    pub fn new(kind: RunErrorKind, span: Span) -> Self {
        Self { kind, span }
    }

    /// Renders the spec-mandated message: `runtime error: <message> at line N`.
    ///
    /// Requires the source text only because spans are byte offsets; use
    /// [`crate::error::RunError::render_with`] when no source is available.
    #[must_use]
    pub fn render(&self, src: &str) -> String {
        let lines = LineMap::new(src);
        self.render_with(Some(&lines))
    }

    /// Like [`render`](RunError::render) but tolerates missing source text:
    /// without a map there is nothing to translate the span into a line, so
    /// the message degrades to quoting the raw span instead of lying.
    #[must_use]
    pub fn render_with(&self, lines: Option<&LineMap>) -> String {
        match lines {
            Some(l) => format!(
                "runtime error: {} at line {}",
                self.kind.message(),
                l.line_of(self.span.start)
            ),
            None => format!(
                "runtime error: {} (at byte offset {})",
                self.kind.message(),
                self.span.start
            ),
        }
    }
}

impl fmt::Display for RunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.render_with(None))
    }
}

impl std::error::Error for RunError {}

// ---------------------------------------------------------------------------
// LineMap — span → 1-based line
// ---------------------------------------------------------------------------

/// Maps byte offsets to 1-based line numbers by remembering newline positions.
///
/// Built once per run (or per diagnostic render); queries are binary search.
pub struct LineMap {
    /// Byte offset of the first character of each line (line 0 starts at 0).
    line_starts: Vec<u32>,
}

impl LineMap {
    /// Scans `src` for newlines.
    #[must_use]
    pub fn new(src: &str) -> Self {
        let mut line_starts = vec![0u32];
        for (i, b) in src.bytes().enumerate() {
            if b == b'\n' {
                line_starts.push(i as u32 + 1);
            }
        }
        Self { line_starts }
    }

    /// 1-based line containing byte offset `offset`.
    ///
    /// `partition_point` finds the first line start strictly greater than
    /// `offset`; its index is exactly the 1-based line number (offset 0 is on
    /// line 1, an offset just past the first newline is on line 2, …). An
    /// offset past end-of-source clamps to the last line, which is what a
    /// diagnostic wants anyway.
    #[must_use]
    pub fn line_of(&self, offset: u32) -> u32 {
        self.line_starts.partition_point(|&s| s <= offset) as u32
    }

    /// Peek at the recorded start offsets (for tests).
    #[must_use]
    pub fn starts(&self) -> &[u32] {
        &self.line_starts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_numbers_are_one_based_and_split_on_newlines() {
        let src = "fn main() {\n  let x = 1;\n  let y = x / z;\n}\n";
        let m = LineMap::new(src);
        // Newlines at offsets 11, 24, 41, 43 -> starts at 0, 12, 25, 42, 44.
        assert_eq!(m.starts(), &[0, 12, 25, 42, 44]);
        assert_eq!(m.line_of(0), 1); // 'f' of fn
        assert_eq!(m.line_of(11), 1); // the newline itself stays on its own line
        assert_eq!(m.line_of(12), 2);
        assert_eq!(m.line_of(24), 2);
        assert_eq!(m.line_of(25), 3);
        assert_eq!(m.line_of(41), 3);
        assert_eq!(m.line_of(42), 4);
    }

    #[test]
    fn error_renders_spec_message() {
        // "fn main() {\n    let q = 1 / 0;\n}\n": newline offsets are 11 and
        // 30, so offset 20 (`/`) sits on line 2.
        let e = RunError::new(RunErrorKind::DivByZero, Span { start: 20, end: 25 });
        let src = "fn main() {\n    let q = 1 / 0;\n}\n";
        assert_eq!(
            e.render(src),
            "runtime error: integer division by zero at line 2"
        );
        assert_eq!(
            e.to_string(),
            "runtime error: integer division by zero (at byte offset 20)"
        );
    }

    #[test]
    fn bounds_message_quotes_index_and_length() {
        let e = RunError::new(
            RunErrorKind::Bounds { len: 4, idx: -1 },
            Span { start: 0, end: 1 },
        );
        assert_eq!(
            e.render_with(None),
            "runtime error: index -1 out of bounds for array of length 4 (at byte offset 0)"
        );
    }
}
