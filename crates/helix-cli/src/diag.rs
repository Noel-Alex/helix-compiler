//! Diagnostic pretty-printing: carets under source spans, shared by all subcommands.

use helix_syntax::Span;

/// Render one diagnostic with a caret line under the offending span.
///
/// ```text
/// error: undeclared variable 'y'
///   --> examples/type_errors.hx:2:23
///    |
///  2 |     let x = 3.5 + y;
///    |                       ^
/// ```
pub fn render(source: &str, filename: &str, span: Span, msg: &str) -> String {
    let (line_no, col, _line_start, line_text) = locate(source, span.start);
    // Caret run length in CHARS within this line (the old clamp mixed bytes
    // and chars, collapsing multi-byte spans to a single caret).
    let chars_on_line = line_text.chars().count();
    let caret_len = ((span.end.saturating_sub(span.start)) as usize)
        .min(chars_on_line.saturating_sub(col - 1) + 1)
        .max(1);
    let prefix = format!("{line_no:>4} | ");
    let mut out = String::new();
    out.push_str(&format!("error: {msg}\n"));
    out.push_str(&format!("  --> {filename}:{line_no}:{col}\n"));
    out.push_str("   |\n");
    out.push_str(&format!("{prefix}{line_text}\n"));
    out.push_str(&format!(
        "   | {}{}\n",
        " ".repeat(col - 1),
        "^".repeat(caret_len)
    ));
    out
}

/// (line number 1-based, column 1-based, byte offset of line start, line text)
fn locate(source: &str, offset: u32) -> (usize, usize, usize, String) {
    let off = offset as usize;
    let mut line_start = 0usize;
    let mut line_no = 1usize;
    for (i, b) in source.bytes().enumerate() {
        if i >= off {
            break;
        }
        if b == b'\n' {
            line_no += 1;
            line_start = i + 1;
        }
    }
    let rest = &source[line_start..];
    let line_end = rest.find('\n').map_or(rest.len(), |p| p);
    let line_text = rest[..line_end].trim_end_matches('\r').to_string();
    let col = source[line_start..off.min(source.len())].chars().count() + 1;
    (line_no, col, line_start, line_text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn caret_lands_under_the_span() {
        // `y` sits 9 chars into line 2 (byte offset 21): column 10, 1-based.
        let src = "fn main() {\n    let x = y;\n}";
        let out = render(src, "t.hx", Span { start: 21, end: 22 }, "undeclared variable 'y'");
        assert!(out.contains("t.hx:2:10"), "{out}");
        assert!(out.contains("^"), "{out}");
        assert!(out.contains("undeclared variable 'y'"), "{out}");
    }
}
