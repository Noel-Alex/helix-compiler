//! Tokens and source spans for the HELIX language.
//!
//! The lexer ([crate::lexer]) turns raw source text into a flat `Vec<Token>`.
//! Every token records the [`Span`] — a half-open range of **byte offsets** —
//! it was produced from. Later stages copy those spans onto AST nodes, so a
//! diagnostic or a runtime error anywhere in the compiler can point back at
//! the exact characters the programmer wrote.
//!
//! Two disjoint word classes exist (see the frozen lexical grammar):
//!
//! * **keywords** (`fn let const if else for return true false as in`) are
//!   recognised by the lexer and surface as [`TokKind::Kw`];
//! * **reserved words** (`while break continue mut by and or not struct
//!   import`) are *not* keywords — they lex as ordinary identifiers and are
//!   rejected by the parser with a dedicated "reserved word" message. This
//!   keeps the token stream uniform while still reserving the names today.

use std::fmt;

use serde::{Deserialize, Serialize};

/// A half-open byte range `[start, end)` identifying a region of source text.
///
/// Offsets are counted in bytes from the beginning of the source string, so a
/// span can be applied directly to `&src[start..end]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Span {
    /// Byte offset of the first character of the region (inclusive).
    pub start: u32,
    /// Byte offset one past the last character of the region (exclusive).
    pub end: u32,
}

impl Span {
    /// Builds a span from two byte offsets.
    #[must_use]
    pub fn new(start: u32, end: u32) -> Self {
        debug_assert!(end >= start, "span end precedes span start");
        Self { start, end }
    }

    /// Returns the smallest span covering both `a` and `b`.
    ///
    /// Used by the parser to grow a node's span outward as it combines
    /// sub-nodes (`1 + 2` spans from the `1` through the `2`).
    #[must_use]
    pub fn join(a: Self, b: Self) -> Self {
        Self {
            start: a.start.min(b.start),
            end: a.end.max(b.end),
        }
    }
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}..{}", self.start, self.end)
    }
}

/// A HELIX keyword.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Kw {
    /// `fn` — function definition.
    Fn,
    /// `let` — variable declaration.
    Let,
    /// `const` — top-level constant definition.
    Const,
    /// `if` — conditional.
    If,
    /// `else` — conditional alternative.
    Else,
    /// `for` — counted loop (the only place `..` ranges may appear).
    For,
    /// `return` — early/explicit return.
    Return,
    /// `true` — boolean literal.
    True,
    /// `false` — boolean literal.
    False,
    /// `as` — explicit numeric cast.
    As,
    /// `in` — introduces the iteration variable bound of a `for` header.
    In,
}

impl Kw {
    /// Every keyword, in specification order.
    pub const ALL: [Kw; 11] = [
        Kw::Fn,
        Kw::Let,
        Kw::Const,
        Kw::If,
        Kw::Else,
        Kw::For,
        Kw::Return,
        Kw::True,
        Kw::False,
        Kw::As,
        Kw::In,
    ];

    /// The source spelling of this keyword.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Kw::Fn => "fn",
            Kw::Let => "let",
            Kw::Const => "const",
            Kw::If => "if",
            Kw::Else => "else",
            Kw::For => "for",
            Kw::Return => "return",
            Kw::True => "true",
            Kw::False => "false",
            Kw::As => "as",
            Kw::In => "in",
        }
    }

    /// Looks up the keyword spelled `name`, or `None` if `name` is not a
    /// keyword (identifiers and reserved words both return `None`).
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "fn" => Some(Kw::Fn),
            "let" => Some(Kw::Let),
            "const" => Some(Kw::Const),
            "if" => Some(Kw::If),
            "else" => Some(Kw::Else),
            "for" => Some(Kw::For),
            "return" => Some(Kw::Return),
            "true" => Some(Kw::True),
            "false" => Some(Kw::False),
            "as" => Some(Kw::As),
            "in" => Some(Kw::In),
            _ => None,
        }
    }
}

impl fmt::Display for Kw {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Words reserved in HELIX v1 but not yet implemented.
///
/// They lex as ordinary identifiers; the parser refuses them wherever a name
/// is expected so future versions can promote them to keywords without
/// breaking existing programs.
pub const RESERVED_WORDS: [&str; 10] = [
    "while", "break", "continue", "mut", "by", "and", "or", "not", "struct", "import",
];

/// Checks whether `name` is a reserved (future-keyword) word.
#[must_use]
pub fn is_reserved(name: &str) -> bool {
    RESERVED_WORDS.contains(&name)
}

/// The kind of a token: its classification plus any payload it carries.
///
/// Punctuation variants are named after the frozen punct list
/// `( ) { } [ ] , ; : :: .. -> + - * / % < > <= >= == != && || ! =`;
/// multi-character operators are matched with maximal munch by the lexer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TokKind {
    /// An identifier (or a reserved word, which is indistinguishable at this
    /// layer; see [`is_reserved`]).
    Ident(String),
    /// An integer literal, already parsed into an `i64`.
    Int(i64),
    /// A floating-point literal, already parsed into an `f64`.
    Float(f64),
    /// A keyword; see [`Kw`].
    Kw(Kw),
    /// `(`
    LParen,
    /// `)`
    RParen,
    /// `{`
    LBrace,
    /// `}`
    RBrace,
    /// `[`
    LBracket,
    /// `]`
    RBracket,
    /// `,`
    Comma,
    /// `;`
    Semi,
    /// `:`
    Colon,
    /// `::` (recognised for completeness; unused by the phrase grammar).
    PathSep,
    /// `..` (the half-open range separator of `for` headers).
    DotDot,
    /// `->` (return-type arrow).
    Arrow,
    /// `+`
    Plus,
    /// `-`
    Minus,
    /// `*`
    Star,
    /// `/`
    Slash,
    /// `%`
    Rem,
    /// `<`
    Lt,
    /// `>`
    Gt,
    /// `<=`
    Le,
    /// `>=`
    Ge,
    /// `==`
    EqEq,
    /// `!=`
    NotEq,
    /// `&&` (short-circuit and).
    AndAnd,
    /// `||` (short-circuit or).
    OrOr,
    /// `!` (logical negation).
    Not,
    /// `=` (assignment; a statement-level operator only).
    Assign,
    /// Synthetic end-of-stream token appended by [`crate::lex`].
    ///
    /// Its span is the empty range at the end of input, which lets the parser
    /// report `expected ..., found end of file` at a sensible location.
    Eof,
}

impl TokKind {
    /// Whether this is the end-of-file marker.
    #[must_use]
    pub fn is_eof(&self) -> bool {
        matches!(self, TokKind::Eof)
    }

    /// The exact source spelling of a punctuation token, if it has one.
    #[must_use]
    pub fn symbol(&self) -> Option<&'static str> {
        Some(match self {
            TokKind::LParen => "(",
            TokKind::RParen => ")",
            TokKind::LBrace => "{",
            TokKind::RBrace => "}",
            TokKind::LBracket => "[",
            TokKind::RBracket => "]",
            TokKind::Comma => ",",
            TokKind::Semi => ";",
            TokKind::Colon => ":",
            TokKind::PathSep => "::",
            TokKind::DotDot => "..",
            TokKind::Arrow => "->",
            TokKind::Plus => "+",
            TokKind::Minus => "-",
            TokKind::Star => "*",
            TokKind::Slash => "/",
            TokKind::Rem => "%",
            TokKind::Lt => "<",
            TokKind::Gt => ">",
            TokKind::Le => "<=",
            TokKind::Ge => ">=",
            TokKind::EqEq => "==",
            TokKind::NotEq => "!=",
            TokKind::AndAnd => "&&",
            TokKind::OrOr => "||",
            TokKind::Not => "!",
            TokKind::Assign => "=",
            _ => return None,
        })
    }

    /// Human-readable description used verbatim inside parse-error messages,
    /// e.g. `` identifier `x` ``, `` reserved word `while` `` or `` `}`` ``.
    #[must_use]
    pub fn describe(&self) -> String {
        match self {
            TokKind::Ident(name) if is_reserved(name) => {
                format!("reserved word `{name}`")
            }
            TokKind::Ident(name) => format!("identifier `{name}`"),
            TokKind::Int(v) => format!("integer `{v}`"),
            TokKind::Float(v) => format!("float `{v:?}`"),
            TokKind::Kw(k) => format!("`{k}`"),
            TokKind::Eof => "end of file".to_string(),
            other => match other.symbol() {
                Some(sym) => format!("`{sym}`"),
                None => "unknown token".to_string(),
            },
        }
    }
}

/// One lexed token: a [`TokKind`] plus the [`Span`] of source it came from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Token {
    /// Classification and payload of the token.
    pub kind: TokKind,
    /// Byte range of the source text this token was produced from.
    pub span: Span,
}

impl Token {
    /// Builds a token.
    #[must_use]
    pub fn new(kind: TokKind, span: Span) -> Self {
        Self { kind, span }
    }

    /// Whether this is the end-of-file marker.
    #[must_use]
    pub fn is_eof(&self) -> bool {
        self.kind.is_eof()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keyword_table_round_trips() {
        for k in Kw::ALL {
            assert_eq!(Kw::from_name(k.as_str()), Some(k), "keyword `{k}`");
        }
        assert_eq!(Kw::from_name("Function"), None);
        assert_eq!(Kw::from_name("iff"), None);
        assert_eq!(Kw::from_name(""), None);
    }

    #[test]
    fn reserved_words_are_recognised() {
        for word in RESERVED_WORDS {
            assert!(is_reserved(word), "`{word}` should be reserved");
            assert_eq!(Kw::from_name(word), None, "`{word}` must NOT be a keyword");
        }
        assert!(!is_reserved("whilst"));
        assert!(!is_reserved("mutable"));
        assert!(!is_reserved("fork"));
    }

    #[test]
    fn every_punct_has_a_symbol_and_description() {
        let punct = [
            TokKind::LParen,
            TokKind::RParen,
            TokKind::LBrace,
            TokKind::RBrace,
            TokKind::LBracket,
            TokKind::RBracket,
            TokKind::Comma,
            TokKind::Semi,
            TokKind::Colon,
            TokKind::PathSep,
            TokKind::DotDot,
            TokKind::Arrow,
            TokKind::Plus,
            TokKind::Minus,
            TokKind::Star,
            TokKind::Slash,
            TokKind::Rem,
            TokKind::Lt,
            TokKind::Gt,
            TokKind::Le,
            TokKind::Ge,
            TokKind::EqEq,
            TokKind::NotEq,
            TokKind::AndAnd,
            TokKind::OrOr,
            TokKind::Not,
            TokKind::Assign,
        ];
        for k in punct {
            assert!(k.symbol().is_some(), "{k:?} missing symbol");
            let d = k.describe();
            assert!(
                d.starts_with('`'),
                "description `{d}` should quote the symbol"
            );
        }
    }

    #[test]
    fn describe_covers_payloads_and_reserved_idents() {
        assert_eq!(
            TokKind::Ident("count".into()).describe(),
            "identifier `count`"
        );
        assert_eq!(
            TokKind::Ident("while".into()).describe(),
            "reserved word `while`"
        );
        assert_eq!(TokKind::Int(-0).describe(), "integer `0`");
        assert_eq!(TokKind::Int(42).describe(), "integer `42`");
        assert_eq!(TokKind::Float(2.5).describe(), "float `2.5`");
        assert_eq!(TokKind::Float(2.0).describe(), "float `2.0`");
        assert_eq!(TokKind::Kw(Kw::Fn).describe(), "`fn`");
        assert_eq!(TokKind::Eof.describe(), "end of file");
    }

    #[test]
    fn span_join_is_the_hull() {
        let a = Span::new(10, 20);
        let b = Span::new(15, 40);
        assert_eq!(Span::join(a, b), Span::new(10, 40));
        assert_eq!(Span::join(b, a), Span::new(10, 40));
        assert_eq!(Span::join(a, a), a);
        assert_eq!(a.to_string(), "10..20");
    }
}
