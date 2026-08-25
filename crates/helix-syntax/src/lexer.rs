//! The HELIX lexer: turns source text into a flat stream of [`Token`]s.
//!
//! # Design
//!
//! A single hand-written scanner walks the source as bytes, tracking a byte
//! offset (`self.pos`). At each step it skips whitespace and comments
//! ([`Lexer::skip_trivia`]), then recognises the next token by its leading
//! character:
//!
//! * digit            → numeric literal (see [`Lexer::lex_number`])
//! * alphabetic / `_` → identifier or keyword (see [`Lexer::lex_word`]);
//!   keywords win over identifiers only when the *whole* word matches, so
//!   `iff` stays an identifier (maximal munch)
//! * punctuation      → longest match first (`<=` before `<`, `&&` before `&`)
//!
//! Comments follow the frozen lexical grammar and are **non-nested** (a
//! deliberate divergence from Rust): `// ... \n` runs to end of line and
//! C-style `/* ... */` ends at the *first* `*/`. An unterminated block
//! comment is an error carrying the span from `/*` to end of input.
//!
//! Numeric literals accept both spec forms — `DIGIT+ '.' DIGIT+ [EXP]` and
//! `DIGIT+ EXP`. A `.` only continues a number when a digit follows it, so
//! range expressions such as `for i in 1..n` lex as `Int DotDot Ident`
//! rather than swallowing `.` into a malformed float.
//!
//! Errors stop scanning immediately and are reported as a [`LexError`]
//! carrying the exact byte span of the offending text.

use serde::{Deserialize, Serialize};

use crate::token::{Kw, Span, TokKind, Token};

/// A lexical error with the source span it was detected at.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LexError {
    /// Where the offending text starts and ends (byte offsets).
    pub span: Span,
    /// Human-readable explanation.
    pub msg: String,
}

impl std::fmt::Display for LexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "lexical error at {}: {}", self.span, self.msg)
    }
}

impl std::error::Error for LexError {}

/// Lexes `src` into a token list ending in one synthetic [`TokKind::Eof`]
/// token whose span is the empty range at end-of-input.
///
/// # Errors
///
/// Returns the first [`LexError`] encountered (comments, characters,
/// integer/float overflow).
pub fn lex(src: &str) -> Result<Vec<Token>, LexError> {
    Lexer { src, pos: 0 }.tokenize()
}

/// Internal scanner state.
struct Lexer<'a> {
    src: &'a str,
    pos: usize,
}

// Two-character operators, longest-match candidates checked before any
// one-character operator that prefixes them ("<=" before "<", "::" before ":").
const TWO_CHAR_OPS: [(TokKind, &str); 9] = [
    (TokKind::PathSep, "::"),
    (TokKind::DotDot, ".."),
    (TokKind::Arrow, "->"),
    (TokKind::Le, "<="),
    (TokKind::Ge, ">="),
    (TokKind::EqEq, "=="),
    (TokKind::NotEq, "!="),
    (TokKind::AndAnd, "&&"),
    (TokKind::OrOr, "||"),
];

impl Lexer<'_> {
    fn tokenize(mut self) -> Result<Vec<Token>, LexError> {
        let mut out = Vec::new();
        loop {
            self.skip_trivia()?;
            if self.pos >= self.src.len() {
                let here = self.here();
                out.push(Token::new(TokKind::Eof, Span::new(here, here)));
                return Ok(out);
            }
            out.push(self.next_token()?);
        }
    }

    // -- position helpers ---------------------------------------------------

    /// Current byte offset as a [`Span`] coordinate.
    fn here(&self) -> u32 {
        u32::try_from(self.pos).expect("source larger than 4 GiB")
    }

    /// The ASCII byte at the current offset, if any.
    fn peek(&self) -> Option<u8> {
        self.src.as_bytes().get(self.pos).copied()
    }

    /// The ASCII byte two ahead of the current offset, if any.
    fn peek2(&self) -> Option<u8> {
        self.src.as_bytes().get(self.pos + 1).copied()
    }

    /// Advances past one ASCII byte.
    fn bump(&mut self) {
        self.pos += 1;
    }

    // -- whitespace and comments -------------------------------------------

    /// Skips whitespace and both comment forms. Block comments do NOT nest:
    /// the first `*/` closes the comment regardless of interior `/*`.
    fn skip_trivia(&mut self) -> Result<(), LexError> {
        loop {
            match self.peek() {
                Some(b' ') | Some(b'\t') | Some(b'\r') | Some(b'\n') => self.bump(),
                Some(b'/') if self.peek2() == Some(b'/') => {
                    // Line comment: consume up to (not including) the newline;
                    // the whitespace arm eats the newline on the next pass.
                    while matches!(self.peek(), Some(c) if c != b'\n') {
                        self.bump();
                    }
                }
                Some(b'/') if self.peek2() == Some(b'*') => {
                    let open = self.here();
                    self.pos += 2; // consume "/*"
                    loop {
                        match self.peek() {
                            None => {
                                return Err(LexError {
                                    span: Span::new(open, self.here()),
                                    msg: "unterminated block comment".to_string(),
                                });
                            }
                            Some(b'*') if self.peek2() == Some(b'/') => {
                                self.pos += 2;
                                break;
                            }
                            _ => self.bump(),
                        }
                    }
                }
                _ => break,
            }
        }
        Ok(())
    }

    // -- tokens -------------------------------------------------------------

    fn next_token(&mut self) -> Result<Token, LexError> {
        let start = self.here();
        let kind = match self.peek().expect("caller checked non-eof") {
            c if c.is_ascii_digit() => self.lex_number(),
            c if c.is_ascii_alphabetic() || c == b'_' => Ok(self.lex_word()),
            _ => self.lex_punct(),
        }?;
        Ok(Token::new(kind, Span::new(start, self.here())))
    }

    /// Lexes an identifier or keyword. Per the frozen spec identifiers are
    /// pure ASCII: `(ALPHA|'_') (ALPHA|DIGIT|'_)*`.
    fn lex_word(&mut self) -> TokKind {
        let start = self.pos;
        while matches!(self.peek(), Some(c) if c.is_ascii_alphanumeric() || c == b'_') {
            self.bump();
        }
        let text = &self.src[start..self.pos];
        match Kw::from_name(text) {
            Some(kw) => TokKind::Kw(kw),
            None => TokKind::Ident(text.to_string()),
        }
    }

    /// Lexes a numeric literal, distinguishing INT_LIT from FLOAT_LIT.
    ///
    /// Grammar: `DIGIT+ | DIGIT+ '.' DIGIT+ [EXP] | DIGIT+ [EXP]` where
    /// `EXP ::= ('e'|'E') ['+'|'-'] DIGIT+`.
    fn lex_number(&mut self) -> Result<TokKind, LexError> {
        let start = self.pos;
        self.skip_digits();

        // Fractional part: a '.' continues the literal ONLY when a digit
        // follows, keeping `1..n` as Int + DotDot + Ident.
        let mut is_float = false;
        if self.peek() == Some(b'.') && self.peek2().is_some_and(|c| c.is_ascii_digit()) {
            is_float = true;
            self.bump(); // '.'
            self.skip_digits();
        }

        // Exponent part, valid on both int-looking and float-looking forms
        // (`1e5` is a float even though it has no '.'). If what follows the
        // e/E is not a well-formed exponent body, the letter belongs to the
        // next token instead.
        if matches!(self.peek(), Some(b'e' | b'E')) && self.well_formed_exponent_follows() {
            is_float = true;
            self.bump(); // e / E
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.bump();
            }
            self.skip_digits();
        }

        let text = &self.src[start..self.pos];
        if is_float {
            // The grammar above guarantees `text` parses; overflow yields
            // INFINITY (never an error), so literals like `1e999` are
            // rejected here — INFINITY would otherwise reach the AST and
            // serialise as JSON `null` in Observatory artifacts.
            let value = text.parse::<f64>().unwrap_or(f64::INFINITY);
            if value.is_infinite() {
                return Err(LexError {
                    span: Span::new(
                        u32::try_from(start).unwrap_or(u32::MAX),
                        self.here(),
                    ),
                    msg: format!("float literal `{text}` is too large"),
                });
            }
            Ok(TokKind::Float(value))
        } else {
            text.parse::<i64>().map(TokKind::Int).map_err(|_| LexError {
                span: Span::new(u32::try_from(start).unwrap_or(u32::MAX), self.here()),
                msg: format!("integer literal `{text}` does not fit in i64"),
            })
        }
    }

    /// Whether a well-formed exponent body follows the current `e`/`E`
    /// (optional sign, then at least one digit).
    fn well_formed_exponent_follows(&self) -> bool {
        let bytes = self.src.as_bytes();
        let mut i = self.pos + 1;
        if bytes.get(i).is_some_and(|c| *c == b'+' || *c == b'-') {
            i += 1;
        }
        bytes.get(i).is_some_and(|c| c.is_ascii_digit())
    }

    fn skip_digits(&mut self) {
        while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
            self.bump();
        }
    }

    /// Lexes one punctuation/operator token using maximal munch: every
    /// two-character operator is tried before its one-character prefix.
    fn lex_punct(&mut self) -> Result<TokKind, LexError> {
        let start = self.pos;
        let rest = &self.src[start..];

        for (kind, sym) in TWO_CHAR_OPS {
            if rest.starts_with(sym) {
                self.pos += sym.len();
                return Ok(kind);
            }
        }

        let one = match self.peek() {
            Some(b'(') => TokKind::LParen,
            Some(b')') => TokKind::RParen,
            Some(b'{') => TokKind::LBrace,
            Some(b'}') => TokKind::RBrace,
            Some(b'[') => TokKind::LBracket,
            Some(b']') => TokKind::RBracket,
            Some(b',') => TokKind::Comma,
            Some(b';') => TokKind::Semi,
            Some(b':') => TokKind::Colon,
            Some(b'+') => TokKind::Plus,
            Some(b'-') => TokKind::Minus,
            Some(b'*') => TokKind::Star,
            Some(b'/') => TokKind::Slash,
            Some(b'%') => TokKind::Rem,
            Some(b'<') => TokKind::Lt,
            Some(b'>') => TokKind::Gt,
            Some(b'!') => TokKind::Not,
            Some(b'=') => TokKind::Assign,
            other => return Err(self.unexpected_char_error(other)),
        };
        self.bump();
        Ok(one)
    }

    /// Builds the "unexpected character" error for the (possibly multi-byte)
    /// character starting at the current position.
    fn unexpected_char_error(&self, first_byte: Option<u8>) -> LexError {
        let start = self.pos;
        let width = first_byte.map_or(1, utf8_width);
        let end = (start + width).min(self.src.len());
        let shown = self.src[start..end].to_string();
        LexError {
            span: Span::new(
                u32::try_from(start).unwrap_or(u32::MAX),
                u32::try_from(end).unwrap_or(u32::MAX),
            ),
            msg: format!("unexpected character `{shown}`"),
        }
    }
}

/// Byte width of the UTF-8 sequence whose first byte is `first_byte`.
fn utf8_width(first_byte: u8) -> usize {
    match first_byte {
        0x00..=0x7F => 1,
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        _ => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Convenience: lex and keep only kinds, dropping the trailing Eof.
    fn kinds(src: &str) -> Vec<TokKind> {
        lex(src)
            .expect("should lex")
            .into_iter()
            .filter(|t| !t.is_eof())
            .map(|t| t.kind)
            .collect()
    }

    #[test]
    fn empty_input_yields_only_eof() {
        let toks = lex("").expect("empty");
        assert_eq!(toks.len(), 1);
        assert!(toks[0].is_eof());
        assert_eq!(toks[0].span, Span::new(0, 0));
    }

    #[test]
    fn whitespace_only_is_eof_at_end() {
        let src = "  \t\r\n ";
        let toks = lex(src).expect("ws");
        // `kinds` drops Eof, so the filtered stream is empty here.
        assert!(kinds(src).is_empty());
        assert_eq!(toks.len(), 1);
        assert!(toks[0].is_eof());
        assert_eq!(toks[0].span, Span::new(6, 6));
    }

    #[test]
    fn every_keyword_lexes_to_kw_variant() {
        let src = "fn let const if else for return true false as in";
        assert_eq!(
            kinds(src),
            vec![
                TokKind::Kw(Kw::Fn),
                TokKind::Kw(Kw::Let),
                TokKind::Kw(Kw::Const),
                TokKind::Kw(Kw::If),
                TokKind::Kw(Kw::Else),
                TokKind::Kw(Kw::For),
                TokKind::Kw(Kw::Return),
                TokKind::Kw(Kw::True),
                TokKind::Kw(Kw::False),
                TokKind::Kw(Kw::As),
                TokKind::Kw(Kw::In),
            ]
        );
    }

    #[test]
    fn reserved_words_lex_as_plain_identifiers() {
        let words: Vec<String> = lex("while break continue mut by and or not struct import")
            .expect("reserved")
            .into_iter()
            .filter_map(|t| match t.kind {
                TokKind::Ident(n) => Some(n),
                _ => None,
            })
            .collect();
        assert_eq!(
            words,
            [
                "while", "break", "continue", "mut", "by", "and", "or", "not", "struct", "import"
            ]
        );
    }

    #[test]
    fn identifiers_vs_keywords_maximal_munch() {
        // `iff` and `fnord` are NOT keywords even though they start with one.
        assert_eq!(
            kinds("iff fnord f fn"),
            vec![
                TokKind::Ident("iff".into()),
                TokKind::Ident("fnord".into()),
                TokKind::Ident("f".into()),
                TokKind::Kw(Kw::Fn),
            ]
        );
        // Trailing underscore keeps a word an identifier: `as_` != `as`.
        assert_eq!(kinds("as_"), vec![TokKind::Ident("as_".into())]);
        assert_eq!(kinds("trueish"), vec![TokKind::Ident("trueish".into())]);
    }

    #[test]
    fn underscores_and_digits_in_identifiers() {
        assert_eq!(
            kinds("_a1 b_c d_9e __"),
            vec![
                TokKind::Ident("_a1".into()),
                TokKind::Ident("b_c".into()),
                TokKind::Ident("d_9e".into()),
                TokKind::Ident("__".into()),
            ]
        );
    }

    #[test]
    fn integer_literals_parse_to_i64() {
        assert_eq!(
            kinds("0 7 12345"),
            vec![TokKind::Int(0), TokKind::Int(7), TokKind::Int(12_345)]
        );
        assert_eq!(kinds("9223372036854775807"), vec![TokKind::Int(i64::MAX)]);
        // No negative literals: `-5` is unary minus applied to 5.
        assert_eq!(kinds("-5"), vec![TokKind::Minus, TokKind::Int(5)]);
        // Consequently `-i64::MIN` cannot even be written: the magnitude
        // alone overflows and is rejected by the lexer.
        assert!(lex("-9223372036854775808").is_err());
    }

    #[test]
    fn integer_overflow_is_an_error_with_span() {
        let err = lex("9223372036854775808").expect_err("overflow");
        assert_eq!(err.span, Span::new(0, 19));
        assert!(err.msg.contains("i64"), "{}", err.msg);
    }

    #[test]
    fn float_forms_including_exponents() {
        assert_eq!(
            kinds("1.5 0.25 100.125"),
            vec![
                TokKind::Float(1.5),
                TokKind::Float(0.25),
                TokKind::Float(100.125)
            ]
        );
        assert_eq!(kinds("2e3"), vec![TokKind::Float(2000.0)]);
        assert_eq!(kinds("1E-2"), vec![TokKind::Float(0.01)]);
        assert_eq!(kinds("3.14e+2"), vec![TokKind::Float(314.0)]);
        assert_eq!(kinds("1.5e10"), vec![TokKind::Float(1.5e10)]);
        assert_eq!(kinds("7e0"), vec![TokKind::Float(7.0)]);
    }

    #[test]
    fn huge_float_overflowing_f64_is_rejected() {
        // INFINITY would reach the AST and serialise as JSON null in
        // Observatory artifacts, so overflow is a lex error instead.
        let err = lex("1e999").expect_err("float overflow");
        assert!(err.msg.contains("too large"), "{}", err.msg);
        assert_eq!(err.span, Span::new(0, 5));
        // Underflow to zero is still fine.
        assert_eq!(kinds("1e-999"), vec![TokKind::Float(0.0)]);
    }

    #[test]
    fn dot_dot_range_does_not_swallow_digits() {
        assert_eq!(
            kinds("for i in 1..n {}"),
            vec![
                TokKind::Kw(Kw::For),
                TokKind::Ident("i".into()),
                TokKind::Kw(Kw::In),
                TokKind::Int(1),
                TokKind::DotDot,
                TokKind::Ident("n".into()),
                TokKind::LBrace,
                TokKind::RBrace,
            ]
        );
        // Real floats around a range still split correctly.
        assert_eq!(
            kinds("1.5..2.5"),
            vec![TokKind::Float(1.5), TokKind::DotDot, TokKind::Float(2.5)]
        );
    }

    #[test]
    fn trailing_dot_before_ident_ends_the_number_then_errors() {
        // `1.x`: no fractional digits, so Int(1) is emitted, then the bare
        // '.' is not a HELIX punctuation character and errors out.
        let err = lex("let x = 1.y").expect_err("bare dot");
        assert!(err.msg.contains('.'), "{}", err.msg);
        assert_eq!(err.span, Span::new(9, 10));
    }

    #[test]
    fn all_punctuation_tokens_in_spec_order() {
        let src = "( ) { } [ ] , ; : :: .. -> + - * / % < > <= >= == != && || ! =";
        assert_eq!(
            kinds(src),
            vec![
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
            ]
        );
    }

    #[test]
    fn maximal_munch_on_operators_without_spaces() {
        assert_eq!(
            kinds("<=>==="),
            vec![TokKind::Le, TokKind::Ge, TokKind::EqEq]
        );
        assert_eq!(
            kinds("a<=b>=c==d!=e&&f||g"),
            vec![
                TokKind::Ident("a".into()),
                TokKind::Le,
                TokKind::Ident("b".into()),
                TokKind::Ge,
                TokKind::Ident("c".into()),
                TokKind::EqEq,
                TokKind::Ident("d".into()),
                TokKind::NotEq,
                TokKind::Ident("e".into()),
                TokKind::AndAnd,
                TokKind::Ident("f".into()),
                TokKind::OrOr,
                TokKind::Ident("g".into()),
            ]
        );
        assert_eq!(kinds("->"), vec![TokKind::Arrow]);
        assert_eq!(kinds("- >"), vec![TokKind::Minus, TokKind::Gt]);
        assert_eq!(kinds("!!= "), vec![TokKind::Not, TokKind::NotEq]);
        assert_eq!(kinds("!=="), vec![TokKind::NotEq, TokKind::Assign]);
        assert_eq!(kinds(":"), vec![TokKind::Colon]);
        assert_eq!(kinds(":::"), vec![TokKind::PathSep, TokKind::Colon]);
        assert_eq!(kinds("...."), vec![TokKind::DotDot, TokKind::DotDot]);
    }

    #[test]
    fn line_comments_skip_to_newline() {
        let src = "a // comment ! @ \" ignored\nb";
        assert_eq!(
            kinds(src),
            vec![TokKind::Ident("a".into()), TokKind::Ident("b".into())]
        );
        let toks = lex(src).expect("ok");
        assert_eq!(toks[1].span, Span::new(27, 28)); // `b` after the comment
    }

    #[test]
    fn block_comments_are_non_nested() {
        // The FIRST */ closes the comment; what follows is code, so the
        // trailing `*/` lexes as Star + Slash — proving no nesting happened.
        let toks = lex("a /* /* still comment */ b */ c").expect("non-nested");
        assert_eq!(
            kinds_of(toks),
            vec![
                TokKind::Ident("a".into()),
                TokKind::Ident("b".into()),
                TokKind::Star,
                TokKind::Slash,
                TokKind::Ident("c".into()),
            ]
        );

        let ok = lex("a /* /* look */ b").expect("single-level");
        assert_eq!(
            kinds_of(ok),
            vec![TokKind::Ident("a".into()), TokKind::Ident("b".into())]
        );
    }

    #[test]
    fn block_comment_content_is_skipped_entirely() {
        assert_eq!(
            kinds("/* fn let 123 **/ x"),
            vec![TokKind::Ident("x".into())]
        );
        assert_eq!(kinds("/*/ x */ y"), vec![TokKind::Ident("y".into())]);
    }

    #[test]
    fn block_comments_can_span_lines() {
        let src = "1 /* two\nlines\nhere */ 2";
        let toks = lex(src).expect("ok");
        assert_eq!(
            kinds_of(toks.clone()),
            vec![TokKind::Int(1), TokKind::Int(2)]
        );
        // The second int sits after the multi-line comment.
        assert_eq!(toks[1].span.start, 23);
        assert!(toks.last().expect("eof").is_eof());
    }

    #[test]
    fn unterminated_block_comment_errors_from_slash_to_eof() {
        let err = lex("a /* oops").expect_err("unterminated");
        assert_eq!(err.span.start, 2);
        assert_eq!(err.span.end, 9);
        assert!(err.msg.contains("unterminated"), "{}", err.msg);
        assert!(lex("/*").is_err());
        assert!(lex("/**").is_err());
    }

    #[test]
    fn unterminated_line_comment_at_eof_is_fine() {
        assert_eq!(
            kinds("let x = 1 // trailing"),
            vec![
                TokKind::Kw(Kw::Let),
                TokKind::Ident("x".into()),
                TokKind::Assign,
                TokKind::Int(1)
            ]
        );
        // Full stream (with spans) still ends in an Eof token.
        let toks = lex("let x = 1 // trailing").expect("ok");
        assert!(toks.last().expect("eof").is_eof());
    }

    #[test]
    fn unexpected_character_reports_span_and_char() {
        let err = lex("a $ b").expect_err("$");
        assert_eq!(err.span, Span::new(2, 3));
        assert!(err.msg.contains('$'), "{}", err.msg);

        let err2 = lex("#").expect_err("#");
        assert_eq!(err2.span, Span::new(0, 1));

        assert!(lex("a ? b").is_err());
        assert!(
            lex("\"no strings\"").is_err(),
            "strings do not exist in HELIX v1"
        );
        assert!(lex("x & y").is_err(), "lone `&` is not an operator");
        assert!(lex("x | y").is_err(), "lone `|` is not an operator");
    }

    #[test]
    fn ascii_only_identifiers_reject_high_bytes() {
        // Non-ASCII identifiers fall outside the frozen lexical grammar:
        // `caf` lexes, then the 2-byte `é` errors with a full-char span.
        let err = lex("café").expect_err("unicode ident");
        assert_eq!(err.span, Span::new(3, 5));
    }

    #[test]
    fn spans_are_byte_offsets_into_source() {
        let src = "let x = 42;";
        let toks = lex(src).expect("spans");
        assert_eq!(toks[0].span, Span::new(0, 3)); // `let`
        assert_eq!(toks[1].span, Span::new(4, 5)); // `x`
        assert_eq!(toks[2].span, Span::new(6, 7)); // `=`
        assert_eq!(toks[3].span, Span::new(8, 10)); // `42`
        assert_eq!(toks[4].span, Span::new(10, 11)); // `;`
        // Slice back through each span to recover the original text.
        for t in &toks {
            if let TokKind::Ident(name) = &t.kind {
                assert_eq!(
                    &src[t.span.start as usize..t.span.end as usize],
                    name.as_str()
                );
            }
        }
    }

    #[test]
    fn eof_token_sits_at_end_of_input() {
        let last = lex("abc").expect("eof pos").pop().expect("nonempty");
        assert_eq!(last.kind, TokKind::Eof);
        assert_eq!(last.span, Span::new(3, 3));
    }

    #[test]
    fn adjacent_tokens_without_spaces_split_correctly() {
        assert_eq!(
            kinds("a=b+c*2"),
            vec![
                TokKind::Ident("a".into()),
                TokKind::Assign,
                TokKind::Ident("b".into()),
                TokKind::Plus,
                TokKind::Ident("c".into()),
                TokKind::Star,
                TokKind::Int(2),
            ]
        );
    }

    #[test]
    fn a_full_program_lexes() {
        let src = r#"
            // demo kernel
            const N: i64 = 8;
            fn main() {
                let acc = 0.0;
                for i in 0..N {
                    acc = acc + i as f64;
                }
                print(acc);
            }
        "#;
        let toks = lex(src).expect("demo program");
        assert!(matches!(toks[0].kind, TokKind::Kw(Kw::Const)));
        assert!(toks.iter().any(|t| t.kind == TokKind::DotDot));
        assert!(toks.iter().any(|t| t.kind == TokKind::Kw(Kw::As)));
        // `fn main()` has no return arrow, but `->` never appears either —
        // check a genuinely present multi-char operator instead (`==`-free
        // demo uses `=` and `+`).
        assert!(toks.iter().any(|t| t.kind == TokKind::Plus));
        assert!(toks.last().expect("eof").is_eof());
    }

    /// Test helper: strip spans, dropping the trailing Eof.
    fn kinds_of(toks: Vec<Token>) -> Vec<TokKind> {
        toks.into_iter()
            .filter(|t| !t.is_eof())
            .map(|t| t.kind)
            .collect()
    }
}
