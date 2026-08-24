//! # helix-syntax — lexer, parser and AST for the HELIX language
//!
//! This crate turns HELIX source text (see the frozen `lang-spec.md`) into a
//! serializable abstract syntax tree. It is stage 1 of the compiler pipeline:
//!
//! ```text
//! source ──lex──▶ Vec<Token> ──parse──▶ Program ──▶ helix-sema …
//! ```
//!
//! ## Module map
//!
//! * [`token`] — [`Span`](token::Span) byte ranges plus [`Token`]/[`TokKind`]
//!   definitions, keywords and the reserved-word list.
//! * [`lexer`] — hand-written scanner; [`lex`] produces the token stream.
//! * [`ast`] — the phrase grammar as Rust types; every node carries a span
//!   and everything derives `serde::Serialize + Deserialize`.
//! * [`parser`] — recursive descent + Pratt precedence climbing; [`parse`]
//!   builds the [`Program`].
//! * [`tree`] — [`Program::print_tree`] for human-readable dumps.
//!
//! ## Design notes (course-report material)
//!
//! **Lexer.** A single forward pass over the source bytes keeps only a byte
//! offset as state, which makes spans trivially correct: a token's span is
//! just `(start_of_match, pos_after_match)`. Recognition is by first
//! character, with maximal munch applied inside each class — longest word
//! wins (`iff` is an identifier, not `if` + `f`) and two-character operators
//! are tested before their one-character prefixes (`<=` before `<`, `&&`
//! before `&`). Comments are skipped in the same trivia loop as whitespace,
//! and block comments do **not** nest (C-style, deliberately unlike Rust).
//!
//! **Parser.** Statements use plain recursive descent — one function per EBNF
//! production, so the code reads like the grammar appendix. Expressions use
//! one Pratt loop driven by [`BinOp::precedence`] instead of seven chained
//! `*_expr` productions; left-associativity falls out of folding each new
//! operand into the accumulated left-hand tree. Levels above binary operators
//! recurse: unary → cast → postfix → primary, which encodes the frozen rule
//! that `-x as i32` means `-(x as i32)` while `-a * b` means `(-a) * b`.
//! Assignment is parsed at *statement* level only (lookahead on
//! `IDENT =` / `IDENT [expr] =`), making `a = b = c` and `if x = y`
//! unrepresentable. Ranges exist solely inside `for` headers, and mandatory
//! braces remove dangling-else entirely; `else if` chains nest as
//! `ElsePart::If(Box<Stmt::If>)`.
//!
//! **Errors.** Both phases report errors as `{span, msg}` structs rather than
//! panicking, per the shared convention; [`SyntaxError`] unifies them for
//! callers that just want "first error, with location".

pub mod ast;
pub mod lexer;
pub mod parser;
pub mod token;
pub mod tree;

/// Convenience module bundling both error types behind one enum.
pub mod syntax_error {
    /// Any failure of the syntax phase.
    #[derive(Debug, Clone, PartialEq)]
    pub enum SyntaxError {
        /// Tokenizer failure.
        Lex(super::LexError),
        /// Parser failure.
        Parse(super::ParseError),
    }

    impl std::fmt::Display for SyntaxError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                SyntaxError::Lex(e) => write!(f, "{e}"),
                SyntaxError::Parse(e) => write!(f, "{e}"),
            }
        }
    }

    impl std::error::Error for SyntaxError {}
}

pub use syntax_error::SyntaxError;

// -- Core contract types -----------------------------------------------------

pub use crate::ast::{
    BinOp, Block, ConstDef, ElsePart, Expr, FnDef, Ident, Item, LValue, Literal, Param, Program,
    ScalarType, Stmt, Type, UnOp,
};
pub use crate::lexer::{LexError, lex};
pub use crate::parser::{ParseError, parse};
pub use crate::token::{Kw, Span, TokKind, Token};

/// Lexes and parses `src` in one call.
///
/// The standard entry point used by `helix-sema` and the CLI.
///
/// # Errors
///
/// Returns the first lexical or syntactic error found, with its span.
pub fn parse_str(src: &str) -> Result<Program, SyntaxError> {
    parser::parse_source(src)
}
