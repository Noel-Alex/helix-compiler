//! The HELIX parser: tokens → [`Program`] via recursive descent plus a Pratt
//! loop for expressions.
//!
//! # Design
//!
//! Statements and declarations are parsed with classic *recursive descent*:
//! one function per EBNF production (`parse_stmt`, `parse_if_stmt`,
//! `parse_for_stmt`, …), mirroring the grammar in `lang-spec.md` line for
//! line so the two can be read side by side.
//!
//! Expressions use a single *Pratt (precedence-climbing) loop*
//! ([`Parser::parse_binary`]) driven by [`BinOp::precedence`]. This replaces
//! the spec's seven chained productions (`or_expr` … `mul_expr`) with one
//! table-driven function that produces exactly the same trees, because
//! left-associativity is encoded as "loop, always fold to the left" and the
//! ladder `|| < && < eq < rel < add < mul` is just the precedence table.
//!
//! Levels above binary operators are layered back into recursion:
//!
//! ```text
//! parse_expr      → parse_binary(min precedence 1)
//! parse_binary    → unary { binop(unary) }*        (Pratt, left-assoc)
//! parse_unary     → (- | !) parse_unary | parse_cast
//! parse_cast      → parse_postfix { as scalar_type }*
//! parse_postfix   → parse_primary { [expr] | (args) }
//! parse_primary   → literal | ident | ( expr )
//! ```
//!
//! Two grammar decisions deserve emphasis:
//!
//! * **Assignment is a statement.** The expression parser never accepts
//!   `=`; an assignment is only recognised at statement position by
//!   lookahead (`IDENT [= | [`) in [`Parser::parse_stmt`]. This makes
//!   `a = b = c` and `if x = y` syntax errors by construction.
//! * **Ranges live only in `for` headers.** There is no range expression;
//!   [`Stmt::For`] consumes the `..` token directly.
//! * **Braces are mandatory** on `if`/`for` bodies, which removes
//!   dangling-else: after `else` we simply parse either another if-statement
//!   ([`ElsePart::If`]) or a block ([`ElsePart::Block`]).

use serde::{Deserialize, Serialize};

use crate::ast::{
    BinOp, Block, ConstDef, ElsePart, Expr, FnDef, Ident, Item, LValue, Literal, Param, Program,
    ScalarType, Stmt, Type, UnOp,
};
use crate::token::{Kw, Span, TokKind, Token, is_reserved};

/// A syntax error with its span and a human-readable message of the form
/// ``expected X, found Y``.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParseError {
    /// Where the problem was detected.
    pub span: Span,
    /// Human-readable explanation.
    pub msg: String,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "syntax error at {}: {}", self.span, self.msg)
    }
}

impl std::error::Error for ParseError {}

/// Parses a full token stream (as produced by [`crate::lex`], ending in
/// `Eof`) into a [`Program`].
pub fn parse(tokens: Vec<Token>) -> Result<Program, ParseError> {
    Parser::new(tokens).parse_program()
}

/// Convenience wrapper: lex + parse in one call.
///
/// # Errors
///
/// Forwards the first [`crate::lexer::LexError`] or [`ParseError`].
pub fn parse_source(src: &str) -> Result<Program, crate::syntax_error::SyntaxError> {
    let tokens = crate::lex(src).map_err(crate::syntax_error::SyntaxError::Lex)?;
    parse(tokens).map_err(crate::syntax_error::SyntaxError::Parse)
}

/// Internal cursor over the token vector.
struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    /// Expression nesting depth. Every recursive expression level (parens,
    /// unary operators) costs one unit; exceeding [`MAX_EXPR_DEPTH`] produces
    /// a clean ParseError instead of a native stack-overflow abort. The cap
    /// also bounds the recursive tree printer and derived `Drop` downstream.
    depth: u32,
    /// Statement nesting depth: each nested block, if/else arm, or for body
    /// costs one unit of [`MAX_BLOCK_DEPTH`], same rationale as the
    /// expression cap — bounded recursion in parser, printer, and Drop.
    block_depth: u32,
}

/// Maximum expression nesting depth. Counts parenthesis levels, unary
/// operators, AND binary right-operand recursion (an alternating-precedence
/// chain like `1+2*3+4%5…` descends one `parse_binary` frame per increase).
/// Generous for human code (~30 in practice); sized so even fat debug-build
/// frames (~10 KB per level across the six-level descent chain) stay within
/// the 2 MiB stack of a tokio worker thread.
const MAX_EXPR_DEPTH: u32 = 128;

/// Maximum statement nesting depth. Counts nested blocks (`{ … }`), if/else
/// chains (each `else if` recurses through [`Parser::parse_if_stmt`]), and
/// for bodies. Far above any human or generated program (~10 in practice);
/// sized like [`MAX_EXPR_DEPTH`] so even debug-build frames stay far inside
/// the 2 MiB stack of a tokio worker thread.
const MAX_BLOCK_DEPTH: u32 = 512;

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            pos: 0,
            depth: 0,
            block_depth: 0,
        }
    }

    /// Enters one deeper expression-nesting level; errors past the cap.
    fn enter_expr(&mut self, open: Span) -> Result<(), ParseError> {
        self.depth += 1;
        if self.depth > MAX_EXPR_DEPTH {
            return Err(ParseError {
                span: open,
                msg: format!(
                    "expression nesting too deep (limit {MAX_EXPR_DEPTH} levels of parentheses/unary operators)"
                ),
            });
        }
        Ok(())
    }

    fn leave_expr(&mut self) {
        self.depth = self.depth.saturating_sub(1);
    }

    /// Enters one deeper statement-nesting level; errors past the cap.
    fn enter_block(&mut self, open: Span) -> Result<(), ParseError> {
        self.block_depth += 1;
        if self.block_depth > MAX_BLOCK_DEPTH {
            return Err(ParseError {
                span: open,
                msg: format!("block nesting too deep (limit {MAX_BLOCK_DEPTH} levels)"),
            });
        }
        Ok(())
    }

    fn leave_block(&mut self) {
        self.block_depth = self.block_depth.saturating_sub(1);
    }

    // -- token cursor helpers ------------------------------------------------

    fn peek(&self) -> &Token {
        &self.tokens[self.pos.min(self.tokens.len() - 1)]
    }

    fn peek_kind(&self) -> &TokKind {
        &self.peek().kind
    }

    fn advance(&mut self) -> Token {
        let tok = self.peek().clone();
        if !tok.is_eof() {
            self.pos += 1;
        }
        tok
    }

    fn prev_span(&self) -> Span {
        self.tokens[self.pos.saturating_sub(1)].span
    }

    fn peek_span(&self) -> Span {
        self.peek().span
    }

    /// Whether the next token has kind `k`.
    fn at(&self, k: &TokKind) -> bool {
        std::mem::discriminant(self.peek_kind()) == std::mem::discriminant(k)
    }

    fn at_kw(&self, k: Kw) -> bool {
        matches!(self.peek_kind(), TokKind::Kw(actual) if *actual == k)
    }

    /// Consumes the next token if it has kind `k`.
    fn eat(&mut self, k: &TokKind) -> bool {
        if self.at(k) {
            self.advance();
            true
        } else {
            false
        }
    }

    /// Consumes keyword `k` if next.
    fn eat_kw(&mut self, k: Kw) -> bool {
        if self.at_kw(k) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn error<T>(&self, expected: &str) -> Result<T, ParseError> {
        Err(self.error_at(self.peek_span(), expected))
    }

    fn error_at(&self, span: Span, expected: &str) -> ParseError {
        ParseError {
            span,
            msg: format!("expected {expected}, found {}", self.peek().kind.describe()),
        }
    }

    /// Consumes kind `k` or fails with `expected <what>`.
    fn expect(&mut self, k: TokKind, what: &str) -> Result<Token, ParseError> {
        if self.at(&k) {
            Ok(self.advance())
        } else {
            self.error(what)
        }
    }

    fn expect_kw(&mut self, k: Kw) -> Result<Token, ParseError> {
        if self.at_kw(k) {
            Ok(self.advance())
        } else {
            self.error(&format!("`{k}`"))
        }
    }

    // -- program -------------------------------------------------------------

    fn parse_program(&mut self) -> Result<Program, ParseError> {
        let mut items = Vec::new();
        while !self.peek().is_eof() {
            items.push(self.parse_item()?);
        }
        Ok(Program { items })
    }

    fn parse_item(&mut self) -> Result<Item, ParseError> {
        if self.at_kw(Kw::Fn) {
            Ok(Item::Fn(self.parse_fn_def()?))
        } else if self.at_kw(Kw::Const) {
            Ok(Item::Const(self.parse_const_def()?))
        } else {
            self.error("`fn` or `const` at top level")
        }
    }

    fn parse_fn_def(&mut self) -> Result<FnDef, ParseError> {
        let start = self.expect_kw(Kw::Fn)?.span;

        let name = match self.peek_kind() {
            TokKind::Ident(name) => {
                if is_reserved(name) {
                    return Err(ParseError {
                        span: self.peek_span(),
                        msg: format!("reserved word `{name}` cannot be used as a function name"),
                    });
                }
                self.ident()
            }
            _ => return self.error("function name"),
        };

        self.expect(TokKind::LParen, "`(`")?;

        let mut params = Vec::new();
        if !self.at(&TokKind::RParen) {
            params.push(self.parse_param()?);
            // Strict per spec: `param {"," param}` — no trailing comma.
            while self.eat(&TokKind::Comma) {
                params.push(self.parse_param()?);
            }
        }
        self.expect(TokKind::RParen, "`)` or parameter list")?;

        let ret = if self.eat(&TokKind::Arrow) {
            Some(self.parse_type()?)
        } else {
            None
        };

        let body = self.parse_block()?;
        let span = Span::join(start, body.span);
        Ok(FnDef {
            name,
            params,
            ret,
            body,
            span,
        })
    }

    fn parse_param(&mut self) -> Result<Param, ParseError> {
        let name = self.parse_ident_strict("parameter name")?;
        self.expect(TokKind::Colon, "`:` after parameter name")?;
        let ty = self.parse_type()?;
        Ok(Param { name, ty })
    }

    fn parse_const_def(&mut self) -> Result<ConstDef, ParseError> {
        let start = self.expect_kw(Kw::Const)?.span;
        let name = self.parse_ident_strict("constant name")?;
        self.expect(TokKind::Colon, "`:` after constant name")?;
        let ty = self.parse_type()?;
        self.expect(TokKind::Assign, "`=` in const definition")?;
        let value = self.parse_literal("const initialiser must be a literal")?;
        // A literal is already consumed here; anything but `;` means the
        // programmer wrote an expression (`const N: i64 = 1 + 2;`).
        let end_tok = self.expect(
            TokKind::Semi,
            "`;` after const definition (initialisers are single literals)",
        )?;
        Ok(ConstDef {
            name,
            ty,
            value,
            span: Span::join(start, end_tok.span),
        })
    }

    // -- types ----------------------------------------------------------------

    /// Parses a type: scalar, `[scalar]`, or `()`. Types are written with
    /// identifier-shaped spellings (`i32`, `bool`, …) except unit.
    fn parse_type(&mut self) -> Result<Type, ParseError> {
        if self.eat(&TokKind::LBracket) {
            let el = self.parse_scalar_type("[T] array element type")?;
            self.expect(TokKind::RBracket, "`]` closing array type")?;
            return Ok(Type::Array(el));
        }
        if self.at(&TokKind::LParen)
            && matches!(
                self.tokens.get(self.pos + 1).map(|t| &t.kind),
                Some(TokKind::RParen)
            )
        {
            // `()` — unit type. Types carry no spans in the AST contract, so
            // both tokens are simply consumed.
            self.advance(); // (
            self.advance(); // )
            return Ok(Type::Unit);
        }
        let el = self.parse_scalar_type("type annotation (e.g. `i64`, `[f64]`, `()`)")?;
        Ok(Type::from_scalar(el))
    }

    fn parse_scalar_type(&mut self, what: &str) -> Result<ScalarType, ParseError> {
        match self.peek_kind().clone() {
            TokKind::Ident(name) => {
                if let Some(sc) = ScalarType::from_name(&name) {
                    self.advance();
                    Ok(sc)
                } else {
                    self.error(what)
                }
            }
            _ => self.error(what),
        }
    }

    // -- statements -----------------------------------------------------------

    fn parse_block(&mut self) -> Result<Block, ParseError> {
        let open = self.expect(TokKind::LBrace, "`{`")?;
        self.enter_block(open.span)?;
        let mut stmts = Vec::new();
        while !self.at(&TokKind::RBrace) && !self.peek().is_eof() {
            stmts.push(self.parse_stmt()?);
        }
        self.leave_block();
        let close = self.expect(TokKind::RBrace, "`}` or statement")?;
        Ok(Block {
            stmts,
            span: Span::join(open.span, close.span),
        })
    }

    fn parse_stmt(&mut self) -> Result<Stmt, ParseError> {
        // Empty statement first: `;`
        if self.at(&TokKind::Semi) {
            self.advance();
            return Ok(Stmt::Empty);
        }
        // Bare nested block introduces a scope.
        if self.at(&TokKind::LBrace) {
            return Ok(Stmt::Block(self.parse_block()?));
        }
        match self.peek_kind().clone() {
            TokKind::Kw(Kw::Let) => self.parse_let_stmt(),
            TokKind::Kw(Kw::If) => self.parse_if_stmt(),
            TokKind::Kw(Kw::For) => self.parse_for_stmt(),
            TokKind::Kw(Kw::Return) => self.parse_return_stmt(),
            // Assignment vs expression statement disambiguated by lookahead:
            // IDENT `=` or IDENT `[` expr `]` `=`.
            TokKind::Ident(_) => {
                if self.next_starts_assignment() {
                    self.parse_assign_stmt()
                } else {
                    self.parse_expr_stmt()
                }
            }
            _ => self.parse_expr_stmt(),
        }
    }

    /// `expr ";"` — a bare expression statement (typically a call).
    fn parse_expr_stmt(&mut self) -> Result<Stmt, ParseError> {
        let expr = self.parse_expr()?;
        self.expect(TokKind::Semi, "`;` after expression")?;
        Ok(Stmt::Expr(expr))
    }

    /// One-token lookahead past the identifier: does an assignment follow?
    fn next_starts_assignment(&self) -> bool {
        let after_ident = self.pos + 1;
        match self.tokens.get(after_ident).map(|t| &t.kind) {
            Some(TokKind::Assign) => true,
            Some(TokKind::LBracket) => {
                // `name[expr] = ...`: scan for the matching `]` then check
                // whether `=` follows it.
                let mut depth = 0usize;
                let mut i = after_ident;
                while i < self.tokens.len() {
                    match &self.tokens[i].kind {
                        TokKind::LBracket => depth += 1,
                        TokKind::RBracket => {
                            depth -= 1;
                            if depth == 0 {
                                return matches!(
                                    self.tokens.get(i + 1).map(|t| &t.kind),
                                    Some(TokKind::Assign)
                                );
                            }
                        }
                        TokKind::Eof => return false,
                        _ => {}
                    }
                    i += 1;
                }
                false
            }
            _ => false,
        }
    }

    fn parse_let_stmt(&mut self) -> Result<Stmt, ParseError> {
        let start = self.expect_kw(Kw::Let)?.span;
        let name = self.parse_ident_strict("variable name after `let`")?;
        let ty = if self.eat(&TokKind::Colon) {
            Some(self.parse_type()?)
        } else {
            None
        };
        self.expect(TokKind::Assign, "`=` in let statement")?;
        let init = self.parse_expr()?;
        let end = self.expect(TokKind::Semi, "`;` after let statement")?;
        Ok(Stmt::Let {
            name,
            ty,
            init,
            span: Span::join(start, end.span),
        })
    }

    fn parse_assign_stmt(&mut self) -> Result<Stmt, ParseError> {
        let target = self.parse_lvalue()?;
        self.expect(TokKind::Assign, "`=` in assignment")?;
        let value = self.parse_expr()?;
        let end = self.expect(TokKind::Semi, "`;` after assignment")?;
        let span = Span::join(target.span, end.span);
        Ok(Stmt::Assign {
            target,
            value,
            span,
        })
    }

    /// `lvalue ::= IDENT ['[' expr ']']`
    fn parse_lvalue(&mut self) -> Result<LValue, ParseError> {
        let base = self.parse_ident_strict("assignment target")?;
        let index = if self.eat(&TokKind::LBracket) {
            let idx = self.parse_expr()?;
            self.expect(TokKind::RBracket, "`]` closing index")?;
            Some(idx)
        } else {
            None
        };
        let end_span = index.as_ref().map_or(base.span, Expr::span);
        Ok(LValue {
            base,
            index,
            span: end_span,
        })
    }

    /// `if_stmt ::= 'if' expr block {'else' (if_stmt | block)}`
    ///
    /// Only ONE else clause is consumed here; chains grow because the else
    /// clause may itself be an if-statement ([`ElsePart::If`]).
    fn parse_if_stmt(&mut self) -> Result<Stmt, ParseError> {
        let start = self.expect_kw(Kw::If)?.span;
        // Charge one level per if-statement: the `else if` chain recurses
        // here (one frame per arm), and each arm's block adds its own level
        // via parse_block.
        self.enter_block(start)?;
        let result = self.parse_if_rest(start);
        self.leave_block();
        result
    }

    /// Continuation of [`Parser::parse_if_stmt`] with the depth already
    /// entered (so every early-return path stays balanced).
    fn parse_if_rest(&mut self, start: Span) -> Result<Stmt, ParseError> {
        let cond = self.parse_expr()?;
        let then_blk = self.parse_block()?;

        let mut span = Span::join(start, then_blk.span);
        let mut else_part = None;
        if self.eat_kw(Kw::Else) {
            if self.at_kw(Kw::If) {
                let nested = self.parse_if_stmt()?;
                span = Span::join(span, nested.span());
                else_part = Some(Box::new(ElsePart::If(Box::new(nested))));
            } else if self.at(&TokKind::LBrace) {
                let blk = self.parse_block()?;
                span = Span::join(span, blk.span);
                else_part = Some(Box::new(ElsePart::Block(blk)));
            } else {
                return self.error("`if` or `{` after `else` (braces are mandatory)");
            }
        }
        Ok(Stmt::If {
            cond,
            then_blk,
            else_part,
            span,
        })
    }

    /// `for_stmt ::= 'for' IDENT 'in' expr '..' expr block`
    fn parse_for_stmt(&mut self) -> Result<Stmt, ParseError> {
        let start = self.expect_kw(Kw::For)?.span;
        let iv = self.parse_ident_strict("iteration variable after `for`")?;
        self.expect_kw(Kw::In)?;
        let start_e = self.parse_expr()?;
        self.expect(
            TokKind::DotDot,
            "`..` between loop bounds (ranges only appear in for headers)",
        )?;
        let end_e = self.parse_expr()?;
        let body = self.parse_block()?;
        let span = Span::join(start, body.span);
        Ok(Stmt::For {
            iv,
            start: start_e,
            end: Box::new(end_e),
            body,
            span,
        })
    }

    fn parse_return_stmt(&mut self) -> Result<Stmt, ParseError> {
        let start = self.expect_kw(Kw::Return)?.span;
        // A return value is present unless the statement ends here.
        let value = if self.at(&TokKind::Semi) || self.at(&TokKind::RBrace) || self.peek().is_eof()
        {
            None
        } else {
            Some(self.parse_expr()?)
        };
        let end = self.expect(TokKind::Semi, "`;` after return")?;
        Ok(Stmt::Return {
            value,
            span: Span::join(start, end.span),
        })
    }

    // -- expressions ------------------------------------------------------------

    fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        self.parse_binary(1)
    }

    /// Pratt / precedence-climbing core. All binary levels associate LEFT:
    /// each iteration folds the freshly parsed right operand into the
    /// accumulated left tree.
    fn parse_binary(&mut self, min_prec: u8) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_unary()?;
        loop {
            let op = match self.peek_kind() {
                TokKind::OrOr => Some(BinOp::Or),
                TokKind::AndAnd => Some(BinOp::And),
                TokKind::EqEq => Some(BinOp::Eq),
                TokKind::NotEq => Some(BinOp::Ne),
                TokKind::Lt => Some(BinOp::Lt),
                TokKind::Gt => Some(BinOp::Gt),
                TokKind::Le => Some(BinOp::Le),
                TokKind::Ge => Some(BinOp::Ge),
                TokKind::Plus => Some(BinOp::Add),
                TokKind::Minus => Some(BinOp::Sub),
                TokKind::Star => Some(BinOp::Mul),
                TokKind::Slash => Some(BinOp::Div),
                TokKind::Rem => Some(BinOp::Rem),
                _ => None,
            };
            let Some(op) = op else { break };
            let prec = op.precedence();
            if prec < min_prec {
                break;
            }
            // A precedence INCREASE recurses into parse_binary for the right
            // operand (one frame per increase); charge that against the same
            // depth budget so `1+2*3+4%5…` chains cannot outgrow the stack.
            let rising = prec + 1 > min_prec;
            if rising {
                self.enter_expr(self.peek_span())?;
            }
            self.advance(); // consume operator
            let rhs = self.parse_binary(prec + 1);
            if rising {
                self.leave_expr();
            }
            let rhs = rhs?;
            let span = Span::join(lhs.span(), rhs.span());
            lhs = Expr::Bin(op, Box::new(lhs), Box::new(rhs), span);
        }
        Ok(lhs)
    }

    /// `unary ::= ('-'|'!') unary | cast` — binds LOOSER than `as`, so
    /// `-x as i32` parses as `Neg(Cast(x))` per the frozen spec.
    fn parse_unary(&mut self) -> Result<Expr, ParseError> {
        let op = match self.peek_kind() {
            TokKind::Minus => Some(UnOp::Neg),
            TokKind::Not => Some(UnOp::Not),
            _ => None,
        };
        if let Some(op) = op {
            let start = self.advance().span;
            self.enter_expr(start)?;
            let inner = self.parse_unary();
            self.leave_expr();
            let inner = inner?;
            let span = Span::join(start, inner.span());
            return Ok(Expr::Unary(op, Box::new(inner), span));
        }
        self.parse_cast()
    }

    /// `cast ::= postfix {'as' scalar_type}` — left-associative chaining of
    /// casts, e.g. `i as i64 as f64`.
    fn parse_cast(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_postfix()?;
        while self.eat_kw(Kw::As) {
            // Casts target SCALAR types only (arrays/unit excluded per spec).
            let ty =
                self.parse_scalar_type("scalar type after `as` (casts exclude arrays and `()`)")?;
            let span = Span::join(expr.span(), self.prev_span());
            expr = Expr::Cast(Box::new(expr), Type::from_scalar(ty), span);
        }
        Ok(expr)
    }

    /// `postfix ::= primary ['[' expr ']'] | primary '(' [expr {',' expr}] ')'`
    fn parse_postfix(&mut self) -> Result<Expr, ParseError> {
        // Indexing applies only to bare identifiers (`a[i]`) and calls only
        // to names (`f(x)`); HELIX has no method calls, no nested arrays and
        // no function values, so chained postfix (`a[i][j]`, `f(x)(y)`,
        // `f(x)[0]`) is rejected with a targeted message.
        let expr = match self.peek_kind().clone() {
            TokKind::Ident(_) => {
                let name = self.ident();
                if self.eat(&TokKind::LBracket) {
                    let index = self.parse_expr()?;
                    let close = self.expect(TokKind::RBracket, "`]` closing index")?;
                    let span = Span::join(name.span, close.span);
                    Expr::Index(name, Box::new(index), span)
                } else if self.eat(&TokKind::LParen) {
                    let args = self.parse_call_args()?;
                    let close = self.expect(TokKind::RParen, "`)` or argument list")?;
                    let span = Span::join(name.span, close.span);
                    Expr::Call {
                        callee: name,
                        args,
                        span,
                    }
                } else {
                    Expr::Var(name)
                }
            }
            _ => self.parse_primary()?,
        };

        // A second postfix segment can never be valid (no nested arrays, no
        // function values), so reject it here with a targeted message rather
        // than letting it surface as a confusing "expected `;`" downstream.
        if self.at(&TokKind::LBracket) || self.at(&TokKind::LParen) {
            return Err(ParseError {
                span: self.peek_span(),
                msg: "only one call or index level is allowed and only on a plain name \
                      (no chained `a[i][j]`, `f(x)(y)` or `f(x)[0]` in HELIX)"
                    .to_string(),
            });
        }
        Ok(expr)
    }

    /// Argument list after the opening `(` (the `(` itself already eaten).
    fn parse_call_args(&mut self) -> Result<Vec<Expr>, ParseError> {
        let mut args = Vec::new();
        if !self.at(&TokKind::RParen) {
            args.push(self.parse_expr()?);
            while self.eat(&TokKind::Comma) {
                args.push(self.parse_expr()?);
            }
        }
        Ok(args)
    }

    /// `primary ::= INT_LIT | FLOAT_LIT | 'true' | 'false' | IDENT | '(' expr ')'`
    fn parse_primary(&mut self) -> Result<Expr, ParseError> {
        let tok = self.peek().clone();
        match tok.kind {
            TokKind::Int(v) => {
                self.advance();
                Ok(Expr::IntLit(v, tok.span))
            }
            TokKind::Float(v) => {
                self.advance();
                Ok(Expr::FloatLit(v, tok.span))
            }
            TokKind::Kw(Kw::True) => {
                self.advance();
                Ok(Expr::Bool(true, tok.span))
            }
            TokKind::Kw(Kw::False) => {
                self.advance();
                Ok(Expr::Bool(false, tok.span))
            }
            TokKind::Ident(ref name) => {
                if is_reserved(name) {
                    Err(ParseError {
                        span: tok.span,
                        msg: format!("reserved word `{name}` cannot be used as a name"),
                    })
                } else {
                    Ok(Expr::Var(self.ident()))
                }
            }
            TokKind::LParen => {
                self.advance();
                self.enter_expr(tok.span)?;
                let inner = self.parse_expr();
                self.leave_expr();
                let inner = inner?;
                self.expect(TokKind::RParen, "`)` to close parenthesised expression")?;
                Ok(inner) // parentheses group; no Paren node in the AST
            }
            _ => self.error("expression (literal, variable, parenthesised expression or call)"),
        }
    }

    // -- small helpers ---------------------------------------------------------

    /// Takes the current token as an identifier without re-checking.
    fn ident(&mut self) -> Ident {
        let tok = self.advance();
        match tok.kind {
            TokKind::Ident(name) => Ident::new(name, tok.span),
            other => unreachable!("ident() called on non-identifier token {other:?}"),
        }
    }

    /// Takes an identifier, rejecting reserved words with a clear message.
    fn parse_ident_strict(&mut self, what: &str) -> Result<Ident, ParseError> {
        match self.peek_kind() {
            TokKind::Ident(name) if is_reserved(name) => {
                let msg = format!("reserved word `{name}` cannot be used as {what}");
                let span = self.peek_span();
                Err(ParseError { span, msg })
            }
            TokKind::Ident(_) => Ok(self.ident()),
            _ => self.error(what),
        }
    }

    /// Parses a standalone literal (used by `const` initialisers).
    ///
    /// Strictly per the frozen grammar, `literal` has no negative form —
    /// there is no `-` case here, so `const N: i64 = -5;` is rejected. This
    /// keeps the parser an exact transcription of the EBNF.
    fn parse_literal(&mut self, what: &str) -> Result<Literal, ParseError> {
        let tok = self.peek().clone();
        match tok.kind {
            TokKind::Int(v) => {
                self.advance();
                Ok(Literal::Int(v))
            }
            TokKind::Float(v) => {
                self.advance();
                Ok(Literal::Float(v))
            }
            TokKind::Kw(Kw::True) | TokKind::Kw(Kw::False) => {
                let v = matches!(tok.kind, TokKind::Kw(Kw::True));
                self.advance();
                Ok(Literal::Bool(v))
            }
            _ => Err(ParseError {
                span: tok.span,
                msg: format!(
                    "{what}, found {} (const initialisers are literal-only; \
                     expressions are not allowed)",
                    tok.kind.describe()
                ),
            }),
        }
    }
}

// -- private AST plumbing ---------------------------------------------------

impl Type {
    /// Wraps a scalar as a full type.
    fn from_scalar(sc: ScalarType) -> Type {
        match sc {
            ScalarType::I32 => Type::I32,
            ScalarType::I64 => Type::I64,
            ScalarType::F32 => Type::F32,
            ScalarType::F64 => Type::F64,
            ScalarType::Bool => Type::Bool,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_ok(src: &str) -> Program {
        crate::parse_str(src).unwrap_or_else(|e| panic!("expected parse success: {e}"))
    }

    #[test]
    fn deeply_nested_parens_error_cleanly_not_stack_overflow() {
        // Regression: 100k paren levels used to overflow the native stack and
        // abort the process. The depth cap turns it into a ParseError.
        let mut src = String::from("fn main() { let x = ");
        src.push_str(&"(".repeat(100_000));
        src.push('1');
        src.push_str(&")".repeat(100_000));
        src.push_str("; }");
        let err = crate::parse_str(&src).expect_err("deep nesting must be rejected, not crash");
        assert!(
            err.to_string().contains("nesting too deep"),
            "wrong error: {err}"
        );
    }

    #[test]
    fn deeply_nested_unary_ops_error_cleanly() {
        let mut src = String::from("fn main() { let x = ");
        src.push_str(&"!".repeat(50_000));
        src.push_str("true; }");
        let err = crate::parse_str(&src).expect_err("deep unary chain must be rejected, not crash");
        assert!(err.to_string().contains("nesting too deep"));
    }

    #[test]
    fn legal_nesting_well_below_cap_still_parses() {
        // ~100 depth units of parens + unary mix: ordinary machine-generated
        // code stays comfortably inside the cap.
        let mut expr = String::new();
        for i in 0..100 {
            if i % 2 == 0 {
                expr.push('(');
            } else {
                expr.push('-');
            }
        }
        expr.push('1');
        for _ in 0..50 {
            expr.push(')');
        }
        let src = format!("fn main() {{ let x = {expr}; }}");
        parse_ok(&src);
    }

    #[test]
    fn normal_expressions_unaffected_by_depth_tracking() {
        parse_ok("fn main() { let x = -(1 + 2) * -(3 - 4); print(x); }");
    }

    // -- statement nesting depth (blocks, if/else chains) -------------------

    fn nested_blocks(depth: usize) -> String {
        let mut src = String::from("fn main() { ");
        for _ in 0..depth {
            src.push_str("{ let x = 1; ");
        }
        for _ in 0..depth {
            src.push('}');
        }
        src.push_str(" }");
        src
    }

    #[test]
    fn deeply_nested_blocks_error_cleanly_not_stack_overflow() {
        // Regression: 100k `{ … }` levels recursed through parse_block and
        // overflowed the native stack; now a clean ParseError.
        let err = crate::parse_str(&nested_blocks(100_000))
            .expect_err("deep block nesting must be rejected, not crash");
        assert!(
            err.to_string().contains("block nesting too deep"),
            "wrong error: {err}"
        );
    }

    #[test]
    fn deeply_nested_else_if_chain_errors_cleanly() {
        // Each `else if` recurses through parse_if_stmt; a long chain must
        // hit the same cap as blocks instead of overflowing the stack.
        let mut src = String::from("fn main() { if true { print(1); }");
        for _ in 1..100_000 {
            src.push_str(" else if false { print(2); }");
        }
        src.push_str(" else { print(3); } }");
        let err =
            crate::parse_str(&src).expect_err("deep else-if chain must be rejected, not crash");
        assert!(err.to_string().contains("block nesting too deep"));
    }

    #[test]
    fn legal_block_nesting_well_below_cap_still_parses() {
        // ~100 levels of mixed blocks + ifs: ordinary machine-generated code.
        parse_ok(&nested_blocks(100));
        let mut src = String::from("fn main() { ");
        for i in 0..50 {
            src.push_str(&format!("if {i} < 100 {{ print({i}); }} else {{ }} "));
        }
        src.push('}');
        parse_ok(&src);
    }

    #[test]
    fn normal_statements_unaffected_by_block_depth_tracking() {
        parse_ok(
            "fn main() { for i in 0..3 { if i > 1 { print(i); } else { print(-i); } } \
             { let y = 2; print(y); } }",
        );
    }
}
