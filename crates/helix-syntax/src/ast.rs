//! The HELIX abstract syntax tree (frozen phrase grammar).
//!
//! One Rust type per EBNF production of `lang-spec.md`. Two invariants hold
//! for every node:
//!
//! 1. **Every node carries a [`Span`]** — either a field (`span`) or, for the
//!    tuple variants of [`Expr`], the trailing offset pair. Spans are byte
//!    offsets into the original source so diagnostics and runtime errors can
//!    quote the exact text the student wrote.
//! 2. **Everything serializes.** All types derive
//!    `serde::Serialize + serde::Deserialize` because the Observatory ships
//!    whole ASTs to the browser as JSON.
//!
//! Shape notes fixed by the spec:
//!
//! * A program is a *flat* list of items ([`Program::items`]) — no nested
//!   functions — which makes the module layout a plain symbol table.
//! * Assignment is a statement, not an expression: [`Stmt::Assign`] exists,
//!   no `Expr` variant does.
//! * Ranges exist only inside `for` headers: [`Stmt::For`] holds separate
//!   `start`/`end` expressions rather than a first-class range value.
//! * Braces are mandatory on every `if`/`for` body, so dangling-else cannot
//!   arise; `else if` chains are represented by nesting
//!   [`ElsePart::If`]`(`[`Stmt::If`]`)`.
//! * Casts bind tighter than unary operators, so `-x as i32` is parsed as
//!   `Neg(Cast(x))` — see the tests on [`crate::parser`].
//!
//! The tree is intentionally "dumb": no type information, no name resolution.
//! Those live in `helix-sema`, which consumes this tree via
//! `sema::check(&Program)`.

use serde::{Deserialize, Serialize};

use crate::token::Span;

/// A whole HELIX source file: function definitions plus scalar constants.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Program {
    /// Top-level items, in source order. The spec allows only functions and
    /// scalar constants here (no statements, no nested modules).
    pub items: Vec<Item>,
}

impl Program {
    /// The function definitions, in source order.
    pub fn fns(&self) -> impl Iterator<Item = &FnDef> {
        self.items.iter().filter_map(|i| match i {
            Item::Fn(f) => Some(f),
            Item::Const(_) => None,
        })
    }

    /// The constant definitions, in source order.
    pub fn consts(&self) -> impl Iterator<Item = &ConstDef> {
        self.items.iter().filter_map(|i| match i {
            Item::Const(c) => Some(c),
            Item::Fn(_) => None,
        })
    }
}

/// A top-level item: [`Item::Fn`] or [`Item::Const`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Item {
    /// `fn name(params) [-> type] block`
    Fn(FnDef),
    /// `const NAME: type = literal;`
    Const(ConstDef),
}

/// `fn name(param, ...) [-> ret] { body }`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FnDef {
    /// Function name.
    pub name: Ident,
    /// Declared parameters, in order.
    pub params: Vec<Param>,
    /// Declared return type; `None` means unit (`()`), i.e. a procedure.
    pub ret: Option<Type>,
    /// Mandatory body block.
    pub body: Block,
    /// Span covering the whole definition from `fn` to closing brace.
    pub span: Span,
}

/// One parameter: `name: type`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Param {
    /// Parameter name.
    pub name: Ident,
    /// Declared parameter type.
    pub ty: Type,
}

/// `const NAME: type = literal;` — a top-level compile-time scalar.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstDef {
    /// Constant name.
    pub name: Ident,
    /// Declared type (the literal must be assignable to it).
    pub ty: Type,
    /// Initialiser, restricted to a single literal by the grammar.
    pub value: Literal,
    /// Span covering the whole definition including the semicolon.
    pub span: Span,
}

/// A name with the span where it was written.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Ident {
    /// The identifier text (never a keyword; reserved words never get this
    /// far — the parser rejects them).
    pub name: String,
    /// Byte span of exactly the identifier characters.
    pub span: Span,
}

impl Ident {
    /// Convenience constructor used by tests and tooling.
    #[must_use]
    pub fn new(name: impl Into<String>, span: Span) -> Self {
        Self {
            name: name.into(),
            span,
        }
    }

    /// The identifier text as a slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.name
    }
}

/// A braced statement sequence `{ stmt* }`.
///
/// Blocks are mandatory for `if`/`for` bodies but also legal as bare
/// statements ([`Stmt::Block`]) introducing a nested scope.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Block {
    /// Statements in source order; possibly empty (`{}`).
    pub stmts: Vec<Stmt>,
    /// Span covering both braces.
    pub span: Span,
}

/// One statement of a block body.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Stmt {
    /// `let name [: type] = init;` — variable declaration.
    Let {
        /// Variable name.
        name: Ident,
        /// Optional annotation; absent means inferred from `init`.
        ty: Option<Type>,
        /// Initialiser expression (mandatory — definite assignment).
        init: Expr,
        /// Span from `let` through `;`.
        span: Span,
    },
    /// `target = value;` — assignment is a statement, never an expression.
    Assign {
        /// Assignment target: variable or array element.
        target: LValue,
        /// Right-hand side.
        value: Expr,
        /// Span from target through `;`.
        span: Span,
    },
    /// `if cond then_blk [else ...]` — braces mandatory, else optional.
    If {
        /// Condition; must type-check to `bool` later.
        cond: Expr,
        /// Mandatory `then` block.
        then_blk: Block,
        /// Optional `else`; `else if` chains nest as
        /// `Some(Box::new(ElsePart::If(Box::new(Stmt::If { .. }))))`.
        else_part: Option<Box<ElsePart>>,
        /// Span from `if` to the end of the taken alternatives.
        span: Span,
    },
    /// `for iv in start..end body` — half-open range `[start, end)`.
    For {
        /// Iteration variable (immutable within its loop per the spec).
        iv: Ident,
        /// Inclusive lower bound expression.
        start: Expr,
        /// Exclusive upper bound; boxed to keep the variant small.
        end: Box<Expr>,
        /// Mandatory body block.
        body: Block,
        /// Span from `for` to the end of the body block.
        span: Span,
    },
    /// `return;` or `return expr;`
    Return {
        /// Value expression; `None` only valid in unit functions.
        value: Option<Expr>,
        /// Span from `return` through `;`.
        span: Span,
    },
    /// A bare expression used as a statement (almost always a call such as
    /// `print(x);`).
    Expr(Expr),
    /// `;` — the empty statement.
    Empty,
    /// A nested block introducing a new scope.
    Block(Block),
}

impl Stmt {
    /// The span of any statement variant.
    #[must_use]
    pub fn span(&self) -> Span {
        match self {
            Stmt::Let { span, .. }
            | Stmt::Assign { span, .. }
            | Stmt::If { span, .. }
            | Stmt::For { span, .. }
            | Stmt::Return { span, .. } => *span,
            Stmt::Expr(e) => e.span(),
            Stmt::Empty => Span::new(0, 0),
            Stmt::Block(b) => b.span,
        }
    }
}

/// The two shapes an `else` clause may take.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ElsePart {
    /// `else if ...`: wraps a full if-statement so chains like
    /// `if a {} else if b {} else {}` become a right-nested spine.
    If(Box<Stmt>),
    /// `else { ... }`: an ordinary block.
    Block(Block),
}

/// An assignment target: `name` or `name[index]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LValue {
    /// Base variable being assigned.
    pub base: Ident,
    /// Index expression for element assignment (`a[i] = v`).
    pub index: Option<Expr>,
    /// Span covering the whole lvalue.
    pub span: Span,
}

/// Unary operators. HELIX has exactly these two.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UnOp {
    /// Numeric negation `-x`.
    Neg,
    /// Logical negation `!b` (bool only — no truthiness).
    Not,
}

impl UnOp {
    /// Source spelling of this operator.
    #[must_use]
    pub fn symbol(self) -> &'static str {
        match self {
            UnOp::Neg => "-",
            UnOp::Not => "!",
        }
    }
}

/// Binary operators, grouped by precedence tier (highest binding last in the
/// listing below; all tiers associate left):
///
/// `||` < `&&` < `== !=` < `< > <= >=` < `+ -` < `* / %`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BinOp {
    /// `+`
    Add,
    /// `-`
    Sub,
    /// `*`
    Mul,
    /// `/` (traps on zero divisor for integers)
    Div,
    /// `%` (truncated remainder, sign of dividend)
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
    Eq,
    /// `!=`
    Ne,
    /// `&&` short-circuiting and
    And,
    /// `||` short-circuiting or
    Or,
}

impl BinOp {
    /// Source spelling of this operator.
    #[must_use]
    pub fn symbol(self) -> &'static str {
        match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Rem => "%",
            BinOp::Lt => "<",
            BinOp::Gt => ">",
            BinOp::Le => "<=",
            BinOp::Ge => ">=",
            BinOp::Eq => "==",
            BinOp::Ne => "!=",
            BinOp::And => "&&",
            BinOp::Or => "||",
        }
    }

    /// Numeric precedence tier: larger binds tighter. Matches the frozen
    /// ladder `|| < && < eq < rel < add < mul`.
    #[must_use]
    pub fn precedence(self) -> u8 {
        match self {
            BinOp::Or => 1,
            BinOp::And => 2,
            BinOp::Eq | BinOp::Ne => 3,
            BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge => 4,
            BinOp::Add | BinOp::Sub => 5,
            BinOp::Mul | BinOp::Div | BinOp::Rem => 6,
        }
    }
}

/// A literal appearing in source. Used directly by `ConstDef` and wrapped by
/// [`Expr::IntLit`] etc. inside general expressions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Literal {
    /// Integer literal (fits `i64`; infers `i64` unless adapted to an `i32`
    /// context by sema).
    Int(i64),
    /// Floating-point literal.
    Float(f64),
    /// Boolean literal `true` / `false`.
    Bool(bool),
}

/// Expressions. Tuple variants carry `(payload…, start_offset, end_offset)`;
/// struct variants have an explicit `span` field. Either way
/// [`Expr::span`] always yields the node's extent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Expr {
    /// Integer literal with its parsed value and span.
    IntLit(i64, Span),
    /// Float literal with its parsed value and span.
    FloatLit(f64, Span),
    /// Boolean literal.
    Bool(bool, Span),
    /// Variable reference.
    Var(Ident),
    /// Unary application: `UnOp operand`.
    Unary(UnOp, Box<Expr>, Span),
    /// Binary application: `BinOp lhs rhs`, left-associative at parse time.
    Bin(BinOp, Box<Expr>, Box<Expr>, Span),
    /// Array indexing `name[index]` (base must be an `[T]` array).
    Index(Ident, Box<Expr>, Span),
    /// Function/builtin call.
    Call {
        /// Called function or builtin name.
        callee: Ident,
        /// Arguments in source order.
        args: Vec<Expr>,
        /// Span from callee name through `)`.
        span: Span,
    },
    /// Explicit numeric cast `expr as Type` (binds tighter than unary ops).
    Cast(Box<Expr>, Type, Span),
}

impl Expr {
    /// The byte span this expression was written at.
    #[must_use]
    pub fn span(&self) -> Span {
        match self {
            Expr::IntLit(_, s)
            | Expr::FloatLit(_, s)
            | Expr::Bool(_, s)
            | Expr::Unary(_, _, s)
            | Expr::Bin(_, _, _, s)
            | Expr::Index(_, _, s)
            | Expr::Cast(_, _, s) => *s,
            Expr::Var(id) => id.span,
            Expr::Call { span, .. } => *span,
        }
    }
}

/// A HELIX type as written in source: annotations, parameters, return types
/// and cast targets.
///
/// Note the asymmetry required by the spec: arrays may appear anywhere a
/// type is written, but their element type is restricted to scalars —
/// nested arrays are unrepresentable, which is enforced *by construction*
/// ([`Type::Array`] holds a [`ScalarType`], not another [`Type`]).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Type {
    /// `i32` — storage/array-element integer.
    I32,
    /// `i64` — the arithmetic integer type.
    I64,
    /// `f32` — single-precision float.
    F32,
    /// `f64` — double-precision float.
    F64,
    /// `bool`
    Bool,
    /// `[T]` where T is a scalar type (flat arrays only).
    Array(ScalarType),
    /// `()` — unit, the return type of procedures.
    Unit,
}

/// The scalar subset of [`Type`]: what may sit inside `[T]` or be targeted
/// by an `as` cast.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ScalarType {
    /// `i32`
    I32,
    /// `i64`
    I64,
    /// `f32`
    F32,
    /// `f64`
    F64,
    /// `bool`
    Bool,
}

impl ScalarType {
    /// Source spelling.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            ScalarType::I32 => "i32",
            ScalarType::I64 => "i64",
            ScalarType::F32 => "f32",
            ScalarType::F64 => "f64",
            ScalarType::Bool => "bool",
        }
    }

    /// Parses a scalar type from its spelling (used by the parser after it
    /// has matched an identifier-shaped token).
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "i32" => Some(ScalarType::I32),
            "i64" => Some(ScalarType::I64),
            "f32" => Some(ScalarType::F32),
            "f64" => Some(ScalarType::F64),
            "bool" => Some(ScalarType::Bool),
            _ => None,
        }
    }
}

impl Type {
    /// Source spelling, including brackets for arrays.
    #[must_use]
    pub fn render(&self) -> String {
        match self {
            Type::I32 => "i32".to_string(),
            Type::I64 => "i64".to_string(),
            Type::F32 => "f32".to_string(),
            Type::F64 => "f64".to_string(),
            Type::Bool => "bool".to_string(),
            Type::Array(el) => format!("[{}]", el.name()),
            Type::Unit => "()".to_string(),
        }
    }

    /// The scalar view of this type, if it is one.
    #[must_use]
    pub fn as_scalar(&self) -> Option<ScalarType> {
        match self {
            Type::I32 => Some(ScalarType::I32),
            Type::I64 => Some(ScalarType::I64),
            Type::F32 => Some(ScalarType::F32),
            Type::F64 => Some(ScalarType::F64),
            Type::Bool => Some(ScalarType::Bool),
            _ => None,
        }
    }

    /// Whether this is the unit type `()`.
    #[must_use]
    pub fn is_unit(&self) -> bool {
        matches!(self, Type::Unit)
    }
}

impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.render())
    }
}

impl std::fmt::Display for Literal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Literal::Int(v) => write!(f, "{v}"),
            Literal::Float(v) => write!(f, "{v:?}"),
            Literal::Bool(b) => write!(f, "{b}"),
        }
    }
}
