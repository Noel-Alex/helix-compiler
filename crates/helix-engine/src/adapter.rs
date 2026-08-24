//! From frontend artifacts to the engine's evaluation tree.
//!
//! ## Why the engine has its own tree
//!
//! Two facts shape this module:
//!
//! 1. `helix-sema` v1's [`TypedExprKind::Call`] records *which* callee a call
//!    resolves to ([`CallTarget::User`] / [`CallTarget::Builtin`]) but not
//!    the argument sub-expressions: the checker type-checks each argument for
//!    diagnostics and then keeps only the callee identity. An evaluator
//!    cannot work without the arguments.
//! 2. Runtime errors must quote a **source line**, but the typed tree only
//!    carries byte-offset spans — the original text is needed to number them.
//!
//! Both are solved the same way: take the *source*, parse it again, and walk
//! the AST against the [`TypedProgram`] that sema produced from it. The typed
//! program stays the semantic authority — its symbol arenas (kinds + types,
//! allocated sequentially in source-walk order), const table and function
//! signatures decide everything — while the AST supplies the structure sema
//! dropped. The result is the engine's own evaluation tree ([`EExpr`] et al.),
//! identical to the typed tree except that calls carry arguments.
//!
//! ## Why replaying name resolution is safe
//!
//! Sema allocates symbol ids sequentially during one forward walk per
//! function (shared consts prefix, then params, then locals/loop vars in
//! encounter order). Replaying the same walk in the same order and claiming
//! symbols in encounter order reproduces every id exactly — including under
//! shadowing, because a shadowing `let x` always claims the *next* unclaimed
//! local named `x`. Scope pushes/pops mirror sema's stack one-to-one, so
//! identifier resolution cannot diverge on programs sema accepted.
//!
//! On any inconsistency (mismatched pair of trees, poisoned symbol) the join
//! fails with `None`; callers report a clean error instead of guessing. It is
//! unreachable for a `TypedProgram` freshly checked from the same source.

use std::collections::HashMap;

use helix_sema::{
    Builtin, ElemTy, SymId, SymKind, Symbol, Ty, TypedConstDef, TypedFnDef, TypedProgram,
};
use helix_syntax::Span;
use helix_syntax::ast::{BinOp, ElsePart, Expr, FnDef, Item, Program, Stmt, Type as SynType, UnOp};

// ---------------------------------------------------------------------------
// Engine evaluation tree (EIR)
// ---------------------------------------------------------------------------

/// A statement sequence.
#[derive(Debug, Clone)]
pub struct EBlock {
    /// Statements in source order.
    pub stmts: Vec<EStmt>,
}

/// One executable statement.
#[derive(Debug, Clone)]
pub enum EStmt {
    /// `let sym = init;` — binds in the current scope.
    Let {
        /// Symbol being defined (already claimed from the arena).
        sym: SymId,
        /// Initialiser, evaluated before binding.
        init: EExpr,
    },
    /// `base = value;` or `base[index] = value;`.
    Assign {
        /// Target variable (array handle for indexed stores).
        base: SymId,
        /// Element index, present iff storing into an array.
        index: Option<Box<EExpr>>,
        /// Right-hand side.
        value: EExpr,
        /// Span of the whole statement (error location for bad stores).
        span: Span,
    },
    /// `if cond {} else ...` with optional else-if chain.
    If {
        /// Condition (bool by construction).
        cond: EExpr,
        /// Taken when the condition holds.
        then_blk: EBlock,
        /// Optional alternative.
        else_arm: Option<EElse>,
    },
    /// `for iv in start..end {}` — half-open, bounds evaluated once.
    For {
        /// Induction variable symbol (bound fresh each iteration).
        iv: SymId,
        /// Inclusive lower bound.
        start: Box<EExpr>,
        /// Exclusive upper bound.
        end: Box<EExpr>,
        /// Body.
        body: EBlock,
        /// Span of the header (error location inside the loop).
        span: Span,
    },
    /// `return;` / `return e;`.
    Return {
        /// Value, absent for procedures.
        value: Option<EExpr>,
        /// Span of the statement.
        span: Span,
    },
    /// Expression evaluated for effects only (`print(..);`, calls).
    Effect(EExpr),
    /// A bare `{ .. }` statement: a nested scope whose statements DO run
    /// (sema v1 models this as an inert effect, but the engine keeps the
    /// statements so side effects are faithful to the grammar).
    Nested(EBlock),
}

/// The `else` side of an [`EStmt::If`].
#[derive(Debug, Clone)]
pub enum EElse {
    /// `else if ...` chains nest.
    If(Box<EIf>),
    /// `else { ... }`.
    Block(EBlock),
}

/// Condition + branches, shared by `if` statements and else-if spines.
#[derive(Debug, Clone)]
pub struct EIf {
    /// Condition.
    pub cond: EExpr,
    /// Then-block.
    pub then_blk: EBlock,
    /// Optional else.
    pub else_arm: Option<EElse>,
}

/// One expression, fully typed and located.
#[derive(Debug, Clone)]
pub struct EExpr {
    /// Static type — decided here, trusted by the evaluator.
    pub ty: Ty,
    /// Source span (runtime errors quote this).
    pub span: Span,
    /// Shape.
    pub kind: EExprKind,
}

/// Expression shapes. Mirrors [`helix_sema::TypedExprKind`] plus one crucial
/// difference: [`EExprKind::Call`] carries its argument sub-trees.
#[derive(Debug, Clone)]
pub enum EExprKind {
    /// Integer literal (width fixed by `ty`: i32 in i32 slots, else i64).
    IntLit(i64),
    /// Float literal (f32 in f32 slots, else f64).
    FloatLit(f64),
    /// Boolean literal.
    BoolLit(bool),
    /// Variable read (arrays never appear as plain values).
    Var(SymId),
    /// Bare array name in argument position — a reference, not a copy.
    ArrayRef(SymId),
    /// `-x` / `!b`.
    Unary(UnOp, Box<EExpr>),
    /// Binary operator; `&&`/`||` short-circuit in the evaluator.
    Bin(BinOp, Box<EExpr>, Box<EExpr>),
    /// Array read `base[index]` (bounds-checked at run time).
    Index(SymId, Box<EExpr>),
    /// Call with arguments attached — the node sema v1 could not store.
    Call(ETarget),
    /// Numeric cast; saturating float→int, truncating int→int.
    Cast(Box<EExpr>, Ty),
}

/// Resolved callee plus evaluated-in-order arguments.
#[derive(Debug, Clone)]
pub enum ETarget {
    /// One of the seven builtins.
    Builtin(Builtin, Vec<EExpr>),
    /// A user function by index (= position in [`AdaptedProgram::funcs`]).
    User {
        /// Callee index.
        idx: u32,
        /// Arguments, evaluated left to right by the evaluator.
        args: Vec<EExpr>,
    },
}

/// A joined, ready-to-run program.
#[derive(Debug, Clone)]
pub struct AdaptedProgram {
    /// Functions in [`TypedProgram::funcs`] order; indices are stable.
    pub funcs: Vec<AdaptedFn>,
    /// Top-level consts, bound into every frame at entry.
    pub consts: Vec<TypedConstDef>,
}

/// One ready-to-run function.
#[derive(Debug, Clone)]
pub struct AdaptedFn {
    /// Position in [`AdaptedProgram::funcs`].
    pub idx: u32,
    /// Function name.
    pub name: String,
    /// Parameter symbols and types, in declaration order.
    pub params: Vec<(SymId, Ty)>,
    /// Return type (`Ty::Unit` for procedures).
    pub ret: Ty,
    /// Executable body.
    pub body: EBlock,
    /// Symbol arena copied verbatim from sema (name/kind/type per [`SymId`]).
    pub symbols: Vec<Symbol>,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Joins a syntax [`Program`] with the [`TypedProgram`] checked from it.
///
/// Returns `None` when the two do not correspond (different sources, or the
/// typed program predates an edit). Unreachable in normal operation because
/// callers pass a freshly parsed + checked pair.
#[must_use]
pub fn adapt_program(ast: &Program, tp: &TypedProgram) -> Option<AdaptedProgram> {
    let ast_fns: Vec<&FnDef> = ast
        .items
        .iter()
        .filter_map(|i| match i {
            Item::Fn(f) => Some(f),
            Item::Const(_) => None,
        })
        .collect();
    if ast_fns.len() != tp.funcs.len() {
        return None;
    }

    // Program-wide signature table: HELIX allows mutual recursion, so callee
    // lookup must not depend on definition order.
    let sigs: HashMap<&str, (u32, Vec<Ty>, Ty)> = tp
        .funcs
        .iter()
        .enumerate()
        .map(|(i, f)| {
            (
                f.name.as_str(),
                (i as u32, f.params.iter().map(|(_, t)| *t).collect(), f.ret),
            )
        })
        .collect();

    let mut funcs = Vec::with_capacity(tp.funcs.len());
    for (tf, af) in tp.funcs.iter().zip(&ast_fns) {
        if tf.name != af.name.name {
            return None;
        }
        let mut ad = Adapter::new(tf, &sigs);
        let body = ad.block(&af.body)?;
        funcs.push(AdaptedFn {
            idx: funcs.len() as u32,
            name: tf.name.clone(),
            params: tf.params.clone(),
            ret: tf.ret,
            body,
            symbols: tf.symbols.clone(),
        });
    }
    Some(AdaptedProgram {
        funcs,
        consts: tp.consts.clone(),
    })
}

// ---------------------------------------------------------------------------
// The per-function adapter
// ---------------------------------------------------------------------------

type Scope = HashMap<String, SymId>;

/// Walks one function's AST, replaying sema's declarations and resolutions.
struct Adapter<'p> {
    /// Function being adapted (its arena is the symbol universe).
    cur: &'p TypedFnDef,
    /// name -> (fn index, param types, ret type) for every function.
    sigs: &'p HashMap<&'p str, (u32, Vec<Ty>, Ty)>,
    /// Sema-style scope stack; root scope holds consts + params.
    scopes: Vec<Scope>,
    /// Lowest symbol id not yet claimed by this replay.
    next_unclaimed: u32,
}

impl<'p> Adapter<'p> {
    fn new(cur: &'p TypedFnDef, sigs: &'p HashMap<&'p str, (u32, Vec<Ty>, Ty)>) -> Self {
        let mut scopes = vec![Scope::new()];
        let mut next_unclaimed = 0u32;
        // Replay the root scope exactly as sema built it: consts occupy the
        // arena prefix shared by all functions, params follow.
        for (i, s) in cur.symbols.iter().enumerate() {
            if matches!(s.kind, SymKind::Const) {
                scopes[0].insert(s.name.clone(), SymId(i as u32));
                next_unclaimed = next_unclaimed.max(i as u32 + 1);
            }
        }
        for (id, _) in &cur.params {
            if let Some(s) = cur.symbols.get(id.0 as usize) {
                scopes[0].insert(s.name.clone(), *id);
            }
            next_unclaimed = next_unclaimed.max(id.0 + 1);
        }
        Self {
            cur,
            sigs,
            scopes,
            next_unclaimed,
        }
    }

    // -- scope plumbing -------------------------------------------------------

    fn resolve(&self, name: &str) -> Option<SymId> {
        self.scopes
            .iter()
            .rev()
            .find_map(|sc| sc.get(name).copied())
    }

    fn ty_of(&self, id: SymId) -> Option<Ty> {
        self.cur.symbols.get(id.0 as usize).map(|s| s.ty)
    }

    fn push_scope(&mut self) {
        self.scopes.push(Scope::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    /// Finds the arena slot sema allocated for a `let` at this site — the
    /// first unclaimed Local matching the name (and, when annotated, the
    /// declared type) — without binding it yet.
    ///
    /// Not binding immediately matters: sema checks `let x = <init>` before
    /// declaring `x`, so the init's `x` refers to the OUTER shadowee.
    fn peek_local(&self, name: &str, ann: Option<&SynType>) -> Option<SymId> {
        let want_ann = ann.map(ann_ty);
        for i in self.next_unclaimed..self.cur.symbols.len() as u32 {
            let s = &self.cur.symbols[i as usize];
            if !matches!(s.kind, SymKind::Local) || s.name != name {
                continue;
            }
            if let Some(want) = want_ann
                && want != s.ty
            {
                continue;
            }
            return Some(SymId(i));
        }
        None
    }

    /// Binds a previously peeked local in the current scope.
    fn bind_local(&mut self, name: &str, id: SymId) {
        self.next_unclaimed = self.next_unclaimed.max(id.0 + 1);
        self.scopes
            .last_mut()
            .expect("scope stack is never empty")
            .insert(name.to_string(), id);
    }

    /// Enters the loop-variable scope, claiming the induction variable.
    fn enter_loop_var(&mut self, name: &str) -> Option<SymId> {
        let mut found = None;
        for i in self.next_unclaimed..self.cur.symbols.len() as u32 {
            let s = &self.cur.symbols[i as usize];
            if matches!(s.kind, SymKind::LoopVar) && s.name == name {
                found = Some(SymId(i));
                break;
            }
        }
        let id = found?;
        self.push_scope();
        self.bind_local(name, id);
        Some(id)
    }

    // -- statements -------------------------------------------------------------

    fn block(&mut self, b: &helix_syntax::ast::Block) -> Option<EBlock> {
        self.push_scope();
        let mut stmts = Vec::with_capacity(b.stmts.len());
        for s in &b.stmts {
            stmts.push(self.stmt(s)?);
        }
        self.pop_scope();
        Some(EBlock { stmts })
    }

    #[allow(clippy::too_many_lines)] // one exhaustive match over the grammar
    fn stmt(&mut self, s: &Stmt) -> Option<EStmt> {
        match s {
            Stmt::Let { name, ty, init, .. } => {
                // Peek first (binding must NOT be visible to the init), take
                // the arena's type as truth, adapt the init towards it, then
                // bind — sema's exact ordering.
                let cand = self.peek_local(&name.name, ty.as_ref())?;
                let want = self.ty_of(cand)?;
                let init_e = self.expr(init, Some(want))?;
                self.bind_local(&name.name, cand);
                Some(EStmt::Let {
                    sym: cand,
                    init: init_e,
                })
            }
            Stmt::Assign { target, value, .. } => {
                let base_sym = self.resolve(&target.base.name)?;
                let base_ty = self.ty_of(base_sym)?;
                match (&target.index, base_ty.elem()) {
                    (None, None) => {
                        let v = self.expr(value, Some(base_ty))?;
                        Some(EStmt::Assign {
                            base: base_sym,
                            index: None,
                            value: v,
                            span: target.span,
                        })
                    }
                    (Some(ix), Some(_)) => {
                        // Index position accepts i32 implicitly widened; typing
                        // the index towards i64 reproduces sema's rule.
                        let idx_e = self.expr(ix, Some(Ty::I64))?;
                        let elem_ty = self.ty_of(base_sym)?.elem().map(elem_scalar_ty)?;
                        let v = self.expr(value, Some(elem_ty))?;
                        Some(EStmt::Assign {
                            base: base_sym,
                            index: Some(Box::new(idx_e)),
                            value: v,
                            span: target.span,
                        })
                    }
                    _ => None,
                }
            }
            Stmt::If {
                cond,
                then_blk,
                else_part,
                ..
            } => {
                let cond_e = self.expr(cond, Some(Ty::Bool))?;
                let then_t = self.block(then_blk)?;
                let else_arm = match else_part.as_deref() {
                    None => None,
                    Some(ElsePart::If(inner)) => match self.stmt(inner)? {
                        EStmt::If {
                            cond,
                            then_blk,
                            else_arm,
                        } => Some(EElse::If(Box::new(EIf {
                            cond,
                            then_blk,
                            else_arm,
                        }))),
                        // The grammar makes an else-if always parse as an if.
                        _ => return None,
                    },
                    Some(ElsePart::Block(b)) => Some(EElse::Block(self.block(b)?)),
                };
                Some(EStmt::If {
                    cond: cond_e,
                    then_blk: then_t,
                    else_arm,
                })
            }
            Stmt::For {
                iv,
                start,
                end,
                body,
                ..
            } => {
                // Bounds first, then the induction variable, then the body —
                // the order sema allocated symbols in.
                let start_e = self.expr(start, Some(Ty::I64))?;
                let end_e = self.expr(end, Some(Ty::I64))?;
                let iv_sym = self.enter_loop_var(&iv.name)?;
                let body_t = self.block(body)?;
                self.pop_scope();
                Some(EStmt::For {
                    iv: iv_sym,
                    start: Box::new(start_e),
                    end: Box::new(end_e),
                    body: body_t,
                    span: iv.span,
                })
            }
            Stmt::Return { value, span } => {
                let want = if self.cur.ret == Ty::Unit {
                    None
                } else {
                    Some(self.cur.ret)
                };
                let value = match value {
                    None => None,
                    Some(v) => Some(self.expr(v, want)?),
                };
                Some(EStmt::Return { value, span: *span })
            }
            Stmt::Expr(e) => Some(EStmt::Effect(self.expr(e, None)?)),
            // `;` contributes nothing.
            Stmt::Empty => Some(EStmt::Effect(inert(Span { start: 0, end: 0 }))),
            // Nested `{ .. }` statement: a real scope whose statements run.
            Stmt::Block(b) => {
                let inner = self.block(b)?;
                Some(EStmt::Nested(inner))
            }
        }
    }

    // -- expressions --------------------------------------------------------------

    /// Adapts an AST expression to [`EExpr`], mirroring sema's bidirectional
    /// rules for everything the evaluator observes: literal width adaptation,
    /// index widening, short-circuit operand typing.
    #[allow(clippy::too_many_lines)]
    fn expr(&mut self, e: &Expr, expected: Option<Ty>) -> Option<EExpr> {
        match e {
            Expr::IntLit(v, span) => Some(EExpr {
                ty: if matches!(expected, Some(Ty::I32)) {
                    Ty::I32
                } else {
                    Ty::I64
                },
                span: *span,
                kind: EExprKind::IntLit(*v),
            }),
            Expr::FloatLit(v, span) => {
                let ty = if matches!(expected, Some(Ty::F32)) {
                    Ty::F32
                } else {
                    Ty::F64
                };
                Some(EExpr {
                    ty,
                    span: *span,
                    kind: EExprKind::FloatLit(*v),
                })
            }
            Expr::Bool(b, span) => Some(EExpr {
                ty: Ty::Bool,
                span: *span,
                kind: EExprKind::BoolLit(*b),
            }),
            Expr::Var(ident) => {
                // Sema rejected bare array values, so a resolved Var is scalar.
                let id = self.resolve(&ident.name)?;
                Some(EExpr {
                    ty: self.ty_of(id)?,
                    span: ident.span,
                    kind: EExprKind::Var(id),
                })
            }
            Expr::Unary(op, operand, span) => {
                // Sema checks unary operands WITHOUT the expected hint (so
                // `let x: i32 = -5;` is rejected upstream); reproduce that.
                let o = self.expr(operand, None)?;
                Some(EExpr {
                    ty: o.ty,
                    span: *span,
                    kind: EExprKind::Unary(*op, Box::new(o)),
                })
            }
            Expr::Bin(op, l, rr, span) => {
                let l_t = self.expr(l, None)?;
                // Comparisons require equal operand types, so the right side
                // is typed towards the left's type; logical operands are bool.
                let want_r = match op {
                    BinOp::And | BinOp::Or => Ty::Bool,
                    _ => l_t.ty,
                };
                let r_t = self.expr(rr, Some(want_r))?;
                let ty = match op {
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem => l_t.ty,
                    BinOp::Lt
                    | BinOp::Gt
                    | BinOp::Le
                    | BinOp::Ge
                    | BinOp::Eq
                    | BinOp::Ne
                    | BinOp::And
                    | BinOp::Or => Ty::Bool,
                };
                Some(EExpr {
                    ty,
                    span: *span,
                    kind: EExprKind::Bin(*op, Box::new(l_t), Box::new(r_t)),
                })
            }
            Expr::Index(base_ident, idx, span) => {
                let id = self.resolve(&base_ident.name)?;
                let elem = self.ty_of(id)?.elem()?;
                let idx_e = self.expr(idx, Some(Ty::I64))?;
                Some(EExpr {
                    ty: elem_scalar_ty(elem),
                    span: *span,
                    kind: EExprKind::Index(id, Box::new(idx_e)),
                })
            }
            Expr::Cast(operand, to_syn, span) => {
                let o = self.expr(operand, None)?;
                let to = cast_target(to_syn)?;
                Some(EExpr {
                    ty: to,
                    span: *span,
                    kind: EExprKind::Cast(Box::new(o), to),
                })
            }
            Expr::Call { callee, args, span } => self.call(callee, args, *span, expected),
        }
    }

    /// Adapts a call site: builtin or user, attaching adapted arguments.
    fn call(
        &mut self,
        callee: &helix_syntax::ast::Ident,
        args: &[Expr],
        span: Span,
        expected: Option<Ty>,
    ) -> Option<EExpr> {
        // Builtins shadow user names in sema, so try them first.
        if let Some(b) = Builtin::from_name(&callee.name) {
            let (ret, args_e) = self.builtin(b, args, expected)?;
            return Some(EExpr {
                ty: ret,
                span,
                kind: EExprKind::Call(ETarget::Builtin(b, args_e)),
            });
        }
        let (idx, params, ret) = self.sigs.get(callee.name.as_str())?;
        let (idx, params, ret) = (*idx, params.clone(), *ret);
        if params.len() != args.len() {
            return None;
        }
        let mut args_e = Vec::with_capacity(args.len());
        for (a, w) in args.iter().zip(&params) {
            args_e.push(self.arg(a, *w)?);
        }
        Some(EExpr {
            ty: ret,
            span,
            kind: EExprKind::Call(ETarget::User { idx, args: args_e }),
        })
    }

    /// Builtin argument/result typing per the spec table. Returns the result
    /// type plus the adapted argument nodes (needed at run time).
    fn builtin(
        &mut self,
        b: Builtin,
        args: &[Expr],
        expected: Option<Ty>,
    ) -> Option<(Ty, Vec<EExpr>)> {
        match b {
            Builtin::Print => {
                let a = self.one(args)?;
                Some((Ty::Unit, vec![self.expr(a, None)?]))
            }
            Builtin::Zeros => {
                // Element type comes from context (annotated binding / array
                // parameter); sema rejects uninferrable sites.
                let ty = expected.filter(Ty::is_array)?;
                let a = self.one(args)?;
                Some((ty, vec![self.expr(a, Some(Ty::I64))?]))
            }
            Builtin::Len => {
                let a = self.one(args)?;
                // len(a) takes a bare array NAME -> ArrayRef node. Its `ty`
                // field is the array's real element-carrying type.
                let Expr::Var(ident) = &a else { return None };
                let id = self.resolve(&ident.name)?;
                let arr_ty = self.ty_of(id)?;
                if !arr_ty.is_array() {
                    return None;
                }
                Some((
                    Ty::I64,
                    vec![EExpr {
                        ty: arr_ty,
                        span: ident.span,
                        kind: EExprKind::ArrayRef(id),
                    }],
                ))
            }
            Builtin::Abs => {
                let a = self.one(args)?;
                let e = self.expr(a, None)?;
                let ty = e.ty;
                Some((ty, vec![e]))
            }
            Builtin::Sqrt => {
                let a = self.one(args)?;
                let e = self.expr(a, None)?;
                if !e.ty.is_float() {
                    return None;
                }
                Some((e.ty, vec![e]))
            }
            Builtin::Min | Builtin::Max => {
                if args.len() != 2 {
                    return None;
                }
                let a0 = self.expr(&args[0], None)?;
                let ty0 = a0.ty;
                let a1 = self.expr(&args[1], Some(ty0))?;
                if a1.ty != ty0 {
                    return None;
                }
                Some((ty0, vec![a0, a1]))
            }
        }
    }

    fn one<'a>(&self, args: &'a [Expr]) -> Option<&'a Expr> {
        match args {
            [a] => Some(a),
            _ => None,
        }
    }

    /// Adapts one user-call argument: a bare array name becomes an ArrayRef
    /// (pass-by-reference); anything else is typed towards the param type.
    fn arg(&mut self, a: &Expr, want: Ty) -> Option<EExpr> {
        if let Expr::Var(ident) = a {
            let id = self.resolve(&ident.name)?;
            let t = self.ty_of(id)?;
            if t.is_array() {
                return if t == want {
                    Some(EExpr {
                        ty: t,
                        span: ident.span,
                        kind: EExprKind::ArrayRef(id),
                    })
                } else {
                    None
                };
            }
        }
        self.expr(a, Some(want))
    }
}

/// An inert effect node standing in for `;` — typed, located, no-op.
fn inert(span: Span) -> EExpr {
    EExpr {
        ty: Ty::Unit,
        span,
        kind: EExprKind::BoolLit(false), // never evaluated: Effects discard the value
    }
}

// ---------------------------------------------------------------------------
// Small conversions shared with the evaluator
// ---------------------------------------------------------------------------

/// Syntactic type -> semantic type (arrays included).
#[must_use]
pub fn ann_ty(t: &SynType) -> Ty {
    match t {
        SynType::I32 => Ty::I32,
        SynType::I64 => Ty::I64,
        SynType::F32 => Ty::F32,
        SynType::F64 => Ty::F64,
        SynType::Bool => Ty::Bool,
        SynType::Array(e) => Ty::Array(match e {
            helix_syntax::ScalarType::I32 => ElemTy::I32,
            helix_syntax::ScalarType::I64 => ElemTy::I64,
            helix_syntax::ScalarType::F32 => ElemTy::F32,
            helix_syntax::ScalarType::F64 => ElemTy::F64,
            helix_syntax::ScalarType::Bool => ElemTy::Bool,
        }),
        SynType::Unit => Ty::Unit,
    }
}

/// Cast target type; `None` for targets sema would have rejected.
fn cast_target(t: &SynType) -> Option<Ty> {
    match t {
        SynType::I32 => Some(Ty::I32),
        SynType::I64 => Some(Ty::I64),
        SynType::F32 => Some(Ty::F32),
        SynType::F64 => Some(Ty::F64),
        _ => None,
    }
}

/// Scalar view of an array element type.
#[must_use]
pub fn elem_scalar_ty(e: ElemTy) -> Ty {
    match e {
        ElemTy::I32 => Ty::I32,
        ElemTy::I64 => Ty::I64,
        ElemTy::F32 => Ty::F32,
        ElemTy::F64 => Ty::F64,
        ElemTy::Bool => Ty::Bool,
    }
}
