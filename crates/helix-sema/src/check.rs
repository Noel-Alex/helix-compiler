//! The semantic checker.
//!
//! Three independent analyses, in order:
//! 1. **Name resolution + type checking** (this module's `check`) — builds symbol
//!    tables, resolves identifiers, enforces typing rules with bidirectional literal
//!    adaptation, and produces the [`TypedProgram`].
//! 2. **Definite assignment** ([`init_analysis`]) — forward dataflow over the typed
//!    tree with proper branch merging (intersection at joins).
//! 3. **All-paths-return** ([`TypedStmt::always_returns`]) — structural check that
//!    value-returning functions cannot fall off the end.
//!
//! Splitting 2 and 3 out keeps each pass simple enough to explain in the course report.

use std::collections::{HashMap, HashSet};

use helix_syntax::ast::{
    BinOp, Block, ConstDef, ElsePart, Expr, FnDef, Ident, Item, LValue, Literal, Program, Stmt,
    Type as SynType, UnOp,
};
use helix_syntax::{ScalarType, Span};
use serde::{Deserialize, Serialize};

use crate::types::{ElemTy, Ty};

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SemDiag {
    pub span: Span,
    pub msg: String,
}

fn diag(span: Span, msg: impl Into<String>) -> SemDiag {
    SemDiag {
        span,
        msg: msg.into(),
    }
}

/// Sentinel id for symbols whose declaration failed; never resolved again because the
/// name never entered scope.
pub(crate) const POISON: SymId = SymId(u32::MAX);

// ---------------------------------------------------------------------------
// Public typed surface
// ---------------------------------------------------------------------------

/// The seven builtins (lang-spec.md).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Builtin {
    Print,
    Zeros,
    Len,
    Abs,
    Sqrt,
    Min,
    Max,
}

impl Builtin {
    pub fn name(self) -> &'static str {
        match self {
            Builtin::Print => "print",
            Builtin::Zeros => "zeros",
            Builtin::Len => "len",
            Builtin::Abs => "abs",
            Builtin::Sqrt => "sqrt",
            Builtin::Min => "min",
            Builtin::Max => "max",
        }
    }

    pub fn from_name(name: &str) -> Option<Builtin> {
        Some(match name {
            "print" => Builtin::Print,
            "zeros" => Builtin::Zeros,
            "len" => Builtin::Len,
            "abs" => Builtin::Abs,
            "sqrt" => Builtin::Sqrt,
            "min" => Builtin::Min,
            "max" => Builtin::Max,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SymKind {
    /// Function parameter — initialized on entry.
    Param,
    /// Top-level constant — always initialized.
    Const,
    /// `let` binding — governed by definite assignment.
    Local,
    /// `for` induction variable — assignment forbidden (affine iteration space).
    LoopVar,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Symbol {
    pub name: String,
    pub ty: Ty,
    pub kind: SymKind,
    pub span: Span,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SymId(pub u32);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TypedProgram {
    pub funcs: Vec<TypedFnDef>,
    pub consts: Vec<TypedConstDef>,
}

impl TypedProgram {
    /// Index of `fn main()` (checked to exist).
    pub fn main_idx(&self) -> usize {
        self.funcs
            .iter()
            .position(|f| f.name == "main")
            .expect("sema guarantees main")
    }

    pub fn find_fn(&self, name: &str) -> Option<usize> {
        self.funcs.iter().position(|f| f.name == name)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TypedConstDef {
    pub name: String,
    pub ty: Ty,
    pub sym: SymId,
    pub value: ConstLit,
    pub span: Span,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ConstLit {
    Int(i64),
    Float(f64),
    Bool(bool),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TypedFnDef {
    pub name: String,
    pub params: Vec<(SymId, Ty)>,
    pub ret: Ty,
    pub body: TypedBlock,
    /// Per-function symbol arena (consts prefix + params + locals). Indices are SymIds.
    pub symbols: Vec<Symbol>,
    pub span: Span,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TypedBlock {
    pub stmts: Vec<TypedStmt>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TypedStmt {
    Let {
        sym: SymId,
        ty: Ty,
        init: TypedExpr,
    },
    Assign {
        target: TypedLValue,
        value: TypedExpr,
    },
    If(Box<TypedIf>),
    For(Box<TypedFor>),
    Return {
        value: Option<TypedExpr>,
        span: Span,
    },
    /// Bare nested `{ .. }` block statement: a real scope whose statements
    /// execute in order. Kept structurally so every consumer of the typed tree
    /// (IR lowering, all-paths-return, definite assignment) sees the same
    /// statements the reference interpreter runs.
    Nested(TypedBlock),
    /// Expression evaluated for effects only (calls, short-circuit operands).
    Effect(TypedExpr),
    /// A bare `;`. Contributes nothing — kept as its own variant so the
    /// poison `TypedExprKind::Error` sentinel is reserved for actual
    /// error-recovery nodes (a legal production must never masquerade as one;
    /// see the 2026-08-25 review wave's "bare blocks" lesson).
    Empty,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TypedIf {
    pub cond: TypedExpr,
    pub then_blk: TypedBlock,
    pub else_arm: Option<ElseArm>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ElseArm {
    If(Box<TypedIf>),
    Block(TypedBlock),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TypedFor {
    pub iv: SymId,
    pub start: TypedExpr,
    pub end: TypedExpr,
    pub body: TypedBlock,
    pub span: Span,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TypedLValue {
    pub base: SymId,
    pub index: Option<TypedExpr>,
    pub span: Span,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TypedExpr {
    pub ty: Ty,
    pub span: Span,
    pub kind: TypedExprKind,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TypedExprKind {
    IntLit(i64),
    FloatLit(f64),
    BoolLit(bool),
    Var(SymId),
    /// An array *reference* appearing in argument position (or len()) — arrays have no
    /// other first-class value use (never copied, never compared).
    ArrayRef(SymId),
    Unary(UnOp, Box<TypedExpr>),
    Bin(BinOp, Box<TypedExpr>, Box<TypedExpr>),
    Index(SymId, Box<TypedExpr>),
    Call(CallTarget),
    Cast(Box<TypedExpr>, Ty),
    /// Poison node produced after reporting an error; downstream passes bail early.
    Error,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum CallTarget {
    Builtin {
        which: Builtin,
        /// Typed argument subtrees (scalars; len() carries an ArrayRef).
        args: Vec<TypedExpr>,
    },
    User {
        fn_idx: u32,
        name: String,
        /// Typed argument subtrees. Kept in the tree so the IR builder and the
        /// interpreter can lower/evaluate calls without re-walking source.
        args: Vec<TypedExpr>,
    },
}

impl TypedExpr {
    pub fn is_error(&self) -> bool {
        matches!(self.kind, TypedExprKind::Error)
    }

    /// Structural "this condition is literally true".
    fn lit_true(&self) -> bool {
        matches!(self.kind, TypedExprKind::BoolLit(true))
    }

    /// Structural "this condition is literally false".
    fn lit_false(&self) -> bool {
        matches!(self.kind, TypedExprKind::BoolLit(false))
    }
}

impl TypedStmt {
    /// Does executing this statement guarantee the enclosing function returns?
    pub fn always_returns(&self) -> bool {
        match self {
            TypedStmt::Return { .. } => true,
            TypedStmt::If(f) => {
                // A literal condition executes exactly one arm; the other is
                // dead code and must not influence the verdict.
                if f.cond.lit_true() || f.cond.lit_false() {
                    let live = if f.cond.lit_true() {
                        block_always_returns(&f.then_blk)
                    } else {
                        match &f.else_arm {
                            Some(ElseArm::Block(b)) => block_always_returns(b),
                            Some(ElseArm::If(inner)) => if_always_returns(inner),
                            None => false,
                        }
                    };
                    return live;
                }
                match f.else_arm.as_ref() {
                    Some(ElseArm::Block(b)) => {
                        block_always_returns(&f.then_blk) && block_always_returns(b)
                    }
                    Some(ElseArm::If(inner)) => {
                        block_always_returns(&f.then_blk) && if_always_returns(inner)
                    }
                    None => false,
                }
            }
            // A nested block guarantees a return exactly when one of its own
            // statements does.
            TypedStmt::Nested(b) => block_always_returns(b),
            _ => false,
        }
    }
}

fn if_always_returns(f: &TypedIf) -> bool {
    TypedStmt::If(Box::new(clone_shallow_if(f))).always_returns()
}

fn clone_shallow_if(f: &TypedIf) -> TypedIf {
    f.clone()
}

fn block_always_returns(b: &TypedBlock) -> bool {
    b.stmts.iter().any(TypedStmt::always_returns)
}

// ---------------------------------------------------------------------------
// Checker state
// ---------------------------------------------------------------------------

struct FnSig {
    name: String,
    params: Vec<Ty>,
    ret: Ty,
}

type Scope = HashMap<String, SymId>;

struct FnCtx<'a> {
    symbols: Vec<Symbol>,
    scopes: Vec<Scope>,
    sigs: &'a [FnSig],
    diags: Vec<SemDiag>,
    /// Enclosing function's return type; None = unit (bare `return;` allowed).
    ret_ty: Option<Ty>,
}

impl FnCtx<'_> {
    fn declare(&mut self, sym: Symbol) -> Result<SymId, ()> {
        let name = sym.name.clone();
        let cur = self.scopes.last_mut().expect("scope stack non-empty");
        if cur.contains_key(&name) {
            self.diags.push(diag(
                sym.span,
                format!("duplicate name '{}' in this scope", name),
            ));
            return Err(());
        }
        let id = SymId(self.symbols.len() as u32);
        self.symbols.push(sym);
        cur.insert(name, id);
        Ok(id)
    }

    fn resolve(&mut self, name: &str, span: Span) -> Result<SymId, ()> {
        for scope in self.scopes.iter().rev() {
            if let Some(&id) = scope.get(name) {
                return Ok(id);
            }
        }
        self.diags
            .push(diag(span, format!("undeclared variable '{}'", name)));
        Err(())
    }

    fn ty_of(&self, id: SymId) -> Ty {
        debug_assert_ne!(id, POISON);
        self.symbols[id.0 as usize].ty
    }

    fn kind_of(&self, id: SymId) -> SymKind {
        self.symbols[id.0 as usize].kind
    }

    fn err(&self, span: Span) -> TypedExpr {
        TypedExpr {
            ty: Ty::Unit,
            span,
            kind: TypedExprKind::Error,
        }
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub fn check(program: &Program) -> Result<TypedProgram, Vec<SemDiag>> {
    let mut diags: Vec<SemDiag> = Vec::new();

    let mut fns: Vec<&FnDef> = Vec::new();
    let mut consts: Vec<&ConstDef> = Vec::new();
    let mut seen_fns: HashMap<&str, ()> = HashMap::new();
    let mut seen_consts: HashMap<&str, ()> = HashMap::new();

    for item in &program.items {
        match item {
            Item::Fn(f) => {
                if seen_fns.insert(f.name.name.as_str(), ()).is_some() {
                    diags.push(diag(
                        f.name.span,
                        format!("duplicate function '{}'", f.name.name),
                    ));
                }
                // A builtin name is reserved: every call site resolves to the
                // builtin, so a user definition would be silently uncallable.
                if Builtin::from_name(&f.name.name).is_some() {
                    diags.push(diag(
                        f.name.span,
                        format!(
                            "'{}' is a builtin and cannot be redefined as a function",
                            f.name.name
                        ),
                    ));
                }
                fns.push(f);
            }
            Item::Const(c) => {
                if seen_consts.insert(c.name.name.as_str(), ()).is_some() {
                    diags.push(diag(
                        c.name.span,
                        format!("duplicate constant '{}'", c.name.name),
                    ));
                }
                consts.push(c);
            }
        }
    }

    // ---- Function signatures (pass 1: enables recursion / call-before-def) ----
    let sigs: Vec<FnSig> = fns
        .iter()
        .map(|f| {
            let ret = f.ret.as_ref().map(conv_type).unwrap_or(Ty::Unit);
            // Arrays are second-class by spec ("arrays are never copied"):
            // they travel by reference through call arguments, so a fn
            // RETURNING one has no lowering — the JIT fails on such programs.
            // Reject at the source instead of diverging between backends.
            if ret.is_array() {
                diags.push(diag(
                    f.span,
                    format!(
                        "function '{}' cannot return an array (arrays are never copied; \
                         write into a caller-provided array parameter instead)",
                        f.name.name
                    ),
                ));
            }
            FnSig {
                name: f.name.name.clone(),
                params: f.params.iter().map(|p| conv_type(&p.ty)).collect(),
                ret,
            }
        })
        .collect();

    // ---- Constants: shared symbol-arena prefix visible in every function ----
    let mut const_syms: Vec<Symbol> = Vec::new();
    let mut typed_consts: Vec<TypedConstDef> = Vec::new();
    let mut const_scope: Scope = Scope::new();

    for c in &consts {
        let ty = conv_type(&c.ty);
        if !matches!(ty, Ty::I64 | Ty::I32 | Ty::F64 | Ty::F32 | Ty::Bool) {
            diags.push(diag(c.span, "consts must have a scalar type"));
            continue;
        }
        let ok = matches!(
            (&c.value, ty),
            (Literal::Int(_), Ty::I64)
                | (Literal::Int(_), Ty::I32)
                | (Literal::Float(_), Ty::F64)
                | (Literal::Float(_), Ty::F32)
                | (Literal::Bool(_), Ty::Bool)
        );
        if !ok {
            diags.push(diag(
                c.span,
                format!("const value does not match declared type {}", ty.name()),
            ));
            continue;
        }
        if let (Literal::Int(v), Ty::I32) = (&c.value, ty)
            && (*v < i32::MIN as i64 || *v > i32::MAX as i64)
        {
            diags.push(diag(
                c.span,
                format!("integer literal {} does not fit in i32", v),
            ));
            continue;
        }
        let value = match &c.value {
            Literal::Int(v) => ConstLit::Int(*v),
            Literal::Float(v) => ConstLit::Float(*v),
            Literal::Bool(b) => ConstLit::Bool(*b),
        };
        let id = SymId(const_syms.len() as u32);
        const_syms.push(Symbol {
            name: c.name.name.clone(),
            ty,
            kind: SymKind::Const,
            span: c.name.span,
        });
        const_scope.insert(c.name.name.clone(), id);
        typed_consts.push(TypedConstDef {
            name: c.name.name.clone(),
            ty,
            sym: id,
            value,
            span: c.span,
        });
    }

    // ---- Function bodies (pass 2) ------------------------------------------
    let mut typed_fns: Vec<TypedFnDef> = Vec::new();
    let mut mains = 0usize;

    for f in &fns {
        let idx = typed_fns.len();
        let ret = sigs[idx].ret;

        if f.name.name == "main" {
            mains += 1;
            if !f.params.is_empty() {
                diags.push(diag(f.span, "'main' must take no parameters"));
            }
            // `-> ()` is EXPLICIT unit and legal (same as any other fn); only
            // a non-unit return type is an error. The old syntactic check
            // rejected the explicit form while accepting bare main.
            if let Some(rt) = &f.ret
                && conv_type(rt) != Ty::Unit
            {
                diags.push(diag(
                    f.span,
                    format!("'main' must return unit, found '{}'", conv_type(rt).name()),
                ));
            }
        }

        let mut ctx = FnCtx {
            symbols: const_syms.clone(),
            scopes: vec![const_scope.clone()],
            sigs: &sigs,
            diags: Vec::new(),
            ret_ty: if ret == Ty::Unit { None } else { Some(ret) },
        };

        let mut params = Vec::new();
        for p in &f.params {
            let ty = conv_type(&p.ty);
            if let Ok(id) = ctx.declare(Symbol {
                name: p.name.name.clone(),
                ty,
                kind: SymKind::Param,
                span: p.name.span,
            }) {
                params.push((id, ty));
            }
        }

        let body = ctx.check_block(&f.body);

        // Value-returning functions must return on all paths.
        if ret != Ty::Unit && !body_always_returns(&body) {
            ctx.diags.push(diag(
                f.span,
                format!(
                    "function '{}' returns {} but not all control-flow paths return a value",
                    f.name.name,
                    ret.name()
                ),
            ));
        }

        diags.extend(ctx.diags);

        let symbols = std::mem::take(&mut ctx.symbols);
        typed_fns.push(TypedFnDef {
            name: f.name.name.clone(),
            params,
            ret,
            body,
            symbols,
            span: f.span,
        });
    }

    if mains == 0 {
        diags.push(diag(
            Span { start: 0, end: 0 },
            "program has no 'fn main()'",
        ));
    }

    // ---- Definite assignment (pass 3, on typed tree) ------------------------
    for tfn in &typed_fns {
        init_analysis_fn(tfn, &mut diags);
    }

    if diags.is_empty() {
        Ok(TypedProgram {
            funcs: typed_fns,
            consts: typed_consts,
        })
    } else {
        diags.sort_by_key(|d| d.span.start);
        Err(diags)
    }
}

fn body_always_returns(b: &TypedBlock) -> bool {
    block_always_returns(b)
}

fn conv_type(t: &SynType) -> Ty {
    match t {
        SynType::I32 => Ty::I32,
        SynType::I64 => Ty::I64,
        SynType::F32 => Ty::F32,
        SynType::F64 => Ty::F64,
        SynType::Bool => Ty::Bool,
        SynType::Array(e) => Ty::Array(conv_elem(e)),
        SynType::Unit => Ty::Unit,
    }
}

fn conv_elem(e: &ScalarType) -> ElemTy {
    match e {
        ScalarType::I32 => ElemTy::I32,
        ScalarType::I64 => ElemTy::I64,
        ScalarType::F32 => ElemTy::F32,
        ScalarType::F64 => ElemTy::F64,
        ScalarType::Bool => ElemTy::Bool,
    }
}

// ---------------------------------------------------------------------------
// Statements
// ---------------------------------------------------------------------------

impl FnCtx<'_> {
    fn check_block(&mut self, block: &Block) -> TypedBlock {
        self.scopes.push(HashMap::new());
        let stmts = block.stmts.iter().map(|s| self.check_stmt(s)).collect();
        self.scopes.pop();
        TypedBlock { stmts }
    }

    fn check_stmt(&mut self, stmt: &Stmt) -> TypedStmt {
        match stmt {
            Stmt::Let { name, ty, init, .. } => {
                let declared = ty.as_ref().map(conv_type);
                let init_t = self.expr(init, declared);
                let inferred = init_t.ty;

                let final_ty = match declared {
                    Some(t) => {
                        if !init_t.is_error() && inferred != t {
                            self.diags.push(diag(
                                init.span(),
                                format!(
                                    "type mismatch: expected {}, found {}",
                                    t.name(),
                                    inferred.name()
                                ),
                            ));
                        }
                        t
                    }
                    None => inferred,
                };

                let sym = if final_ty == Ty::Unit && !init_t.is_error() {
                    self.diags
                        .push(diag(name.span, "cannot bind a unit expression"));
                    POISON
                } else {
                    match self.declare(Symbol {
                        name: name.name.clone(),
                        ty: final_ty,
                        kind: SymKind::Local,
                        span: name.span,
                    }) {
                        Ok(id) => id,
                        Err(()) => POISON,
                    }
                };
                TypedStmt::Let {
                    sym,
                    ty: final_ty,
                    init: init_t,
                }
            }
            Stmt::Assign { target, value, .. } => {
                // Expected type flows from the assignment target into the value
                // (bidirectional), e.g. `x = 3` where x: f64 is still an error, but
                // element stores benefit from knowing the element type for literals.
                let want = self.peek_assign_target_ty(target);
                let value_t = self.expr(value, want);
                self.check_assign(target, value_t)
            }
            Stmt::If {
                cond,
                then_blk,
                else_part,
                ..
            } => {
                let cond_t = self.expr(cond, Some(Ty::Bool));
                if cond_t.ty != Ty::Bool && !cond_t.is_error() {
                    self.diags.push(diag(
                        cond_t.span,
                        format!("if condition must be bool, found {}", cond_t.ty.name()),
                    ));
                }
                let then_t = self.check_block(then_blk);
                let else_t = else_part.as_ref().map(|e| match e.as_ref() {
                    ElsePart::If(inner) => match self.check_stmt(inner) {
                        TypedStmt::If(boxed) => ElseArm::If(boxed),
                        _ => unreachable!("else-if is syntactically always an if"),
                    },
                    ElsePart::Block(b) => ElseArm::Block(self.check_block(b)),
                });
                TypedStmt::If(Box::new(TypedIf {
                    cond: cond_t,
                    then_blk: then_t,
                    else_arm: else_t,
                }))
            }
            Stmt::For {
                iv,
                start,
                end,
                body,
                ..
            } => {
                let start_t = self.expr(start, Some(Ty::I64));
                let end_t = self.expr(end, Some(Ty::I64));
                for (t, what) in [(&start_t, "start"), (&end_t, "end")] {
                    if !t.ty.is_integral() && !t.is_error() {
                        self.diags.push(diag(
                            t.span,
                            format!(
                                "for range {} must be an integer, found {}",
                                what,
                                t.ty.name()
                            ),
                        ));
                    }
                }

                self.scopes.push(HashMap::new());
                let iv_sym = match self.declare(Symbol {
                    name: iv.name.clone(),
                    ty: Ty::I64,
                    kind: SymKind::LoopVar,
                    span: iv.span,
                }) {
                    Ok(id) => id,
                    Err(()) => POISON,
                };
                let body_t = self.check_block(body);
                self.scopes.pop();
                TypedStmt::For(Box::new(TypedFor {
                    iv: iv_sym,
                    start: start_t,
                    end: end_t,
                    body: body_t,
                    span: iv.span,
                }))
            }
            Stmt::Return { value, span } => match (&value, self.ret_ty) {
                // Bare `return;` in a value-returning function is an error too.
                (None, Some(_)) => {
                    self.diags.push(diag(
                        *span,
                        "bare 'return;' in a function that returns a value — return an expression",
                    ));
                    TypedStmt::Return {
                        value: None,
                        span: *span,
                    }
                }
                (None, None) => TypedStmt::Return {
                    value: None,
                    span: *span,
                },
                (Some(v), None) => {
                    let t = self.expr(v, None);
                    if !t.is_error() {
                        self.diags
                            .push(diag(*span, "cannot return a value from a unit function"));
                    }
                    TypedStmt::Return {
                        value: None,
                        span: *span,
                    }
                }
                (Some(v), Some(ret)) => {
                    let t = self.expr(v, Some(ret));
                    if !t.is_error() && t.ty != ret {
                        self.diags.push(diag(
                            v.span(),
                            format!(
                                "return type mismatch: expected {}, found {}",
                                ret.name(),
                                t.ty.name()
                            ),
                        ));
                    }
                    TypedStmt::Return {
                        value: Some(t),
                        span: *span,
                    }
                }
            },
            Stmt::Expr(e) => TypedStmt::Effect(self.expr(e, None)),
            Stmt::Empty => TypedStmt::Empty,
            Stmt::Block(b) => TypedStmt::Nested(self.check_block(b)),
        }
    }

    /// Target type of an assignment, resolved WITHOUT reporting errors (used only to
    /// steer literal adaptation in the value expression). Returns None on failure —
    /// the real check below reports problems once.
    fn peek_assign_target_ty(&mut self, target: &LValue) -> Option<Ty> {
        let id = self.resolve(&target.base.name, target.base.span).ok()?;
        let base_ty = self.ty_of(id);
        match &target.index {
            None => Some(base_ty),
            Some(_) => base_ty.elem().map(elem_scalar_ty),
        }
    }

    fn check_assign(&mut self, target: &LValue, value: TypedExpr) -> TypedStmt {
        let base_sym = match self.resolve(&target.base.name, target.base.span) {
            Ok(id) => id,
            Err(()) => {
                return TypedStmt::Assign {
                    target: TypedLValue {
                        base: POISON,
                        index: None,
                        span: target.span,
                    },
                    value,
                };
            }
        };
        let base_ty = self.ty_of(base_sym);

        match self.kind_of(base_sym) {
            crate::SymKind::LoopVar => {
                self.diags.push(diag(
                    target.base.span,
                    format!(
                        "cannot assign to loop variable '{}' (loops must stay affine)",
                        target.base.name
                    ),
                ));
            }
            crate::SymKind::Const => {
                self.diags.push(diag(
                    target.base.span,
                    format!("cannot assign to constant '{}'", target.base.name),
                ));
            }
            _ => {}
        }

        let lv = match &target.index {
            None => {
                if base_ty.is_array() {
                    self.diags.push(diag(
                        target.span,
                        "cannot assign a whole array; assign individual elements instead",
                    ));
                } else if value.ty != base_ty && !value.is_error() {
                    self.diags.push(diag(
                        value.span,
                        format!(
                            "type mismatch: cannot assign {} to '{}' of type {}",
                            value.ty.name(),
                            target.base.name,
                            base_ty.name()
                        ),
                    ));
                }
                TypedLValue {
                    base: base_sym,
                    index: None,
                    span: target.span,
                }
            }
            Some(idx_expr) => {
                let idx_t = self.expr(idx_expr, Some(Ty::I64));
                if !idx_t.ty.is_integral() && !idx_t.is_error() {
                    self.diags.push(diag(
                        idx_t.span,
                        format!("array index must be an integer, found {}", idx_t.ty.name()),
                    ));
                }
                match base_ty.elem() {
                    None => {
                        self.diags.push(diag(
                            target.base.span,
                            format!("'{}' is not an array", target.base.name),
                        ));
                    }
                    Some(elem) => {
                        let want = elem_scalar_ty(elem);
                        if value.ty != want && !value.is_error() {
                            self.diags.push(diag(
                                value.span,
                                format!(
                                    "type mismatch: cannot store {} into [{}]",
                                    value.ty.name(),
                                    elem_name(elem)
                                ),
                            ));
                        }
                    }
                }
                TypedLValue {
                    base: base_sym,
                    index: Some(idx_t),
                    span: target.span,
                }
            }
        };
        TypedStmt::Assign { target: lv, value }
    }
}

fn elem_name(e: ElemTy) -> &'static str {
    match e {
        ElemTy::I32 => "i32",
        ElemTy::I64 => "i64",
        ElemTy::F32 => "f32",
        ElemTy::F64 => "f64",
        ElemTy::Bool => "bool",
    }
}

fn elem_scalar_ty(e: ElemTy) -> Ty {
    match e {
        ElemTy::I32 => Ty::I32,
        ElemTy::I64 => Ty::I64,
        ElemTy::F32 => Ty::F32,
        ElemTy::F64 => Ty::F64,
        ElemTy::Bool => Ty::Bool,
    }
}

// ---------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------

impl FnCtx<'_> {
    /// Check an expression against an optional expected type (bidirectional checking).
    pub fn expr(&mut self, e: &Expr, expected: Option<Ty>) -> TypedExpr {
        match e {
            Expr::IntLit(v, span) => {
                let ty = match expected {
                    Some(Ty::I32) => {
                        if *v < i32::MIN as i64 || *v > i32::MAX as i64 {
                            self.diags.push(diag(
                                *span,
                                format!("integer literal {} does not fit in i32", v),
                            ));
                        }
                        Ty::I32
                    }
                    _ => Ty::I64,
                };
                TypedExpr {
                    ty,
                    span: *span,
                    kind: TypedExprKind::IntLit(*v),
                }
            }
            Expr::FloatLit(v, span) => {
                let ty = if matches!(expected, Some(Ty::F32)) {
                    Ty::F32
                } else {
                    Ty::F64
                };
                TypedExpr {
                    ty,
                    span: *span,
                    kind: TypedExprKind::FloatLit(*v),
                }
            }
            Expr::Bool(b, span) => TypedExpr {
                ty: Ty::Bool,
                span: *span,
                kind: TypedExprKind::BoolLit(*b),
            },
            Expr::Var(ident) => {
                let id = match self.resolve(&ident.name, ident.span) {
                    Ok(id) => id,
                    Err(()) => return self.err(ident.span),
                };
                let ty = self.ty_of(id);
                if ty.is_array() {
                    self.diags.push(diag(
                        ident.span,
                        format!(
                            "array '{}' cannot be used as a value (index it, or pass it to a call)",
                            ident.name
                        ),
                    ));
                    return self.err(ident.span);
                }
                TypedExpr {
                    ty,
                    span: ident.span,
                    kind: TypedExprKind::Var(id),
                }
            }
            Expr::Unary(op, operand, span) => {
                let o = self.expr(operand, None);
                if o.is_error() {
                    return self.err(*span);
                }
                let ok = match op {
                    UnOp::Neg => o.ty.is_numeric(),
                    UnOp::Not => o.ty == Ty::Bool,
                };
                if !ok {
                    let sym = match op {
                        UnOp::Neg => "-",
                        UnOp::Not => "!",
                    };
                    self.diags.push(diag(
                        o.span,
                        format!("unary {} cannot be applied to {}", sym, o.ty.name()),
                    ));
                    return self.err(*span);
                }
                TypedExpr {
                    ty: o.ty,
                    span: *span,
                    kind: TypedExprKind::Unary(*op, Box::new(o)),
                }
            }
            Expr::Bin(op, l, r, span) => {
                let (want_r, result_ty) = match op {
                    BinOp::And | BinOp::Or => (Some(Ty::Bool), Ty::Bool),
                    BinOp::Eq | BinOp::Ne => (None, Ty::Bool),
                    BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge => (None, Ty::Bool),
                    _ => (None, Ty::Unit), // arithmetic: unified with lhs below
                };
                let lt = self.expr(l, None);
                let want_for_r = if result_ty == Ty::Unit {
                    lt.ty
                } else {
                    want_r.unwrap_or(lt.ty)
                };
                let rt = self.expr(r, Some(want_for_r));
                self.binop(*op, lt, rt, *span)
            }
            Expr::Index(base_ident, idx, span) => {
                let id = match self.resolve(&base_ident.name, base_ident.span) {
                    Ok(id) => id,
                    Err(()) => return self.err(*span),
                };
                let base_ty = self.ty_of(id);
                let idx_t = self.expr(idx, Some(Ty::I64));
                match base_ty.elem() {
                    Some(elem) => {
                        if !idx_t.ty.is_integral() && !idx_t.is_error() {
                            self.diags.push(diag(
                                idx_t.span,
                                format!(
                                    "array index must be an integer, found {}",
                                    idx_t.ty.name()
                                ),
                            ));
                        }
                        TypedExpr {
                            ty: elem_scalar_ty(elem),
                            span: *span,
                            kind: TypedExprKind::Index(id, Box::new(idx_t)),
                        }
                    }
                    None => {
                        self.diags.push(diag(
                            base_ident.span,
                            format!("'{}' is not an array", base_ident.name),
                        ));
                        self.err(*span)
                    }
                }
            }
            Expr::Cast(operand, to_syn, span) => {
                let o = self.expr(operand, None);
                let to = conv_type(to_syn);
                if !matches!(to, Ty::I32 | Ty::I64 | Ty::F32 | Ty::F64) {
                    self.diags
                        .push(diag(*span, "can only cast to numeric types"));
                    return self.err(*span);
                }
                if !o.ty.is_numeric() && !o.is_error() {
                    self.diags.push(diag(
                        o.span,
                        format!("can only cast numeric types, found {}", o.ty.name()),
                    ));
                    return self.err(*span);
                }
                TypedExpr {
                    ty: to,
                    span: *span,
                    kind: TypedExprKind::Cast(Box::new(o), to),
                }
            }
            Expr::Call { callee, args, span } => self.call(callee, args, *span, expected),
        }
    }

    fn binop(&mut self, op: BinOp, l: TypedExpr, r: TypedExpr, span: Span) -> TypedExpr {
        use BinOp::*;
        if l.is_error() || r.is_error() {
            return self.err(span);
        }
        let arith_ok =
            l.ty == r.ty && l.ty.is_numeric() && matches!(op, Add | Sub | Mul | Div | Rem);
        let cmp_ok = l.ty == r.ty && l.ty.is_numeric() && matches!(op, Lt | Gt | Le | Ge);
        let eq_ok = l.ty == r.ty && l.ty.is_scalar() && matches!(op, Eq | Ne);
        let logic_ok = l.ty == Ty::Bool && r.ty == Ty::Bool && matches!(op, And | Or);

        if arith_ok {
            TypedExpr {
                ty: l.ty,
                span,
                kind: TypedExprKind::Bin(op, Box::new(l), Box::new(r)),
            }
        } else if cmp_ok || eq_ok || logic_ok {
            TypedExpr {
                ty: Ty::Bool,
                span,
                kind: TypedExprKind::Bin(op, Box::new(l), Box::new(r)),
            }
        } else {
            let what = match op {
                Add | Sub | Mul | Div | Rem => "arithmetic",
                Lt | Gt | Le | Ge => "comparison",
                Eq | Ne => "equality",
                And | Or => "'&&'/'||'",
            };
            self.diags.push(diag(
                span,
                format!(
                    "{} needs two compatible operands, got {} and {}",
                    what,
                    l.ty.name(),
                    r.ty.name()
                ),
            ));
            self.err(span)
        }
    }

    #[allow(clippy::too_many_lines)]
    fn call(
        &mut self,
        callee: &Ident,
        args: &[Expr],
        span: Span,
        expected: Option<Ty>,
    ) -> TypedExpr {
        if let Some(b) = Builtin::from_name(&callee.name) {
            return self.builtin(b, args, span, expected);
        }

        // User function?
        let sig_info = self
            .sigs
            .iter()
            .enumerate()
            .find(|(_, sg)| sg.name == callee.name)
            .map(|(fi, sg)| (fi, sg.params.clone(), sg.ret));
        if let Some((fi, want_params, ret)) = sig_info {
            if args.len() != want_params.len() {
                self.diags.push(diag(
                    span,
                    format!(
                        "function '{}' expects {} argument(s), got {}",
                        callee.name,
                        want_params.len(),
                        args.len()
                    ),
                ));
                for a in args {
                    self.expr(a, None);
                }
                return self.err(span);
            }
            let mut typed_args = Vec::new();
            for (a, want) in args.iter().zip(&want_params) {
                typed_args.push(self.arg_expr(a, *want));
            }
            // Alias protection: same array variable twice in one call (spec normative —
            // dependence analysis cannot see through aliasing).
            let mut seen_arrays: HashSet<u32> = HashSet::new();
            for (a, t) in args.iter().zip(&typed_args) {
                if let TypedExprKind::ArrayRef(id) = t.kind
                    && !seen_arrays.insert(id.0)
                {
                    let name = &a.ident_name();
                    self.diags.push(diag(
                        a.span(),
                        format!(
                            "array '{}' passed multiple times to '{}' (aliasing rejected)",
                            name, callee.name
                        ),
                    ));
                }
            }
            return TypedExpr {
                ty: ret,
                span,
                kind: TypedExprKind::Call(CallTarget::User {
                    fn_idx: fi as u32,
                    name: callee.name.clone(),
                    args: typed_args,
                }),
            };
        }

        self.diags.push(diag(
            callee.span,
            format!("unknown function '{}'", callee.name),
        ));
        for a in args {
            self.expr(a, None);
        }
        self.err(span)
    }

    /// Check one argument position. Arrays are allowed here as [`TypedExprKind::ArrayRef`].
    fn arg_expr(&mut self, a: &Expr, want: Ty) -> TypedExpr {
        if let Expr::Var(ident) = a
            && let Ok(id) = self.resolve(&ident.name, ident.span)
        {
            let ty = self.ty_of(id);
            if ty.is_array() {
                // Array passed to an array parameter: element types must match.
                // Array passed to a scalar parameter: rejected (arrays aren't values).
                if !want.is_array() || ty != want {
                    self.diags.push(diag(
                        ident.span,
                        format!(
                            "argument type mismatch: expected {}, found {}",
                            want.name(),
                            ty.name()
                        ),
                    ));
                    return self.err(ident.span);
                }
                return TypedExpr {
                    ty,
                    span: ident.span,
                    kind: TypedExprKind::ArrayRef(id),
                };
            }
        }
        let t = self.expr(a, Some(want));
        if !t.is_error() && t.ty != want {
            self.diags.push(diag(
                t.span,
                format!(
                    "argument type mismatch: expected {}, found {}",
                    want.name(),
                    t.ty.name()
                ),
            ));
            return self.err(t.span);
        }
        t
    }

    #[allow(clippy::too_many_lines)]
    fn builtin(
        &mut self,
        b: Builtin,
        args: &[Expr],
        span: Span,
        expected: Option<Ty>,
    ) -> TypedExpr {
        // Arity check helper: reports + type-checks stray args + returns poison.
        macro_rules! need_arity {
            ($n:expr) => {{
                if args.len() != $n {
                    self.diags.push(diag(
                        span,
                        format!(
                            "builtin '{}' expects {} argument(s), got {}",
                            b.name(),
                            $n,
                            args.len()
                        ),
                    ));
                    for a in args {
                        self.expr(a, None);
                    }
                    return self.err(span);
                }
            }};
        }

        match b {
            Builtin::Print => {
                need_arity!(1);
                let t = self.expr(&args[0], None);
                if !t.ty.is_scalar() && !t.is_error() {
                    self.diags.push(diag(
                        t.span,
                        format!("print takes a scalar, found {}", t.ty.name()),
                    ));
                    return self.err(span);
                }
                TypedExpr {
                    ty: Ty::Unit,
                    span,
                    kind: TypedExprKind::Call(CallTarget::Builtin {
                        which: b,
                        args: vec![t],
                    }),
                }
            }
            Builtin::Len => {
                need_arity!(1);
                match &args[0] {
                    Expr::Var(ident) => {
                        let id = match self.resolve(&ident.name, ident.span) {
                            Ok(id) => id,
                            Err(()) => return self.err(span),
                        };
                        let ty = self.ty_of(id);
                        if !ty.is_array() {
                            self.diags.push(diag(
                                ident.span,
                                format!(
                                    "len() expects an array, '{}' is {}",
                                    ident.name,
                                    ty.name()
                                ),
                            ));
                            return self.err(span);
                        }
                        TypedExpr {
                            ty: Ty::I64,
                            span,
                            kind: TypedExprKind::Call(CallTarget::Builtin {
                                which: b,
                                args: vec![TypedExpr {
                                    ty,
                                    span: ident.span,
                                    kind: TypedExprKind::ArrayRef(id),
                                }],
                            }),
                        }
                    }
                    other => {
                        let t = self.expr(other, None);
                        self.diags
                            .push(diag(other.span(), "len() expects an array variable"));
                        let _ = t;
                        self.err(span)
                    }
                }
            }
            Builtin::Zeros => {
                need_arity!(1);
                let n = self.expr(&args[0], Some(Ty::I64));
                if !n.ty.is_integral() && !n.is_error() {
                    self.diags
                        .push(diag(n.span, "zeros(n): n must be an integer"));
                }
                match expected {
                    Some(w @ Ty::Array(_)) => TypedExpr {
                        ty: w,
                        span,
                        kind: TypedExprKind::Call(CallTarget::Builtin {
                            which: b,
                            args: vec![n],
                        }),
                    },
                    _ => {
                        self.diags.push(diag(
                            span,
                            "cannot infer element type of zeros(n); annotate the binding: let a: [f64] = zeros(n)",
                        ));
                        self.err(span)
                    }
                }
            }
            Builtin::Abs => {
                need_arity!(1);
                let t = self.expr(&args[0], None);
                if !t.ty.is_numeric() && !t.is_error() {
                    self.diags.push(diag(t.span, "abs expects a number"));
                    return self.err(span);
                }
                TypedExpr {
                    ty: t.ty,
                    span,
                    kind: TypedExprKind::Call(CallTarget::Builtin {
                        which: b,
                        args: vec![t],
                    }),
                }
            }
            Builtin::Sqrt => {
                need_arity!(1);
                let t = self.expr(&args[0], None);
                if !t.ty.is_float() && !t.is_error() {
                    self.diags.push(diag(
                        t.span,
                        "sqrt expects f32/f64 (cast integers explicitly)",
                    ));
                    return self.err(span);
                }
                TypedExpr {
                    ty: t.ty,
                    span,
                    kind: TypedExprKind::Call(CallTarget::Builtin {
                        which: b,
                        args: vec![t],
                    }),
                }
            }
            Builtin::Min | Builtin::Max => {
                need_arity!(2);
                let a = self.expr(&args[0], None);
                let c = self.expr(&args[1], Some(a.ty));
                if !(a.ty.is_numeric() && a.ty == c.ty) && !a.is_error() && !c.is_error() {
                    self.diags.push(diag(
                        span,
                        format!(
                            "{} expects two numbers of the same type, got {} and {}",
                            b.name(),
                            a.ty.name(),
                            c.ty.name()
                        ),
                    ));
                    return self.err(span);
                }
                TypedExpr {
                    ty: a.ty,
                    span,
                    kind: TypedExprKind::Call(CallTarget::Builtin {
                        which: b,
                        args: vec![a, c],
                    }),
                }
            }
        }
    }
}

trait ArgName {
    fn ident_name(&self) -> String;
}

impl ArgName for Expr {
    fn ident_name(&self) -> String {
        match self {
            Expr::Var(i) => i.name.clone(),
            _ => "<expression>".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Definite assignment (forward dataflow with branch merging)
// ---------------------------------------------------------------------------

fn init_analysis_fn(f: &TypedFnDef, diags: &mut Vec<SemDiag>) {
    // Params + consts start initialized (they are the arena prefix + params).
    let mut init: HashSet<u32> = HashSet::new();
    for (i, s) in f.symbols.iter().enumerate() {
        if matches!(s.kind, crate::SymKind::Param | crate::SymKind::Const) {
            init.insert(i as u32);
        }
    }
    let mut cx = InitCx {
        symbols: &f.symbols,
        diags,
    };
    let _after = cx.block(&f.body, init);
}

struct InitCx<'a, 'd> {
    symbols: &'a [Symbol],
    diags: &'d mut Vec<SemDiag>,
}

impl InitCx<'_, '_> {
    fn block(&mut self, b: &TypedBlock, mut init: HashSet<u32>) -> HashSet<u32> {
        for s in &b.stmts {
            init = self.stmt(s, init);
        }
        init
    }

    fn stmt(&mut self, s: &TypedStmt, mut init: HashSet<u32>) -> HashSet<u32> {
        match s {
            TypedStmt::Let { sym, ty, init: rhs } => {
                self.expr(rhs, &init);
                if *sym != POISON && !ty.is_unit() {
                    init.insert(sym.0);
                }
                init
            }
            TypedStmt::Assign { target, value } => {
                self.expr(value, &init);
                if let Some(idx) = &target.index {
                    self.expr(idx, &init);
                }
                if target.index.is_none() && target.base != POISON {
                    init.insert(target.base.0);
                }
                init
            }
            TypedStmt::If(f) => {
                self.expr(&f.cond, &init);
                let then_after = self.block(&f.then_blk, init.clone());
                let else_after = match &f.else_arm {
                    Some(crate::ElseArm::Block(b)) => self.block(b, init.clone()),
                    Some(crate::ElseArm::If(inner)) => self.stmt_if(inner, init.clone()),
                    None => init.clone(),
                };
                // Merge: definitely-assigned on BOTH paths.
                then_after.intersection(&else_after).copied().collect()
            }
            TypedStmt::For(f) => {
                self.expr(&f.start, &init);
                self.expr(&f.end, &init);
                let mut body_init = init.clone();
                body_init.insert(f.iv.0);
                let _ = self.block(&f.body, body_init);
                // Zero-trip possible: nothing gained.
                init
            }
            TypedStmt::Return { value, .. } => {
                if let Some(v) = value {
                    self.expr(v, &init);
                }
                // Everything counts as initialized after return (path ends).

                (0..self.symbols.len()).map(|i| i as u32).collect()
            }
            TypedStmt::Effect(e) => {
                self.expr(e, &init);
                init
            }
            // `;` touches nothing.
            TypedStmt::Empty => init,
            // A bare nested block is a scoped sequence: assignments inside it
            // persist (scoping of *names* was resolved during checking).
            TypedStmt::Nested(b) => self.block(b, init),
        }
    }

    fn stmt_if(&mut self, f: &TypedIf, init: HashSet<u32>) -> HashSet<u32> {
        self.expr(&f.cond, &init);
        let then_after = self.block(&f.then_blk, init.clone());
        let else_after = match &f.else_arm {
            Some(crate::ElseArm::Block(b)) => self.block(b, init.clone()),
            Some(crate::ElseArm::If(inner)) => self.stmt_if(inner, init.clone()),
            None => init.clone(),
        };
        then_after.intersection(&else_after).copied().collect()
    }

    fn expr(&mut self, e: &TypedExpr, init: &HashSet<u32>) {
        match &e.kind {
            TypedExprKind::Var(id) | TypedExprKind::ArrayRef(id) => {
                if let Some(sym) = self.symbols.get(id.0 as usize)
                    && matches!(sym.kind, crate::SymKind::Local)
                    && !init.contains(&id.0)
                {
                    self.diags.push(diag(
                        e.span,
                        format!("'{}' may be used before initialization", sym.name),
                    ));
                }
            }
            TypedExprKind::Index(_, idx) => self.expr(idx, init),
            TypedExprKind::Unary(_, o) => self.expr(o, init),
            TypedExprKind::Bin(_, l, r) => {
                self.expr(l, init);
                self.expr(r, init);
            }
            TypedExprKind::Cast(o, _) => self.expr(o, init),
            _ => {}
        }
    }
}
