//! The tree-walking evaluator — the executable form of `lang-spec.md`.
//!
//! ## State
//!
//! One [`Interp`] per run, holding deliberately little:
//!
//! * **`scopes`** — a stack of `HashMap<SymId, Value>` for the *current*
//!   frame. Name resolution was finished by sema (identifiers are already
//!   [`SymId`]s), so lookup is O(1) per scope, and shadowing is handled by
//!   the stack itself: entering a block pushes, leaving pops.
//! * **No heap table.** Array buffers live inside [`Value::Array`] handles
//!   (`Rc<RefCell<Vec<Value>>>`) that flow through the environment like any
//!   other value. Passing an array clones the `Rc`, so callee stores mutate
//!   the caller's buffer directly — the spec's "assignment/passing is BY
//!   REFERENCE; callee writes ESCAPE" falls out of the representation.
//! * **`call_depth`** — bounds recursion so runaway programs fail with a
//!   clean [`RunErrorKind::StackExhausted`] instead of killing the host
//!   thread. Real HELIX recursion is shallow (`fib(24)` needs 24 frames).
//!
//! ## Control flow
//!
//! `return` unwinds via [`Stop::Return`] — a plain value carried in the
//! `Result`, not a panic. Panics would poison active `RefCell` borrows and
//! cannot cross the API boundary cleanly; an explicit stop makes every
//! fallible step visible in the types.
//!
//! ## Faithfulness notes (spec §Operator semantics)
//!
//! * `/` `%` trap on zero divisor and on the overflow edge (`i64::MIN / -1`,
//!   and the i32 analogue) *before* any arithmetic runs — Rust would panic,
//!   the spec wants `runtime error: … at line N`.
//! * Rust's `%` IS truncated remainder (sign of dividend): `-7 % 2 == -1`.
//! * Float division is IEEE: `x / 0.0` gives `inf`/`NaN`, never an error.
//! * Integer overflow elsewhere wraps (two's complement), as documented.
//! * Casts are Rust's `as`: float→int saturating (NaN→0, clamp at limits),
//!   int→int truncating two's complement, int↔float rounding toward zero.
//! * `&&`/`||` short-circuit before evaluating the right operand — observable
//!   through side-effecting calls (`examples/shortcircuit.hx`).
//! * The induction variable is written fresh each iteration and never read
//!   back-modified; sema forbids assigning it anyway.
//! * `min`/`max` implement IEEE minNum/maxNum: a NaN operand loses against a
//!   real number.
//! * `print` appends one line per call using the canonical formatter shared
//!   with the JIT backend.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use helix_sema::{Builtin, ElemTy, SymId, Ty, TypedConstDef};
use helix_syntax::Span;
use helix_syntax::ast::{BinOp, UnOp};

use crate::adapter::{
    AdaptedFn, AdaptedProgram, EBlock, EElse, EExpr, EExprKind, EIf, EStmt, ETarget,
};
use crate::error::{RunError, RunErrorKind};
use crate::value::Value;

/// Maximum interpreter call depth before [`RunErrorKind::StackExhausted`].
const MAX_DEPTH: usize = 2000;

/// FNV-1a offset basis.
pub(crate) const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a prime.
pub(crate) const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Why execution stopped inside a function body.
#[derive(Debug)]
enum Stop {
    /// A `return` executed (value carried; `Unit` for bare `return;`).
    Return(Value),
    /// A runtime error occurred and is propagating to the driver.
    Fail(RunError),
}

/// Everything below `Result<T, Stop>` either produced a `T` or stopped.
type Flow<T> = Result<T, Stop>;

/// Executes one program from `fn main()`.
pub struct Interp<'p> {
    program: &'p AdaptedProgram,
    /// Scope stack of the current frame (swapped across calls).
    scopes: Vec<HashMap<SymId, Value>>,
    /// Printed lines, in order.
    printed: Vec<String>,
    /// Running FNV-1a over printed lines (+ newlines) and final array bytes.
    checksum: u64,
    /// Current recursion depth.
    call_depth: usize,
}

impl<'p> Interp<'p> {
    fn new(program: &'p AdaptedProgram) -> Self {
        Self {
            program,
            scopes: Vec::new(),
            printed: Vec::new(),
            checksum: FNV_OFFSET,
            call_depth: 0,
        }
    }

    // -- environment -----------------------------------------------------------

    fn lookup(&self, id: SymId) -> Flow<Value> {
        for scope in self.scopes.iter().rev() {
            if let Some(v) = scope.get(&id) {
                return Ok(v.clone());
            }
        }
        Err(internal(format!("symbol #{} not bound at run time", id.0)))
    }

    /// Assigns through the scope stack: shadowing resolves to the nearest
    /// enclosing binding, exactly as reads do.
    fn assign(&mut self, id: SymId, v: Value) -> Flow<()> {
        for scope in self.scopes.iter_mut().rev() {
            if let Some(slot) = scope.get_mut(&id) {
                *slot = v;
                return Ok(());
            }
        }
        Err(internal(format!("assignment to unbound symbol #{}", id.0)))
    }

    // -- statements --------------------------------------------------------------

    fn block(&mut self, b: &EBlock) -> Flow<()> {
        self.scopes.push(HashMap::new());
        let mut flow = Ok(());
        for s in &b.stmts {
            match self.stmt(s) {
                Ok(()) => {}
                Err(stop) => {
                    flow = Err(stop);
                    break;
                }
            }
        }
        self.scopes.pop();
        flow
    }

    /// Executes a block's statements in the CURRENT top scope, without
    /// pushing a fresh layer. Used for `main`'s body so its bindings survive
    /// into the post-run state snapshot ([`Interp::hash_final_arrays`]).
    fn block_in_current_scope(&mut self, b: &EBlock) -> Flow<()> {
        let mut flow = Ok(());
        for s in &b.stmts {
            match self.stmt(s) {
                Ok(()) => {}
                Err(stop) => {
                    flow = Err(stop);
                    break;
                }
            }
        }
        flow
    }

    fn stmt(&mut self, s: &EStmt) -> Flow<()> {
        match s {
            EStmt::Let { sym, init } => {
                let v = self.expr(init)?;
                self.scopes
                    .last_mut()
                    .expect("statements execute inside a block scope")
                    .insert(*sym, v);
                Ok(())
            }
            EStmt::Assign {
                base, index, value, ..
            } => match index {
                None => {
                    let v = self.expr(value)?;
                    self.assign(*base, v)
                }
                Some(idx) => {
                    // Evaluate index, then RHS, then store — a fixed order
                    // that makes OOB reports deterministic.
                    let i = self.index_value(idx)?;
                    let v = self.expr(value)?;
                    self.array_store(*base, i, v, idx.span)
                }
            },
            EStmt::If {
                cond,
                then_blk,
                else_arm,
            } => {
                let Value::Bool(b) = self.expr(cond)? else {
                    return Err(internal("if condition evaluated to non-bool"));
                };
                if b {
                    self.block(then_blk)
                } else {
                    match else_arm {
                        None => Ok(()),
                        Some(EElse::If(inner)) => self.if_chain(inner),
                        Some(EElse::Block(b2)) => self.block(b2),
                    }
                }
            }
            EStmt::For {
                iv,
                start,
                end,
                body,
                ..
            } => {
                // Bounds evaluate ONCE, before iteration begins.
                let lo = self.range_bound(start)?;
                let hi = self.range_bound(end)?;
                self.scopes.push(HashMap::new());
                let mut flow = Ok(());
                let mut i = lo;
                while i < hi {
                    self.scopes
                        .last_mut()
                        .expect("loop scope was just pushed")
                        .insert(*iv, Value::I64(i));
                    match self.block(body) {
                        Ok(()) => i += 1,
                        Err(stop) => {
                            flow = Err(stop);
                            break;
                        }
                    }
                }
                self.scopes.pop();
                flow
            }
            EStmt::Return { value, .. } => {
                let v = match value {
                    None => Value::Unit,
                    Some(e) => self.expr(e)?,
                };
                Err(Stop::Return(v))
            }
            EStmt::Effect(e) => {
                self.expr(e)?;
                Ok(())
            }
            EStmt::Nested(b) => self.block(b),
        }
    }

    /// Walks an else-if spine; each link evaluates its own condition.
    fn if_chain(&mut self, f: &EIf) -> Flow<()> {
        let Value::Bool(b) = self.expr(&f.cond)? else {
            return Err(internal("if condition evaluated to non-bool"));
        };
        if b {
            self.block(&f.then_blk)
        } else {
            match &f.else_arm {
                None => Ok(()),
                Some(EElse::If(inner)) => self.if_chain(inner),
                Some(EElse::Block(b2)) => self.block(b2),
            }
        }
    }

    // -- expressions ---------------------------------------------------------------

    fn expr(&mut self, e: &EExpr) -> Flow<Value> {
        match &e.kind {
            EExprKind::IntLit(v) => Ok(int_lit(*v, e.ty)),
            EExprKind::FloatLit(v) => Ok(float_lit(*v, e.ty)),
            EExprKind::BoolLit(b) => Ok(Value::Bool(*b)),
            EExprKind::Var(id) | EExprKind::ArrayRef(id) => self.lookup(*id),
            EExprKind::Unary(op, o) => {
                let v = self.expr(o)?;
                unary(*op, v)
            }
            EExprKind::Bin(op, l, r) => {
                // Short-circuit BEFORE touching the right operand.
                if *op == BinOp::And {
                    return self.short_circuit(l, r, false);
                }
                if *op == BinOp::Or {
                    return self.short_circuit(l, r, true);
                }
                let lv = self.expr(l)?;
                let rv = self.expr(r)?;
                binary(*op, lv, rv, e.span)
            }
            EExprKind::Index(base, idx) => {
                let i = self.index_value(idx)?;
                let arr = self.lookup(*base)?;
                let Value::Array(h) = arr else {
                    return Err(internal("indexing a non-array"));
                };
                let buf = h.borrow();
                check_bounds(buf.len(), i, idx.span)?;
                // check_bounds proved 0 <= i < len, so narrowing is safe.
                Ok(buf[i as usize].clone())
            }
            EExprKind::Cast(o, to) => {
                let v = self.expr(o)?;
                Ok(cast(v, *to))
            }
            EExprKind::Call(target) => match target {
                ETarget::Builtin(b, args) => self.builtin(*b, args, e.ty, e.span),
                ETarget::User { idx, args } => {
                    // Arguments evaluate strictly left to right.
                    let mut vals = Vec::with_capacity(args.len());
                    for a in args {
                        vals.push(self.expr(a)?);
                    }
                    self.call_user(*idx, &vals, e.span)
                }
            },
        }
    }

    /// `&&` (or_wanted=false) / `||` (or_wanted=true) with short-circuiting.
    fn short_circuit(&mut self, l: &EExpr, r: &EExpr, or_wanted: bool) -> Flow<Value> {
        let lv = self.expr(l)?;
        let Value::Bool(lb) = lv else {
            return Err(internal("logical operand was not bool"));
        };
        if lb == or_wanted {
            // Left side already decides the result; right side NOT evaluated.
            return Ok(Value::Bool(or_wanted));
        }
        let rv = self.expr(r)?;
        match rv {
            Value::Bool(_) => Ok(rv),
            _ => Err(internal("logical operand was not bool")),
        }
    }

    /// Evaluates an index expression: any integer widens to i64 (the spec's
    /// single implicit coercion).
    fn index_value(&mut self, idx: &EExpr) -> Flow<i64> {
        let v = self.expr(idx)?;
        match v.as_i64_widen() {
            Some(i) => Ok(i),
            None => Err(internal("array index evaluated to a non-integer")),
        }
    }

    fn range_bound(&mut self, e: &EExpr) -> Flow<i64> {
        let v = self.expr(e)?;
        match v.as_i64_widen() {
            Some(i) => Ok(i),
            None => Err(internal("range bound evaluated to a non-integer")),
        }
    }

    fn array_store(&mut self, base: SymId, i: i64, v: Value, idx_span: Span) -> Flow<()> {
        let arr = self.lookup(base)?;
        let Value::Array(h) = arr else {
            return Err(internal("element assignment into a non-array"));
        };
        let mut buf = h.borrow_mut();
        check_bounds(buf.len(), i, idx_span)?;
        // check_bounds proved 0 <= i < len, so narrowing is safe.
        buf[i as usize] = v;
        Ok(())
    }

    // -- calls ------------------------------------------------------------------------

    fn call_user(&mut self, fi: u32, args: &[Value], span: Span) -> Flow<Value> {
        // Borrow of self.program must end before we mutate self.scopes.
        let (params, body): (&[(SymId, Ty)], &AdaptedFn) = {
            let Some(f) = self.program.funcs.get(fi as usize) else {
                return Err(internal(format!("no function with index {fi}")));
            };
            if args.len() != f.params.len() {
                return Err(internal("argument count mismatch"));
            }
            (f.params.as_slice(), f)
        };
        if self.call_depth >= MAX_DEPTH {
            return Err(Stop::Fail(RunError::new(
                RunErrorKind::StackExhausted,
                span,
            )));
        }

        // Callee gets a fresh scope stack rooted at params + consts.
        let saved = std::mem::take(&mut self.scopes);
        let mut root: HashMap<SymId, Value> =
            HashMap::with_capacity(params.len() + self.program.consts.len());
        for ((sym, _), v) in params.iter().zip(args) {
            root.insert(*sym, v.clone());
        }
        bind_consts(&mut root, body, &self.program.consts);
        self.scopes = vec![root];

        self.call_depth += 1;
        let out = self.block(&body.body);
        self.call_depth -= 1;
        self.scopes = saved;

        match out {
            Ok(()) => Ok(Value::Unit), // procedure fell off the end
            Err(Stop::Return(v)) => Ok(v),
            Err(stop @ Stop::Fail(_)) => Err(stop),
        }
    }

    #[allow(clippy::too_many_lines)] // one arm per builtin, each tiny
    fn builtin(&mut self, b: Builtin, arg_nodes: &[EExpr], call_ty: Ty, span: Span) -> Flow<Value> {
        match b {
            Builtin::Print => {
                let v = self.expr(&arg_nodes[0])?;
                let line = v.render();
                self.hash_line(&line);
                self.printed.push(line);
                Ok(Value::Unit)
            }
            Builtin::Zeros => {
                let n = self.range_bound(&arg_nodes[0])?;
                if n < 0 {
                    return Err(Stop::Fail(RunError::new(
                        RunErrorKind::NegativeZeros { n },
                        span,
                    )));
                }
                // The element type rides on the CALL node's static type
                // (an [T]); the argument only supplies the length.
                let elem_ty =
                    elem_ty_of(call_ty).ok_or_else(|| internal("zeros element type unresolved"))?;
                let buf = vec![
                    Value::zero(elem_ty);
                    usize::try_from(n).map_err(|_| {
                        internal("zeros length does not fit the address space")
                    })?
                ];
                Ok(Value::Array(Rc::new(RefCell::new(buf))))
            }
            Builtin::Len => {
                let arr = self.expr(&arg_nodes[0])?;
                let Value::Array(h) = arr else {
                    return Err(internal("len() of a non-array"));
                };
                Ok(Value::I64(h.borrow().len() as i64))
            }
            Builtin::Abs => {
                let v = self.expr(&arg_nodes[0])?;
                Ok(match v {
                    // abs(i32::MIN) overflows: wrap like every other integer op.
                    Value::I32(x) => Value::I32(x.wrapping_abs()),
                    Value::I64(x) => Value::I64(x.wrapping_abs()),
                    Value::F32(x) => Value::F32(x.abs()),
                    Value::F64(x) => Value::F64(x.abs()),
                    other => return Err(internal(format!("abs over {}", other.ty_name()))),
                })
            }
            Builtin::Sqrt => {
                let v = self.expr(&arg_nodes[0])?;
                Ok(match v {
                    // IEEE: negative input yields NaN, never an error.
                    Value::F32(x) => Value::F32(x.sqrt()),
                    Value::F64(x) => Value::F64(x.sqrt()),
                    other => return Err(internal(format!("sqrt over {}", other.ty_name()))),
                })
            }
            Builtin::Min | Builtin::Max => {
                let a = self.expr(&arg_nodes[0])?;
                let c = self.expr(&arg_nodes[1])?;
                match (a, c) {
                    (Value::I32(x), Value::I32(y)) => Ok(int_minmax_i32(b, x, y)),
                    (Value::I64(x), Value::I64(y)) => Ok(int_minmax_i64(b, x, y)),
                    (Value::F32(x), Value::F32(y)) => Ok(float_minmax_f32(b, x, y)),
                    (Value::F64(x), Value::F64(y)) => Ok(float_minmax_f64(b, x, y)),
                    (a, c) => Err(internal(format!(
                        "min/max over {} and {}",
                        a.ty_name(),
                        c.ty_name()
                    ))),
                }
            }
        }
    }

    // -- finish -------------------------------------------------------------------------

    /// Feeds one printed line plus its newline into the checksum.
    ///
    /// Including the terminator makes line boundaries part of the output's
    /// identity: "1","23" can never collide with "12","3".
    fn hash_line(&mut self, line: &str) {
        for byte in line.bytes().chain(std::iter::once(b'\n')) {
            self.checksum ^= u64::from(byte);
            self.checksum = self.checksum.wrapping_mul(FNV_PRIME);
        }
    }

    /// Folds every array still visible in main's final environment into the
    /// checksum, iterating bindings in symbol-id order so the byte stream is
    /// deterministic across runs.
    ///
    /// Arrays whose only binding lived in an exited block scope are gone by
    /// now (Rc dropped); that is faithful — they were unreachable.
    fn hash_final_arrays(&mut self) {
        let mut ids: Vec<u32> = self
            .scopes
            .iter()
            .flat_map(|sc| sc.keys())
            .map(|id| id.0)
            .collect();
        ids.sort_unstable();
        ids.dedup();
        for raw in ids {
            let id = SymId(raw);
            // Arrays live only in main's root scope by the end of a run
            // (block scopes have been popped), so root is the right place
            // to look them up.
            if let Some(v @ Value::Array(_)) = self.scopes[0].get(&id) {
                v.hash_bits_into(&mut self.checksum);
            }
        }
    }
}

/// Native stack given to the worker thread. The interpreter recurses per
/// AST node (block → stmt → expr), and each *HELIX* frame costs many native
/// frames, so [`MAX_DEPTH`] must sit under a generous stack to be reachable
/// as a clean error rather than STATUS_STACK_OVERFLOW. 64 MiB covers the
/// bound with two orders of magnitude of headroom.
const WORKER_STACK_BYTES: usize = 64 * 1024 * 1024;

/// Runs `main()` to completion and collects the output.
///
/// Execution happens on a dedicated thread with [`WORKER_STACK_BYTES`] of
/// stack. This is sound because everything shipped across the boundary is
/// plain owned data ([`AdaptedProgram`] in; [`RunOutcome`]/[`RunError`] out)
/// — the `Rc`/`RefCell` values live only inside the worker.
///
/// Runtime errors surface as `Err`; successful runs always include every
/// printed line and the state checksum.
pub(crate) fn execute(program: &AdaptedProgram) -> Result<RunOutcome, RunError> {
    // The borrowed tree cannot cross the thread boundary (no 'static), so a
    // clone ships instead. One deep copy of the EIR is negligible next to
    // interpreting it.
    let owned = program.clone();
    let handle = std::thread::Builder::new()
        .name("helix-interp".to_string())
        .stack_size(WORKER_STACK_BYTES)
        .spawn(move || execute_in_worker(&owned))
        .expect("worker thread spawn cannot fail under normal conditions");
    handle.join().unwrap_or_else(|panic_payload| {
        // The worker should not panic (no unwinding across FFI here), but if
        // it ever does, report it as an internal error instead of crashing
        // the host process.
        let detail = panic_payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| panic_payload.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_else(|| "unknown panic".to_string());
        Err(RunError {
            kind: RunErrorKind::Internal(format!("interpreter panicked: {detail}")),
            span: Span { start: 0, end: 0 },
        })
    })
}

fn execute_in_worker(program: &AdaptedProgram) -> Result<RunOutcome, RunError> {
    let mut it = Interp::new(program);
    let Some(mi) = it.program.funcs.iter().position(|f| f.name == "main") else {
        return Err(RunError {
            kind: RunErrorKind::Internal("program has no main".to_string()),
            span: Span { start: 0, end: 0 },
        });
    };

    // Inline main's prologue and run its body IN the root scope so the final
    // environment survives for the state checksum.
    let main_fn: &AdaptedFn = &it.program.funcs[mi];
    let mut root: HashMap<SymId, Value> =
        HashMap::with_capacity(main_fn.params.len() + program.consts.len());
    bind_consts(&mut root, main_fn, &program.consts);
    it.scopes = vec![root];

    match it.block_in_current_scope(&main_fn.body) {
        Ok(()) | Err(Stop::Return(_)) => {}
        Err(Stop::Fail(e)) => return Err(e),
    }
    it.hash_final_arrays();
    Ok(RunOutcome {
        printed: std::mem::take(&mut it.printed),
        checksum: it.checksum,
    })
}

/// What a completed run produced.
pub(crate) struct RunOutcome {
    /// Every `print` line, in order.
    pub printed: Vec<String>,
    /// FNV-1a over printed lines and final array bytes.
    pub checksum: u64,
}

/// Copies const values into a fresh root scope.
///
/// Const symbols occupy the same arena slots in every function (the shared
/// prefix sema builds), so binding by id is sound; the debug assertion pins
/// the invariant.
fn bind_consts(root: &mut HashMap<SymId, Value>, f: &AdaptedFn, consts: &[TypedConstDef]) {
    for c in consts {
        debug_assert_eq!(
            f.symbols.get(c.sym.0 as usize).map(|s| s.name.as_str()),
            Some(c.name.as_str()),
            "const arenas must align across functions"
        );
        root.insert(c.sym, Value::from_const(&c.value, c.ty));
    }
}

// ---------------------------------------------------------------------------
// Pure value operations (no interpreter state)
// ---------------------------------------------------------------------------

fn internal(msg: impl Into<String>) -> Stop {
    Stop::Fail(RunError {
        kind: RunErrorKind::Internal(msg.into()),
        span: Span { start: 0, end: 0 },
    })
}

/// Element type behind an array type.
fn elem_ty_of(t: Ty) -> Option<ElemTy> {
    t.elem()
}

/// Integer literal adapted to its slot width.
fn int_lit(v: i64, ty: Ty) -> Value {
    if ty == Ty::I32 {
        Value::I32(v as i32)
    } else {
        Value::I64(v)
    }
}

/// Float literal adapted to its slot width.
fn float_lit(v: f64, ty: Ty) -> Value {
    if ty == Ty::F32 {
        Value::F32(v as f32)
    } else {
        Value::F64(v)
    }
}

fn unary(op: UnOp, v: Value) -> Flow<Value> {
    Ok(match op {
        UnOp::Neg => match v {
            Value::I32(x) => Value::I32(x.wrapping_neg()), // -i32::MIN wraps
            Value::I64(x) => Value::I64(x.wrapping_neg()),
            Value::F32(x) => Value::F32(-x),
            Value::F64(x) => Value::F64(-x),
            other => return Err(internal(format!("negation of {}", other.ty_name()))),
        },
        UnOp::Not => match v {
            Value::Bool(b) => Value::Bool(!b),
            other => return Err(internal(format!("'!' applied to {}", other.ty_name()))),
        },
    })
}

#[allow(clippy::too_many_lines)] // exhaustive width × operator matrix
fn binary(op: BinOp, lv: Value, rv: Value, span: Span) -> Flow<Value> {
    let sym = op.symbol();
    let l_desc = lv.ty_name();
    let r_desc = rv.ty_name();
    let fallback = move || internal(format!("'{sym}' over {l_desc} and {r_desc}"));
    match (op, lv, rv) {
        // Integer arithmetic: wrapping everywhere EXCEPT div/rem (trap below).
        (BinOp::Add, Value::I32(a), Value::I32(b)) => Ok(Value::I32(a.wrapping_add(b))),
        (BinOp::Add, Value::I64(a), Value::I64(b)) => Ok(Value::I64(a.wrapping_add(b))),
        (BinOp::Sub, Value::I32(a), Value::I32(b)) => Ok(Value::I32(a.wrapping_sub(b))),
        (BinOp::Sub, Value::I64(a), Value::I64(b)) => Ok(Value::I64(a.wrapping_sub(b))),
        (BinOp::Mul, Value::I32(a), Value::I32(b)) => Ok(Value::I32(a.wrapping_mul(b))),
        (BinOp::Mul, Value::I64(a), Value::I64(b)) => Ok(Value::I64(a.wrapping_mul(b))),
        (BinOp::Div, Value::I32(a), Value::I32(b)) => {
            // Guarded at i32 width: i32::MIN / -1 overflows i32 and must trap
            // (the JIT's idiv would fault identically), NOT wrap silently.
            if b == 0 {
                return Err(Stop::Fail(RunError::new(RunErrorKind::DivByZero, span)));
            }
            if a == i32::MIN && b == -1 {
                return Err(Stop::Fail(RunError::new(RunErrorKind::IdivOverflow, span)));
            }
            Ok(Value::I32(a / b))
        }
        (BinOp::Div, Value::I64(a), Value::I64(b)) => checked_div(a, b, span).map(Value::I64),
        (BinOp::Rem, Value::I32(a), Value::I32(b)) => {
            // Same edge as division: i32::MIN % -1 is the hardware fault case.
            if b == 0 {
                return Err(Stop::Fail(RunError::new(RunErrorKind::DivByZero, span)));
            }
            if a == i32::MIN && b == -1 {
                return Err(Stop::Fail(RunError::new(RunErrorKind::IdivOverflow, span)));
            }
            Ok(Value::I32(a % b))
        }
        (BinOp::Rem, Value::I64(a), Value::I64(b)) => checked_rem(a, b, span).map(Value::I64),

        // Float arithmetic: IEEE, never traps.
        (BinOp::Add, Value::F32(a), Value::F32(b)) => Ok(Value::F32(a + b)),
        (BinOp::Add, Value::F64(a), Value::F64(b)) => Ok(Value::F64(a + b)),
        (BinOp::Sub, Value::F32(a), Value::F32(b)) => Ok(Value::F32(a - b)),
        (BinOp::Sub, Value::F64(a), Value::F64(b)) => Ok(Value::F64(a - b)),
        (BinOp::Mul, Value::F32(a), Value::F32(b)) => Ok(Value::F32(a * b)),
        (BinOp::Mul, Value::F64(a), Value::F64(b)) => Ok(Value::F64(a * b)),
        (BinOp::Div, Value::F32(a), Value::F32(b)) => Ok(Value::F32(a / b)),
        (BinOp::Div, Value::F64(a), Value::F64(b)) => Ok(Value::F64(a / b)),

        // Comparisons. Sema guarantees equal operand widths.
        (BinOp::Lt, Value::I32(a), Value::I32(b)) => Ok(Value::Bool(a < b)),
        (BinOp::Lt, Value::I64(a), Value::I64(b)) => Ok(Value::Bool(a < b)),
        (BinOp::Lt, Value::F32(a), Value::F32(b)) => Ok(Value::Bool(a < b)),
        (BinOp::Lt, Value::F64(a), Value::F64(b)) => Ok(Value::Bool(a < b)),
        (BinOp::Gt, Value::I32(a), Value::I32(b)) => Ok(Value::Bool(a > b)),
        (BinOp::Gt, Value::I64(a), Value::I64(b)) => Ok(Value::Bool(a > b)),
        (BinOp::Gt, Value::F32(a), Value::F32(b)) => Ok(Value::Bool(a > b)),
        (BinOp::Gt, Value::F64(a), Value::F64(b)) => Ok(Value::Bool(a > b)),
        (BinOp::Le, Value::I32(a), Value::I32(b)) => Ok(Value::Bool(a <= b)),
        (BinOp::Le, Value::I64(a), Value::I64(b)) => Ok(Value::Bool(a <= b)),
        (BinOp::Le, Value::F32(a), Value::F32(b)) => Ok(Value::Bool(a <= b)),
        (BinOp::Le, Value::F64(a), Value::F64(b)) => Ok(Value::Bool(a <= b)),
        (BinOp::Ge, Value::I32(a), Value::I32(b)) => Ok(Value::Bool(a >= b)),
        (BinOp::Ge, Value::I64(a), Value::I64(b)) => Ok(Value::Bool(a >= b)),
        (BinOp::Ge, Value::F32(a), Value::F32(b)) => Ok(Value::Bool(a >= b)),
        (BinOp::Ge, Value::F64(a), Value::F64(b)) => Ok(Value::Bool(a >= b)),

        // Equality. Floats follow IEEE `==` (NaN != anything).
        (BinOp::Eq, Value::I32(a), Value::I32(b)) => Ok(Value::Bool(a == b)),
        (BinOp::Eq, Value::I64(a), Value::I64(b)) => Ok(Value::Bool(a == b)),
        (BinOp::Eq, Value::F32(a), Value::F32(b)) => Ok(Value::Bool(a == b)),
        (BinOp::Eq, Value::F64(a), Value::F64(b)) => Ok(Value::Bool(a == b)),
        (BinOp::Eq, Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(a == b)),
        (BinOp::Ne, Value::I32(a), Value::I32(b)) => Ok(Value::Bool(a != b)),
        (BinOp::Ne, Value::I64(a), Value::I64(b)) => Ok(Value::Bool(a != b)),
        (BinOp::Ne, Value::F32(a), Value::F32(b)) => Ok(Value::Bool(a != b)),
        (BinOp::Ne, Value::F64(a), Value::F64(b)) => Ok(Value::Bool(a != b)),
        (BinOp::Ne, Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(a != b)),

        _ => Err(fallback()),
    }
}

/// Bounds guard: `0 <= i < len`, else a spec-shaped runtime error located at
/// the index expression's span.
fn check_bounds(len: usize, i: i64, span: Span) -> Flow<()> {
    let ok = 0 <= i && usize::try_from(i).is_ok_and(|u| u < len);
    if !ok {
        return Err(Stop::Fail(RunError::new(
            RunErrorKind::Bounds { len, idx: i },
            span,
        )));
    }
    Ok(())
}

/// Saturating/truncating casts — bit-for-bit Rust's `as`.
///
/// The matrix is written out explicitly (rather than through a helper trait)
/// because each of the 16 numeric conversions has its own documented rule:
/// float→int saturates with NaN→0; int→int truncates the two's complement;
/// int↔float rounds toward zero; f32→f64 is exact, f64→f32 rounds to nearest.
fn cast(v: Value, to: Ty) -> Value {
    match (v, to) {
        (Value::I32(x), Ty::I32) => Value::I32(x),
        (Value::I32(x), Ty::I64) => Value::I64(i64::from(x)),
        (Value::I32(x), Ty::F32) => Value::F32(x as f32),
        (Value::I32(x), Ty::F64) => Value::F64(f64::from(x)),
        (Value::I64(x), Ty::I32) => Value::I32(x as i32),
        (Value::I64(x), Ty::I64) => Value::I64(x),
        (Value::I64(x), Ty::F32) => Value::F32(x as f32),
        (Value::I64(x), Ty::F64) => Value::F64(x as f64),
        (Value::F32(x), Ty::I32) => Value::I32(x as i32),
        (Value::F32(x), Ty::I64) => Value::I64(x as i64),
        (Value::F32(x), Ty::F32) => Value::F32(x),
        (Value::F32(x), Ty::F64) => Value::F64(f64::from(x)),
        (Value::F64(x), Ty::I32) => Value::I32(x as i32),
        (Value::F64(x), Ty::I64) => Value::I64(x as i64),
        (Value::F64(x), Ty::F32) => Value::F32(x as f32),
        (Value::F64(x), Ty::F64) => Value::F64(x),
        // bool/array cannot be cast (sema rejects); unreachable here.
        (v, _) => v,
    }
}

/// Checked division computed in i64 (so the i32 overflow edge is caught by
/// the same MIN/-1 test) and narrowed by the caller.
fn checked_div(a: i64, b: i64, span: Span) -> Flow<i64> {
    if b == 0 {
        return Err(Stop::Fail(RunError::new(RunErrorKind::DivByZero, span)));
    }
    if a == i64::MIN && b == -1 {
        return Err(Stop::Fail(RunError::new(RunErrorKind::IdivOverflow, span)));
    }
    Ok(a / b)
}

/// Checked truncated remainder — Rust's `%` already has sign-of-dividend.
fn checked_rem(a: i64, b: i64, span: Span) -> Flow<i64> {
    if b == 0 {
        return Err(Stop::Fail(RunError::new(RunErrorKind::DivByZero, span)));
    }
    if a == i64::MIN && b == -1 {
        return Err(Stop::Fail(RunError::new(RunErrorKind::IdivOverflow, span)));
    }
    Ok(a % b)
}

/// Integer min/max (sema guarantees equal widths).
fn int_minmax_i32(b: Builtin, x: i32, y: i32) -> Value {
    if b == Builtin::Min {
        Value::I32(x.min(y))
    } else {
        Value::I32(x.max(y))
    }
}

/// Integer min/max, 64-bit.
fn int_minmax_i64(b: Builtin, x: i64, y: i64) -> Value {
    if b == Builtin::Min {
        Value::I64(x.min(y))
    } else {
        Value::I64(x.max(y))
    }
}

/// IEEE minNum/maxNum, f32: a NaN operand loses to a real number.
fn float_minmax_f32(b: Builtin, x: f32, y: f32) -> Value {
    let r = if x.is_nan() {
        y
    } else if y.is_nan() {
        x
    } else if b == Builtin::Min {
        x.min(y)
    } else {
        x.max(y)
    };
    Value::F32(r)
}

/// IEEE minNum/maxNum, f64.
fn float_minmax_f64(b: Builtin, x: f64, y: f64) -> Value {
    let r = if x.is_nan() {
        y
    } else if y.is_nan() {
        x
    } else if b == Builtin::Min {
        x.min(y)
    } else {
        x.max(y)
    };
    Value::F64(r)
}
