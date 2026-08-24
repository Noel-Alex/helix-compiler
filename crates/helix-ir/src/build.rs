//! Lowering the typed tree ([`TypedProgram`]) into CFG form ([`FuncIr`]).
//!
//! The builder is recursive descent over statements tracking a *current*
//! block. Control constructs follow canonical shapes:
//!
//! ```text
//! if c { T } else { E }:        for i in s..e { B }:
//!
//!      cond                          pre:  iv_cell = eval(s); end_t = eval(e)
//!      /   \                               jump hdr(iv_cell)
//!   Branch                                hdr ←────────────┐
//!    /    \                     iv = φ(pre: iv0, lat: iv+1)
//!  then   else                  cond = iv < end ; Branch(body, exit)
//!    \     /                          body ───────────┐
//!     \   /                                 lat: iv+1 ─┘
//!      merge                          exit:
//! ```
//!
//! ## The cell convention
//!
//! Every source variable occupies one *cell*: [`ValueId`] equal to its
//! [`LocalId`]. An assignment lowers its right-hand side so that the **root
//! definition lands directly on the cell** (`let x = 5` becomes
//! `Inst::Const { dst: cell(x), c: 5 }`), and every read spells the bare cell
//! id. Consequences:
//!
//! * straight-line code is trivially correct — the latest def shadows older
//!   ones in program order;
//! * at joins, two paths may reach a use through different defs, which is
//!   exactly the shape [`crate::ssa::to_ssa`] exists to repair: it classifies
//!   such variables as *global*, places φ-nodes, and renames each def to a
//!   fresh SSA name;
//! * compiler temporaries follow the same rule — `$sc` cells carry
//!   short-circuit results, `$ret` accumulates return values, loop induction
//!   variables are plain cells — so the whole function converts through one
//!   mechanism.
//!
//! The one construct that cannot define a cell directly is copying a variable
//! (`let y = x`) — HELIX has no move instruction. It lowers to the identity
//! computation `y = x + 0`; downstream `copy_prop` removes the arithmetic.
//!
//! # Known upstream interface gap
//!
//! `helix_sema::TypedExprKind::Call` wraps a `CallTarget` that records the
//! callee but **not the argument expressions** (sema type-checks arguments
//! and discards them). Until sema is extended, this builder emits
//! `Inst::Call` with an empty scalar `args` list; the callee, destination and
//! array references (`zeros`) are modelled faithfully. Everything else about
//! the lowering is complete.

use std::collections::HashSet;

use helix_sema::{
    Builtin, CallTarget, ConstLit, ElseArm, Ty, TypedBlock, TypedExpr, TypedExprKind, TypedFnDef,
    TypedLValue, TypedProgram, TypedStmt,
};
use helix_syntax::ast::{BinOp, UnOp};

use crate::ir::{BlockId, Call, Constant, FuncIr, Inst, Load, LocalId, Phi, Term, ValueId};

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

/// Build IR for every function of the program, preserving source order.
#[must_use]
pub fn build(program: &TypedProgram) -> Vec<FuncIr> {
    (0..program.funcs.len())
        .map(|i| build_fn(program, i))
        .collect()
}

/// Scratch counted by a quick pre-scan so every temporary slot is reserved
/// *before* value allocation begins (keeping the fresh-value cursor strictly
/// above the cell range).
struct Reservations {
    /// One bool cell per short-circuit operator occurrence.
    short_circuits: usize,
}

fn reserve_scan(f: &TypedFnDef) -> Reservations {
    let mut r = Reservations { short_circuits: 0 };
    scan_stmts(&f.body.stmts, &mut r);
    r
}

fn scan_stmts(stmts: &[TypedStmt], r: &mut Reservations) {
    for s in stmts {
        scan_stmt(s, r);
    }
}

fn scan_stmt(s: &TypedStmt, r: &mut Reservations) {
    match s {
        TypedStmt::Let { init, .. } | TypedStmt::Effect(init) => scan_expr(init, r),
        TypedStmt::Assign { target, value } => {
            scan_expr(value, r);
            if let Some(i) = &target.index {
                scan_expr(i, r);
            }
        }
        TypedStmt::If(f) => {
            scan_expr(&f.cond, r);
            scan_stmts(&f.then_blk.stmts, r);
            match &f.else_arm {
                Some(ElseArm::Block(b)) => scan_stmts(&b.stmts, r),
                Some(ElseArm::If(inner)) => {
                    scan_stmt(&TypedStmt::If(Box::new((**inner).clone())), r);
                }
                None => {}
            }
        }
        TypedStmt::For(f) => {
            scan_expr(&f.start, r);
            scan_expr(&f.end, r);
            scan_stmts(&f.body.stmts, r);
        }
        TypedStmt::Return { value, .. } => {
            if let Some(v) = value {
                scan_expr(v, r);
            }
        }
    }
}

fn scan_expr(e: &TypedExpr, r: &mut Reservations) {
    match &e.kind {
        TypedExprKind::Bin(op, l, rr) => {
            if matches!(op, BinOp::And | BinOp::Or) {
                r.short_circuits += 1;
            }
            scan_expr(l, r);
            scan_expr(rr, r);
        }
        TypedExprKind::Unary(_, o) | TypedExprKind::Cast(o, _) => scan_expr(o, r),
        TypedExprKind::Index(_, i) => scan_expr(i, r),
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Per-function construction
// ---------------------------------------------------------------------------

fn build_fn(program: &TypedProgram, fidx: usize) -> FuncIr {
    let f = &program.funcs[fidx];
    let n_source_locals = f.symbols.len();
    let mut ir = FuncIr::new(&f.name, f.ret, n_source_locals);

    // Mirror the sema arena (names/types) and preset the type table for the
    // cell range [0, n_source_locals).
    for (i, sym) in f.symbols.iter().enumerate() {
        ir.declare_local(LocalId(i as u32), sym.ty, &sym.name);
        if ir.types.val_tys.len() <= i {
            ir.types.val_tys.resize(i + 1, sym.ty);
        }
        ir.types.val_tys[i] = sym.ty;
    }

    // ---- reserve compiler-temporary cells BEFORE allocating values ---------
    let want = reserve_scan(f);
    let mut sc_temps: Vec<LocalId> = Vec::new();
    for _ in 0..want.short_circuits {
        sc_temps.push(alloc_temp(&mut ir, "$sc", Ty::Bool));
    }
    // Value-returning functions get one `$ret` accumulator cell; every return
    // statement defines it and the shared exit block's φ merges the edges.
    let ret_slot = if f.ret == Ty::Unit {
        None
    } else {
        Some(alloc_temp(&mut ir, "$ret", f.ret))
    };

    // Fresh value allocation starts strictly above every reserved slot.
    ir.next_value = ir.n_locals as u32;

    // ---- entry block: top-level constant definitions -----------------------
    let mut const_defs = std::collections::HashMap::new();
    for c in &program.consts {
        let v = ir.new_value(c.ty);
        let k = match (&c.value, c.ty) {
            (ConstLit::Int(v), Ty::I32) => Constant::I32(*v as i32),
            (ConstLit::Int(v), _) => Constant::I64(*v),
            (ConstLit::Float(v), Ty::F32) => Constant::F32(*v as f32),
            (ConstLit::Float(v), _) => Constant::F64(*v),
            (ConstLit::Bool(b), _) => Constant::Bool(*b),
        };
        ir.block_mut(ir.entry)
            .insts
            .push(Inst::Const { dst: v, c: k });
        const_defs.insert(c.sym, v);
    }

    let mut b = Builder {
        ir,
        cur: BlockId(0),
        closed: HashSet::new(),
        const_defs,
        sc_temps,
        ret_slot,
        exit_block: None,
        n_return_sites: 0,
    };

    // Parameter definitions: argument-less entry phis act as block params.
    for (sym, _ty) in &f.params {
        b.define_param(*sym);
    }

    b.block(&f.body);

    // Close the tail. Cases:
    //   * the body ended with `return` — control funneled into the exit block
    //     already;
    //   * the body fell through — a value function is rejected by sema, but we
    //     keep the IR total by jumping to the exit; a unit function gets the
    //     implicit `return;`.
    if !b.terminated() {
        if f.ret == Ty::Unit {
            let v = None;
            b.finish_return(v);
        } else {
            let zero = b.val(f.ret);
            b.emit(Inst::Const {
                dst: zero,
                c: match f.ret {
                    Ty::I32 => Constant::I32(0),
                    Ty::F32 => Constant::F32(0.0),
                    Ty::F64 => Constant::F64(0.0),
                    _ => Constant::I64(0),
                },
            });
            b.finish_return(Some(zero));
        }
    }
    // Terminate the shared exit block now that every return edge is known.
    b.close_exit();

    b.ir.normalize_phis();
    b.ir.recompute_edges();
    b.ir
}

/// Allocate one compiler-temporary cell above the current high-water mark.
fn alloc_temp(ir: &mut FuncIr, tag: &str, ty: Ty) -> LocalId {
    let l = LocalId(ir.n_locals as u32);
    ir.types.local_tys.push(ty);
    ir.types.local_names.push(format!("{tag}{}", l.0));
    ir.n_locals += 1;
    if ir.types.val_tys.len() <= l.0 as usize {
        ir.types.val_tys.resize(l.0 as usize + 1, ty);
    }
    l
}

// ---------------------------------------------------------------------------
// Builder state
// ---------------------------------------------------------------------------

struct Builder {
    ir: FuncIr,
    /// Block currently receiving emissions.
    cur: BlockId,
    /// Blocks that received a real terminator.
    closed: HashSet<BlockId>,
    /// Per-function value of each top-level constant.
    const_defs: std::collections::HashMap<helix_sema::SymId, ValueId>,
    /// Short-circuit result cells in evaluation order.
    sc_temps: Vec<LocalId>,
    /// Return accumulator cell (value-returning functions only).
    ret_slot: Option<LocalId>,
    /// Lazily created shared exit block (early-return funnel).
    exit_block: Option<BlockId>,
    /// Number of return statements seen so far.
    n_return_sites: usize,
}

impl Builder {
    // -- plumbing ------------------------------------------------------------

    fn new_block(&mut self) -> BlockId {
        self.ir.new_block()
    }

    fn val(&mut self, ty: Ty) -> ValueId {
        self.ir.new_value(ty)
    }

    fn emit(&mut self, inst: Inst) {
        self.ir.block_mut(self.cur).insts.push(inst);
    }

    fn jump(&mut self, target: BlockId) {
        self.ir.set_term(self.cur, Term::Jump(target, Vec::new()));
        self.closed.insert(self.cur);
    }

    fn branch(&mut self, cond: ValueId, t: BlockId, f: BlockId) {
        self.ir.set_term(self.cur, Term::Branch { cond, t, f });
        self.closed.insert(self.cur);
    }

    fn ret(&mut self, v: Option<ValueId>) {
        self.ir.set_term(self.cur, Term::Return(v));
        self.closed.insert(self.cur);
    }

    fn terminated(&self) -> bool {
        self.closed.contains(&self.cur)
    }

    /// Declare a parameter as an argument-less entry phi (Cranelift block
    /// parameter analogue).
    fn define_param(&mut self, sym: helix_sema::SymId) {
        let l = LocalId(sym.0);
        self.ir.block_mut(self.ir.entry).phis.push(Phi {
            dst: ValueId(l.0),
            var: l,
            args: Vec::new(),
        });
    }

    // -- expression lowering ---------------------------------------------------

    /// Evaluate `e`, yielding the id of a freshly-defined value.
    fn expr(&mut self, e: &TypedExpr) -> ValueId {
        match &e.kind {
            // Variable reads ARE their cell — no fresh def needed.
            TypedExprKind::Var(sym) => {
                if let Some(v) = self.const_defs.get(sym) {
                    *v
                } else {
                    ValueId(sym.0)
                }
            }
            // Arrays have no first-class value; callers use arr_refs instead.
            TypedExprKind::ArrayRef(sym) => ValueId(sym.0),
            _ => {
                let tmp = self.val(e.ty);
                self.expr_into(e, tmp);
                tmp
            }
        }
    }

    /// Evaluate `e` so its **root definition** is `dst`. Intermediate
    /// sub-expressions still receive fresh temporaries.
    fn expr_into(&mut self, e: &TypedExpr, dst: ValueId) {
        match &e.kind {
            TypedExprKind::IntLit(_) | TypedExprKind::FloatLit(_) | TypedExprKind::BoolLit(_) => {
                let c = self.literal(e);
                self.emit(Inst::Const { dst, c });
            }
            TypedExprKind::Var(sym) | TypedExprKind::ArrayRef(sym) => {
                // Copy a variable into a new cell via an identity computation
                // (`copy_prop` reduces it afterwards).
                let ty = self.ir.types.val_tys.get(dst.0 as usize).copied();
                let src = match self.const_defs.get(sym) {
                    Some(v) => *v,
                    None => ValueId(sym.0),
                };
                self.copy_value(dst, src, scalar_ty(ty));
            }
            TypedExprKind::Unary(op, o) => {
                let a = self.expr(o);
                self.emit(Inst::Unary { op: *op, dst, a });
            }
            TypedExprKind::Bin(op, l, r) => {
                if matches!(op, BinOp::And | BinOp::Or) {
                    self.short_circuit(*op, l, r, dst);
                } else {
                    let a = self.expr(l);
                    let b = self.expr(r);
                    self.emit(Inst::Bin { op: *op, dst, a, b });
                }
            }
            TypedExprKind::Index(arr, idx) => {
                let i = self.expr(idx);
                self.emit(Inst::Load(Load {
                    dst,
                    arr: LocalId(arr.0),
                    idx: i,
                }));
            }
            TypedExprKind::Cast(o, to) => {
                let v = self.expr(o);
                self.emit(Inst::Cast {
                    dst,
                    val: v,
                    to: *to,
                });
            }
            TypedExprKind::Call(target) => {
                self.call_expr(target, e.ty, Some(dst));
            }
            TypedExprKind::Error => {
                // Poison from failed sema recovery; well-typed inputs never
                // reach the builder. Zero keeps downstream typing total.
                self.emit(Inst::Const {
                    dst,
                    c: Constant::I64(0),
                });
            }
        }
    }

    /// Constant payload of a literal expression (types already resolved by
    /// sema's bidirectional checking).
    fn literal(&self, e: &TypedExpr) -> Constant {
        match &e.kind {
            TypedExprKind::IntLit(v) => {
                if e.ty == Ty::I32 {
                    Constant::I32(*v as i32)
                } else {
                    Constant::I64(*v)
                }
            }
            TypedExprKind::FloatLit(v) => {
                if e.ty == Ty::F32 {
                    Constant::F32(*v as f32)
                } else {
                    Constant::F64(*v)
                }
            }
            TypedExprKind::BoolLit(b) => Constant::Bool(*b),
            _ => Constant::I64(0),
        }
    }

    /// Short-circuit diamond. `a && b` branches `a ? rhs : merge`;
    /// `a || b` branches `a ? merge : rhs`. Both operands are defined **into
    /// the same `$sc` cell**, so whichever path reaches the merge determines
    /// the observed value — and the rhs genuinely executes only on its own
    /// path (observable through side-effecting calls). `to_ssa` places the
    /// merge φ over the cell; the builder stays phi-free.
    ///
    /// ```text
    /// sc_cell = eval(a)
    /// Branch(sc_cell, rhs, merge)          // && polarity; || mirrors
    /// rhs:  sc_cell = eval(b); jump merge
    /// merge: dst = copy(sc_cell)           // to_ssa turns this into a φ
    /// ```
    fn short_circuit(&mut self, op: BinOp, l: &TypedExpr, r: &TypedExpr, dst: ValueId) {
        // Pop the reserved cell for this operator (evaluation order).
        let sc = self.sc_temps.pop().expect("short-circuit reservation");
        let sc_cell = ValueId(sc.0);

        // Left operand defines the cell in the current block.
        self.expr_into(l, sc_cell);

        let merge = self.new_block();
        let rhs_b = self.new_block();
        let (t, f) = match op {
            BinOp::And => (rhs_b, merge),
            _ => (merge, rhs_b),
        };
        self.branch(sc_cell, t, f);

        // Right path: redefines the same cell.
        self.cur = rhs_b;
        self.expr_into(r, sc_cell);
        self.jump(merge);

        // Merge: copy the (phi-to-be) cell into the caller's destination.
        self.cur = merge;
        if dst != sc_cell {
            self.copy_value(dst, sc_cell, Ty::Bool);
        }
    }

    /// Emit `dst = copy(src)` for a scalar of type `ty`.
    ///
    /// HELIX has no move instruction, so copies are expressed as an identity
    /// computation: numeric types add a zero of their own width; bool doubles
    /// through `!` (`!!b == b`). `copy_prop` reduces both shapes afterwards.
    fn copy_value(&mut self, dst: ValueId, src: ValueId, ty: Ty) {
        match ty {
            Ty::Bool => {
                let t1 = self.val(Ty::Bool);
                self.emit(Inst::Unary {
                    op: UnOp::Not,
                    dst: t1,
                    a: src,
                });
                self.emit(Inst::Unary {
                    op: UnOp::Not,
                    dst,
                    a: t1,
                });
            }
            ty => {
                let zero = self.val(ty);
                let c = match ty {
                    Ty::F32 => Constant::F32(0.0),
                    Ty::F64 => Constant::F64(0.0),
                    Ty::I32 => Constant::I32(0),
                    _ => Constant::I64(0),
                };
                self.emit(Inst::Const { dst: zero, c });
                self.emit(Inst::Bin {
                    op: BinOp::Add,
                    dst,
                    a: src,
                    b: zero,
                });
            }
        }
    }

    /// Calls. See the module-level note about the missing argument list.
    fn call_expr(&mut self, target: &CallTarget, ret: Ty, dst: Option<ValueId>) {
        match target {
            CallTarget::Builtin {
                which: Builtin::Zeros,
                ..
            } => {
                // The array lands in the destination cell directly (arrays are
                // referenced by local slot, never moved as values).
                let out_local = dst
                    .map(|d| LocalId(d.0))
                    .unwrap_or_else(|| alloc_temp(&mut self.ir, "$arr", ret));
                self.emit(Inst::Call(Call {
                    dst: None,
                    callee: "zeros".into(),
                    args: Vec::new(),
                    arr_refs: vec![out_local],
                }));
            }
            CallTarget::Builtin { which: b, .. } => {
                self.emit(Inst::Call(Call {
                    dst,
                    callee: b.name().into(),
                    args: Vec::new(),
                    arr_refs: Vec::new(),
                }));
            }
            CallTarget::User { name, .. } => {
                self.emit(Inst::Call(Call {
                    dst,
                    callee: name.clone(),
                    args: Vec::new(),
                    arr_refs: Vec::new(),
                }));
            }
        }
    }

    // -- statements ------------------------------------------------------------

    fn block(&mut self, blk: &TypedBlock) {
        for s in &blk.stmts {
            if self.terminated() {
                // Statements after a return are unreachable; skip them so the
                // terminator stays last in the block.
                break;
            }
            self.stmt(s);
        }
    }

    fn stmt(&mut self, s: &TypedStmt) {
        if self.terminated() {
            return;
        }
        match s {
            TypedStmt::Let { sym, init, ty, .. } => {
                let cell = ValueId(sym.0);
                if init_is_zeros(init) {
                    // Arrays bind by reference: the call writes the cell.
                    self.call_expr(
                        &CallTarget::Builtin {
                            which: Builtin::Zeros,
                            args: Vec::new(),
                        },
                        *ty,
                        Some(cell),
                    );
                } else {
                    self.expr_into(init, cell);
                }
            }
            TypedStmt::Assign { target, value } => self.assign(target, value),
            TypedStmt::If(f) => self.if_stmt(f),
            TypedStmt::For(f) => self.for_stmt(f),
            TypedStmt::Return { value, .. } => self.return_stmt(value.as_ref()),
            TypedStmt::Effect(e) => {
                self.effect_expr(e);
            }
        }
    }

    /// Evaluate an expression for effects only (calls, discarded values).
    fn effect_expr(&mut self, e: &TypedExpr) {
        match &e.kind {
            TypedExprKind::Call(target) => {
                self.call_expr(target, e.ty, None);
            }
            other_kind => {
                let _ = other_kind;
                let tmp = self.val(e.ty);
                self.expr_into(e, tmp);
                // Dead value; DCE removes pure leftovers, effects remain.
            }
        }
    }

    fn assign(&mut self, target: &TypedLValue, value: &TypedExpr) {
        match &target.index {
            Some(idx) => {
                let v = self.expr(value);
                let i = self.expr(idx);
                self.emit(Inst::Store {
                    arr: LocalId(target.base.0),
                    idx: i,
                    val: v,
                });
            }
            None => {
                let cell = ValueId(target.base.0);
                self.expr_into(value, cell);
            }
        }
    }

    /// `return e;` defines the `$ret` accumulator, then jumps to the shared
    /// exit block passing that cell as the edge argument. The exit φ merges
    /// one value per return edge, and the function's single `Term::Return`
    /// lives there — the canonical "dedicated exit block" shape.
    fn return_stmt(&mut self, value: Option<&TypedExpr>) {
        self.n_return_sites += 1;
        match (self.ret_slot, value) {
            (Some(slot), Some(e)) => {
                // Evaluate into the accumulator cell so every return edge
                // passes the same well-defined cell to the exit φ.
                self.expr_into(e, ValueId(slot.0));
            }
            (Some(_slot), None) => {
                // Malformed input rejected by sema; evaluate for effects.
                if let Some(e) = value {
                    self.expr(e);
                }
            }
            (None, Some(e)) => {
                self.expr(e);
            }
            (None, None) => {}
        }
        let exit = self.ensure_exit();
        // Register this edge in the exit phi with the actual value passed.
        if let Some(slot) = self.ret_slot {
            let from = self.cur;
            for p in self.ir.block_mut(exit).phis.iter_mut() {
                if !p.args.iter().any(|(b, _)| *b == from) {
                    p.args.push((from, ValueId(slot.0)));
                    p.args.sort_unstable_by_key(|(b, _)| *b);
                }
            }
        }
        let args: Vec<ValueId> = match self.ret_slot {
            Some(slot) => vec![ValueId(slot.0)],
            None => Vec::new(),
        };
        self.ir.set_term(self.cur, Term::Jump(exit, args));
        self.closed.insert(self.cur);
    }

    /// Fallback return used when control falls off the end of the body.
    fn finish_return(&mut self, _v: Option<ValueId>) {
        let exit = self.ensure_exit();
        if let Some(slot) = self.ret_slot {
            let from = self.cur;
            for p in self.ir.block_mut(exit).phis.iter_mut() {
                if !p.args.iter().any(|(b, _)| *b == from) {
                    p.args.push((from, ValueId(slot.0)));
                    p.args.sort_unstable_by_key(|(b, _)| *b);
                }
            }
        }
        let args: Vec<ValueId> = match self.ret_slot {
            Some(slot) => vec![ValueId(slot.0)],
            None => Vec::new(),
        };
        self.ir.set_term(self.cur, Term::Jump(exit, args));
        self.closed.insert(self.cur);
    }

    /// Terminate the shared exit block after all edges are known: its single
    /// φ argument list mirrors the return sites, and `Return` yields the
    /// merged value (or nothing for unit functions).
    fn close_exit(&mut self) {
        let Some(exit) = self.exit_block else {
            return;
        };
        if self.closed.contains(&exit) {
            return;
        }
        self.cur = exit;
        let v = self.ret_slot.map(|slot| ValueId(slot.0));
        self.ret(v);
    }

    /// Lazily create the shared exit block with a `$ret` merge φ; every jump
    /// into it passes the current value of the accumulator cell positionally.
    fn ensure_exit(&mut self) -> BlockId {
        if let Some(e) = self.exit_block {
            return e;
        }
        let e = self.new_block();
        if let Some(slot) = self.ret_slot {
            self.ir.block_mut(e).phis.push(Phi {
                dst: ValueId(slot.0),
                var: slot,
                args: Vec::new(),
            });
        }
        self.exit_block = Some(e);
        e
    }

    /// if/else diamond; `else if` chains recurse, sharing one merge block.
    fn if_stmt(&mut self, f: &helix_sema::TypedIf) {
        let cond = self.expr(&f.cond);
        let then_b = self.new_block();
        let merge = self.new_block();
        let else_b = match &f.else_arm {
            Some(_) => self.new_block(),
            None => merge,
        };
        self.branch(cond, then_b, else_b);

        self.cur = then_b;
        self.block(&f.then_blk);
        if !self.terminated() {
            self.jump(merge);
        }

        if let Some(arm) = &f.else_arm {
            self.cur = else_b;
            match arm {
                ElseArm::Block(b) => self.block(b),
                ElseArm::If(inner) => self.if_stmt(inner),
            }
            if !self.terminated() {
                self.jump(merge);
            }
        }

        self.cur = merge;
    }

    /// Canonical for-loop lowering (see module docs). `start` defines the iv
    /// cell in the preheader; `end` is evaluated once; the latch redefines the
    /// iv cell with `iv = iv + 1`. `to_ssa` later inserts the header φ that
    /// merges the two incoming iv values; the builder stays phi-free.
    fn for_stmt(&mut self, fw: &helix_sema::TypedFor) {
        let iv_cell = ValueId(LocalId(fw.iv.0).0);

        // Preheader part (in the current block).
        self.expr_into(&fw.start, iv_cell);
        let end = self.expr(&fw.end);

        let header = self.new_block();
        let body_b = self.new_block();
        let latch = self.new_block();
        let exit = self.new_block();

        self.jump(header);

        // Header: cond = iv < end.
        self.cur = header;
        let cond = self.val(Ty::Bool);
        self.emit(Inst::Bin {
            op: BinOp::Lt,
            dst: cond,
            a: iv_cell,
            b: end,
        });
        self.branch(cond, body_b, exit);

        // Body.
        self.cur = body_b;
        self.block(&fw.body);
        if !self.terminated() {
            self.jump(latch);
        }

        // Latch: iv = iv + 1 (defines the cell so the back edge carries it).
        self.cur = latch;
        let one = self.val(Ty::I64);
        self.emit(Inst::Const {
            dst: one,
            c: Constant::I64(1),
        });
        self.emit(Inst::Bin {
            op: BinOp::Add,
            dst: iv_cell,
            a: iv_cell,
            b: one,
        });
        self.jump(header);

        self.cur = exit;
    }
}

/// Does this expression initialize an array from `zeros(n)`?
fn init_is_zeros(e: &TypedExpr) -> bool {
    matches!(
        &e.kind,
        TypedExprKind::Call(CallTarget::Builtin {
            which: Builtin::Zeros,
            ..
        })
    )
}

/// Scalar type of a value whose recorded type may be missing or array-shaped;
/// used by [`Builder::copy_value`] to pick an identity computation of the
/// right width.
fn scalar_ty(ty: Option<Ty>) -> Ty {
    match ty {
        Some(t) if t.is_scalar() && !t.is_unit() => t,
        _ => Ty::I64,
    }
}
