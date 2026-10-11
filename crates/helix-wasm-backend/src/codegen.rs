//! Lowering one post-SSA [`FuncIr`] into a WebAssembly function body.
//!
//! ## Value representation
//!
//! Every IR value owns a wasm local of its own width: `i64 → i64`, `i32 → i32`,
//! `f32 → f32`, `f64 → f64`, `bool → i32` (0/1). The mapping is one-to-one
//! with the post-SSA id space, so there is no register allocator to get wrong:
//! HELIX's optimizing pipeline has already run, and the browser lab needs a
//! module whose observable behaviour matches the interpreter exactly, not the
//! last spill squeezed out of it.
//!
//! Arrays are not values in wasm. An array binding is modelled as a pair of
//! `i32` locals — base pointer and element count — that stand in for the
//! interpreter's `Rc<Vec<Value>>` handle: `Inst::Load`/`Inst::Store` name the
//! array by local slot, and an array argument to a user call is passed as that
//! same pair, so a callee's stores escape to the caller exactly as the
//! specification demands.
//!
//! Linear memory layout (one bump region, based above the host's print
//! scratch):
//!
//! ```text
//!   base+0 .. base+4 : element count (i32)
//!   base+4 .. base+8 : padding
//!   base+8 ..        : elements, each `elem_size` bytes wide
//! ```
//!
//! ## Control flow: a dispatch loop instead of unstructured branching
//!
//! wasm has no `goto`: a branch can only leave an *enclosing* label, so a CFG
//! with cross edges cannot be written as a flat sequence of labelled blocks.
//! Rather than restructure the graph — a relooper, loop forests, block
//! duplication, each of which adds its own failure modes — every function is
//! emitted as one dispatch loop:
//!
//! ```text
//!   $pc = entry
//!   loop $dispatch
//!     block $bb{n-1} … block $bb0          ; labels, innermost last
//!       br_table $pc → $bb0 … $bb{n-1}     ; a branch leaves one label and
//!       unreachable                        ; lands on that block's code
//!     end $bb0 ; <code of bb0> ; end $bb1 ; <code of bb1> ; … ; end $loop
//! ```
//!
//! A terminated block writes its successor into `$pc` and re-enters the loop,
//! so every edge — forward, backward or cross — costs one `local.set` plus a
//! jump-table branch. A successor that is the next block in emission order
//! falls through and pays nothing, which is the common case inside loop
//! bodies. This is the honest price of a backend that never reorders or
//! duplicates user code.
//!
//! ## Traps
//!
//! Every trapping operation calls the imported `hx_trap(code, a, b)` rather
//! than letting wasm's built-in `div_s` fault, so the host renders the
//! interpreter's exact message (with the printed prefix intact) instead of a
//! bare "unreachable executed".

use std::collections::HashMap;

use helix_ir::{BinOp, BlockId, Constant, FuncIr, Inst, LocalId, Term, UnOp, ValueId};
use helix_sema::{ElemTy, Ty};

use super::EmitError;
use super::encode::{Body, F32, F64, I32, I64, push_u32};

/// Function indices of the imported host hooks.
pub(super) const IMPORT_PRINT_I32: u32 = 0;
pub(super) const IMPORT_PRINT_I64: u32 = 1;
pub(super) const IMPORT_PRINT_F32: u32 = 2;
pub(super) const IMPORT_PRINT_F64: u32 = 3;
pub(super) const IMPORT_PRINT_BOOL: u32 = 4;
pub(super) const IMPORT_TRAP: u32 = 5;
/// Function indices of the internal helpers that follow the imports.
pub(super) const FN_FOLD: u32 = 6;
pub(super) const FN_ALLOC: u32 = 7;
/// First function index owned by the IR-derived functions, in source order.
pub(super) const FIRST_USER_FN: u32 = 8;

/// Byte offset of the element region inside an array header.
pub(super) const ELEMS_OFFSET: i32 = 8;

/// Module globals. 0 is the bump pointer, 1 the running checksum, 2 the bytes
/// allocated by the current run, and 3 the number of final arrays recorded in
/// the fold table. `hx_run` resets all four.
pub(super) const GLOBAL_HEAP: u32 = 0;
pub(super) const GLOBAL_CHECKSUM: u32 = 1;
pub(super) const GLOBAL_BYTES: u32 = 2;
pub(super) const GLOBAL_FOLDS: u32 = 3;

/// Linear-memory layout. The host's print scratch occupies `[0, 4096)`, the
/// array fold table `[4096, 8192)`, and the bump heap starts above both.
pub(super) const FOLD_TABLE_BASE: i32 = 4096;
/// Bytes per table entry: element base, then byte length.
pub(super) const FOLD_TABLE_STRIDE: i32 = 8;
/// How many final arrays one program may fold into the state checksum. A
/// program that declares more is refused at emission time rather than
/// silently folding a subset.
pub(super) const FOLD_TABLE_CAPACITY: usize = 256;

/// Trap codes; the host renders the matching interpreter message.
pub(super) const TRAP_DIV_ZERO: i32 = 0;
pub(super) const TRAP_IDIV_OVERFLOW: i32 = 1;
pub(super) const TRAP_BOUNDS: i32 = 2;
pub(super) const TRAP_NEGATIVE_ZEROS: i32 = 3;
pub(super) const TRAP_ALLOC: i32 = 5;

/// Depth budget for the nested-label dispatch. wasm engines refuse to validate
/// deeply nested blocks; 512 covers every example in the repository (the
/// largest needs 16 blocks) with room to spare for hand-written programs.
pub(super) const MAX_DISPATCH_BLOCKS: usize = 512;

/// Byte size of one element, used for addressing and checksum folding.
fn elem_size(elem: ElemTy) -> i32 {
    match elem {
        ElemTy::I32 | ElemTy::F32 => 4,
        ElemTy::I64 | ElemTy::F64 => 8,
        ElemTy::Bool => 1,
    }
}

/// `(opcode, alignment, offset)` of the load for one element type.
///
/// A memory instruction carries both an alignment and an address offset; the
/// element address already includes the header size, so the offset is always
/// zero and the alignment is the element's natural alignment.
fn load_op(elem: ElemTy) -> (u8, u8, u8) {
    match elem {
        ElemTy::I32 => (0x28, 2, 0),
        ElemTy::I64 => (0x29, 3, 0),
        ElemTy::F32 => (0x2A, 2, 0),
        ElemTy::F64 => (0x2B, 3, 0),
        ElemTy::Bool => (0x2D, 0, 0), // i32.load8_u
    }
}

/// `(opcode, alignment, offset)` of the store for one element type.
fn store_op(elem: ElemTy) -> (u8, u8, u8) {
    match elem {
        ElemTy::I32 => (0x36, 2, 0),
        ElemTy::I64 => (0x37, 3, 0),
        ElemTy::F32 => (0x38, 2, 0),
        ElemTy::F64 => (0x39, 3, 0),
        ElemTy::Bool => (0x3A, 0, 0), // i32.store8
    }
}

/// Array locals of `main` whose `zeros` call unconditionally reaches the end of
/// the function — the bindings in `main`'s root scope, which is exactly the set
/// the interpreter folds into its final state checksum.
///
/// A call inside a loop body or inside an `if` arm does not dominate the exit
/// block, so it is excluded; that matches the interpreter, whose nested scopes
/// are popped before the checksum is taken. The one shape the IR cannot see is
/// a *bare* nested `{ … }` block: the builder inlines it into the surrounding
/// block, so such an array is folded here although the interpreter's scope
/// stack has already dropped it.
#[must_use]
pub(super) fn folded_arrays(main: &FuncIr) -> Vec<LocalId> {
    let Some(exit) = main
        .blocks
        .iter()
        .position(|block| matches!(block.term, Term::Return(_)))
    else {
        return Vec::new();
    };
    let doms = helix_ir::dominators(main);
    let exit = BlockId(exit as u32);
    let mut out: Vec<LocalId> = main
        .blocks
        .iter()
        .enumerate()
        .filter(|(index, _)| doms.dominates(BlockId(*index as u32), exit))
        .flat_map(|(_, block)| {
            block
                .insts
                .iter()
                .filter_map(|inst| match inst {
                    Inst::Call(call) if call.callee == "zeros" => call.arr_refs.first().copied(),
                    _ => None,
                })
                .collect::<Vec<_>>()
        })
        .collect();
    out.sort_unstable_by_key(|local| local.0);
    out.dedup();
    out
}

/// Non-trapping binary opcode for one operator and operand width.
fn bin_opcode(op: BinOp, ty: Ty) -> Result<&'static [u8], EmitError> {
    let code: &'static [u8] = match (op, ty) {
        (BinOp::Add, Ty::I32) => &[0x6A],
        (BinOp::Sub, Ty::I32) => &[0x6B],
        (BinOp::Mul, Ty::I32) => &[0x6C],
        (BinOp::Add, Ty::I64) => &[0x7C],
        (BinOp::Sub, Ty::I64) => &[0x7D],
        (BinOp::Mul, Ty::I64) => &[0x7E],
        (BinOp::Add, Ty::F32) => &[0x92],
        (BinOp::Sub, Ty::F32) => &[0x93],
        (BinOp::Mul, Ty::F32) => &[0x94],
        (BinOp::Div, Ty::F32) => &[0x95],
        (BinOp::Add, Ty::F64) => &[0xA0],
        (BinOp::Sub, Ty::F64) => &[0xA1],
        (BinOp::Mul, Ty::F64) => &[0xA2],
        (BinOp::Div, Ty::F64) => &[0xA3],
        // Comparisons yield an i32 bool; equality also applies to bools.
        (BinOp::Lt, Ty::I32) => &[0x48],
        (BinOp::Gt, Ty::I32) => &[0x4A],
        (BinOp::Le, Ty::I32) => &[0x4C],
        (BinOp::Ge, Ty::I32) => &[0x4E],
        (BinOp::Eq, Ty::I32 | Ty::Bool) => &[0x46],
        (BinOp::Ne, Ty::I32 | Ty::Bool) => &[0x47],
        (BinOp::Lt, Ty::I64) => &[0x53],
        (BinOp::Gt, Ty::I64) => &[0x55],
        (BinOp::Le, Ty::I64) => &[0x57],
        (BinOp::Ge, Ty::I64) => &[0x59],
        (BinOp::Eq, Ty::I64) => &[0x51],
        (BinOp::Ne, Ty::I64) => &[0x52],
        (BinOp::Lt, Ty::F32) => &[0x5D],
        (BinOp::Gt, Ty::F32) => &[0x5E],
        (BinOp::Le, Ty::F32) => &[0x5F],
        (BinOp::Ge, Ty::F32) => &[0x60],
        (BinOp::Eq, Ty::F32) => &[0x5B],
        (BinOp::Ne, Ty::F32) => &[0x5C],
        (BinOp::Lt, Ty::F64) => &[0x63],
        (BinOp::Gt, Ty::F64) => &[0x64],
        (BinOp::Le, Ty::F64) => &[0x65],
        (BinOp::Ge, Ty::F64) => &[0x66],
        (BinOp::Eq, Ty::F64) => &[0x61],
        (BinOp::Ne, Ty::F64) => &[0x62],
        (op, ty) => {
            return Err(EmitError::Unsupported(format!(
                "'{}' over {} is not a wasm operation",
                op.symbol(),
                ty.name()
            )));
        }
    };
    Ok(code)
}

/// Width-specific opcodes for the checked integer division sequence.
struct IntOps {
    eqz: u8,
    eq: u8,
    div: u8,
    rem: u8,
}

fn int_ops(ty: Ty) -> IntOps {
    match ty {
        Ty::I32 => IntOps {
            eqz: 0x45,
            eq: 0x46,
            div: 0x6D,
            rem: 0x6F,
        },
        _ => IntOps {
            eqz: 0x50,
            eq: 0x51,
            div: 0x7F,
            rem: 0x81,
        },
    }
}

/// Per-function lowering state.
struct Lowerer {
    /// Wasm local index of every scalar value id.
    scalar: Vec<Option<u32>>,
    /// `(ptr, len)` wasm local indices of every array value id.
    arrays: Vec<Option<(u32, u32)>>,
    /// Wasm function index of every IR function, by source name.
    functions: HashMap<String, u32>,
    /// The dispatch register, the first declared local.
    pc: u32,
    /// Number of value ids covered by the tables above.
    value_count: usize,
    /// Number of blocks, needed to compute dispatch-relative branch depths.
    block_count: usize,
    /// `(ptr local, len local, element bytes)` of the arrays folded at the
    /// function's exit — only `main` has any, because only `main`'s final
    /// environment is part of the state checksum.
    fold_at_exit: Vec<(u32, u32, i32)>,
    body: Body,
}

impl Lowerer {
    fn new(ir: &FuncIr, functions: HashMap<String, u32>) -> Result<Self, EmitError> {
        let value_count = ir.max_value_id() as usize + 1;
        let mut lowerer = Self {
            scalar: vec![None; value_count],
            arrays: vec![None; value_count],
            functions,
            pc: 0,
            value_count,
            block_count: ir.blocks.len(),
            fold_at_exit: Vec::new(),
            body: Body::new(0),
        };
        lowerer.allocate_locals(ir)?;
        Ok(lowerer)
    }

    /// Records the arrays the state checksum has to fold when this function
    /// exits. Only `main` is asked: the interpreter folds its root scope, and
    /// every other function's arrays are gone by then.
    fn set_exit_fold(&mut self, ir: &FuncIr) {
        self.fold_at_exit = folded_arrays(ir)
            .into_iter()
            .filter_map(|local| self.folded_array_locals(ir, local).ok())
            .collect();
    }

    // -- local allocation -----------------------------------------------------

    /// Entry-block phis with no arguments are the parameters; array bindings
    /// additionally own a `(ptr, len)` pair. Every other live value gets one
    /// declared local, so an id is never aliased with a different width.
    fn allocate_locals(&mut self, ir: &FuncIr) -> Result<(), EmitError> {
        let params: Vec<ValueId> = ir
            .block(ir.entry)
            .phis
            .iter()
            .filter(|phi| phi.args.is_empty())
            .map(|phi| phi.dst)
            .collect();
        let mut param_count = 0;
        for param in &params {
            // A parameter is named twice: by its phi destination (the SSA id)
            // and by its cell id, which is what `Load`/`Store` and array
            // arguments spell. Both must reach the same wasm local(s).
            let cell = self.param_cell(ir, *param);
            match self.value_ty(ir, *param)? {
                Ty::Array(_) => {
                    self.arrays[param.0 as usize] = Some((param_count, param_count + 1));
                    if let Some(cell) = cell {
                        self.arrays[cell.0 as usize] = Some((param_count, param_count + 1));
                    }
                    param_count += 2;
                }
                _ => {
                    self.scalar[param.0 as usize] = Some(param_count);
                    if let Some(cell) = cell {
                        self.scalar[cell.0 as usize] = Some(param_count);
                    }
                    param_count += 1;
                }
            }
        }
        self.body = Body::new(param_count);
        self.pc = self.body.declare_local(I32);

        let mut needed = vec![false; self.value_count];
        for block in &ir.blocks {
            for phi in &block.phis {
                self.mark(phi.dst, &mut needed);
                for (_, arg) in &phi.args {
                    self.mark(*arg, &mut needed);
                }
            }
            for inst in &block.insts {
                if let Some(dst) = inst.dst() {
                    self.mark(dst, &mut needed);
                }
                for used in inst.uses() {
                    self.mark(used, &mut needed);
                }
                // Arrays are addressed through their local slot, which lives
                // in the same id space (LocalId(i) == ValueId(i)).
                match inst {
                    Inst::Load(load) => self.mark(ValueId(load.arr.0), &mut needed),
                    Inst::Store { arr, .. } => self.mark(ValueId(arr.0), &mut needed),
                    Inst::Call(call) => {
                        for arr in &call.arr_refs {
                            self.mark(ValueId(arr.0), &mut needed);
                        }
                    }
                    _ => {}
                }
            }
            match &block.term {
                Term::Jump(_, args) => {
                    for arg in args {
                        self.mark(*arg, &mut needed);
                    }
                }
                Term::Branch { cond, .. } => self.mark(*cond, &mut needed),
                Term::Return(Some(v)) => self.mark(*v, &mut needed),
                Term::Return(None) => {}
            }
        }

        for (id, needed) in needed.iter().enumerate().take(self.value_count) {
            if !needed || self.scalar[id].is_some() || self.arrays[id].is_some() {
                continue;
            }
            match self.value_ty(ir, ValueId(id as u32))? {
                Ty::Array(_) => {
                    let ptr = self.body.declare_local(I32);
                    let len = self.body.declare_local(I32);
                    self.arrays[id] = Some((ptr, len));
                }
                Ty::I32 | Ty::Bool => self.scalar[id] = Some(self.body.declare_local(I32)),
                Ty::I64 => self.scalar[id] = Some(self.body.declare_local(I64)),
                Ty::F32 => self.scalar[id] = Some(self.body.declare_local(F32)),
                Ty::F64 => self.scalar[id] = Some(self.body.declare_local(F64)),
                Ty::Unit => {}
            }
        }
        Ok(())
    }

    fn mark(&self, v: ValueId, needed: &mut [bool]) {
        if let Some(slot) = needed.get_mut(v.0 as usize) {
            *slot = true;
        }
    }

    /// The cell id that also names a parameter, or `None` when the parameter's
    /// phi is its only spelling.
    fn param_cell(&self, ir: &FuncIr, dst: ValueId) -> Option<ValueId> {
        ir.block(ir.entry)
            .phis
            .iter()
            .find(|phi| phi.dst == dst)
            .map(|phi| ValueId(phi.var.0))
            .filter(|cell| *cell != dst)
    }

    // -- lookups --------------------------------------------------------------

    fn value_ty(&self, ir: &FuncIr, v: ValueId) -> Result<Ty, EmitError> {
        ir.types.val_tys.get(v.0 as usize).copied().ok_or_else(|| {
            EmitError::Malformed(format!("{}: value v{} has no recorded type", ir.name, v.0))
        })
    }

    fn elem_ty(&self, ir: &FuncIr, arr: LocalId) -> Result<ElemTy, EmitError> {
        ir.types.elem_ty(arr).ok_or_else(|| {
            EmitError::Malformed(format!(
                "{}: local #{} is addressed as an array but has no element type",
                ir.name, arr.0
            ))
        })
    }

    fn local_of(&self, v: ValueId) -> Result<u32, EmitError> {
        self.scalar
            .get(v.0 as usize)
            .copied()
            .flatten()
            .ok_or_else(|| EmitError::Malformed(format!("value v{} has no local", v.0)))
    }

    fn array_of(&self, v: ValueId) -> Result<(u32, u32), EmitError> {
        self.arrays
            .get(v.0 as usize)
            .copied()
            .flatten()
            .ok_or_else(|| EmitError::Malformed(format!("array v{} has no pointer pair", v.0)))
    }

    /// The `(pointer, length, element bytes)` triple behind an array binding,
    /// so `hx_checksum` folds exactly the memory the lowered body wrote
    /// through.
    fn folded_array_locals(
        &self,
        ir: &FuncIr,
        local: LocalId,
    ) -> Result<(u32, u32, i32), EmitError> {
        let (ptr, len) = self.array_of(ValueId(local.0))?;
        Ok((ptr, len, elem_size(self.elem_ty(ir, local)?)))
    }

    // -- raw emission ---------------------------------------------------------
    //
    // Thin wrappers over the shared [`Body`] writer; they exist so the lowering
    // reads as a sequence of operations rather than opcode arithmetic.

    fn raw(&mut self, bytes: &[u8]) {
        self.body.raw(bytes);
    }

    fn local_get(&mut self, index: u32) {
        self.body.local_get(index);
    }

    fn local_set(&mut self, index: u32) {
        self.body.local_set(index);
    }

    fn i32_const(&mut self, value: i32) {
        self.body.i32_const(value);
    }

    fn i64_const(&mut self, value: i64) {
        self.body.i64_const(value);
    }

    fn f32_const(&mut self, value: f32) {
        self.body.f32_const(value);
    }

    fn f64_const(&mut self, value: f64) {
        self.body.f64_const(value);
    }

    fn call(&mut self, index: u32) {
        self.body.call(index);
    }

    fn if_block(&mut self) {
        self.body.if_block();
    }

    fn end(&mut self) {
        self.body.end();
    }

    fn branch_to(&mut self, depth: u32) {
        self.body.branch(depth);
    }

    // -- value access ---------------------------------------------------------

    /// Pushes a scalar value.
    fn push_value(&mut self, ir: &FuncIr, v: ValueId) -> Result<(), EmitError> {
        match self.value_ty(ir, v)? {
            Ty::Array(_) | Ty::Unit => Err(EmitError::Malformed(format!(
                "{}: value v{} is not a scalar",
                ir.name, v.0
            ))),
            _ => {
                let index = self.local_of(v)?;
                self.local_get(index);
                Ok(())
            }
        }
    }

    /// Pushes a value widened to `i64`. Array indices and range bounds are the
    /// two places the spec lets an `i32` flow into an `i64` position.
    fn push_value_i64(&mut self, ir: &FuncIr, v: ValueId) -> Result<(), EmitError> {
        match self.value_ty(ir, v)? {
            Ty::I64 => self.push_value(ir, v),
            Ty::I32 => {
                let index = self.local_of(v)?;
                self.local_get(index);
                self.raw(&[0xAC]); // i64.extend_i32_s
                Ok(())
            }
            other => Err(EmitError::Malformed(format!(
                "{}: value v{} is {} where an integer was expected",
                ir.name,
                v.0,
                other.name()
            ))),
        }
    }

    /// Consumes the top of the stack into `dst`, or discards it when the call
    /// site dropped the result.
    ///
    /// A `Unit` destination means the callee produced nothing at all — HEXLIX
    /// procedures and `print` — so there is nothing to store or drop. That
    /// matters because statement-position calls carry no destination, and
    /// dropping a value that was never pushed is a validation error.
    fn store_or_drop(&mut self, ir: &FuncIr, dst: Option<ValueId>) -> Result<(), EmitError> {
        let Some(dst) = dst else {
            return Ok(());
        };
        match self.value_ty(ir, dst)? {
            Ty::Unit => Ok(()),
            _ => {
                let index = self.local_of(dst)?;
                self.local_set(index);
                Ok(())
            }
        }
    }

    /// Calls the host trap hook as `hx_trap(code, a, b)` with the operands
    /// widened to `i64`, matching the interpreter's message arguments.
    fn trap(
        &mut self,
        ir: &FuncIr,
        code: i32,
        a: Option<ValueId>,
        b: Option<ValueId>,
    ) -> Result<(), EmitError> {
        self.i32_const(code);
        for operand in [a, b] {
            match operand {
                Some(v) => self.push_value_i64(ir, v)?,
                None => self.i64_const(0),
            }
        }
        self.call(IMPORT_TRAP);
        Ok(())
    }

    /// Shared bounds guard for loads and stores: `0 <= idx < len`, unsigned
    /// compared so a negative index trips the same test.
    fn bounds_guard(&mut self, ir: &FuncIr, arr: LocalId, idx: ValueId) -> Result<(), EmitError> {
        let (_, len) = self.array_of(ValueId(arr.0))?;
        self.push_value_i64(ir, idx)?;
        self.local_get(len);
        self.raw(&[0xAD]); // i64.extend_i32_u
        self.raw(&[0x5A]); // i64.ge_u
        self.if_block();
        self.i32_const(TRAP_BOUNDS);
        self.local_get(len);
        self.raw(&[0xAD]); // i64.extend_i32_u
        self.push_value_i64(ir, idx)?;
        self.call(IMPORT_TRAP);
        self.end();
        Ok(())
    }

    /// Pushes the element address of `arr[idx]`.
    fn element_address(
        &mut self,
        ir: &FuncIr,
        arr: LocalId,
        idx: ValueId,
    ) -> Result<(), EmitError> {
        let (ptr, _) = self.array_of(ValueId(arr.0))?;
        self.local_get(ptr);
        self.i32_const(ELEMS_OFFSET);
        self.raw(&[0x6A]); // i32.add
        self.push_value_i64(ir, idx)?;
        self.raw(&[0xA7]); // i32.wrap_i64
        self.i32_const(elem_size(self.elem_ty(ir, arr)?));
        self.raw(&[0x6C]); // i32.mul
        self.raw(&[0x6A]); // i32.add
        Ok(())
    }

    // -- body -----------------------------------------------------------------

    fn emit_body(&mut self, ir: &FuncIr) -> Result<(), EmitError> {
        let count = ir.blocks.len();
        // A single returning block needs no dispatch machinery at all.
        if count == 1 && matches!(ir.blocks[0].term, Term::Return(_)) {
            self.emit_block(ir, 0)?;
            return Ok(());
        }

        let live = helix_ir::reachability(ir);
        let entry = ir.entry.0 as usize;
        self.i32_const(entry as i32);
        self.local_set(self.pc);
        self.raw(&[0x03, 0x40]); // loop $dispatch

        // Labels in reverse order: the last block is outermost, the entry is
        // innermost, so at the dispatch point block k sits at depth k.
        for _ in 0..count {
            self.raw(&[0x02, 0x40]); // block
        }
        self.local_get(self.pc);
        self.body.code().push(0x0E); // br_table
        push_u32(self.body.code(), count as u32);
        for depth in 0..count {
            push_u32(self.body.code(), depth as u32);
        }
        push_u32(self.body.code(), (count - 1) as u32); // default: last label
        self.raw(&[0x00]); // unreachable
        for index in 0..count {
            self.end(); // closes block index's label
            if live.get(index).copied().unwrap_or(false) {
                self.emit_block(ir, index)?;
            } else {
                self.raw(&[0x00]); // unreachable
            }
        }
        self.end(); // closes the dispatch loop
        // Control never falls out of the loop: every block either returns or
        // re-enters the dispatch. The `unreachable` is not decorative — wasm's
        // unreachable flag is per control frame, so without it the validator
        // would treat the code after the loop (and therefore the function's
        // own `end`) as reachable and demand the result value here.
        self.raw(&[0x00]); // unreachable
        Ok(())
    }

    fn emit_block(&mut self, ir: &FuncIr, index: usize) -> Result<(), EmitError> {
        for inst in &ir.blocks[index].insts {
            self.emit_inst(ir, inst)?;
        }
        self.emit_terminator(ir, index)
    }

    fn emit_terminator(&mut self, ir: &FuncIr, index: usize) -> Result<(), EmitError> {
        let term = ir.blocks[index].term.clone();
        match term {
            Term::Return(None) => {
                if ir.types.ret != Ty::Unit {
                    return Err(EmitError::Malformed(format!(
                        "{}: 'return;' inside a function returning {}",
                        ir.name,
                        ir.types.ret.name()
                    )));
                }
                self.emit_exit_fold();
                self.raw(&[0x0F]); // return
                Ok(())
            }
            Term::Return(Some(v)) => {
                if ir.types.ret == Ty::Unit {
                    return Err(EmitError::Malformed(format!(
                        "{}: a value is returned from a unit function",
                        ir.name
                    )));
                }
                self.push_value(ir, v)?;
                self.emit_exit_fold();
                self.raw(&[0x0F]); // return
                Ok(())
            }
            Term::Jump(target, args) => {
                self.move_phis(ir, target, None, &args)?;
                self.emit_edge(index, target, true, 0);
                Ok(())
            }
            Term::Branch { cond, t, f } => {
                let from = BlockId(index as u32);
                self.push_value(ir, cond)?;
                self.if_block();
                self.move_phis(ir, t, Some(from), &[])?;
                // An `if` arm cannot fall through, and both arms sit inside
                // the `if`'s own label, so the dispatch loop is one deeper.
                self.emit_edge(index, t, false, 1);
                self.raw(&[0x05]); // else
                self.move_phis(ir, f, Some(from), &[])?;
                self.emit_edge(index, f, true, 1);
                self.end();
                Ok(())
            }
        }
    }

    // -- instructions ---------------------------------------------------------

    fn emit_inst(&mut self, ir: &FuncIr, inst: &Inst) -> Result<(), EmitError> {
        match inst {
            Inst::Const { dst, c } => self.emit_const(ir, *dst, *c)?,
            Inst::Bin { op, dst, a, b } => self.emit_bin(ir, *op, *dst, *a, *b)?,
            Inst::Unary { op, dst, a } => self.emit_unary(ir, *op, *dst, *a)?,
            Inst::Cast { dst, val, to } => self.emit_cast(ir, *dst, *val, *to)?,
            Inst::Load(load) => {
                let (opcode, align, offset) = load_op(self.elem_ty(ir, load.arr)?);
                self.bounds_guard(ir, load.arr, load.idx)?;
                self.element_address(ir, load.arr, load.idx)?;
                self.raw(&[opcode, align, offset]);
                self.store_or_drop(ir, Some(load.dst))?;
            }
            Inst::Store { arr, idx, val } => {
                let (opcode, align, offset) = store_op(self.elem_ty(ir, *arr)?);
                self.bounds_guard(ir, *arr, *idx)?;
                self.element_address(ir, *arr, *idx)?;
                self.push_value(ir, *val)?;
                self.raw(&[opcode, align, offset]);
            }
            Inst::Call(call) => self.emit_call(ir, call)?,
        }
        Ok(())
    }

    fn emit_const(&mut self, ir: &FuncIr, dst: ValueId, c: Constant) -> Result<(), EmitError> {
        match c {
            Constant::I32(v) => self.i32_const(v),
            Constant::I64(v) => self.i64_const(v),
            Constant::F32(v) => self.f32_const(v),
            Constant::F64(v) => self.f64_const(v),
            Constant::Bool(b) => self.i32_const(i32::from(b)),
        }
        self.store_or_drop(ir, Some(dst))
    }

    fn emit_bin(
        &mut self,
        ir: &FuncIr,
        op: BinOp,
        dst: ValueId,
        a: ValueId,
        b: ValueId,
    ) -> Result<(), EmitError> {
        let ty = self.value_ty(ir, a)?;
        match op {
            BinOp::And | BinOp::Or => {
                return Err(EmitError::Unsupported(format!(
                    "'{}' must be lowered to control flow before emission",
                    op.symbol()
                )));
            }
            BinOp::Div | BinOp::Rem if ty.is_integral() => {
                self.emit_checked_div(ir, op, dst, a, b, ty)?;
            }
            _ => {
                self.push_value(ir, a)?;
                self.push_value(ir, b)?;
                self.raw(bin_opcode(op, ty)?);
                self.store_or_drop(ir, Some(dst))?;
            }
        }
        Ok(())
    }

    /// Integer division and remainder: the two traps the spec defines (zero
    /// divisor, and the `MIN / -1` overflow edge) are reported through the
    /// host hook so the message matches the interpreter's, and only then is
    /// the (oracle-free) wasm operation executed.
    fn emit_checked_div(
        &mut self,
        ir: &FuncIr,
        op: BinOp,
        dst: ValueId,
        a: ValueId,
        b: ValueId,
        ty: Ty,
    ) -> Result<(), EmitError> {
        let ops = int_ops(ty);

        self.push_value(ir, b)?;
        self.raw(&[ops.eqz]);
        self.if_block();
        self.trap(ir, TRAP_DIV_ZERO, Some(a), Some(b))?;
        self.end();

        self.push_value(ir, a)?;
        match ty {
            Ty::I32 => self.i32_const(i32::MIN),
            _ => self.i64_const(i64::MIN),
        }
        self.raw(&[ops.eq]);
        self.push_value(ir, b)?;
        match ty {
            Ty::I32 => self.i32_const(-1),
            _ => self.i64_const(-1),
        }
        self.raw(&[ops.eq]);
        self.raw(&[0x71]); // i32.and — comparison results are always i32
        self.if_block();
        self.trap(ir, TRAP_IDIV_OVERFLOW, Some(a), Some(b))?;
        self.end();

        self.push_value(ir, a)?;
        self.push_value(ir, b)?;
        self.raw(&[if op == BinOp::Div { ops.div } else { ops.rem }]);
        self.store_or_drop(ir, Some(dst))
    }

    fn emit_unary(
        &mut self,
        ir: &FuncIr,
        op: UnOp,
        dst: ValueId,
        a: ValueId,
    ) -> Result<(), EmitError> {
        match op {
            UnOp::Neg => {
                match self.value_ty(ir, a)? {
                    Ty::I32 => {
                        self.i32_const(0);
                        self.push_value(ir, a)?;
                        self.raw(&[0x6B]); // i32.sub — wrapping, like the spec
                    }
                    Ty::I64 => {
                        self.i64_const(0);
                        self.push_value(ir, a)?;
                        self.raw(&[0x7D]); // i64.sub
                    }
                    Ty::F32 => {
                        self.push_value(ir, a)?;
                        self.raw(&[0x8C]); // f32.neg
                    }
                    Ty::F64 => {
                        self.push_value(ir, a)?;
                        self.raw(&[0x9A]); // f64.neg
                    }
                    other => {
                        return Err(EmitError::Unsupported(format!(
                            "negation of {}",
                            other.name()
                        )));
                    }
                }
            }
            UnOp::Not => {
                self.push_value(ir, a)?;
                self.raw(&[0x45]); // i32.eqz
            }
        }
        self.store_or_drop(ir, Some(dst))
    }

    /// Numeric conversion with the frozen `as` semantics: float→int saturates
    /// (NaN→0) which is exactly wasm's `trunc_sat`, int→int truncates, and
    /// float↔float is promote/demote.
    fn emit_cast(
        &mut self,
        ir: &FuncIr,
        dst: ValueId,
        val: ValueId,
        to: Ty,
    ) -> Result<(), EmitError> {
        let from = self.value_ty(ir, val)?;
        if from == to {
            self.push_value(ir, val)?;
            return self.store_or_drop(ir, Some(dst));
        }
        self.push_value(ir, val)?;
        let code: &[u8] = match (from, to) {
            (Ty::I32, Ty::I64) => &[0xAC],       // i64.extend_i32_s
            (Ty::I64, Ty::I32) => &[0xA7],       // i32.wrap_i64
            (Ty::I32, Ty::F32) => &[0xB2],       // f32.convert_i32_s
            (Ty::I32, Ty::F64) => &[0xB7],       // f64.convert_i32_s
            (Ty::I64, Ty::F32) => &[0xB4],       // f32.convert_i64_s
            (Ty::I64, Ty::F64) => &[0xB9],       // f64.convert_i64_s
            (Ty::F32, Ty::I32) => &[0xFC, 0x00], // i32.trunc_sat_f32_s
            (Ty::F32, Ty::I64) => &[0xFC, 0x04], // i64.trunc_sat_f32_s
            (Ty::F64, Ty::I32) => &[0xFC, 0x02], // i32.trunc_sat_f64_s
            (Ty::F64, Ty::I64) => &[0xFC, 0x06], // i64.trunc_sat_f64_s
            (Ty::F32, Ty::F64) => &[0xBB],       // f64.promote_f32
            (Ty::F64, Ty::F32) => &[0xB6],       // f32.demote_f64
            (from, to) => {
                return Err(EmitError::Unsupported(format!(
                    "cast from {} to {}",
                    from.name(),
                    to.name()
                )));
            }
        };
        self.raw(code);
        self.store_or_drop(ir, Some(dst))
    }

    fn emit_call(&mut self, ir: &FuncIr, call: &helix_ir::Call) -> Result<(), EmitError> {
        let name = call.callee.as_str();
        match name {
            "print" => self.emit_print(ir, call)?,
            "zeros" => self.emit_zeros(ir, call)?,
            "len" => {
                let (Some(arr), Some(dst)) = (call.args.first(), call.dst) else {
                    return Err(EmitError::Malformed(format!(
                        "{}: malformed len() call",
                        ir.name
                    )));
                };
                if call.args.len() != 1 {
                    return Err(EmitError::Malformed(format!(
                        "{}: len() takes one argument",
                        ir.name
                    )));
                }
                let (_, len) = self.array_of(*arr)?;
                self.local_get(len);
                self.raw(&[0xAD]); // i64.extend_i32_u
                self.store_or_drop(ir, Some(dst))?;
            }
            "abs" | "sqrt" | "min" | "max" => self.emit_math_builtin(ir, call)?,
            other => {
                let Some(index) = self.functions.get(other).copied() else {
                    return Err(EmitError::Unsupported(format!(
                        "{}: call to unknown function '{other}'",
                        ir.name
                    )));
                };
                for arg in &call.args {
                    // Arrays travel as their pointer pair; scalars inline.
                    if matches!(self.value_ty(ir, *arg)?, Ty::Array(_)) {
                        let (ptr, len) = self.array_of(*arg)?;
                        self.local_get(ptr);
                        self.local_get(len);
                    } else {
                        self.push_value(ir, *arg)?;
                    }
                }
                self.call(index);
                self.store_or_drop(ir, call.dst)?;
            }
        }
        Ok(())
    }

    fn emit_print(&mut self, ir: &FuncIr, call: &helix_ir::Call) -> Result<(), EmitError> {
        if call.args.len() != 1 {
            return Err(EmitError::Malformed(format!(
                "{}: print() takes exactly one argument",
                ir.name
            )));
        }
        let value = call.args[0];
        self.push_value(ir, value)?;
        let import = match self.value_ty(ir, value)? {
            Ty::I32 => IMPORT_PRINT_I32,
            Ty::I64 => IMPORT_PRINT_I64,
            Ty::F32 => IMPORT_PRINT_F32,
            Ty::F64 => IMPORT_PRINT_F64,
            Ty::Bool => IMPORT_PRINT_BOOL,
            other => {
                return Err(EmitError::Unsupported(format!("print of {}", other.name())));
            }
        };
        self.call(import);
        Ok(())
    }

    /// `zeros(n)`: allocates the header plus `n` elements and binds the array's
    /// pointer/length pair. The negative-length trap fires inside the
    /// allocator before any memory is touched, so the host reports it with the
    /// interpreter's message.
    fn emit_zeros(&mut self, ir: &FuncIr, call: &helix_ir::Call) -> Result<(), EmitError> {
        if call.args.len() != 1 || call.arr_refs.len() != 1 {
            return Err(EmitError::Malformed(format!(
                "{}: zeros() allocates one array of one length",
                ir.name
            )));
        }
        let arr = call.arr_refs[0];
        let length = call.args[0];
        let elem = self.elem_ty(ir, arr)?;
        let (ptr, len_local) = self.array_of(ValueId(arr.0))?;

        self.push_value_i64(ir, length)?;
        self.i32_const(elem_size(elem));
        self.call(FN_ALLOC);
        self.local_set(ptr);

        self.push_value(ir, length)?;
        self.raw(&[0xA7]); // i32.wrap_i64 — the allocator proved it fits
        self.local_set(len_local);
        Ok(())
    }

    /// Pushes an integer constant at the width of `ty`.
    fn push_int_const(&mut self, ty: Ty, value: i64) {
        match ty {
            Ty::I32 => self.i32_const(value as i32),
            _ => self.i64_const(value),
        }
    }

    /// `abs`, `sqrt`, `min`, `max` — the numeric builtins that return a scalar.
    /// Integer min/max are selected comparisons because wasm has no integer
    /// min/max opcodes; float min/max are IEEE minNum/maxNum, which is exactly
    /// what the interpreter implements (a NaN operand loses).
    fn emit_math_builtin(&mut self, ir: &FuncIr, call: &helix_ir::Call) -> Result<(), EmitError> {
        let name = call.callee.as_str();
        if call.args.is_empty() || call.args.len() > 2 {
            return Err(EmitError::Malformed(format!(
                "{}: {name}() takes one or two arguments",
                ir.name
            )));
        }
        let first = call.args[0];
        let ty = self.value_ty(ir, first)?;

        match name {
            "abs" if call.args.len() == 1 => match ty {
                Ty::F32 => {
                    self.push_value(ir, first)?;
                    self.raw(&[0x8B]); // f32.abs
                }
                Ty::F64 => {
                    self.push_value(ir, first)?;
                    self.raw(&[0x99]); // f64.abs
                }
                Ty::I32 | Ty::I64 => {
                    // select(0 - x, x, x < 0): wrapping subtraction gives the
                    // two's-complement negation the spec asks for.
                    let (sub, lt) = if ty == Ty::I64 {
                        (0x7D, 0x53)
                    } else {
                        (0x6B, 0x48)
                    };
                    self.push_int_const(ty, 0);
                    self.push_value(ir, first)?;
                    self.raw(&[sub]);
                    self.push_value(ir, first)?;
                    self.push_value(ir, first)?;
                    self.push_int_const(ty, 0);
                    self.raw(&[lt]);
                    self.raw(&[0x1B]); // select
                }
                other => return Err(EmitError::Unsupported(format!("abs of {}", other.name()))),
            },
            "sqrt" if call.args.len() == 1 => match ty {
                Ty::F32 => {
                    self.push_value(ir, first)?;
                    self.raw(&[0x91]); // f32.sqrt
                }
                Ty::F64 => {
                    self.push_value(ir, first)?;
                    self.raw(&[0x9F]); // f64.sqrt
                }
                other => return Err(EmitError::Unsupported(format!("sqrt of {}", other.name()))),
            },
            "min" | "max" if call.args.len() == 2 => {
                let second = call.args[1];
                let min = name == "min";
                match ty {
                    Ty::I32 | Ty::I64 => {
                        let cmp = match (ty == Ty::I64, min) {
                            (true, true) => 0x53,   // i64.lt_s
                            (true, false) => 0x55,  // i64.gt_s
                            (false, true) => 0x48,  // i32.lt_s
                            (false, false) => 0x4A, // i32.gt_s
                        };
                        self.push_value(ir, first)?;
                        self.push_value(ir, second)?;
                        self.push_value(ir, first)?;
                        self.push_value(ir, second)?;
                        self.raw(&[cmp]);
                        self.raw(&[0x1B]); // select
                    }
                    Ty::F32 | Ty::F64 => {
                        let (low, high) = if ty == Ty::F64 {
                            (0xA4, 0xA5)
                        } else {
                            (0x96, 0x97)
                        };
                        self.push_value(ir, first)?;
                        self.push_value(ir, second)?;
                        self.raw(&[if min { low } else { high }]);
                    }
                    other => {
                        return Err(EmitError::Unsupported(format!(
                            "{name} of {}",
                            other.name()
                        )));
                    }
                }
            }
            _ => {
                return Err(EmitError::Malformed(format!(
                    "{}: {name}() has the wrong argument count",
                    ir.name
                )));
            }
        }
        self.store_or_drop(ir, call.dst)
    }

    /// Records this function's final arrays for [`abi::checksum`] to fold.
    ///
    /// The call sites sit at the exit block, which is where the interpreter
    /// takes its post-run snapshot: after every print, with the environment
    /// still intact. Each array is appended to a table in linear memory as
    /// `(element base, byte length)`, and `hx_checksum` folds the table — so
    /// the host has already folded the printed lines by then, and the byte
    /// stream matches the interpreter's exactly (lines first, arrays second).
    fn emit_exit_fold(&mut self) {
        for (slot, &(ptr, len, elem_bytes)) in self.fold_at_exit.iter().enumerate() {
            let entry = FOLD_TABLE_BASE + slot as i32 * FOLD_TABLE_STRIDE;
            self.body.i32_const(entry);
            self.body.local_get(ptr);
            self.body.i32_const(ELEMS_OFFSET);
            self.body.raw(&[0x6A]); // i32.add
            self.body.raw(&[0x36, 0x02, 0x00]); // i32.store
            self.body.i32_const(entry + 4);
            self.body.local_get(len);
            self.body.i32_const(elem_bytes);
            self.body.raw(&[0x6C]); // i32.mul
            self.body.raw(&[0x36, 0x02, 0x00]); // i32.store
        }
        // One extra `global.set` per recorded array is worse than a single
        // count write, so the table length is published once.
        let count = self.fold_at_exit.len();
        if count > 0 {
            self.body
                .i32_const(i32::try_from(count).unwrap_or(FOLD_TABLE_CAPACITY as i32));
            self.body.global_set(GLOBAL_FOLDS);
        }
    }

    /// Writes the successor's phi destinations for one edge. `Jump` passes the
    /// values positionally; `Branch` looks the predecessor up in each phi.
    fn move_phis(
        &mut self,
        ir: &FuncIr,
        target: BlockId,
        pred: Option<BlockId>,
        args: &[ValueId],
    ) -> Result<(), EmitError> {
        let phis = &ir.block(target).phis;
        if pred.is_none() && phis.len() != args.len() {
            return Err(EmitError::Malformed(format!(
                "{}: bb{} forwards {} values for {} phis",
                ir.name,
                target.0,
                args.len(),
                phis.len()
            )));
        }
        for (slot, phi) in phis.iter().enumerate() {
            let arg = match pred {
                None => args[slot],
                Some(from) => phi
                    .args
                    .iter()
                    .find(|(block, _)| *block == from)
                    .map(|(_, value)| *value)
                    .ok_or_else(|| {
                        EmitError::Malformed(format!(
                            "{}: bb{} has no phi argument for the edge from bb{}",
                            ir.name, target.0, from.0
                        ))
                    })?,
            };
            let dst = self.local_of(phi.dst)?;
            let src = self.local_of(arg)?;
            self.local_get(src);
            self.local_set(dst);
        }
        Ok(())
    }

    /// Transfers control to `target` — by falling through when it is the next
    /// block in emission order, otherwise through the dispatch loop.
    ///
    /// `nested` counts enclosing labels that are not block labels: a branch
    /// arm is inside its `if`, which pushes the dispatch loop one label
    /// deeper than the block's own code is.
    fn emit_edge(&mut self, from: usize, target: BlockId, may_fall_through: bool, nested: u32) {
        if may_fall_through && target == BlockId(from as u32 + 1) {
            return;
        }
        self.i32_const(target.0 as i32);
        self.local_set(self.pc);
        // Inside block `from`'s code the dispatch loop is `block_count - from -
        // 1` labels away: every block label opened after `from` is still open.
        let depth = self.block_count - from - 1 + nested as usize;
        self.branch_to(depth as u32);
    }
}

/// Lowers one function into its serialized code entry. `main`'s body
/// additionally folds the final arrays the state checksum needs, because the
/// interpreter takes that snapshot at the end of `main` rather than in a
/// separate routine.
pub(super) fn lower(
    ir: &FuncIr,
    functions: HashMap<String, u32>,
    fold_exit: bool,
) -> Result<Vec<u8>, EmitError> {
    if ir.blocks.len() > MAX_DISPATCH_BLOCKS {
        return Err(EmitError::TooManyBlocks {
            function: ir.name.clone(),
            blocks: ir.blocks.len(),
        });
    }
    let mut lowerer = Lowerer::new(ir, functions)?;
    if fold_exit {
        lowerer.set_exit_fold(ir);
    }
    lowerer.emit_body(ir)?;
    Ok(lowerer.body.finish())
}
