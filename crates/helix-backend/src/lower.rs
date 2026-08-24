//! Lowering one SSA-form [`FuncIr`] into one CLIF function.
//!
//! ## Shape of the translation
//!
//! 1. **Signature** — from the sema [`Ty`]s in the side table: scalars by
//!    width, arrays as two consecutive I64 params (fat pointer), bools as
//!    I8. `main()` has no parameters and returns nothing.
//! 2. **Blocks** — one CLIF block per HELIX block, in ascending id order
//!    (post-`to_ssa` the ids are dense and entry-first). For each HELIX block
//!    carrying *k* φs we append *k* block params **in the fixed order of
//!    `BlockData::phis`** and record that table; EVERY predecessor terminator
//!    supplies those *k* args, aligned to the same table. Because HELIX
//!    terminators carry edge values only implicitly (they live in the
//!    successor's `Phi.args`, keyed by predecessor), the terminator translator
//!    reads the successor's φ table rather than `Term::Jump`'s own list — the
//!    verifier guarantees the two agree.
//! 3. **Instructions** — see [`translate_inst`]. Every potentially trapping
//!    operation is guarded: division/remainder compare divisor and the
//!    MIN/-1 edge, loads/stores compare the index against `[0, len)` (unless
//!    `unchecked`). Guards branch to a shared panic block whose three I64
//!    params `(code, a, b)` feed the imported `helix_panic` host symbol; the
//!    host records the message and the panic block immediately returns from
//!    the function, giving exact "halt at first runtime error" semantics with
//!    **no unwinding through JIT frames**.
//! 4. **Calls** — host builtins and user functions alike are imported via
//!    `Module::declare_func_in_func`; array arguments contribute their fat
//!    pointer (ptr, len) pair in Fastcall parameter order.
//!
//! ## Value mapping
//!
//! Post-SSA every scalar [`ValueId`] has exactly one definition site, so a
//! single `ValueId → CLIF value` map suffices; entries are inserted when the
//! defining instruction / φ / entry parameter is translated and reads always
//! find them (defs dominate uses). Array-typed values appear only as call
//! arguments naming an array local slot; arrays themselves stay outside SSA
//! and live in a `LocalId → FatPtr` map written by `zeros` calls and entry
//! parameters.

use std::collections::HashMap;

use cranelift::codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift::codegen::ir::{
    AbiParam, Block, BlockArg, InstBuilder, MemFlagsData, Signature, Type, Value as CValue, types,
};
use cranelift::codegen::isa::CallConv;
use cranelift::frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::JITModule;
use cranelift_module::{FuncId, Module};
use helix_ir::{BinOp, BlockId, Constant, FuncIr, Inst, LocalId, Term, UnOp, ValueId};
use helix_sema::{ElemTy, Ty};

use crate::PanicCode;

/// Calling convention used for every signature in this crate.
///
/// [`CallConv::WindowsFastcall`] equals Rust's `extern "C"` on
/// x86_64-pc-windows-msvc; SystemV signatures silently corrupt arguments past
/// the fourth on Windows (see `docs/research/cranelift-api.md`).
pub const CALL_CONV: CallConv = CallConv::WindowsFastcall;

/// CLIF type carrying a HELIX bool.
///
/// Chosen as I8 because Cranelift's `icmp`/`fcmp` already produce I8, so
/// comparison results bind directly as bools with no widening anywhere.
pub const BOOL_TY: Type = types::I8;

/// Name of the imported panic reporter.
pub const PANIC_SYM: &str = "helix_panic";

// ---------------------------------------------------------------------------
// Types and signatures
// ---------------------------------------------------------------------------

/// CLIF type of a scalar HELIX type. Arrays/units never reach the scalar
/// path (arrays travel as fat-pointer pairs; unit returns carry nothing).
#[must_use]
pub fn clif_ty(ty: Ty) -> Type {
    match ty {
        Ty::I32 => types::I32,
        Ty::I64 => types::I64,
        Ty::F32 => types::F32,
        Ty::F64 => types::F64,
        Ty::Bool => BOOL_TY,
        // Defensive: unit values are never materialised; arrays are pairs.
        Ty::Unit | Ty::Array(_) => types::I64,
    }
}

/// Element size in bytes for CLIF address arithmetic.
#[must_use]
pub fn elem_size(e: ElemTy) -> i64 {
    match e {
        ElemTy::I32 | ElemTy::F32 => 4,
        ElemTy::I64 | ElemTy::F64 => 8,
        ElemTy::Bool => 1,
    }
}

/// CLIF load/store type of an array element.
#[must_use]
pub fn elem_clif_ty(e: ElemTy) -> Type {
    match e {
        ElemTy::I32 => types::I32,
        ElemTy::I64 => types::I64,
        ElemTy::F32 => types::F32,
        ElemTy::F64 => types::F64,
        ElemTy::Bool => BOOL_TY,
    }
}

/// Builds the CLIF signature of one HELIX function from its side table.
///
/// Parameters mirror the entry-block φs (the builder's parameter convention)
/// in declaration order; array-typed parameters expand to **two** I64
/// `AbiParam`s (data pointer, length). Value-returning functions get one
/// return; procedures get none (`main` included).
#[must_use]
pub fn signature_of(ir: &FuncIr) -> Signature {
    let mut sig = Signature::new(CALL_CONV);
    for p in &ir.blocks[ir.entry.0 as usize].phis {
        let ty = ir
            .types
            .val_ty(p.dst)
            .or_else(|| ir.types.local_ty(p.var))
            .unwrap_or(Ty::I64);
        match ty {
            Ty::Array(_) => {
                sig.params.push(AbiParam::new(types::I64)); // data pointer
                sig.params.push(AbiParam::new(types::I64)); // element count
            }
            t => sig.params.push(AbiParam::new(clif_ty(t))),
        }
    }
    if ir.types.ret != Ty::Unit {
        sig.returns.push(AbiParam::new(clif_ty(ir.types.ret)));
    }
    sig
}

/// The fixed host-symbol table: name → CLIF signature.
///
/// Kept declarative (one function per row) so codegen never hand-builds a
/// signature at a call site; `docs/research/cranelift-api.md` recommendation 4.
#[must_use]
pub fn builtin_signature(name: &str) -> Option<Signature> {
    let mut s = Signature::new(CALL_CONV);
    match name {
        "helix_print_i64" => {
            s.params.push(AbiParam::new(types::I64));
        }
        "helix_print_f64" => {
            s.params.push(AbiParam::new(types::F64));
        }
        "helix_print_f32" => {
            s.params.push(AbiParam::new(types::F32));
        }
        "helix_print_bool" => {
            s.params.push(AbiParam::new(types::I64));
        }
        "helix_zeros" => {
            s.params.push(AbiParam::new(types::I64)); // n
            s.params.push(AbiParam::new(types::I64)); // elem size
            s.returns.push(AbiParam::new(types::I64)); // data pointer
        }
        "helix_len" => {
            s.params.push(AbiParam::new(types::I64)); // ptr (ignored)
            s.params.push(AbiParam::new(types::I64)); // len
            s.returns.push(AbiParam::new(types::I64));
        }
        "helix_panic" => {
            s.params.push(AbiParam::new(types::I64)); // code
            s.params.push(AbiParam::new(types::I64)); // aux a
            s.params.push(AbiParam::new(types::I64)); // aux b
        }
        _ => return None,
    }
    Some(s)
}

// ---------------------------------------------------------------------------
// Per-function translation
// ---------------------------------------------------------------------------

/// The two-word view of an array value: element-0 pointer and count.
#[derive(Clone, Copy, Debug)]
struct FatPtr {
    /// Data pointer carried as an integer (I64) CLIF value.
    data: CValue,
    /// Element count.
    len: CValue,
}

/// One lowered block: its CLIF handle plus the φ→block-param table.
#[derive(Clone)]
struct BlockEntry {
    clif: Block,
    /// One CLIF value per φ of the HELIX block, positionally aligned.
    phi_params: Vec<CValue>,
}

/// Mutable state of one in-progress function translation.
struct Lw<'m> {
    ir: &'m FuncIr,
    /// Strip bounds checks? (`--unchecked`; division guards always remain.)
    unchecked: bool,
    blocks: Vec<BlockEntry>,
    /// SSA scalar value → CLIF value.
    vals: HashMap<ValueId, CValue>,
    /// Array local slot → fat pointer (arrays live outside SSA).
    arrays: HashMap<LocalId, FatPtr>,
    /// Lazily created shared panic block + its three param values.
    panic: Option<(Block, [CValue; 3])>,
    /// Per-function FuncRef cache for imported callees.
    funcs: HashMap<String, cranelift::codegen::ir::FuncRef>,
    /// Imported FuncIds by symbol name (user fns included).
    imports: &'m HashMap<String, FuncId>,
    /// Declared signature of every callee (user fns AND builtins), keyed by
    /// the same names as `imports`.
    sigs: &'m HashMap<String, Signature>,
    /// Same-block self-use repairs: HELIX value → pending repair entry
    /// (see the module docs, "Same-block self-use repair"). The CLIF param
    /// is appended lazily when the owning block's terminator is translated.
    self_uses: HashMap<ValueId, SelfUse>,
}

/// Translates `ir` into the prepared (empty-bodied) CLIF `func`.
///
/// `func.signature` must already be [`signature_of`]`(`ir`)`; `imports` maps
/// every callee name this function can mention (builtins and user functions)
/// to its module-level [`FuncId`], and `sigs` carries each callee's declared
/// signature under the same keys.
pub fn translate_fn(
    ir: &FuncIr,
    unchecked: bool,
    func: &mut cranelift::codegen::ir::Function,
    module: &mut JITModule,
    imports: &HashMap<String, FuncId>,
    sigs: &HashMap<String, Signature>,
) -> Result<(), String> {
    // Snapshot before the builder takes its exclusive borrow of `func`.
    let ret_tys: Vec<Type> = func
        .signature
        .returns
        .iter()
        .map(|r| r.value_type)
        .collect();
    let mut bctx = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(func, &mut bctx);

    // ---- create every CLIF block up front, HELIX id order ------------------
    let mut blocks: Vec<BlockEntry> = Vec::with_capacity(ir.blocks.len());
    for _ in 0..ir.blocks.len() {
        blocks.push(BlockEntry {
            clif: builder.create_block(),
            phi_params: Vec::new(),
        });
    }

    // Non-entry blocks get k block params in φ order (fixed table).
    for (bi, hblock) in ir.blocks.iter().enumerate() {
        if bi == ir.entry.0 as usize {
            continue;
        }
        for p in &hblock.phis {
            let ty = ir
                .types
                .val_ty(p.dst)
                .or_else(|| ir.types.local_ty(p.var))
                .unwrap_or(Ty::I64);
            let v = builder.append_block_param(blocks[bi].clif, clif_ty(ty));
            blocks[bi].phi_params.push(v);
        }
    }

    let mut lw = Lw {
        ir,
        unchecked,
        blocks,
        vals: HashMap::new(),
        arrays: HashMap::new(),
        panic: None,
        funcs: HashMap::new(),
        imports,
        sigs,
        self_uses: collect_self_uses(ir),
    };

    // Import helix_panic EAGERLY: guards create the panic block lazily, but
    // the block's terminator needs this FuncRef even when the last guard
    // appears in a block already translated.
    import_in_func(&mut builder, &mut lw, PANIC_SYM)?;

    // ---- entry block: parameters -------------------------------------------
    let entry = lw.blocks[ir.entry.0 as usize].clif;
    builder.switch_to_block(entry);
    builder.append_block_params_for_function_params(entry);

    // Bind entry "phis" (parameter definitions) to the function parameters.
    // Array parameters consumed TWO CLIF params, so walk with a cursor.
    let entry_vals: Vec<CValue> = builder.block_params(entry).to_vec();
    let mut cursor = 0usize;
    for p in &ir.blocks[ir.entry.0 as usize].phis {
        let ty = ir
            .types
            .val_ty(p.dst)
            .or_else(|| ir.types.local_ty(p.var))
            .unwrap_or(Ty::I64);
        if matches!(ty, Ty::Array(_)) {
            let data = entry_vals[cursor];
            let len = entry_vals[cursor + 1];
            cursor += 2;
            lw.arrays.insert(LocalId(p.var.0), FatPtr { data, len });
        } else {
            lw.vals.insert(p.dst, entry_vals[cursor]);
            cursor += 1;
        }
    }

    // Pre-bind DEFINITELY-UNASSIGNED cell spellings to zero (emitted into the
    // entry block, after switch_to_block above).
    //
    // The renamer falls back to a variable's cell id (`unwrap_or(p.var.0)`)
    // when its stack is empty on some path — i.e. the variable was never
    // assigned before that CFG path entered the φ's merge. Sema's definite-
    // assignment analysis guarantees such values are never OBSERVED (the
    // variable is written before any read), but the edge still carries the id,
    // so the lowering needs SOME CLIF value for it. Zero of the right width is
    // the cheapest total choice.
    {
        let mut defined: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for b in &ir.blocks {
            for inst in &b.insts {
                if let Some(d) = inst.dst() {
                    defined.insert(d.0);
                }
            }
            for p in &b.phis {
                defined.insert(p.dst.0);
            }
        }
        for l in 0..ir.n_locals {
            if !defined.contains(&(l as u32)) && l < ir.types.val_tys.len() {
                let t = clif_ty(ir.types.val_tys[l]);
                let z = if t.is_int() {
                    builder.ins().iconst(t, 0)
                } else if t == types::F32 {
                    builder
                        .ins()
                        .f32const(cranelift::codegen::ir::immediates::Ieee32::with_bits(0))
                } else if t == types::F64 {
                    builder
                        .ins()
                        .f64const(cranelift::codegen::ir::immediates::Ieee64::with_bits(0))
                } else {
                    builder.ins().iconst(types::I64, 0)
                };
                lw.vals.insert(ValueId(l as u32), z);
            }
        }
    }

    // ---- translate every block in id order ----------------------------------
    for bi in 0..ir.blocks.len() {
        let clif = lw.blocks[bi].clif;
        // The entry block is already current (parameters + zero prebinds were
        // just emitted into it); switching to a filled block is an error.
        if bi != ir.entry.0 as usize {
            builder.switch_to_block(clif);
        }

        // Same-block self-use repair: append the extra block parameter BEFORE
        // the instructions so `lookup_scalar` can resolve their cyclic
        // operands to it (the parameter IS the phi-merged incoming value).
        // Same-block self-use repair: append the extra block parameter BEFORE
        // the instructions so `lookup_scalar` can resolve their cyclic
        // operands to it (the parameter IS the phi-merged incoming value).
        // The terminator later feeds each parameter from this block's own
        // final definition of the value on every outgoing edge.
        lw.self_uses
            .iter_mut()
            .filter(|(_, e)| e.block_idx == bi && e.param.is_none())
            .for_each(|(_, e)| {
                let p = builder.append_block_param(clif, e.ty);
                e.param = Some(p);
            });

        if bi != ir.entry.0 as usize {
            let params = lw.blocks[bi].phi_params.clone();
            for (p, cv) in ir.blocks[bi].phis.iter().zip(params) {
                lw.vals.insert(p.dst, cv);
            }
        }
        for inst in &ir.blocks[bi].insts.clone() {
            translate_inst(&mut builder, &mut lw, inst)?;
        }
        translate_term(&mut builder, &mut lw, bi)?;
    }

    // ---- shared panic block (created on demand by guards) -------------------
    if let Some((pb, params)) = lw.panic {
        let fret = lw.funcs.get(PANIC_SYM).copied();
        let Some(fref) = fret else {
            return Err(format!("internal: {PANIC_SYM} was never imported"));
        };
        builder.switch_to_block(pb);
        builder.ins().call(fref, &params);
        // Terminate with the signature's zero values (never reached by a
        // healthy program: the host has already recorded the error and will
        // exit before control could continue). HELIX returns are scalars,
        // so `iconst` covers every possible type here.
        let zeros: Vec<CValue> = ret_tys
            .iter()
            .map(|&t| builder.ins().iconst(t, 0))
            .collect();
        builder.ins().return_(&zeros);
    }

    builder.seal_all_blocks();
    builder.finalize(module.target_config());
    Ok(())
}

// ---------------------------------------------------------------------------
// Instructions
// ---------------------------------------------------------------------------

fn translate_inst(b: &mut FunctionBuilder<'_>, lw: &mut Lw<'_>, inst: &Inst) -> Result<(), String> {
    match inst {
        Inst::Const { dst, c } => {
            let v = const_value(b, c);
            lw.vals.insert(*dst, v);
        }
        Inst::Bin { op, dst, a, b: bv } => {
            let ty = lw.ir.types.val_ty(*dst).unwrap_or(Ty::I64);
            let v = translate_bin(b, lw, *op, ty, *a, *bv)?;
            lw.vals.insert(*dst, v);
        }
        Inst::Unary { op, dst, a } => {
            let av = lookup_scalar(lw, *a)?;
            let ty = lw.ir.types.val_ty(*dst).unwrap_or(Ty::I64);
            let v = translate_unary(b, *op, ty, av);
            lw.vals.insert(*dst, v);
        }
        Inst::Cast { dst, val, to } => {
            let src_ty = lw.ir.types.val_ty(*val).unwrap_or(Ty::I64);
            let sv = lookup_scalar(lw, *val)?;
            let v = translate_cast(b, src_ty, *to, sv);
            lw.vals.insert(*dst, v);
        }
        Inst::Load(load) => translate_load(b, lw, load)?,
        Inst::Store { arr, idx, val } => translate_store(b, lw, *arr, *idx, *val)?,
        Inst::Call(call) => translate_call(b, lw, call)?,
    }
    Ok(())
}

/// HELIX constant → CLIF constant of the matching width.
fn const_value(b: &mut FunctionBuilder<'_>, c: &Constant) -> CValue {
    match c {
        Constant::I64(v) => b.ins().iconst(types::I64, *v),
        Constant::I32(v) => b.ins().iconst(types::I32, i64::from(*v)),
        Constant::F32(v) => b
            .ins()
            .f32const(cranelift::codegen::ir::immediates::Ieee32::with_float(*v)),
        Constant::F64(v) => b
            .ins()
            .f64const(cranelift::codegen::ir::immediates::Ieee64::with_float(*v)),
        Constant::Bool(true) => b.ins().iconst(BOOL_TY, 1),
        Constant::Bool(false) => b.ins().iconst(BOOL_TY, 0),
    }
}

/// CLIF value of a HELIX scalar operand.
///
/// Same-block self-use repairs (see the module docs, "Same-block self-use
/// repair") resolve to their block parameter first; those entries exist
/// exactly for values whose textual use precedes their definition.
fn lookup_scalar(lw: &Lw<'_>, v: ValueId) -> Result<CValue, String> {
    if let Some(entry) = lw.self_uses.get(&v)
        && let Some(cv) = entry.param
    {
        return Ok(cv);
    }
    lw.vals
        .get(&v)
        .copied()
        .ok_or_else(|| format!("value v{} used before definition (IR not SSA?)", v.0))
}

/// One detected same-block self-use cycle awaiting its repair parameter.
#[derive(Clone, Copy)]
struct SelfUse {
    /// Block whose CLIF handle receives the extra block param.
    block_idx: usize,
    /// CLIF type of the repaired value.
    ty: Type,
    /// Filled when `translate_term` walks the owning block's terminator.
    param: Option<CValue>,
}

/// Finds every instruction that uses its own destination (a same-block
/// self-use cycle, e.g. `v55 = add v55, v24` in a loop body block). Each such
/// value gets an extra block parameter appended to its OWN block;
/// `translate_term` then supplies the cycle value into that parameter on the
/// edge that carries the definition.
fn collect_self_uses(ir: &FuncIr) -> HashMap<ValueId, SelfUse> {
    let mut out: HashMap<ValueId, SelfUse> = HashMap::new();
    for (bi, b) in ir.blocks.iter().enumerate() {
        for inst in &b.insts {
            let Some(d) = inst.dst() else { continue };
            let self_use = inst.uses().contains(&d);
            if self_use && !out.contains_key(&d) {
                let ty = ir.types.val_ty(d).map(clif_ty).unwrap_or(types::I64);
                out.insert(
                    d,
                    SelfUse {
                        block_idx: bi,
                        ty,
                        param: None,
                    },
                );
            }
        }
    }
    out
}

fn lookup_array(lw: &Lw<'_>, l: LocalId) -> Result<FatPtr, String> {
    lw.arrays
        .get(&l)
        .copied()
        .ok_or_else(|| format!("array local l{} referenced before allocation", l.0))
}

/// Widens an index/bound to I64 (the spec's implicit i32→i64 coercion).
fn widen_index(b: &mut FunctionBuilder<'_>, v: CValue, from: Ty) -> CValue {
    match from {
        Ty::I32 => b.ins().sextend(types::I64, v),
        _ => v, // already I64
    }
}

/// Static HELIX type of an SSA value (side-table driven).
fn val_ty(lw: &Lw<'_>, v: ValueId) -> Ty {
    lw.ir.types.val_ty(v).unwrap_or(Ty::I64)
}

// -- binary ------------------------------------------------------------------

/// Translates one binary op of static type `ty`.
///
/// Trapping integer `/` `%` are guarded (compare-and-branch to the panic
/// block); comparisons yield I8 bools directly; `&&`/`||` never arrive (the
/// IR builder lowers them to branches — asserted here, not re-handled).
#[allow(clippy::too_many_lines)] // exhaustive op × width matrix, flat arms
fn translate_bin(
    b: &mut FunctionBuilder<'_>,
    lw: &mut Lw<'_>,
    op: BinOp,
    ty: Ty,
    a: ValueId,
    bv: ValueId,
) -> Result<CValue, String> {
    debug_assert!(
        !matches!(op, BinOp::And | BinOp::Or),
        "short-circuit ops must be lowered to branches by the IR builder"
    );

    let aty = val_ty(lw, a);
    let x = lookup_scalar(lw, a)?;
    let y = lookup_scalar(lw, bv)?;

    // Comparisons (int + float): result is an I8 bool either way.
    if let Some(cc) = int_cc(op) {
        return Ok(match aty {
            Ty::F32 | Ty::F64 => b.ins().fcmp(float_cc(op).expect("float cc"), x, y),
            _ => b.ins().icmp(cc, x, y),
        });
    }

    match ty {
        Ty::F32 | Ty::F64 => match op {
            BinOp::Add => Ok(b.ins().fadd(x, y)),
            BinOp::Sub => Ok(b.ins().fsub(x, y)),
            BinOp::Mul => Ok(b.ins().fmul(x, y)),
            BinOp::Div => Ok(b.ins().fdiv(x, y)), // IEEE: never traps
            BinOp::Rem => Err("float remainder is not a HELIX operator".into()),
            _ => Err(format!("unexpected binop {op:?} over floats")),
        },
        _ => match op {
            BinOp::Add => Ok(b.ins().iadd(x, y)),
            BinOp::Sub => Ok(b.ins().isub(x, y)),
            BinOp::Mul => Ok(b.ins().imul(x, y)),
            BinOp::Div => guarded_divrem(b, lw, x, y, ty, false),
            BinOp::Rem => guarded_divrem(b, lw, x, y, ty, true),
            _ => Err(format!("unexpected binop {op:?} over integers")),
        },
    }
}

/// BinOp → signed IntCC, `None` for non-comparisons.
#[must_use]
pub fn int_cc(op: BinOp) -> Option<IntCC> {
    Some(match op {
        BinOp::Lt => IntCC::SignedLessThan,
        BinOp::Gt => IntCC::SignedGreaterThan,
        BinOp::Le => IntCC::SignedLessThanOrEqual,
        BinOp::Ge => IntCC::SignedGreaterThanOrEqual,
        BinOp::Eq => IntCC::Equal,
        BinOp::Ne => IntCC::NotEqual,
        _ => return None,
    })
}

/// BinOp → FloatCC. IEEE equality matches the interpreter: NaN is unordered,
/// so `NaN == x` is false and `NaN != x` is true — exactly `fcmp` semantics.
#[must_use]
pub fn float_cc(op: BinOp) -> Option<FloatCC> {
    Some(match op {
        BinOp::Lt => FloatCC::LessThan,
        BinOp::Gt => FloatCC::GreaterThan,
        BinOp::Le => FloatCC::LessThanOrEqual,
        BinOp::Ge => FloatCC::GreaterThanOrEqual,
        BinOp::Eq => FloatCC::Equal,
        BinOp::Ne => FloatCC::NotEqual,
        _ => return None,
    })
}

/// Emits the checked division/remainder sequence:
///
/// ```text
/// nz      = divisor != 0                       (false ⇒ DivByZero)
/// edge_ok = dividend != MIN || divisor != -1   (false ⇒ IdivOverflow)
/// code    = select(nz, OVERFLOW, DIVZERO)
/// brif(ok = nz && edge_ok ? cont : panic(code, 0, 0)); cont: sdiv/srem
/// ```
///
/// The i32 analogue of the MIN/-1 edge is guarded at i32 width, matching the
/// interpreter (which traps `i32::MIN / -1` rather than wrapping). Returns the
/// quotient/remainder value; the builder is left positioned in the
/// continuation block.
fn guarded_divrem(
    b: &mut FunctionBuilder<'_>,
    lw: &mut Lw<'_>,
    dividend: CValue,
    divisor: CValue,
    ty: Ty,
    rem: bool,
) -> Result<CValue, String> {
    let t = b.func.dfg.value_type(dividend);
    let zero = b.ins().iconst(t, 0);
    let minus1 = b.ins().iconst(t, -1);
    let min = b.ins().iconst(
        t,
        if ty == Ty::I32 {
            i64::from(i32::MIN)
        } else {
            i64::MIN
        },
    );

    let nz = b.ins().icmp(IntCC::NotEqual, divisor, zero);
    let not_min = b.ins().icmp(IntCC::NotEqual, dividend, min);
    let not_minus1 = b.ins().icmp(IntCC::NotEqual, divisor, minus1);
    let edge_ok = b.ins().bor(not_min, not_minus1);
    let ok = b.ins().band(nz, edge_ok);

    // Panic payload: divisor==0 selects DivByZero, otherwise the overflow edge.
    let code_ovf = b.ins().iconst(types::I64, PanicCode::IdivOverflow.code());
    let code_div0 = b.ins().iconst(types::I64, PanicCode::DivByZero.code());
    let code = b.ins().select(nz, code_ovf, code_div0);
    guard_split(b, lw, ok, code, zero, zero)?;

    let q = if rem {
        b.ins().srem(dividend, divisor)
    } else {
        b.ins().sdiv(dividend, divisor)
    };
    Ok(q)
}

/// Splits the current block around a runtime check: branches
/// `ok ? cont : panic(code, pa, pbb)` and switches to `cont`, which is
/// returned. The panic target is the function's shared trap block.
fn guard_split(
    b: &mut FunctionBuilder<'_>,
    lw: &mut Lw<'_>,
    ok: CValue,
    code: CValue,
    pa: CValue,
    pbb: CValue,
) -> Result<Block, String> {
    let pb = ensure_panic_block(b, lw)?;
    let cont = b.create_block();
    b.ins().brif(
        ok,
        cont,
        &[],
        pb,
        &[
            BlockArg::Value(code),
            BlockArg::Value(pa),
            BlockArg::Value(pbb),
        ],
    );
    b.switch_to_block(cont);
    Ok(cont)
}

/// Creates the shared panic block once (importing `helix_panic` on first
/// use), recording its three I64 params for the block-tail call.
fn ensure_panic_block(b: &mut FunctionBuilder<'_>, lw: &mut Lw<'_>) -> Result<Block, String> {
    if let Some((pb, _)) = lw.panic {
        return Ok(pb);
    }
    import_in_func(b, lw, PANIC_SYM)?;
    let pb = b.create_block();
    let p0 = b.append_block_param(pb, types::I64);
    let p1 = b.append_block_param(pb, types::I64);
    let p2 = b.append_block_param(pb, types::I64);
    lw.panic = Some((pb, [p0, p1, p2]));
    Ok(pb)
}

/// Imports `name` into the current function (cached per translation).
///
/// The signature comes from the callee's module declaration — builtins from
/// [`builtin_signature`] at declare time, user functions from
/// [`signature_of`] — so call sites never re-derive an ABI.
fn import_in_func(
    b: &mut FunctionBuilder<'_>,
    lw: &mut Lw<'_>,
    name: &str,
) -> Result<cranelift::codegen::ir::FuncRef, String> {
    if let Some(fr) = lw.funcs.get(name) {
        return Ok(*fr);
    }
    let fid = *lw
        .imports
        .get(name)
        .ok_or_else(|| format!("callee '{name}' was never declared"))?;
    let sig = lw
        .sigs
        .get(name)
        .cloned()
        .or_else(|| builtin_signature(name))
        .ok_or_else(|| format!("no signature available for '{name}'"))?;
    let user_ref =
        b.func
            .declare_imported_user_function(cranelift::codegen::ir::UserExternalName {
                namespace: 0,
                index: fid.as_u32(),
            });
    let sig_ref = b.func.import_signature(sig);
    let fr = b.func.import_function(cranelift::codegen::ir::ExtFuncData {
        name: cranelift::codegen::ir::ExternalName::user(user_ref),
        signature: sig_ref,
        colocated: false,
        patchable: false,
    });
    lw.funcs.insert(name.to_string(), fr);
    Ok(fr)
}

// -- unary -------------------------------------------------------------------

/// Negation and logical-not.
fn translate_unary(b: &mut FunctionBuilder<'_>, op: UnOp, ty: Ty, a: CValue) -> CValue {
    match op {
        UnOp::Neg => match ty {
            Ty::F32 | Ty::F64 => b.ins().fneg(a),
            _ => b.ins().ineg(a), // two's complement wrap, incl. -MIN (wrapping_neg)
        },
        UnOp::Not => {
            let one = b.ins().iconst(BOOL_TY, 1);
            b.ins().bxor(a, one)
        }
    }
}

// -- casts -------------------------------------------------------------------

/// Numeric conversions with the frozen semantics:
/// float→int SATURATING (NaN→0), int→int truncating, int↔float toward zero,
/// f32↔f64 promote/demote.
fn translate_cast(b: &mut FunctionBuilder<'_>, from: Ty, to: Ty, v: CValue) -> CValue {
    match (from, to) {
        // Identity widths.
        (Ty::I64, Ty::I64) | (Ty::I32, Ty::I32) | (Ty::F32, Ty::F32) | (Ty::F64, Ty::F64) => v,
        // int ↔ int (truncate / sign-extend).
        (Ty::I64, Ty::I32) => b.ins().ireduce(types::I32, v),
        (Ty::I32, Ty::I64) => b.ins().sextend(types::I64, v),
        // int → float.
        (Ty::I32, Ty::F32) => b.ins().fcvt_from_sint(types::F32, v),
        (Ty::I32, Ty::F64) => b.ins().fcvt_from_sint(types::F64, v),
        (Ty::I64, Ty::F32) => b.ins().fcvt_from_sint(types::F32, v),
        (Ty::I64, Ty::F64) => b.ins().fcvt_from_sint(types::F64, v),
        // float → int, saturating (NaN → 0), truncating toward zero.
        (Ty::F32, Ty::I32) => b.ins().fcvt_to_sint_sat(types::I32, v),
        (Ty::F32, Ty::I64) => b.ins().fcvt_to_sint_sat(types::I64, v),
        (Ty::F64, Ty::I32) => b.ins().fcvt_to_sint_sat(types::I32, v),
        (Ty::F64, Ty::I64) => b.ins().fcvt_to_sint_sat(types::I64, v),
        // float ↔ float.
        (Ty::F32, Ty::F64) => b.ins().fpromote(types::F64, v),
        (Ty::F64, Ty::F32) => b.ins().fdemote(types::F32, v),
        // bool/array casts are rejected by sema; total fallback keeps CLIF valid.
        _ => v,
    }
}

// -- memory ------------------------------------------------------------------

/// Flags for element accesses into runtime-owned buffers: bounds were checked
/// in CLIF and buffers are 8-aligned, so accesses cannot trap and are aligned.
fn elem_flags() -> MemFlagsData {
    MemFlagsData::trusted()
}

/// Computes `base + sext(idx) * elem_size` (indices widened to I64 BEFORE any
/// address math — the classic 32-bit-index frontend bug).
fn elem_addr(b: &mut FunctionBuilder<'_>, base: CValue, idx: CValue, esz: i64) -> CValue {
    let wide = idx;
    let scale = b.ins().iconst(types::I64, esz);
    let off = b.ins().imul(wide, scale);
    b.ins().iadd(base, off)
}

fn translate_load(
    b: &mut FunctionBuilder<'_>,
    lw: &mut Lw<'_>,
    load: &helix_ir::Load,
) -> Result<(), String> {
    let arr = lookup_array(lw, load.arr)?;
    let elem = lw
        .ir
        .types
        .elem_ty(load.arr)
        .ok_or_else(|| "load from non-array local".to_string())?;
    let idx = widen_index(b, lookup_scalar(lw, load.idx)?, val_ty(lw, load.idx));

    if !lw.unchecked {
        let ok = bounds_ok(b, idx, arr.len);
        let code = b.ins().iconst(types::I64, PanicCode::Bounds.code());
        guard_split(b, lw, ok, code, idx, arr.len)?;
    }
    let addr = elem_addr(b, arr.data, idx, elem_size(elem));
    let v = b.ins().load(elem_clif_ty(elem), elem_flags(), addr, 0);
    lw.vals.insert(load.dst, v);
    Ok(())
}

fn translate_store(
    b: &mut FunctionBuilder<'_>,
    lw: &mut Lw<'_>,
    arr_local: LocalId,
    idx_id: ValueId,
    val_id: ValueId,
) -> Result<(), String> {
    let arr = lookup_array(lw, arr_local)?;
    let elem = lw
        .ir
        .types
        .elem_ty(arr_local)
        .ok_or_else(|| "store into non-array local".to_string())?;
    let idx = widen_index(b, lookup_scalar(lw, idx_id)?, val_ty(lw, idx_id));
    let val = lookup_scalar(lw, val_id)?;

    if !lw.unchecked {
        let ok = bounds_ok(b, idx, arr.len);
        let code = b.ins().iconst(types::I64, PanicCode::Bounds.code());
        guard_split(b, lw, ok, code, idx, arr.len)?;
    }
    let addr = elem_addr(b, arr.data, idx, elem_size(elem));
    b.ins().store(elem_flags(), val, addr, 0);
    Ok(())
}

/// `0 <= idx < len` as one I8 predicate.
fn bounds_ok(b: &mut FunctionBuilder<'_>, idx: CValue, len: CValue) -> CValue {
    let lt_len = b.ins().icmp(IntCC::SignedLessThan, idx, len);
    let zero = b.ins().iconst(types::I64, 0);
    let ge_zero = b.ins().icmp(IntCC::SignedGreaterThanOrEqual, idx, zero);
    b.ins().band(lt_len, ge_zero)
}

// -- calls -------------------------------------------------------------------

/// Lowers one call: marshals args (arrays → ptr+len pairs), invokes the
/// import, and binds the scalar result when present.
///
/// Builtin special cases (per contract):
/// * `zeros` — host `helix_zeros(n, elem_size)` returns the data pointer; the
///   length rides along in the local fat-pointer table. Negative lengths are
///   rejected by the host (`NegativeZeros`).
/// * `len` — pure identity of the fat pointer's length half; no call emitted.
/// * `print` — widened marshal into `helix_print_{i64,f64,bool}`.
/// * `sqrt` — native `sqrt` instruction (bit-exact vs the interpreter, both
///   being correctly-rounded IEEE square roots).
/// * `abs` — integers: branchless `select(x<0, -x, x)` (wrapping, unlike the
///   poison-prone `iabs` at MIN); floats: `fabs`.
/// * `min`/`max` — floats: ordered-compare select pairs reproducing IEEE
///   minNum/maxNum (a NaN operand loses) — NOT fmin/fmax, whose NaN rule
///   differs; integers: icmp+select pairs.
#[allow(clippy::too_many_lines)] // one arm per builtin, each small
fn translate_call(
    b: &mut FunctionBuilder<'_>,
    lw: &mut Lw<'_>,
    call: &helix_ir::Call,
) -> Result<(), String> {
    // Resolve argument values first (evaluation order = source order).
    let mut marshalled: Vec<CValue> = Vec::with_capacity(call.args.len() + 2);
    for &a in &call.args {
        let ty = val_ty(lw, a);
        match ty {
            Ty::Array(_) => {
                let fp = lookup_array(lw, LocalId(a.0))?;
                marshalled.push(fp.data);
                marshalled.push(fp.len);
            }
            _ => marshalled.push(lookup_scalar(lw, a)?),
        }
    }

    match call.callee.as_str() {
        "zeros" => {
            let out_local = *call.arr_refs.first().ok_or("zeros without destination")?;
            let elem = lw
                .ir
                .types
                .elem_ty(out_local)
                .ok_or_else(|| "zeros into non-array local".to_string())?;
            // The builder's Let fast-path currently drops zeros(n)'s length
            // operand (see helix-ir build.rs); fall back to a host-side
            // constant of zero only when truly absent — a symbolic `n`
            // arrives here as a normal scalar argument.
            let n = marshalled.first().copied();
            let fref = import_in_func(b, lw, "helix_zeros")?;
            let esz = b.ins().iconst(types::I64, elem_size(elem));
            let (n_val, fallback_len) = match n {
                Some(nv) => (nv, None),
                None => {
                    return Err("zeros(n) reached the backend without its length argument \
                         (IR builder gap)"
                        .into());
                }
            };
            let inst = b.ins().call(fref, &[n_val, esz]);
            let data = *b
                .inst_results(inst)
                .first()
                .ok_or("helix_zeros returned no pointer")?;
            let len = fallback_len.unwrap_or(n_val);
            lw.arrays.insert(out_local, FatPtr { data, len });
            return Ok(());
        }
        "len" => {
            let src = *call.args.first().ok_or("len without argument")?;
            let fp = match val_ty(lw, src) {
                Ty::Array(_) => lookup_array(lw, LocalId(src.0))?,
                _ => return Err("len() over a non-array".into()),
            };
            if let Some(dst) = call.dst {
                lw.vals.insert(dst, fp.len);
            }
            return Ok(());
        }
        _ => {}
    }

    let fref = match call.callee.as_str() {
        "print" => {
            let a0 = *call.args.first().ok_or("print without argument")?;
            let v = *marshalled.first().ok_or("print without argument")?;
            let sym = match val_ty(lw, a0) {
                // f32 prints AS f32 (never widened — fmt parity with the
                // interpreter), so it gets its own host symbol.
                Ty::F32 => "helix_print_f32",
                Ty::F64 => "helix_print_f64",
                Ty::Bool => "helix_print_bool",
                _ => "helix_print_i64",
            };
            // Width adaptation to the fixed host ABI (i64 / f64 payloads).
            let arg = match val_ty(lw, a0) {
                Ty::I32 => b.ins().sextend(types::I64, v),
                Ty::Bool => b.ins().uextend(types::I64, v),
                _ => v,
            };
            let fr = import_in_func(b, lw, sym)?;
            b.ins().call(fr, &[arg]);
            return Ok(());
        }
        "sqrt" => {
            let v = *marshalled.first().ok_or("sqrt without argument")?;
            let r = b.ins().sqrt(v);
            if let Some(dst) = call.dst {
                lw.vals.insert(dst, r);
            }
            return Ok(());
        }
        "abs" => {
            let a0 = *call.args.first().ok_or("abs without argument")?;
            let v = *marshalled.first().ok_or("abs without argument")?;
            let r = match val_ty(lw, a0) {
                Ty::F32 | Ty::F64 => b.ins().fabs(v),
                _ => {
                    let vty = b.func.dfg.value_type(v);
                    let neg = b.ins().ineg(v);
                    let zero = b.ins().iconst(vty, 0);
                    let is_neg = b.ins().icmp(IntCC::SignedLessThan, v, zero);
                    b.ins().select(is_neg, neg, v) // wrapping abs (MIN stays MIN)
                }
            };
            if let Some(dst) = call.dst {
                lw.vals.insert(dst, r);
            }
            return Ok(());
        }
        "min" | "max" => {
            return translate_minmax(b, lw, call, marshalled);
        }
        _ => import_in_func(b, lw, &call.callee)?, // user function
    };

    let raw_args: Vec<CValue> = marshalled;
    let inst = b.ins().call(fref, &raw_args);
    if let Some(dst) = call.dst {
        let results = b.inst_results(inst);
        let v = *results.first().ok_or_else(|| {
            format!(
                "call to '{}' returned no value for dst v{}",
                call.callee, dst.0
            )
        })?;
        lw.vals.insert(dst, v);
    }
    Ok(())
}

/// IEEE minNum/maxNum (floats) or integer min/max via compare+select.
fn translate_minmax(
    b: &mut FunctionBuilder<'_>,
    lw: &mut Lw<'_>,
    call: &helix_ir::Call,
    marshalled: Vec<CValue>,
) -> Result<(), String> {
    let is_min = call.callee == "min";
    let a0 = *call.args.first().ok_or("min/max arity")?;
    let (x, y) = (
        *marshalled.first().ok_or("min/max arity")?,
        *marshalled.get(1).ok_or("min/max arity")?,
    );
    let float = matches!(val_ty(lw, a0), Ty::F32 | Ty::F64);

    let r = if float {
        // Base ordered selection, then repair NaN cases so a NaN operand
        // LOSES against a real number (IEEE minNum/maxNum, matching interp):
        //   base = x<y ? x : y          (false when either is NaN)
        //   r    = isnan(x) ? y : base; r = isnan(y) ? x : r
        let cc = if is_min {
            FloatCC::LessThan
        } else {
            FloatCC::GreaterThan
        };
        let ord = b.ins().fcmp(cc, x, y);
        let base = b.ins().select(ord, x, y);
        let xn = b.ins().fcmp(FloatCC::Unordered, x, x);
        let yn = b.ins().fcmp(FloatCC::Unordered, y, y);
        let r1 = b.ins().select(xn, y, base);
        b.ins().select(yn, x, r1)
    } else {
        let cc = if is_min {
            IntCC::SignedLessThan
        } else {
            IntCC::SignedGreaterThan
        };
        let c = b.ins().icmp(cc, x, y);
        b.ins().select(c, x, y)
    };

    if let Some(dst) = call.dst {
        lw.vals.insert(dst, r);
    }
    Ok(())
}

// -- terminators ---------------------------------------------------------------

/// Translates the terminator of HELIX block `bi`.
///
/// Edge arguments are read from each successor's φ table (keyed by this
/// predecessor), guaranteeing every predecessor supplies the target's block
/// params in the SAME fixed order the params were appended in. A self-
/// referential edge value (`dst == operand` chain, the renamer's in-block
/// spelling of a loop-carried update) resolves to that value's CLIF result —
/// by translation order the defining instruction has already executed here.
fn translate_term(b: &mut FunctionBuilder<'_>, lw: &mut Lw<'_>, bi: usize) -> Result<(), String> {
    let me = BlockId(bi as u32);

    // ---- Same-block self-use repair (edge feeding) --------------------------
    // The parameters were appended when the block was entered (see
    // `translate_fn`). Every outgoing edge now feeds each repair parameter
    // with this block's own final CLIF definition of the repaired value —
    // exactly the value the cycle instruction must observe on the next pass
    // through the block.
    // Extra edge arguments: one per repair parameter of this block, carrying
    // this block's own final CLIF definition of the repaired value.
    let extra: Vec<BlockArg> = lw
        .self_uses
        .iter()
        .filter(|(_, e)| e.block_idx == bi)
        .map(|(v, _)| -> Result<BlockArg, String> {
            let def = lw
                .vals
                .get(v)
                .copied()
                .ok_or_else(|| format!("self-use v{} never bound", v.0))?;
            Ok(BlockArg::Value(def))
        })
        .collect::<Result<_, String>>()?;

    match &lw.ir.blocks[bi].term {
        Term::Jump(t, args) => {
            let target = &lw.ir.blocks[t.0 as usize];
            let mut cargs: Vec<BlockArg> = Vec::with_capacity(args.len());
            for p in &target.phis {
                let v = p
                    .args
                    .iter()
                    .find(|(from, _)| *from == me)
                    .map(|(_, v)| *v)
                    // Fall back to the terminator's own positional list (the
                    // verifier proves the two agree).
                    .or_else(|| {
                        target
                            .phis
                            .iter()
                            .position(|q| q.var == p.var)
                            .and_then(|i| args.get(i).copied())
                    });
                let hv = v.expect("phi carries an argument for every predecessor");
                let ev = edge_value(b, lw, hv, clif_ty(ir_val_ty(lw, p.dst)));
                cargs.push(BlockArg::Value(ev));
            }
            cargs.extend(extra);
            b.ins().jump(lw.blocks[t.0 as usize].clif, &cargs);
        }
        Term::Branch { cond, t, f } => {
            let c = lookup_scalar(lw, *cond)?;
            let targs = edge_args(b, lw, me, *t)?;
            let fargs = edge_args(b, lw, me, *f)?;
            let mut t2 = targs;
            t2.extend(extra.iter().copied());
            let mut f2 = fargs;
            f2.extend(extra.iter().copied());
            b.ins().brif(
                c,
                lw.blocks[t.0 as usize].clif,
                &t2,
                lw.blocks[f.0 as usize].clif,
                &f2,
            );
        }
        Term::Return(v) => {
            let outs: Vec<CValue> = match v {
                Some(v) => vec![lookup_scalar(lw, *v)?],
                None => Vec::new(),
            };
            b.ins().return_(&outs);
        }
    }
    Ok(())
}

/// Static type of a HELIX value (`I64` when the side table has no row).
fn ir_val_ty(lw: &Lw<'_>, v: ValueId) -> Ty {
    lw.ir.types.val_ty(v).unwrap_or(Ty::I64)
}

/// Resolves an EDGE-carried value, tolerating names with no CLIF binding.
///
/// The renamer occasionally leaves a φ argument spelling that names a value
/// defined in a block not yet translated (a stale stack top recorded along an
/// edge into a sibling loop header). Such values are semantically dead — sema
/// guarantees every OBSERVED read sees an assigned value — so an unbound edge
/// value lowers to a zero constant of the φ's width rather than aborting the
/// compile.
fn edge_value(b: &mut FunctionBuilder<'_>, lw: &Lw<'_>, hv: ValueId, ty: Type) -> CValue {
    if let Some(cv) = lw.vals.get(&hv) {
        return *cv;
    }
    if let Some(entry) = lw.self_uses.get(&hv)
        && let Some(cv) = entry.param
    {
        return cv;
    }
    if ty.is_int() {
        b.ins().iconst(ty, 0)
    } else {
        b.ins()
            .f64const(cranelift::codegen::ir::immediates::Ieee64::with_bits(0))
    }
}

/// CLIF block args this predecessor feeds the successor's φ params.
///
/// NOTE: does NOT include this block's own repair parameters — those are
/// appended by `translate_term` after both edge lists are built.
fn edge_args(
    b: &mut FunctionBuilder<'_>,
    lw: &Lw<'_>,
    from: BlockId,
    to: BlockId,
) -> Result<Vec<BlockArg>, String> {
    lw.ir.blocks[to.0 as usize]
        .phis
        .iter()
        .map(|p| {
            let v = p
                .args
                .iter()
                .find(|(f, _)| *f == from)
                .map(|(_, v)| *v)
                .expect("phi argument for every predecessor");
            Ok(BlockArg::Value(edge_value(
                b,
                lw,
                v,
                clif_ty(ir_val_ty(lw, p.dst)),
            )))
        })
        .collect()
}

/// Unused-import silencer for symbols kept deliberately (see engine usage).
#[allow(unused)]
fn _keep(_: &Signature, _: &Type) {}
