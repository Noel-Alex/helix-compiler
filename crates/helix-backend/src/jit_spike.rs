//! M0 de-risk spike: prove cranelift-jit 0.135 works on this machine (Windows x64, MSVC ABI).
//!
//! Validates, against the real pinned crates:
//! - the canonical JIT flow (settings → native ISA → JITBuilder::with_isa → define → finalize),
//! - CallConv::WindowsFastcall == Rust `extern "C"` on x86_64-pc-windows-msvc,
//! - host symbol registration (`JITBuilder::symbol`) and calling INTO Rust from JIT code,
//! - i64 arithmetic and f64 loads through a pointer parameter (array-access shape),
//! - a two-block CFG with brif/jump carrying block arguments.

use std::mem;

use cranelift::codegen::ir::{types, AbiParam, Signature};
use cranelift::frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift::prelude::*;
use cranelift::codegen::ir::BlockArg;
use cranelift::codegen::isa::CallConv;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, FuncId, Linkage, Module};

fn new_jit() -> JITModule {
    let flag_builder = settings::builder();
    let isa_builder = cranelift_native::builder().expect("host machine features");
    let isa = isa_builder
        .finish(settings::Flags::new(flag_builder))
        .expect("isa setup");
    let jb = JITBuilder::with_isa(isa, default_libcall_names());
    JITModule::new(jb)
}

/// Build one function into `module`, finalize everything, return its code pointer.
///
/// `emit` receives the builder plus the module (shared reborrow) so it can resolve
/// imports via `declare_func_in_func`. Pointers are valid only after
/// `finalize_definitions`, which this performs.
fn build_and_finalize(
    module: &mut JITModule,
    name: &str,
    sig: Signature,
    emit: impl FnOnce(&mut FunctionBuilder, &mut JITModule),
) -> *const u8 {
    let mut ctx = module.make_context();
    ctx.func.signature = sig;
    let fid = module
        .declare_function(name, Linkage::Local, &ctx.func.signature)
        .expect("declare function");

    {
        let mut bctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut bctx);
        let entry = builder.create_block();
        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);
        builder.seal_block(entry);
        emit(&mut builder, module);
        builder.finalize(module.target_config());
    }

    module.define_function(fid, &mut ctx).expect("define function");
    module.clear_context(&mut ctx);
    module.finalize_definitions().expect("finalize definitions");
    module.get_finalized_function(fid)
}

#[test]
fn jit_add_two_i64() {
    let mut module = new_jit();
    let mut sig = Signature::new(CallConv::WindowsFastcall);
    sig.params.push(AbiParam::new(types::I64));
    sig.params.push(AbiParam::new(types::I64));
    sig.returns.push(AbiParam::new(types::I64));

    let ptr = build_and_finalize(&mut module, "add", sig, |b, _| {
        let params = b.block_params(b.current_block().unwrap());
        let (x, y) = (params[0], params[1]);
        let sum = b.ins().iadd(x, y);
        b.ins().return_(&[sum]);
    });

    let f = unsafe { mem::transmute::<*const u8, extern "C" fn(i64, i64) -> i64>(ptr) };
    assert_eq!(f(2, 3), 5);
    assert_eq!(f(-7, 7), 0);
    assert_eq!(f(i64::MAX - 1, 1), i64::MAX);
}

#[test]
fn jit_loads_f64_through_pointer() {
    let mut module = new_jit();
    let mut sig = Signature::new(CallConv::WindowsFastcall);
    sig.params.push(AbiParam::new(types::I64)); // pointer carried as i64
    sig.returns.push(AbiParam::new(types::F64));

    let ptr = build_and_finalize(&mut module, "sum3", sig, |b, _| {
        let p = b.block_params(b.current_block().unwrap())[0];
        let flags = MemFlagsData::trusted().with_notrap().with_aligned();
        // p[0], p[1], p[2] via offset immediates — array element access shape.
        let x0 = b.ins().load(types::F64, flags, p, 0);
        let x1 = b.ins().load(types::F64, flags, p, 8);
        let x2 = b.ins().load(types::F64, flags, p, 16);
        let s01 = b.ins().fadd(x0, x1);
        let s = b.ins().fadd(s01, x2);
        b.ins().return_(&[s]);
    });

    let mut data = [1.5f64, 2.25, -4.0];
    let f = unsafe { mem::transmute::<*const u8, extern "C" fn(*mut f64) -> f64>(ptr) };
    let s = f(data.as_mut_ptr());
    assert!((s - (-0.25)).abs() < 1e-12);
}

#[test]
fn jit_calls_host_symbol() {
    static CALLS: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);
    extern "C" fn record(v: i64) {
        CALLS.fetch_add(v, std::sync::atomic::Ordering::SeqCst);
    }

    // Host symbols must be registered on the JITBuilder before the module exists.
    let flag_builder = settings::builder();
    let isa_builder = cranelift_native::builder().unwrap();
    let isa = isa_builder.finish(settings::Flags::new(flag_builder)).unwrap();
    let mut jb = JITBuilder::with_isa(isa, default_libcall_names());
    jb.symbol("record", record as extern "C" fn(i64) as *const u8);
    let mut module = JITModule::new(jb);

    let mut host_sig = Signature::new(CallConv::WindowsFastcall);
    host_sig.params.push(AbiParam::new(types::I64));
    let host_fid = module
        .declare_function("record", Linkage::Import, &host_sig)
        .expect("declare import");

    let mut sig = Signature::new(CallConv::WindowsFastcall);
    sig.params.push(AbiParam::new(types::I64));

    let ptr = build_and_finalize(&mut module, "call_host", sig, |b, m| {
        let arg = b.block_params(b.current_block().unwrap())[0];
        let two = b.ins().iconst(types::I64, 2);
        let v = b.ins().imul(arg, two);
        let func_ref = m.declare_func_in_func(host_fid, &mut b.func);
        b.ins().call(func_ref, &[arg]);
        b.ins().call(func_ref, &[v]);
        b.ins().return_(&[]);
    });

    let f = unsafe { mem::transmute::<*const u8, extern "C" fn(i64)>(ptr) };
    f(10); // record(10) + record(20)
    assert_eq!(CALLS.load(std::sync::atomic::Ordering::SeqCst), 30);
}

#[test]
fn jit_branches_with_block_args() {
    // max(a,b) via a real diamond: bb0 -> brif -> bb1/bb2 -> jump(bb3, chosen).
    let mut module = new_jit();
    let mut sig = Signature::new(CallConv::WindowsFastcall);
    sig.params.push(AbiParam::new(types::I64));
    sig.params.push(AbiParam::new(types::I64));
    sig.returns.push(AbiParam::new(types::I64));

    let ptr = build_and_finalize(&mut module, "max2", sig, |b, _| {
        let params = b.block_params(b.current_block().unwrap());
        let (a, c) = (params[0], params[1]);
        let then_blk = b.create_block();
        let else_blk = b.create_block();
        let join = b.create_block();
        b.append_block_param(join, types::I64);

        let cmp = b.ins().icmp(IntCC::SignedGreaterThan, a, c);
        b.ins().brif(cmp, then_blk, &[], else_blk, &[]);

        b.switch_to_block(then_blk);
        b.ins().jump(join, &[BlockArg::Value(a)]);
        b.switch_to_block(else_blk);
        b.ins().jump(join, &[BlockArg::Value(c)]);
        b.switch_to_block(join);
        b.seal_block(then_blk);
        b.seal_block(else_blk);
        b.seal_block(join);
        let result = b.block_params(join)[0];
        b.ins().return_(&[result]);
    });

    let f = unsafe { mem::transmute::<*const u8, extern "C" fn(i64, i64) -> i64>(ptr) };
    assert_eq!(f(3, 9), 9);
    assert_eq!(f(42, 7), 42);
}
