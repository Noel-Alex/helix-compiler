//! # helix-wasm-backend — the WebAssembly emitter for HELIX
//!
//! Consumes post-SSA, optimized IR (`helix_ir::build` → `to_ssa` →
//! `run_optimization_pipeline`) and produces one real WebAssembly module that
//! runs the program in a browser with no host runtime of its own.
//!
//! ```text
//! &[FuncIr] ──▶ module bytes ──▶ WebAssembly.instantiate({ env: … })
//! ```
//!
//! The module is self-describing: it exports its memory and four functions,
//! imports six host hooks for printing and trapping, and keeps all program
//! state (arrays plus the running checksum) inside linear memory and globals.
//! A host therefore only has to feed formatted print lines into the checksum
//! and read two numbers back.
//!
//! ## Module contract
//!
//! **Imports** (module `env`):
//!
//! | import | signature | purpose |
//! |---|---|---|
//! | `hx_print_i32` | `(i32) -> ()` | print an i32 |
//! | `hx_print_i64` | `(i64) -> ()` | print an i64 (integers widen to i64) |
//! | `hx_print_f32` | `(f32) -> ()` | print an f32 at f32 width |
//! | `hx_print_f64` | `(f64) -> ()` | print an f64 |
//! | `hx_print_bool` | `(i32) -> ()` | print a bool |
//! | `hx_trap` | `(i32 code, i64 a, i64 b) -> ()` | report a runtime trap and unwind |
//!
//! **Exports**:
//!
//! | export | signature | behaviour |
//! |---|---|---|
//! | `memory` | 32 pages minimum | element storage, based above the host print scratch |
//! | `hx_run` | `() -> ()` | reset all state, then run `fn main()` |
//! | `hx_fold` | `(i32 ptr, i32 len) -> ()` | fold `len` bytes of formatted print output into the checksum |
//! | `hx_checksum` | `() -> i64` | fold `main`'s final arrays, return the FNV-1a checksum |
//! | `hx_heap_bytes` | `() -> i32` | bytes this run allocated |
//!
//! **Trap codes** (see [`codegen`]): 0 division by zero, 1 division overflow
//! (`i64::MIN / -1`), 2 bounds violation (`a` = length, `b` = index), 3
//! negative `zeros(`a`)`, 4 call depth exceeded, 5 allocation failure.
//!
//! ## Checksum parity
//!
//! The checksum stream is the interpreter's, byte for byte: every printed line
//! plus its `\n` (folded by `hx_fold` from the host, which renders each line
//! with `helix_sema`'s canonical formatter), then every array still bound in
//! `main`'s root scope, in ascending symbol order, element by element, as
//! little-endian bytes — folded by `hx_checksum`. A browser run therefore
//! produces the same number the reference interpreter does.
//!
//! ## Why hand-written encoding
//!
//! The emitter writes the binary format directly rather than depending on a
//! wasm builder crate: the shipped asset is the whole surface the browser
//! sees, so the encoder stays auditable, and the crate keeps its dependency
//! list at `helix-ir` + `helix-sema`.

#![forbid(unsafe_code)]

use std::collections::HashMap;

use helix_ir::FuncIr;

mod abi;
mod codegen;
mod encode;

use abi::{FNV_OFFSET, HEAP_BASE};
use codegen::{FIRST_USER_FN, FN_FOLD, FOLD_TABLE_CAPACITY, MAX_DISPATCH_BLOCKS};
use encode::{
    F32, F64, I32, I64, SECTION_CODE, SECTION_EXPORT, SECTION_FUNCTION, SECTION_GLOBAL,
    SECTION_IMPORT, SECTION_MEMORY, SECTION_NAME, SECTION_TYPE, code_body, push_name, push_section,
    push_u32, vector_body,
};

/// The emitted module: the bytes *are* the artifact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmittedModule {
    /// A complete WebAssembly module, ready to instantiate.
    pub bytes: Vec<u8>,
}

/// Why emission refused to produce a module.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EmitError {
    /// No function named `main` exists.
    NoMain,
    /// Two functions share a name, so a call site could not be resolved.
    DuplicateName(String),
    /// The function is too large for the dispatch loop's label nesting.
    TooManyBlocks {
        /// Function that overflowed.
        function: String,
        /// Its block count.
        blocks: usize,
    },
    /// The IR contains a construct the emitter refuses to guess about.
    Unsupported(String),
    /// The IR violates a structural invariant the emitter relies on.
    Malformed(String),
}

impl std::fmt::Display for EmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoMain => write!(f, "no function named 'main' in the module"),
            Self::DuplicateName(name) => write!(f, "duplicate function name '{name}'"),
            Self::TooManyBlocks { function, blocks } => write!(
                f,
                "function '{function}' has {blocks} blocks, above the {MAX_DISPATCH_BLOCKS}-block limit"
            ),
            Self::Unsupported(detail) => write!(f, "unsupported IR: {detail}"),
            Self::Malformed(detail) => write!(f, "malformed IR: {detail}"),
        }
    }
}

impl std::error::Error for EmitError {}

/// Wasm type indices for the ABI shapes. A user function whose signature
/// matches one of these reuses the entry instead of duplicating it. Index 0
/// (`() -> ()`) is unnamed because it is reached through the folded signature
/// table and by `hx_run`'s call to `main`.
const T_UNIT_UNIT: u32 = 0;
const T_FOLD: u32 = 1; // (i32, i32) -> ()
const T_CHECKSUM: u32 = 2; // () -> i64
const T_HEAP: u32 = 3; // () -> i32
const T_PRINT_I32: u32 = 4; // (i32) -> ()
const T_PRINT_I64: u32 = 5; // (i64) -> ()
const T_PRINT_F32: u32 = 6; // (f32) -> ()
const T_PRINT_F64: u32 = 7; // (f64) -> ()
const T_TRAP: u32 = 8; // (i32, i64, i64) -> ()
const T_ALLOC: u32 = 9; // (i32, i64) -> i32
const FIRST_USER_TYPE: u32 = 10;

/// Emits a complete WebAssembly module for `funcs`.
///
/// The IR must be post-SSA and post-pipeline (which is what the pipeline hands
/// over), must contain exactly one `main`, and must not contain short-circuit
/// operators — the builder lowers those to control flow.
///
/// # Errors
///
/// Returns [`EmitError`] instead of emitting a module whose behaviour could
/// differ from the interpreter: an unknown callee, a missing type, a malformed
/// phi edge, or a function above [`codegen::MAX_DISPATCH_BLOCKS`] blocks.
pub fn emit_wasm(funcs: &[FuncIr]) -> Result<EmittedModule, EmitError> {
    let main_index = funcs
        .iter()
        .position(|f| f.name == "main")
        .ok_or(EmitError::NoMain)?;

    let mut indices: HashMap<String, u32> = HashMap::with_capacity(funcs.len());
    for (offset, function) in funcs.iter().enumerate() {
        if indices
            .insert(function.name.clone(), FIRST_USER_FN + offset as u32)
            .is_some()
        {
            return Err(EmitError::DuplicateName(function.name.clone()));
        }
    }

    // ---- type section -------------------------------------------------------
    let mut types = vec![
        signature(&[], &[]),
        signature(&[I32, I32], &[]),
        signature(&[], &[I64]),
        signature(&[], &[I32]),
        signature(&[I32], &[]),
        signature(&[I64], &[]),
        signature(&[F32], &[]),
        signature(&[F64], &[]),
        signature(&[I32, I64, I64], &[]),
        signature(&[I64, I32], &[I32]), // $alloc: count, element size → base
    ];
    let mut signatures: HashMap<(Vec<u8>, Vec<u8>), u32> = HashMap::new();
    let mut function_types = Vec::with_capacity(funcs.len());
    for function in funcs {
        let (params, results) = wasm_signature(function)?;
        let key = (params.clone(), results.clone());
        let next = FIRST_USER_TYPE + signatures.len() as u32;
        let index = *signatures.entry(key.clone()).or_insert(next);
        if index == next {
            types.push(signature(&params, &results));
        }
        function_types.push(index);
    }

    // ---- import section -----------------------------------------------------
    let mut imports = Vec::with_capacity(6);
    for (name, ty) in [
        ("hx_print_i32", T_PRINT_I32),
        ("hx_print_i64", T_PRINT_I64),
        ("hx_print_f32", T_PRINT_F32),
        ("hx_print_f64", T_PRINT_F64),
        ("hx_print_bool", T_PRINT_I32),
        ("hx_trap", T_TRAP),
    ] {
        let mut entry = Vec::new();
        push_name(&mut entry, "env");
        push_name(&mut entry, name);
        entry.push(0x00); // kind: function
        push_u32(&mut entry, ty);
        imports.push(entry);
    }

    // ---- function section ---------------------------------------------------
    let run_fn = FIRST_USER_FN + funcs.len() as u32;
    let checksum_fn = run_fn + 1;
    let heap_fn = checksum_fn + 1;
    let mut function_section = Vec::new();
    push_u32(&mut function_section, funcs.len() as u32 + 5);
    push_u32(&mut function_section, T_FOLD); // $fold_bytes
    push_u32(&mut function_section, T_ALLOC); // $alloc
    for ty in &function_types {
        push_u32(&mut function_section, *ty);
    }
    push_u32(&mut function_section, T_UNIT_UNIT); // hx_run
    push_u32(&mut function_section, T_CHECKSUM); // hx_checksum
    push_u32(&mut function_section, T_HEAP); // hx_heap_bytes

    // ---- memory section -----------------------------------------------------
    // 32 pages (2 MiB) minimum: enough for the teaching programs and every
    // example in the repository, and `memory.grow` extends it on demand. The
    // first two pages hold the host's print scratch and the fold table.
    let memory_section = push_section(SECTION_MEMORY, {
        let mut out = Vec::new();
        push_u32(&mut out, 1); // one memory
        out.push(0x00); // no maximum
        push_u32(&mut out, 32); // pages
        out
    });

    // ---- global section -----------------------------------------------------
    // 0: bump pointer. 1: running checksum. 2: bytes allocated by this run.
    // 3: how many final arrays are recorded in the fold table.
    let mut globals = Vec::with_capacity(4);
    for (ty, value) in [
        (I32, i64::from(HEAP_BASE)),
        (I64, FNV_OFFSET as i64),
        (I32, 0),
        (I32, 0),
    ] {
        let mut entry = Vec::new();
        entry.push(ty);
        entry.push(0x01); // mutable
        if ty == I32 {
            entry.push(0x41); // i32.const
            encode::push_i32(&mut entry, value as i32);
        } else {
            entry.push(0x42); // i64.const
            encode::push_i64(&mut entry, value);
        }
        entry.push(0x0B); // end
        globals.push(entry);
    }

    // ---- export section -----------------------------------------------------
    let exports = vec![
        export_entry("memory", 0x02, 0),
        export_entry("hx_run", 0x00, run_fn),
        export_entry("hx_fold", 0x00, FN_FOLD),
        export_entry("hx_checksum", 0x00, checksum_fn),
        export_entry("hx_heap_bytes", 0x00, heap_fn),
    ];

    // ---- code section -------------------------------------------------------
    let mut bodies = vec![abi::fold_bytes(), abi::alloc()];
    for (offset, function) in funcs.iter().enumerate() {
        if offset == main_index {
            let folded_at_exit = codegen::folded_arrays(function).len();
            if folded_at_exit > FOLD_TABLE_CAPACITY {
                return Err(EmitError::Unsupported(format!(
                    "'main' has {folded_at_exit} arrays to fold into the checksum, above the \
                     {FOLD_TABLE_CAPACITY}-entry fold table"
                )));
            }
        }
        bodies.push(codegen::lower(
            function,
            indices.clone(),
            offset == main_index,
        )?);
    }
    bodies.push(abi::run(FIRST_USER_FN + main_index as u32));
    bodies.push(abi::checksum());
    bodies.push(abi::heap_bytes());

    // ---- name section (diagnostics only) ------------------------------------
    // A custom section: the section name, then one subsection holding one
    // entry per function index. The browser shows these in stack traces.
    let mut names_body = Vec::new();
    push_name(&mut names_body, "name");
    let mut names: Vec<Vec<u8>> = Vec::with_capacity(funcs.len() + 10);
    let hook_names = [
        "hx_print_i32",
        "hx_print_i64",
        "hx_print_f32",
        "hx_print_f64",
        "hx_print_bool",
        "hx_trap",
        "$fold_bytes",
        "$alloc",
    ];
    for (offset, name) in hook_names.iter().enumerate() {
        names.push(name_entry(offset as u32, name));
    }
    for (offset, function) in funcs.iter().enumerate() {
        names.push(name_entry(
            FIRST_USER_FN + offset as u32,
            &wasm_ident(&function.name),
        ));
    }
    for (offset, name) in [
        (run_fn, "hx_run"),
        (checksum_fn, "hx_checksum"),
        (heap_fn, "hx_heap_bytes"),
    ] {
        names.push(name_entry(offset, name));
    }
    let mut function_names = Vec::new();
    push_u32(&mut function_names, names.len() as u32);
    for entry in &names {
        function_names.extend_from_slice(entry);
    }
    push_u32(&mut names_body, 1); // subsection id: function names
    push_u32(&mut names_body, function_names.len() as u32);
    names_body.extend_from_slice(&function_names);
    let name_section = push_section(SECTION_NAME, names_body);

    let mut module = Vec::with_capacity(bodies.iter().map(Vec::len).sum::<usize>() + 512);
    module.extend_from_slice(&[0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00]);
    module.extend_from_slice(&push_section(SECTION_TYPE, vector_body(&types)));
    module.extend_from_slice(&push_section(SECTION_IMPORT, vector_body(&imports)));
    module.extend_from_slice(&push_section(SECTION_FUNCTION, function_section));
    module.extend_from_slice(&memory_section);
    module.extend_from_slice(&push_section(SECTION_GLOBAL, vector_body(&globals)));
    module.extend_from_slice(&push_section(SECTION_EXPORT, vector_body(&exports)));
    module.extend_from_slice(&push_section(SECTION_CODE, code_body(&bodies)));
    module.extend_from_slice(&name_section);
    Ok(EmittedModule { bytes: module })
}

/// A wasm function type entry: `0x60`, parameter types, result types.
fn signature(params: &[u8], results: &[u8]) -> Vec<u8> {
    let mut entry = Vec::with_capacity(1 + params.len() + results.len() + 2);
    entry.push(0x60);
    push_u32(&mut entry, params.len() as u32);
    entry.extend_from_slice(params);
    push_u32(&mut entry, results.len() as u32);
    entry.extend_from_slice(results);
    entry
}

/// One export entry: name, kind, index.
fn export_entry(name: &str, kind: u8, index: u32) -> Vec<u8> {
    let mut entry = Vec::new();
    push_name(&mut entry, name);
    entry.push(kind);
    push_u32(&mut entry, index);
    entry
}

/// One function-name entry for the name section: index then name.
fn name_entry(index: u32, name: &str) -> Vec<u8> {
    let mut entry = Vec::new();
    push_u32(&mut entry, index);
    push_name(&mut entry, name);
    entry
}

/// The wasm signature of one IR function: arrays occupy two `i32` parameters
/// (pointer, length), unit-returning functions have no result.
fn wasm_signature(function: &FuncIr) -> Result<(Vec<u8>, Vec<u8>), EmitError> {
    let mut params = Vec::new();
    for phi in function
        .block(function.entry)
        .phis
        .iter()
        .filter(|phi| phi.args.is_empty())
    {
        match function
            .types
            .val_tys
            .get(phi.dst.0 as usize)
            .copied()
            .ok_or_else(|| {
                EmitError::Malformed(format!(
                    "{}: parameter v{} has no type",
                    function.name, phi.dst.0
                ))
            })? {
            helix_sema::Ty::Array(_) => params.extend_from_slice(&[I32, I32]),
            helix_sema::Ty::I32 | helix_sema::Ty::Bool => params.push(I32),
            helix_sema::Ty::I64 => params.push(I64),
            helix_sema::Ty::F32 => params.push(F32),
            helix_sema::Ty::F64 => params.push(F64),
            helix_sema::Ty::Unit => {}
        }
    }
    let results = match function.types.ret {
        helix_sema::Ty::Unit => Vec::new(),
        helix_sema::Ty::I32 | helix_sema::Ty::Bool => vec![I32],
        helix_sema::Ty::I64 => vec![I64],
        helix_sema::Ty::F32 => vec![F32],
        helix_sema::Ty::F64 => vec![F64],
        helix_sema::Ty::Array(_) => {
            return Err(EmitError::Unsupported(format!(
                "{} returns an array",
                function.name
            )));
        }
    };
    Ok((params, results))
}

/// The name section is diagnostic only, so a HELIX identifier that would not
/// survive a round trip is transliterated rather than mangled in place.
fn wasm_ident(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '_' | '$' | '.') {
                c
            } else {
                '_'
            }
        })
        .collect()
}
