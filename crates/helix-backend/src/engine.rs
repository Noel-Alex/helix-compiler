//! The [`JitEngine`]: module lifecycle, host runtime symbols, and execution.
//!
//! ## Ownership discipline
//!
//! The engine owns the `JITModule`; its code pages stay valid exactly as long
//! as the engine lives (research digest pitfall 5). Every host symbol is
//! registered on the `JITBuilder` *before* the module exists and declared
//! `Linkage::Import` per function; user functions cross-reference through
//! `declare_func_in_func`. One `finalize_definitions()` covers all functions;
//! `run_main` then transmutes main's finalized pointer to
//! `extern "C" fn()` — legal because every signature used
//! [`CallConv::WindowsFastcall`], which IS Rust's `"C"` ABI on this target.
//!
//! ## Panic containment at the host boundary
//!
//! `helix_panic` records the error message into a thread-safe slot and exits
//! the process with status 1 (spec: runtime errors terminate the program).
//! The JITed panic block returns normally afterwards, so **nothing ever
//! unwinds through JIT frames**. As defense in depth for interpreter-side
//! bugs, `run_main` still wraps the call in `catch_unwind`.
//!
//! ## ParallelPlan seam (M10)
//!
//! With an empty `plan.regions` the compile is fully sequential. When M10
//! lands: each approved loop gets an extracted body function
//! `extern "C" fn(i64 iter, *const Ctx)` compiled into the same module, the
//! header branch is rewritten to a call of imported `helix_parallel_for`,
//! and body pointers are pushed into `helix_runtime::register_body` AFTER
//! finalize (never embedded in code before their addresses exist).

use std::collections::HashMap;

use cranelift::codegen::settings;
use cranelift::prelude::Configurable;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module, default_libcall_names};
use helix_ir::FuncIr;

use crate::lower::{self, CALL_CONV};

/// Host runtime state shared by all JIT-registered symbols of one run.
///
/// A single process-global (the JITBuilder only accepts raw fn pointers) that
/// carries the recorded runtime error between `helix_panic` and `run_main`.
pub(crate) struct HostRt;

impl HostRt {
    /// Pointer-width view for `JITBuilder::symbol` registration.
    pub(crate) fn symbols() -> Vec<(&'static str, *const u8)> {
        vec![
            (
                "helix_print_i64",
                helix_print_i64 as extern "C" fn(i64) as *const u8,
            ),
            (
                "helix_print_f32",
                helix_print_f32 as extern "C" fn(f32) as *const u8,
            ),
            (
                "helix_print_f64",
                helix_print_f64 as extern "C" fn(f64) as *const u8,
            ),
            (
                "helix_print_bool",
                helix_print_bool as extern "C" fn(i64) as *const u8,
            ),
            (
                "helix_zeros",
                helix_zeros as extern "C" fn(i64, i64) -> i64 as *const u8,
            ),
            (
                "helix_len",
                helix_len as extern "C" fn(i64, i64) -> i64 as *const u8,
            ),
            (
                "helix_panic",
                helix_panic as extern "C" fn(i64, i64, i64) as *const u8,
            ),
        ]
    }
}

/// The last recorded runtime error message (set only by `helix_panic`).
static LAST_PANIC: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

/// Recorded trap payload: `(code, aux_a, aux_b)` of the most recent guard hit.
type TrapRecord = (i64, i64, i64);

/// Trap-recorder state for in-process guard tests: `armed` selects
/// record-and-return behaviour inside `helix_panic`; `last` keeps the most
/// recent record. Under an armed recorder control resumes at the JIT panic
/// block's defined fallback return instead of exiting — which is what makes
/// guard tests possible without killing the test runner.
static TRAP_RECORDER: std::sync::Mutex<(bool, Option<TrapRecord>)> =
    std::sync::Mutex::new((false, None));

/// Spec-shaped message text for a numeric panic code (byte-identical to the
/// reference engine's wording where the spec defines one).
fn panic_message(code: i64, a: i64, b: i64) -> String {
    match code {
        0 => "integer division by zero".to_string(),
        1 => "integer division overflow (i64::MIN / -1)".to_string(),
        2 => format!("index {a} out of bounds for array of length {b}"),
        3 => format!("zeros({a}): array length must be non-negative"),
        other => format!("unknown helix panic code {other}"),
    }
}

/// Records the spec-shaped runtime error and terminates with status 1.
///
/// Never returns from a healthy program: `exit(1)` matches the CLI contract
/// (`runtime error: <message>`, exit status 1). Called exclusively from the
/// JITed shared panic block, so no unwinding crosses any JIT frame. The
/// `extern "C"` signature (not `-> !`) is what the JIT side declares; the
/// diverging body is invisible across the boundary because `exit` is `noreturn`.
extern "C" fn helix_panic(code: i64, a: i64, b: i64) {
    let msg = panic_message(code, a, b);
    if let Ok(mut slot) = LAST_PANIC.lock() {
        *slot = Some(msg.clone());
    }
    // Armed recorder: record and return instead of exiting. Sound because
    // every guard's trap block ends in a defined return of zero values (the
    // lowering guarantees it), so resumption never executes garbage.
    if let Ok(mut st) = TRAP_RECORDER.lock()
        && st.0
    {
        st.1 = Some((code, a, b));
        return;
    }
    eprintln!("runtime error: {msg}");
    std::process::exit(1);
}

/// Arms the trap recorder: subsequent traps are recorded (readable via
/// [`take_last_trap`]) instead of exiting the process. Exposed through
/// [`crate::testutil`].
#[doc(hidden)]
pub fn arm_trap_recorder() {
    match TRAP_RECORDER.lock() {
        Ok(mut st) => {
            st.0 = true;
            st.1 = None;
        }
        Err(e) => {
            let mut st = e.into_inner();
            st.0 = true;
            st.1 = None;
        }
    }
}

/// Disarms the trap recorder (restores exiting production behaviour).
#[doc(hidden)]
pub fn disarm_trap_recorder() {
    match TRAP_RECORDER.lock() {
        Ok(mut st) => {
            st.0 = false;
            st.1 = None;
        }
        Err(e) => {
            let mut st = e.into_inner();
            st.0 = false;
            st.1 = None;
        }
    }
}

/// Reads and clears the recorded trap `(code, a, b)`; `None` when no trap
/// fired since arming or when the recorder is disarmed (production mode).
#[doc(hidden)]
#[must_use]
pub fn take_last_trap() -> Option<(i64, i64, i64)> {
    match TRAP_RECORDER.lock() {
        Ok(mut st) => st.1.take(),
        Err(e) => e.into_inner().1.take(),
    }
}

// Output sink of the host print builtins.
//
// Normally lines go to stdout. When a capture buffer is installed for the
// current thread (tests, and later the Observatory), lines are appended
// there instead — the JIT runs on the installing thread, so a thread-local
// is exactly the right scope.
thread_local! {
    static CAPTURE: std::cell::RefCell<Option<Vec<String>>> = const { std::cell::RefCell::new(None) };
}

/// Installs a print capture for the duration of `f` (tests + selftests).
///
/// Returns every line printed while `f` ran, in order, without newlines.
/// Exposed through [`crate::testutil`]; not part of the production surface.
#[doc(hidden)]
pub fn capture_prints<R>(f: impl FnOnce() -> R) -> (Vec<String>, R) {
    CAPTURE.with(|c| *c.borrow_mut() = Some(Vec::new()));
    let out = f();
    let lines = CAPTURE.with(|c| c.borrow_mut().take().unwrap_or_default());
    (lines, out)
}

/// Serializes engine runs that touch process-global state (`LAST_PANIC`,
/// hook installation). Tests hold this guard; single-program production runs
/// never contend.
#[doc(hidden)]
pub fn serial_lock() -> &'static std::sync::Mutex<()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    &LOCK
}

/// Reads and clears the last recorded runtime error (tests).
#[doc(hidden)]
#[allow(dead_code)] // reserved for embedders that pull the last panic after run
pub(crate) fn take_last_panic() -> Option<String> {
    match LAST_PANIC.lock() {
        Ok(mut slot) => slot.take(),
        Err(e) => e.into_inner().take(),
    }
}

/// Emits one formatted line to the active sink.
fn emit_line(line: String) {
    CAPTURE.with(|c| {
        if let Some(buf) = c.borrow_mut().as_mut() {
            buf.push(line);
            return;
        }
        println!("{line}");
    });
}

/// Host side of `print` for integers (any width; widened by lowering).
extern "C" fn helix_print_i64(v: i64) {
    emit_line(helix_sema::fmt_i64(v));
}

extern "C" fn helix_print_f32(v: f32) {
    emit_line(helix_sema::fmt_f32(v));
}

extern "C" fn helix_print_f64(v: f64) {
    emit_line(helix_sema::fmt_f64(v));
}

extern "C" fn helix_print_bool(v: i64) {
    emit_line(helix_sema::fmt_bool(v != 0));
}

/// Allocates a zero-initialized buffer of `n` elements of `elem_size` bytes.
///
/// Live allocations from `helix_zeros`, tracked so [`reset_host_heap`] can
/// reclaim them. The bench harness runs hundreds of programs per process;
/// without this table each run's arrays would accumulate (measured: 18 GB
/// during a full campaign).
static LIVE_BUFS: std::sync::Mutex<Vec<(i64, usize)>> = std::sync::Mutex::new(Vec::new());

/// Frees every array allocation made by JITed code since the last call, and
/// clears print/panic/trap state.
///
/// # Safety-critical contract
/// Call ONLY between program executions — never while JITed code is running
/// and never dereferencing a fat pointer handed out before the reset (the
/// bench harness and tests satisfy this by construction). Single CLI runs
/// never need to call it (buffers die with the process).
pub fn reset_host_heap() {
    if let Ok(mut bufs) = LIVE_BUFS.lock() {
        for (ptr, len) in bufs.drain(..) {
            // SAFETY: `ptr` came from `Vec::into_raw_parts_style` allocation
            // below with this exact length/capacity and is freed exactly once
            // (entries are drained under the lock).
            unsafe {
                drop(Vec::from_raw_parts(ptr as *mut u8, len, len));
            }
        }
    }
    LAST_PANIC.lock().map(|mut g| g.take()).ok();
    TRAP_RECORDER.lock().map(|mut g| *g = (g.0, None)).ok();
}

/// Allocating variant used by `helix_zeros`: tracks `(ptr, len)` for reclamation.
extern "C" fn helix_zeros(n: i64, elem_size: i64) -> i64 {
    if n < 0 {
        helix_panic(3, n, 0);
    }
    let bytes =
        usize::try_from(n.checked_mul(elem_size).unwrap_or(i64::MAX)).unwrap_or(usize::MAX - 1);
    let vec = vec![0u8; bytes.max(1)]; // ≥1 byte keeps dangling-pointer rules happy at n=0
    let len = vec.len();
    let ptr = Box::into_raw(vec.into_boxed_slice());
    if let Ok(mut bufs) = LIVE_BUFS.lock() {
        bufs.push((ptr as *const u8 as i64, len));
    }
    ptr as *mut u8 as i64
}

/// Fat-pointer `len` identity (the pointer half is ignored).
extern "C" fn helix_len(_ptr: i64, len: i64) -> i64 {
    len
}

// ---------------------------------------------------------------------------

/// A compiled program: all functions resident, main callable.
///
/// Dropping the engine frees the JIT code pages (`free_memory`); function
/// pointers handed out by [`JitEngine::function_ptr`] die with it — keep the
/// engine alive while any code may still run (research digest pitfall 5).
/// Manual `Debug` (JITModule lacks one): print shape, not code bytes.
impl std::fmt::Debug for JitEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JitEngine")
            .field("functions", &self.funcs.keys().collect::<Vec<_>>())
            .field("main", &"FuncId(main)")
            .finish_non_exhaustive()
    }
}

pub struct JitEngine {
    /// Owns the executable memory; `Some` until Drop consumes it.
    module: Option<JITModule>,
    funcs: HashMap<String, FuncId>,
    main_fid: FuncId,
}

impl JitEngine {
    /// Compiles every function of `program` into one JIT module and finalizes.
    ///
    /// `plan`'s approved loops become extracted body functions dispatched
    /// through the helix-runtime fork/join machinery ([`crate::parallel`]);
    /// regions whose shape the transform cannot express are silently demoted
    /// to the sequential lowering. `unchecked` strips bounds checks
    /// (`--unchecked`); division guards always remain.
    ///
    /// # Errors
    /// Any CLIF/module-level failure (verification, define, finalize), or a
    /// missing `main`.
    pub fn compile(
        program: &[FuncIr],
        plan: &ParallelPlan,
        unchecked: bool,
    ) -> Result<JitEngine, String> {
        // ---- plan preparation -------------------------------------------------
        // Extract every expressible region up front; parents with at least one
        // surviving region get the dispatch hook during lowering, everything
        // else lowers exactly as before.
        let regions = crate::parallel::prepare(plan, program);
        let mut hooks: HashMap<usize, crate::parallel::RtHook> = HashMap::new();
        let mut bodies: Vec<crate::parallel::BodyArtifact> = Vec::new();
        for r in &regions {
            hooks
                .entry(r.func_idx)
                .or_insert_with(|| crate::parallel::RtHook::of(r));
            let body = r
                .body
                .clone()
                .expect("prepare keeps extractable regions only");
            bodies.push(crate::parallel::BodyArtifact {
                name: r.body_fn_name.clone(),
                ir: body,
                prebind: r.array_prebind.clone(),
            });
            crate::parallel::register_spec(r.region_id, r.spec_meta());
        }
        // Dispatcher metadata must exist even for regions that later fail to
        // compile — registration above happens once per surviving region.

        // --- flags + ISA + builder (jit-minimal.rs skeleton) ---------------
        let mut flag_builder = settings::builder();
        flag_builder
            .set("use_colocated_libcalls", "false")
            .map_err(|e| e.to_string())?;
        flag_builder
            .set("is_pic", "false")
            .map_err(|e| e.to_string())?;
        let isa_builder = cranelift_native::builder().map_err(|e| e.to_string())?;
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .map_err(|e| format!("ISA setup failed: {e}"));

        let mut jb = JITBuilder::with_isa(isa?, default_libcall_names());
        for (name, ptr) in HostRt::symbols() {
            jb.symbol(name, ptr);
        }
        for (name, ptr) in crate::parallel::host_symbols() {
            jb.symbol(name, ptr);
        }
        let mut module = JITModule::new(jb);

        // --- declare every function first (cross-references need ids) ------
        let mut fn_sigs: Vec<cranelift::codegen::ir::Signature> = Vec::with_capacity(program.len());
        let mut funcs: HashMap<String, FuncId> = HashMap::new();
        let mut sigs_by_name: HashMap<String, cranelift::codegen::ir::Signature> = HashMap::new();
        for f in program {
            let sig = lower::signature_of(f);
            let fid = module
                .declare_function(&f.name, Linkage::Local, &sig)
                .map_err(|e| format!("declaring '{}': {e}", f.name))?;
            fn_sigs.push(sig.clone());
            sigs_by_name.insert(f.name.clone(), sig);
            funcs.insert(f.name.clone(), fid);
        }

        // Declare extracted parallel-region bodies: `extern "C" fn(i64 iter,
        // i64 ctx)` — two I64 params, no returns (WindowsFastcall == Rust's
        // extern "C" on this target).
        let body_sig = crate::parallel::body_signature();
        for body in &bodies {
            let name = &body.name;
            let fid = module
                .declare_function(name, Linkage::Local, &body_sig)
                .map_err(|e| format!("declaring region body '{name}': {e}"))?;
            sigs_by_name.insert(name.clone(), body_sig.clone());
            funcs.insert(name.clone(), fid);
        }

        // Declare the host builtins this program can reach.
        const BUILTINS: &[&str] = &[
            "helix_print_i64",
            "helix_print_f32",
            "helix_print_f64",
            "helix_print_bool",
            "helix_zeros",
            "helix_len",
            "helix_panic",
        ];
        for name in BUILTINS {
            let Some(sig) = lower::builtin_signature(name) else {
                return Err(format!("internal: no signature table row for {name}"));
            };
            let fid = module
                .declare_function(name, Linkage::Import, &sig)
                .map_err(|e| format!("declaring builtin '{name}': {e}"))?;
            sigs_by_name.insert((*name).to_string(), sig);
            funcs.insert((*name).to_string(), fid);
        }

        // Declare the dispatch-ABI host symbols (stash/dispatch/readback and
        // the body-context imports extracted regions call).
        for name in crate::parallel::HOST_SYMBOL_NAMES {
            let Some(sig) = lower::builtin_signature(name) else {
                return Err(format!("internal: no dispatch-ABI signature for {name}"));
            };
            let fid = module
                .declare_function(name, Linkage::Import, &sig)
                .map_err(|e| format!("declaring dispatch symbol '{name}': {e}"))?;
            sigs_by_name.insert((*name).to_string(), sig);
            funcs.insert((*name).to_string(), fid);
        }

        // --- translate + define bodies -------------------------------------
        for (fi, f) in program.iter().enumerate() {
            let mut ctx = module.make_context();
            ctx.func.signature = fn_sigs[fi].clone();
            lower::translate_fn_rt(
                f,
                unchecked,
                &mut ctx.func,
                &mut module,
                &funcs,
                &sigs_by_name,
                hooks.remove(&fi),
            )?;
            let fid = funcs[&f.name];
            // Verify first so failures carry a precise CLIF location instead
            // of the opaque "Verifier errors" wrapper.
            if let Err(errs) = ctx.verify(module.isa()) {
                return Err(format!("verifying '{}': {errs:?}", f.name));
            }
            module
                .define_function(fid, &mut ctx)
                .map_err(|e| format!("defining '{}': {e}", f.name))?;
            module.clear_context(&mut ctx);
        }

        // Extracted region bodies: lowered with array fat pointers prebound to
        // their ctx slots (no zeros call runs inside the region).
        for body in &bodies {
            let mut ctx = module.make_context();
            ctx.func.signature = crate::parallel::body_signature();
            lower::translate_body_fn(
                &body.ir,
                unchecked,
                &mut ctx.func,
                &mut module,
                &funcs,
                &sigs_by_name,
                &body.prebind,
            )?;
            let fid = funcs[&body.name];
            if let Err(errs) = ctx.verify(module.isa()) {
                return Err(format!("verifying region body '{}': {errs:?}", body.name));
            }
            module
                .define_function(fid, &mut ctx)
                .map_err(|e| format!("defining region body '{}': {e}", body.name))?;
            module.clear_context(&mut ctx);
        }

        module
            .finalize_definitions()
            .map_err(|e| format!("finalizing definitions: {e}"))?;

        // --- register region bodies/combines AFTER finalize ------------------
        // Contract rule: never embed or share pointers before their addresses
        // exist; registry entries are keyed by the same ids baked into the
        // parent's dispatch calls.
        let _registered = {
            let mut reg = Vec::with_capacity(regions.len());
            for r in &regions {
                let Some(fid) = funcs.get(&r.body_fn_name) else {
                    continue;
                };
                let ptr = module.get_finalized_function(*fid);
                debug_assert!(!ptr.is_null(), "region body finalized");
                // SAFETY: `ptr` is a finalized `extern "C" fn(i64, *mut u8)`
                // emitted from `body_signature` (see above); transmute mirrors
                // `run_main`'s established pattern.
                let body_fn: helix_runtime::BodyFn =
                    unsafe { std::mem::transmute::<*const u8, helix_runtime::BodyFn>(ptr) };
                helix_runtime::register_body(r.region_id, body_fn);
                helix_runtime::register_combine(r.region_id, crate::parallel::combine_for(r.kind));
                reg.push(r.region_id);
            }
            reg
        };

        let main_fid = funcs
            .get("main")
            .copied()
            .ok_or_else(|| "program has no fn main()".to_string())?;

        Ok(JitEngine {
            module: Some(module),
            funcs,
            main_fid,
        })
    }

    /// Calls the JIT-compiled `main()` (no params, unit return).
    ///
    /// Panics inside JIT code cannot happen by construction (all runtime
    /// errors funnel through `helix_panic`, which exits), but the call is
    /// wrapped in `catch_unwind` anyway per the research digest's host-
    /// boundary rule.
    ///
    /// # Errors
    /// A caught unwind (never expected) or a recorded-but-unreported panic.
    pub fn run_main(&self) -> Result<(), String> {
        let module = self.module.as_ref().expect("module alive until Drop");
        let ptr = module.get_finalized_function(self.main_fid);
        debug_assert!(!ptr.is_null(), "main finalized");
        let main_fn: extern "C" fn() = unsafe {
            // SAFETY: `ptr` comes from get_finalized_function after
            // finalize_definitions succeeded; the module (and thus the code
            // pages) is owned by self and outlives the call. The signature
            // matches our WindowsFastcall main() (no params, no returns),
            // which equals extern "C" on x86_64-pc-windows-msvc.
            std::mem::transmute::<*const u8, extern "C" fn()>(ptr)
        };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| main_fn()));
        match result {
            Ok(()) => Ok(()),
            Err(payload) => Err(format!(
                "JIT code panicked unexpectedly: {}",
                payload
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "non-string panic payload".into())
            )),
        }
    }

    /// Finalized pointer of a named function (tests / future parallel driver).
    #[must_use]
    pub fn function_ptr(&self, name: &str) -> Option<*const u8> {
        self.funcs.get(name).map(|fid| {
            self.module
                .as_ref()
                .expect("module alive until Drop")
                .get_finalized_function(*fid)
        })
    }

    /// Calling convention used for every emitted signature (docs/tests).
    #[must_use]
    pub const fn calling_convention() -> cranelift::codegen::isa::CallConv {
        CALL_CONV
    }
}

impl Drop for JitEngine {
    fn drop(&mut self) {
        // Free executable memory deterministically. SAFETY: the engine owned
        // the code pages and no calls are in flight (run_main borrows self
        // for its whole duration).
        if let Some(module) = self.module.take() {
            unsafe { module.free_memory() };
        }
    }
}

/// Placeholder for the M10 plan type (contract: `helix_analysis` will own the
/// real struct; backend consumes it read-only).
///
/// Defined locally for now so the backend has zero dependency on
/// helix-analysis's evolving internals; M10 swaps this for
/// `helix_analysis::plan::ParallelPlan` verbatim (same shape).
#[derive(Clone, Debug, Default)]
pub struct ParallelPlan {
    /// One descriptor per approved/reduction loop (empty ⇒ sequential).
    pub regions: Vec<RegionDesc>,
}

/// One parallel region description (M10 seam; see interface-contracts.md).
#[derive(Clone, Debug)]
pub struct RegionDesc {
    /// Index of the owning function in the compiled slice.
    pub func_idx: usize,
    /// Header block of the loop being replaced.
    pub header: helix_ir::BlockId,
    /// DoAll or Reduction(op) — drives accumulator plumbing later.
    pub kind: RegionKind,
    /// Extracted body function name (e.g. "main.loop0.body").
    pub body_fn_name: String,
}

/// Kind of a planned region (M10 seam).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegionKind {
    /// Independent iterations.
    DoAll,
    /// Private accumulators + post-join combine.
    Reduction(helix_analysis_stub::ReductionOp),
}

/// Temporary stand-in so RegionKind compiles without depending on
/// helix-analysis; replaced wholesale in M10.
#[allow(unused)]
pub mod helix_analysis_stub {
    /// Reduction operator tag (M10 swaps in helix_analysis::ReductionOp).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum ReductionOp {
        /// Sum/difference accumulation.
        Add,
        /// Product accumulation.
        Mul,
        /// Running minimum.
        Min,
        /// Running maximum.
        Max,
    }
}
