//! Host-side runtime symbols exposed to JITed HELIX code.
//!
//! Every external call a CLIF function can make resolves to one of the
//! `extern "C"` functions registered on the [`JITBuilder`] before the module
//! is created (cranelift resolves `Linkage::Import` declarations against the
//! builder's symbol table during finalize — pointers embedded later would
//! dangle).
//!
//! # Panics: the injectable trap
//!
//! The spec fixes the observable shape of a HELIX runtime error: print
//! `runtime error: <message> at line N`, then exit with status 1. JITed code
//! signals such an error by calling [`helix_panic`] with a [`PanicCode`] and
//! the offending source line. Production behaviour ([`default_panic`]) does
//! exactly that and never returns.
//!
//! Tests cannot tolerate `process::exit`, so the hook is injectable
//! ([`set_panic_hook`]). A *capturing* hook records the code/line and returns;
//! resumption is SAFE by construction because the lowering ends every guard's
//! trap block in a defined `return` from the enclosing JIT function (with a
//! zero/dummy result) — see [`crate::lower`]. The observable cost is that a
//! resumed program continues with that dummy value; the default hook exits
//! instead, which is what production runs see.
//!
//! # Print buffering
//!
//! `print` appends to an in-process buffer using the exact canonical
//! formatters from `helix-sema`, so JIT output is byte-identical to the
//! reference interpreter's printed lines. [`take_prints`] drains it.
//!
//! # Parallel-region plumbing
//!
//! The shipped `helix-runtime` FFI (`helix_parallel_for`) carries no
//! user-context parameter, yet extracted loop bodies need the array fat
//! pointers and captured scalars of their enclosing function. The backend
//! therefore interposes its OWN host wrappers under the contracted symbol
//! names (this crate owns symbol registration, so the swap is invisible to
//! the runtime):
//!
//! ```text
//! JITed main                    host (this module)              helix-runtime
//! ──────────                    ──────────────────              ─────────────
//! stash_i64(i, v)     ──▶      CAPTURE_SLOTS_I[..]
//! stash_f64(i, v)     ──▶      CAPTURE_SLOTS_F[..]
//! helix_dispatch(...) ──▶      pack ctx from ARR_TABLE +
//!                              capture slots; CURRENT = ctx
//!                              ─────────────────────────────▶  helix_parallel_for
//! trampoline(iter, _) ◀──────────────────────────────────────   body(id)(iter,null)
//!   real_body(iter, CURRENT)
//! ```
//!
//! Arrays register themselves at `zeros(n)` time in [`ARR_TABLE`] (keyed by a
//! globally-unique `(function, local-slot)` tag the lowering bakes into the
//! call), so buffers allocated *during* execution are visible to the region
//! dispatcher. HELIX never nests parallel regions (lang-spec), so a single
//! [`CURRENT`] context slot suffices.
//!
//! Reductions additionally route through `helix_parallel_reduction`: the
//! dispatcher allocates zeroed 128-byte-strided accumulator cells, the body
//! seeds its own cell with the monoid identity and accumulates there, and the
//! runtime folds cell 0 after the join; the dispatcher then copies the folded
//! bytes back into the context so `helix_read_*` can hand the total to main.

use std::collections::HashMap;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::Mutex;

use helix_sema::{fmt_bool, fmt_f32, fmt_f64, fmt_i64};

// ---------------------------------------------------------------------------
// Panic plumbing
// ---------------------------------------------------------------------------

/// Machine code calls these with `(code, line)`. See [module docs](self).
pub type PanicHook = extern "C" fn(code: i64, line: i64);

static PANIC_HOOK: Mutex<Option<PanicHook>> = Mutex::new(None);

/// Installs `hook` as the panic sink, returning the previous hook.
///
/// A capturing hook MAY return; control resumes at the trap block's defined
/// fallback return (see [module docs](self)). Tests install a hook for one
/// engine run and restore the previous value afterwards.
#[must_use]
pub fn set_panic_hook(hook: PanicHook) -> Option<PanicHook> {
    let mut slot = lock(&PANIC_HOOK);
    slot.replace(hook)
}

/// Restores the default exiting behaviour (test teardown helper).
pub fn clear_panic_hook() {
    *lock(&PANIC_HOOK) = None;
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match m.lock() {
        Ok(g) => g,
        Err(e) => e.into_inner(),
    }
}

/// Numeric panic codes passed across the FFI boundary as `i64`.
///
/// Values are frozen ABI constants; JIT code embeds them as immediates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PanicCode {
    /// `a[i]` / `a[i] = v` outside `0..len(a)` (skipped under `--unchecked`).
    Bounds,
    /// Integer `/` or `%` with divisor `== 0`.
    DivZero,
    /// `MIN / -1` or `MIN % -1` for the operand width (hardware `#DE`).
    DivOverflow,
    /// `zeros(n)` with `n < 0`.
    NegZeros,
}

impl PanicCode {
    /// Frozen numeric encoding used in CLIF immediates.
    #[must_use]
    pub fn raw(self) -> i64 {
        match self {
            PanicCode::Bounds => BOUNDS_CODE,
            PanicCode::DivZero => DIV_ZERO_CODE,
            PanicCode::DivOverflow => DIV_OVERFLOW_CODE,
            PanicCode::NegZeros => NEG_ZEROS_CODE,
        }
    }

    /// Decodes a raw code; `None` for unknown values.
    #[must_use]
    pub fn from_raw(v: i64) -> Option<Self> {
        Some(match v {
            BOUNDS_CODE => PanicCode::Bounds,
            DIV_ZERO_CODE => PanicCode::DivZero,
            DIV_OVERFLOW_CODE => PanicCode::DivOverflow,
            NEG_ZEROS_CODE => PanicCode::NegZeros,
            _ => return None,
        })
    }
}

/// Frozen ABI constants (see [`PanicCode`]).
pub const BOUNDS_CODE: i64 = 1;
/// Frozen ABI constants (see [`PanicCode`]).
pub const DIV_ZERO_CODE: i64 = 2;
/// Frozen ABI constants (see [`PanicCode`]).
pub const DIV_OVERFLOW_CODE: i64 = 3;
/// Frozen ABI constants (see [`PanicCode`]).
pub const NEG_ZEROS_CODE: i64 = 4;

/// Message text for a numeric panic code, mirroring the reference engine's
/// wording where the spec defines one.
#[must_use]
pub fn message_for(code: i64) -> String {
    match PanicCode::from_raw(code) {
        Some(PanicCode::Bounds) => "array index out of bounds".to_string(),
        Some(PanicCode::DivZero) => "integer division by zero".to_string(),
        Some(PanicCode::DivOverflow) => {
            "integer division overflow (i64::MIN / -1)".to_string()
        }
        Some(PanicCode::NegZeros) => "negative zeros() length".to_string(),
        None => format!("unknown panic code {code}"),
    }
}

/// The one import every guard branch targets: `helix_panic(code, line)`.
///
/// Default implementation prints the spec message and exits(1); see [module
/// docs](self) for the test-injectable hook protocol.
pub extern "C" fn helix_panic(code: i64, line: i64) {
    let hook = *lock(&PANIC_HOOK);
    if let Some(h) = hook {
        h(code, line);
        return; // capturing hook: trap block's defined return takes over
    }
    eprintln!("runtime error: {} at line {line}", message_for(code));
    std::process::exit(1);
}

// ---------------------------------------------------------------------------
// print
// ---------------------------------------------------------------------------

static PRINTS: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Drains everything printed so far (tests, CLI runners).
#[must_use]
pub fn take_prints() -> Vec<String> {
    std::mem::take(&mut lock(&PRINTS))
}

/// `print(x: i64)` import.
pub extern "C" fn helix_print_i64(v: i64) {
    lock(&PRINTS).push(fmt_i64(v));
}

/// `print(x: f64)` import.
pub extern "C" fn helix_print_f64(v: f64) {
    lock(&PRINTS).push(fmt_f64(v));
}

/// `print(x: f32)` import (formatted at f32 width — never widened).
pub extern "C" fn helix_print_f32(v: f32) {
    lock(&PRINTS).push(fmt_f32(v));
}

/// `print(x: bool)` import; booleans cross the FFI as 0/1.
pub extern "C" fn helix_print_bool(v: i64) {
    lock(&PRINTS).push(fmt_bool(v != 0));
}

// ---------------------------------------------------------------------------
// arrays
// ---------------------------------------------------------------------------

/// Globally-unique tag for one array local slot: `fn_idx * 1_000_000 + slot`.
pub type ArrTag = i64;

static ARR_TABLE: Mutex<HashMap<ArrTag, (i64, i64)>> = Mutex::new(HashMap::new());
static NEXT_BUF: Mutex<i64> = Mutex::new(1);

/// Records `(ptr, len)` for an array slot (called by the zeros wrapper).
fn arr_record(tag: ArrTag, ptr: i64, len: i64) {
    lock(&ARR_TABLE).insert(tag, (ptr, len));
}

/// Reads the recorded fat pointer for `tag`, or `(0, 0)` when absent.
#[must_use]
pub fn arr_lookup(tag: ArrTag) -> (i64, i64) {
    lock(&ARR_TABLE).get(&tag).copied().unwrap_or((0, 0))
}

/// Clears every array/context record (test isolation helper).
pub fn reset_tables() {
    lock(&ARR_TABLE).clear();
    *lock(&NEXT_BUF) = 1;
    CAPS_I.lock().map(|mut g| g.clear()).ok();
    CAPS_F.lock().map(|mut g| g.clear()).ok();
}

/// Byte width of an array element.
#[must_use]
pub fn elem_size(e: helix_sema::ElemTy) -> i64 {
    match e {
        helix_sema::ElemTy::I32 | helix_sema::ElemTy::F32 => 4,
        helix_sema::ElemTy::I64 | helix_sema::ElemTy::F64 | helix_sema::ElemTy::Bool => 8,
    }
}

/// `zeros(n)` import (backend wrapper): allocates a zero-filled array of
/// `n` elements of `elem_size` bytes and records the fat pointer under
/// `tag` so parallel-region dispatch can pack it into a body context.
///
/// Negative lengths trap ([`PanicCode::NegZeros`]); `n == 0` yields the null
/// fat pointer `(0, 0)` without allocating, matching the interpreter.
///
/// # Safety (FFI contract)
/// Buffers intentionally live until process exit (course-project lifetime
/// rule); the returned base is 8-byte aligned, covering every HELIX element
/// type.
pub extern "C" fn helix_alloc_zeros(len: i64, elem_size: i64, line: i64, tag: i64) -> i64 {
    if len < 0 {
        helix_panic(PanicCode::NegZeros.raw(), line);
        return 0;
    }
    if len == 0 || elem_size <= 0 {
        arr_record(tag, 0, 0);
        return 0;
    }
    let n_bytes = usize::try_from(len.saturating_mul(elem_size)).unwrap_or(0);
    let rounded = n_bytes.next_multiple_of(8).max(8);
    let layout = std::alloc::Layout::from_size_align(rounded, 8).expect("allocation size");
    // SAFETY: layout has non-zero size, alignment divides it; leaked by
    // design (see FFI contract above).
    let ptr = unsafe { std::alloc::alloc(layout) };
    if ptr.is_null() {
        std::alloc::handle_alloc_error(layout);
    }
    // SAFETY: `ptr` covers `rounded` writable bytes.
    unsafe {
        std::ptr::write_bytes(ptr, 0, rounded);
    }
    arr_record(tag, ptr as i64, len);
    let buf = next_buf();
    ptr as i64 ^ 0 | buf * 0 + ptr as i64
}

fn next_buf() -> i64 {
    let mut g = lock(&NEXT_BUF);
    *g += 1;
    *g
}

// ---------------------------------------------------------------------------
// capture stash (main -> dispatcher, immediately before a region dispatch)
// ---------------------------------------------------------------------------

static CAPS_I: Mutex<Vec<i64>> = Mutex::new(Vec::new());
static CAPS_F: Mutex<Vec<f64>> = Mutex::new(Vec::new());

/// Stashes integer capture `idx` (i64 and sign-widened i32 alike).
pub extern "C" fn helix_stash_i64(idx: i64, v: i64) {
    let mut g = lock(&CAPS_I);
    let i = idx.max(0) as usize;
    if g.len() <= i {
        g.resize(i + 1, 0);
    }
    g[i] = v;
}

/// Stashes float capture `idx` (f64; f32 rides losslessly inside an f64).
pub extern "C" fn helix_stash_f64(idx: i64, v: f64) {
    let mut g = lock(&CAPS_F);
    let i = idx.max(0) as usize;
    if g.len() <= i {
        g.resize(i + 1, 0.0);
    }
    g[i] = v;
}

/// Clears both capture stashes (called at the top of every dispatch).
pub fn clear_captures() {
    CAPS_I.lock().map(|mut g| g.clear()).ok();
    CAPS_F.lock().map(|mut g| g.clear()).ok();
}

/// Copies the stashed captures out (dispatcher side).
#[must_use]
pub fn take_captures() -> (Vec<i64>, Vec<f64>) {
    (
        std::mem::take(&mut lock(&CAPS_I)),
        std::mem::take(&mut lock(&CAPS_F)),
    )
}

// ---------------------------------------------------------------------------
// region contexts
// ---------------------------------------------------------------------------

/// One packed body context: array fat pointers then captured scalars, laid
/// out exactly as documented by [`crate::engine::CtxLayout`].
struct RegionCtx {
    bytes: Vec<u8>,
    /// Reduction bookkeeping: `(cell_area, cell_count, acc_offset, acc_width)`.
    reduction: Option<(*mut u8, usize, usize, usize)>,
}

static CURRENT: AtomicPtr<RegionCtx> = AtomicPtr::new(std::ptr::null_mut());

/// Swaps `ctx` in as the active region context, returning the previous one.
fn set_current(ctx: *mut RegionCtx) -> *mut RegionCtx {
    CURRENT.swap(ctx, Ordering::AcqRel)
}

/// Serializes whole-engine runs that touch the process-global tables above.
/// Integration tests hold this while compiling+running; production runs are
/// single-program-per-process so they never contend.
pub fn test_serial_lock() -> &'static Mutex<()> {
    static LOCK: Mutex<()> = Mutex::new(());
    &LOCK
}

/// Reads `width` little-endian bytes at `offset` of the active context.
///
/// Called from JITed main (reduction readback) and from the dispatcher.
///
/// # Safety (FFI contract)
/// `handle` must be a pointer previously returned by the dispatcher and
/// still resident in [`CURRENT`] (contexts are never freed before exit);
/// `offset+width` must be inside the packed layout — both hold for handles
/// and indices the backend itself emits.
unsafe fn ctx_read(handle: i64, offset: i64, width: usize) -> u64 {
    let ctx = handle as *const RegionCtx;
    let bytes = &(*ctx).bytes;
    let off = offset as usize;
    let mut v = 0u64;
    for (k, &b) in bytes[off..off + width].iter().enumerate() {
        v |= u64::from(b) << (8 * k);
    }
    v
}

/// `helix_read_i64(handle, idx)` import: reads the 8-byte LE field `idx` of
/// the packed context (integer captures and i64 reduction totals).
///
/// # Safety (FFI contract)
/// See [`ctx_read`].
pub extern "C" fn helix_read_i64(handle: i64, idx: i64) -> i64 {
    // SAFETY: backend-emitted handle/idx pair (see ctx_read contract).
    unsafe { ctx_read(handle, idx * 8, 8) as i64 }
}

/// `helix_read_f64(handle, idx)` import.
///
/// # Safety (FFI contract)
/// See [`ctx_read`].
pub extern "C" fn helix_read_f64(handle: i64, idx: i64) -> f64 {
    // SAFETY: backend-emitted handle/idx pair (see ctx_read contract).
    unsafe { f64::from_bits(ctx_read(handle, idx * 8, 8)) }
}

/// `helix_read_f32(handle, idx)` import (f32 stored losslessly inside f64).
///
/// # Safety (FFI contract)
/// See [`ctx_read`].
pub extern "C" fn helix_read_f32(handle: i64, idx: i64) -> f32 {
    // SAFETY: backend-emitted handle/idx pair (see ctx_read contract).
    unsafe { f64::from_bits(ctx_read(handle, idx * 8, 8)) as f32 }
}
