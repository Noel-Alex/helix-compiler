//! # helix-backend — CLIF lowering and JIT execution
//!
//! Stage 5 of the HELIX pipeline: consumes SSA-form [`helix_ir::FuncIr`]s and
//! lowers each one to a Cranelift CLIF function, compiled to native x86-64
//! machine code in-process by `cranelift-jit`.
//!
//! ```text
//! Vec<FuncIr> ──lower──▶ Vec<CLIF> ──finalize──▶ native code ──run_main──▶ stdout
//! ```
//!
//! ## Module map
//!
//! * [`lower`] — per-function lowering: signatures from sema [`Ty`]s, φ →
//!   block parameters, checked arithmetic/bounds guards as compare-and-branch
//!   into a shared panic block, host builtins as imported symbols, user calls
//!   via `declare_func_in_func`.
//! * [`engine`] — the [`JitEngine`]: owns the `JITModule`, builds every
//!   function, finalizes once, and exposes `run_main`. Host runtime symbols
//!   (`helix_print_*`, `helix_zeros`, `helix_panic`, …) live here.
//!
//! ## Calling convention (normative)
//!
//! Every signature uses [`CallConv::WindowsFastcall`], which is bit-for-bit
//! Rust's `extern "C"` on x86_64-pc-windows-msvc. SystemV signatures on
//! Windows read arguments from the wrong registers past four parameters and
//! corrupt silently — never used (see `docs/research/cranelift-api.md`).
//!
//! Scalars map by width: `i32→I32`, `i64→I64`, `f32→F32`, `f64→F64`,
//! `bool→I8`. Arrays are fat pointers lowered as **two consecutive I64
//! parameters** (data pointer, element count), which keeps every array a
//! first-class pair under Fastcall's register ordering.
//!
//! ## Checked semantics without OS faults
//!
//! Division/remainder guards and array bounds checks are emitted as explicit
//! compares branching to a per-function panic block that calls the imported
//! host symbol `helix_panic(code)`; the host prints the spec-shaped message
//! and exits(1). A faulting `sdiv` would otherwise surface as an SEH exception
//! deep inside JIT frames — fragile and Windows-specific.
//!
//! ## Parallel seam (M10)
//!
//! [`engine::JitEngine::compile`] accepts a `helix_analysis::ParallelPlan`.
//! With an empty plan the lowering is fully sequential; the parallel-region
//! extension (extracted loop bodies, context packing, runtime registration)
//! lands in M10 on top of the seams documented in [`lower`] and [`engine`].

pub mod engine;
pub mod lower;
#[doc(hidden)]
#[allow(missing_docs)]
pub mod parallel;

/// M0 de-risk spike: proves the pinned cranelift 0.135 JIT flow works on this
/// machine (Windows x64, MSVC ABI). Kept as a permanent regression net.
#[cfg(test)]
mod jit_spike;

/// The panic codes passed across the FFI boundary to `helix_panic`.
///
/// Kept as a plain `i64` enum so the JIT side needs no type sharing: codes are
/// stable integers documented here and matched in [`engine::helix_panic`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PanicCode {
    /// Integer `/` or `%` with divisor 0.
    DivByZero = 0,
    /// `i64::MIN / -1` or `% -1` (and the i32 analogue).
    IdivOverflow = 1,
    /// Array index outside `0..len` (payload carries idx,len).
    Bounds = 2,
    /// `zeros(n)` with negative `n` (payload carries n).
    NegativeZeros = 3,
}

impl PanicCode {
    /// The fixed `<message>` text, byte-identical to the interpreter's wording
    /// (`helix_engine::error::RunErrorKind::message`).
    #[must_use]
    pub fn message(self) -> &'static str {
        match self {
            Self::DivByZero => "integer division by zero",
            Self::IdivOverflow => "integer division overflow (i64::MIN / -1)",
            Self::Bounds => "index out of bounds for array", // payload appends details
            Self::NegativeZeros => "array length must be non-negative",
        }
    }

    /// Numeric encoding sent through `helix_panic(i64)`.
    #[must_use]
    pub fn code(self) -> i64 {
        self as i64
    }
}

// Re-exports (contract surface, interface-contracts.md).
pub use engine::JitEngine;
pub use engine::{ParallelPlan, RegionDesc, RegionKind};

/// Test-support surface (integration tests + downstream selftests).
///
/// The engine's `print` builtins write to stdout normally; tests install a
/// thread-local capture around one `run_main` call instead. The panic hook is
/// injectable for the same reason: the default records the spec message and
/// exits(1), which a test process cannot survive.
pub mod testutil {
    /// Installs a print capture for the duration of `f`, returning
    /// `(lines, f's result)`. Lines are captured without newlines, in order.
    pub fn capture_prints<R>(f: impl FnOnce() -> R) -> (Vec<String>, R) {
        crate::engine::capture_prints(f)
    }

    /// Serializes whole-engine runs that install the panic hook or reset the
    /// recorded-error slot (process-global state).
    ///
    /// Returns a held guard: `let _g = serial_lock();` blocks until every
    /// other engine run in the process finishes. (Returning a bare `&Mutex`
    /// would acquire nothing — that bug let trap tests race and randomly
    /// killed the whole test binary via `helix_panic`'s disarmed-window exit.)
    pub fn serial_lock() -> std::sync::MutexGuard<'static, ()> {
        crate::engine::serial_lock()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    /// Arms the trap recorder: subsequent runtime traps are recorded instead
    /// of exiting the process (the JIT trap block resumes into its defined
    /// fallback return). Always pair with [`disarm_trap_recorder`] in a
    /// [`serial_lock`](Self::serial_lock) critical section.
    pub fn arm_trap_recorder() {
        crate::engine::arm_trap_recorder();
    }

    /// Disarms the trap recorder (restores exiting production behaviour).
    pub fn disarm_trap_recorder() {
        crate::engine::disarm_trap_recorder();
    }

    /// The recorded `(code, aux_a, aux_b)` of the last trap, if any.
    pub fn take_last_trap() -> Option<(i64, i64, i64)> {
        crate::engine::take_last_trap()
    }

    /// Panic codes as emitted by the lowering (frozen ABI constants).
    pub mod codes {
        /// Integer `/` or `%` with divisor 0.
        pub const DIV_BY_ZERO: i64 = 0;
        /// `MIN / -1` (or `% -1`) at i64 or i32 width.
        pub const DIV_OVERFLOW: i64 = 1;
        /// Array index outside `0..len` (payload carries idx, len).
        pub const BOUNDS: i64 = 2;
        /// `zeros(n)` with negative n (payload carries n).
        pub const NEG_ZEROS: i64 = 3;
    }
}

/// Frees all JIT-host allocations between program executions (see
/// [`engine::reset_host_heap`]). The bench harness calls this after every
/// timed run so repeated executions do not accumulate arrays.
pub use engine::reset_host_heap;
