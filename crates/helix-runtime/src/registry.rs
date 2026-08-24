//! Body / combine function registries.
//!
//! The M10 backend lowers each approved loop body to a Cranelift helper with
//! the `extern "C"` signature `fn(i64 iter, *mut u8 ctx)`, finalizes the JIT
//! module, and then registers the resulting raw pointer here under a stable
//! integer id. JITed code calls [`crate::helix_parallel_for`] with that id;
//! the runtime resolves it back to a callable pointer per region.
//!
//! Pointers are captured only AFTER `JitEngine::finalize_definitions`
//! (contract: "host registry maps body_id→ptr AFTER finalize, never embed
//! unknown pointers"), so every stored pointer is valid for the lifetime of
//! its [`helix_backend::JitEngine`]-equivalent module — i.e. the whole run.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use crate::pool::BodyFn;

/// Combines two partial reduction values (`dst = combine(dst, src)`), both
/// byte buffers of the reduction's element type. Registered alongside bodies
/// by the backend when lowering a reduction region.
pub type CombineFn = extern "C" fn(dst: *mut u8, src: *const u8);

/// One global registry slot: id -> function pointer.
type Map<T> = Mutex<HashMap<i64, T>>;

fn bodies() -> &'static Map<BodyFn> {
    static BODIES: OnceLock<Map<BodyFn>> = OnceLock::new();
    BODIES.get_or_init(|| Mutex::new(HashMap::new()))
}

fn combines() -> &'static Map<CombineFn> {
    static COMBINES: OnceLock<Map<CombineFn>> = OnceLock::new();
    COMBINES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Registers (or replaces) the body for `id`.
pub fn register_body(id: i64, f: BodyFn) {
    // Lock poisoning is impossible in practice (no user closures panic while
    // holding it); recover anyway so the runtime can never panic on the FFI
    // path.
    let mut map = match bodies().lock() {
        Ok(m) => m,
        Err(e) => e.into_inner(),
    };
    map.insert(id, f);
}

/// Registers (or replaces) the combine fn for `id`.
pub fn register_combine(id: i64, f: CombineFn) {
    let mut map = match combines().lock() {
        Ok(m) => m,
        Err(e) => e.into_inner(),
    };
    map.insert(id, f);
}

/// Resolves a registered body; `None` for unknown ids.
///
/// Unknown ids mean "compiler/runtime out of sync" — a bug in HELIX itself,
/// not user input — so callers surface it as a clean error, never a panic.
pub fn lookup_body(id: i64) -> Option<BodyFn> {
    let map = match bodies().lock() {
        Ok(m) => m,
        Err(e) => e.into_inner(),
    };
    map.get(&id).copied()
}

/// Resolves a registered combine fn; `None` for unknown ids.
pub fn lookup_combine(id: i64) -> Option<CombineFn> {
    let map = match combines().lock() {
        Ok(m) => m,
        Err(e) => e.into_inner(),
    };
    map.get(&id).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    extern "C" fn add_one(iter: i64, ctx: *mut u8) {
        // SAFETY: test-owned i64 accumulator passed as ctx.
        unsafe {
            *(ctx as *mut i64) += iter;
        }
    }

    extern "C" fn add_i64(dst: *mut u8, src: *const u8) {
        // SAFETY: test-owned i64 buffers, both valid.
        unsafe {
            *(dst as *mut i64) += *(src as *const i64);
        }
    }

    #[test]
    fn register_and_lookup_roundtrip() {
        const ID: i64 = 424_242;
        // Compare addresses (fn-ptr equality is not meaningful across CGUs).
        let addr = |f: Option<BodyFn>| f.map(|g| g as *const u8).unwrap_or(std::ptr::null());
        let caddr = |f: Option<CombineFn>| f.map(|g| g as *const u8).unwrap_or(std::ptr::null());
        register_body(ID, add_one);
        assert_eq!(addr(lookup_body(ID)), addr(Some(add_one)));
        assert!(lookup_combine(ID).is_none());

        register_combine(ID, add_i64);
        assert_eq!(caddr(lookup_combine(ID)), caddr(Some(add_i64)));

        // Re-registration replaces (stable id semantics).
        register_body(ID, add_one);
        assert_eq!(addr(lookup_body(ID)), addr(Some(add_one)));
    }

    #[test]
    fn unknown_ids_are_none_not_panic() {
        assert!(lookup_body(-1_000_000).is_none());
        assert!(lookup_combine(i64::MAX).is_none());
    }

    #[test]
    fn registered_fns_are_callable_through_the_pointer() {
        register_body(777_777, add_one);
        let f = lookup_body(777_777).expect("just registered");
        let mut acc: i64 = 0;
        f(5, (&mut acc as *mut i64).cast());
        assert_eq!(acc, 5);
        register_combine(777_777, add_i64);
        let g = lookup_combine(777_777).expect("just registered");
        let (mut dst, src): (i64, i64) = (40, 2);
        g((&mut dst as *mut i64).cast(), (&src as *const i64).cast());
        assert_eq!(dst, 42);
    }
}
