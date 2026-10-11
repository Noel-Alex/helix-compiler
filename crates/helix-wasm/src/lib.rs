//! A deliberately small raw-WASM bridge for the HELIX reference interpreter.
//!
//! The browser lab uses this adapter for short, educational programs. It
//! shares the real syntax, semantic checker, and tree-walking interpreter;
//! it does not claim to ship the native Cranelift/JIT backend or parallel
//! execution. The ABI is intentionally tiny so the shipped asset needs no
//! generated binding runtime: JavaScript writes UTF-8 into linear memory,
//! calls an entry, then reads the JSON result back.
//!
//! * [`helix_run`] — run one program, store its result.
//! * [`helix_apply`] — one JSON entry for the rest of the pipeline: the
//!   artifact builder, the WebAssembly emitter, the canonical print
//!   formatters, and a bare interpreter run (see [`apply`]).
//!
//! Both entries answer through the same shared result buffer, read with
//! [`helix_output_ptr`]/[`helix_output_len`].

use std::cell::RefCell;

use serde::Serialize;

mod apply;
mod b64;

const MAX_SOURCE_BYTES: usize = 64 * 1024;

/// Raw request cap for [`helix_apply`]: a maximal legal `source` plus the JSON
/// envelope around it. A blob beyond that is not a request the bridge answers,
/// and is refused the same defensive way `helix_run` refuses oversized input.
const MAX_REQUEST_BYTES: usize = MAX_SOURCE_BYTES + 4096;

/// Engine label every browser result names itself with.
const ENGINE: &str = "HELIX reference interpreter / wasm32";

thread_local! {
    static LAST_RESULT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

#[derive(Serialize)]
struct BrowserResult {
    ok: bool,
    status: &'static str,
    printed: Vec<String>,
    checksum: Option<String>,
    message: Option<String>,
    span: Option<SpanResult>,
    engine: &'static str,
}

#[derive(Serialize)]
struct SpanResult {
    start: u32,
    end: u32,
    line: u32,
}

/// What one program run produced: the interpreter's output, or the first
/// failure with everything `print` had already emitted.
pub(crate) enum ProgramRun {
    Completed {
        printed: Vec<String>,
        checksum: u64,
    },
    Failed {
        status: &'static str,
        message: String,
        printed: Vec<String>,
        span: Option<SpanResult>,
    },
}

/// Stores `result` as the JSON the JS bridge reads back.
fn store(result: BrowserResult) {
    let bytes = serde_json::to_vec(&result).unwrap_or_else(|_| {
        br#"{"ok":false,"status":"error","printed":[],"checksum":null,"message":"could not serialize browser result","span":null,"engine":"HELIX reference interpreter / wasm32"}"#.to_vec()
    });
    store_bytes(bytes);
}

/// Stores an already-encoded answer (the `helix_apply` ops).
pub(crate) fn store_bytes(bytes: Vec<u8>) {
    LAST_RESULT.with(|slot| *slot.borrow_mut() = bytes);
}

/// Builds the browser-shaped error for a request `helix_run` refuses before
/// the pipeline runs: empty, oversized, or not valid UTF-8.
fn error_result(message: String, status: &'static str, span: Option<SpanResult>) -> BrowserResult {
    BrowserResult {
        ok: false,
        status,
        printed: Vec::new(),
        checksum: None,
        message: Some(message),
        span,
        engine: ENGINE,
    }
}

/// Allocates a UTF-8 input buffer in wasm linear memory.
#[unsafe(no_mangle)]
pub extern "C" fn helix_alloc(len: usize) -> *mut u8 {
    if len == 0 {
        return std::ptr::null_mut();
    }
    let mut bytes = vec![0u8; len].into_boxed_slice();
    let ptr = bytes.as_mut_ptr();
    std::mem::forget(bytes);
    ptr
}

/// Releases a buffer previously returned by [`helix_alloc`].
#[unsafe(no_mangle)]
pub extern "C" fn helix_free(ptr: *mut u8, len: usize) {
    if ptr.is_null() || len == 0 {
        return;
    }
    // The pointer/length pair is produced by helix_alloc and never retained
    // after this call. Reconstructing the boxed slice returns it to Rust's
    // allocator without exposing a Rust reference across the ABI.
    unsafe {
        drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len)));
    }
}

/// Runs one bounded HELIX program and stores a JSON result for the JS bridge.
///
/// # Safety
/// `ptr` must point to `len` readable bytes of UTF-8 source for the duration
/// of the call — the JS bridge passes pointers returned by `helix_alloc`.
///
/// Returns `1` when the program parsed, passed semantic checking, and ran to
/// completion; returns `0` for a lexical, syntax, semantic, or runtime error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn helix_run(ptr: *const u8, len: usize) -> i32 {
    // Null/empty still produces a JSON error so the JS bridge always has a
    // readable result even when the safety contract is violated.
    if ptr.is_null() || len == 0 {
        store(error_result(
            "source must not be empty".to_string(),
            "error",
            None,
        ));
        return 0;
    }
    if len > MAX_SOURCE_BYTES {
        store(error_result(
            format!("source exceeds {MAX_SOURCE_BYTES} bytes"),
            "error",
            None,
        ));
        return 0;
    }

    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    let source = match std::str::from_utf8(bytes) {
        Ok(source) => source,
        Err(_) => {
            store(error_result(
                "source is not valid UTF-8".to_string(),
                "error",
                None,
            ));
            return 0;
        }
    };

    match run_source(source) {
        ProgramRun::Completed { printed, checksum } => {
            store(BrowserResult {
                ok: true,
                status: "ok",
                printed,
                checksum: Some(format!("0x{checksum:016x}")),
                message: None,
                span: None,
                engine: ENGINE,
            });
            1
        }
        ProgramRun::Failed {
            status,
            message,
            printed,
            span,
        } => {
            store(BrowserResult {
                ok: false,
                status,
                printed,
                checksum: None,
                message: Some(message),
                span,
                engine: ENGINE,
            });
            0
        }
    }
}

/// Entry: `helix_apply(ptr, len) -> i32`.
///
/// Input JSON: `{"op":"build"|"emit"|"format_line"|"bench_index"|"interp_run",
/// …}` — [`apply`] holds the op table. Output: JSON stored in the shared
/// result buffer, read via [`helix_output_ptr`]/[`helix_output_len`].
///
/// Returns `1` on a well-formed answer (including answers that carry an
/// embedded `"ok":false` payload, which is how a rejected request still
/// reaches the caller), `0` for an unreadable request (empty, invalid UTF-8,
/// or oversized — the same defensive style as [`helix_run`]).
///
/// # Safety
/// `ptr` must point to `len` readable bytes for the duration of the call —
/// the JS bridge passes pointers returned by [`helix_alloc`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn helix_apply(ptr: *const u8, len: usize) -> i32 {
    if ptr.is_null() || len == 0 {
        store_bytes(apply::error_answer("request must not be empty"));
        return 0;
    }
    if len > MAX_REQUEST_BYTES {
        store_bytes(apply::error_answer(format!(
            "request exceeds {MAX_REQUEST_BYTES} bytes"
        )));
        return 0;
    }

    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    let request = match std::str::from_utf8(bytes) {
        Ok(request) => request,
        Err(_) => {
            store_bytes(apply::error_answer("request is not valid UTF-8"));
            return 0;
        }
    };

    store_bytes(apply::dispatch(request));
    1
}

/// Returns a pointer to the most recent JSON result.
#[unsafe(no_mangle)]
pub extern "C" fn helix_output_ptr() -> *const u8 {
    LAST_RESULT.with(|slot| slot.borrow().as_ptr())
}

/// Returns the byte length of the most recent JSON result.
#[unsafe(no_mangle)]
pub extern "C" fn helix_output_len() -> usize {
    LAST_RESULT.with(|slot| slot.borrow().len())
}

/// Runs one source through the real pipeline, mapping every stage failure
/// onto the browser result shape.
///
/// Shared by [`helix_run`] and the `interp_run` op, so a program's output, its
/// error message, and its partial prints are identical whichever entry the
/// bridge was called through.
pub(crate) fn run_source(source: &str) -> ProgramRun {
    let program = match helix_syntax::parse_str(source) {
        Ok(program) => program,
        Err(error) => {
            let span = match &error {
                helix_syntax::SyntaxError::Lex(error) => error.span,
                helix_syntax::SyntaxError::Parse(error) => error.span,
            };
            return ProgramRun::Failed {
                status: "syntax_error",
                message: error.to_string(),
                printed: Vec::new(),
                span: Some(SpanResult {
                    start: span.start,
                    end: span.end,
                    line: line_of(source, span.start),
                }),
            };
        }
    };

    let typed = match helix_sema::check(&program) {
        Ok(typed) => typed,
        Err(diags) => {
            let first = diags.first();
            return ProgramRun::Failed {
                status: "semantic_error",
                message: sema_message(source, &diags),
                printed: Vec::new(),
                span: first.map(|diag| SpanResult {
                    start: diag.span.start,
                    end: diag.span.end,
                    line: line_of(source, diag.span.start),
                }),
            };
        }
    };

    match helix_engine::run_with_source(source, &typed) {
        Ok(output) => ProgramRun::Completed {
            printed: output.printed,
            checksum: output.checksum,
        },
        Err(error) => {
            let line = line_of(source, error.span.start);
            ProgramRun::Failed {
                status: "runtime_error",
                message: format!("{} (line {line})", error.kind.message()),
                printed: error.printed_so_far,
                span: Some(SpanResult {
                    start: error.span.start,
                    end: error.span.end,
                    line,
                }),
            }
        }
    }
}

/// Every semantic diagnostic, one per line with its source line number.
pub(crate) fn sema_message(source: &str, diags: &[helix_sema::SemDiag]) -> String {
    diags
        .iter()
        .map(|diag| format!("line {}: {}", line_of(source, diag.span.start), diag.msg))
        .collect::<Vec<_>>()
        .join("\n")
}

fn line_of(source: &str, byte_offset: u32) -> u32 {
    let end = (byte_offset as usize).min(source.len());
    1 + source[..end].bytes().filter(|b| *b == b'\n').count() as u32
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::*;

    /// The smallest program the interpreter can run.
    const HELLO: &str = "fn main() {\n    print(21 * 2);\n}\n";

    /// Writes `source` into a bridge buffer exactly like the worker does.
    fn input(source: &str) -> (*mut u8, usize) {
        let ptr = helix_alloc(source.len());
        assert!(!ptr.is_null(), "bridge must allocate the input buffer");
        unsafe { std::ptr::copy_nonoverlapping(source.as_ptr(), ptr, source.len()) };
        (ptr, source.len())
    }

    /// Reads the shared result buffer the way `helix_output_*` exposes it.
    fn output() -> Vec<u8> {
        let ptr = helix_output_ptr();
        let len = helix_output_len();
        assert!(!ptr.is_null() && len > 0, "an answer must be stored");
        unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
    }

    fn answer_json(bytes: &[u8]) -> Value {
        serde_json::from_slice(bytes).expect("answer is JSON")
    }

    #[test]
    fn helix_run_keeps_the_worker_contract() {
        let (ptr, len) = input(HELLO);
        let ok = unsafe { helix_run(ptr, len) };
        let stored = output();
        helix_free(ptr, len);
        assert_eq!(ok, 1);
        let answer = answer_json(&stored);
        assert_eq!(answer["ok"], true);
        assert_eq!(answer["status"], "ok");
        assert_eq!(answer["printed"], serde_json::json!(["42"]));
        assert_eq!(answer["engine"], ENGINE);
    }

    #[test]
    fn helix_apply_round_trips_a_request_through_the_buffer() {
        let request = serde_json::json!({"op": "interp_run", "source": HELLO});
        let (ptr, len) = input(&request.to_string());
        let ok = unsafe { helix_apply(ptr, len) };
        let stored = output();
        helix_free(ptr, len);
        assert_eq!(ok, 1);
        assert_eq!(answer_json(&stored)["printed"], serde_json::json!(["42"]));
    }

    #[test]
    fn helix_run_reports_empty_input_defensively() {
        let ok = unsafe { helix_run(std::ptr::null(), 0) };
        let answer = answer_json(&output());
        assert_eq!(ok, 0);
        assert_eq!(answer["ok"], false);
        assert_eq!(answer["message"], "source must not be empty");
    }
}
