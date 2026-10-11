//! The `helix_apply` request/response surface.
//!
//! One raw-memory entry dispatches a small JSON op table, so the browser lab
//! can reach the rest of the pipeline — the artifact builder, the
//! WebAssembly emitter, the canonical print formatters, the recorded
//! benchmark index — without growing the raw ABI one export at a time:
//!
//! | op | request | answer |
//! |---|---|---|
//! | `build` | `{"op":"build","source":"…","example":"name"}` | the [`helix_artifact::CompileArtifact`] JSON (compact) |
//! | `emit` | `{"op":"emit","source":"…","example":"name"}` | `{"ok":true,"module_b64":"…","engine":"helix-wasm-backend"}` |
//! | `format_line` | `{"op":"format_line","tag":"i32|i64|f32|f64|bool","bits":"<u64 decimal>"}` | `{"text":"…"}` |
//! | `bench_index` | `{"op":"bench_index"}` | the recorded campaign index |
//! | `interp_run` | `{"op":"interp_run","source":"…"}` | `{"ok":true,"printed":[…],"checksum":"0x…","alloc_bytes":N,"elapsed_ms":0}` |
//!
//! Every answer lands in the crate's shared result buffer and is read back
//! with `helix_output_ptr`/`helix_output_len`, exactly like `helix_run`. An
//! answer that describes a failure is still an answer: a rejected op returns 1
//! with an `{"ok":false,"message":"…"}` payload, and only a request the raw
//! layer cannot read at all (empty, invalid UTF-8, oversized) returns 0 from
//! [`crate::helix_apply`].

use serde::{Deserialize, Serialize};

use crate::{MAX_SOURCE_BYTES, ProgramRun, b64, run_source};

/// Engine marker for modules this bridge compiled itself.
const ENGINE: &str = "helix-wasm-backend";

/// Elapsed time this ABI reports: wasm32 has no monotonic clock, so the
/// browser worker measures wall-clock around the bridge call itself.
const ELAPSED_MS: u64 = 0;

/// The request envelope. Unknown fields are ignored so a later op can add
/// options without breaking an older bridge; a missing `op` fails the parse.
#[derive(Debug, Deserialize)]
struct ApplyRequest {
    op: String,
    #[serde(default)]
    source: Option<String>,
    #[serde(default)]
    example: Option<String>,
    #[serde(default)]
    tag: Option<String>,
    #[serde(default)]
    bits: Option<String>,
}

/// `{"ok":true,"module_b64":"…","engine":"helix-wasm-backend"}`
#[derive(Debug, Serialize)]
struct EmitAnswer {
    ok: bool,
    module_b64: String,
    engine: &'static str,
}

/// `{"text":"…"}` — one scalar through `helix_sema`'s canonical formatters.
#[derive(Debug, Serialize)]
struct FormattedLine {
    text: String,
}

/// `{"ok":true,"printed":[…],"checksum":"0x…","alloc_bytes":N,"elapsed_ms":0}`
#[derive(Debug, Serialize)]
struct InterpAnswer {
    ok: bool,
    printed: Vec<String>,
    checksum: String,
    alloc_bytes: u64,
    elapsed_ms: u64,
}

/// `{"ok":false,"message":"…","printed":[…]}` — the failing twin of
/// [`InterpAnswer`], with everything `print` produced before the failure.
#[derive(Debug, Serialize)]
struct InterpFailure {
    ok: bool,
    message: String,
    printed: Vec<String>,
}

/// `{"ok":false,"message":"…"}` — every failure that is not a program run.
#[derive(Debug, Serialize)]
struct Failure {
    ok: bool,
    message: String,
}

/// Parses `request` and answers its op.
///
/// Even an unparsable request is answered rather than rejected: the caller
/// stores the bytes and returns 1, so the bridge always leaves a readable
/// JSON trail.
pub(crate) fn dispatch(request: &str) -> Vec<u8> {
    let parsed: ApplyRequest = match serde_json::from_str(request) {
        Ok(parsed) => parsed,
        Err(error) => return error_answer(format!("malformed request: {error}")),
    };
    match parsed.op.as_str() {
        "build" => op_build(&parsed),
        "emit" => op_emit(&parsed),
        "format_line" => op_format_line(&parsed),
        "bench_index" => op_bench_index(),
        "interp_run" => op_interp_run(&parsed),
        other => error_answer(format!("unknown op '{other}'")),
    }
}

/// The `{"ok":false,"message":…}` answer every non-run failure uses.
pub(crate) fn error_answer(message: impl Into<String>) -> Vec<u8> {
    encode_answer(&Failure {
        ok: false,
        message: message.into(),
    })
}

/// `build`: the full artifact document for `source`.
///
/// Diagnostics travel inside the artifact — a program that does not compile
/// answers with the artifact that says so, which is the wire contract the
/// Observatory already speaks, so this op never fails the call.
///
/// Recorded evidence attaches through the same identity gate the Observatory
/// server uses (token-kind identity against the frozen catalog revision), so
/// a Pages-mode artifact carries the same provenance-stamped `bench` field as
/// the served one — the browser never invents campaign numbers, it reports
/// the recorded ones with their stamps.
fn op_build(request: &ApplyRequest) -> Vec<u8> {
    let source = match source_of(request) {
        Ok(source) => source,
        Err(message) => return error_answer(message),
    };
    encode_answer(&helix_artifact::artifact_for_example(
        example_of(request),
        source,
    ))
}

/// `emit`: the WebAssembly module for `source`, base64 so it survives JSON.
///
/// The lowering runs the ONE shared compile sequence
/// ([`helix_artifact::prepare_pipeline`]: parse → check → IR → SSA →
/// optimizer → loop analysis → plan) — the same post-SSA optimized IR the
/// native JIT and the Observatory consume — and emits a module from exactly
/// that IR, so the browser instantiates the module the parity tests ran.
fn op_emit(request: &ApplyRequest) -> Vec<u8> {
    let source = match source_of(request) {
        Ok(source) => source,
        Err(message) => return error_answer(message),
    };
    let units = match helix_artifact::prepare_pipeline(source) {
        Ok(units) => units,
        Err(message) => return error_answer(message),
    };
    match helix_wasm_backend::emit_wasm(&units.funcs) {
        Ok(module) => encode_answer(&EmitAnswer {
            ok: true,
            module_b64: b64::encode(&module.bytes),
            engine: ENGINE,
        }),
        Err(error) => error_answer(error.to_string()),
    }
}

/// `format_line`: one scalar as `print` would have rendered it.
///
/// `bits` travels as a decimal string because JSON numbers lose precision
/// above 2^53; every tag's payload fits the 64-bit word, zero-extended.
fn op_format_line(request: &ApplyRequest) -> Vec<u8> {
    let Some(tag) = request.tag.as_deref() else {
        return error_answer("request must carry a `tag` string");
    };
    let Some(bits) = request.bits.as_deref() else {
        return error_answer("request must carry a `bits` string");
    };
    let bits: u64 = match bits.trim().parse() {
        Ok(bits) => bits,
        Err(_) => return error_answer(format!("`bits` is not a u64 decimal value: '{bits}'")),
    };
    let text = match tag {
        // An i32 arrives zero-extended; read it back as a signed 32-bit value.
        "i32" => helix_sema::fmt_i64(i64::from(bits as i32)),
        "i64" => helix_sema::fmt_i64(bits as i64),
        // Rebuild at f32 width: widening the pattern first would print 0.1f32
        // as 0.10000000149011612.
        "f32" => helix_sema::fmt_f32(f32::from_bits(bits as u32)),
        "f64" => helix_sema::fmt_f64(f64::from_bits(bits)),
        "bool" => helix_sema::fmt_bool(bits != 0),
        other => return error_answer(format!("unknown format_line tag '{other}'")),
    };
    encode_answer(&FormattedLine { text })
}

/// `bench_index`: the recorded campaign index, the same document the
/// Observatory's `/api/benchmarks` route serves.
fn op_bench_index() -> Vec<u8> {
    encode_answer(&helix_artifact::bench_index())
}

/// `interp_run`: the reference interpreter alone, with `zeros` accounting.
///
/// Failures reuse the mapping [`crate::helix_run`] stores, so a runtime error
/// reads identically whichever entry the bridge was called through. The
/// allocation counter is read straight after a successful run because the
/// engine resets it on read.
fn op_interp_run(request: &ApplyRequest) -> Vec<u8> {
    let source = match source_of(request) {
        Ok(source) => source,
        Err(message) => return error_answer(message),
    };
    match run_source(source) {
        ProgramRun::Completed { printed, checksum } => encode_answer(&InterpAnswer {
            ok: true,
            printed,
            checksum: format!("0x{checksum:016x}"),
            alloc_bytes: helix_engine::take_alloc_bytes(),
            elapsed_ms: ELAPSED_MS,
        }),
        ProgramRun::Failed {
            message, printed, ..
        } => encode_answer(&InterpFailure {
            ok: false,
            message,
            printed,
        }),
    }
}

/// The program text an op runs on: present, and within the source cap the
/// bridge enforces exactly like `helix_run`.
fn source_of(request: &ApplyRequest) -> Result<&str, String> {
    let source = request
        .source
        .as_deref()
        .filter(|source| !source.is_empty())
        .ok_or_else(|| "request must carry a non-empty `source` string".to_string())?;
    if source.len() > MAX_SOURCE_BYTES {
        return Err(format!("source exceeds {MAX_SOURCE_BYTES} bytes"));
    }
    Ok(source)
}

/// The name an artifact is filed under; editor submissions pass none.
fn example_of(request: &ApplyRequest) -> &str {
    request.example.as_deref().unwrap_or("<adhoc>")
}

/// Compact JSON for any serializable answer, with a readable last-resort
/// payload in case one ever stops serializing.
fn encode_answer<T: Serialize>(answer: &T) -> Vec<u8> {
    serde_json::to_vec(answer)
        .unwrap_or_else(|_| br#"{"ok":false,"message":"could not serialize answer"}"#.to_vec())
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::*;

    /// A teaching program: an array, a loop, one print.
    const SQUARES: &str = "fn main() {\n    let n = 4;\n    let x: [i64] = zeros(n);\n    for i in 0..n {\n        x[i] = i * i;\n    }\n    print(x[3]);\n}\n";

    /// The smallest program the interpreter runs.
    const HELLO: &str = "fn main() {\n    print(21 * 2);\n}\n";

    /// A program whose array write traps at runtime.
    const OUT_OF_BOUNDS: &str = "fn main() {\n    let a: [i64] = zeros(2);\n    a[5] = 1;\n}\n";

    /// A program the checker rejects — a bool is not an i64.
    const TYPE_ERROR: &str = "fn main() {\n    let x: i64 = true;\n}\n";

    /// Dispatches `request` and parses the stored answer, the same path the
    /// `helix_apply` entry walks.
    fn answer(request: Value) -> Value {
        let bytes = dispatch(&request.to_string());
        serde_json::from_slice(&bytes).expect("answer is JSON")
    }

    /// The module bytes of a successful `emit`, decoded from base64.
    fn emitted_module(source: &str) -> Vec<u8> {
        let answer = answer(json!({"op": "emit", "source": source}));
        assert_eq!(answer["ok"], true, "emit must succeed for {source:?}");
        let module_b64 = answer["module_b64"].as_str().expect("module_b64");
        b64::decode(module_b64)
    }

    #[test]
    fn build_answers_the_full_artifact_document() {
        let answer = answer(json!({"op": "build", "source": HELLO, "example": "unit"}));
        assert_eq!(answer["schema"], 1);
        assert_eq!(answer["example"], "unit");
        assert_eq!(answer["source"], HELLO);
        assert_eq!(answer["diags_sem"], json!([]));
        assert_eq!(answer["exec"]["printed"], json!(["42"]));
    }

    #[test]
    fn build_carries_diagnostics_instead_of_failing() {
        let answer = answer(json!({"op": "build", "source": TYPE_ERROR, "example": "unit"}));
        assert!(
            answer["diags_sem"]
                .as_array()
                .is_some_and(|diags| !diags.is_empty()),
            "a type error must travel inside the artifact"
        );
        assert!(answer["exec"].is_null(), "a rejected program never runs");
    }

    /// Pages mode has no server, so the artifact must carry the recorded
    /// evidence itself: a named example whose source still matches the frozen
    /// catalog gets the same provenance-stamped `bench` field the Observatory
    /// serves, and an unrelated program gets none.
    ///
    /// The probe catalog (not the campaign) is what the simple example files
    /// match: the campaign kernels carry seed loops the teaching examples do
    /// not, and the identity gate correctly refuses that mismatch.
    #[test]
    fn build_attaches_recorded_bench_evidence_for_named_examples() {
        let source = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../examples/casts_demo.hx"
        ))
        .expect("examples/casts_demo.hx is readable");
        let built = answer(json!({"op": "build", "source": source, "example": "casts_demo"}));
        assert!(built["diags_sem"].as_array().is_some_and(Vec::is_empty));
        assert_eq!(
            built["bench"]["kernel"], "casts_demo",
            "recorded probe must attach"
        );
        assert_eq!(built["bench"]["variants"][0]["name"], "interp-local");
        // An unknown program must not inherit somebody else's evidence.
        let unrelated = answer(json!({"op": "build", "source": HELLO, "example": "casts_demo"}));
        assert!(
            unrelated["bench"].is_null(),
            "a source that is not the recorded one must not attach evidence"
        );
    }

    #[test]
    fn emit_answers_a_structurally_valid_module() {
        let answer = answer(json!({"op": "emit", "source": HELLO, "example": "unit"}));
        assert_eq!(answer["ok"], true);
        assert_eq!(answer["engine"], ENGINE);

        let module = emitted_module(HELLO);
        assert_eq!(&module[..4], b"\0asm", "wasm magic");
        assert_eq!(&module[4..8], &1u32.to_le_bytes(), "wasm version 1");
        assert!(module.len() > 8, "a module has sections after the header");
    }

    #[test]
    fn emit_answers_the_syntax_error_message() {
        let answer = answer(json!({"op": "emit", "source": "fn main( {\n"}));
        assert_eq!(answer["ok"], false);
        assert!(
            answer["message"].as_str().is_some_and(|m| !m.is_empty()),
            "the failure must quote the diagnostic"
        );
    }

    #[test]
    fn emit_reports_a_missing_main() {
        let answer = answer(
            json!({"op": "emit", "source": "fn helper(n: i64) -> i64 {\n    return n;\n}\n"}),
        );
        assert_eq!(answer["ok"], false);
        assert!(
            answer["message"]
                .as_str()
                .is_some_and(|m| m.contains("main")),
            "the emitter's message must name the missing main"
        );
    }

    #[test]
    fn format_line_renders_every_tag_through_sema() {
        let line = |tag: &str, bits: u64| -> String {
            answer(json!({"op": "format_line", "tag": tag, "bits": bits.to_string()}))["text"]
                .as_str()
                .expect("text is a string")
                .to_string()
        };
        // An i32 arrives zero-extended and is read back as a signed 32-bit value.
        assert_eq!(line("i32", u64::from(21u32)), "21");
        assert_eq!(line("i32", u64::from(0xffff_ffffu32)), "-1");
        assert_eq!(line("i64", 42), "42");
        assert_eq!(line("i64", i64::MIN as u64), "-9223372036854775808");
        // f32 is rebuilt at f32 width: widening the pattern first would print
        // 0.10000000149011612.
        assert_eq!(line("f32", u64::from(0.1f32.to_bits())), "0.1");
        assert_eq!(line("f64", 0.1f64.to_bits()), "0.1");
        assert_eq!(line("bool", 1), "true");
        assert_eq!(line("bool", 0), "false");
    }

    #[test]
    fn format_line_rejects_unknown_tags_and_bits() {
        let unknown_tag = answer(json!({"op": "format_line", "tag": "i128", "bits": "1"}));
        assert_eq!(unknown_tag["ok"], false);
        assert!(
            unknown_tag["message"]
                .as_str()
                .is_some_and(|m| m.contains("i128"))
        );

        let bad_bits = answer(json!({"op": "format_line", "tag": "i64", "bits": "not-a-number"}));
        assert_eq!(bad_bits["ok"], false);
        assert!(
            bad_bits["message"]
                .as_str()
                .is_some_and(|m| m.contains("not-a-number"))
        );
    }

    #[test]
    fn interp_run_reports_output_checksum_and_allocations() {
        let answer = answer(json!({"op": "interp_run", "source": SQUARES}));
        assert_eq!(answer["ok"], true);
        assert_eq!(answer["printed"], json!(["9"]));
        // The engine counts per thread and interprets on a worker thread on
        // native hosts, so the count is only observable in the single-threaded
        // wasm build this bridge ships; the field must exist either way.
        assert!(
            answer["alloc_bytes"].is_u64(),
            "alloc_bytes must be a count"
        );
        assert_eq!(answer["elapsed_ms"], 0);
        let checksum = answer["checksum"].as_str().expect("checksum string");
        assert!(
            checksum.starts_with("0x") && checksum.len() == 18,
            "checksum is 0x plus sixteen hex digits, got {checksum}"
        );
    }

    #[test]
    fn interp_run_reports_runtime_failures_like_helix_run() {
        let answer = answer(json!({"op": "interp_run", "source": OUT_OF_BOUNDS}));
        assert_eq!(answer["ok"], false);
        assert_eq!(answer["printed"], json!([]));
        assert_eq!(
            answer["message"],
            "index 5 out of bounds for array of length 2 (line 3)"
        );
    }

    #[test]
    fn interp_run_reports_front_end_failures() {
        let answer = answer(json!({"op": "interp_run", "source": "fn main( {\n"}));
        assert_eq!(answer["ok"], false);
        assert!(
            answer["message"]
                .as_str()
                .is_some_and(|m| m.contains("syntax error")),
            "a syntax error quotes the front end's own message"
        );
    }

    #[test]
    fn interp_run_keeps_lines_printed_before_a_failure() {
        let answer = answer(
            json!({"op": "interp_run", "source": "fn main() {\n    print(1);\n    let a: [i64] = zeros(1);\n    a[3] = 1;\n}\n"}),
        );
        assert_eq!(answer["ok"], false);
        assert_eq!(answer["printed"], json!(["1"]));
    }

    #[test]
    fn bench_index_answers_the_recorded_campaign() {
        let answer = answer(json!({"op": "bench_index"}));
        assert_eq!(answer["schema"], "helix-campaign-v1");
        assert!(answer["point_count"].as_u64().is_some_and(|n| n > 0));
        assert!(
            answer["kernels"]
                .as_array()
                .is_some_and(|kernels| kernels.iter().any(|k| k["name"] == "saxpy")),
            "the index lists the measured kernels"
        );
    }

    #[test]
    fn unknown_op_answers_a_failure() {
        let answer = answer(json!({"op": "explode"}));
        assert_eq!(answer["ok"], false);
        assert!(
            answer["message"]
                .as_str()
                .is_some_and(|m| m.contains("explode"))
        );
    }

    #[test]
    fn missing_op_is_a_malformed_request() {
        let answer = answer(json!({"source": HELLO}));
        assert_eq!(answer["ok"], false);
        assert!(
            answer["message"]
                .as_str()
                .is_some_and(|m| m.contains("malformed request"))
        );
    }

    #[test]
    fn missing_or_oversized_source_answers_a_failure() {
        let no_source = answer(json!({"op": "interp_run"}));
        assert_eq!(no_source["ok"], false);
        assert!(
            no_source["message"]
                .as_str()
                .is_some_and(|m| m.contains("source")),
            "the answer must say what the request lacked"
        );

        let too_big =
            answer(json!({"op": "interp_run", "source": "x".repeat(MAX_SOURCE_BYTES + 1)}));
        assert_eq!(too_big["ok"], false);
        assert!(
            too_big["message"]
                .as_str()
                .is_some_and(|m| m.contains("exceeds"))
        );
    }
}
