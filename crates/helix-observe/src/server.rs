//! The Observatory HTTP server (axum).
//!
//! Serves the embedded web UI plus the JSON API the UI already speaks:
//!
//! | Route                              | Purpose                                    |
//! |------------------------------------|--------------------------------------------|
//! | `GET /`                            | `index.html` (embedded)                    |
//! | `GET /about`                       | compiler guide (embedded)                 |
//! | `GET /app.js`, `/style.css`        | UI assets (embedded, correct MIME)         |
//! | `GET /about.js`, `/wasm/*`         | browser lab assets (embedded, worker included) |
//! | `GET /assets/*`                     | guide illustration (embedded)              |
//! | `GET /vendor/d3.v7.min.js`         | vendored d3 (embedded)                     |
//! | `GET /api/examples`                | example names from the repo's `examples/`  |
//! | `GET /api/artifact?example=name`   | full [`CompileArtifact`] for one example   |
//! | `GET /api/benchmarks`              | recorded campaign index                    |
//! | `POST /api/run {source}`           | artifact for editor-submitted source       |
//! | `POST /api/benchmark {source}`     | bounded custom interpreter measurement     |
//! | `GET /api/dot?example=…&fn=…`      | Graphviz `.dot` of one function            |
//! | `GET /api/artifact/export?…`       | standalone offline HTML with inline data   |
//!
//! Assets are compiled in with `include_bytes!`, so the server binary is
//! self-contained — the repo checkout is only needed to read example files.
//! The compile pipeline itself does not live here: `helix-artifact` owns it so
//! the same builder runs natively and under wasm32.
//!
//! ## Standalone export design
//!
//! The export route returns a small wrapper page that embeds the artifact and
//! installs a `fetch` shim *before* loading the real `app.js`: every
//! `/api/...` request is answered from the inlined object instead of the
//! network. Nothing inside `web/` changes, so the shipped UI and the offline
//! report can never drift apart.

use std::net::SocketAddr;
use std::path::PathBuf;

use axum::Json as AxJson;
use axum::extract::{Path as AxPath, Query, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use serde::Deserialize;
use serde_json::json;

use helix_artifact::{
    CompileArtifact, MAX_SOURCE_BYTES, artifact_for_example, bench_index, build_artifact,
    measure_interpreter,
};

/// Server settings.
#[derive(Debug, Clone)]
pub struct ServeConfig {
    /// Address to bind (default `127.0.0.1:8931`).
    pub addr: SocketAddr,
    /// Open the default browser after binding.
    pub open_browser: bool,
    /// Directory holding `<name>.hx` examples; falls back to a built-in list.
    pub examples_dir: Option<PathBuf>,
}

impl ServeConfig {
    /// Config bound to `addr`, browser auto-open on.
    #[must_use]
    pub fn new(addr: SocketAddr) -> Self {
        Self {
            addr,
            open_browser: true,
            examples_dir: None,
        }
    }
}

/// Shared handler state.
#[derive(Debug, Clone)]
struct AppState {
    examples_dir: Option<PathBuf>,
}

/// Builds the Observatory router (public so tests and `helix-cli` can mount
/// or probe it without spawning a listener).
pub fn router(examples_dir: Option<PathBuf>) -> axum::Router {
    axum::Router::new()
        .route("/", get(index_html))
        .route("/index.html", get(index_html))
        .route("/about", get(about_html))
        .route("/about.html", get(about_html))
        .route("/app.js", get(app_js))
        .route("/style.css", get(style_css))
        .route("/about.css", get(about_css))
        .route("/about.js", get(about_js))
        .route("/fonts/silkscreen-400.ttf", get(font_400))
        .route("/fonts/silkscreen-700.ttf", get(font_700))
        .route("/wasm/helix-browser.js", get(wasm_js))
        .route("/wasm/helix-worker.js", get(wasm_worker_js))
        .route("/wasm/helix-lab.wasm", get(wasm_bin))
        .route(
            "/assets/helix-loop-observatory.png",
            get(helix_loop_observatory),
        )
        .route("/vendor/d3.v7.min.js", get(d3_js))
        .route("/api/examples", get(api_examples))
        .route("/api/artifact", get(api_artifact))
        .route("/api/benchmarks", get(api_benchmarks))
        .route("/api/artifact/export", get(api_export))
        .route("/api/run", post(api_run))
        .route("/api/benchmark", post(api_benchmark))
        .route("/api/dot/{example}", get(api_dot))
        .fallback(not_found)
        .with_state(AppState { examples_dir })
}

/// Binds, serves until Ctrl-C, optionally opening the browser.
///
/// # Errors
/// Propagates bind/serve failures (port in use, …).
pub async fn serve(cfg: ServeConfig) -> Result<(), String> {
    let url = format!("http://{}", cfg.addr);
    let listener = tokio::net::TcpListener::bind(cfg.addr)
        .await
        .map_err(|e| format!("cannot bind {url}: {e}"))?;
    eprintln!("HELIX Observatory serving at {url}  (Ctrl-C to stop)");

    if cfg.open_browser {
        // Best effort: failure to open a tab must not kill the server.
        let _ = webbrowser::open(&url);
    }

    axum::serve(listener, router(cfg.examples_dir))
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|e| format!("server error: {e}"))
}

/// Resolves when Ctrl-C arrives.
async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}

// ---------------------------------------------------------------------------
// Static assets (embedded)
// ---------------------------------------------------------------------------

/// `web/index.html`.
const INDEX_HTML: &[u8] = include_bytes!("../../../web/index.html");
/// `web/about.html`.
const ABOUT_HTML: &[u8] = include_bytes!("../../../web/about.html");
/// `web/app.js`.
const APP_JS: &[u8] = include_bytes!("../../../web/app.js");
/// `web/style.css`.
const STYLE_CSS: &[u8] = include_bytes!("../../../web/style.css");
/// `web/about.css`.
const ABOUT_CSS: &[u8] = include_bytes!("../../../web/about.css");
/// `web/about.js`.
const ABOUT_JS: &[u8] = include_bytes!("../../../web/about.js");
/// Bundled bitmap UI fonts; keeping these on the observer makes the console
/// genuinely usable when the browser has no network access.
const FONT_400: &[u8] = include_bytes!("../../../web/fonts/silkscreen-400.ttf");
const FONT_700: &[u8] = include_bytes!("../../../web/fonts/silkscreen-700.ttf");
/// Browser lab assets: the real syntax/sema/reference-interpreter path
/// compiled for wasm32, plus its deliberately tiny raw-memory loader.
const WASM_JS: &[u8] = include_bytes!("../../../web/wasm/helix-browser.js");
/// Web Worker body for the browser lab; isolated per call with a deadline.
const WASM_WORKER_JS: &[u8] = include_bytes!("../../../web/wasm/helix-worker.js");
const WASM_BIN: &[u8] = include_bytes!("../../../web/wasm/helix-lab.wasm");
/// Original guide illustration; embedded so the college submission remains
/// readable and self-contained when served from the Rust observer.
const HELIX_LOOP_OBSERVATORY: &[u8] =
    include_bytes!("../../../web/assets/helix-loop-observatory.png");
/// Vendored d3 v7.
const D3_JS: &[u8] = include_bytes!("../../../web/vendor/d3.v7.min.js");

async fn index_html() -> impl IntoResponse {
    bytes_response(INDEX_HTML, "text/html; charset=utf-8")
}

async fn about_html() -> impl IntoResponse {
    bytes_response(ABOUT_HTML, "text/html; charset=utf-8")
}

async fn app_js() -> impl IntoResponse {
    bytes_response(APP_JS, "text/javascript; charset=utf-8")
}

async fn style_css() -> impl IntoResponse {
    bytes_response(STYLE_CSS, "text/css; charset=utf-8")
}

async fn about_css() -> impl IntoResponse {
    bytes_response(ABOUT_CSS, "text/css; charset=utf-8")
}

async fn about_js() -> impl IntoResponse {
    bytes_response(ABOUT_JS, "text/javascript; charset=utf-8")
}

async fn font_400() -> impl IntoResponse {
    bytes_response(FONT_400, "font/ttf")
}

async fn font_700() -> impl IntoResponse {
    bytes_response(FONT_700, "font/ttf")
}

async fn wasm_js() -> impl IntoResponse {
    bytes_response(WASM_JS, "text/javascript; charset=utf-8")
}

async fn wasm_worker_js() -> impl IntoResponse {
    bytes_response(WASM_WORKER_JS, "text/javascript; charset=utf-8")
}

async fn wasm_bin() -> impl IntoResponse {
    bytes_response(WASM_BIN, "application/wasm")
}

async fn helix_loop_observatory() -> impl IntoResponse {
    bytes_response(HELIX_LOOP_OBSERVATORY, "image/png")
}

async fn d3_js() -> impl IntoResponse {
    bytes_response(D3_JS, "text/javascript; charset=utf-8")
}

/// Byte payload + content type + no-store (artifacts change per compile).
fn bytes_response(bytes: &'static [u8], content_type: &str) -> Response {
    let mut resp = ([(header::CONTENT_TYPE, content_type.to_string())], bytes).into_response();
    resp.headers_mut().insert(
        header::CACHE_CONTROL,
        header::HeaderValue::from_static("no-store"),
    );
    resp
}

// ---------------------------------------------------------------------------
// API handlers
// ---------------------------------------------------------------------------

/// Example names: live directory listing when available, built-in fallback
/// otherwise (release binaries run far from a checkout).
async fn api_examples(State(state): State<AppState>) -> Response {
    let names = match &state.examples_dir {
        Some(dir) => list_examples(dir),
        None => builtin_examples(),
    };
    AxJson(names).into_response()
}

/// Compact index of the committed measured campaign. The shaping lives in
/// `helix-artifact` so the browser lab can show the same evidence without a
/// server; the route only decides the response envelope.
async fn api_benchmarks() -> Response {
    AxJson(bench_index()).into_response()
}

#[derive(Debug, Deserialize)]
struct ArtifactQuery {
    /// Example stem (`saxpy` → `examples/saxpy.hx`).
    example: String,
}

/// Full artifact for a stored example; 404 with a JSON body when unknown.
async fn api_artifact(State(state): State<AppState>, Query(q): Query<ArtifactQuery>) -> Response {
    match load_example(&state, &q.example) {
        Ok(src) => AxJson(artifact_for_example(&q.example, &src)).into_response(),
        Err(msg) => error_json(StatusCode::NOT_FOUND, &msg),
    }
}

/// Standalone offline HTML: same UI, artifact inlined, network not required.
async fn api_export(State(state): State<AppState>, Query(q): Query<ArtifactQuery>) -> Response {
    match load_example(&state, &q.example) {
        Ok(src) => {
            let art = artifact_for_example(&q.example, &src);
            let html = standalone_export(&q.example, &art);
            (
                [(header::CONTENT_TYPE, "text/html; charset=utf-8".to_string())],
                html,
            )
                .into_response()
        }
        Err(msg) => error_json(StatusCode::NOT_FOUND, &msg),
    }
}

#[derive(Debug, Deserialize)]
struct RunBody {
    /// HELIX source text from the editor pane.
    source: String,
}

/// Compiles POSTed source into an artifact (64 KiB cap).
async fn api_run(State(_state): State<AppState>, body: String) -> Response {
    if body.len() > MAX_SOURCE_BYTES {
        return error_json(
            StatusCode::PAYLOAD_TOO_LARGE,
            &format!("source exceeds {} bytes", MAX_SOURCE_BYTES),
        );
    }
    // The raw-body variant keeps malformed JSON from blocking compiles: any
    // text is treated as source unless it is a JSON object, which must carry
    // a usable `source` — silently compiling `{}` helps nobody.
    let src = match source_from_body(&body) {
        Ok(src) => src,
        Err(msg) => return error_json(StatusCode::BAD_REQUEST, &msg),
    };
    AxJson(build_artifact("<adhoc>", &src)).into_response()
}

/// Measures custom source after the normal compile pipeline has succeeded.
/// Syntax/sema failures are returned as normal artifacts so the UI can keep
/// showing the diagnostic stage; only a valid-but-unmeasurable program gets a
/// 422 with the reason.
async fn api_benchmark(State(_state): State<AppState>, body: String) -> Response {
    if body.len() > MAX_SOURCE_BYTES {
        return error_json(
            StatusCode::PAYLOAD_TOO_LARGE,
            &format!("source exceeds {} bytes", MAX_SOURCE_BYTES),
        );
    }
    let src = match source_from_body(&body) {
        Ok(src) => src,
        Err(msg) => return error_json(StatusCode::BAD_REQUEST, &msg),
    };
    let mut art = build_artifact("<adhoc>", &src);
    if art.diags_lex.is_some() || !art.diags_sem.is_empty() {
        return AxJson(art).into_response();
    }
    match measure_interpreter("<adhoc>", &src) {
        Ok(bench) => {
            art.bench = Some(bench);
            AxJson(art).into_response()
        }
        Err(msg) => error_json(StatusCode::UNPROCESSABLE_ENTITY, &msg),
    }
}

#[derive(Debug, Deserialize)]
struct DotQuery {
    /// Function name; defaults to the first one (`main` usually).
    #[serde(default)]
    r#fn: Option<String>,
}

/// `.dot` escape hatch for one function of one example.
async fn api_dot(
    State(state): State<AppState>,
    AxPath(example): AxPath<String>,
    Query(q): Query<DotQuery>,
) -> Response {
    match load_example(&state, &example) {
        Ok(src) => {
            let art = build_artifact(&example, &src);
            match dot_for(&art, q.r#fn.as_deref()) {
                Some(dot) => (
                    [(
                        header::CONTENT_TYPE,
                        "text/vnd.graphviz; charset=utf-8".to_string(),
                    )],
                    dot,
                )
                    .into_response(),
                None => error_json(
                    StatusCode::NOT_FOUND,
                    "no CFG for that function — did compilation reach IR?",
                ),
            }
        }
        Err(msg) => error_json(StatusCode::NOT_FOUND, &msg),
    }
}

async fn not_found() -> Response {
    error_json(StatusCode::NOT_FOUND, "no such route")
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Error body shaped like everything else the API emits.
fn error_json(status: StatusCode, msg: &str) -> Response {
    (status, AxJson(json!({ "error": msg }))).into_response()
}

/// Reads `examples/<name>.hx`; rejects traversal attempts outright.
fn load_example(state: &AppState, name: &str) -> Result<String, String> {
    if name.contains("..") || name.contains('/') || name.contains('\\') || name.is_empty() {
        return Err(format!("invalid example name '{name}'"));
    }
    let dir = state
        .examples_dir
        .as_ref()
        .ok_or_else(|| "examples directory unavailable (binary built without it)".to_string())?;
    std::fs::read_to_string(dir.join(format!("{name}.hx")))
        .map_err(|_| format!("unknown example '{name}'"))
}

/// Normalizes both documented JSON (`{"source":"…"}`) and raw-source POSTs.
fn source_from_body(body: &str) -> Result<String, String> {
    match serde_json::from_str::<RunBody>(body) {
        Ok(payload) if !payload.source.trim().is_empty() => Ok(payload.source),
        Ok(_) => Err("source must not be empty".to_string()),
        Err(_) if serde_json::from_str::<serde_json::Value>(body).is_ok_and(|v| v.is_object()) => {
            Err("JSON body must carry a non-empty `source` string".to_string())
        }
        Err(_) if body.trim().is_empty() => Err("source must not be empty".to_string()),
        Err(_) => Ok(body.to_string()),
    }
}

/// Sorted `.hx` stems in `dir`.
fn list_examples(dir: &std::path::Path) -> Vec<String> {
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for e in entries.flatten() {
            let p = e.path();
            if p.extension().is_some_and(|x| x == "hx")
                && let Some(stem) = p.file_stem().and_then(|s| s.to_str())
            {
                out.push(stem.to_string());
            }
        }
    }
    out.sort();
    if out.is_empty() {
        builtin_examples()
    } else {
        out
    }
}

/// Fallback list mirroring the repository's `examples/` directory.
fn builtin_examples() -> Vec<String> {
    vec![
        "casts_demo".into(),
        "const_globals".into(),
        "count_primes_sieve".into(),
        "div_guard".into(),
        "dot_reduction".into(),
        "fib_recursion".into(),
        "gcd_box_test".into(),
        "jacobi_2d".into(),
        "matmul".into(),
        "minmax_reduction".into(),
        "recurrence_reject".into(),
        "saxpy".into(),
        "scale".into(),
        "shortcircuit".into(),
        "small_n".into(),
        "ssa_demo".into(),
        "stencil_2d_reject".into(),
        "type_errors".into(),
        "writer_call".into(),
    ]
}

/// Minimal standard base64 (RFC 4648, padded) — no external crates in this
/// workspace, and the export needs fonts inlined as data: URIs.
fn base64(data: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [
            chunk[0],
            chunk.get(1).copied().unwrap_or(0),
            chunk.get(2).copied().unwrap_or(0),
        ];
        let n = u32::from_be_bytes([0, b[0], b[1], b[2]]);
        out.push(TABLE[(n >> 18) as usize & 63] as char);
        out.push(TABLE[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            TABLE[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            TABLE[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

/// Inline scripts must not contain a literal `</script>` (the HTML parser
/// closes the element at the first one, inside any JS string). Escaping the
/// slash as `<\/script` is a no-op inside JS strings and comments.
fn script_safe(js: &str) -> String {
    js.replace("</script", "<\\/script")
}

/// Renders the standalone export page.
///
/// Strategy: take the embedded `index.html` and inline EVERY asset it loads —
/// style sheet (with fonts as data: URIs), d3, and app.js — plus one shim
/// `<script>` that answers `/api/examples` and `/api/artifact` from an
/// inlined copy of the artifact. The saved file boots the real UI with no
/// server and no network: the advertised "standalone offline HTML".
fn standalone_export(name: &str, art: &CompileArtifact) -> String {
    let artifact_json = serde_json::to_string(art).unwrap_or_else(|_| "{\"schema\":1}".to_string());
    // A literal "</script>" inside the JSON would close the tag early; break
    // up every "</" sequence, which JSON string syntax tolerates as "\/".
    let safe_json = artifact_json.replace("</", "<\\/");

    let css = std::str::from_utf8(STYLE_CSS)
        .unwrap_or("")
        .replace(
            "./fonts/silkscreen-400.ttf",
            &format!("data:font/ttf;base64,{}", base64(FONT_400)),
        )
        .replace(
            "./fonts/silkscreen-700.ttf",
            &format!("data:font/ttf;base64,{}", base64(FONT_700)),
        );

    let shim = format!(
        r#"<script data-observe-export="{name}">
window.__INLINE_ARTIFACT_JSON__ = {safe_json};
(function () {{
  'use strict';
  // The inline value may be an object literal (same-origin document) —
  // normalise both spellings to a parsed object.
  var a = window.__INLINE_ARTIFACT_JSON__;
  var artifact = typeof a === 'string' ? null : (a || null);
  if (typeof a === 'string') {{
    try {{ artifact = JSON.parse(a); }} catch (e) {{ artifact = null; }}
  }}
  var realFetch = window.fetch ? window.fetch.bind(window) : null;
  window.fetch = function (input, init) {{
    var url = typeof input === 'string' ? input : (input && input.url) || '';
    function respond(obj) {{
      return Promise.resolve(new Response(JSON.stringify(obj), {{
        status: 200,
        headers: {{ 'Content-Type': 'application/json' }}
      }}));
    }}
    if (url.indexOf('/api/examples') === 0) {{
      return respond(artifact ? [artifact.example] : []);
    }}
    if (url.indexOf('/api/artifact') === 0) {{
      return respond(artifact || {{ schema: 1, example: '(export)', source: '' }});
    }}
    if (realFetch) return realFetch(input, init);
    return Promise.reject(new Error('offline export: ' + url));
  }};
}})();
</script>"#
    );

    let shell = std::str::from_utf8(INDEX_HTML).unwrap_or("");
    // Inline the style sheet, then swap the two script tags for
    // shim + d3 + app.js, all inline, app.js last.
    let shell = shell.replace(
        r#"<link rel="stylesheet" href="./style.css">"#,
        &format!("<style>\n{css}\n</style>"),
    );
    let mut out = String::with_capacity(shell.len() + shim.len() + D3_JS.len() + APP_JS.len() + 64);
    match shell.find(r#"<script src="./vendor/d3.v7.min.js"></script>"#) {
        Some(pos) => {
            out.push_str(&shell[..pos]);
            out.push_str(&shim);
            out.push_str("\n<script>\n");
            out.push_str(&script_safe(&String::from_utf8_lossy(D3_JS)));
            out.push_str("\n</script>\n<script>\n");
            out.push_str(&script_safe(&String::from_utf8_lossy(APP_JS)));
            out.push_str("\n</script>");
            let tail = pos + r#"<script src="./vendor/d3.v7.min.js"></script>"#.len();
            // Skip the original /app.js tag (now inlined above).
            let tail = shell[tail..]
                .find("</script>")
                .map_or(shell.len(), |end| tail + end + "</script>".len());
            out.push_str(&shell[tail..]);
        }
        None => {
            // index.html shape unexpected — keep the original shell and append
            // the shim; boot may degrade but the artifact stays inspectable in
            // __INLINE_ARTIFACT_JSON__.
            out.push_str(&shell);
            out.push_str(&shim);
        }
    }
    out
}

/// Extracts the `.dot` text for one function from a built artifact.
#[must_use]
pub fn dot_for(art: &CompileArtifact, fn_name: Option<&str>) -> Option<String> {
    let fns = art.cfg.as_ref()?.functions.as_slice();
    let pick = match fn_name {
        Some(want) => fns.iter().find(|f| f.name == want)?,
        None => fns.first()?,
    };
    Some(crate::dot::cfg_to_dot(pick))
}
