//! The Observatory HTTP server (axum).
//!
//! Serves the embedded web UI plus the JSON API the UI already speaks:
//!
//! | Route                              | Purpose                                    |
//! |------------------------------------|--------------------------------------------|
//! | `GET /`                            | `index.html` (embedded)                    |
//! | `GET /app.js`, `/style.css`        | UI assets (embedded, correct MIME)         |
//! | `GET /vendor/d3.v7.min.js`         | vendored d3 (embedded)                     |
//! | `GET /api/examples`                | example names from the repo's `examples/`  |
//! | `GET /api/artifact?example=name`   | full [`CompileArtifact`] for one example   |
//! | `POST /api/run {source}`           | artifact for editor-submitted source       |
//! | `GET /api/dot?example=…&fn=…`      | Graphviz `.dot` of one function            |
//! | `GET /api/artifact/export?…`       | standalone offline HTML with inline data   |
//!
//! Assets are compiled in with `include_bytes!`, so the server binary is
//! self-contained — the repo checkout is only needed to read example files.
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

use crate::artifact::CompileArtifact;
use crate::pipeline::{MAX_SOURCE_BYTES, build_artifact};

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
        .route("/app.js", get(app_js))
        .route("/style.css", get(style_css))
        .route("/vendor/d3.v7.min.js", get(d3_js))
        .route("/api/examples", get(api_examples))
        .route("/api/artifact", get(api_artifact))
        .route("/api/artifact/export", get(api_export))
        .route("/api/run", post(api_run))
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
/// `web/app.js`.
const APP_JS: &[u8] = include_bytes!("../../../web/app.js");
/// `web/style.css`.
const STYLE_CSS: &[u8] = include_bytes!("../../../web/style.css");
/// Vendored d3 v7.
const D3_JS: &[u8] = include_bytes!("../../../web/vendor/d3.v7.min.js");

async fn index_html() -> impl IntoResponse {
    bytes_response(INDEX_HTML, "text/html; charset=utf-8")
}

async fn app_js() -> impl IntoResponse {
    bytes_response(APP_JS, "text/javascript; charset=utf-8")
}

async fn style_css() -> impl IntoResponse {
    bytes_response(STYLE_CSS, "text/css; charset=utf-8")
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

#[derive(Debug, Deserialize)]
struct ArtifactQuery {
    /// Example stem (`saxpy` → `examples/saxpy.hx`).
    example: String,
}

/// Full artifact for a stored example; 404 with a JSON body when unknown.
async fn api_artifact(State(state): State<AppState>, Query(q): Query<ArtifactQuery>) -> Response {
    match load_example(&state, &q.example) {
        Ok(src) => AxJson(build_artifact(&q.example, &src)).into_response(),
        Err(msg) => error_json(StatusCode::NOT_FOUND, &msg),
    }
}

/// Standalone offline HTML: same UI, artifact inlined, network not required.
async fn api_export(State(state): State<AppState>, Query(q): Query<ArtifactQuery>) -> Response {
    match load_example(&state, &q.example) {
        Ok(src) => {
            let art = build_artifact(&q.example, &src);
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
    let src = match serde_json::from_str::<RunBody>(&body) {
        Ok(b) => b.source,
        Err(_) if serde_json::from_str::<serde_json::Value>(&body).is_ok_and(|v| v.is_object()) => {
            return error_json(
                StatusCode::BAD_REQUEST,
                "JSON body must carry a non-empty `source` string",
            );
        }
        Err(_) => body.clone(),
    };
    if src.trim().is_empty() {
        return error_json(StatusCode::BAD_REQUEST, "source must not be empty");
    }
    AxJson(build_artifact("<adhoc>", &src)).into_response()
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
        "saxpy".into(),
        "matmul".into(),
        "jacobi_2d".into(),
        "dot_reduction".into(),
        "minmax_reduction".into(),
        "count_primes_sieve".into(),
        "fib_recursion".into(),
        "gcd_box_test".into(),
        "recurrence_reject".into(),
        "stencil_2d_reject".into(),
        "type_errors".into(),
    ]
}

/// Renders the standalone export page.
///
/// Strategy: take the **unmodified** embedded `index.html` and inject one
/// `<script>` block immediately before the existing script tags. The block
/// installs a `fetch` shim that answers `/api/examples` and `/api/artifact`
/// from an inlined copy of the artifact, so the real UI boots unchanged and
/// runs with no server at all. Nothing inside `web/` is touched, so the
/// shipped UI and the offline export can never drift apart.
fn standalone_export(name: &str, art: &CompileArtifact) -> String {
    let artifact_json = serde_json::to_string(art).unwrap_or_else(|_| "{\"schema\":1}".to_string());
    // A literal "</script>" inside the JSON would close the tag early; break
    // up every "</" sequence, which JSON string syntax tolerates as "\/".
    let safe_json = artifact_json.replace("</", "<\\/");

    let shell = std::str::from_utf8(INDEX_HTML).unwrap_or("");
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
</script>
<script src="/vendor/d3.v7.min.js"></script>"#
    );

    // Replace BOTH original script tags with the shim block (which itself
    // re-includes d3 first), keeping app.js last.
    if let Some(pos) = shell.find(r#"<script src="/vendor/d3.v7.min.js"></script>"#) {
        let mut out = String::with_capacity(shell.len() + shim.len());
        out.push_str(&shell[..pos]);
        out.push_str(&shim);
        out.push('\n');
        out.push_str(&shell[pos + r#"<script src="/vendor/d3.v7.min.js"></script>"#.len()..]);
        out
    } else {
        // index.html shape unexpected — append the shim; boot may degrade but
        // the artifact stays inspectable in __INLINE_ARTIFACT_JSON__.
        format!("{shell}{shim}\n")
    }
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
