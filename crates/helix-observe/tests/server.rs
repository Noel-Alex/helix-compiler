//! Server integration: spawn the axum router on an ephemeral port inside a
//! tokio runtime and probe it with a raw `std::net::TcpStream` HTTP/1.1
//! client — no new dependencies, exactly what a browser would send.
//!
//! The blocking client runs on `spawn_blocking`: the default test runtime is
//! current-thread, so a synchronous read on the main future's thread would
//! starve the server half and deadlock.
//!
//! Covered: static assets (status + content-type), `/api/examples`,
//! `/api/artifact` happy + missing paths, `POST /api/run` round trip
//! (JSON body and raw source), payload cap, dot route, standalone export,
//! and traversal rejection.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::PathBuf;

use helix_observe::server::router;

/// Resolved repo examples directory.
fn examples_dir() -> PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples")
        .canonicalize()
        .expect("examples dir")
}

/// Spawns the router on 127.0.0.1:0 and returns its bound address.
async fn spawn() -> SocketAddr {
    spawn_with_examples(examples_dir()).await
}

async fn spawn_with_examples(dir: PathBuf) -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("ephemeral bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move {
        axum::serve(listener, router(Some(dir)))
            .await
            .expect("serve");
    });
    addr
}

/// Minimal blocking HTTP/1.1 GET/POST; returns (status, content_type, body).
///
/// Runs entirely on a blocking thread so the async runtime stays live.
async fn http(
    addr: SocketAddr,
    method: &str,
    target: &str,
    body: Option<&str>,
) -> (u16, String, String) {
    let method = method.to_string();
    let target = target.to_string();
    let body = body.map(str::to_string);
    tokio::task::spawn_blocking(move || http_blocking(addr, &method, &target, body.as_deref()))
        .await
        .expect("client task")
}

/// The same exchange without the async wrapper (used by spawn_blocking).
fn http_blocking(
    addr: SocketAddr,
    method: &str,
    target: &str,
    body: Option<&str>,
) -> (u16, String, String) {
    let mut stream = TcpStream::connect(addr).expect("connect");
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(30)))
        .expect("read timeout");
    let head = match body {
        Some(b) => format!(
            "{method} {target} HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n",
            b.len()
        ),
        None => format!("{method} {target} HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n"),
    };
    stream.write_all(head.as_bytes()).expect("write head");
    if let Some(b) = body {
        stream.write_all(b.as_bytes()).expect("write body");
    }
    let mut buf = Vec::new();
    stream.read_to_end(&mut buf).expect("read response");
    let text = String::from_utf8_lossy(&buf).to_string();
    let status: u16 = text
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let ctype = text
        .lines()
        .find(|l| l.to_ascii_lowercase().starts_with("content-type:"))
        .map(|l| l.split_once(':').expect("header").1.trim().to_string())
        .unwrap_or_default();
    let body_start = buf
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .map_or(buf.len(), |i| i + 4);
    // Chunked responses are re-assembled naively: strip chunk-size lines.
    let raw = String::from_utf8_lossy(&buf[body_start..]).to_string();
    let is_chunked = text
        .to_ascii_lowercase()
        .contains("transfer-encoding: chunked");
    let body_text = if is_chunked { dechunk(&raw) } else { raw };
    (status, ctype, body_text)
}

/// Strips HTTP chunked-encoding framing.
fn dechunk(raw: &str) -> String {
    let mut out = String::new();
    let mut rest = raw.to_string();
    while let Some(line_end) = rest.find("\r\n") {
        let size = usize::from_str_radix(rest[..line_end].trim(), 16).unwrap_or(0);
        if size == 0 {
            break;
        }
        let start = line_end + 2;
        let end = (start + size).min(rest.len());
        out.push_str(&rest[start..end]);
        rest = rest[(end + 2).min(rest.len())..].to_string();
    }
    out
}

#[tokio::test]
async fn static_assets_serve_with_correct_mime() {
    let addr = spawn().await;
    for (path, want) in [
        ("/", "text/html"),
        ("/about", "text/html"),
        ("/app.js", "text/javascript"),
        ("/style.css", "text/css"),
        ("/about.css", "text/css"),
        ("/about.js", "text/javascript"),
        ("/wasm/helix-browser.js", "text/javascript"),
        ("/wasm/helix-worker.js", "text/javascript"),
        ("/wasm/helix-lab.wasm", "application/wasm"),
        ("/assets/helix-loop-observatory.png", "image/png"),
        ("/vendor/d3.v7.min.js", "text/javascript"),
    ] {
        let (status, ctype, body) = http(addr, "GET", path, None).await;
        assert_eq!(status, 200, "{path}");
        assert!(ctype.starts_with(want), "{path}: got {ctype}");
        assert!(!body.is_empty(), "{path}: non-empty");
    }
}

#[tokio::test]
async fn benchmark_index_exposes_recorded_campaign_provenance() {
    let addr = spawn().await;
    let (status, ctype, body) = http(addr, "GET", "/api/benchmarks", None).await;
    assert_eq!(status, 200);
    assert!(ctype.starts_with("application/json"));
    let index: serde_json::Value = serde_json::from_str(&body).expect("benchmark index json");
    assert_eq!(index["schema"], "helix-campaign-v1");
    assert_eq!(index["point_count"], 17);
    assert!(
        index["kernels"]
            .as_array()
            .is_some_and(|kernels| { kernels.iter().any(|kernel| kernel["name"] == "saxpy") })
    );
    assert!(index["points"].as_array().is_some_and(|points| {
        points.iter().any(|point| {
            point["kernel"] == "scale"
                && point["n"] == 65536
                && point["variants"]
                    .as_array()
                    .is_some_and(|variants| variants.len() == 3)
        })
    }));
    assert!(index["machine"]["cpu_brand"].as_str().is_some());
    assert!(index["examples"].as_array().is_some_and(|examples| {
        examples
            .iter()
            .any(|example| example["name"] == "casts_demo" && example["sample_count"] == 7)
    }));
}

#[tokio::test]
async fn benchmark_index_preserves_probe_provenance() {
    let addr = spawn().await;
    let (_, _, body) = http(addr, "GET", "/api/benchmarks", None).await;
    let index: serde_json::Value = serde_json::from_str(&body).expect("index");
    let catalog: serde_json::Value = serde_json::from_str(include_str!(
        "../../../docs/benchmarks/data/example-probes.json"
    ))
    .expect("probe catalog");
    let provenance = &index["probe_provenance"];
    assert_eq!(provenance["schema"], catalog["schema"]);
    assert_eq!(provenance["timestamp_utc"], catalog["timestamp_utc"]);
    assert_ne!(provenance["timestamp_utc"], index["timestamp_utc"]);
    assert_eq!(provenance["git_rev"], catalog["machine"]["git_rev"]);
    for field in [
        "cpu_brand",
        "logical_cores",
        "physical_cores",
        "rustc_version",
    ] {
        assert_eq!(provenance["machine"][field], catalog["machine"][field]);
    }
}

#[tokio::test]
async fn api_examples_lists_directory_contents() {
    let addr = spawn().await;
    let (status, ctype, body) = http(addr, "GET", "/api/examples", None).await;
    assert_eq!(status, 200);
    assert!(ctype.starts_with("application/json"));
    let names: Vec<String> = serde_json::from_str(&body).expect("json array");
    assert!(names.contains(&"saxpy".to_string()));
    assert!(names.iter().all(|n| !n.ends_with(".hx")), "stems only");
}

#[tokio::test]
async fn artifact_route_returns_full_document() {
    let addr = spawn().await;
    let (status, ctype, body) = http(addr, "GET", "/api/artifact?example=saxpy", None).await;
    assert_eq!(status, 200);
    assert!(ctype.starts_with("application/json"));
    let art: serde_json::Value = serde_json::from_str(&body).expect("artifact json");
    assert_eq!(art["schema"], 1);
    assert_eq!(art["example"], "saxpy");
    assert!(art["cfg"]["functions"].as_array().is_some());
    assert!(art["loops"].as_array().is_some_and(|l| !l.is_empty()));
    // The teaching example omits the measured campaign's initialization loop.
    assert!(
        art["bench"].is_null(),
        "a matching name is not measured-source identity"
    );
}

#[tokio::test]
async fn non_kernel_bundled_example_exposes_recorded_interpreter_probe() {
    let addr = spawn().await;
    let (status, _ctype, body) = http(addr, "GET", "/api/artifact?example=casts_demo", None).await;
    assert_eq!(status, 200);
    let art: serde_json::Value = serde_json::from_str(&body).expect("artifact json");
    assert_eq!(art["bench"]["kernel"], "casts_demo");
    assert_eq!(art["bench"]["variants"][0]["name"], "interp-local");
    assert_eq!(
        art["bench"]["variants"][0]["samples"]
            .as_array()
            .map(Vec::len),
        Some(7)
    );
    assert!(
        art["bench"]["efficiency"]
            .as_array()
            .is_some_and(Vec::is_empty)
    );
}

#[tokio::test]
async fn mounted_examples_require_recorded_source_identity() {
    let sources: serde_json::Value = serde_json::from_str(include_str!(
        "../../helix-artifact/src/recorded-sources.json"
    ))
    .expect("recorded sources");
    let dir = std::env::temp_dir().join(format!("helix-observe-identity-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("test examples dir");
    let addr = spawn_with_examples(dir.clone()).await;
    let saxpy = sources["campaign"]["saxpy"]
        .as_str()
        .unwrap()
        .replace("33554432", "64");
    let probe = sources["probes"]["casts_demo"].as_str().unwrap();
    for (name, source) in [("saxpy", saxpy.as_str()), ("casts_demo", probe)] {
        for (text, measured) in [
            (format!("// presentation-only comment\n{source}"), true),
            (format!("{source}\nfn unmeasured() {{ print(1); }}"), false),
        ] {
            std::fs::write(dir.join(format!("{name}.hx")), text).expect("example");
            let (status, _, body) =
                http(addr, "GET", &format!("/api/artifact?example={name}"), None).await;
            assert_eq!(status, 200);
            let artifact: serde_json::Value = serde_json::from_str(&body).expect("artifact");
            assert_eq!(!artifact["bench"].is_null(), measured, "{name}");
        }
    }
    std::fs::remove_dir_all(dir).expect("remove test examples");
}

#[tokio::test]
async fn artifact_route_404s_unknown_example() {
    let addr = spawn().await;
    let (status, _ctype, body) = http(addr, "GET", "/api/artifact?example=nope", None).await;
    assert_eq!(status, 404);
    assert!(body.contains("error"));
}

#[tokio::test]
async fn artifact_route_rejects_traversal() {
    let addr = spawn().await;
    let (status, _, _) = http(
        addr,
        "GET",
        "/api/artifact?example=..%2FCargo.toml%00",
        None,
    )
    .await;
    assert_ne!(status, 200, "traversal must not read arbitrary files");
}

#[tokio::test]
async fn run_post_round_trips_source_to_artifact() {
    let addr = spawn().await;
    let src = "fn main() { print(1 + 2); }";
    let payload = serde_json::to_string(&serde_json::json!({ "source": src })).expect("body");

    let (status, ctype, body) = http(addr, "POST", "/api/run", Some(&payload)).await;
    assert_eq!(status, 200);
    assert!(ctype.starts_with("application/json"));
    let art: serde_json::Value = serde_json::from_str(&body).expect("artifact");
    assert_eq!(art["example"], "<adhoc>");
    assert_eq!(art["source"], src);
    // The tiny program fully compiles AND executes.
    assert_eq!(art["exec"]["printed"].as_array().expect("prints")[0], "3");
    assert!(
        art["exec"]["checksum"]
            .as_str()
            .is_some_and(|c| c.starts_with("0x"))
    );
}

#[tokio::test]
async fn run_post_accepts_raw_source_without_json_envelope() {
    let addr = spawn().await;
    let (status, _ctype, body) =
        http(addr, "POST", "/api/run", Some("fn main() { print(7); }")).await;
    assert_eq!(status, 200);
    let art: serde_json::Value = serde_json::from_str(&body).expect("artifact");
    assert_eq!(art["exec"]["printed"][0], "7");
}

#[tokio::test]
async fn benchmark_post_measures_custom_source_without_inventing_parallel_data() {
    let addr = spawn().await;
    let src = "fn main() { print(7); }";
    let payload = serde_json::to_string(&serde_json::json!({ "source": src })).expect("body");

    let (status, ctype, body) = http(addr, "POST", "/api/benchmark", Some(&payload)).await;
    assert_eq!(status, 200);
    assert!(ctype.starts_with("application/json"));
    let art: serde_json::Value = serde_json::from_str(&body).expect("artifact");
    assert_eq!(art["example"], "<adhoc>");
    assert_eq!(art["bench"]["kernel"], "<adhoc>");
    assert_eq!(art["bench"]["variants"][0]["name"], "interp-local");
    assert_eq!(
        art["bench"]["variants"][0]["samples"]
            .as_array()
            .map(Vec::len),
        Some(7)
    );
    assert!(
        art["bench"]["efficiency"]
            .as_array()
            .is_some_and(Vec::is_empty)
    );
    assert_eq!(art["exec"]["printed"][0], "7");
}

#[tokio::test]
async fn run_post_rejects_json_without_usable_source() {
    let addr = spawn().await;
    // A JSON object without a usable `source` must 400 with a showable
    // message, not silently compile empty source into a full artifact.
    for payload in [
        "{}",
        "{\"other\":1}",
        "{\"source\":\"\"}",
        "{\"source\":\"  \"}",
    ] {
        let (status, ctype, body) = http(addr, "POST", "/api/run", Some(payload)).await;
        assert_eq!(status, 400, "{payload}");
        assert!(ctype.starts_with("application/json"), "{payload}");
        let err: serde_json::Value = serde_json::from_str(&body).expect("error json");
        assert!(
            err["error"].as_str().is_some_and(|m| m.contains("source")),
            "{payload}: {body}"
        );
    }
    // Valid JSON that is NOT an object stays on the raw-source path.
    let (status, _, _) = http(addr, "POST", "/api/run", Some("null")).await;
    assert_eq!(status, 200);
}

#[tokio::test]
async fn run_post_rejects_oversized_sources() {
    let addr = spawn().await;
    let big = format!(
        "{{\"source\":\"{}\"}}",
        "// x\n".repeat(helix_observe::MAX_SOURCE_BYTES / 4 + 10)
    );
    let (status, _ctype, _body) = http(addr, "POST", "/api/run", Some(&big)).await;
    assert_eq!(status, 413);
}

#[tokio::test]
async fn dot_route_emits_graphviz_text() {
    let addr = spawn().await;
    let (status, ctype, body) = http(addr, "GET", "/api/dot/saxpy", None).await;
    assert_eq!(status, 200);
    assert!(ctype.starts_with("text/vnd.graphviz"), "got {ctype}");
    assert!(body.starts_with("digraph \"main\""));
    assert!(body.contains("->"));
    // Named function lookup works too.
    let (status, _, _) = http(addr, "GET", "/api/dot/fib_recursion?fn=fib", None).await;
    assert_eq!(status, 200);
}

#[tokio::test]
async fn export_route_inlines_artifact_into_standalone_html() {
    let addr = spawn().await;
    let (status, ctype, body) =
        http(addr, "GET", "/api/artifact/export?example=ssa_demo", None).await;
    assert_eq!(status, 200);
    assert!(ctype.starts_with("text/html"));
    assert!(
        body.contains("__INLINE_ARTIFACT_JSON__"),
        "artifact embedded"
    );
    // The saved file must boot with no server: every asset the shell loads is
    // inlined (style sheet, both scripts, fonts as data: URIs) — no absolute
    // asset URLs may remain for the file:// origin to fail on.
    assert!(!body.contains(r#"src="/"#), "no external script tags");
    assert!(
        !body.contains(r#"href="/style.css""#),
        "style sheet inlined"
    );
    assert!(
        body.contains("renderExecReadout"),
        "app.js source inlined into the export"
    );
    assert!(body.contains("data-observe-export"), "fetch shim present");
    assert!(
        body.contains("data:font/ttf;base64,"),
        "fonts inlined as data URIs"
    );
}
