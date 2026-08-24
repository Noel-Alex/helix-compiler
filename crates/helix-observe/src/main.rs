//! `helix observe` — the Observatory server binary.
//!
//! Serves the embedded web UI plus the artifact API on 127.0.0.1 (default
//! port 8931) and opens the default browser unless `--no-open` is passed:
//!
//! ```text
//! cargo run -p helix-observe            # http://127.0.0.1:8931/
//! cargo run -p helix-observe -- --port 9000 --no-open
//! ```
//!
//! Examples are read from `../../examples` relative to this crate at runtime
//! so freshly edited `.hx` files appear without a rebuild; when the directory
//! is absent (installed binary), a built-in example list is served instead.

use std::path::PathBuf;

#[tokio::main]
async fn main() {
    let mut port: u16 = 8931;
    let mut open = true;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--port" => {
                port = args.next().and_then(|v| v.parse().ok()).unwrap_or_else(|| {
                    eprintln!("--port needs a number");
                    std::process::exit(2);
                });
            }
            "--no-open" => open = false,
            "--help" | "-h" => {
                println!("helix observe [--port N] [--no-open]");
                return;
            }
            other => {
                eprintln!("unknown flag '{other}' (try --help)");
                std::process::exit(2);
            }
        }
    }

    // Examples live two levels up from this crate in a repo checkout.
    let examples_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples")
        .canonicalize()
        .ok();

    let cfg = helix_observe::ServeConfig {
        addr: std::net::SocketAddr::from(([127, 0, 0, 1], port)),
        open_browser: open,
        examples_dir,
    };
    if let Err(e) = helix_observe::serve(cfg).await {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
