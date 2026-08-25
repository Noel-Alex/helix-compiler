//! The `helix` command-line driver.
//!
//! Subcommands:
//! - `run <file.hx>`          — parse → check → interpret (reference backend)
//! - `check <file.hx>`        — frontend only; prints diagnostics with carets
//! - `dump <stage> <file.hx>` — print a pipeline stage (tokens|ast|ir|ssa)
//!   for eyeballing and golden tests
//!
//! The JIT backend (`--backend jit`) and the Observatory server live in later
//! milestones; the argument surface already reserves them.

use std::path::PathBuf;

use helix_ir::print::print_ir;
use helix_sema::TypedProgram;
use helix_syntax::Span;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = dispatch(&args);
    std::process::exit(code);
}

fn dispatch(args: &[String]) -> i32 {
    match args.first().map(String::as_str) {
        Some("run") => cmd_run(&args[1..]),
        Some("check") => cmd_check(args.get(1)),
        Some("dump") => match (args.get(1), args.get(2)) {
            (Some(stage), Some(file)) => cmd_dump(stage, file),
            _ => usage("dump requires <stage> <file>"),
        },
        Some("loops") => match args.get(1) {
            Some(file) => cmd_loops(file),
            None => usage("loops requires <file>"),
        },
        Some("bench") => cmd_bench(&args[1..]),
        Some("observe") => cmd_observe(&args[1..]),
        Some("selftest") => cmd_selftest(),
        Some("--help" | "-h" | "help") | None => {
            print_help();
            0
        }
        Some(other) => usage(&format!("unknown subcommand '{other}'")),
    }
}

fn usage(msg: &str) -> i32 {
    eprintln!("error: {msg}");
    print_help();
    1
}

fn print_help() {
    println!(
        "HELIX Lite — automatic parallelizing compiler

USAGE:
    helix run <file.hx>              run via the reference interpreter
    helix check <file.hx>            type-check only, print diagnostics
    helix dump <stage> <file.hx>     print a pipeline stage
                                     stages: tokens | ast | ir | ssa
    helix loops <file.hx>            loop detection + dependence verdicts
    helix run --backend jit <f.hx>   run through the Cranelift JIT
                                     [--unchecked] strips bounds checks
    helix bench [--quick] [--out D]  benchmark campaign (JSON into docs/benchmarks/data)
    helix observe [--port N] [--no-open]
                                     the Observatory web UI
    helix selftest                   interp-vs-JIT differential gauntlet
    helix help                       this message"
    );
}

/// Shared frontend: source string → checked program, printing diags on failure.
fn frontend(path: &str) -> Result<TypedProgram, i32> {
    let src = std::fs::read_to_string(PathBuf::from(path)).map_err(|e| {
        eprintln!("error: cannot read {path}: {e}");
        1
    })?;
    let program = match helix_syntax::parse_str(&src) {
        Ok(p) => p,
        Err(e) => {
            let span = match &e {
                helix_syntax::SyntaxError::Lex(x) => x.span,
                helix_syntax::SyntaxError::Parse(x) => x.span,
            };
            print_diag(&src, span, &e.to_string());
            return Err(1);
        }
    };
    helix_sema::check(&program).map_err(|diags| {
        for d in &diags {
            print_diag(&src, d.span, &d.msg);
        }
        eprintln!("{} error(s)", diags.len());
        1
    })
}

fn cmd_run(rest: &[String]) -> i32 {
    let mut path: Option<&str> = None;
    let mut backend = "interp";
    let mut unchecked = false;
    let mut it = rest.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--backend" | "-b" => backend = it.next().map(String::as_str).unwrap_or("interp"),
            // Combined `--backend=jit` form: silently ignoring it ran the
            // INTERPRETER while the user believed they tested the JIT.
            flag if flag.starts_with("--backend=") || flag.starts_with("-b=") => {
                backend = &flag[flag.find('=').expect("prefix checked") + 1..];
            }
            "jit" | "interp" => backend = a,
            "--unchecked" => unchecked = true,
            other if path.is_none() => path = Some(other),
            _ => {}
        }
    }
    let Some(path) = path else {
        return usage("run requires <file>");
    };
    if backend != "interp" && backend != "jit" {
        eprintln!("error: unknown backend '{backend}' (expected 'interp' or 'jit')");
        return 2;
    }
    let src = std::fs::read_to_string(PathBuf::from(path)).unwrap_or_default();
    let program = match frontend(path) {
        Ok(p) => p,
        Err(code) => return code,
    };
    if backend == "jit" {
        return run_jit(&src, &program, unchecked);
    }
    match helix_engine::run_with_source(&src, &program) {
        Ok(out) => {
            for line in &out.printed {
                println!("{line}");
            }
            0
        }
        Err(e) => {
            eprintln!("{e}");
            1
        }
    }
}

/// JIT execution path: lower -> analyze -> plan -> compile -> run.
fn run_jit(_src: &str, program: &TypedProgram, unchecked: bool) -> i32 {
    let mut funcs = helix_ir::build(program);
    for f in &mut funcs {
        helix_ir::to_ssa(f);
    }
    // Analysis + plan (regions only when parallelization is sound).
    let mut loops_per_fn = Vec::new();
    let mut reports_per_fn = Vec::new();
    for f in &funcs {
        let li = helix_analysis::find_loops(f);
        let reps = helix_analysis::analyze(f, &li);
        loops_per_fn.push(li);
        reports_per_fn.push(reps);
    }
    let plan = helix_analysis::build_plan(&funcs, &loops_per_fn, &reports_per_fn);

    // Production JIT runs use the engine's normal error path: the first
    // runtime guard prints `runtime error: ...` and exits with status 1
    // (helix_panic). The test-only trap recorder must stay disarmed here —
    // arming it made guards record-and-RESUME, so a trapping program kept
    // executing on dummy values and printed garbage after the error.
    let engine = match helix_backend::JitEngine::compile(&funcs, &to_backend_plan(&plan), unchecked)
    {
        Ok(e) => e,
        Err(msg) => {
            eprintln!("jit compile error: {msg}");
            return 1;
        }
    };
    let (prints, result) = helix_backend::engine::capture_prints(|| engine.run_main());
    for line in prints {
        println!("{line}");
    }
    if let Err(msg) = result {
        eprintln!("{msg}");
        return 1;
    }
    0
}

fn cmd_check(path: Option<&String>) -> i32 {
    let Some(path) = path else {
        return usage("check requires <file>");
    };
    match frontend(path) {
        Ok(_) => {
            println!("ok");
            0
        }
        Err(code) => code,
    }
}

fn cmd_dump(stage: &str, path: &str) -> i32 {
    let src = std::fs::read_to_string(PathBuf::from(path)).unwrap_or_default();
    let tokens = match helix_syntax::lex(&src) {
        Ok(t) => t,
        Err(e) => {
            print_diag(&src, e.span, &e.msg);
            return 1;
        }
    };

    match stage {
        "tokens" => {
            for tok in &tokens {
                let text = &src[tok.span.start as usize..tok.span.end as usize];
                println!(
                    "{:>4}..{:<4} {:?} {text:?}",
                    tok.span.start, tok.span.end, tok.kind
                );
            }
            0
        }
        "ast" => match helix_syntax::parse_str(&src) {
            Ok(p) => {
                println!("{}", p.print_tree());
                0
            }
            Err(e) => {
                let span = match &e {
                    helix_syntax::SyntaxError::Lex(x) => x.span,
                    helix_syntax::SyntaxError::Parse(x) => x.span,
                };
                print_diag(&src, span, &e.to_string());
                1
            }
        },
        "ir" | "ssa" => {
            let want_ssa = stage == "ssa";
            let program = match frontend(path) {
                Ok(p) => p,
                Err(code) => return code,
            };
            let mut funcs = helix_ir::build(&program);
            if want_ssa {
                for f in &mut funcs {
                    helix_ir::to_ssa(f);
                    if let Err(e) = helix_ir::verify_ssa(f) {
                        eprintln!("ssa verification failed for '{}': {e}", f.name);
                        return 1;
                    }
                }
            }
            for f in &funcs {
                println!("=== {} ===", f.name);
                println!("{}", print_ir(f, want_ssa));
            }
            0
        }
        other => usage(&format!("unknown stage '{other}' (tokens|ast|ir|ssa)")),
    }
}

/// Print one diagnostic with a caret line under the offending span.
///
/// Columns count CHARS, not bytes (UTF-8 sources get correct carets), and
/// `\r` is excluded from the rendered line so CRLF files don't drift the
/// caret right by one per preceding line break.
fn print_diag(src: &str, span: Span, msg: &str) {
    let line_no = src[..span.start as usize].matches('\n').count() + 1;
    let line_start = src[..span.start as usize].rfind('\n').map_or(0, |i| i + 1);
    let line_end = src[span.start as usize..]
        .find('\n')
        .map_or(src.len(), |i| span.start as usize + i);
    let line_raw = &src[line_start..line_end];
    let line = line_raw.trim_end_matches('\r');
    // Char distance from line start to the span start (tabs render as one
    // column; matches how terminals with tabstop=8 would still misalign less
    // than byte-counting does for any non-ASCII source).
    let caret_col = src[line_start..span.start as usize].chars().count();
    let caret_len = (span.end.saturating_sub(span.start)).max(1);

    eprintln!("--> line {line_no}: {msg}");
    eprintln!("{line}");
    eprintln!(
        "{}{}",
        " ".repeat(caret_col),
        "^".repeat(caret_len as usize)
    );
}

// ---------------------------------------------------------------------------
// loops: dependence verdicts per loop
// ---------------------------------------------------------------------------

fn cmd_loops(path: &str) -> i32 {
    let program = match frontend(path) {
        Ok(p) => p,
        Err(code) => return code,
    };
    let funcs = helix_ir::build(&program);
    for f in &funcs {
        let mut f = clone_fn(f);
        helix_ir::to_ssa(&mut f);
        let li = helix_analysis::find_loops(&f);
        if li.loops.is_empty() {
            continue;
        }
        println!("==== {} loop analysis ====", f.name);
        for rep in helix_analysis::analyze(&f, &li) {
            println!("{}", rep.summary_line());
            for line in &rep.accesses {
                println!("    {line}");
            }
            for d in rep
                .raw_deps
                .iter()
                .chain(&rep.war_deps)
                .chain(&rep.waw_deps)
            {
                println!("    {}", d.explain);
            }
        }
    }
    0
}

fn clone_fn(f: &helix_ir::FuncIr) -> helix_ir::FuncIr {
    f.clone()
}

// ---------------------------------------------------------------------------
// bench / observe / selftest
// ---------------------------------------------------------------------------

fn cmd_bench(args: &[String]) -> i32 {
    let quick = args.iter().any(|a| a == "--quick");
    let out = args
        .iter()
        .position(|a| a == "--out")
        .and_then(|i| args.get(i + 1))
        .map_or_else(|| PathBuf::from("docs/benchmarks/data"), PathBuf::from);
    std::fs::create_dir_all(&out).expect("create bench dir");
    match helix_bench::campaign_main(&out) {
        Ok(()) => {
            if !quick {
                println!("campaign complete — JSONs in {}", out.display());
            }
            0
        }
        Err(e) => {
            eprintln!("bench failed: {e}");
            1
        }
    }
}

fn cmd_observe(args: &[String]) -> i32 {
    let mut port: u16 = 8931;
    let mut open = true;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--port" => {
                port = it.next().and_then(|v| v.parse().ok()).unwrap_or(port);
            }
            "--no-open" => open = false,
            other => {
                eprintln!("unknown observe flag '{other}'");
                return 2;
            }
        }
    }
    let examples_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples")
        .canonicalize()
        .ok();
    let cfg = helix_observe::ServeConfig {
        addr: std::net::SocketAddr::from(([127, 0, 0, 1], port)),
        open_browser: open,
        examples_dir,
    };
    // The server runs forever; hand the thread to tokio.
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");
    match rt.block_on(helix_observe::serve(cfg)) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("observe failed: {e}");
            1
        }
    }
}

/// Differential gauntlet: every example through interpreter AND JIT, outputs must agree.
fn cmd_selftest() -> i32 {
    let examples_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples");
    let entries = match std::fs::read_dir(&examples_dir) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("cannot read examples dir: {e}");
            return 1;
        }
    };
    let mut pass = 0usize;
    let mut fail = 0usize;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("hx") {
            continue;
        }
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let src = match std::fs::read_to_string(&path) {
            Ok(s) => s,
            Err(_) => continue,
        };
        // Frontend errors are expected for the deliberately-broken examples.
        let program = (|| -> Option<TypedProgram> {
            let parsed = helix_syntax::parse_str(&src).ok()?;
            helix_sema::check(&parsed).ok()
        })();
        let Some(program) = program else {
            println!("  skip {name} (frontend rejects, as intended)");
            continue;
        };
        // Interpreter oracle.
        let interp = helix_engine::run_with_source(&src, &program);
        // JIT.
        let mut funcs = helix_ir::build(&program);
        for f in &mut funcs {
            helix_ir::to_ssa(f);
        }
        let li: Vec<_> = funcs.iter().map(helix_analysis::find_loops).collect();
        let reps: Vec<_> = funcs
            .iter()
            .zip(&li)
            .map(|(f, l)| helix_analysis::analyze(f, l))
            .collect();
        let plan = helix_analysis::build_plan(&funcs, &li, &reps);

        // The recorder stays DISARMED in selftest too: with it armed, a JIT
        // run that hits a runtime guard records-and-resumes and reports Ok,
        // so the "both backends reject" comparison could never fire (every
        // trap looked like success). Without it, helix_panic exits the
        // process — but examples/*.hx are all guard-clean by construction
        // (the deliberately-trapping ones are frontend rejects), so no
        // example should ever reach a guard. If one does, the abort IS the
        // signal; the harness treats a missing exit as a failure upstream.
        let engine = match helix_backend::JitEngine::compile(&funcs, &to_backend_plan(&plan), false)
        {
            Ok(e) => e,
            Err(msg) => {
                println!("  FAIL {name}: jit compile: {msg}");
                fail += 1;
                continue;
            }
        };
        let (jit_prints, jit_res) = helix_backend::engine::capture_prints(|| engine.run_main());

        match (&interp, &jit_res) {
            (Ok(i), Ok(())) => {
                if i.printed == jit_prints {
                    println!(
                        "  ok   {name} ({} lines, checksum {:016x})",
                        i.printed.len(),
                        i.checksum
                    );
                    pass += 1;
                } else {
                    println!(
                        "  FAIL {name}: output mismatch\n    interp: {:?}\n    jit:    {:?}",
                        i.printed, jit_prints
                    );
                    fail += 1;
                }
            }
            (Err(_), Err(_)) => {
                println!("  ok   {name} (both backends reject at runtime)");
                pass += 1;
            }
            (a, b) => {
                println!(
                    "  FAIL {name}: backend disagreement interp={:?} jit={:?}",
                    a.is_ok(),
                    b.is_ok()
                );
                fail += 1;
            }
        }
    }
    println!("selftest: {pass} passed, {fail} failed");
    if fail > 0 { 1 } else { 0 }
}

/// Convert an analysis plan to the backend's seam type (M10: the fields map
/// 1:1 and reduction operators pass through so min/max regions combine with
/// the right monoid).
fn to_backend_plan(p: &helix_analysis::ParallelPlan) -> helix_backend::ParallelPlan {
    use helix_backend::engine::helix_analysis_stub::ReductionOp as BOp;
    let mut out = helix_backend::ParallelPlan::default();
    for r in &p.regions {
        out.regions.push(helix_backend::RegionDesc {
            func_idx: r.func_idx,
            header: r.header,
            kind: match r.kind {
                helix_analysis::RegionKind::DoAll => helix_backend::RegionKind::DoAll,
                helix_analysis::RegionKind::Reduction(op) => {
                    helix_backend::RegionKind::Reduction(match op {
                        helix_analysis::ReductionOp::Add => BOp::Add,
                        helix_analysis::ReductionOp::Mul => BOp::Mul,
                        helix_analysis::ReductionOp::Min => BOp::Min,
                        helix_analysis::ReductionOp::Max => BOp::Max,
                    })
                }
            },
            body_fn_name: r.body_fn_name.clone(),
        });
    }
    out
}
