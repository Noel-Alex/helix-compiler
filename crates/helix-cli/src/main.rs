//! `helix` — the HELIX Lite compiler driver.
//!
//! Subcommands:
//! - `run    <file.hx> [--backend interp|jit] [--unchecked]` — compile & execute
//! - `check  <file.hx>` — parse + semantic check only
//! - `dump   <stage> <file.hx>` — tokens | ast | ir | ssa | loops | all
//! - `bench  [--quick|--full] [--out dir]` — benchmark campaign
//! - `observe [--port N] [--no-open]` — the Observatory web UI
//! - `selftest` — differential gauntlet: interpreter vs JIT on every example

mod diag;

use std::path::PathBuf;

use helix_ir as hir;
use helix_syntax::parse_str;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = run(args);
    std::process::exit(code);
}

fn run(args: Vec<String>) -> i32 {
    let Some(cmd) = args.first() else {
        print_usage();
        return 2;
    };
    let rest = &args[1..];
    match cmd.as_str() {
        "run" => cmd_run(rest),
        "check" => cmd_check(rest),
        "dump" => cmd_dump(rest),
        "bench" => cmd_bench(rest),
        "observe" => cmd_observe(rest),
        "selftest" => cmd_selftest(rest),
        "--help" | "-h" | "help" => {
            print_usage();
            0
        }
        other => {
            eprintln!("unknown command '{other}'");
            print_usage();
            2
        }
    }
}

fn print_usage() {
    println!(
        "HELIX Lite — automatic parallelizing compiler\n\
         \n\
         USAGE:\n  \
         helix run     <file.hx> [--backend interp|jit] [--unchecked]\n  \
         helix check   <file.hx>\n  \
         helix dump    <tokens|ast|ir|ssa|loops|all> <file.hx>\n  \
         helix bench   [--quick|--full] [--out DIR]\n  \
         helix observe [--port N] [--no-open]\n  \
         helix selftest"
    );
}

struct Frontend {
    source: String,
    program: helix_sema::TypedProgram,
}

fn frontend(path: &str) -> Result<Frontend, i32> {
    let source = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read {path}: {e}");
            return Err(1);
        }
    };
    let parsed = match parse_str(&source) {
        Ok(p) => p,
        Err(e) => {
            let (span, msg) = match &e {
                helix_syntax::SyntaxError::Lex(l) => (l.span, l.msg.clone()),
                helix_syntax::SyntaxError::Parse(p) => (p.span, p.msg.clone()),
            };
            eprint!("{}", diag::render(&source, path, span, &msg));
            return Err(1);
        }
    };
    match helix_sema::check(&parsed) {
        Ok(tp) => Ok(Frontend { source, program: tp }),
        Err(diags) => {
            for d in &diags {
                eprint!("{}", diag::render(&source, path, d.span, &d.msg));
            }
            eprintln!("{} error(s)", diags.len());
            Err(1)
        }
    }
}

fn lower_all(program: &helix_sema::TypedProgram) -> Vec<hir::FuncIr> {
    let mut fns = hir::build(program);
    for f in &mut fns {
        hir::to_ssa(f);
    }
    fns
}

fn cmd_run(args: &[String]) -> i32 {
    let mut backend = "interp";
    let mut unchecked = false;
    let mut file = None;
    for a in args {
        match a.as_str() {
            "--backend" | "-b" => {}
            "interp" | "jit" => backend = a,
            "--unchecked" => unchecked = true,
            other if file.is_none() => file = Some(other.to_string()),
            _ => {}
        }
    }
    let Some(file) = file else {
        eprintln!("usage: helix run <file.hx> [--backend interp|jit]");
        return 2;
    };
    let Ok(fe) = frontend(&file) else { return 1 };

    match backend {
        "interp" => match helix_engine::run_with_source(&fe.source, &fe.program) {
            Ok(out) => {
                for line in out.printed {
                    println!("{line}");
                }
                0
            }
            Err(e) => {
                eprintln!("{}", e.render(&fe.source));
                1
            }
        },
        "jit" => {
            // Backend integration lands with Wave 3; keep the surface stable.
            eprintln!("error: jit backend not wired yet (pending helix-backend build)");
            let _ = (unchecked, lower_all(&fe.program));
            3
        }
        _ => unreachable!(),
    }
}

fn cmd_check(args: &[String]) -> i32 {
    let Some(file) = args.first() else {
        eprintln!("usage: helix check <file.hx>");
        return 2;
    };
    match frontend(file) {
        Ok(_) => {
            println!("ok");
            0
        }
        Err(code) => code,
    }
}

fn cmd_dump(args: &[String]) -> i32 {
    if args.len() < 2 {
        eprintln!("usage: helix dump <tokens|ast|ir|ssa|loops|all> <file.hx>");
        return 2;
    }
    let stage = args[0].as_str();
    let file = &args[1];
    let fe = match frontend_stage(file, stage) {
        Ok(fe) => fe,
        Err(code) => return code,
    };

    match stage {
        "tokens" => {
            let toks = helix_syntax::lex(&fe.source).expect("checked above");
            for t in &toks {
                let text = &fe.source[t.span.start as usize..t.span.end as usize];
                println!("{:>4}..{:<4} {:?} {text:?}", t.span.start, t.span.end, t.kind);
            }
        }
        "ast" => {
            let p = helix_syntax::parse_str(&fe.source).expect("checked above");
            println!("{}", p.print_tree());
        }
        "ir" | "ssa" | "loops" | "all" => {
            let fns = lower_all(&fe.program);
            if matches!(stage, "ir" | "all") {
                for f in &fns {
                    println!("==== {} (pre-SSA) ====", f.name);
                    println!("{}", hir::print_ir(f, false));
                }
            }
            if matches!(stage, "ssa" | "all") {
                for f in &fns {
                    println!("==== {} (SSA) ====", f.name);
                    println!("{}", hir::print_ir(f, true));
                }
            }
            if matches!(stage, "loops" | "all") {
                for f in &fns {
                    let li = helix_analysis::loops::find_loops(f);
                    let reports = helix_analysis::analyze(f, &li);
                    if reports.is_empty() {
                        continue;
                    }
                    println!("==== {} loop analysis ====", f.name);
                    for r in &reports {
                        println!("{}", r.summary_line());
                        for line in &r.accesses {
                            println!("    {line}");
                        }
                        for d in r.raw_deps.iter().chain(&r.war_deps).chain(&r.waw_deps) {
                            println!("    {}", d.explain);
                        }
                    }
                }
            }
        }
        other => {
            eprintln!("unknown stage '{other}'");
            return 2;
        }
    }
    0
}

/// Like frontend() but tolerates later-stage failures when dumping early stages.
fn frontend_stage(file: &str, stage: &str) -> Result<Frontend, i32> {
    let needs_full =
        matches!(stage, "ir" | "ssa" | "loops" | "all");
    let source = std::fs::read_to_string(file).map_err(|e| {
        eprintln!("error: cannot read {file}: {e}");
        1
    })?;
    let parsed = parse_str(&source).map_err(|e| {
        let (span, msg) = match &e {
            helix_syntax::SyntaxError::Lex(l) => (l.span, l.msg.clone()),
            helix_syntax::SyntaxError::Parse(p) => (p.span, p.msg.clone()),
        };
        eprint!("{}", diag::render(&source, file, span, &msg));
        1
    })?;
    if !needs_full && matches!(stage, "tokens" | "ast") {
        // A syntactic dump doesn't need sema.
        return Ok(Frontend { source, program: empty_program() });
    }
    match helix_sema::check(&parsed) {
        Ok(tp) => Ok(Frontend { source, program: tp }),
        Err(diags) => {
            for d in &diags {
                eprint!("{}", diag::render(&source, file, d.span, &d.msg));
            }
            Err(1)
        }
    }
}

fn empty_program() -> helix_sema::TypedProgram {
    helix_sema::TypedProgram { funcs: Vec::new(), consts: Vec::new() }
}

fn cmd_bench(_args: &[String]) -> i32 {
    eprintln!("bench campaign lands with Wave 3 (helix-bench)");
    3
}

fn cmd_observe(_args: &[String]) -> i32 {
    eprintln!("observatory server lands with Wave 3 (helix-observe)");
    3
}

fn cmd_selftest(_args: &[String]) -> i32 {
    eprintln!("selftest lands after jit wiring");
    3
}

#[allow(dead_code)]
fn unused_path_guard(_: PathBuf) {}
