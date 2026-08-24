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
        Some("run") => cmd_run(args.get(1)),
        Some("check") => cmd_check(args.get(1)),
        Some("dump") => match (args.get(1), args.get(2)) {
            (Some(stage), Some(file)) => cmd_dump(stage, file),
            _ => usage("dump requires <stage> <file>"),
        },
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

fn cmd_run(path: Option<&String>) -> i32 {
    let Some(path) = path else {
        return usage("run requires <file>");
    };
    let src = std::fs::read_to_string(PathBuf::from(path)).unwrap_or_default();
    let program = match frontend(path) {
        Ok(p) => p,
        Err(code) => return code,
    };
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
fn print_diag(src: &str, span: Span, msg: &str) {
    let line_no = src[..span.start as usize].matches('\n').count() + 1;
    let line_start = src[..span.start as usize].rfind('\n').map_or(0, |i| i + 1);
    let line_end = src[span.start as usize..]
        .find('\n')
        .map_or(src.len(), |i| span.start as usize + i);
    let line = &src[line_start..line_end];
    let caret_col = span.start.saturating_sub(line_start as u32);
    let caret_len = (span.end.saturating_sub(span.start)).max(1);

    eprintln!("--> line {line_no}: {msg}");
    eprintln!("{line}");
    eprintln!(
        "{}{}",
        " ".repeat(caret_col as usize),
        "^".repeat(caret_len as usize)
    );
}
