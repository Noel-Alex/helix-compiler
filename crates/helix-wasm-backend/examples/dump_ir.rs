//! Dumps post-SSA optimized IR for every example (development aid).

use helix_ir::{build, run_optimization_pipeline, to_ssa};
use helix_sema::check;
use helix_syntax::parse_str;

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args
        .next()
        .unwrap_or_else(|| "N:\\projects\\Helix\\examples".to_string());
    let mut entries: Vec<_> = std::fs::read_dir(&dir)
        .expect("read examples")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "hx"))
        .collect();
    entries.sort();
    for path in entries {
        let src = std::fs::read_to_string(&path).expect("read");
        println!("======== {}", path.display());
        let ast = match parse_str(&src) {
            Ok(a) => a,
            Err(e) => {
                println!("  parse error: {e}");
                continue;
            }
        };
        let tp = match check(&ast) {
            Ok(t) => t,
            Err(_) => {
                println!("  sema rejected");
                continue;
            }
        };
        let mut funcs = build(&tp);
        for f in &mut funcs {
            to_ssa(f);
            run_optimization_pipeline(f);
            println!(
                "fn {} -> {:?} ({} blocks, locals {})",
                f.name,
                f.types.ret,
                f.blocks.len(),
                f.n_locals
            );
            for (bi, b) in f.blocks.iter().enumerate() {
                println!(" bb{}:", bi);
                for p in &b.phis {
                    print!("   phi v{} (var {}) <-", p.dst.0, p.var.0);
                    for (f, v) in &p.args {
                        print!(" [bb{}: v{}]", f.0, v.0);
                    }
                    println!();
                }
                for i in &b.insts {
                    println!("   {i:?}");
                }
                match &b.term {
                    helix_ir::Term::Jump(t, a) => {
                        let args = a.iter().map(|v| format!("v{}", v.0)).collect::<Vec<_>>();
                        println!("   jump bb{}({})", t.0, args.join(","));
                    }
                    helix_ir::Term::Branch { cond, t, f } => {
                        println!("   branch v{} ? bb{} : bb{}", cond.0, t.0, f.0);
                    }
                    helix_ir::Term::Return(v) => {
                        println!(
                            "   return {}",
                            v.map(|v| format!("v{}", v.0)).unwrap_or_default()
                        );
                    }
                }
            }
        }
    }
}
