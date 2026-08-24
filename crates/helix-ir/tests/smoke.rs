use helix_ir::{build, print_ir, run_passes_to_fixpoint, to_ssa, verify};

fn compile(src: &str) -> Vec<helix_ir::FuncIr> {
    let ast = helix_syntax::parse_str(src).expect("parse");
    let typed = helix_sema::check(&ast).expect("sema");
    build(&typed)
}

#[test]
fn ssa_demo_shape() {
    let src = r#"
        fn main() {
            let x = 5;
            let cond = 1 < 2;
            if cond {
                x = 10;
            }
            print(x);
        }
    "#;
    let irs = compile(src);
    let mut f = irs.into_iter().next().unwrap();
    println!("=== PRE-SSA ===\n{}", print_ir(&f, false));
    to_ssa(&mut f);
    println!("=== SSA ===\n{}", print_ir(&f, true));
    verify(&f).unwrap_or_else(|e| panic!("verify failed: {e}"));
}

#[test]
fn for_loop_shape() {
    let src = r#"
        fn main() {
            let a: [i64] = zeros(10);
            for i in 0..10 {
                a[i] = i * 2;
            }
        }
    "#;
    let mut f = compile(src).into_iter().next().unwrap();
    println!("=== FOR PRE-SSA ===\n{}", print_ir(&f, false));
    to_ssa(&mut f);
    println!("=== FOR SSA ===\n{}", print_ir(&f, true));
    verify(&f).unwrap_or_else(|e| panic!("verify failed: {e}"));
}

#[test]
fn shortcircuit_shape() {
    let src = r#"
        fn main() {
            let a: [i64] = zeros(5);
            let r = len(a) > 1 && len(a) < 4;
            print(r);
        }
    "#;
    let mut f = compile(src).into_iter().next().unwrap();
    println!("=== SC PRE-SSA ===\n{}", print_ir(&f, false));
    to_ssa(&mut f);
    println!("=== SC SSA ===\n{}", print_ir(&f, true));
    verify(&f).unwrap_or_else(|e| panic!("verify failed: {e}"));
}

#[test]
fn pipeline_end_to_end() {
    let src = r#"
        fn main() {
            let a: [i64] = zeros(8);
            for i in 0..8 {
                a[i] = i * 2 + 3 * 4;
            }
            print(len(a));
        }
    "#;
    for f in compile(src) {
        let mut f = f;
        to_ssa(&mut f);
        verify(&f).unwrap_or_else(|e| panic!("pre-pipeline: {e}"));
        let reports = run_passes_to_fixpoint(&mut f);
        for r in &reports {
            if r.changed {
                println!("--- {} ---\n{}", r.pass.name(), r.after);
            }
        }
        verify(&f).unwrap_or_else(|e| panic!("post-pipeline: {e}"));
    }
}
