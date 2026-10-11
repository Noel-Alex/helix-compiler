//! Temporary diagnostic walker for a few programs' operator streams.

use helix_ir::FuncIr;
use helix_syntax::parse_str;
use helix_wasm_backend::emit_wasm;
use wasmparser::{Parser, Payload, Validator};

fn lower(source: &str) -> Vec<FuncIr> {
    let program = parse_str(source).unwrap();
    let typed = helix_sema::check(&program).unwrap();
    let mut funcs = helix_ir::build(&typed);
    for function in &mut funcs {
        helix_ir::to_ssa(function);
        helix_ir::run_optimization_pipeline(function);
    }
    funcs
}

fn dump_source(source: &str) {
    eprintln!("######## {source}");
    let funcs = lower(source);
    for function in &funcs {
        eprintln!(
            "--- fn {} ({} blocks)",
            function.name,
            function.blocks.len()
        );
        for (bi, block) in function.blocks.iter().enumerate() {
            eprintln!("  bb{bi}: preds {:?} succs {:?}", block.preds, block.succs);
            for p in &block.phis {
                eprintln!("    phi v{} (var {}) {:?}", p.dst.0, p.var.0, p.args);
            }
            for i in &block.insts {
                eprintln!("    {i:?}");
            }
            eprintln!("    term {:?}", block.term);
        }
    }
    let module = emit_wasm(&funcs).expect("emit succeeds");
    match Validator::new().validate_all(&module.bytes) {
        Ok(_) => eprintln!("validates"),
        Err(error) => eprintln!("INVALID at {}: {error}", error.offset()),
    }
    for payload in Parser::new(0).parse_all(&module.bytes) {
        match payload {
            Ok(Payload::CodeSectionEntry(body)) => {
                let range = body.range();
                eprintln!("--- code body at {range:?}");
                for operator in body
                    .get_operators_reader()
                    .expect("ops")
                    .into_iter_with_offsets()
                {
                    match operator {
                        Ok((op, offset)) => eprintln!("{:>4} {op:?}", offset),
                        Err(error) => {
                            eprintln!("OPERATOR ERROR: {error}");
                            break;
                        }
                    }
                }
            }
            Ok(_) => {}
            Err(error) => {
                eprintln!("PARSE ERROR: {error}");
                break;
            }
        }
    }
}

#[test]
fn dump() {
    dump_source(
        "fn poke(a: [i64], v: i64) { a[0] = v; }\nfn main() { let h: [i64] = zeros(5); poke(h, 3); print(h[0]); }\n",
    );
    dump_source(
        "fn side(v: i64, hit: [i64]) -> bool { hit[v] = hit[v] + 1; return v == 3; }\nfn main() { let hits: [i64] = zeros(5); let r = side(1, hits) && side(3, hits); print(r); print(hits[1]); }\n",
    );
}
