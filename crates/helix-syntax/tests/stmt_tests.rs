//! Statement- and program-level parse tests: every grammar production,
//! if/else-if chains, for headers, assignment forms, error positions, the
//! ASCII tree printer and serde round-trips.

use helix_syntax::{
    BinOp as B, ElsePart, Expr, Item, Program, Stmt, Type, UnOp, lex, parse, parse_str,
};

/// Parses a full program, panicking on failure.
fn ok(src: &str) -> Program {
    parse_str(src).unwrap_or_else(|e| panic!("parse failed for {src:?}: {e}"))
}

/// Expects failure; returns the message text.
fn err(src: &str) -> String {
    match parse_str(src) {
        Ok(_) => panic!("expected parse error for {src:?}"),
        Err(e) => e.to_string(),
    }
}

fn single_fn(src: &str) -> helix_syntax::FnDef {
    match ok(src).items.into_iter().next().expect("one item") {
        Item::Fn(f) => f,
        Item::Const(_) => panic!("expected fn"),
    }
}

// ------------------------------------------------------------------- consts

#[test]
fn const_defs_of_every_scalar_type() {
    let p = ok("const N: i64 = 1024;\n\
         const EPS: f64 = 0.0001;\n\
         const FLAG: bool = true;");
    assert_eq!(p.items.len(), 3);
    let Item::Const(c0) = p.items.first().unwrap() else {
        panic!()
    };
    assert_eq!(c0.name.name, "N");
    assert_eq!(c0.ty, Type::I64);
    assert!(matches!(c0.value, helix_syntax::Literal::Int(1024)));

    let Item::Const(c1) = p.items.get(1).unwrap() else {
        panic!()
    };
    assert_eq!(c1.ty, Type::F64);
    assert!(matches!(c1.value, helix_syntax::Literal::Float(f) if f == 0.0001));

    let Item::Const(c2) = p.items.get(2).unwrap() else {
        panic!()
    };
    assert!(matches!(c2.value, helix_syntax::Literal::Bool(true)));
}

#[test]
fn negative_const_literal_is_rejected_per_frozen_grammar() {
    // `literal` in the frozen EBNF has no sign; `-5` would need an
    // extension the spec does not grant.
    let msg = err("const NEG: i64 = -5;");
    assert!(msg.contains("literal"), "{msg}");
}

#[test]
fn const_requires_literal_initializer() {
    let msg = err("const N: i64 = 1 + 2;");
    assert!(msg.contains("literal"), "{msg}");

    // Non-literal initialisers fail the same way.
    assert!(err("const N: i64 = f();").contains("literal"));
    assert!(
        err("const N: i64 = x;").contains("literal") || err("const N: i64 = x;").contains("`x`")
    );
}

#[test]
fn const_requires_type_annotation_and_semi() {
    assert!(err("const N = 5;").contains("`:`"));
    assert!(err("const N: i64 = 5").contains(";"));
}

// --------------------------------------------------------------- fn headers

#[test]
fn params_and_array_types() {
    let f = single_fn("fn dot(a: [f64], b: [f64], n: i64) -> f64 { return 0.0; }");
    assert_eq!(f.params.len(), 3);
    assert_eq!(f.params[0].ty, Type::Array(helix_syntax::ScalarType::F64));
    assert_eq!(f.params[2].name.name, "n");
    assert_eq!(f.ret, Some(Type::F64));
    // Span covers `fn` through final `}`.
    assert_eq!(f.span.start, 0);
    assert_eq!(
        f.span.end as usize,
        src_len("fn dot(a: [f64], b: [f64], n: i64) -> f64 { return 0.0; }")
    );
}

#[test]
fn unit_return_type_written_explicitly() {
    let f = single_fn("fn noisy() -> () { return; }");
    assert_eq!(f.ret, Some(Type::Unit));
}

#[test]
fn nested_arrays_are_unrepresentable() {
    let msg = err("fn bad(a: [[i32]]) {}");
    assert!(msg.contains("]"), "{msg}");
}

// ---------------------------------------------------------------- statements

#[test]
fn let_forms() {
    let f = single_fn("fn t() { let x = 1; let y: f64 = 2.0; }");
    assert_eq!(f.body.stmts.len(), 2);
    match f.body.stmts.first() {
        Some(Stmt::Let { name, ty, .. }) => {
            assert_eq!(name.name, "x");
            assert!(ty.is_none());
        }
        other => panic!("{other:?}"),
    }
    match f.body.stmts.get(1) {
        Some(Stmt::Let { name, ty, .. }) => {
            assert_eq!(name.name, "y");
            assert_eq!(ty.as_ref(), Some(&Type::F64));
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn scalar_and_element_assignment() {
    let f = single_fn("fn t(a: [i64]) { a[0] = 7; b = a[i]; }");
    assert_eq!(f.body.stmts.len(), 2);
    match f.body.stmts.first() {
        Some(Stmt::Assign { target, value, .. }) => {
            assert_eq!(target.base.name, "a");
            assert!(target.index.is_some());
            assert!(matches!(value, Expr::IntLit(7, _)));
        }
        other => panic!("{other:?}"),
    }
    match f.body.stmts.get(1) {
        Some(Stmt::Assign { target, .. }) => {
            assert_eq!(target.base.name, "b");
            assert!(target.index.is_none());
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn empty_statement_and_nested_block_statements() {
    let f = single_fn("fn t() { ; { ; } }");
    match f.body.stmts.first() {
        Some(Stmt::Empty) => {}
        other => panic!("{other:?}"),
    }
    match f.body.stmts.get(1) {
        Some(Stmt::Block(b)) => {
            assert_eq!(b.stmts.len(), 1);
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn bare_expression_statement_is_a_call() {
    let f = single_fn("fn t() { print(x); }");
    match f.body.stmts.first() {
        Some(Stmt::Expr(Expr::Call { callee, .. })) => assert_eq!(callee.name, "print"),
        other => panic!("{other:?}"),
    }
}

// ------------------------------------------------------------------ if/else

#[test]
fn if_without_else() {
    let f = single_fn("fn t(x: bool) { if x { print(1); } }");
    match f.body.stmts.first() {
        Some(Stmt::If {
            cond,
            then_blk,
            else_part,
            span,
        }) => {
            assert!(matches!(cond, Expr::Var(_)));
            assert_eq!(then_blk.stmts.len(), 1);
            assert!(else_part.is_none());
            // The if-statement's span starts at `if` and covers the body.
            assert_eq!(span.start, 16); // position of `if` in `fn t(x: bool) { ... }`
            assert_eq!(span.end, then_blk.span.end);
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn if_with_else_block() {
    let f = single_fn("fn t(x: bool) { if x { print(1); } else { print(2); } }");
    match f.body.stmts.first() {
        Some(Stmt::If { else_part, .. }) => match else_part.as_deref() {
            Some(ElsePart::Block(b)) => assert_eq!(b.stmts.len(), 1),
            other => panic!("expected else block, got {other:?}"),
        },
        other => panic!("{other:?}"),
    }
}

#[test]
fn else_if_chain_nests_as_elsepart_if() {
    let src = r#"
        fn grade(s: i64) {
            if s > 90 {
                print(4);
            } else if s > 80 {
                print(3);
            } else if s > 70 {
                print(2);
            } else {
                print(0);
            }
        }
    "#;
    let f = single_fn(src);
    // Unwrap the spine: If → ElsePart::If(Box<If>) → ... → ElsePart::Block.
    let Some(Stmt::If { else_part: e1, .. }) = f.body.stmts.first() else {
        panic!("outer if missing");
    };
    let Some(ElsePart::If(inner1)) = e1.as_deref() else {
        panic!("second level must be an else-if");
    };
    let Stmt::If { else_part: e2, .. } = inner1.as_ref() else {
        panic!("else-if must wrap a Stmt::If");
    };
    let Some(ElsePart::If(inner2)) = e2.as_deref() else {
        panic!("third level must be an else-if");
    };
    let Stmt::If { else_part: e3, .. } = inner2.as_ref() else {
        panic!("else-if must wrap a Stmt::If");
    };
    match e3.as_deref() {
        Some(ElsePart::Block(b)) => assert_eq!(b.stmts.len(), 1),
        other => panic!("innermost should be a plain block, got {other:?}"),
    }

    // And the tree printer shows the chain as nested `If` spines under
    // `else` rows (three else rows: two else-ifs + one final block).
    let tree = single_fn_tree(src);
    let else_rows = tree
        .lines()
        .filter(|l| l.trim_start().starts_with("else"))
        .count();
    assert_eq!(else_rows, 3, "three else rows in the chain: {tree}");
}

#[test]
fn braces_are_mandatory_on_if_bodies() {
    assert!(err("fn t(x: bool) { if x print(1); }").contains("{"));
    assert!(err("fn t(x: bool) { if x } ").contains("{"));
    assert!(err("fn t(x: bool) { if x { } else print(2); }").contains("`if` or `{`"));
}

#[test]
fn dangling_else_attaches_to_nearest_if_by_nesting() {
    // With mandatory braces there IS no ambiguity; check the shape.
    let f = single_fn("fn t(a: bool, b: bool) { if a { if b { print(1); } } else { print(2); } }");
    match f.body.stmts.first() {
        Some(Stmt::If {
            then_blk,
            else_part,
            ..
        }) => {
            assert!(matches!(then_blk.stmts.first(), Some(Stmt::If { .. })));
            assert!(else_part.is_some(), "outer else belongs to OUTER if");
        }
        other => panic!("{other:?}"),
    }
}

// ---------------------------------------------------------------------- for

#[test]
fn for_header_shape_and_range_spanning_exprs() {
    let f = single_fn("fn t(n: i64) { for i in n - 1..n * 2 { acc = acc + i; } }");
    match f.body.stmts.first() {
        Some(Stmt::For {
            iv,
            start,
            end,
            body,
            ..
        }) => {
            assert_eq!(iv.name, "i");
            assert!(matches!(start, Expr::Bin(B::Sub, _, _, _)));
            assert!(matches!(end.as_ref(), Expr::Bin(B::Mul, _, _, _)));
            assert_eq!(body.stmts.len(), 1);
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn ranges_only_in_for_headers() {
    // `..` in an expression is a syntax error.
    let msg = err("fn t(n: i64) { let r = 1..2; }");
    assert!(msg.contains("`..`") || msg.contains(".."), "{msg}");
}

#[test]
fn for_body_braces_mandatory() {
    assert!(err("fn t(n: i64) { for i in 0..n print(i); }").contains("{"));
}

// ------------------------------------------------------------------- return

#[test]
fn return_with_and_without_value() {
    let f = single_fn("fn t() { return; }");
    assert!(matches!(
        f.body.stmts.first(),
        Some(Stmt::Return { value: None, .. })
    ));
    let f = single_fn("fn t() -> i64 { return 42; }");
    match f.body.stmts.first() {
        Some(Stmt::Return { value: Some(e), .. }) => {
            assert!(matches!(e, Expr::IntLit(42, _)));
        }
        other => panic!("{other:?}"),
    }
}

// ------------------------------------------------------------ reserved words

#[test]
fn reserved_words_rejected_as_names() {
    for word in [
        "while", "break", "continue", "mut", "by", "and", "or", "not", "struct", "import",
    ] {
        let src = format!("fn main() {{ let {word} = 1; }}");
        let msg = err(&src);
        assert!(msg.contains("reserved"), "`{word}`: {msg}");
    }
    assert!(err("fn while() { }").contains("reserved"));
    assert!(!err("while = 3;").is_empty());
}

// -------------------------------------------------------------- error positions

/// Unwraps a `SyntaxError` into its span/message, failing on success.
fn fail(src: &str) -> (helix_syntax::Span, String) {
    match parse_str(src) {
        Ok(_) => panic!("expected error for {src:?}"),
        Err(helix_syntax::SyntaxError::Parse(e)) => (e.span, e.msg),
        Err(helix_syntax::SyntaxError::Lex(e)) => (e.span, e.msg),
    }
}

#[test]
fn errors_report_expected_found_with_spans() {
    // Missing semicolon: error points at the `}`.
    let (span, msg) = fail("fn t() { let x = 1 }");
    assert!(msg.contains("expected"), "{msg}");
    assert!(msg.contains(';'), "{msg}");
    assert_eq!(span.start, 19); // position of `}`

    // Missing closing brace at EOF.
    let (span, _) = fail("fn t() { let x = 1;");
    assert!(span.start >= 19);

    // Bad top-level item.
    let (span, msg) = fail("let x = 1;");
    assert!(msg.contains("`fn` or `const`"), "{msg}");
    assert_eq!(span.start, 0);
}

#[test]
fn unclosed_paren_and_bracket_positions() {
    let (span, msg) = fail("fn t() { let y = (1 + 2; }");
    assert!(msg.contains(')'), "{msg}");
    assert_eq!(span.start, 23); // the `;`

    let (_, msg) = fail("fn t(a: [i64]) { a[0 = 1; }");
    assert!(msg.contains(']'), "{msg}");
}

#[test]
fn expression_cannot_start_a_statement_with_operator() {
    assert!(err("fn t() { * = 3; }").contains("expression"));
    assert!(err("fn t() { let x = *a; }").contains("expression"));
}

#[test]
fn assignment_is_not_an_expression() {
    // `a = b = c` must fail: after parsing `b`, parser sees `=` where
    // `;` was expected.
    let msg = err("fn t() { a = b = c; }");
    assert!(msg.contains(';'), "{msg}");

    // Assignment inside a condition is likewise impossible.
    let msg = err("fn t(x: bool) { if x = true { } }");
    assert!(msg.contains('{') || msg.contains("if"), "{msg}");
}

#[test]
fn chained_indexing_rejected_targeted() {
    let msg = err("fn t(a: [i64]) { a[0][1] = 2; }");
    assert!(
        msg.contains("named variables") || msg.contains(']'),
        "{msg}"
    );
}

#[test]
fn trailing_comma_in_params_or_args_rejected() {
    assert!(!err("fn t(a: i64,) { }").is_empty());
    assert!(!err("fn t() { print(1,); }").is_empty());
}

// ------------------------------------------------------------------ pipeline

#[test]
fn lex_then_parse_api_contract() {
    // The contracted API: lex(src) -> Vec<Token>, parse(tokens) -> Program.
    let toks = lex("fn main() { print(42); }").expect("lexes");
    let ends_in_eof = toks.last().expect("eof").is_eof();
    let prog = parse(toks).expect("parses");
    assert_eq!(prog.items.len(), 1);
    assert!(ends_in_eof);
}

#[test]
fn lexer_error_surfaces_through_parse_str() {
    match parse_str("fn main() { let x = /* oops }") {
        Err(e) => assert!(e.to_string().contains("unterminated"), "{e}"),
        Ok(_) => panic!("should fail"),
    }
}

// --------------------------------------------------------------- tree output

fn single_fn_tree(src: &str) -> String {
    single_fn_tree_of(&ok(src))
}

fn single_fn_tree_of(p: &Program) -> String {
    p.print_tree()
}

#[test]
fn tree_printer_structure() {
    let p = ok("const N: i64 = 8;\n\
         fn sum(a: [i64]) -> i64 {\n\
             let acc = 0;\n\
             for i in 0..N {\n\
                 acc = acc + a[i];\n\
             }\n\
             return acc;\n\
         }");
    let tree = p.print_tree();

    // Stable skeleton lines exist.
    assert!(tree.starts_with("Program\n"), "{tree}");
    assert!(tree.contains("|- Const N: i64 = 8"), "{tree}");
    assert!(tree.contains("`- Fn sum(a: [i64]) -> i64"), "{tree}");
    assert!(tree.contains("Block @"));
    assert!(tree.contains("Let acc"));
    assert!(tree.contains("For i in [start, end)"));
    assert!(tree.contains("Bin `+`"));
    assert!(tree.contains("Index a"));
    assert!(tree.contains("Return"));

    // Indentation grows monotonically with nesting depth.
    fn indent(line: &str) -> usize {
        line.len() - line.trim_start().len()
    }
    let let_line = tree.lines().find(|l| l.contains("Let acc")).expect("let");
    let lit_line = tree.lines().find(|l| l.contains("IntLit 0")).expect("lit");
    let for_line = tree.lines().find(|l| l.contains("For i in")).expect("for");
    let idx_line = tree.lines().find(|l| l.contains("Index a")).expect("idx");
    assert!(indent(lit_line) > indent(let_line), "{tree}");
    assert!(indent(idx_line) > indent(for_line), "{tree}");

    // Determinism: same input, same output.
    assert_eq!(tree, p.print_tree());
}

#[test]
fn tree_shows_else_chain_and_unary() {
    let p = ok("fn t(a: bool, b: bool, x: i64) {\n\
             if a { } else if b { } else { }\n\
             let y = -x as i64;\n\
             let z = !a && b;\n\
         }");
    let tree = p.print_tree();
    assert!(tree.contains("Unary - "), "{tree}");
    assert!(tree.contains("Cast as i64"), "{tree}");
    assert!(tree.contains("Unary ! "), "{tree}");
    assert!(tree.contains("Bin `&&`"), "{tree}");
    // The else-if chain appears as labelled rows under the outer If,
    // with a nested If spine for `else if b`.
    let else_rows = tree
        .lines()
        .filter(|l| l.trim_start().starts_with("else"))
        .count();
    assert_eq!(
        else_rows, 2,
        "two else rows (if-chain + final block): {tree}"
    );
}

#[test]
fn empty_program_tree_has_placeholder() {
    assert_eq!(ok("").print_tree(), "Program\n  `(empty)`\n");
}

// --------------------------------------------------------------------- serde

// The Observatory serializes whole ASTs and both error types to JSON, so the
// contract requires `Serialize + Deserialize` on every public type. This crate
// deliberately adds no test-time JSON dependency; instead we pin the trait
// bounds at compile time (a missing derive fails this build) and exercise
// clone/equality semantics that the artifacts rely on.
#[test]
fn every_contract_type_implements_serde_and_clone_eq_debug() {
    fn asserts<'de, T>()
    where
        T: serde::Serialize + serde::Deserialize<'de>,
    {
    }
    asserts::<Program>();
    asserts::<helix_syntax::Item>();
    asserts::<helix_syntax::FnDef>();
    asserts::<helix_syntax::ConstDef>();
    asserts::<helix_syntax::Param>();
    asserts::<helix_syntax::Ident>();
    asserts::<helix_syntax::Block>();
    asserts::<Stmt>();
    asserts::<ElsePart>();
    asserts::<helix_syntax::LValue>();
    asserts::<Expr>();
    asserts::<UnOp>();
    asserts::<B>();
    asserts::<helix_syntax::Literal>();
    asserts::<Type>();
    asserts::<helix_syntax::ScalarType>();
    asserts::<helix_syntax::Span>();
    asserts::<helix_syntax::Token>();
    asserts::<helix_syntax::TokKind>();
    asserts::<helix_syntax::LexError>();
    asserts::<helix_syntax::ParseError>();

    // Program additionally needs Clone/PartialEq/Debug for artifact diffs.
    fn clone_eq_debug<T: Clone + PartialEq + std::fmt::Debug>() {}
    clone_eq_debug::<Program>();
}

#[test]
fn ast_is_clonable_and_comparable_for_artifact_snapshots() {
    let src = "const N: i64 = 4;\n\
               fn main() {\n\
                   let a: [f64] = zeros(N);\n\
                   for i in 0..N {\n\
                       a[i] = i as f64;\n\
                   }\n\
                   if a[0] > 0.5 { print(a); } else { print(len(a)); }\n\
               }";
    let p = ok(src);
    let twin = p.clone();
    assert_eq!(p, twin);
}

// ------------------------------------------------------------------- helpers

fn src_len(s: &str) -> usize {
    s.len()
}
