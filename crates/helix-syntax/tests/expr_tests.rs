//! Expression-level parse tests: precedence, associativity, unary/cast
//! interaction, postfix forms and expression error positions.

use helix_syntax::{BinOp as B, Expr, Item, Span, Stmt, Type, UnOp, parse_str};

/// Parses `body_src` as the single statement of `fn t() { ... }`.
fn stmt_of(body_src: &str) -> Stmt {
    let src = format!("fn t() {{ {body_src} }}");
    let prog = parse_str(&src).unwrap_or_else(|e| panic!("parse failed: {e}"));
    match prog.items.first() {
        Some(Item::Fn(f)) => f.body.stmts.first().expect("a statement").clone(),
        other => panic!("unexpected shape: {other:?}"),
    }
}

/// The expression inside a bare `let _ = EXPR;`.
fn expr_of(src: &str) -> Expr {
    match stmt_of(&format!("let _t = {src};")) {
        Stmt::Let { init, .. } => init,
        other => panic!("expected let stmt, got {other:?}"),
    }
}

fn bin(e: &Expr) -> (B, &Expr, &Expr) {
    match e {
        Expr::Bin(op, l, r, _) => (*op, l.as_ref(), r.as_ref()),
        other => panic!("expected Bin, got {other:?}"),
    }
}

fn un(e: &Expr) -> (UnOp, &Expr) {
    match e {
        Expr::Unary(op, inner, _) => (*op, inner.as_ref()),
        other => panic!("expected Unary, got {other:?}"),
    }
}

// ------------------------------------------------------------------ literals

#[test]
fn int_float_bool_literals() {
    let e = expr_of("2.5");
    assert!(matches!(e, Expr::FloatLit(v, _) if v == 2.5));
    let e = expr_of("2e3");
    assert!(matches!(e, Expr::FloatLit(v, _) if v == 2000.0));
    assert!(matches!(expr_of("true"), Expr::Bool(true, _)));
    assert!(matches!(expr_of("false"), Expr::Bool(false, _)));
    // No negative literal exists: unary minus wraps the positive literal.
    let neg = expr_of("-7");
    let (op, inner) = un(&neg);
    assert_eq!(op, UnOp::Neg);
    assert!(
        matches!(inner, Expr::IntLit(7, _)),
        "unary minus wraps the literal"
    );
}

#[test]
fn literal_spans_are_exact() {
    // Inside `fn t() { let x = 123; }` the literal sits at bytes 17..20.
    let src = "fn t() { let x = 123; }";
    let p = parse_str(src).expect("parses");
    let Item::Fn(f) = p.items.first().unwrap() else {
        panic!()
    };
    match f.body.stmts.first() {
        Some(Stmt::Let {
            init: Expr::IntLit(_, s),
            ..
        }) => assert_eq!(*s, Span::new(17, 20)),
        other => panic!("{other:?}"),
    }
}

// ------------------------------------------------------- precedence ladder

#[test]
fn mul_binds_tighter_than_add_shape_1_plus_2_times_3() {
    // 1+2*3 == 1+(2*3)
    let e = expr_of("1 + 2 * 3");
    let (op, l, r) = bin(&e);
    assert_eq!(op, B::Add);
    assert!(matches!(l, Expr::IntLit(1, _)));
    let (op2, _, _) = bin(r);
    assert_eq!(op2, B::Mul);
}

#[test]
fn left_associativity_of_same_level() {
    // 1-2-3 == ((1-2)-3), NOT 1-(2-3)
    let e = expr_of("1 - 2 - 3");
    let (op, l, r) = bin(&e);
    assert_eq!(op, B::Sub);
    assert!(matches!(r, Expr::IntLit(3, _)));
    let (op2, ll, lr) = bin(l);
    assert_eq!(op2, B::Sub);
    assert!(matches!(ll, Expr::IntLit(1, _)));
    assert!(matches!(lr, Expr::IntLit(2, _)));

    // Division too: 100/10/5 == ((100/10)/5)
    let d = expr_of("100 / 10 / 5");
    let (_, l2, r2) = bin(&d);
    assert!(matches!(r2, Expr::IntLit(5, _)));
    let (o, _, _) = bin(l2);
    assert_eq!(o, B::Div);
}

#[test]
fn mixed_add_mul_left_assoc_chain() {
    // a + b * c - d  ==  ((a + (b*c)) - d)
    let e = expr_of("a + b * c - d");
    let (top, tl, tr) = bin(&e);
    assert_eq!(top, B::Sub);
    assert!(matches!(tr, Expr::Var(v) if v.name == "d"));
    let (mid, ml, mr) = bin(tl);
    assert_eq!(mid, B::Add);
    assert!(matches!(ml, Expr::Var(v) if v.name == "a"));
    let (inner, _, _) = bin(mr);
    assert_eq!(inner, B::Mul);
}

#[test]
fn rel_below_eq_and_logical_ladder() {
    // a < b == c && d || e
    // ladder: || lowest → ( (a<b == c) && d ) || e
    let e = expr_of("a < b == c && d || e");
    let (top, tl, tr) = bin(&e);
    assert_eq!(top, B::Or);
    assert!(matches!(tr, Expr::Var(v) if v.name == "e"));

    let (and_op, al, ar) = bin(tl);
    assert_eq!(and_op, B::And);
    assert!(matches!(ar, Expr::Var(v) if v.name == "d"));

    let (eq_op, el, er) = bin(al);
    assert_eq!(eq_op, B::Eq);
    let (lt_op, _, _) = bin(el);
    assert_eq!(lt_op, B::Lt);
    assert!(matches!(er, Expr::Var(v) if v.name == "c"));
}

#[test]
fn comparison_chains_parse_left_assoc() {
    // a < b < c parses ((a<b)<c); sema will reject bool<int later,
    // but syntactically it is a valid left-assoc chain.
    let e = expr_of("a < b < c");
    let (top, l, _) = bin(&e);
    assert_eq!(top, B::Lt);
    let (lo, _, _) = bin(l);
    assert_eq!(lo, B::Lt);
}

#[test]
fn rem_sits_with_mul_level() {
    let e = expr_of("a % b * c");
    let (top, l, r) = bin(&e);
    assert_eq!(top, B::Mul);
    assert!(matches!(r, Expr::Var(v) if v.name == "c"));
    let (lo, _, _) = bin(l);
    assert_eq!(lo, B::Rem);
}

#[test]
fn parens_override_precedence() {
    // (1 + 2) * 3 → Mul(Add, 3)
    let e = expr_of("(1 + 2) * 3");
    let (op, l, r) = bin(&e);
    assert_eq!(op, B::Mul);
    assert!(matches!(r, Expr::IntLit(3, _)));
    let (inner, _, _) = bin(l);
    assert_eq!(inner, B::Add);

    // -(1 + 2): parens make the whole sum the operand
    let neg_expr = expr_of("-(1 + 2)");
    let (neg, inner_e) = un(&neg_expr);
    assert_eq!(neg, UnOp::Neg);
    let (iop, _, _) = bin(inner_e);
    assert_eq!(iop, B::Add);
}

#[test]
fn nested_parens_grouping() {
    let e = expr_of("((a))");
    assert!(matches!(e, Expr::Var(v) if v.name == "a"));
}

// ------------------------------------------------------------ unary / cast

#[test]
fn double_negation_and_not_not() {
    let dd = expr_of("--x");
    let (op1, inner1) = un(&dd);
    assert_eq!(op1, UnOp::Neg);
    let (op2, _) = un(inner1);
    assert_eq!(op2, UnOp::Neg);

    let nn = expr_of("!!b");
    let (n1, n_inner) = un(&nn);
    assert_eq!(n1, UnOp::Not);
    let (n2, _) = un(n_inner);
    assert_eq!(n2, UnOp::Not);
}

#[test]
fn neg_cast_is_neg_of_cast_per_spec() {
    // "-x as i32" MUST be Neg(Cast(x)) because `as` binds tighter than unary.
    let src = "fn t() -> i32 { let y = 3; return -y as i32; }";
    let p = parse_str(src).expect("parses");
    let Item::Fn(f) = p.items.first().unwrap() else {
        panic!()
    };
    let Some(Stmt::Return { value: Some(e), .. }) = f.body.stmts.get(1) else {
        panic!("expected return as second statement");
    };
    let (op, inner) = un(e);
    assert_eq!(op, UnOp::Neg, "outer node must be Neg");
    match inner {
        Expr::Cast(castee, ty, _) => {
            assert_eq!(ty, &Type::I32);
            let castee = castee.as_ref();
            assert!(matches!(castee, Expr::Var(v) if v.name == "y"));
        }
        other => panic!("expected Cast under Neg, got {other:?}"),
    }
}

#[test]
fn neg_times_is_neg_of_a_times_b() {
    // -a*b == (-a)*b : unary binds tighter than binary minus/mul level.
    let e = expr_of("-a * b");
    let (op, l, r) = bin(&e);
    assert_eq!(op, B::Mul);
    assert!(matches!(r, Expr::Var(v) if v.name == "b"));
    let (uop, uinner) = un(l);
    assert_eq!(uop, UnOp::Neg);
    assert!(matches!(uinner, Expr::Var(v) if v.name == "a"));
}

#[test]
fn cast_chain_is_left_nested() {
    // x as i64 as f64 → Cast(Cast(x, i64), f64)
    let e = expr_of("x as i64 as f64");
    match e {
        Expr::Cast(outer, ty_out, _) => {
            assert_eq!(ty_out, Type::F64);
            match outer.as_ref() {
                Expr::Cast(inner, ty_in, _) => {
                    assert_eq!(*ty_in, Type::I64);
                    assert!(matches!(inner.as_ref(), Expr::Var(v) if v.name == "x"));
                }
                other => panic!("{other:?}"),
            }
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn cast_binds_tighter_than_mul_but_looser_than_postfix() {
    // a[i] as f64 + 1.0 → Add(Cast(Index(a,i)), 1.0)
    let e = expr_of("a[i] as f64 + 1.0");
    let (add, l, r) = bin(&e);
    assert_eq!(add, B::Add);
    assert!(matches!(r, Expr::FloatLit(v, _) if *v == 1.0));
    match l {
        Expr::Cast(castee, ty, _) => {
            assert_eq!(*ty, Type::F64);
            assert!(matches!(castee.as_ref(), Expr::Index(..)));
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn cast_target_must_be_scalar() {
    let msg = err_of_expr("x as [f64]");
    assert!(msg.contains("scalar"), "{msg}");
    let msg = err_of_expr("x as ()");
    assert!(msg.contains("scalar") || msg.contains("`as`"), "{msg}");
    let msg = err_of_expr("x as banana");
    assert!(msg.contains("scalar"), "{msg}");
}

#[test]
fn unary_applies_to_full_unary_chain_not_just_primary() {
    // -a + b == (-a) + b ; and !a && b == (!a) && b
    let e = expr_of("!a && b");
    let (and, l, r) = bin(&e);
    assert_eq!(and, B::And);
    assert!(matches!(r, Expr::Var(v) if v.name == "b"));
    let (nop, _) = un(l);
    assert_eq!(nop, UnOp::Not);
}

// ------------------------------------------------------------- postfix forms

#[test]
fn index_vs_binary_sub_disambiguation() {
    // a[i]-1 : Index then Sub, never something else.
    let e = expr_of("a[i] - 1");
    let (op, l, r) = bin(&e);
    assert_eq!(op, B::Sub);
    assert!(matches!(r, Expr::IntLit(1, _)));
    match l {
        Expr::Index(base, idx, _) => {
            assert_eq!(base.name, "a");
            assert!(matches!(idx.as_ref(), Expr::Var(v) if v.name == "i"));
        }
        other => panic!("expected Index on lhs, got {other:?}"),
    }

    // a[i-1] : the subtraction lives INSIDE the index.
    match expr_of("a[i-1]") {
        Expr::Index(base, idx, _) => {
            assert_eq!(base.name, "a");
            let (iop, il, ir) = bin(idx.as_ref());
            assert_eq!(iop, B::Sub);
            assert!(matches!(il, Expr::Var(v) if v.name == "i"));
            assert!(matches!(ir, Expr::IntLit(1, _)));
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn call_forms() {
    match expr_of("print(x)") {
        Expr::Call { callee, args, .. } => {
            assert_eq!(callee.name, "print");
            assert_eq!(args.len(), 1);
        }
        other => panic!("{other:?}"),
    }
    match expr_of("min(a, b)") {
        Expr::Call { callee, args, .. } => {
            assert_eq!(callee.name, "min");
            assert_eq!(args.len(), 2);
            assert!(matches!(args[1], Expr::Var(ref v) if v.name == "b"));
        }
        other => panic!("{other:?}"),
    }
    match expr_of("f()") {
        Expr::Call { callee, args, .. } => {
            assert_eq!(callee.name, "f");
            assert!(args.is_empty());
        }
        other => panic!("{other:?}"),
    }
    // Call arguments are full expressions incl. nested calls.
    match expr_of("f(g(1), a[i])") {
        Expr::Call { callee, args, .. } => {
            assert_eq!(callee.name, "f");
            assert_eq!(args.len(), 2);
            assert!(matches!(&args[0], Expr::Call { callee, .. } if callee.name == "g"));
            assert!(matches!(&args[1], Expr::Index(base, _, _) if base.name == "a"));
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn index_of_call_result_rejected() {
    let msg = err_of_expr("f(x)[0]");
    assert!(msg.contains("only one call or index level"), "{msg}");
}

// ------------------------------------------------------------------ helpers

fn err_of_expr(expr_src: &str) -> String {
    let src = format!("fn t() {{ let _t = {expr_src}; }}");
    match parse_str(&src) {
        Ok(_) => panic!("expected error for {expr_src:?}"),
        Err(e) => e.to_string(),
    }
}
