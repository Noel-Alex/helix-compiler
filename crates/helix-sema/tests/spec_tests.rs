//! Semantic-analysis conformance tests: every static rule from lang-spec.md.
use helix_sema::check;
use helix_syntax::parse_str;

fn ok(src: &str) -> helix_sema::TypedProgram {
    match parse_str(src) {
        Ok(p) => match check(&p) {
            Ok(tp) => tp,
            Err(diags) => panic!("expected OK, got diags: {diags:#?} for\n{src}"),
        },
        Err(e) => panic!("parse failed: {e:?} for\n{src}"),
    }
}

fn errs(src: &str) -> Vec<String> {
    let p = parse_str(src).expect("should parse");
    check(&p)
        .expect_err("expected diagnostics")
        .into_iter()
        .map(|d| d.msg)
        .collect()
}

fn has(diags: &[String], needle: &str) -> bool {
    diags.iter().any(|m| m.contains(needle))
}

#[test]
fn accepts_the_canonical_kernels() {
    ok(
        "fn main() { let n = 10; let a: [f64] = zeros(n); let out: [f64] = zeros(n); \
        for i in 0..n { out[i] = a[i] * 5.0; } print(out[42]); }",
    );
    ok("fn main() { let s = 0; for i in 1..100 { s = s + i; } print(s); }");
    ok("const N: i64 = 8; fn main() { let a: [i64] = zeros(N * N); print(a[N]); }");
    // recursion + early return + else-if
    ok(
        "fn fib(n: i64) -> i64 { if n < 2 { return n; } else if n < 15 { return fib(n-1) + fib(n-2); } \
        return fib(n - 3) + 2 * fib(n - 2) - fib(n - 4) + 4; } fn main() { print(fib(24)); }",
    );
}

#[test]
fn undeclared_and_duplicate_names() {
    assert!(has(&errs("fn main() { x = 3; }"), "undeclared variable"));
    assert!(has(
        &errs("fn main() { let x = 3; let x = 4; }"),
        "duplicate name"
    ));
    assert!(has(
        &errs("fn f(a: i64, a: i64) {} fn main() {}"),
        "duplicate name 'a'"
    ));
    assert!(has(
        &errs("fn main() {} fn main() {}"),
        "duplicate function 'main'"
    ));
    assert!(has(
        &errs("fn go() {} fn go() {} fn main() { go(); }"),
        "duplicate function 'go'"
    ));
}

#[test]
fn missing_main_and_bad_main_signature() {
    assert!(has(&errs("fn notmain() {}"), "no 'fn main()'"));
    assert!(has(&errs("fn main(n: i64) {}"), "must take no parameters"));
    assert!(has(
        &errs("fn main() -> i64 { return 1; }"),
        "must return unit"
    ));
}

#[test]
fn type_mismatches() {
    assert!(has(
        &errs("fn main() { let b: bool = 5; }"),
        "type mismatch: expected bool"
    ));
    assert!(has(
        &errs("fn main() { let x = 3.5 + y; }"),
        "undeclared variable 'y'"
    ));
    assert!(has(
        &errs("fn main() { let x = 3; let y = x + 1.5; }"),
        "compatible operands"
    ));
    assert!(has(
        &errs("fn main() { let c = 1 < 2; let d = c + 1; }"),
        "compatible operands"
    ));
    assert!(has(
        &errs("fn main() { if 3 { } }"),
        "condition must be bool"
    ));
}

#[test]
fn zero_implicit_coercions_but_literal_adaptation() {
    // int literal adapts into i32 slot
    ok("fn take(v: i32) -> i32 { return v; } fn main() { print(take(7)); }");
    // but a float does NOT coerce to int and vice versa
    assert!(has(
        &errs("fn take(v: i32) -> i32 { return v; } fn main() { print(take(7.5)); }"),
        "argument type mismatch: expected i32, found f64"
    ));
    assert!(has(
        &errs("fn main() { let n = 5; let f = n * 1.5; }"),
        "got i64 and f64"
    ));
}

#[test]
fn array_rules() {
    // whole-array value use rejected
    assert!(has(
        &errs("fn main() { let a: [i64] = zeros(4); let b = a; }"),
        "cannot be used as a value"
    ));
    assert!(has(
        &errs("fn main() { let a: [i64] = zeros(4); if a == a { } }"),
        "cannot be used as a value"
    ));
    // index typing + widening
    assert!(has(
        &errs("fn main() { let a: [i64] = zeros(4); a[true] = 1; }"),
        "integer"
    ));
    ok("fn main() { let a: [i64] = zeros(4); let i: i32 = 1; a[i] = 2; print(a[i]); }");
    // element store type must match
    assert!(has(
        &errs("fn main() { let a: [i64] = zeros(4); a[0] = 1.5; }"),
        "cannot store"
    ));
    // zeros needs annotation
    assert!(has(
        &errs("fn main() { let a = zeros(4); }"),
        "cannot infer element type"
    ));
    // len works on arrays only
    assert!(has(
        &errs("fn main() { let x = 3; print(len(x)); }"),
        "len() expects an array"
    ));
    ok("fn main() { let n = 7; let a: [i64] = zeros(n); print(len(a)); }");
}

#[test]
fn aliasing_rejected() {
    assert!(has(
        &errs(
            "fn f(x: [i64], y: [i64]) { x[0] = 1; } fn main() { let a: [i64] = zeros(4); f(a, a); }"
        ),
        "aliasing rejected"
    ));
    ok(
        "fn f(x: [i64], y: [i64]) { x[0] = 1; } fn main() { let a: [i64] = zeros(4); let b: [i64] = zeros(4); f(a, b); }",
    );
}

#[test]
fn loop_variable_is_immutable() {
    assert!(has(
        &errs("fn main() { for i in 0..10 { i = 5; } }"),
        "loop variable 'i'"
    ));
    // shadowing in nested scope is fine
    ok("fn main() { for i in 0..10 { } }");
}

#[test]
fn definite_assignment() {
    //  without initializer is a syntax error by construction (definite assignment)
    assert!(helix_syntax::parse_str("fn main() { let x: i64; }").is_err());
    // Mandatory initializers make use-before-init impossible BY CONSTRUCTION:
    // every binding is born initialized; self-reference is caught as undeclared.
    assert!(has(
        &errs("fn main() { let x = x + 1; }"),
        "undeclared variable 'x'"
    ));
    ok("fn main() { let c = 1 < 2; let x = 5; if c { x = 9; } else { x = 8; } print(x); }");
    ok(
        "fn f(c: bool) -> i64 { let v = 0; if c { v = 1; } return v; } fn main() { print(f(true)); }",
    );
    // loop body may assign then use within same iteration
    ok("fn main() { let s = 0; for i in 0..4 { let t = i * 2; s = s + t; } print(s); }");
}

#[test]
fn all_paths_must_return_values() {
    assert!(has(
        &errs("fn f(n: i64) -> i64 { if n < 2 { return 1; } } fn main() { print(f(3)); }"),
        "not all control-flow paths return"
    ));
    ok("fn f(n: i64) -> i64 { if n < 2 { return 1; } return 2; } fn main() { print(f(3)); }");
    // unit fns may fall off the end and use bare return;
    ok("fn p(x: i64) { if x > 0 { print(x); return; } print(-x); } fn main() { p(3); p(-3); }");
    // returning a value from unit fn is an error
    assert!(has(
        &errs("fn p() { return 3; } fn main() { p(); }"),
        "unit function"
    ));
    // returning nothing from value fn is an error
    assert!(has(
        &errs("fn f() -> i64 { return; } fn main() { print(f()); }"),
        "bare 'return;'"
    ));
}

#[test]
fn casts_and_builtins() {
    ok(
        "fn main() { let x = (3.7 as i64) % 3; print(x); print((0.0 / 0.0) as i64); \
        print(abs(-3.5)); print(sqrt(2.25)); print(min(2, 9)); print(max(2.0, 9.0)); }",
    );
    assert!(has(&errs("fn main() { print(sqrt(4)); }"), "sqrt expects"));
    assert!(has(&errs("fn main() { print(abs(true)); }"), "abs expects"));
    assert!(has(&errs("fn main() { print(max(1, 2.0)); }"), "same type"));
    assert!(has(
        &errs("fn main() { print(zeros(4)); }"),
        "cannot infer element type"
    ));
    assert!(has(
        &errs("fn main() { let x = true as i64; }"),
        "numeric types"
    ));
}

#[test]
fn consts_are_scalar_and_typed() {
    assert!(has(
        &errs("const A: [i64] = 0; fn main() {}"),
        "consts must have a scalar type"
    ));
    assert!(has(
        &errs("const A: i64 = 1.5; fn main() {}"),
        "does not match declared type"
    ));
    // negative consts are unrepresentable (literal-only grammar); positive ones work
    ok(
        "const A: i64 = 5; const B: f64 = 2.5; const C: bool = true; fn main() { print(A); print(B); print(C); }",
    );
    assert!(helix_syntax::parse_str("const A: i64 = -5; fn main() {}").is_err());
}

// ---------------------------------------------------------------------------
// 2026-08-25 review wave 2: checker-completeness regressions
// ---------------------------------------------------------------------------

#[test]
fn main_with_explicit_unit_return_is_accepted() {
    // `-> ()` is explicit unit — legal, same as any other fn. The old
    // syntactic check rejected it while accepting a bare `fn main()`.
    ok("fn main() -> () { print(1); }");
    // A genuinely wrong return type stays rejected.
    assert!(has(
        &errs("fn main() -> i64 { return 0; }"),
        "'main' must return unit"
    ));
}

#[test]
fn array_returning_functions_are_rejected() {
    // Arrays are never copied (spec); a fn returning one has no lowering and
    // used to pass sema while failing JIT compilation — an accepted-invalid
    // program. Both spellings must now be rejected at the source.
    assert!(has(
        &errs("fn make(n: i64) -> [i64] { return zeros(n); } fn main() { let a = make(3); }"),
        "cannot return an array"
    ));
}

#[test]
fn builtin_names_cannot_be_redefined_as_functions() {
    // Every call site resolves to the builtin, so a user definition was
    // silently uncallable. Reject at the definition instead.
    assert!(has(
        &errs("fn len(a: [i64]) -> i64 { return 0; } fn main() {}"),
        "is a builtin and cannot be redefined"
    ));
    assert!(has(
        &errs("fn print(x: i64) {} fn main() {}"),
        "is a builtin and cannot be redefined"
    ));
}

// ---------------------------------------------------------------------------
// 2026-08-25 review wave 3: diagnostic-matrix gaps
// ---------------------------------------------------------------------------

#[test]
fn definite_assignment_both_branches_then_use() {
    // `let x;` never parses (initializers are mandatory, pinned above), so
    // the observable contract is: assigning in BOTH branches satisfies the
    // join and the post-if read is well-typed.
    ok("fn main() { let c = 1 < 2; let x: i64 = 0; \
         if c { x = 1; } else { x = 2; } print(x); }");
}

#[test]
fn all_paths_return_else_if_chains_and_nested_blocks() {
    // An else-if spine whose LAST link lacks an else can fall through.
    assert!(has(
        &errs(
            "fn f(n: i64) -> i64 { if n < 2 { return 1; } else if n < 4 { return 2; } } \
             fn main() { print(f(3)); }"
        ),
        "not all control-flow paths return"
    ));
    // Same chain plus a trailing return accepts.
    ok(
        "fn f(n: i64) -> i64 { if n < 2 { return 1; } else if n < 4 { return 2; } return 3; } \
         fn main() { print(f(3)); }",
    );
    // A bare nested block guarantees a return exactly when its contents do.
    ok("fn f(n: i64) -> i64 { { return n; } } fn main() { print(f(3)); }");
}

#[test]
fn constants_are_read_only() {
    // Companion to loop-variable immutability (same check_assign walk).
    assert!(has(
        &errs("const N: i64 = 5; fn main() { N = 6; }"),
        "cannot assign to constant 'N'"
    ));
}

#[test]
fn i32_literal_adaptation_at_boundary() {
    // i32::MAX adapts into an annotated i32 slot...
    ok(
        "fn take(v: i32) -> i32 { return v; } fn main() { let x: i32 = 2147483647; print(take(x)); }",
    );
    // ...but MAX+1 is rejected at the literal, not silently wrapped.
    assert!(has(
        &errs("fn main() { let x: i32 = 2147483648; }"),
        "integer literal 2147483648 does not fit in i32"
    ));
    // The same range check guards consts.
    assert!(has(
        &errs("const C: i32 = 2147483648; fn main() {}"),
        "integer literal 2147483648 does not fit in i32"
    ));
    // Unannotated literals stay i64: no annotation means no adaptation.
    ok("fn main() { let x = 2147483648; print(x); }");
}

#[test]
fn array_index_must_be_integer_in_every_position() {
    // Read position.
    assert!(has(
        &errs("fn main() { let a: [i64] = zeros(4); print(a[1.5]); }"),
        "array index must be an integer, found f64"
    ));
    // Store position.
    assert!(has(
        &errs("fn main() { let a: [f64] = zeros(4); a[0.5] = 1.0; }"),
        "array index must be an integer, found f64"
    ));
    // i32 indices widen implicitly (the spec's single coercion) — accepted.
    ok("fn main() { let a: [i64] = zeros(4); let i: i32 = 2; print(a[i]); }");
}

#[test]
fn call_arity_mismatch_rejected() {
    assert!(has(
        &errs("fn add(a: i64, b: i64) -> i64 { return a + b; } fn main() { print(add(1)); }"),
        "function 'add' expects 2 argument(s), got 1"
    ));
    assert!(has(
        &errs("fn add(a: i64, b: i64) -> i64 { return a + b; } fn main() { print(add(1, 2, 3)); }"),
        "function 'add' expects 2 argument(s), got 3"
    ));
    // Correct arity accepts.
    ok("fn add(a: i64, b: i64) -> i64 { return a + b; } fn main() { print(add(1, 2)); }");
    // Builtins enforce arity too.
    assert!(has(
        &errs("fn main() { print(len()); }"),
        "builtin 'len' expects 1 argument(s), got 0"
    ));
}

// ---------------------------------------------------------------------------
// 2026-08-25 review wave 3: literal-condition all-paths-return regressions
// ---------------------------------------------------------------------------

#[test]
fn literal_true_then_no_return_else_returns_is_rejected() {
    // `if true` executes ONLY the then-branch; the else arm is dead code, so
    // its `return` cannot save the function. (Inverted condition used to
    // accept this and the IR builder masked it with a synthesized zero.)
    assert!(has(
        &errs(
            "fn f() -> i64 { if true { let x = 1; } else { return 7; } } fn main() { print(f()); }"
        ),
        "not all control-flow paths return"
    ));
}

#[test]
fn literal_true_then_returns_else_not_is_accepted() {
    // Dead else arm may omit the return — only the then-branch runs.
    ok("fn f() -> i64 { if true { return 7; } else { let x = 1; } } fn main() { print(f()); }");
}

#[test]
fn literal_false_branch_skipped_correctly() {
    // `if false` executes ONLY the else arm: accepted when it returns...
    ok(
        "fn f(n: i64) -> i64 { if false { print(n); } else { return n; } } fn main() { print(f(3)); }",
    );
    // ...and rejected when neither branch does.
    assert!(has(
        &errs(
            "fn f() -> i64 { if false { let x = 1; } else { let y = 2; } } fn main() { print(f()); }"
        ),
        "not all control-flow paths return"
    ));
}

#[test]
fn nested_else_if_with_literal_conditions_rejected_when_fallthrough_exists() {
    // Outer cond is not a literal, so BOTH the then-branch and the whole
    // else-if spine must return. The inner `if true` link falls through, so
    // the chain does not guarantee a return.
    assert!(has(
        &errs(
            "fn f(p: bool) -> i64 { if p { return 1; } else if true { let x = 2; } } \
             fn main() { print(f(true)); }"
        ),
        "not all control-flow paths return"
    ));
    // Same shape but with the dead-spine link returning: accepts.
    ok(
        "fn f(p: bool) -> i64 { if p { return 1; } else if true { return 2; } } \
         fn main() { print(f(true)); }",
    );
}
