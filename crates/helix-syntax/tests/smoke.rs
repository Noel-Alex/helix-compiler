//! End-to-end smoke test: a representative kernel parses, prints and counts.

use helix_syntax::parse_str;

#[test]
fn representative_kernel_parses_and_prints() {
    let src = r#"
        // matvec-style kernel
        const N: i64 = 4;
        fn matvec(a: [f64], m: [f64], n: i64) -> f64 {
            let acc = 0.0;
            for i in 0..n {
                if a[i] > 0.0 {
                    acc = acc + a[i] * m[i];
                } else {
                    acc = acc - a[i] as f64;
                }
            }
            return sqrt(acc);
        }
        fn main() {
            let v: [f64] = zeros(N);
            print(matvec(v, v, N));
        }
    "#;
    // Lex+parse must succeed.
    let p = parse_str(src).unwrap_or_else(|e| panic!("kernel failed to parse: {e}"));

    // Flat symbol table: 2 fns + 1 const.
    assert_eq!(p.items.len(), 3);
    assert_eq!(p.fns().count(), 2);
    assert_eq!(p.consts().count(), 1);
    assert_eq!(p.fns().next().expect("matvec").name.name, "matvec");

    // Tree printer output covers the constructs used.
    let tree = p.print_tree();
    for needle in [
        "Const N",
        "Fn matvec",
        "Fn main()",
        "For i in",
        "If @",
        "Index a",
        "Call sqrt",
    ] {
        assert!(tree.contains(needle), "tree missing `{needle}`:\n{tree}");
    }

    // The whole tree is span-annotated: every FnDef spans its full source.
    let matvec = p.fns().next().expect("matvec");
    assert!(matvec.span.end > matvec.span.start);
}
