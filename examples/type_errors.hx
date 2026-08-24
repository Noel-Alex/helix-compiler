// intentionally broken: exercises sema diagnostics (used by tests, never run)
fn main() {
    let x = 3.5 + y;          // undeclared variable
    let b: bool = 5;          // type mismatch
    let a: [i64] = zeros(10);
    a[true] = 2;              // bad index type
    let c = a + a;            // arrays not first-class values
}
