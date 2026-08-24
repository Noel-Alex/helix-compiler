// checked division semantics demo
fn main() {
    let a = -7;
    let b = 2;
    print(a % b);   // -1 : sign follows dividend (srem)
    print(7 % -2);  // 1
    print(a / b);   // -3 (truncating)
}
