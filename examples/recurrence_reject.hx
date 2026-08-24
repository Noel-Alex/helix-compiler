// recurrence: RAW dependence distance 1 -> MUST be rejected
fn main() {
    let n = 100000;
    let a: [i64] = zeros(n);
    a[0] = 1;
    for i in 1..n {
        a[i] = a[i - 1] + 1;
    }
    print(a[n - 1]);
}
