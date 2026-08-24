// a[2*i] vs a[i]: GCD test inconclusive, exact box-intersection decides (dependences exist)
fn main() {
    let n = 1000;
    let a: [i64] = zeros(n);
    for i in 1..n / 2 {
        a[2 * i] = a[i] + 1;
    }
    print(a[998]);
}
