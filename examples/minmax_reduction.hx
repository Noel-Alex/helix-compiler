// two reductions in one loop: min over array, max of squares
fn main() {
    let n = 16777216;
    let a: [f64] = zeros(n);
    let lo = 1.0e300;
    let hi = 0.0;
    for i in 0..n {
        let sq = a[i] * a[i];
        lo = min(lo, a[i]);
        hi = max(hi, sq);
    }
    print(lo);
    print(hi);
}
