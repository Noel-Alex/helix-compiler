// honest case where threading LOSES: tiny trip count
fn main() {
    let n = 1000;
    let a: [f64] = zeros(n);
    let out: [f64] = zeros(n);
    for i in 0..n {
        out[i] = a[i] + 1.0;
    }
    print(out[999]);
}
