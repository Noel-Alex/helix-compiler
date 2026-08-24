// dot product: +-reduction over products of two arrays
fn main() {
    let n = 33554432;
    let a: [f64] = zeros(n);
    let b: [f64] = zeros(n);
    let dot = 0.0;
    for i in 0..n {
        dot = dot + a[i] * b[i];
    }
    print(dot);
}
