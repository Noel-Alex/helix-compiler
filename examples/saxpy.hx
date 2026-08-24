// saxpy: y = s*x + y — memory-bound streaming kernel
fn main() {
    let n = 33554432;
    let x: [f64] = zeros(n);
    let y: [f64] = zeros(n);
    let s = 2.5;
    for i in 0..n {
        y[i] = s * x[i] + y[i];
    }
    print(y[7]);
}
