// scale: independent iterations -> SAFE parallel
fn main() {
    let n = 50000000;   // NOTE: underscores in literals not yet specced — see below
    let a: [f64] = zeros(n);
    let out: [f64] = zeros(n);
    for i in 0..n {
        out[i] = a[i] * 5.0;
    }
    print(out[42]);
}
