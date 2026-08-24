// level-2 carried dependence: inner loop over j reads a[i][j-1] written this i-iteration? no:
// writes row i, reads row i-1 => carried by OUTER loop; inner j is parallel.
fn main() {
    const R: i64 = 256;
    const C: i64 = 256;
    let n = R * C;
    let a: [f64] = zeros(n);
    for i in 1..R {
        for j in 0..C {
            a[i * C + j] = a[(i - 1) * C + j] * 2.0;
        }
    }
    print(a[255 * C + 255]);
}
