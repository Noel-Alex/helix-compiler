// 5-point Jacobi stencil, flattened 2D: inner loop rows are independent (level-2 parallel)
const SIZE: i64 = 4096;
const ITER: i64 = 10;

fn main() {
    let n = SIZE * SIZE;
    let cur: [f64] = zeros(n);
    let next: [f64] = zeros(n);
    for k in 0..ITER {
        for i in 1..SIZE - 1 {
            let base = i * SIZE;
            for j in 1..SIZE - 1 {
                next[base + j] = 0.25 * (cur[base + j - 1] + cur[base + j + 1]
                                       + cur[base - SIZE + j] + cur[base + SIZE + j]);
            }
        }
        for i in 0..n {
            cur[i] = next[i];
        }
    }
    print(cur[SIZE * (SIZE / 2) + SIZE / 2]);
}
