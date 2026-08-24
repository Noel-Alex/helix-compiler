// 5-point Jacobi stencil on a flattened SIZE x SIZE grid: each interior
// element becomes the average of its four neighbours. Rows of `next` are
// independent within a sweep (the previous ROW is read, not the next one),
// so the j-loop is DOALL and the nest is a level-2-parallel showcase.
// Consts are top-level scalars per the frozen grammar; sizes are kept small
// here so the example doubles as a correctness case (see helix-bench for the
// 4096x4096 performance variant).
const SIZE: i64 = 32;
const ITER: i64 = 4;

fn main() {
    let n = SIZE * SIZE;
    let cur: [f64] = zeros(n);
    let next: [f64] = zeros(n);
    // Hot spot at the centre; heat diffuses outward one row/col per sweep.
    cur[SIZE * (SIZE / 2) + SIZE / 2] = 256.0;
    for k in 0..ITER {
        for i in 1..SIZE - 1 {
            let base = i * SIZE;
            for j in 1..SIZE - 1 {
                next[base + j] = 0.25 * (cur[base + j - 1] + cur[base + j + 1]
                                       + cur[base - SIZE + j] + cur[base + SIZE + j]);
            }
        }
        // Swap-free double buffer copy: next -> cur (DOALL over elements).
        for i in 0..n {
            cur[i] = next[i];
        }
    }
    print(cur[SIZE * (SIZE / 2) + SIZE / 2]);
}
