// C = A*B, naive triple loop: outer i parallel (rows of C independent),
// inner k is a classic +-reduction into `acc`. i-k-j would be faster from
// stride-1 access alone; the plain i-j-k form is kept because it is THE
// canonical example of "one reduction loop nested inside a DOALL loop".
// Consts are top-level scalars per the frozen grammar. 512 here is the
// performance size; helix-bench rewrites N for correctness runs.
const N: i64 = 512;

fn main() {
    let nn = N * N;
    let a: [f64] = zeros(nn);
    let b: [f64] = zeros(nn);
    let c: [f64] = zeros(nn);
    // Deterministic init in ~[0,48) x {0.5} — normal-range magnitudes only.
    for i in 0..nn {
        a[i] = (i % 97) as f64 * 0.5;
        b[i] = ((i * 7) % 89) as f64 * 0.25;
    }
    for i in 0..N {
        let ibase = i * N;
        for j in 0..N {
            let acc = 0.0;
            for k in 0..N {
                acc = acc + a[ibase + k] * b[k * N + j];
            }
            c[ibase + j] = acc;
        }
    }
    print(c[N * (N / 2) + N / 2]);
}
