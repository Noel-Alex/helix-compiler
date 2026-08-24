// C = A*B with outer-i parallel and inner-k +-reduction
const N: i64 = 512;

fn main() {
    let nn = N * N;
    let a: [f64] = zeros(nn);
    let b: [f64] = zeros(nn);
    let c: [f64] = zeros(nn);
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
