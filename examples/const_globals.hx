// top-level consts + shadowing
const N: i64 = 8;

fn main() {
    let n = N * N;
    let a: [i64] = zeros(n);
    for i in 0..n {
        a[i] = i % N;
    }
    let sum = 0;
    for i in 0..n {
        sum = sum + a[i];
    }
    print(sum);
}
