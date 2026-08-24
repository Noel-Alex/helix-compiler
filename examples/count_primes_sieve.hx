// sieve of Eratosthenes: interesting because the INNER loop has WAR/WAW-free structure
fn main() {
    let n = 10000000;
    let composite: [bool] = zeros(n);
    let count = 0;
    for i in 2..n {
        if !composite[i] {
            count = count + 1;
            for j in i + i..n {
                composite[j] = true;
            }
        }
    }
    print(count);
}
