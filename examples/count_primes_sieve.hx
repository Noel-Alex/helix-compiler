// Sieve of Eratosthenes: the OUTER loop is inherently sequential (it reads
// composite[i] and branches), but each INNER sweep only WRITES composite at
// strided indices and never reads them, so its iterations are independent --
// a textbook "outer serial, inner DOALL" nest. HELIX's frozen grammar has no
// step clause, so the stride is expressed as a contiguous driver k with the
// affine subscript start + k*i (j from 2*i step i, rewritten).
// Correctness anchor: pi(100) = 25 primes below 100.
fn main() {
    let n = 100;
    let composite: [bool] = zeros(n);
    let count = 0;
    for i in 2..n {
        if !composite[i] {
            count = count + 1;
            let start = i + i;
            let sweeps = (n - start + i - 1) / i;
            for k in 0..sweeps {
                composite[start + k * i] = true;
            }
        }
    }
    print(count);
}
