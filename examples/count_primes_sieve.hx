// Sieve of Eratosthenes: the OUTER loop is inherently sequential (it reads
// composite[i] and branches). Each INNER sweep only WRITES composite at
// strided indices, so its iterations are independent in principle — but the
// stride multiplier `i` is loop-invariant and cannot be proven non-zero, so
// HELIX soundly REFUSES the sweep: if the stride were ever zero, every
// iteration would alias the same element. Recovering this loop needs range
// reasoning over the outer induction variable, which the analysis does not
// yet do. (Run `helix loops examples/count_primes_sieve.hx` to see the
// refusal and its reason.) HELIX's frozen grammar has no step clause, so the
// stride is expressed as a contiguous driver k with the affine subscript
// start + k*i (j from 2*i step i, rewritten).
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
