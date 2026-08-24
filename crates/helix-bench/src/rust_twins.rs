//! Hand-written Rust release twins of the numeric kernels.
//!
//! Purpose (methodology rec. 10): anchor the report. A tree-walking
//! interpreter landing 10–150x behind *compiled* code is expected
//! (Crafting Interpreters' jlox/clox gap); what validates the Cranelift
//! pipeline is HELIX-native landing **within ~2x** of plain idiomatic Rust.
//! These twins are that yardstick — written exactly as a practitioner would,
//! no `unsafe`, no SIMD, release-profile assumptions only.
//!
//! Every twin returns the same checksum the HELIX program's printed lines
//! encode: the twins hash their result through the same FNV-1a stream as
//! `helix-engine`'s `print` (`fmt_f64` = Rust `Debug` formatting, newline
//! terminated), so `twin_checksum == interpreter_checksum` is a real
//! differential assertion, not an eyeball comparison. See [`printed_checksum`].

use helix_sema::fmt_f64;

/// FNV-1a over canonical printed lines (+`\n` each) — byte-identical to
/// `helix_engine`'s print hashing so checksums are directly comparable.
#[must_use]
pub fn printed_checksum(lines: &[String]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for line in lines {
        for b in line.bytes().chain(std::iter::once(b'\n')) {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

// ---------------------------------------------------------------------------
// Input generators (shared shape with the HELIX sources)
// ---------------------------------------------------------------------------

/// The deterministic init used by every HELIX streaming/reduction kernel:
/// all-zeros arrays. Kept here for documentation; twins allocate identically.
#[must_use]
pub fn zeros(n: usize) -> Vec<f64> {
    vec![0.0; n]
}

/// Matmul init mirroring `matmul.hx`: `a[i]=(i%97)*0.5`, `b[i]=((i*7)%89)*0.25`.
#[must_use]
pub fn matmul_inputs(n: usize) -> (Vec<f64>, Vec<f64>) {
    let nn = n * n;
    let mut a = vec![0.0f64; nn];
    let mut b = vec![0.0f64; nn];
    for (i, slot) in a.iter_mut().enumerate() {
        *slot = ((i % 97) as f64) * 0.5;
    }
    for (i, slot) in b.iter_mut().enumerate() {
        *slot = (((i * 7) % 89) as f64) * 0.25;
    }
    (a, b)
}

// ---------------------------------------------------------------------------
// Twins
// ---------------------------------------------------------------------------

/// Twin of `saxpy.hx`: `y[i] += s*x[i]`, over SEEDED inputs (`x=1.0`,
/// `y=2.0`) so every store is live and no pass can fold the loop away.
/// Returns the rewritten `y`.
#[must_use]
pub fn saxpy(n: usize, s: f64) -> Vec<f64> {
    let x = vec![1.0f64; n];
    let mut y = vec![2.0f64; n];
    for i in 0..n {
        y[i] += s * x[i];
    }
    std::hint::black_box(&x);
    y
}

/// Twin of `dot_reduction.hx` over two seeded vectors (zeros would make the
/// sum trivially 0 and let a broken kernel pass vacuously).
#[must_use]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut acc = 0.0f64;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}

/// Twin of `matmul.hx` at edge `n` (naive i-j-k, identical order so FP
/// rounding matches the HELIX program bit-for-bit). Returns `c[centre]`.
#[must_use]
pub fn matmul_centre(n: usize) -> f64 {
    let (a, b) = matmul_inputs(n);
    let mut c = vec![0.0f64; n * n];
    for i in 0..n {
        let ib = i * n;
        for j in 0..n {
            let mut acc = 0.0f64;
            for k in 0..n {
                acc += a[ib + k] * b[k * n + j];
            }
            c[ib + j] = acc;
        }
    }
    c[n * (n / 2) + n / 2]
}

/// Convenience: checksum of one f64 printed the way HELIX prints it.
#[must_use]
pub fn checksum_f64(v: f64) -> u64 {
    printed_checksum(&[fmt_f64(v)])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn saxpy_math_is_elementwise() {
        let y = saxpy(1024, 2.5);
        assert_eq!(y.len(), 1024);
        // Seeded x=1, y=2 => y' = 2.5*1 + 2 = 4.5 everywhere.
        assert!(y.iter().all(|&v| v == 4.5));
    }

    #[test]
    fn dot_matches_closed_form_for_seeded_inputs() {
        // a[i] = 1, b[i] = 2 => sum = 2n exactly in f64 up to huge n.
        let n = 4096;
        let a = vec![1.0; n];
        let b = vec![2.0; n];
        assert_eq!(dot(&a, &b), 8192.0);
        // Non-trivial case vs direct computation.
        let c: Vec<f64> = (0..1000).map(|i| f64::from(i % 7)).collect();
        let d: Vec<f64> = (0..1000).map(|i| f64::from(i % 11)).collect();
        let expect: f64 = (0..1000)
            .map(|i| f64::from(i % 7) * f64::from(i % 11))
            .sum();
        assert_eq!(dot(&c, &d), expect);
    }

    #[test]
    fn matmul_twin_agrees_with_the_interpreted_example() {
        // examples/matmul.hx at N=8 prints 1626.625 (pinned by helix-bench's
        // parity test); the twin must produce the identical value because the
        // summation ORDER matches.
        assert_eq!(matmul_centre(8), 1626.625);
    }

    #[test]
    fn printed_checksum_is_fnv_and_terminator_sensitive() {
        let one_line = vec!["12".to_string()];
        let split = vec!["1".to_string(), "2".to_string()];
        assert_ne!(printed_checksum(&one_line), printed_checksum(&split));
        // Empty input yields the FNV offset basis.
        assert_eq!(printed_checksum(&[]), 0xcbf2_9ce4_8422_2325);
        // Deterministic.
        assert_eq!(
            printed_checksum(&["36.0".to_string()]),
            printed_checksum(&["36.0".to_string()])
        );
        assert_eq!(checksum_f64(36.0), printed_checksum(&["36.0".to_string()]));
    }

    #[test]
    fn matmul_inputs_match_source_formulas() {
        let (a, b) = matmul_inputs(16);
        // a[i] = (i % 97) * 0.5 — index 97 wraps the modulus to 0.
        assert_eq!(a[1], 0.5);
        assert_eq!(a[96], 48.0);
        assert_eq!(a[97], 0.0);
        // b[i] = ((i*7) % 89) * 0.25 — index 13 gives 91 % 89 = 2.
        assert_eq!(b[7], (49.0f64 % 89.0) * 0.25);
        assert_eq!(b[13], 2.0 * 0.25);
        // i=89: 623 % 89 == 0.
        assert_eq!(b[89], 0.0);
        assert_eq!(a.len(), 256);
    }

    #[test]
    fn zeros_are_zeros() {
        assert_eq!(zeros(10), vec![0.0; 10]);
        assert!(zeros(0).is_empty());
    }
}
