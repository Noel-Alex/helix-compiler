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

/// The deterministic init of `saxpy.hx`: sign/fraction-mixed values so no
/// execution pass can fold the kernel or pass parity vacuously.
/// `x[i] = ((i*17+3) % 251)/17 - 4`, `y[i] = ((i*29+11) % 241)/19 - 5`.
#[must_use]
pub fn saxpy_inputs(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut x = vec![0.0f64; n];
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        x[i] = ((i * 17 + 3) % 251) as f64 / 17.0 - 4.0;
        y[i] = ((i * 29 + 11) % 241) as f64 / 19.0 - 5.0;
    }
    (x, y)
}

/// The deterministic init of `scale.hx`:
/// `a[i] = ((i*13+5) % 199)/7 - 12`.
#[must_use]
pub fn scale_inputs(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| ((i * 13 + 5) % 199) as f64 / 7.0 - 12.0)
        .collect()
}

/// The deterministic init of `dot_reduction.hx`:
/// `a[i] = ((i*7+1) % 97)/9 - 4`, `b[i] = ((i*11+2) % 89)/11 - 3`. Products
/// cancel in aggregate, so a zero-initialized or half-seeded dot cannot pass.
#[must_use]
pub fn dot_inputs(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = vec![0.0f64; n];
    let mut b = vec![0.0f64; n];
    for i in 0..n {
        a[i] = ((i * 7 + 1) % 97) as f64 / 9.0 - 4.0;
        b[i] = ((i * 11 + 2) % 89) as f64 / 11.0 - 3.0;
    }
    (a, b)
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

/// Twin of `saxpy.hx`: `y[i] += s*x[i]` over the seeded [`saxpy_inputs`] so
/// every store is live and no pass can fold the loop away. Returns the
/// rewritten `y`.
#[must_use]
pub fn saxpy(n: usize, s: f64) -> Vec<f64> {
    let (x, mut y) = saxpy_inputs(n);
    for i in 0..n {
        y[i] += s * x[i];
    }
    std::hint::black_box(&x);
    y
}

/// Twin of `dot_reduction.hx` over two seeded vectors (the HELIX sources'
/// formulas make the sum a cancelling mix — zeros would let a broken kernel
/// pass vacuously).
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
        let n = 1024;
        let (x, y0) = saxpy_inputs(n);
        let y = saxpy(n, 2.5);
        assert_eq!(y.len(), n);
        for i in 0..n {
            assert_eq!(y[i], 2.5 * x[i] + y0[i], "element {i}");
        }
    }

    #[test]
    fn input_generators_are_nonzero_and_sign_mixed() {
        let gens: Vec<Box<dyn Fn(usize) -> Vec<f64>>> = vec![
            Box::new(|n| saxpy_inputs(n).0),
            Box::new(|n| saxpy_inputs(n).1),
            Box::new(scale_inputs),
            Box::new(|n| dot_inputs(n).0),
            Box::new(|n| dot_inputs(n).1),
        ];
        for make in &gens {
            let v = make(512);
            assert!(v.iter().any(|&x| x > 0.0), "no positive values");
            assert!(v.iter().any(|&x| x < 0.0), "no negative values");
            assert!(v.iter().any(|&x| x != x.trunc()), "no fractional values");
            // The vacuity this module exists to prevent:
            assert!(v.iter().any(|&x| x != 0.0), "all-zero seed");
        }
    }

    #[test]
    fn dot_matches_closed_form_for_seeded_inputs() {
        // Non-trivial case vs direct computation.
        let c: Vec<f64> = (0..1000).map(|i| f64::from(i % 7)).collect();
        let d: Vec<f64> = (0..1000).map(|i| f64::from(i % 11)).collect();
        let expect: f64 = (0..1000)
            .map(|i| f64::from(i % 7) * f64::from(i % 11))
            .sum();
        assert_eq!(dot(&c, &d), expect);
    }

    // -- Oracle pins ---------------------------------------------------------
    //
    // These three values are EXACTLY what kernels.rs expects at its
    // correctness sizes; they are computed here independently of the HELIX
    // toolchain (plain Rust loops over the same formulas) so a drift in
    // either side fails loudly.

    #[test]
    fn oracle_saxpy_y7_at_n1024() {
        let n = 1024;
        let (x, mut y) = saxpy_inputs(n);
        for i in 0..n {
            y[i] = 2.5 * x[i] + y[i];
        }
        assert_eq!(fmt_f64(y[7]), "14.204334365325078");
    }

    #[test]
    fn oracle_scale_out42_at_n1000() {
        let a = scale_inputs(1000);
        let out42 = a[42] * 5.0;
        assert_eq!(fmt_f64(out42), "49.28571428571429");
    }

    #[test]
    fn oracle_dot_sum_at_n4096() {
        let n = 4096;
        let (a, b) = dot_inputs(n);
        let mut acc = 0.0f64;
        for i in 0..n {
            acc += a[i] * b[i];
        }
        assert_eq!(fmt_f64(acc), "5206.808080808095");
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
    fn input_generators_match_source_formulas() {
        let n = 300; // spans several modulus wrap points of every formula
        let (x, y) = saxpy_inputs(n);
        let a = scale_inputs(n);
        let (da, db) = dot_inputs(n);
        for i in [0usize, 1, 14, 17, 97, 199, 240, 250, 299] {
            let f = i as f64;
            assert_eq!(x[i], ((f * 17.0 + 3.0) % 251.0) / 17.0 - 4.0, "x[{i}]");
            assert_eq!(y[i], ((f * 29.0 + 11.0) % 241.0) / 19.0 - 5.0, "y[{i}]");
            assert_eq!(a[i], ((f * 13.0 + 5.0) % 199.0) / 7.0 - 12.0, "a[{i}]");
            assert_eq!(da[i], ((f * 7.0 + 1.0) % 97.0) / 9.0 - 4.0, "dot_a[{i}]");
            assert_eq!(db[i], ((f * 11.0 + 2.0) % 89.0) / 11.0 - 3.0, "dot_b[{i}]");
        }
    }
}
