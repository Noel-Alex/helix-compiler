//! The affine dependence battery. Per subscript-dimension, cheapest test first;
//! the first test that PROVES independence kills the pair's dependence entirely.
//! Surviving tests refine distance/direction; anything unproven is conservative `*`.
//!
//! All arithmetic in i128 (LLVM widens similarly — the ANALYZER must not overflow
//! even when the program's own i64 math does).
//!
//! References: Goff/Kennedy/Tseng PLDI'91 (the SIV battery); Allen & Kennedy ch.2/8;
//! LLVM DependenceAnalysis.cpp (same scheme, same return conventions).

use serde::{Deserialize, Serialize};

/// One dimension of a subscript: value = a*i + b (i = the loop index).
/// Coefficients already normalized to this form by access extraction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Affine {
    pub a: i128,
    pub b: i128,
}

impl Affine {
    pub const fn const_val(v: i64) -> Self {
        Affine { a: 0, b: v as i128 }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DirVec {
    pub lt: bool,
    pub eq: bool,
    pub gt: bool,
}

impl DirVec {
    pub fn star() -> Self {
        DirVec {
            lt: true,
            eq: true,
            gt: true,
        }
    }

    pub fn exact(dir: Dir) -> Self {
        match dir {
            Dir::Lt => DirVec {
                lt: true,
                eq: false,
                gt: false,
            },
            Dir::Eq => DirVec {
                lt: false,
                eq: true,
                gt: false,
            },
            Dir::Gt => DirVec {
                lt: false,
                eq: false,
                gt: true,
            },
        }
    }

    pub fn is_star(&self) -> bool {
        self.lt && self.eq && self.gt
    }

    /// Merge (intersect) with another direction set.
    pub fn intersect(&mut self, other: &DirVec) {
        self.lt &= other.lt;
        self.eq &= other.eq;
        self.gt &= other.gt;
    }

    pub fn describe(&self) -> String {
        let mut s = String::new();
        if self.lt {
            s.push('<');
        }
        if self.eq {
            s.push('=');
        }
        if self.gt {
            s.push('>');
        }
        s
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dir {
    Lt,
    Eq,
    Gt,
}

/// Outcome of testing one dependence pair across all dimensions.
#[derive(Clone, Debug, PartialEq)]
pub enum DepOutcome {
    /// Proven independent — no dependence can exist.
    Independent,
    /// A dependence may exist; refined description.
    Dependence {
        /// Exact distance per dimension when fully determined (all '=' or single dir).
        distance: Option<i64>,
        dirs: Vec<DirVec>,
    },
}

/// Loop bounds for the analyzed dimension: iterations i ∈ [lo, hi].
#[derive(Clone, Copy, Debug)]
pub struct IterRange {
    pub lo: i128,
    pub hi: i128,
}

/// Test ONE dimension of a subscript pair.
///
/// Battery order: ZIV → Strong SIV → Weak-Zero SIV → Weak-Crossing SIV →
/// gcd+box (general two-variable Diophantine with bound intersection).
///
/// `range` constrains both source and sink iterations of THIS loop level.
pub fn test_dimension(src: Affine, dst: Affine, range: IterRange) -> DepOutcome {
    // ---- ZIV: no index variable on either side ------------------------------
    if src.a == 0 && dst.a == 0 {
        return if src.b == dst.b {
            DepOutcome::Dependence {
                distance: Some(0),
                dirs: vec![DirVec::exact(Dir::Eq)],
            }
        } else {
            DepOutcome::Independent
        };
    }

    // ---- Strong SIV: same coefficient on both sides --------------------------
    if src.a == dst.a && src.a != 0 {
        let delta = dst.b - src.b; // dst(i') = src(i)  =>  a*i' + q = a*i + p
        if delta % src.a != 0 {
            return DepOutcome::Independent;
        }
        let d = delta / src.a; // distance = sink - source = i' - i
        // d must be representable and the iteration pair in-range.
        if d > i64::MAX as i128 || d < i64::MIN as i128 {
            return DepOutcome::Independent; // unreachable pair within any real trip count
        }
        let dv = i64::try_from(d).expect("checked above");
        // Bounds: source at i, sink at i + d; both must lie in [lo, hi] for some i.
        let feasible = pairs_feasible(range, d);
        if !feasible {
            return DepOutcome::Independent;
        }
        let dir = match d.cmp(&0) {
            std::cmp::Ordering::Less => DirVec::exact(Dir::Lt),
            std::cmp::Ordering::Equal => DirVec::exact(Dir::Eq),
            std::cmp::Ordering::Greater => DirVec::exact(Dir::Gt),
        };
        return DepOutcome::Dependence {
            distance: Some(dv),
            dirs: vec![dir],
        };
    }

    // ---- Weak-Zero SIV: one side has zero coefficient ------------------------
    if src.a == 0 || dst.a == 0 {
        let (a, p, c, _swap) = if dst.a == 0 {
            // dst constant c = src(i): single point i = (c - b_src)/a_src
            (src.a, src.b, dst.b, false)
        } else {
            // src constant p = dst(i'): i' = (p - b_dst)/a_dst
            (dst.a, dst.b, src.b, true)
        };
        debug_assert!(a != 0);
        let diff = c - p;
        if diff % a != 0 {
            return DepOutcome::Independent;
        }
        let point = diff / a;
        if point < range.lo || point > range.hi {
            return DepOutcome::Independent;
        }
        // The fixed iteration touches the same location from one side only:
        // direction relative to the *variable* side. For our purposes the
        // dependence is loop-independent ('=') when both sides refer to that point,
        // which they do by construction.
        return DepOutcome::Dependence {
            distance: None,
            dirs: vec![DirVec {
                lt: true,
                eq: true,
                gt: true,
            }],
        };
    }

    // ---- Weak-Crossing SIV: coefficients are negatives of each other ----------
    if src.a == -dst.a {
        // src(i) = a*i + p, dst(j) = -a*j + q  =>  a*(i + j) = q - p.
        let a = src.a;
        debug_assert!(a != 0);
        let num = dst.b - src.b;
        if num % a != 0 {
            return DepOutcome::Independent;
        }
        let sum = num / a; // i + j must equal `sum`
        // Feasible when the box admits two values adding to `sum`:
        // max(lo, sum - hi) <= min(hi, sum - lo).
        if range.lo.max(sum - range.hi) <= range.hi.min(sum - range.lo) {
            // '<' and '>' possible; '=' only when sum is even and sum/2 in
            // range. Conservative: allow all three directions.
            return DepOutcome::Dependence {
                distance: None,
                dirs: vec![DirVec::star()],
            };
        }
        return DepOutcome::Independent;
    }

    // ---- General case: gcd test + bounded box intersection --------------------
    // Solve a_s*i - a_d*j = b_d - b_s  over the box [lo,hi]^2.
    let lhs_a = src.a;
    let rhs_a = dst.a;
    let k = dst.b - src.b;
    let g = gcd(lhs_a.abs(), rhs_a.abs());
    if g == 0 || k % g != 0 {
        return DepOutcome::Independent;
    }
    // Parametric solution via extended Euclid on |coefficients| with sign fixups:
    // find one solution (i0, j0) to lhs_a*i - rhs_a*j = k.
    if let Some((i0, j0)) = solve_diophantine(lhs_a, rhs_a, k) {
        // General solution: i = i0 + (rhs_a/g)*t, j = j0 + (lhs_a/g)*t.
        let si = rhs_a / g;
        let sj = lhs_a / g;
        if box_has_solution(range, i0, j0, si, sj) {
            return DepOutcome::Dependence {
                distance: None,
                dirs: vec![DirVec::star()],
            };
        }
        return DepOutcome::Independent;
    }
    // Unsolvable Diophantine equation => independent.
    DepOutcome::Independent
}

fn gcd(a: i128, b: i128) -> i128 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Find one integer solution (i, j) to `lhs_a*i - rhs_a*j = k`, if any exists.
fn solve_diophantine(lhs_a: i128, rhs_a: i128, k: i128) -> Option<(i128, i128)> {
    // lhs_a*i ≡ k (mod rhs_a) — reduce by gcd (already checked divisible).
    // Extended Euclid on (lhs_a, rhs_a).
    let (mut old_r, mut r) = (lhs_a, rhs_a);
    let (mut old_s, mut s) = (1i128, 0i128);
    while r != 0 {
        let q = old_r / r;
        (old_r, r) = (r, old_r - q * r);
        (old_s, s) = (s, old_s - q * s);
    }
    // old_r = gcd, old_s = coefficient for lhs_a: lhs_a*old_s ≡ gcd (mod rhs_a)
    let gg = old_r;
    if gg == 0 {
        return None;
    }
    if k % gg != 0 {
        return None;
    }
    let scale = k / gg;
    // lhs_a * (old_s*scale) - rhs_a * j = k has solution i = old_s*scale.
    let i0 = old_s * scale;
    // Corresponding j from the equation itself.
    let j_num = lhs_a * i0 - k;
    debug_assert!(j_num % rhs_a == 0);
    let j0 = j_num / rhs_a;
    Some((i0, j0))
}

/// Does the parametric family (i,j) = (i0 + si*t, j0 + sj*t) intersect [lo,hi]^2?
fn box_has_solution(range: IterRange, i0: i128, j0: i128, si: i128, sj: i128) -> bool {
    // Need t with lo <= i0 + si*t <= hi and lo <= j0 + sj*t <= hi.
    let lo = range.lo;
    let hi = range.hi;
    // Feasible-t interval per dimension. A negative step flips the inequality
    // directions; normalizing the sign BEFORE dividing keeps the bounds ordered
    // (the previous sign-branch produced [upper, lower] pairs and reported
    // genuinely dependent pairs as independent — see the negative_step tests).
    let iv = |step: i128, c: i128| -> Option<(i128, i128)> {
        if step == 0 {
            return if c >= lo && c <= hi {
                Some((i128::MIN / 4, i128::MAX / 4))
            } else {
                None
            };
        }
        // lo <= c + step*t <= hi  ⇔  tmin_raw/|step| <= t <= tmax_raw/|step|
        // with the raw numerator endpoints swapped when step < 0.
        let (tmin_raw, tmax, s) = if step > 0 {
            (lo - c, hi - c, step)
        } else {
            (c - hi, c - lo, -step)
        };
        let lower = ceil_div(tmin_raw, s);
        let upper = floor_div(tmax, s);
        if lower <= upper {
            Some((lower, upper))
        } else {
            None
        }
    };

    match (iv(si, i0), iv(sj, j0)) {
        (Some((a1, a2)), Some((b1, b2))) => a2 >= b1 && b2 >= a1,
        _ => false,
    }
}

fn floor_div(a: i128, b: i128) -> i128 {
    a.div_euclid(b)
}

fn ceil_div(a: i128, b: i128) -> i128 {
    -(-a).div_euclid(b)
}

/// Brute-force ground truth for small boxes: does ANY (i,j) in [lo,hi]^2 with
/// src(i) == dst(j) exist? Used by property tests against the algebra.
#[cfg(test)]
fn box_brute_force(range: IterRange, src: Affine, dst: Affine) -> bool {
    for i in range.lo..=range.hi {
        for j in range.lo..=range.hi {
            if src.a * i + src.b == dst.a * j + dst.b {
                return true;
            }
        }
    }
    false
}

fn pairs_feasible(range: IterRange, d: i128) -> bool {
    // Exists i with i in [lo,hi] and i+d in [lo,hi].
    let lo = range.lo.max(range.lo - d);
    let hi = range.hi.min(range.hi - d);
    // i ranges over [max(lo, lo-d), min(hi, hi-d)] ∩ original bounds handled above:
    // need max(lo, lo - d) <= min(hi, hi - d)? Careful: i in [lo,hi] AND i+d in [lo,hi]
    // means i in [lo, hi] ∩ [lo-d, hi-d].
    let l = lo.max(range.lo);
    let h = hi.min(range.hi);
    let _ = (l, h);
    let i_lo = range.lo.max(range.lo - d);
    let i_hi = range.hi.min(range.hi - d);
    i_lo <= i_hi
}

// ---------------------------------------------------------------------------
// Multi-dimension driver
// ---------------------------------------------------------------------------

/// Test all dimensions; combine per the conjoin rule: ANY dimension proving
/// independence kills the dependence; surviving dimensions intersect directions.
pub fn test_pair(
    subscripts_src: &[Affine],
    subscripts_dst: &[Affine],
    range: IterRange,
) -> DepOutcome {
    debug_assert_eq!(subscripts_src.len(), subscripts_dst.len());
    let mut combined_dirs: Vec<DirVec> = Vec::new();
    let mut distance: Option<i64> = Some(0);

    for (s, d) in subscripts_src.iter().zip(subscripts_dst.iter()) {
        match test_dimension(*s, *d, range) {
            DepOutcome::Independent => return DepOutcome::Independent,
            DepOutcome::Dependence {
                distance: dist,
                dirs,
            } => {
                if combined_dirs.is_empty() {
                    combined_dirs = dirs;
                } else {
                    for (cd, nd) in combined_dirs.iter_mut().zip(&dirs) {
                        cd.intersect(nd);
                    }
                }
                distance = match (distance, dist) {
                    (Some(acc), Some(x)) => Some(acc + x), // 1-D nesting: distances add
                    _ => None,
                };
            }
        }
    }

    if combined_dirs.iter().all(|d| d.lt || d.eq || d.gt) && combined_dirs.is_empty() {
        // No dimensions at all — treat as dependent conservatively? Empty subscript
        // lists never reach here (access pairing guarantees >= 1).
        return DepOutcome::Dependence {
            distance: None,
            dirs: vec![DirVec::star()],
        };
    }
    DepOutcome::Dependence {
        distance,
        dirs: combined_dirs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const R: IterRange = IterRange { lo: 0, hi: 99 };

    fn outcome_is_independent(src: Affine, dst: Affine) -> bool {
        matches!(test_dimension(src, dst, R), DepOutcome::Independent)
    }

    #[test]
    fn ziv_cases() {
        assert!(outcome_is_independent(
            Affine { a: 0, b: 3 },
            Affine { a: 0, b: 5 }
        ));
        assert!(!outcome_is_independent(
            Affine { a: 0, b: 3 },
            Affine { a: 0, b: 3 }
        ));
    }

    #[test]
    fn strong_siv_distance_one() {
        // a[i-1] vs a[i]: src coeff 1 b -1; dst coeff 1 b 0 => distance = (0-(-1))/1 = 1
        match test_dimension(Affine { a: 1, b: -1 }, Affine { a: 1, b: 0 }, R) {
            DepOutcome::Dependence {
                distance: Some(1),
                dirs,
            } => {
                assert!(dirs[0].gt && !dirs[0].lt && !dirs[0].eq);
            }
            other => panic!("expected distance-1 dependence, got {other:?}"),
        }
    }

    #[test]
    fn strong_siv_proves_independence_on_nondivisible() {
        // 2i vs 2i+1: delta = 1 not divisible by 2 -> independent
        assert!(outcome_is_independent(
            Affine { a: 2, b: 0 },
            Affine { a: 2, b: 1 }
        ));
    }

    #[test]
    fn weak_zero_hits_single_point() {
        // src a[i], dst a[7]: single point i=7 in range -> dependence exists
        assert!(!outcome_is_independent(
            Affine { a: 1, b: 0 },
            Affine { a: 0, b: 7 }
        ));
        // but a[200] out of range -> independent
        assert!(outcome_is_independent(
            Affine { a: 1, b: 0 },
            Affine { a: 0, b: 200 }
        ));
    }

    #[test]
    fn weak_crossing() {
        // src a[i], dst a[-i + 100]: crossing at i=50 in range
        assert!(!outcome_is_independent(
            Affine { a: 1, b: 0 },
            Affine { a: -1, b: 100 }
        ));
        // a[i] vs a[-i + 300]: crossing at 150 out of [0,99]
        assert!(outcome_is_independent(
            Affine { a: 1, b: 0 },
            Affine { a: -1, b: 300 }
        ));
    }

    #[test]
    fn weak_crossing_even_coefficient_solutions() {
        // Regression: the old crossing test divided by 2a instead of a, so
        // |a| > 1 pairs with even sums were wrongly declared independent.
        // src = -2i - 1, dst = 2j - 3 over [0,12]: sum i+j = (-3+1)/-2 = 1,
        // solutions (0,1),(1,0) — dependent.
        assert!(!outcome_is_independent(
            Affine { a: -2, b: -1 },
            Affine { a: 2, b: -3 }
        ));
        // Same coefficients but no in-range pair adds to the sum:
        // src = -2i (a=-2,b=0), dst = 2j + 50 over [0,12]: i+j = 25 needs
        // one side >= 13 — out of range. Independent.
        assert!(outcome_is_independent(
            Affine { a: -2, b: 0 },
            Affine { a: 2, b: 50 }
        ));
    }

    #[test]
    fn gcd_box_general_case() {
        // a[2i] vs a[i]: 2i - j = 0, gcd(2,1)=1 divides 0 -> solutions exist (i=j even)
        assert!(!outcome_is_independent(
            Affine { a: 2, b: 0 },
            Affine { a: 1, b: 0 }
        ));
        // a[2i] vs a[2j+1]: 2i - 2j = 1, gcd 2 does not divide 1 -> independent
        assert!(outcome_is_independent(
            Affine { a: 2, b: 0 },
            Affine { a: 2, b: 1 }
        ));
    }

    #[test]
    fn negative_range_guards() {
        // a[i-100] vs a[i] over 200 trips: distance-100 pairs exist (i >= 100)
        match test_dimension(
            Affine { a: 1, b: -100 },
            Affine { a: 1, b: 0 },
            IterRange { lo: 0, hi: 199 },
        ) {
            DepOutcome::Dependence {
                distance: Some(100),
                ..
            } => {}
            other => panic!("expected distance 100, got {other:?}"),
        }
        // Over exactly 100 trips [0,99] the reader needs iteration >= 100:
        // unreachable -> independent (bounds-awareness catches off-by-one bugs)
        assert!(matches!(
            test_dimension(
                Affine { a: 1, b: -100 },
                Affine { a: 1, b: 0 },
                IterRange { lo: 0, hi: 99 }
            ),
            DepOutcome::Independent
        ));
    }

    #[test]
    fn multi_dim_conjunction() {
        // 2D: first dim proves independence -> overall independent
        assert!(matches!(
            test_pair(
                &[Affine { a: 0, b: 1 }, Affine { a: 1, b: 0 }],
                &[Affine { a: 0, b: 2 }, Affine { a: 1, b: 0 }],
                R
            ),
            DepOutcome::Independent
        ));
    }

    // ---- negative-coefficient regression tests --------------------------------
    //
    // box_has_solution previously flipped the feasible-t interval whenever the
    // parametric step was negative (a negative subscript coefficient reaches
    // the general gcd/box path via Unary::Neg, `n - i` invariant collapse, or a
    // literal like `5 - 3*i`), reporting truly dependent pairs as Independent —
    // an unsound SAFE verdict. These cases all have real in-range solutions.

    #[test]
    fn negative_step_dependent_pairs_are_never_independent() {
        // src = 5 - 2i (a[5-2i]), dst = i (a[i]) over [0,9]:
        // solutions at (i=0,j=5),(1,3),(2,1). Old code said Independent.
        let src = Affine { a: -2, b: 5 };
        let dst = Affine { a: 1, b: 0 };
        let r = IterRange { lo: 0, hi: 9 };
        assert!(
            matches!(test_dimension(src, dst, r), DepOutcome::Dependence { .. }),
            "a[5-2i] vs a[i] has real dependences at i=0..2"
        );
        // Cross-check against brute force.
        assert!(box_brute_force(r, src, dst));
    }

    #[test]
    fn negative_step_both_sides_still_detected() {
        // src = 2i (write a[2i]), dst = 5 - i (read a[5-i]) over [0,9]:
        // equation 2i + j = 5; solutions (0,5),(1,3),(2,1).
        let src = Affine { a: 2, b: 0 };
        let dst = Affine { a: -1, b: 5 };
        let r = IterRange { lo: 0, hi: 9 };
        assert!(matches!(
            test_dimension(src, dst, r),
            DepOutcome::Dependence { .. }
        ));
        assert!(box_brute_force(r, src, dst));
    }

    #[test]
    fn negative_coefficient_out_of_range_is_still_independent() {
        // Same shapes but the crossing points fall outside the range.
        let r = IterRange { lo: 100, hi: 199 };
        // src = 5 - 2i vs dst = i over [100,199]: lhs <= -195, rhs >= 100.
        assert!(matches!(
            test_dimension(
                Affine { a: -2, b: 5 },
                Affine { a: 1, b: 0 },
                r
            ),
            DepOutcome::Independent
        ));
        assert!(!box_brute_force(r, Affine { a: -2, b: 5 }, Affine { a: 1, b: 0 }));
    }

    #[test]
    fn box_agrees_with_brute_force_over_sign_grid() {
        // Property sweep: every coefficient-sign combination and offset over a
        // small box must agree with brute force. This is the net that catches
        // any future sign-handling regression in the general-case path.
        let r = IterRange { lo: 0, hi: 12 };
        for sa in [-3i128, -2, -1, 1, 2, 3] {
            for sb in [-4i128, -1, 0, 3, 7] {
                for da in [-2i128, -1, 1, 2] {
                    for db in [-3i128, 0, 2, 6] {
                        let src = Affine { a: sa, b: sb };
                        let dst = Affine { a: da, b: db };
                        // Skip pairs routed to earlier battery stages? No:
                        // brute force must agree with the WHOLE test_dimension,
                        // which is exactly what production runs.
                        let algebra = matches!(
                            test_dimension(src, dst, r),
                            DepOutcome::Dependence { .. }
                        );
                        let truth = box_brute_force(r, src, dst);
                        assert_eq!(
                            algebra, truth,
                            "src={sa}*i{sb:+} dst={da}*j{db:+} range=[0,12]"
                        );
                    }
                }
            }
        }
    }
}
