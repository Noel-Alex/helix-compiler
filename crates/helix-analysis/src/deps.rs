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
        DirVec { lt: true, eq: true, gt: true }
    }

    pub fn exact(dir: Dir) -> Self {
        match dir {
            Dir::Lt => DirVec { lt: true, eq: false, gt: false },
            Dir::Eq => DirVec { lt: false, eq: true, gt: false },
            Dir::Gt => DirVec { lt: false, eq: false, gt: true },
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
            DepOutcome::Dependence { distance: Some(0), dirs: vec![DirVec::exact(Dir::Eq)] }
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
        return DepOutcome::Dependence { distance: Some(dv), dirs: vec![dir] };
    }

    // ---- Weak-Zero SIV: one side has zero coefficient ------------------------
    if src.a == 0 || dst.a == 0 {
        let (a, p, c, swap) = if dst.a == 0 {
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
            dirs: vec![DirVec { lt: true, eq: true, gt: true }],
        };
    }

    // ---- Weak-Crossing SIV: coefficients are negatives of each other ----------
    if src.a == -dst.a {
        // a*i + p = -a*i' + q => a*(i + i') = q - p => crossing x = (q-p)/(2a)
        let num = dst.b - src.b;
        let den = 2 * src.a;
        if den == 0 || num % den != 0 {
            return DepOutcome::Independent;
        }
        let cross = num / den;
        if cross < range.lo || cross > range.hi {
            return DepOutcome::Independent;
        }
        // Crossing exists: '<' and '>' possible; '=' only if i == i' == cross,
        // i.e. 2*i == cross... only when cross is even relative to a==±1 forms.
        // Conservative: allow all three directions.
        return DepOutcome::Dependence {
            distance: None,
            dirs: vec![DirVec::star()],
        };
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
            return DepOutcome::Dependence { distance: None, dirs: vec![DirVec::star()] };
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
    let t_interval = |c: i128, step: i128| -> Option<(i128, i128)> {
        // lo <= c + step*t <= hi  =>  (lo-c)/step <= t <= (hi-c)/step  (careful with sign)
        if step == 0 {
            return if c >= lo && c <= hi { Some(i128::MIN / 4, i128::MAX / 4) } else { None };
        }
        let (t1, t2) = ((lo - c).div_euclid(step), (hi - c).div_euclid(step));
        let (t1b, t2b) = ((lo - c).div_ceil(step), (hi - c).div_ceil(step));
        let _ = t2;
        let _ = t2b;
        Some((t1.min(t1b), t1.max(t1b)))
    };
    // Simpler robust approach: iterate interval intersection numerically.
    let iv = |step: i128, c: i128| -> Option<(i128, i128)> {
        if step == 0 {
            return if c >= lo && c <= hi {
                Some((i128::MIN / 4, i128::MAX / 4))
            } else {
                None
            };
        }
        // t >= (lo - c)/step and t <= (hi - c)/step with correct rounding per sign.
        let lower = if step > 0 {
            ceil_div(lo - c, step)
        } else {
            floor_div(lo - c, step)
        };
        let upper = if step > 0 {
            floor_div(hi - c, step)
        } else {
            ceil_div(hi - c, step)
        };
        if lower <= upper { Some((lower, upper)) } else { None }
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
pub fn test_pair(subscripts_src: &[Affine], subscripts_dst: &[Affine], range: IterRange) -> DepOutcome {
    debug_assert_eq!(subscripts_src.len(), subscripts_dst.len());
    let mut combined_dirs: Vec<DirVec> = Vec::new();
    let mut distance: Option<i64> = Some(0);

    for (s, d) in subscripts_src.iter().zip(subscripts_dst.iter()) {
        match test_dimension(*s, *d, range) {
            DepOutcome::Independent => return DepOutcome::Independent,
            DepOutcome::Dependence { distance: dist, dirs } => {
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
        return DepOutcome::Dependence { distance: None, dirs: vec![DirVec::star()] };
    }
    DepOutcome::Dependence { distance, dirs: combined_dirs }
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
        assert!(outcome_is_independent(Affine { a: 0, b: 3 }, Affine { a: 0, b: 5 }));
        assert!(!outcome_is_independent(Affine { a: 0, b: 3 }, Affine { a: 0, b: 3 }));
    }

    #[test]
    fn strong_siv_distance_one() {
        // a[i-1] vs a[i]: src coeff 1 b -1; dst coeff 1 b 0 => distance = (0-(-1))/1 = 1
        match test_dimension(Affine { a: 1, b: -1 }, Affine { a: 1, b: 0 }, R) {
            DepOutcome::Dependence { distance: Some(1), dirs } => {
                assert!(dirs[0].gt && !dirs[0].lt && !dirs[0].eq);
            }
            other => panic!("expected distance-1 dependence, got {other:?}"),
        }
    }

    #[test]
    fn strong_siv_proves_independence_on_nondivisible() {
        // 2i vs 2i+1: delta = 1 not divisible by 2 -> independent
        assert!(outcome_is_independent(Affine { a: 2, b: 0 }, Affine { a: 2, b: 1 }));
    }

    #[test]
    fn weak_zero_hits_single_point() {
        // src a[i], dst a[7]: single point i=7 in range -> dependence exists
        assert!(!outcome_is_independent(Affine { a: 1, b: 0 }, Affine { a: 0, b: 7 }));
        // but a[200] out of range -> independent
        assert!(outcome_is_independent(Affine { a: 1, b: 0 }, Affine { a: 0, b: 200 }));
    }

    #[test]
    fn weak_crossing() {
        // src a[i], dst a[-i + 100]: crossing at i=50 in range
        assert!(!outcome_is_independent(Affine { a: 1, b: 0 }, Affine { a: -1, b: 100 }));
        // a[i] vs a[-i + 300]: crossing at 150 out of [0,99]
        assert!(outcome_is_independent(Affine { a: 1, b: 0 }, Affine { a: -1, b: 300 }));
    }

    #[test]
    fn gcd_box_general_case() {
        // a[2i] vs a[i]: 2i - j = 0, gcd(2,1)=1 divides 0 -> solutions exist (i=j even)
        assert!(!outcome_is_independent(Affine { a: 2, b: 0 }, Affine { a: 1, b: 0 }));
        // a[2i] vs a[2j+1]: 2i - 2j = 1, gcd 2 does not divide 1 -> independent
        assert!(outcome_is_independent(Affine { a: 2, b: 0 }, Affine { a: 2, b: 1 }));
    }

    #[test]
    fn negative_range_guards() {
        // Distance beyond trip count: a[i-100] vs a[i] with only 100 trips
        match test_dimension(
            Affine { a: 1, b: -100 },
            Affine { a: 1, b: 0 },
            IterRange { lo: 0, hi: 99 },
        ) {
            DepOutcome::Dependence { distance: Some(100), .. } => {}
            other => panic!("expected distance 100, got {other:?}"),
        }
        // But with only 10 trips the pair is unreachable -> independent
        assert!(matches!(
            test_dimension(
                Affine { a: 1, b: -100 },
                Affine { a: 1, b: 0 },
                IterRange { lo: 0, hi: 9 }
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
}
