//! Verdicts and reports: walk each loop, pair its accesses, run the battery,
//! exempt recognized reductions, and produce the polished [`LoopReport`] that
//! feeds both the parallelizer backend and the Observatory UI.

use crate::access;
use crate::canon::{self, CanonicalLoop};
use crate::deps::{self, DepOutcome, IterRange};
use crate::loops::LoopInfo;
use crate::reduce;
use crate::Bound;
use helix_ir::FuncIr;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReductionOp {
    Add,
    Mul,
    Min,
    Max,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Reduction {
    pub op: ReductionOp,
    /// Source-level variable name.
    pub var: String,
}

/// One classified cross-iteration dependence edge.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DepEdge {
    /// e.g. "RAW a[i] <- a[i-1]"
    pub kind_label: String,
    pub array: String,
    /// Exact distance when determinable.
    pub distance: Option<i64>,
    /// Allen-Kennedy level: which nest level carries it (1-based).
    pub level: u32,
    pub direction: String,
    /// Full human explanation for tooltips/reports.
    pub explain: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Verdict {
    /// Independent iterations → DOALL.
    SafeParallel,
    /// Recognized associative reduction → private accumulators + combine.
    ReductionParallel(Reduction),
    /// Must stay sequential; carries the precise reason.
    Sequential(String),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LoopReport {
    pub loop_id: usize,
    pub depth: u32,
    /// Header block label (bbN).
    pub header: String,
    pub blocks: Vec<String>,
    /// Induction variable name when canonical (e.g. "i").
    pub iv: Option<String>,
    pub bounds: Option<(String, String)>,
    /// Pretty access lines: "READ a[i]", "WRITE out[i]".
    pub accesses: Vec<String>,
    pub raw_deps: Vec<DepEdge>,
    pub war_deps: Vec<DepEdge>,
    pub waw_deps: Vec<DepEdge>,
    /// Non-reduction reasons collected during analysis (side effects, unknown shapes).
    pub notes: Vec<String>,
    pub verdict: Verdict,
}

impl LoopReport {
    /// One-line summary like the demo slide:
    /// `Loop #1: RAW 0 / WAR 0 / WAW 0 => SAFE`
    pub fn summary_line(&self) -> String {
        let verdict = match &self.verdict {
            Verdict::SafeParallel => "SAFE".to_string(),
            Verdict::ReductionParallel(r) => format!("REDUCTION({})", r.op.symbol()),
            Verdict::Sequential(reason) => format!("SEQUENTIAL ({reason})"),
        };
        format!(
            "Loop #{}: RAW {} / WAR {} / WAW {} => {}",
            self.loop_id + 1,
            self.raw_deps.len(),
            self.war_deps.len(),
            self.waw_deps.len(),
            verdict
        )
    }
}

/// Analyze all loops of one function.
pub fn analyze(func: &FuncIr, loops: &LoopInfo) -> Vec<LoopReport> {
    let mut reports = Vec::new();

    for lp in &loops.loops {
        let mut notes = Vec::new();

        // Side effects kill parallelization immediately (spec normative).
        if func.loop_has_print(lp) {
            notes.push("contains a side effect (print)".to_string());
        }

        // Canonical shape?
        let canonical: Option<CanonicalLoop> = canon::canon(func, lp);
        let iv_name = canonical.as_ref().map(|c| func.local_name(c.iv)).unwrap_or("?");
        if canonical.is_none() {
            notes.push("non-canonical loop shape".to_string());
        }

        // Access extraction per block/instruction.
        let accesses = access::collect_accesses(func, lp);

        // Reduction recognition first — an approved reduction exempts its own
        // distance-1 RAW self-dependence on the accumulator.
        let reductions = reduce::find_reductions(func, &lp.blocks);
        let reduction_report = reductions
            .first()
            .map(|r| Reduction { op: r.op, var: func.local_name(r.var).to_string() });

        // Pair accesses on the same array; classify RAW/WAR/WAW.
        let mut raw_deps = Vec::new();
        let mut war_deps = Vec::new();
        let mut waw_deps = Vec::new();

        let range = canonical.as_ref().map_or(
            IterRange { lo: i128::MIN / 4, hi: i128::MAX / 4 },
            |c| match (&c.start, &c.end) {
                (Bound::Const(lo), Bound::Const(hi)) => IterRange {
                    lo: *lo as i128,
                    hi: hi.saturating_sub(1) as i128, // half-open [start, end)
                },
                _ => IterRange { lo: i128::MIN / 4, hi: i128::MAX / 4 },
            },
        );

        for a in &accesses {
            for b in &accesses {
                if a.arr != b.arr {
                    continue;
                }
                let (src, dst) = (a, b);
                let kind = match (src.is_write, dst.is_write) {
                    (true, false) => "RAW",
                    (false, true) => "WAR",
                    (true, true) => "WAW",
                    (false, false) => continue, // RAR never a dependence
                };
                // Same access pair in program order only (src executes before dst
                // in some iteration ordering); skip self-pairs of reads already skipped.
                let (aff_s, _) = src.affine.clone().into_tuple();
                let _ = aff_s;
                let aff_src = src.affine.unwrap_or(deps::Affine { a: 0, b: 0 });
                let aff_dst = dst.affine.unwrap_or(deps::Affine { a: 0, b: 1 });
                if src.affine.is_none() || dst.affine.is_none() {
                    notes.push(format!(
                        "unanalyzable subscript on '{}' — assuming dependence",
                        func.local_name(src.arr)
                    ));
                    push_edge(
                        if kind == "RAW" { &mut raw_deps } else if kind == "WAR" { &mut war_deps } else { &mut waw_deps },
                        kind,
                        func.local_name(src.arr),
                        None,
                        "*",
                        lp.depth,
                        "subscript not affine — conservative dependence",
                    );
                    continue;
                }
                match deps::test_pair(&[aff_src], &[aff_dst], range) {
                    DepOutcome::Independent => {}
                    DepOutcome::Dependence { distance, dirs } => {
                        // Reduction exemption: WAW/RAW/WAR between the SAME scalar is not
                        // possible here (scalars aren't arrays); reduction exemption shows
                        // up as: the only deps are on the accumulator var, handled by
                        // lowering. Array-level exemption does not apply.
                        let dir_str: String = dirs.iter().map(|d| d.describe()).collect();
                        let dist_txt = distance.map(|d| d.to_string()).unwrap_or("?".into());
                        let label = format!(
                            "{kind} {}[{}] <- {}[{}]",
                            func.local_name(src.arr),
                            render_affine(aff_src, iv_name),
                            func.local_name(dst.arr),
                            render_affine(aff_dst, iv_name)
                        );
                        push_edge(
                            if kind == "RAW" { &mut raw_deps } else if kind == "WAR" { &mut war_deps } else { &mut waw_deps },
                            kind,
                            func.local_name(src.arr),
                            distance,
                            &dir_str,
                            lp.depth,
                            &format!("{label} (distance {dist_txt}, level {})", lp.depth),
                        );
                    }
                }
            }
        }

        // Verdict assembly.
        let verdict = if !notes.is_empty() {
            Verdict::Sequential(notes.join("; "))
        } else if !raw_deps.is_empty() || !war_deps.is_empty() || !waw_deps.is_empty() {
            let first = raw_deps
                .first()
                .or_else(|| war_deps.first())
                .or_else(|| waw_deps.first())
                .expect("non-empty");
            Verdict::Sequential(first.explain.clone())
        } else if let Some(red) = reduction_report {
            Verdict::ReductionParallel(red)
        } else {
            Verdict::SafeParallel
        };

        reports.push(LoopReport {
            loop_id: lp.id,
            depth: lp.depth,
            header: format!("bb{}", lp.header.0),
            blocks: lp.blocks.iter().map(|b| format!("bb{}", b.0)).collect(),
            iv: Some(iv_name.to_string()),
            bounds: canonical.as_ref().map(|c| (bound_text(func, &c.start), bound_text(func, &c.end))),
            accesses: accesses
                .iter()
                .map(|a| {
                    format!(
                        "{} {}[{}]",
                        if a.is_write { "WRITE" } else { "READ" },
                        func.local_name(a.arr),
                        a.raw_index
                    )
                })
                .collect(),
            raw_deps,
            war_deps,
            waw_deps,
            notes,
            verdict,
        });
    }
    reports
}

fn push_edge(
    sink: &mut Vec<DepEdge>,
    kind: &str,
    array: &str,
    distance: Option<i64>,
    dir: &str,
    level: u32,
    explain: &str,
) {
    sink.push(DepEdge {
        kind_label: format!("{kind} on '{array}'"),
        array: array.to_string(),
        distance,
        level,
        direction: dir.to_string(),
        explain: explain.to_string(),
    });
}

fn render_affine(a: deps::Affine, iv: &str) -> String {
    match (a.a, a.b) {
        (1, 0) => iv.to_string(),
        (1, c) => format!("{iv} + {c}"),
        (-1, 0) => format!("-{iv}"),
        (-1, c) => format!("{c} - {iv}"),
        (k, 0) => format!("{k}*{iv}"),
        (k, c) => format!("{k}*{iv} + {c}"),
        (0, c) => c.to_string(),
    }
}

fn bound_text(_func: &FuncIr, b: &Bound) -> String {
    match b {
        Bound::Const(c) => c.to_string(),
        Bound::Sym(_) => "n".to_string(), // symbolic bound rendered by name at call sites
    }
}

trait IntoTuple {
    fn into_tuple(self) -> (Option<deps::Affine>, ());
}

impl IntoTuple for Option<deps::Affine> {
    fn into_tuple(self) -> (Option<deps::Affine>, ()) {
        (self, ())
    }
}
