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
    /// e.g. "RAW on 'a'" — display label.
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
    pub header: String,
    pub blocks: Vec<String>,
    pub iv: Option<String>,
    pub bounds: Option<(String, String)>,
    /// Pretty access lines: "READ a[i]", "WRITE out[i]".
    pub accesses: Vec<String>,
    pub raw_deps: Vec<DepEdge>,
    pub war_deps: Vec<DepEdge>,
    pub waw_deps: Vec<DepEdge>,
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

/// Analyze all loops of one function. `func` must be in SSA form (the pipeline
/// runs to_ssa before analysis).
pub fn analyze(func: &FuncIr, loops: &LoopInfo) -> Vec<LoopReport> {
    let mut reports = Vec::new();

    for lp in &loops.loops {
        let mut notes = Vec::new();

        // Side effects kill parallelization immediately (spec normative).
        if has_print(func, lp) {
            notes.push("contains a side effect (print)".to_string());
        }

        // Canonical shape?
        let canonical: Option<CanonicalLoop> = canon::canon(func, lp);
        if canonical.is_none() {
            notes.push("non-canonical loop shape".to_string());
        }
        let iv_name = canonical
            .as_ref()
            .map(|c| local_name(func, c.iv))
            .unwrap_or_else(|| "?".to_string());

        // Access extraction relative to the induction value.
        let accesses = match &canonical {
            Some(c) => access::collect(func, lp, c.iv_value_in_loop),
            None => Vec::new(),
        };

        // Reduction recognition first — an approved reduction exempts its own
        // distance-1 RAW self-dependence on the accumulator.
        let reductions = reduce::find_reductions(func, &lp.blocks);
        let reduction_report = reductions
            .first()
            .map(|r| Reduction { op: r.op, var: local_name(func, r.var) });

        // Iteration range for bound-aware testing (half-open [start, end)).
        let range = canonical.as_ref().map_or(
            IterRange { lo: i128::MIN / 4, hi: i128::MAX / 4 },
            |c| match (&c.start, &c.end) {
                (Bound::Const(lo), Bound::Const(hi)) => IterRange {
                    lo: *lo as i128,
                    hi: hi.saturating_sub(1) as i128,
                },
                _ => IterRange { lo: -1 << 40, hi: 1 << 40 }, // symbolic bounds: wide box
            },
        );

        // Pair same-array accesses; classify RAW/WAR/WAW.
        let mut raw_deps = Vec::new();
        let mut war_deps = Vec::new();
        let mut waw_deps = Vec::new();
        let arr_name =
            |l: helix_ir::LocalId| local_name(func, l);

        for (i, src) in accesses.iter().enumerate() {
            for dst in accesses.iter().skip(i + 1) {
                if src.arr != dst.arr {
                    continue;
                }
                // Order pairs so the WRITE side is the "source" when present
                // (dependence direction follows program order per iteration;
                // cross-iteration both orderings are covered by the battery's
                // ±distance handling).
                let (w, r) = if src.is_write && !dst.is_write {
                    (src, dst)
                } else if dst.is_write && !src.is_write {
                    (dst, src)
                } else {
                    (src, dst)
                };
                let kind = match (src.is_write, dst.is_write) {
                    (true, false) | (false, true) => "RAW",
                    (true, true) => "WAW",
                    (false, false) => continue, // RAR never a dependence
                };

                let (Some(aff_w), Some(aff_r)) = (w.affine, r.affine) else {
                    notes.push(format!(
                        "unanalyzable subscript on '{}' — assuming dependence",
                        arr_name(src.arr)
                    ));
                    push_edge(
                        sink_for(kind, &mut raw_deps, &mut war_deps, &mut waw_deps),
                        kind,
                        &arr_name(src.arr),
                        None,
                        "*",
                        lp.depth,
                        &format!(
                            "{} {}[?] vs {}[?] — subscript not affine, conservative",
                            kind,
                            arr_name(src.arr),
                            arr_name(src.arr)
                        ),
                    );
                    continue;
                };

                match deps::test_pair(&[aff_r], &[aff_w], range) {
                    DepOutcome::Independent => {}
                    DepOutcome::Dependence { distance, dirs } => {
                        let dir_str: String = dirs.iter().map(|d| d.describe()).collect();
                        let dist_txt = distance.map(|d| d.to_string()).unwrap_or("?".into());
                        let name = arr_name(src.arr);
                        let label = format!(
                            "{kind} {name}[{}] ← {name}[{}]",
                            render_affine(aff_w, &iv_name),
                            render_affine(aff_r, &iv_name),
                        );
                        push_edge(
                            sink_for(kind, &mut raw_deps, &mut war_deps, &mut waw_deps),
                            kind,
                            &name,
                            distance,
                            &dir_str,
                            lp.depth,
                            &format!("{label} (distance {dist_txt}, level {})", lp.depth),
                        );
                    }
                }
            }
        }

        // Verdict assembly: notes → sequential; real dependences → sequential
        // with the first edge as the reason; otherwise reduction or plain DOALL.
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
            iv: Some(iv_name.clone()),
            bounds: canonical
                .as_ref()
                .map(|c| (bound_text(&c.start), bound_text(&c.end))),
            accesses: accesses
                .iter()
                .map(|a| {
                    let sub = a
                        .affine
                        .map(|f| render_affine(f, &iv_name))
                        .unwrap_or_else(|| "?".into());
                    format!(
                        "{} {}[{}]",
                        if a.is_write { "WRITE" } else { "READ" },
                        arr_name(a.arr),
                        sub
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

fn local_name(func: &FuncIr, l: helix_ir::LocalId) -> String {
    func.types.local_name(l).unwrap_or("?").to_string()
}

fn sink_for<'s>(
    kind: &str,
    raw: &'s mut Vec<DepEdge>,
    war: &'s mut Vec<DepEdge>,
    waw: &'s mut Vec<DepEdge>,
) -> &'s mut Vec<DepEdge> {
    match kind {
        "WAW" => waw,
        "WAR" => war,
        _ => raw,
    }
}

fn has_print(func: &FuncIr, lp: &crate::loops::Loop) -> bool {
    lp.blocks
        .iter()
        .any(|b| func.block(*b).insts.iter().any(|i| is_print_call(i)))
}

fn is_print_call(i: &helix_ir::Inst) -> bool {
    matches!(i, helix_ir::Inst::Call(c) if c.callee == "print")
}

fn push_edge(sink: &mut Vec<DepEdge>, kind: &str, array: &str, distance: Option<i64>, dir: &str, level: u32, explain: &str) {
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
        (1, c) if c > 0 => format!("{iv} + {c}"),
        (1, c) => format!("{iv} - {}", -c),
        (-1, 0) => format!("-{iv}"),
        (-1, c) if c > 0 => format!("{c} - {iv}"),
        (-1, c) => format!("-{iv} - {}", -c),
        (k, 0) => format!("{k}*{iv}"),
        (k, c) if c > 0 => format!("{k}*{iv} + {c}"),
        (k, c) => format!("{k}*{iv} - {}", -c),
        _ => "?".to_string(),
    }
}

fn bound_text(b: &Bound) -> String {
    match b {
        Bound::Const(c) => c.to_string(),
        Bound::Sym(_) => "n".to_string(), // rendered symbolically at call sites
    }
}
