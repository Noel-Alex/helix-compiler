//! Verdicts, per-loop reports, and the parallelization plan — the layer that
//! walks each canonical loop, pairs its array accesses, runs the dependence
//! battery, exempts recognized reductions, and hands polished results to both
//! the backend ([`build_plan`]) and the Observatory UI ([`LoopReport`]).
//!
//! ## The verdict ladder
//!
//! Per loop, in order:
//! 1. **Side effects** (`print`) → [`Verdict::Sequential`] (spec normative).
//! 2. **Non-canonical shape** → `Sequential` (no induction variable ⇒ no
//!    affine iteration space to distribute).
//! 3. **Dependence battery** over same-array access pairs. Surviving edges
//!    split into loop-*independent* ones (exact distance 0 — both touches in
//!    the same iteration; harmless under DOALL and not recorded) and
//!    loop-*carried* ones ([`DepEdge`]s).
//! 4. **Reduction exemption**: when every carried edge sits on a recognized
//!    reduction's accumulator variable, the edges are dropped — that
//!    distance-1 self-flow is precisely what the reduction transform breaks
//!    (private accumulators + associative combine) — and the loop becomes
//!    [`Verdict::ReductionParallel`]. Any edge on real memory survives and
//!    vetoes with its explanation as the printed reason.
//! 5. Otherwise [`Verdict::SafeParallel`].
//!
//! ## Reports are data
//!
//! Everything here derives `Serialize` because the Observatory ships reports
//! to the browser verbatim; the string fields are pre-rendered so the UI
//! stays markup-free.

use crate::Bound;
use crate::access;
use crate::canon::{self, CanonicalLoop};
use crate::deps::{self, DepOutcome, IterRange};
use crate::loops::{Loop, LoopInfo};
use crate::reduce;
use helix_ir::{BlockId, FuncIr, LocalId};
use serde::{Deserialize, Serialize};

/// Associative operators approved for parallel combination.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReductionOp {
    /// Sum family (subtraction folds in via negated terms at lowering).
    Add,
    /// Product.
    Mul,
    /// IEEE `minNum` builtin.
    Min,
    /// IEEE `maxNum` builtin.
    Max,
}

/// A recognized reduction, reported with its source-level name.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Reduction {
    pub op: ReductionOp,
    /// Source-level variable name (report lines, UI cards, runtime ctx).
    pub var: String,
}

/// One classified cross-iteration dependence edge.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DepEdge {
    /// Display label, e.g. `"RAW on 'a'"`.
    pub kind_label: String,
    /// Array (or variable) name the edge is carried on.
    pub array: String,
    /// Exact distance when determinable (`None` = unknown / multi-valued).
    pub distance: Option<i64>,
    /// Allen-Kennedy level carrying the edge (the analyzed loop's depth).
    pub level: u32,
    /// Direction vector rendering, e.g. `"="`, `">"`, `"*"`.
    pub direction: String,
    /// Full human explanation for tooltips/reports.
    pub explain: String,
}

/// What the parallelizer may do with one loop.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Verdict {
    /// Independent iterations → DOALL.
    SafeParallel,
    /// Recognized associative reduction → private accumulators + combine.
    ReductionParallel(Reduction),
    /// Must stay sequential; carries the precise reason.
    Sequential(String),
}

/// Everything the Observatory card shows about one analyzed loop.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LoopReport {
    pub loop_id: usize,
    pub depth: u32,
    /// Header block, rendered `bbN`.
    pub header: String,
    /// All body blocks (header included), rendered `bbN`.
    pub blocks: Vec<String>,
    /// Induction variable name when the loop is canonical.
    pub iv: Option<String>,
    /// `(start, end)` bounds text for canonical loops.
    pub bounds: Option<(String, String)>,
    /// Pretty access lines: `"READ a[i]"`, `"WRITE out[i]"`.
    pub accesses: Vec<String>,
    pub raw_deps: Vec<DepEdge>,
    pub war_deps: Vec<DepEdge>,
    pub waw_deps: Vec<DepEdge>,
    /// Analysis remarks (side effects, unanalyzable subscripts…). Any note
    /// vetoes parallelization for the loop it belongs to.
    pub notes: Vec<String>,
    pub verdict: Verdict,
}

impl LoopReport {
    /// One-line summary like the demo slide:
    /// `Loop #1: RAW 0 / WAR 0 / WAW 0 => SAFE`.
    ///
    /// Exempted reduction dependences no longer appear in the counts (they
    /// were removed from the lists when exempted); the verdict suffix is what
    /// distinguishes a reduction loop from a plain-safe one.
    #[must_use]
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

/// Analyze all loops of one function. `func` must be in SSA form (the
/// pipeline runs `to_ssa` before analysis).
///
/// Loops are reported in discovery order (outermost-first per nest); nested
/// loops get independent verdicts — an inner loop may parallelize inside a
/// sequential outer one.
#[must_use]
pub fn analyze(func: &FuncIr, loops: &LoopInfo) -> Vec<LoopReport> {
    let mut reports = Vec::new();

    for lp in &loops.loops {
        let mut notes = Vec::new();

        // Side effects kill parallelization immediately (spec normative):
        // print AND any user-defined call (which may itself print or mutate
        // shared state). Pure builtins are exempt.
        if has_print(func, lp) {
            notes.push("contains a side effect (print)".to_string());
        }
        for name in user_call_names(func, lp) {
            notes.push(format!("calls non-pure function '{name}' inside the loop"));
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

        // Reduction recognition — the canonical iv is excluded because its
        // header φ has exactly the additive shape of a reduction.
        const NO_EXCLUSIONS: &[LocalId] = &[];
        let excluded: &[LocalId] = match &canonical {
            Some(c) => std::slice::from_ref(&c.iv),
            None => NO_EXCLUSIONS,
        };
        let reductions = reduce::find_reductions(func, &lp.blocks, excluded);
        let reduction_report = reductions.first().map(|r| Reduction {
            op: r.op,
            var: local_name(func, r.var),
        });

        // Iteration range for bound-aware testing (half-open [start, end)).
        let range = canonical.as_ref().map_or(
            IterRange {
                lo: i128::MIN / 4,
                hi: i128::MAX / 4,
            },
            |c| match (&c.start, &c.end) {
                (Bound::Const(lo), Bound::Const(hi)) => IterRange {
                    lo: i128::from(*lo),
                    hi: i128::from(*hi).saturating_sub(1),
                },
                _ => IterRange {
                    lo: -1 << 40,
                    hi: 1 << 40,
                }, // symbolic bounds: wide box
            },
        );

        // Pair same-array accesses; classify RAW/WAR/WAW.
        let mut raw_deps = Vec::new();
        let mut war_deps = Vec::new();
        let mut waw_deps = Vec::new();
        let arr_name = |l: LocalId| local_name(func, l);

        for (i, src) in accesses.iter().enumerate() {
            for dst in accesses.iter().skip(i + 1) {
                if src.arr != dst.arr {
                    continue;
                }
                let kind = match (src.is_write, dst.is_write) {
                    (true, false) | (false, true) => "RAW",
                    (true, true) => "WAW",
                    (false, false) => continue, // RAR never a dependence
                };

                let (Some(aff_w), Some(aff_r)) = (w_of(src, dst), r_of(src, dst)) else {
                    let name = arr_name(src.arr);
                    notes.push(format!(
                        "unanalyzable subscript on '{name}' — assuming dependence"
                    ));
                    push_edge(
                        sink_for(kind, &mut raw_deps, &mut war_deps, &mut waw_deps),
                        kind,
                        &name,
                        None,
                        "*",
                        lp.depth,
                        format!(
                            "{kind} {name}[?] vs {name}[?] — subscript not affine, conservative"
                        ),
                    );
                    continue;
                };
                // Battery convention: the READ side is the first operand, so a
                // classic `a[i] = a[i-1] + c` reports distance +1.
                match deps::test_pair(&[aff_r], &[aff_w], range) {
                    DepOutcome::Independent => {}
                    DepOutcome::Dependence { distance, dirs } => {
                        // Exact distance 0 = both touches in ONE iteration:
                        // loop-independent, survives DOALL, not carried.
                        if distance == Some(0) && dirs.iter().all(|d| d.eq && !d.lt && !d.gt) {
                            continue;
                        }
                        let dir_str: String = dirs.iter().map(|d| d.describe()).collect();
                        let dist_txt = distance.map_or("?".to_string(), |d| d.to_string());
                        let name = arr_name(src.arr);
                        let detail = if distance.is_some() {
                            format!("carried by iteration distance {dist_txt}")
                        } else {
                            "gcd/box test inconclusive — integer solutions exist in range"
                                .to_string()
                        };
                        push_edge(
                            sink_for(kind, &mut raw_deps, &mut war_deps, &mut waw_deps),
                            kind,
                            &name,
                            distance,
                            &dir_str,
                            lp.depth,
                            format!(
                                "{kind} {name}[{}] <- {name}[{}] ({detail}, level {})",
                                render_affine(aff_w, &iv_name),
                                render_affine(aff_r, &iv_name),
                                lp.depth
                            ),
                        );
                    }
                }
            }
        }

        // ---- Reduction exemption ------------------------------------------
        //
        // Every surviving carried edge that sits on the accumulator variable
        // (scalar chains surface under the accumulator's name) is dropped:
        // the reduction transform breaks exactly this self-flow. Edges on
        // real array memory stay and veto below.
        if let Some(acc) = reductions
            .first()
            .map(|r| r.var)
            .filter(|_| notes.is_empty())
        {
            let acc_name = local_name(func, acc);
            drop_edges_on(&acc_name, &mut raw_deps);
            drop_edges_on(&acc_name, &mut war_deps);
            drop_edges_on(&acc_name, &mut waw_deps);
        }

        // Verdict assembly: notes → sequential; surviving carried dependences
        // → sequential citing the first; any recognized reduction → approval
        // (whether its own chain produced exemptable edges or the body had
        // none); nothing at all → plain DOALL.
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

// ---------------------------------------------------------------------------
// ParallelPlan (interface-contracts.md, Addendum 2)
// ---------------------------------------------------------------------------

/// How one loop region will be executed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegionKind {
    /// Plain DOALL over iterations `[start, end)`.
    DoAll,
    /// DOALL with per-thread accumulators combined after the join.
    Reduction(ReductionOp),
}

/// One approved parallel region: a loop the backend lowers into a
/// fork/join body function.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegionDesc {
    /// Index into the program's `Vec<FuncIr>`.
    pub func_idx: usize,
    /// The loop header block (identifies the region inside its function).
    pub header: BlockId,
    pub kind: RegionKind,
    /// Present iff `kind == Reduction`; the combine operator for the backend
    /// and runtime.
    pub reduction: Option<ReductionOp>,
    /// Body-function symbol name, e.g. `"main.loop0.body"` — unique per
    /// region, registered against the runtime's body registry.
    pub body_fn_name: String,
    /// SSA id of the start bound (`None` = constant folded into the ctx).
    pub start_val: Option<u32>,
    /// SSA id of the end bound (`None` = constant folded into the ctx).
    pub end_val: Option<u32>,
}

/// A whole-program parallelization plan: one entry per approved loop.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParallelPlan {
    pub regions: Vec<RegionDesc>,
}

/// Build the plan from per-function analyses (Addendum 2 signature).
///
/// Approval rule: a loop enters the plan when it is **canonical** and its
/// verdict is [`Verdict::SafeParallel`] or [`Verdict::ReductionParallel`] at
/// nest depth 1 or in innermost position — parallelizing a mid-level loop
/// would serialize every level below it, so those stay out (their inner
/// candidates carry the parallelism instead). `Sequential` loops never enter.
#[must_use]
pub fn build_plan(
    funcs: &[FuncIr],
    loops: &[LoopInfo],
    reports: &[Vec<LoopReport>],
) -> ParallelPlan {
    let mut regions = Vec::new();
    for (fi, info) in loops.iter().enumerate() {
        let Some(reps) = reports.get(fi) else {
            continue;
        };
        let fname = funcs.get(fi).map_or("fn", |f| f.name.as_str());
        for lp in &info.loops {
            let Some(rep) = reps.iter().find(|r| r.loop_id == lp.id) else {
                continue;
            };
            let (kind, red) = match &rep.verdict {
                Verdict::SafeParallel => (RegionKind::DoAll, None),
                Verdict::ReductionParallel(r) => (RegionKind::Reduction(r.op), Some(r.op)),
                Verdict::Sequential(_) => continue,
            };
            let innermost = is_innermost(info, lp);
            if lp.depth != 1 && !innermost {
                continue;
            }
            let (start_val, end_val) = canon::canon(&funcs[fi], lp)
                .map_or((None, None), |c| (bound_sym(&c.start), bound_sym(&c.end)));
            regions.push(RegionDesc {
                func_idx: fi,
                header: lp.header,
                kind,
                reduction: red,
                body_fn_name: format!("{fname}.loop{}.body", lp.id),
                start_val,
                end_val,
            });
        }
    }
    ParallelPlan { regions }
}

/// A loop is innermost when no other discovered loop's body is a strict
/// subset of its own (nesting always produces strictly smaller bodies).
fn is_innermost(info: &LoopInfo, lp: &Loop) -> bool {
    !info.loops.iter().any(|c| {
        c.id != lp.id
            && c.blocks.len() < lp.blocks.len()
            && c.blocks.iter().all(|b| lp.blocks.contains(b))
    })
}

/// Symbolic view of a bound (`None` = compile-time constant).
fn bound_sym(b: &Bound) -> Option<u32> {
    match b {
        Bound::Sym(v) => Some(*v),
        Bound::Const(_) => None,
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Write/read sides of an ordered access pair (`src` earlier in program
/// order). Kept as tiny functions to make the pairing symmetric obvious.
fn w_of<'a>(src: &'a access::Access, dst: &'a access::Access) -> Option<deps::Affine> {
    if src.is_write { src.affine } else { dst.affine }
}

fn r_of<'a>(src: &'a access::Access, dst: &'a access::Access) -> Option<deps::Affine> {
    if src.is_write { dst.affine } else { src.affine }
}

/// Remove every edge carried on `name` (the accumulator variable).
fn drop_edges_on(name: &str, edges: &mut Vec<DepEdge>) {
    edges.retain(|e| e.array != name);
}

fn local_name(func: &FuncIr, l: LocalId) -> String {
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

fn has_print(func: &FuncIr, lp: &Loop) -> bool {
    lp.blocks
        .iter()
        .any(|b| func.block(*b).insts.iter().any(is_print_call))
}

/// Names of every user-defined (non-builtin) function called inside `lp`,
/// in first-appearance order.
fn user_call_names(func: &FuncIr, lp: &Loop) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for b in &lp.blocks {
        for inst in &func.block(*b).insts {
            if let helix_ir::Inst::Call(c) = inst {
                let is_user = match c.callee.as_str() {
                    "min" | "max" | "sqrt" | "abs" | "len" | "zeros" | "print" => false,
                    _ => true,
                };
                if is_user && !out.iter().any(|n| n == &c.callee) {
                    out.push(c.callee.clone());
                }
            }
        }
    }
    out
}

fn is_print_call(i: &helix_ir::Inst) -> bool {
    matches!(i, helix_ir::Inst::Call(c) if c.callee == "print")
}

fn push_edge(
    sink: &mut Vec<DepEdge>,
    kind: &str,
    array: &str,
    distance: Option<i64>,
    dir: &str,
    level: u32,
    explain: String,
) {
    sink.push(DepEdge {
        kind_label: format!("{kind} on '{array}'"),
        array: array.to_string(),
        distance,
        level,
        direction: dir.to_string(),
        explain,
    });
}

/// Render `a*iv + b` the way the report cards show subscripts.
fn render_affine(a: deps::Affine, iv: &str) -> String {
    match (a.a, a.b) {
        (_, b) if a.a == 0 && b == 0 => "0".to_string(), // collapsed invariant
        (0, b) => b.to_string(),
        (1, 0) => iv.to_string(),
        (1, c) if c > 0 => format!("{iv} + {c}"),
        (1, c) => format!("{iv} - {}", -c),
        (-1, 0) => format!("-{iv}"),
        (-1, c) if c > 0 => format!("{c} - {iv}"),
        (-1, c) => format!("-{iv} - {}", -c),
        (k, 0) => format!("{k}*{iv}"),
        (k, c) if c > 0 => format!("{k}*{iv} + {c}"),
        (k, c) => format!("{k}*{iv} - {}", -c),
    }
}

fn bound_text(b: &Bound) -> String {
    match b {
        Bound::Const(c) => (*c).to_string(),
        Bound::Sym(_) => "n".to_string(), // rendered symbolically at call sites
    }
}
