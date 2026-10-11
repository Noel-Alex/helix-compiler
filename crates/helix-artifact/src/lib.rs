//! # helix-artifact — the pure compile-artifact builder
//!
//! Turns one HELIX compile into the single self-contained [`CompileArtifact`]
//! JSON document the Observatory renders (tokens, AST, IR dumps, per-pass
//! snapshots, laid-out CFGs, dominator trees, loop verdicts, execution
//! output) and attaches the committed measurement evidence that travels with
//! it.
//!
//! ```text
//! source ──pipeline──▶ CompileArtifact ─┬─▶ Observatory (axum server)
//!                          │            └─▶ browser lab (wasm32)
//!                          └── bench_index() ──▶ recorded evidence
//! ```
//!
//! No server, no OS: the dependency list is compiler crates plus serde, and
//! nothing here touches the filesystem, the network, or the clock. The
//! recorded campaign is compiled in with `include_bytes!`, never read from
//! disk, so the identical pipeline runs natively under the Observatory and
//! inside `wasm32-unknown-unknown` for the browser lab.
//!
//! ## Module map
//!
//! * [`artifact`] — the wire data contract (`docs/notes/artifact-schema.md`
//!   is normative; field names there are frozen). Everything derives serde so
//!   the whole artifact serializes with one call.
//! * [`pipeline`] — `build_artifact(example, source)`: runs the real compiler
//!   front to back, snapshotting each stage and stopping at the first failure.
//! * [`prepare`] — `prepare_pipeline(source)`: the one shared
//!   parse → sema → IR → SSA → optimizer → analysis → plan sequence every
//!   executing consumer (CLI JIT, bench campaign, selftest) runs.
//! * [`recorded`] — the committed campaign/probe evidence, the frozen-source
//!   identity check that gates it, and the bench points those measurements
//!   attach.
//! * [`cfg_layout`] — server-independent tidy CFG layout. The browser paints
//!   coordinates, it never computes them.
//!
//! ## Design notes (course-report material)
//!
//! **Graceful degradation.** Compilation may stop at any stage, so every
//! post-parse stage is optional and the UI greys absent ones out. A lex error
//! yields tokens + `diags_lex`; a type error yields everything through the
//! AST plus underlined diagnostics; success yields the full stack including
//! interpreter output. The artifact is always valid JSON for its prefix.
//!
//! **Layout is a pure function.** `cfg_layout` consumes only the IR and loop
//! info, so identical compiles produce byte-identical artifacts — golden-test
//! friendly and cacheable without invalidation subtleties.
//!
//! **No panics across the boundary.** Every stage that could fail returns
//! rather than unwraps; a caller maps unexpected failures to its own error
//! response instead of dropping the connection.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, HashSet};

pub mod artifact;
pub mod pipeline;
pub mod prepare;
pub mod recorded;

// -- Contract surface --------------------------------------------------------

pub use artifact::*;
pub use pipeline::{
    BuildOpts, CUSTOM_BENCH_SAMPLE_COUNT, CUSTOM_BENCH_WORK_BUDGET, INTERACTIVE_WORK_BUDGET,
    MAX_SOURCE_BYTES, build_artifact, build_artifact_with_opts, estimated_work,
    measure_interpreter,
};
pub use prepare::{CompiledUnits, prepare_pipeline, prepare_pipeline_checked};

// ---------------------------------------------------------------------------
// Shared CFG geometry constants
// ---------------------------------------------------------------------------

/// Horizontal gap between neighbouring CFG columns.
const COL_GAP: f64 = 46.0;
/// Vertical gap between CFG rows.
const ROW_GAP: f64 = 78.0;
/// Canvas margin on every side.
const MARGIN: f64 = 28.0;
/// Monospace glyph advance used to size CFG boxes (px per char).
const CHAR_W: f64 = 9.0;
/// Horizontal padding inside a CFG box.
const PAD_X: f64 = 28.0;
/// Line height inside a CFG box.
const LINE_H: f64 = 18.0;
/// Vertical padding of a CFG box (title row + bottom air).
const PAD_Y: f64 = 26.0;
/// Minimum box width so tiny blocks still hold `"bb0 · exit"`.
const MIN_NODE_W: f64 = 120.0;
/// Maximum box width; wider lines are clipped by the UI anyway.
const MAX_NODE_W: f64 = 460.0;
/// Gap between boxes stacked in the same CFG layer.
const NODE_GAP_X: f64 = 30.0;

// ---------------------------------------------------------------------------
// CFG layout
// ---------------------------------------------------------------------------

/// Input edge before routing (ids + classification only).
#[derive(Debug, Clone)]
struct RawEdge {
    from: u32,
    to: u32,
    kind: EdgeKind,
    label: &'static str,
}

/// Lays out one function's CFG.
///
/// Pipeline: classify edges (an edge whose target dominates its source closes
/// a natural loop ⇒ backedge) → longest-path layering over the remaining
/// acyclic edge set → DFS-discovery order inside each layer → monospace box
/// sizing → placement → port-exact edge routing. Runs after SSA + passes, so
/// φ lines are shown when present.
#[must_use]
pub fn cfg_layout(
    name: &str,
    ir: &helix_ir::FuncIr,
    loops: &helix_analysis::LoopInfo,
) -> CfgFunction {
    let n = ir.blocks.len();
    let doms = helix_ir::dominators(ir);

    // ---- backedge predicate --------------------------------------------------
    let mut backedge: HashSet<(u32, u32)> = HashSet::new();
    for bi in 0..n {
        for s in ir.succs(helix_ir::BlockId(bi as u32)) {
            let si = s.0;
            if si == bi as u32
                || doms.dominates(helix_ir::BlockId(si), helix_ir::BlockId(bi as u32))
            {
                backedge.insert((bi as u32, si));
            }
        }
    }

    // ---- roles ---------------------------------------------------------------
    // Priority: loop_header > join > entry/exit, because the UI's spotlight
    // and colour story key off loops first; a block that both joins edges
    // and returns still reads best as a join.
    let mut role = vec![BlockRole::Straight; n];
    for (bi, r) in role.iter_mut().enumerate() {
        if matches!(
            ir.term(helix_ir::BlockId(bi as u32)),
            helix_ir::Term::Return(_)
        ) {
            *r = BlockRole::Exit;
        }
        if ir.block(helix_ir::BlockId(bi as u32)).preds.len() > 1 && *r != BlockRole::Exit {
            *r = BlockRole::Join;
        }
    }
    let mut loop_of: Vec<Option<usize>> = vec![None; n];
    for lp in &loops.loops {
        let h = lp.header.0 as usize;
        if h < n {
            role[h] = BlockRole::LoopHeader;
        }
        for b in &lp.blocks {
            let i = b.0 as usize;
            if i >= n {
                continue;
            }
            // Keep the deepest (innermost) matching loop.
            let deeper = loop_of[i].is_none_or(|cur| loops.loops[cur].depth <= lp.depth);
            if deeper {
                loop_of[i] = Some(lp.id);
            }
        }
    }
    // Entry wins last (it is the most navigational of the roles).
    if n > 0 {
        role[ir.entry.0 as usize] = BlockRole::Entry;
    }

    // ---- layering + ordering ---------------------------------------------------
    let order = discovery_order(ir, n);
    let layer = longest_path_layers(ir, n, &backedge);

    // ---- box sizing ------------------------------------------------------------
    let lines_per_block: Vec<Vec<String>> = (0..n)
        .map(|bi| block_lines(ir, helix_ir::BlockId(bi as u32)))
        .collect();
    let widths: Vec<f64> = lines_per_block
        .iter()
        .map(|lines| {
            let widest = lines.iter().map(String::len).max().unwrap_or(6);
            (widest as f64 * CHAR_W + PAD_X).clamp(MIN_NODE_W, MAX_NODE_W)
        })
        .collect();
    let heights: Vec<f64> = lines_per_block
        .iter()
        .map(|lines| lines.len() as f64 * LINE_H + PAD_Y)
        .collect();

    // Column geometry per layer: widest box defines the column.
    let mut col_w: BTreeMap<u32, f64> = BTreeMap::new();
    for (bi, l) in layer.iter().enumerate() {
        col_w
            .entry(*l)
            .and_modify(|w| *w = (*w).max(widths[bi]))
            .or_insert(widths[bi]);
    }

    // ---- place nodes -----------------------------------------------------------
    // Column left edge for layer L = Σ widths of narrower layers + gaps.
    let mut prefix: BTreeMap<u32, f64> = BTreeMap::new();
    let mut acc = MARGIN;
    for (l, w) in &col_w {
        prefix.insert(*l, acc);
        acc += w + COL_GAP;
    }
    let mut used_in_col: BTreeMap<u32, f64> = BTreeMap::new();
    let mut cx = vec![MARGIN; n];
    for &bi in &order {
        let l = layer[bi];
        let col_left = prefix.get(&l).copied().unwrap_or(MARGIN);
        let slot = used_in_col.entry(l).or_insert(col_left);
        cx[bi] = *slot + widths[bi] / 2.0;
        *slot += widths[bi] + NODE_GAP_X;
    }

    let mut nodes = Vec::with_capacity(n);
    for bi in 0..n {
        let (w, h) = (widths[bi], heights[bi]);
        let x = (cx[bi] - w / 2.0).max(0.0);
        let y = MARGIN + layer[bi] as f64 * ROW_GAP;
        nodes.push(CfgNode {
            id: format!("bb{bi}"),
            x,
            y,
            w,
            h,
            role: role[bi],
            lines: lines_per_block[bi].clone(),
            loop_id: loop_of[bi],
        });
    }

    // ---- route edges -------------------------------------------------------------
    let mut raw_edges: Vec<RawEdge> = Vec::new();
    for bi in 0..n {
        match ir.term(helix_ir::BlockId(bi as u32)) {
            helix_ir::Term::Jump(t, _) => raw_edges.push(RawEdge {
                from: bi as u32,
                to: t.0,
                kind: classify(bi as u32, t.0, &backedge),
                label: "",
            }),
            helix_ir::Term::Branch { t, f, .. } => {
                raw_edges.push(RawEdge {
                    from: bi as u32,
                    to: t.0,
                    kind: classify(bi as u32, t.0, &backedge),
                    label: "T",
                });
                raw_edges.push(RawEdge {
                    from: bi as u32,
                    to: f.0,
                    kind: classify(bi as u32, f.0, &backedge),
                    label: "F",
                });
            }
            helix_ir::Term::Return(_) => {}
        }
    }
    let edges: Vec<CfgEdge> = raw_edges.iter().map(|e| route_edge(e, &nodes)).collect();

    CfgFunction {
        name: name.to_string(),
        nodes,
        edges,
    }
}

/// Backedge test shared by layering and routing.
fn classify(from: u32, to: u32, backedge: &HashSet<(u32, u32)>) -> EdgeKind {
    if backedge.contains(&(from, to)) {
        EdgeKind::Backedge
    } else {
        EdgeKind::Fallthrough
    }
}

/// DFS preorder over successor lists, entry first, then any unreachable
/// blocks in id order (they still deserve rectangles). Iterative — deep
/// chains cannot blow the native stack.
fn discovery_order(ir: &helix_ir::FuncIr, n: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(n);
    let mut seen = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();
    for root in 0..n {
        if seen[root] {
            continue;
        }
        seen[root] = true;
        stack.push(root);
        while let Some(b) = stack.pop() {
            out.push(b);
            // Reverse-push keeps successor visit order = terminator order.
            for s in ir.succs(helix_ir::BlockId(b as u32)).iter().rev() {
                let si = s.0 as usize;
                if !seen[si] {
                    seen[si] = true;
                    stack.push(si);
                }
            }
        }
    }
    out
}

/// Longest-path layering: `layer[s] ≥ layer[b] + 1` for every forward edge.
///
/// Relax Bellman-Ford style (bounded by block count — the forward-edge graph
/// is acyclic, so this converges quickly). Backedges are skipped, which is
/// exactly what breaks the cycle and keeps loop bodies beside their header.
fn longest_path_layers(
    ir: &helix_ir::FuncIr,
    n: usize,
    backedge: &HashSet<(u32, u32)>,
) -> Vec<u32> {
    let mut layer = vec![0u32; n];
    for _round in 0..n.max(1) {
        let mut changed = false;
        for b in 0..n {
            let lb = layer[b];
            for s in ir.succs(helix_ir::BlockId(b as u32)) {
                let si = s.0 as usize;
                if backedge.contains(&(b as u32, s.0)) || layer[si] > lb {
                    continue;
                }
                layer[si] = lb + 1;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    layer
}

/// Rendered instruction lines of one block (φs first, terminator last) — the
/// same content `print_ir` writes, minus the `fn`/`bbN:` scaffolding.
fn block_lines(ir: &helix_ir::FuncIr, b: helix_ir::BlockId) -> Vec<String> {
    let block = ir.block(b);
    let mut lines = Vec::new();
    for p in &block.phis {
        if p.args.is_empty() {
            lines.push(format!("v{} = param", p.dst.0));
        } else {
            let args = p
                .args
                .iter()
                .map(|(from, v)| format!("[bb{}: v{}]", from.0, v.0))
                .collect::<Vec<_>>()
                .join(" ");
            lines.push(format!("v{} = φ {args}", p.dst.0));
        }
    }
    for inst in &block.insts {
        lines.push(inst_line_short(inst));
    }
    match ir.term(b) {
        helix_ir::Term::Jump(t, _) => lines.push(format!("jump bb{}", t.0)),
        helix_ir::Term::Branch { t, f, .. } => {
            lines.push(format!("branch ? bb{} : bb{}", t.0, f.0));
        }
        helix_ir::Term::Return(None) => lines.push("return".to_string()),
        helix_ir::Term::Return(Some(v)) => lines.push(format!("return v{}", v.0)),
    }
    lines
}

/// Compact instruction spelling for box lines. Ids stay numeric (stable,
/// short) — full pretty names live in the IR panes.
fn inst_line_short(i: &helix_ir::Inst) -> String {
    match i {
        helix_ir::Inst::Const { dst, c } => format!("v{} = const {}", dst.0, const_text(c)),
        helix_ir::Inst::Bin { op, dst, a, b } => {
            format!("v{} = {} v{}, v{}", dst.0, op.symbol(), a.0, b.0)
        }
        helix_ir::Inst::Unary { op, dst, a } => format!("v{} = {} v{}", dst.0, op.symbol(), a.0),
        helix_ir::Inst::Cast { dst, val, to } => {
            format!("v{} = cast v{} as {}", dst.0, val.0, to.name())
        }
        helix_ir::Inst::Load(l) => format!("v{} = load v{}", l.dst.0, l.idx.0),
        helix_ir::Inst::Store { idx, val, .. } => format!("store [..] = v{}, v{}", idx.0, val.0),
        helix_ir::Inst::Call(c) => {
            let args = c
                .args
                .iter()
                .map(|a| format!("v{}", a.0))
                .collect::<Vec<_>>();
            match c.dst {
                Some(d) => format!("v{} = call {}({})", d.0, c.callee, args.join(",")),
                None => format!("call {}({})", c.callee, args.join(",")),
            }
        }
    }
}

/// Constant rendering kept deliberately terse for box lines.
fn const_text(c: &helix_ir::Constant) -> String {
    match c {
        helix_ir::Constant::I64(v) => v.to_string(),
        helix_ir::Constant::I32(v) => format!("{v}i"),
        helix_ir::Constant::F32(v) => format!("{v:?}f"),
        helix_ir::Constant::F64(v) => format!("{v:?}"),
        helix_ir::Constant::Bool(b) => b.to_string(),
    }
}

/// Routes one edge around/through the placed boxes.
///
/// * **Fallthrough**: vertical straight line when aligned, otherwise a soft
///   4-point elbow dropping out of the source bottom into the target top.
/// * **Branch**: leaves through a side port (true = right wall, false = left
///   wall) so the two arms never overlap; same-row targets are entered
///   through their facing wall.
/// * **Backedge**: 3 points `[start, control, end]` — the browser draws a
///   quadratic bezier bowing past the right side of both boxes.
fn route_edge(e: &RawEdge, nodes: &[CfgNode]) -> CfgEdge {
    let id = |i: u32| format!("bb{i}");
    let (Some(a), Some(b)) = (nodes.get(e.from as usize), nodes.get(e.to as usize)) else {
        return CfgEdge {
            from: id(e.from),
            to: id(e.to),
            kind: e.kind,
            points: Vec::new(),
            label: e.label.to_string(),
        };
    };

    match e.kind {
        EdgeKind::Backedge => {
            let start = [a.x + a.w, a.y + a.h * 0.7];
            let end = [b.x + b.w, b.y + b.h * 0.25];
            let bulge_x = (start[0] + 34.0).max(end[0] + 34.0);
            let ctrl = [bulge_x, (start[1] + end[1]) / 2.0];
            CfgEdge {
                from: a.id.clone(),
                to: b.id.clone(),
                kind: e.kind,
                points: vec![start, ctrl, end],
                label: e.label.to_string(),
            }
        }
        EdgeKind::Branch => {
            let start_x = if e.label == "F" { a.x } else { a.x + a.w };
            let start = [start_x, a.y + a.h * 0.5];
            let same_row = (a.y - b.y).abs() < 1.0;
            let points = if same_row {
                // Sideways hop: enter through the facing wall of the target.
                let end_x = if start_x < b.x { b.x } else { b.x + b.w };
                let mid_x = (start_x + end_x) / 2.0;
                let end_y = b.y + b.h * 0.5;
                vec![start, [mid_x, start[1]], [mid_x, end_y], [end_x, end_y]]
            } else {
                // Elbow out of the side wall, drop past the source row,
                // enter the target top.
                let mid_y = a.y + a.h + (b.y - a.y - a.h).max(0.0) / 2.0 + LINE_H / 2.0;
                vec![
                    start,
                    [start[0], mid_y],
                    [b.x + b.w / 2.0, mid_y],
                    [b.x + b.w / 2.0, b.y],
                ]
            };
            CfgEdge {
                from: a.id.clone(),
                to: b.id.clone(),
                kind: e.kind,
                points,
                label: e.label.to_string(),
            }
        }
        EdgeKind::Fallthrough => {
            let start = [a.x + a.w / 2.0, a.y + a.h];
            let end = [b.x + b.w / 2.0, b.y];
            let dx = end[0] - start[0];
            let points = if dx.abs() < 4.0 {
                vec![start, end]
            } else {
                let bend_y = start[1] + (end[1] - start[1]).max(0.0) * 0.55 + LINE_H / 2.0;
                vec![start, [start[0], bend_y], [end[0], bend_y], end]
            };
            CfgEdge {
                from: a.id.clone(),
                to: b.id.clone(),
                kind: e.kind,
                points,
                label: String::new(),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Recorded evidence attachment
// ---------------------------------------------------------------------------

/// Builds the artifact for a stored example and attaches its matching
/// recorded bench point, when the source is the measured one.
///
/// A point with the most complete variant set is preferred (the smaller
/// correctness-sized point in the current campaign), then the smallest size
/// wins ties so interpreter/native comparisons remain visible. Non-kernel
/// teaching examples fall back to their dated interpreter probe.
///
/// Attachment needs measured-source identity, so a merely similarly named
/// example or an edited one gets no bench section — see [`recorded`].
#[must_use]
pub fn artifact_for_example(example: &str, source: &str) -> CompileArtifact {
    let mut art = build_artifact(example, source);
    if art.diags_lex.is_none() && art.diags_sem.is_empty() {
        art.bench = recorded::recorded_bench(example, source)
            .or_else(|| recorded::probe_bench(example, source));
    }
    art
}

// ---------------------------------------------------------------------------
// Benchmark index
// ---------------------------------------------------------------------------

/// Compact index of the committed measured campaign, in the shape the
/// Observatory's `/api/benchmarks` route serves.
///
/// Raw samples stay in `docs/benchmarks/data/campaign.json`; the index only
/// carries provenance, sizes, and the list of kernels for honest availability
/// labels. Probe provenance is kept separate from campaign provenance: neither
/// says anything about the current build's performance.
///
/// Both catalogs are embedded at compile time, so the only failure left is a
/// malformed commit. That keeps the index answerable as JSON: the reason
/// travels in an `error` field instead of taking the route down.
#[must_use]
pub fn bench_index() -> serde_json::Value {
    let Ok(campaign) = recorded::read_campaign() else {
        return serde_json::json!({ "error": "benchmark campaign is unavailable" });
    };

    let mut sizes: BTreeMap<String, Vec<u64>> = BTreeMap::new();
    for point in &campaign.points {
        if let Ok(n) = u64::try_from(point.n) {
            sizes.entry(point.kernel.clone()).or_default().push(n);
        }
    }
    // Keep the index compact, but retain enough measured evidence for the UI
    // to compare each recorded size without shipping the raw sample arrays.
    // This is additive to the original catalog shape.
    let points = campaign
        .points
        .iter()
        .filter_map(|point| {
            let n = u64::try_from(point.n).ok()?;
            Some(serde_json::json!({
                "kernel": point.kernel,
                "n": n,
                "variants": point.variants.iter().map(|variant| serde_json::json!({
                    "name": variant.name,
                    "median_ms": variant.median_ms,
                    "sample_count": variant.samples_ms.len(),
                })).collect::<Vec<_>>(),
                "efficiency_points": point.efficiency.len(),
            }))
        })
        .collect::<Vec<_>>();
    let kernels = sizes
        .into_iter()
        .map(|(name, mut ns)| {
            ns.sort_unstable();
            ns.dedup();
            serde_json::json!({ "name": name, "sizes": ns })
        })
        .collect::<Vec<_>>();
    let probes = recorded::read_example_probes().ok();
    let probe_provenance = probes.as_ref().map(|catalog| {
        serde_json::json!({
            "schema": catalog.schema,
            "timestamp_utc": catalog.timestamp_utc,
            "git_rev": catalog.machine.git_rev,
            "machine": {
                "cpu_brand": catalog.machine.cpu_brand,
                "logical_cores": catalog.machine.logical_cores,
                "physical_cores": catalog.machine.physical_cores,
                "rustc_version": catalog.machine.rustc_version,
            },
        })
    });
    let examples = probes
        .map(|catalog| {
            catalog
                .examples
                .into_iter()
                .map(|probe| {
                    serde_json::json!({
                        "name": probe.example,
                        "median_ms": probe.median_ms,
                        "sample_count": probe.samples_ms.len(),
                    })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    serde_json::json!({
        "schema": campaign.schema,
        "timestamp_utc": campaign.timestamp_utc,
        "git_rev": campaign.meta.git_rev,
        "point_count": campaign.points.len(),
        "kernels": kernels,
        "examples": examples,
        "probe_provenance": probe_provenance,
        "points": points,
        "machine": {
            "cpu_brand": campaign.meta.cpu_brand,
            "logical_cores": campaign.meta.logical_cores,
            "physical_cores": campaign.meta.physical_cores,
            "rustc_version": campaign.meta.rustc_version,
        },
    })
}
