//! Server-side graph layout: the browser paints coordinates, it never
//! computes them (`artifact-schema.md` rule: "cfg layout coordinates are
//! FINAL"). Two tidy algorithms live here.
//!
//! * [`ast_tree`] — Reingold–Tilford-flavoured layered tree over the display
//!   hierarchy produced from a serde `Program`: depth ⇒ row (`y`), in-order
//!   leaf slots ⇒ column (`x`), parents centred over their children. Small,
//!   dependency-free, and deterministic for identical trees.
//! * [`program_to_tree`] — the adapter converting externally-tagged serde
//!   enum JSON (`{"Bin": ["+", …]}`) into that hierarchy, labelling nodes by
//!   variant plus a payload summary so the picture reads like the grammar.
//! * [`cfg_layout`] — per-function CFG layout: longest-path layering from the
//!   entry block (backedges excluded ⇒ acyclic ⇒ termination), DFS-discovery
//!   order within each layer, monospace box sizing, and edge routing with
//!   straight/elbow polylines plus 3-point quadratic curves for backedges.
//!
//! Every coordinate is finite by construction (no division, no trig); the
//! artifact tests assert this anyway so NaN can never reach the SVG.

use std::collections::{BTreeMap, HashSet};

use crate::artifact::{BlockRole, CfgEdge, CfgFunction, CfgNode, EdgeKind};

// ---------------------------------------------------------------------------
// Shared geometry constants
// ---------------------------------------------------------------------------

/// Horizontal gap between neighbouring leaf slots / CFG columns.
const COL_GAP: f64 = 46.0;
/// Minimum horizontal room a leaf claims (its own label must fit).
const LEAF_MIN_W: f64 = 90.0;
/// Vertical gap between tree depths / CFG rows.
const ROW_GAP: f64 = 78.0;
/// Canvas margin on every side.
pub(crate) const MARGIN: f64 = 28.0;
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
// AST tidy tree
// ---------------------------------------------------------------------------

/// One node of the display hierarchy handed to [`ast_tree`].
#[derive(Debug, Clone)]
pub struct TreeNode {
    /// Primary label (`"FnDef"`, `"Bin(+)"`, `"42"`).
    pub label: String,
    /// Secondary detail line (`"main"`, a type, a literal tag).
    pub detail: String,
    /// Ordered children.
    pub children: Vec<TreeNode>,
}

impl TreeNode {
    /// Builds a leaf (label + optional detail).
    #[must_use]
    pub fn leaf(label: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            detail: detail.into(),
            children: Vec::new(),
        }
    }

    /// Builds an internal node.
    #[must_use]
    pub fn node(
        label: impl Into<String>,
        detail: impl Into<String>,
        children: Vec<TreeNode>,
    ) -> Self {
        Self {
            label: label.into(),
            detail: detail.into(),
            children,
        }
    }
}

/// A laid-out tree node: payload plus final canvas coordinates.
#[derive(Debug, Clone)]
pub struct LaidOutNode {
    /// Payload subtree rooted here.
    pub node: TreeNode,
    /// Final x (column centre), canvas pixels.
    pub x: f64,
    /// Final y (row top), canvas pixels.
    pub y: f64,
}

/// Lay out a tidy tree.
///
/// Leaves claim successive column slots (in-order traversal); an internal
/// node is centred over its children. Depth `d` lands at row `d`. Returns
/// nodes in preorder with the root first.
#[must_use]
pub fn ast_tree(root: &TreeNode) -> Vec<LaidOutNode> {
    let mut out = Vec::new();
    let mut cursor = MARGIN; // next free leaf-slot centre
    layout_node(root, 0, &mut cursor, &mut out);
    out
}

/// Recursive worker. `cursor` is advanced past everything this subtree uses,
/// so sibling subtrees can never overlap.
fn layout_node(node: &TreeNode, depth: usize, cursor: &mut f64, out: &mut Vec<LaidOutNode>) {
    let y = MARGIN + depth as f64 * ROW_GAP;
    if node.children.is_empty() {
        out.push(LaidOutNode {
            node: node.clone(),
            x: *cursor,
            y,
        });
        *cursor += LEAF_MIN_W.max(COL_GAP);
        return;
    }
    let first_child = out.len();
    for child in &node.children {
        layout_node(child, depth + 1, cursor, out);
    }
    let kids = &out[first_child..];
    let left = kids.first().map_or(*cursor, |k| k.x);
    let right = kids.last().map_or(left, |k| k.x);
    let x = (left + right) / 2.0;
    out.insert(
        first_child,
        LaidOutNode {
            node: node.clone(),
            x,
            y,
        },
    );
}

// ---------------------------------------------------------------------------
// AST adapter — serde_json Program → TreeNode hierarchy
// ---------------------------------------------------------------------------

/// Converts the raw serialized [`helix_syntax::ast::Program`] JSON into the
/// parent-child hierarchy the AST view draws.
///
/// Externally tagged enums arrive as `{"Variant": payload}` — tuple variants
/// carry an array payload, struct variants an object. Nodes are labelled by
/// variant plus a short payload summary (`Let i`, `Bin(*)`, literal values).
/// Returns `None` only when the JSON is not object-shaped (the caller then
/// simply omits the AST view).
#[must_use]
pub fn program_to_tree(ast_json: &serde_json::Value) -> Option<TreeNode> {
    let items = ast_json.get("items")?.as_array()?;
    let kids = items
        .iter()
        .map(|item| match variant_of(item) {
            ("Fn", payload) => fn_def_tree(payload),
            ("Const", payload) => const_def_tree(payload),
            _ => generic_tree("Item", item),
        })
        .collect();
    Some(TreeNode::node("Program", items.len().to_string(), kids))
}

/// `{"Variant": payload}` → `(variant, payload)`; plain objects yield
/// `("", self)`.
fn variant_of(value: &serde_json::Value) -> (&str, &serde_json::Value) {
    match value {
        serde_json::Value::Object(map) if map.len() == 1 => {
            let (k, v) = map.iter().next().expect("len == 1");
            (k.as_str(), v)
        }
        _ => ("", value),
    }
}

/// `FnDef` → `FnDef name` node with param/ret/body children.
fn fn_def_tree(f: &serde_json::Value) -> TreeNode {
    let name = ident_text(f.get("name"));
    let mut kids = Vec::new();
    if let Some(params) = f.get("params").and_then(|p| p.as_array()) {
        for p in params {
            kids.push(TreeNode::leaf(
                format!("Param {}", ident_text(p.get("name"))),
                ty_text(p.get("ty")),
            ));
        }
    }
    if let Some(ret) = f.get("ret") {
        kids.push(TreeNode::leaf("Ret", ty_text(Some(ret))));
    }
    kids.extend(block_children(f.get("body")));
    TreeNode::node("FnDef", name, kids)
}

/// `ConstDef` → node showing `NAME: ty` with the value as detail.
fn const_def_tree(c: &serde_json::Value) -> TreeNode {
    let name = ident_text(c.get("name"));
    let ty = ty_text(c.get("ty"));
    let val = c
        .get("value")
        .map(literal_text)
        .unwrap_or_else(|| "?".into());
    TreeNode::leaf(format!("Const {name}: {ty}"), format!("= {val}"))
}

/// Renders a `Block`'s statements as child nodes behind a `Block n` header.
fn block_children(body: Option<&serde_json::Value>) -> Vec<TreeNode> {
    let stmts = body.and_then(|b| b.get("stmts")).and_then(|s| s.as_array());
    let mut kids = vec![TreeNode::leaf(
        "Block",
        stmts.map_or("0".into(), |a| a.len().to_string()),
    )];
    for s in stmts.into_iter().flatten() {
        kids.push(stmt_tree(s));
    }
    kids
}

/// Statement dispatcher — mirrors `helix_syntax::ast::Stmt`.
fn stmt_tree(stmt: &serde_json::Value) -> TreeNode {
    let (variant, p) = variant_of(stmt);
    match variant {
        "Let" => {
            let mut kids = Vec::new();
            if let Some(ty) = p.get("ty") {
                kids.push(TreeNode::leaf("Ty", ty_text(Some(ty))));
            }
            if let Some(init) = p.get("init") {
                kids.push(expr_tree(init));
            }
            TreeNode::node("Let", ident_text(p.get("name")), kids)
        }
        "Assign" => TreeNode::node(
            "Assign",
            "",
            vec![
                p.get("target")
                    .map(lvalue_tree)
                    .unwrap_or_else(|| TreeNode::leaf("LVal", "?")),
                p.get("value")
                    .map(expr_tree)
                    .unwrap_or_else(|| TreeNode::leaf("Expr", "?")),
            ],
        ),
        "If" => {
            let mut kids = vec![
                p.get("cond")
                    .map(expr_tree)
                    .unwrap_or_else(|| TreeNode::leaf("Cond", "?")),
            ];
            kids.extend(block_children(p.get("then_blk")));
            if let Some(ep) = p.get("else_part") {
                match ep.get("If").or_else(|| ep.get("Block")) {
                    Some(inner) if inner.get("stmts").is_some() => {
                        kids.extend(block_children(Some(inner)));
                    }
                    Some(inner) => kids.push(stmt_tree(inner)),
                    None => kids.push(TreeNode::leaf("Else", "?")),
                }
            }
            TreeNode::node("If", "", kids)
        }
        "For" => {
            let mut kids = vec![
                p.get("start")
                    .map(expr_tree)
                    .unwrap_or_else(|| TreeNode::leaf("Start", "?")),
                p.get("end")
                    .map(expr_tree)
                    .unwrap_or_else(|| TreeNode::leaf("End", "?")),
            ];
            kids.extend(block_children(p.get("body")));
            TreeNode::node("For", ident_text(p.get("iv")), kids)
        }
        "Return" => TreeNode::node(
            "Return",
            "",
            p.get("value")
                .filter(|v| !v.is_null())
                .map(|v| vec![expr_tree(v)])
                .unwrap_or_default(),
        ),
        "Expr" => TreeNode::node("ExprStmt", "", vec![expr_tree(p)]),
        "Empty" => TreeNode::leaf("Empty", ""),
        "Block" => TreeNode::node(
            "Block",
            "",
            p.get("stmts")
                .and_then(|s| s.as_array())
                .map(|a| a.iter().map(stmt_tree).collect())
                .unwrap_or_default(),
        ),
        other => generic_tree(other, p),
    }
}

/// `LValue` → `Var base` / `Index base[idx]`.
fn lvalue_tree(lv: &serde_json::Value) -> TreeNode {
    let base = ident_text(lv.get("base"));
    match lv.get("index").filter(|i| !i.is_null()) {
        Some(idx) => TreeNode::node("Index", base, vec![expr_tree(idx)]),
        None => TreeNode::leaf("Var", base),
    }
}

/// Expression dispatcher — mirrors `helix_syntax::ast::Expr`; tuple variants
/// address fields positionally, struct variants by name.
fn expr_tree(expr: &serde_json::Value) -> TreeNode {
    let (variant, p) = variant_of(expr);
    let fields = p.as_array();
    let field = |i: usize| -> Option<&serde_json::Value> {
        fields.and_then(|f| f.get(i)).filter(|v| !v.is_null())
    };
    match variant {
        "IntLit" => TreeNode::leaf(field(0).map(value_text).unwrap_or_default(), "int"),
        "FloatLit" => TreeNode::leaf(field(0).map(value_text).unwrap_or_default(), "float"),
        "Bool" => TreeNode::leaf(field(0).map(value_text).unwrap_or_default(), "bool"),
        "Var" => TreeNode::leaf(ident_text(Some(p)), "var"),
        "Unary" => TreeNode::node(
            format!("Un({})", field(0).map(op_symbol).unwrap_or_default()),
            "",
            field(1).map(|e| vec![expr_tree(e)]).unwrap_or_default(),
        ),
        "Bin" => TreeNode::node(
            format!("Bin({})", field(0).map(op_symbol).unwrap_or_default()),
            "",
            [field(1), field(2)]
                .into_iter()
                .flatten()
                .map(expr_tree)
                .collect(),
        ),
        "Index" => TreeNode::node(
            "Index",
            ident_text(field(0)),
            field(1).map(|e| vec![expr_tree(e)]).unwrap_or_default(),
        ),
        "Call" => TreeNode::node(
            format!("{}()", ident_text(p.get("callee"))),
            "",
            p.get("args")
                .and_then(|a| a.as_array())
                .map(|a| a.iter().map(expr_tree).collect())
                .unwrap_or_default(),
        ),
        "Cast" => TreeNode::node(
            "Cast",
            ty_text(field(1)),
            field(0).map(|e| vec![expr_tree(e)]).unwrap_or_default(),
        ),
        other => generic_tree(other, p),
    }
}

/// Fallback renderer for anything unanticipated: labels by shape.
fn generic_tree(label: &str, value: &serde_json::Value) -> TreeNode {
    match value {
        serde_json::Value::Array(items) => TreeNode::node(
            format!("{label}[…]"),
            items.len().to_string(),
            items.iter().map(|v| generic_tree("", v)).collect(),
        ),
        serde_json::Value::Object(map) => TreeNode::node(
            label.to_string(),
            "",
            map.iter()
                .filter(|(_, v)| !v.is_null())
                .map(|(k, v)| generic_tree(k, v))
                .collect(),
        ),
        scalar => TreeNode::leaf(label.to_string(), value_text(scalar)),
    }
}

// -- little render helpers ---------------------------------------------------

/// Identifier text (`{"name":"x","span":…}` → `"x"`).
fn ident_text(v: Option<&serde_json::Value>) -> String {
    v.and_then(|v| v.get("name"))
        .and_then(|n| n.as_str())
        .unwrap_or("?")
        .to_string()
}

/// Type rendering mirroring `Type::render` over the serde shape.
fn ty_text(v: Option<&serde_json::Value>) -> String {
    let Some(ty) = v else { return "?".into() };
    let (variant, payload) = variant_of(ty);
    match variant {
        "I32" => "i32".into(),
        "I64" => "i64".into(),
        "F32" => "f32".into(),
        "F64" => "f64".into(),
        "Bool" => "bool".into(),
        "Unit" => "()".into(),
        "Array" => format!("[{}]", scalar_name(payload)),
        "" => "?".into(),
        other => other.to_lowercase(),
    }
}

/// Scalar element spelling inside `[T]`.
fn scalar_name(v: &serde_json::Value) -> String {
    match variant_of(v).0 {
        "I32" => "i32".into(),
        "I64" => "i64".into(),
        "F32" => "f32".into(),
        "F64" => "f64".into(),
        "Bool" => "bool".into(),
        other => other.into(),
    }
}

/// Literal payload spelling (`{"Int": 3}` → `"3"`).
fn literal_text(v: &serde_json::Value) -> String {
    value_text(variant_of(v).1)
}

/// Operator symbol for `UnOp`/`BinOp` payloads (`"Add"` → `"+"`).
fn op_symbol(v: &serde_json::Value) -> String {
    match variant_of(v).0 {
        "Add" => "+",
        "Sub" => "-",
        "Mul" => "*",
        "Div" => "/",
        "Rem" => "%",
        "Lt" => "<",
        "Gt" => ">",
        "Le" => "<=",
        "Ge" => ">=",
        "Eq" => "==",
        "Ne" => "!=",
        "And" => "&&",
        "Or" => "||",
        "Neg" => "-",
        "Not" => "!",
        other => other,
    }
    .to_string()
}

/// Any JSON scalar as display text.
fn value_text(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Null => String::new(),
        other => other.to_string(),
    }
}

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
