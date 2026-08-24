//! Server-side graph layout: the browser paints coordinates, it never
//! computes them (`artifact-schema.md` rule). Two tidy algorithms live here.
//!
//! * [`ast_tree`] — Reingold–Tilford-flavoured layered tree over the serde
//!   JSON of a `Program`. Depth ⇒ row (`y`), in-order leaf slots ⇒ column
//!   (`x`), parents centred over their children. ~60 lines, no dependencies,
//!   and deterministic for identical trees — golden-test friendly.
//! * [`cfg_layout`] — longest-path layering from the entry block (backedges
//!   ignored for layering), DFS-discovery order within each column, block
//!   boxes sized from their monospace line estimate, straight/elbow/curve
//!   edge routing with 3-point quadratic beziers for backedges.
//!
//! All arithmetic is `f64`; every emitted coordinate is checked finite by
//! [`CompileArtifact`] tests, so NaN can never reach the SVG.

use std::collections::BTreeMap;

use crate::artifact::{BlockRole, CfgEdge, CfgFunction, CfgNode, EdgeKind};

// ---------------------------------------------------------------------------
// Shared geometry constants
// ---------------------------------------------------------------------------

/// Horizontal gap between sibling subtrees / CFG columns.
pub(crate) const COL_GAP: f64 = 46.0;
/// Vertical gap between tree depths / CFG rows.
pub(crate) const ROW_GAP: f64 = 74.0;
/// Left/top margin of the canvas.
pub(crate) const MARGIN: f64 = 28.0;
/// Monospace glyph advance used to size CFG boxes (px per char).
const CHAR_W: f64 = 9.0;
/// Horizontal padding inside a CFG box.
const PAD_X: f64 = 24.0;
/// Line height inside a CFG box.
const LINE_H: f64 = 18.0;
/// Vertical padding of a CFG box (title row + bottom air).
const PAD_Y: f64 = 26.0;
/// Minimum box width so tiny blocks still fit `"bb0 · exit"`.
const MIN_NODE_W: f64 = 120.0;

// ---------------------------------------------------------------------------
// AST tidy tree
// ---------------------------------------------------------------------------

/// One node of the display hierarchy handed to [`ast_tree`].
#[derive(Debug, Clone)]
pub struct TreeNode {
    /// Short kind label rendered as the primary text (`"FnDef"`, `"Bin(+)"`).
    pub label: String,
    /// Secondary detail line (`"main"`, literal value, type name).
    pub detail: String,
    /// Ordered children.
    pub children: Vec<TreeNode>,
}

impl TreeNode {
    /// Builds a leaf.
    #[must_use]
    pub fn leaf(label: impl Into<String>, detail: impl Into<String>) -> Self {
        Self { label: label.into(), detail: detail.into(), children: Vec::new() }
    }

    /// Builds an internal node from labelled children.
    #[must_use]
    pub fn node(
        label: impl Into<String>,
        detail: impl Into<String>,
        children: Vec<TreeNode>,
    ) -> Self {
        Self { label: label.into(), detail: detail.into(), children }
    }
}

/// A laid-out tree node: payload plus final canvas coordinates.
#[derive(Debug, Clone)]
pub struct LaidOutNode {
    /// The payload subtree rooted here.
    pub node: TreeNode,
    /// Final x (column centre), canvas pixels.
    pub x: f64,
    /// Final y (row top), canvas pixels.
    pub y: f64,
}

/// Lay out a tidy tree: depth ⇒ row, in-order leaves ⇒ columns, parents
/// centred over children (the textbook Reingold–Tilford simplification).
///
/// Returns nodes in preorder; index 0 is always the root.
#[must_use]
pub fn ast_tree(root: &TreeNode) -> Vec<LaidOutNode> {
    let mut out = Vec::new();
    if root.label.is_empty() && root.children.is_empty() && root.detail.is_empty() {
        // Degenerate empty payload still renders one dot rather than nothing.
    }
    let mut cursor = 0.0f64; // next free leaf slot
    layout(root, 0, &mut cursor, &mut out);
    out
}

/// Recursive worker: assigns `x` after all children claim leaf slots.
fn layout(
    node: &TreeNode,
    depth: usize,
    cursor: &mut f64,
    out: &mut Vec<LaidOutNode>,
) {
    let y = MARGIN + depth as f64 * ROW_GAP;
    if node.children.is_empty() {
        let x = *cursor;
        *cursor += COL_GAP.max(90.0); // leaves need room for their own label
        out.push(LaidOutNode { node: node.clone(), x, y });
        return;
    }
    let first_child = out.len();
    for child in &node.children {
        layout(child, depth + 1, cursor, out);
    }
    let kids = &out[first_child..];
    let left = kids.first().map_or(0.0, |k| k.x);
    let right = kids.last().map_or(0.0, |k| k.x);
    let x = (left + right) / 2.0;
    // A wide internal label may not overlap its left sibling; nudge right.
    if x < *cursor - 40.0 {
        *cursor = x + 40.0 + 40.0;
    } else {
        *cursor = (*cursor).max(x);
    }
    out.insert(first_child, LaidOutNode { node: node.clone(), x, y });
}

// ---------------------------------------------------------------------------
// AST adapter — serde_json Program → TreeNode hierarchy
// ---------------------------------------------------------------------------

/// Converts the raw serialized [`helix_syntax::ast::Program`] JSON into the
/// parent-child hierarchy the AST view draws.
///
/// Externally tagged enums arrive as `{"Variant": payload}` (tuple variants:
/// payload is an array; struct variants: payload is an object). Labels are
/// formed from the variant plus a short payload summary so the picture reads
/// like the grammar.
#[must_use]
pub fn program_to_tree(ast_json: &serde_json::Value) -> Option<TreeNode> {
    let items = ast_json.get("items")?.as_array()?;
    let mut kids = Vec::new();
    for item in items {
        match variant_of(item) {
            ("Fn", payload) => kids.push(fn_def_tree(payload)),
            ("Const", payload) => kids.push(const_def_tree(payload)),
            _ => kids.push(generic_tree("Item", item)),
        }
    }
    Some(TreeNode::node("Program", items.len().to_string(), kids))
}

/// `{"Variant": payload}` → `(variant, payload)`; plain objects yield
/// `("", self)`.
fn variant_of(value: &serde_json::Value) -> (&str, &serde_json::Value) {
    match value {
        serde_json::Value::Object(map) => {
            if map.len() == 1 {
                let (k, v) = map.iter().next().expect("len == 1");
                return (k.as_str(), v);
            }
            ("", value)
        }
        _ => ("", value),
    }
}

/// `FnDef` → `FnDef name` node with params/ret/body children.
fn fn_def_tree(f: &serde_json::Value) -> TreeNode {
    let name = ident_text(f.get("name"));
    let mut kids = Vec::new();
    if let Some(params) = f.get("params").and_then(|p| p.as_array()) {
        for p in params {
            let pname = ident_text(p.get("name"));
            let ty = ty_text(p.get("ty"));
            kids.push(TreeNode::leaf(format!("Param {pname}"), ty));
        }
    }
    if let Some(ret) = f.get("ret") {
        kids.push(TreeNode::leaf("Ret", ty_text(Some(ret))));
    }
    kids.extend(block_children(f.get("body")));
    TreeNode::node("FnDef", name, kids)
}

/// `ConstDef` → leaf-ish node showing `NAME: ty = value`.
fn const_def_tree(c: &serde_json::Value) -> TreeNode {
    let name = ident_text(c.get("name"));
    let ty = ty_text(c.get("ty"));
    let val = c
        .get("value")
        .map(literal_text)
        .unwrap_or_else(|| "?".to_string());
    TreeNode::leaf(format!("Const {name}: {ty}"), format!("= {val}"))
}

/// Renders a `Block`'s statements as child nodes.
fn block_children(body: Option<&serde_json::Value>) -> Vec<TreeNode> {
    let stmts = body
        .and_then(|b| b.get("stmts"))
        .and_then(|s| s.as_array())
        .map_or(&[][..], std::vec::Vec::as_slice);
    let mut kids = vec![TreeNode::leaf("Block", stmts.len().to_string())];
    for s in stmts {
        kids.push(stmt_tree(s));
    }
    kids
}

/// Statement dispatcher — mirrors `helix_syntax::ast::Stmt` variants.
fn stmt_tree(stmt: &serde_json::Value) -> TreeNode {
    let (variant, payload) = variant_of(stmt);
    match variant {
        "Let" => {
            let mut kids = Vec::new();
            if let Some(ty) = payload.get("ty") {
                kids.push(TreeNode::leaf("Ty", ty_text(Some(ty))));
            }
            if let Some(init) = payload.get("init") {
                kids.push(expr_tree(init));
            }
            TreeNode::node("Let", ident_text(payload.get("name")), kids)
        }
        "Assign" => {
            let target = payload
                .get("target")
                .map(lvalue_tree)
                .unwrap_or_else(|| TreeNode::leaf("LVal", "?"));
            let value = payload
                .get("value")
                .map(expr_tree)
                .unwrap_or_else(|| TreeNode::leaf("Expr", "?"));
            TreeNode::node("Assign", "", vec![target, value])
        }
        "If" => {
            let mut kids = vec![
                payload
                    .get("cond")
                    .map(expr_tree)
                    .unwrap_or_else(|| TreeNode::leaf("Cond", "?")),
            ];
            kids.extend(block_children(payload.get("then_blk")));
            if let Some(else_part) = payload.get("else_part") {
                let inner = else_part.get("If").or_else(|| else_part.get("Block"));
                match inner {
                    Some(ep) if ep.get("stmts").is_some() => {
                        kids.extend(block_children(Some(ep)));
                    }
                    Some(ep) => kids.push(stmt_tree(ep)),
                    None => kids.push(TreeNode::leaf("Else", "?")),
                }
            }
            TreeNode::node("If", "", kids)
        }
        "For" => TreeNode::node(
            "For",
            ident_text(payload.get("iv")),
            vec![
                payload
                    .get("start")
                    .map(expr_tree)
                    .unwrap_or_else(|| TreeNode::leaf("Start", "?")),
                payload
                    .get("end")
                    .map(|e| expr_tree(e))
                    .unwrap_or_else(|| TreeNode::leaf("End", "?")),
            ]
            .into_iter()
            .chain(block_children(payload.get("body")))
            .collect(),
        ),
        "Return" => {
            let kids = payload
                .get("value")
                .filter(|v| !v.is_null())
                .map(|v| vec![expr_tree(v)])
                .unwrap_or_default();
            TreeNode::node("Return", "", kids)
        }
        "Expr" => TreeNode::node("ExprStmt", "", vec![expr_tree(payload)]),
        "Empty" => TreeNode::leaf("Empty", ""),
        "Block" => TreeNode::node(
            "Block",
            "",
            payload
                .get("stmts")
                .and_then(|s| s.as_array())
                .map(|a| a.iter().map(stmt_tree).collect())
                .unwrap_or_default(),
        ),
        other => generic_tree(other, payload),
    }
}

/// `LValue` → `Var base` / `Index base[expr]`.
fn lvalue_tree(lv: &serde_json::Value) -> TreeNode {
    let base = ident_text(lv.get("base"));
    match lv.get("index").filter(|i| !i.is_null()) {
        Some(idx) => TreeNode::node("Index", base, vec![expr_tree(idx)]),
        None => TreeNode::leaf("Var", base),
    }
}

/// Expression dispatcher — mirrors `helix_syntax::ast::Expr` tuple/struct
/// variants (payload arrays keep field order).
fn expr_tree(expr: &serde_json::Value) -> TreeNode {
    let (variant, payload) = variant_of(expr);
    let fields = payload.as_array();
    let field = |i: usize| -> Option<&serde_json::Value> {
        fields.and_then(|f| f.get(i)).filter(|v| !v.is_null())
    };
    match variant {
        "IntLit" => TreeNode::leaf(field(0).map(value_text).unwrap_or_default(), "int"),
        "FloatLit" => TreeNode::leaf(field(0).map(value_text).unwrap_or_default(), "float"),
        "Bool" => TreeNode::leaf(field(0).map(value_text).unwrap_or_default(), "bool"),
        "Var" => TreeNode::leaf(ident_text(Some(payload)), "var"),
        "Unary" => TreeNode::node(
            format!("Un({})", field(0).map(op_symbol).unwrap_or_default()),
            "",
            field(1).map(|e| vec![expr_tree(e)]).unwrap_or_default(),
        ),
        "Bin" => TreeNode::node(
            format!("Bin({})", field(0).map(op_symbol).unwrap_or_default()),
            "",
            vec![field(1), field(2)]
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
        "Call" => {
            let args = payload
                .get("args")
                .and_then(|a| a.as_array())
                .map(|a| a.iter().map(expr_tree).collect())
                .unwrap_or_default();
            TreeNode::node(format!("{}()", ident_text(payload.get("callee"))), "", args)
        }
        "Cast" => TreeNode::node(
            "Cast",
            ty_text(field(1)),
            field(0).map(|e| vec![expr_tree(e)]).unwrap_or_default(),
        ),
        other => generic_tree(other, payload),
    }
}

/// Fallback renderer for anything unanticipated: labels by shape.
fn generic_tree(label: &str, value: &serde_json::Value) -> TreeNode {
    match value {
        serde_json::Value::Array(items) => TreeNode::node(
            format!("{label}[…]") ,
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

/// Identifier text (`{name: "x", span…}` → `"x"`).
fn ident_text(v: Option<&serde_json::Value>) -> String {
    v.and_then(|v| v.get("name"))
        .and_then(|n| n.as_str())
        .unwrap_or("?")
        .to_string()
}

/// Type rendering mirroring `Type::render` on the serde shape.
fn ty_text(v: Option<&serde_json::Value>) -> String {
    let Some(ty) = v else { return "?".to_string() };
    let (variant, payload) = variant_of(ty);
    if variant.is_empty() {
        return "?".to_string();
    }
    match variant {
        "I32" => "i32".into(),
        "I64" => "i64".into(),
        "F32" => "f32".into(),
        "F64" => "f64".into(),
        "Bool" => "bool".into(),
        "Unit" => "()".into(),
        "Array" => format!("[{}]", scalar_name(payload)),
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
    let (_, payload) = variant_of(v);
    value_text(payload)
}

/// Operator symbol for Un/Bin payloads (`"Add"` → `"+"`).
fn op_symbol(v: &serde_json::Value) -> String {
    let sym = match variant_of(v).0 {
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
    };
    sym.to_string()
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

/// Input edge before routing (block ids only).
#[derive(Debug, Clone)]
struct RawEdge {
    from: u32,
    to: u32,
    kind: EdgeKind,
    label: String,
}

/// Lays out one function's CFG.
///
/// Algorithm (longest-path layering + barycentre-free DFS ordering):
///
/// 1. classify edges (backedge when `to` dominates `from`, i.e. loop latch);
/// 2. DFS from entry over non-backedges recording discovery order and depth;
/// 3. `layer[b] = max(layer[pred]) + 1` over discovered preds (backedges
///    ignored ⇒ acyclic ⇒ termination guaranteed);
/// 4. within-layer order = DFS discovery order, columns packed left→right;
/// 5. box size from the widest rendered line, centred in its column.
///
/// Edge routing happens after boxes exist so ports sit exactly on borders:
/// fallthroughs get a straight or elbow polyline, branch arms elbow through
/// side ports, backedges curve via a sideways control point (3 points ⇒
/// quadratic bezier in the browser).
#[must_use]
pub fn cfg_layout(name: &str, ir: &helix_ir::FuncIr, loops: &helix_analysis::LoopInfo) -> CfgFunction {
    let n = ir.blocks.len();

    // ---- roles + loop membership ------------------------------------------
    let doms = helix_ir::dominators(ir);
    let mut role = vec![BlockRole::Straight; n];
    if n > 0 && role[ir.entry.0 as usize] != BlockRole::LoopHeader {
        role[ir.entry.0 as usize] = BlockRole::Entry;
    }

    // Innermost loop containing each block (for colour coding + spotlight).
    let mut loop_of: Vec<Option<usize>> = vec![None; n];
    for lp in &loops.loops {
        for b in &lp.blocks {
            let i = b.0 as usize;
            if i < n
                && loop_of[i].is_none_or(|cur| {
                    loops.loops[cur].depth >= lp.depth
                })
            {
                loop_of[i] = Some(lp.id);
            }
        }
        let h = lp.header.0 as usize;
        if h < n {
            role[h] = BlockRole::LoopHeader;
        }
    }

    // ---- edge classification ------------------------------------------------
    let mut raw_edges: Vec<RawEdge> = Vec::new();
    for bi in 0..n {
        let id = helix_ir::BlockId(bi as u32);
        let term = ir.term(id);
        match term {
            helix_ir::Term::Jump(t, _) => raw_edges.push(RawEdge {
                from: bi as u32,
                to: t.0,
                kind: edge_kind(bi as u32, t.0, &doms),
                label: String::new(),
            }),
            helix_ir::Term::Branch { t, f, .. } => {
                raw_edges.push(RawEdge {
                    from: bi as u32,
                    to: t.0,
                    kind: edge_kind(bi as u32, t.0, &doms),
                    label: "T".to_string(),
                });
                raw_edges.push(RawEdge {
                    from: bi as u32,
                    to: f.0,
                    kind: edge_kind(bi as u32, f.0, &doms),
                    label: "F".to_string(),
                });
            }
            helix_ir::Term::Return(_) => {
                role[bi] = BlockRole::Exit;
            }
        }
    }

    // ---- layering + ordering -------------------------------------------------
    let (layer, order_pos) = layer_and_order(ir, n);

    // ---- box sizing -----------------------------------------------------------
    let lines_per_block: Vec<Vec<String>> = (0..n)
        .map(|bi| block_lines(ir, helix_ir::BlockId(bi as u32)))
        .collect();
    let widths: Vec<f64> = lines_per_block
        .iter()
        .map(|lines| {
            let widest = lines.iter().map(String::len).max().unwrap_or(6);
            (widest as f64 * CHAR_W + PAD_X).clamp(MIN_NODE_W, 460.0)
        })
        .collect();
    let heights: Vec<f64> = lines_per_block
        .iter()
        .map(|lines| lines.len() as f64 * LINE_H + PAD_Y)
        .collect();

    // Column width per layer accommodates its widest box.
    let mut col_w: BTreeMap<u32, f64> = BTreeMap::new();
    for (bi, &l) in layer.iter().enumerate() {
        col_w
            .entry(l)
            .and_modify(|w| *w = (*w).max(widths[bi]))
            .or_insert(widths[bi]);
    }
    let total_w: f64 = col_w.values().sum::<f64>()
        + col_w.len().saturating_sub(1) as f64 * COL_GAP;

    // ---- place nodes ------------------------------------------------------------
    let mut cx: Vec<Option<f64>> = vec![None; n];
    let mut col_cursor: BTreeMap<u32, f64> = BTreeMap::new();
    for &bi in &order_pos {
        let l = layer[bi];
        let col_left = col_w.range(..l).map(|(_, w)| *w).sum::<f64>()
            + col_w.range(..l).count().saturating_sub(1) as f64 * COL_GAP;
        let slot = col_cursor.entry(l).or_insert(col_left);
        cx[bi] = Some(*slot + widths[bi] / 2.0);
        *slot += widths[bi] + COL_GAP.min(18.0); // intra-column stacking offset
    }

    let mut nodes = Vec::with_capacity(n);
    for bi in 0..n {
        let w = widths[bi];
        let h = heights[bi];
        let centre = cx[bi].unwrap_or(MARGIN + w / 2.0);
        let x = (MARGIN + centre - w / 2.0).max(0.0);
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

    // ---- route edges ---------------------------------------------------------------
    let edges: Vec<CfgEdge> = raw_edges
        .iter()
        .map(|e| route_edge(e, &nodes))
        .collect();

    CfgFunction { name: name.to_string(), nodes, edges }
}

/// Fallthrough unless the edge closes a natural loop (`to` dominates `from`).
fn edge_kind(from: u32, to: u32, doms: &helix_ir::Doms) -> EdgeKind {
    let (f, t) = (helix_ir::BlockId(from), helix_ir::BlockId(to));
    if from >= to || doms.dominates(t, f) {
        EdgeKind::Backedge
    } else {
        EdgeKind::Fallthrough
    }
}

/// Longest-path layering from entry over forward edges only, plus DFS
/// discovery order. Iterative DFS — deep chains must not blow the stack.
fn layer_and_order(ir: &helix_ir::FuncIr, n: usize) -> (Vec<u32>, Vec<usize>) {
    let mut order: Vec<usize> = Vec::new();
    let mut seen = vec![false; n];
    let mut stack: Vec<(usize, usize)> = if n == 0 { Vec::new() } else { vec![(ir.entry.0 as usize, 0)] };
    if n > 0 {
        seen[ir.entry.0 as usize] = true;
    }
    while let Some(&(b, ref mut _c)) = stack.last().copied().as_mut().map(|_| stack.last().copied().expect("checked")).as_mut() {
        let (_b, ci) = stack.last_mut().expect("nonempty");
        let succs = ir.succs(helix_ir::BlockId(_b as u32)).to_vec();
        if ci < succs.len() {
            *stack.last_mut().expect("nonempty") = (_b, ci + 1);
            let s = succs[ci].0 as usize;
            if !seen[s] {
                seen[s] = true;
                stack.push((s, 0));
            }
        } else {
            stack.pop();
            order.push(_b);
        }
    }
    // Any block unreachable from entry still deserves a rectangle.
    for bi in 0..n {
        if !seen[bi] {
            seen[bi] = true;
            order.push(bi);
        }
    }
    order.reverse(); // discovery order (preorder)

    // Longest path over forward edges, relaxing in discovery order.
    let pos: Vec<usize> = {
        let mut p = vec![usize::MAX; n];
        for (i, &b) in order.iter().enumerate() {
            p[b] = i;
        }
        p
    };
    let mut layer = vec![0u32; n];
    for &b in &order {
        let lb = layer[b];
        for s in ir.succs(helix_ir::BlockId(b as u32)) {
            let si = s.0 as usize;
            // Ignore backedges (target already ordered before us in a loop).
            if pos[si] != usize::MAX && pos[si] < pos[b] && si != b {
                continue;
            }
            if layer[si] < lb + 1 {
                layer[si] = lb + 1;
            }
        }
    }
    (layer, order)
}

/// Rendered instruction lines of one block (phis first, terminator last) —
/// the same content `print_ir` writes, minus headers/preds noise.
fn block_lines(ir: &helix_ir::FuncIr, b: helix_ir::BlockId) -> Vec<String> {
    use std::fmt::Write as _;
    let block = ir.block(b);
    let mut lines = Vec::new();
    for p in &block.phis {
        let args = p
            .args
            .iter()
            .map(|(from, v)| format!("[bb{}: v{}]", from.0, v.0))
            .collect::<Vec<_>>()
            .join(" ");
        if p.args.is_empty() {
            lines.push(format!("v{} = param", p.dst.0));
        } else {
            let mut l = format!("v{} = φ({args})", p.dst.0);
            let _ = write!(l, "");
            lines.push(l);
        }
    }
    for inst in &block.insts {
        lines.push(print_inst_short(inst));
    }
    match ir.term(b) {
        helix_ir::Term::Jump(t, _) => lines.push(format!("jump bb{}", t.0)),
        helix_ir::Term::Branch { t, f, .. } => {
            lines.push(format!("branch ? bb{} : bb{}", t.0, f.0))
        }
        helix_ir::Term::Return(None) => lines.push("return".to_string()),
        helix_ir::Term::Return(Some(v)) => lines.push(format!("return v{}", v.0)),
    }
    lines
}

/// Compact instruction spelling for box lines (no local-name resolution —
/// ids stay stable and short enough for the monospace estimate).
fn print_inst_short(i: &helix_ir::Inst) -> String {
    match i {
        helix_ir::Inst::Const { dst, .. } => format!("v{} = const …", dst.0),
        helix_ir::Inst::Bin { op, dst, .. } => {
            format!("v{} = bin {} …", dst.0, op.symbol())
        }
        helix_ir::Inst::Unary { op, dst, .. } => format!("v{} = {} …", dst.0, op.symbol()),
        helix_ir::Inst::Cast { dst, .. } => format!("v{} = cast …", dst.0),
        helix_ir::Inst::Load(l) => format!("v{} = load …", l.dst.0),
        helix_ir::Inst::Store { arr, .. } => format!("store [l{}] …", arr.0),
        helix_ir::Inst::Call(c) => match c.dst {
            Some(d) => format!("v{} = call {}", d.0, c.callee),
            None => format!("call {}", c.callee),
        },
    }
}

/// Routes one edge around/through the placed boxes.
fn route_edge(e: &RawEdge, nodes: &[CfgNode]) -> CfgEdge {
    let (Some(a), Some(b)) = (nodes.get(e.from as usize), nodes.get(e.to as usize)) else {
        return CfgEdge {
            from: format!("bb{}", e.from),
            to: format!("bb{}", e.to),
            kind: e.kind,
            points: Vec::new(),
            label: String::new(),
        };
    };

    let a_bottom = (a.x + a.w / 2.0, a.y + a.h);
    let b_top = (b.x + b.w / 2.0, b.y);

    if matches!(e.kind, EdgeKind::Backedge) {
        // Curve out to the side: start at the source's bottom-right corner
        // region, bow past the right of both boxes, re-enter the header's top.
        let bulge_x = a.x + a.w + 34.0;
        let mid_y = (a_bottom.1 + b_top.1) / 2.0;
        let start = (a.x + a.w, a.y + a.h * 0.75);
        let end = (b.x + b.w, b.y + b.h * 0.25);
        let ctrl = (bulge_x.max(start.0 + 30.0), mid_y);
        return CfgEdge {
            from: a.id.clone(),
            to: b.id.clone(),
            kind: e.kind,
            points: vec![[start.0, start.1], [ctrl.0, ctrl.1], [end.0, end.1]],
            label: e.label.clone(),
        };
    }

    if matches!(e.kind, EdgeKind::Branch) {
        // Branch arms leave through side ports so T/F never overlap.
        let same_row = (a.y - b.y).abs() < 1.0;
        let start_x = if e.label == "F" { a.x } else { a.x + a.w };
        let start_y = a.y + a.h * 0.5;
        let end_x = b.x + b.w / 2.0;
        let end_y = b.y;
        let pts = if same_row {
            // Sideways hop on one row: enter through the facing wall.
            let ex = if start_x < b.x { b.x } else { b.x + b.w };
            vec![
                [start_x, start_y],
                [(start_x + ex) / 2.0, start_y],
                [ex, b.y + b.h * 0.5],
            ]
        } else {
            // Elbow: out of the side, drop below the source row, into the top.
            vec![
                [start_x, start_y],
                [start_x, (start_y + end_y) / 2.0],
                [end_x, end_y],
            ]
        };
        return CfgEdge {
            from: a.id.clone(),
            to: b.id.clone(),
            kind: e.kind,
            points: pts,
            label: e.label.clone(),
        };
    }

    // Plain fallthrough: straight when aligned, else a soft elbow.
    let dx = b_top.0 - a_bottom.0;
    let dy = b_top.1 - a_bottom.1;
    let points = if dx.abs() < 4.0 {
        vec![[a_bottom.0, a_bottom.1], [b_top.0, b_top.1]]
    } else {
        let bend_y = a_bottom.1 + dy * 0.55;
        vec![[a_bottom.0, a_bottom.1], [a_bottom.0, bend_y], [b_top.0, bend_y], [b_top.0, b_top.1]]
    };
    CfgEdge {
        from: a.id.clone(),
        to: b.id.clone(),
        kind: e.kind,
        points,
        label: String::new(),
    }
}
