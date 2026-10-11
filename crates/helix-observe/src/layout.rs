//! Browser-side tidy layout of the AST display hierarchy: the browser paints
//! coordinates, it never computes them (`artifact-schema.md` rule). One tidy
//! algorithm lives here.
//!
//! * [`ast_tree`] — Reingold–Tilford-flavoured layered tree over the display
//!   hierarchy produced from a serde `Program`: depth ⇒ row (`y`), in-order
//!   leaf slots ⇒ column (`x`), parents centred over their children. Small,
//!   dependency-free, and deterministic for identical trees.
//! * [`program_to_tree`] — the adapter converting externally-tagged serde
//!   enum JSON (`{"Bin": ["+", …]}`) into that hierarchy, labelling nodes by
//!   variant plus a payload summary so the picture reads like the grammar.
//!
//! The per-function CFG layout is a different graph and a different algorithm
//! — it lives in `helix-artifact` as [`helix_artifact::cfg_layout`] because
//! the artifact carries its coordinates.
//!
//! Every coordinate is finite by construction (no division, no trig); the
//! artifact tests assert this anyway so NaN can never reach the SVG.

// ---------------------------------------------------------------------------
// Tree geometry constants
// ---------------------------------------------------------------------------
// The CFG box geometry is a separate set in `helix-artifact`; these four size
// the AST tree only, so a change to one algorithm cannot shift the other.

/// Horizontal gap between neighbouring leaf slots.
const COL_GAP: f64 = 46.0;
/// Minimum horizontal room a leaf claims (its own label must fit).
const LEAF_MIN_W: f64 = 90.0;
/// Vertical gap between tree depths.
const ROW_GAP: f64 = 78.0;
/// Canvas margin on every side.
const MARGIN: f64 = 28.0;

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
