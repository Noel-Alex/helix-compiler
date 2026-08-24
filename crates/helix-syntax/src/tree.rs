//! Pretty-printing an AST as an indented ASCII tree.
//!
//! [`Program::print_tree`] renders one line per node, indented by two spaces
//! per nesting level (`|-` marks non-final items at the program level). It is
//! consumed by the CLI `dump ast` stage and by the Observatory's AST view,
//! shown next to the JSON form. The output is deterministic — no timestamps,
//! no addresses — so dumps can be diffed in tests and stored in artifacts.
//!
//! Layout rules:
//!
//! ```text
//! Program
//!   Fn main() -> () @0..24
//!     Block @10..24
//!       Let x @12..22
//!         IntLit 1 @20..21
//! ```
//!
//! Composite nodes introduce labelled rows (`cond`, `then`, `else`, `start`,
//! `end`, `body`, `lhs`, `rhs`, …) whose own subtree follows one level deeper.

use crate::ast::{
    Block, ConstDef, ElsePart, Expr, FnDef, Item, LValue, Literal, Param, Program, Stmt,
};

impl Program {
    /// Renders the whole program as an indented ASCII tree.
    #[must_use]
    pub fn print_tree(&self) -> String {
        let mut out = String::new();
        out.push_str("Program\n");
        if self.items.is_empty() {
            out.push_str("  `(empty)`\n");
            return out;
        }
        let last = self.items.len() - 1;
        for (i, item) in self.items.iter().enumerate() {
            // Header line carries the branch marker; descendant lines use a
            // continuation prefix aligned under it.
            let (header, kids_prefix) = if i == last {
                ("  `- ".to_string(), "      ".to_string())
            } else {
                ("  |- ".to_string(), "  |   ".to_string())
            };
            match item {
                Item::Fn(f) => f.write_tree(&mut out, &header, &kids_prefix),
                Item::Const(c) => c.write_tree(&mut out, &header),
            }
        }
        out
    }
}

/// One node line: `prefix label detail\n`.
fn line(out: &mut String, prefix: &str, label: &str, detail: &str) {
    out.push_str(prefix);
    out.push_str(label);
    if !detail.is_empty() {
        out.push(' ');
        out.push_str(detail);
    }
    out.push('\n');
}

/// Prefix for a labelled child one level under `prefix`.
fn sub(prefix: &str, label: &str) -> String {
    format!("{prefix}{label}")
}

/// Two extra spaces of indent (a plain nesting step).
fn deeper(prefix: &str) -> String {
    format!("{prefix}  ")
}

impl FnDef {
    /// `header` prefixes this definition's own line; `kids` prefixes the
    /// body block's lines (continuation indent under the item marker).
    fn write_tree(&self, out: &mut String, header: &str, kids: &str) {
        let ret = match &self.ret {
            Some(t) => format!("-> {}", t.render()),
            None => "()".to_string(),
        };
        line(
            out,
            header,
            "Fn",
            &format!(
                "{}({}) {} @{}",
                self.name.name,
                render_params(&self.params),
                ret,
                self.span
            ),
        );
        self.body.write_tree(out, kids);
    }
}

impl ConstDef {
    fn write_tree(&self, out: &mut String, prefix: &str) {
        line(
            out,
            prefix,
            "Const",
            &format!(
                "{}: {} = {} @{}",
                self.name.name,
                self.ty.render(),
                render_literal(&self.value),
                self.span
            ),
        );
    }
}

impl Block {
    fn write_tree(&self, out: &mut String, prefix: &str) {
        line(out, prefix, "Block", &format!("@{}", self.span));
        for stmt in &self.stmts {
            stmt.write_tree(out, &deeper(prefix));
        }
    }
}

impl Stmt {
    fn write_tree(&self, out: &mut String, prefix: &str) {
        match self {
            Stmt::Let {
                name,
                ty,
                init,
                span,
            } => {
                let ty_s = ty
                    .as_ref()
                    .map_or(String::new(), |t| format!(": {}", t.render()));
                line(
                    out,
                    prefix,
                    "Let",
                    &format!("{}{ty_s} @{}", name.name, span),
                );
                init.write_tree(out, &deeper(prefix));
            }
            Stmt::Assign {
                target,
                value,
                span,
            } => {
                line(
                    out,
                    prefix,
                    "Assign",
                    &format!("{} @{}", render_lvalue(target), span),
                );
                line(out, &deeper(prefix), "target", "");
                target.write_tree(out, &sub(&deeper(prefix), ""));
                line(out, &deeper(prefix), "value", "");
                value.write_tree(out, &sub(&deeper(prefix), ""));
            }
            Stmt::If {
                cond,
                then_blk,
                else_part,
                span,
            } => {
                line(out, prefix, "If", &format!("@{}", span));
                line(out, &deeper(prefix), "cond", "");
                cond.write_tree(out, &sub(&deeper(prefix), ""));
                line(out, &deeper(prefix), "then", "");
                then_blk.write_tree(out, &sub(&deeper(prefix), ""));
                if let Some(ep) = else_part {
                    // An `else if` chain shows up as a nested `If` spine.
                    match ep.as_ref() {
                        ElsePart::If(stmt) => {
                            line(out, &deeper(prefix), "else", "");
                            stmt.write_tree(out, &sub(&deeper(prefix), ""));
                        }
                        ElsePart::Block(blk) => {
                            line(out, &deeper(prefix), "else", "");
                            blk.write_tree(out, &sub(&deeper(prefix), ""));
                        }
                    }
                }
            }
            Stmt::For {
                iv,
                start,
                end,
                body,
                span,
            } => {
                line(
                    out,
                    prefix,
                    "For",
                    &format!("{} in [start, end) @{}", iv.name, span),
                );
                line(out, &deeper(prefix), "iv", &iv.name);
                line(out, &deeper(prefix), "from", "");
                start.write_tree(out, &sub(&deeper(prefix), ""));
                line(out, &deeper(prefix), "to", "");
                end.write_tree(out, &sub(&deeper(prefix), ""));
                line(out, &deeper(prefix), "body", "");
                body.write_tree(out, &sub(&deeper(prefix), ""));
            }
            Stmt::Return { value, span } => {
                line(out, prefix, "Return", &format!("@{}", span));
                if let Some(v) = value {
                    v.write_tree(out, &deeper(prefix));
                }
            }
            Stmt::Expr(e) => {
                e.write_tree(out, &sub(prefix, "expr"));
            }
            Stmt::Empty => line(out, prefix, "Empty", ""),
            Stmt::Block(b) => b.write_tree(out, prefix),
        }
    }
}

impl LValue {
    fn write_tree(&self, out: &mut String, prefix: &str) {
        match &self.index {
            Some(_) => line(
                out,
                prefix,
                "Elem",
                &format!("{} @{}", self.base.name, self.span),
            ),
            None => line(
                out,
                prefix,
                "Var",
                &format!("{} @{}", self.base.name, self.span),
            ),
        }
    }
}

impl Expr {
    fn write_tree(&self, out: &mut String, prefix: &str) {
        match self {
            Expr::IntLit(v, span) => line(out, prefix, "IntLit", &format!("{v} @{span}")),
            Expr::FloatLit(v, span) => {
                line(out, prefix, "FloatLit", &format!("{v:?} @{span}"));
            }
            Expr::Bool(b, span) => line(out, prefix, "BoolLit", &format!("{b} @{span}")),
            Expr::Var(id) => line(out, prefix, "Var", &format!("{} @{}", id.name, id.span)),
            Expr::Unary(op, inner, span) => {
                line(out, prefix, "Unary", &format!("{} @{}", op.symbol(), span));
                inner.write_tree(out, &deeper(prefix));
            }
            Expr::Bin(op, lhs, rhs, span) => {
                line(out, prefix, "Bin", &format!("`{}` @{}", op.symbol(), span));
                lhs.write_tree(out, &sub(prefix, "lhs:"));
                rhs.write_tree(out, &sub(prefix, "rhs:"));
            }
            Expr::Index(base, index, span) => {
                line(out, prefix, "Index", &format!("{} @{}", base.name, span));
                index.write_tree(out, &sub(prefix, "idx:"));
            }
            Expr::Call { callee, args, span } => {
                line(out, prefix, "Call", &format!("{} @{}", callee.name, span));
                for (i, arg) in args.iter().enumerate() {
                    arg.write_tree(out, &format!("{prefix}  arg{i}: "));
                }
            }
            Expr::Cast(inner, ty, span) => {
                line(
                    out,
                    prefix,
                    "Cast",
                    &format!("as {} @{}", ty.render(), span),
                );
                inner.write_tree(out, &deeper(prefix));
            }
        }
    }
}

fn render_lvalue(lv: &LValue) -> String {
    if lv.index.is_some() {
        format!("{}[...]", lv.base.name)
    } else {
        lv.base.name.clone()
    }
}

fn render_params(params: &[Param]) -> String {
    params
        .iter()
        .map(|p| format!("{}: {}", p.name.name, p.ty.render()))
        .collect::<Vec<_>>()
        .join(", ")
}

fn render_literal(lit: &Literal) -> String {
    match lit {
        Literal::Int(v) => v.to_string(),
        Literal::Float(v) => format!("{v:?}"),
        Literal::Bool(b) => b.to_string(),
    }
}
