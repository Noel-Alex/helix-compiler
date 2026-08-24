//! Stable `bb0`-style textual rendering of [`FuncIr`].
//!
//! The format is the project brief's example shape:
//!
//! ```text
//! fn main() -> () {
//! bb0:                          ; preds:
//!   i0 = const 5
//!   x0 = ...
//!   jump bb1()
//! bb1:                          ; preds: bb0
//!   x1 = φ [bb0: x0] [bb2: x2]
//!   call print, args=[]
//!   return
//! }
//! ```
//!
//! Stability contract for golden tests and the Observatory diff view:
//! * blocks in ascending id order;
//! * phis first (only shown when `ssa == true`), then instructions in order,
//!   terminator last with a blank-line separation never injected mid-block;
//! * value names `vN` (or the source variable name + version suffix once SSA
//!   names exist — see [`ssa_name`]);
//! * terminator spelled `jump bbK(a0, …)` / `branch cond ? bbT : bbF` /
//!   `return` / `return v`.

use std::fmt::Write as _;


use crate::ir::{BlockId, Constant, FuncIr, Inst, Term};

/// Render a whole function.
#[must_use]
pub fn print_ir(ir: &FuncIr, ssa: bool) -> String {
    let mut out = String::new();
    let ret = ir.types.ret.name();
    let _ = writeln!(out, "fn {} -> {} {{", ir.name, ret);
    for (bi, b) in ir.blocks.iter().enumerate() {
        let _bid = BlockId(bi as u32);
        let _ = writeln!(out, "bb{}:", bi);
        if !b.preds.is_empty() {
            let preds = b
                .preds
                .iter()
                .map(|p| format!("bb{}", p.0))
                .collect::<Vec<_>>()
                .join(" ");
            let _ = writeln!(out, "  ; preds: {preds}");
        } else {
            let _ = writeln!(out, "  ; preds:");
        }

        if ssa {
            for p in &b.phis {
                if p.args.is_empty() {
                    // Entry parameter phi: render like a block param.
                    let _ = writeln!(
                        out,
                        "  {} = param {}",
                        name_of(ir, p.dst),
                        local_name(ir, p.var)
                    );
                } else {
                    let args = p
                        .args
                        .iter()
                        .map(|(from, v)| format!("[bb{}: {}]", from.0, name_of(ir, *v)))
                        .collect::<Vec<_>>()
                        .join(" ");
                    let _ = writeln!(
                        out,
                        "  {} = φ(v{}) {args}",
                        name_of(ir, p.dst),
                        p.var.0
                    );
                }
            }
        }

        for inst in &b.insts {
            let _ = writeln!(out, "  {}", inst_line(ir, inst));
        }

        match &b.term {
            Term::Jump(t, args) => {
                let a = args
                    .iter()
                    .map(|v| name_of(ir, *v))
                    .collect::<Vec<_>>()
                    .join(", ");
                let _ = writeln!(out, "  jump bb{}({a})", t.0);
            }
            Term::Branch { cond, t, f } => {
                let _ = writeln!(
                    out,
                    "  branch {} ? bb{} : bb{}",
                    name_of(ir, *cond),
                    t.0,
                    f.0
                );
            }
            Term::Return(Some(v)) => {
                let _ = writeln!(out, "  return {}", name_of(ir, *v));
            }
            Term::Return(None) => {
                let _ = writeln!(out, "  return");
            }
        }
    }
    let _ = writeln!(out, "}}");
    out.trim_end().to_string()
}

/// One instruction line (`dst = op operands`).
fn inst_line(ir: &FuncIr, inst: &Inst) -> String {
    match inst {
        Inst::Const { dst, c } => format!("{} = const {}", name_of(ir, *dst), const_str(c)),
        Inst::Bin { op, dst, a, b } => format!(
            "{} = bin {} {}, {}",
            name_of(ir, *dst),
            op.symbol(),
            name_of(ir, *a),
            name_of(ir, *b)
        ),
        Inst::Unary { op, dst, a } => format!(
            "{} = unary {} {}",
            name_of(ir, *dst),
            op.symbol(),
            name_of(ir, *a)
        ),
        Inst::Cast { dst, val, to } => format!(
            "{} = cast {} as {}",
            name_of(ir, *dst),
            name_of(ir, *val),
            to.name()
        ),
        Inst::Load(l) => format!(
            "{} = load {}[{}]",
            name_of(ir, l.dst),
            local_name(ir, l.arr),
            name_of(ir, l.idx)
        ),
        Inst::Store { arr, idx, val } => format!(
            "store {}[{}] = {}",
            local_name(ir, *arr),
            name_of(ir, *idx),
            name_of(ir, *val)
        ),
        Inst::Call(c) => {
            let mut parts: Vec<String> = c.args.iter().map(|v| name_of(ir, *v)).collect();
            for (k, arr) in c.arr_refs.iter().enumerate() {
                parts.push(if k + 1 == c.arr_refs.len() && c.dst.is_none() && c.args.is_empty() {
                    // Array-returning callee: destination rendered as output.
                    format!("out={}", local_name(ir, *arr))
                } else {
                    format!("&{}", local_name(ir, *arr))
                });
            }
            let d = match c.dst {
                Some(d) => format!("{} = ", name_of(ir, d)),
                None => String::new(),
            };
            format!("{d}call {}, args=[{}]", c.callee, parts.join(", "))
        }
    }
}

fn const_str(c: &Constant) -> String {
    match c {
        Constant::I64(v) => v.to_string(),
        Constant::I32(v) => format!("{v}:i32"),
        Constant::F32(v) => format!("{v:?}:f32"),
        Constant::F64(v) => format!("{v:?}"),
        Constant::Bool(b) => b.to_string(),
    }
}

/// Display name of a value. SSA-renamed ids decode to `<local><version>`;
/// cell ids print their source variable name; compiler temporaries print
/// `$tag`-style names from the side table.
#[must_use]
pub fn ssa_name(local_idx: u32, version: u32) -> String {
    format!("t{local_idx}_{version}")
}

fn name_of(ir: &FuncIr, v: crate::ir::ValueId) -> String {
    if ir.is_slot_value(v) {
        match ir.types.local_name(crate::ir::LocalId(v.0)) {
            Some(n) if !n.is_empty() => n.to_string(),
            _ => format!("v{}", v.0),
        }
    } else {
        format!("v{}", v.0)
    }
}

fn local_name(ir: &FuncIr, l: crate::ir::LocalId) -> String {
    ir.types
        .local_name(l)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| format!("l{}", l.0))
}
