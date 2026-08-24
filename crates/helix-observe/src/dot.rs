//! Graphviz `.dot` emission — the escape hatch for publication-quality CFG
//! figures without going through the browser.
//!
//! The output mirrors [`crate::layout::cfg_layout`] exactly (same roles, same
//! edge classes, same labels) but lets Graphviz do its own layout, so it is
//! also a handy cross-check: if the server-side tidy algorithm disagrees with
//! `dot`'s idea of the graph, one of them is wrong.
//!
//! ```text
//! digraph "main" {
//!   graph [rankdir=TB];
//!   node  [shape=record, fontname="monospace"];
//!   bb0 [label="{bb0|v0 = const 5\ljump bb1()\l}", shape=box];
//!   bb1 -> bb2 [label=T];
//! }
//! ```

use crate::artifact::{BlockRole, CfgFunction, EdgeKind};

/// Renders one laid-out function as a Graphviz digraph.
///
/// The text is plain ASCII, escaped per the DOT grammar (backslashes and
/// double quotes), and ends in a newline.
#[must_use]
pub fn cfg_to_dot(fn_layout: &CfgFunction) -> String {
    let mut out = String::new();
    out.push_str(&format!("digraph \"{}\" {{\n", escape(&fn_layout.name)));
    out.push_str("  graph [rankdir=TB, bgcolor=transparent];\n");
    out.push_str("  node [shape=box, style=\"rounded,filled\", fillcolor=\"#161b22\", ");
    out.push_str(
        "fontname=\"monospace\", fontsize=10, color=\"#30363d\", fontcolor=\"#e6edf3\"];\n",
    );
    out.push_str(
        "  edge [color=\"#8b949e\", fontcolor=\"#79c0ff\", fontname=\"monospace\", fontsize=9];\n",
    );

    for node in &fn_layout.nodes {
        let mut label = String::new();
        label.push_str(node.id.as_str());
        match node.role {
            BlockRole::Entry => label.push_str("\\n(entry)"),
            BlockRole::Exit => label.push_str("\\n(exit)"),
            BlockRole::LoopHeader => {
                let lp = node
                    .loop_id
                    .map(|id| format!(" (loop {id})"))
                    .unwrap_or_default();
                label.push_str(&format!("\\n(loop header{lp})"));
            }
            BlockRole::Join => label.push_str("\\n(join)"),
            BlockRole::Straight => {}
        }
        for line in &node.lines {
            // `\l` = left-justified line break in a record/label.
            label.push_str("\\n");
            label.push_str(&escape(line));
        }
        let color = role_color(node.role);
        out.push_str(&format!(
            "  {} [label=\"{}\", color=\"{}\"];\n",
            escape(&node.id),
            label,
            color
        ));
    }

    for edge in &fn_layout.edges {
        let style = match edge.kind {
            EdgeKind::Backedge => ", style=dashed, color=\"#f85149\", constraint=false",
            EdgeKind::Branch => ", color=\"#61afef\"",
            EdgeKind::Fallthrough => "",
        };
        let lbl = if edge.label.is_empty() {
            String::new()
        } else {
            format!("label={}, ", escape(&edge.label))
        };
        out.push_str(&format!(
            "  {} -> {} [{}{}];\n",
            escape(&edge.from),
            escape(&edge.to),
            lbl,
            style.trim_start_matches(", ")
        ));
    }

    out.push_str("}\n");
    out
}

/// Node stroke colour per block role (matches the UI palette).
fn role_color(role: BlockRole) -> &'static str {
    match role {
        BlockRole::Entry => "#00c896",
        BlockRole::Exit => "#e6edf3",
        BlockRole::LoopHeader => "#e69f00",
        BlockRole::Join => "#56b4e9",
        BlockRole::Straight => "#30363d",
    }
}

/// Escapes a string for inclusion inside a DOT double-quoted string.
fn escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifact::{CfgEdge, CfgNode};

    fn sample() -> CfgFunction {
        CfgFunction {
            name: "main".into(),
            nodes: vec![
                CfgNode {
                    id: "bb0".into(),
                    x: 40.0,
                    y: 20.0,
                    w: 180.0,
                    h: 64.0,
                    role: BlockRole::Entry,
                    lines: vec!["v0 = const 5".into(), "jump bb1".into()],
                    loop_id: None,
                },
                CfgNode {
                    id: "bb1".into(),
                    x: 40.0,
                    y: 120.0,
                    w: 180.0,
                    h: 82.0,
                    role: BlockRole::LoopHeader,
                    lines: vec!["branch ? bb2 : bb3".into()],
                    loop_id: Some(0),
                },
            ],
            edges: vec![
                CfgEdge {
                    from: "bb0".into(),
                    to: "bb1".into(),
                    kind: EdgeKind::Fallthrough,
                    points: vec![[130.0, 84.0], [130.0, 120.0]],
                    label: String::new(),
                },
                CfgEdge {
                    from: "bb1".into(),
                    to: "bb1".into(),
                    kind: EdgeKind::Backedge,
                    points: vec![[220.0, 180.0], [260.0, 160.0], [220.0, 140.0]],
                    label: String::new(),
                },
            ],
        }
    }

    #[test]
    fn emits_parseable_dot_skeleton() {
        let dot = cfg_to_dot(&sample());
        assert!(dot.starts_with("digraph \"main\" {"));
        assert!(dot.trim_end().ends_with('}'));
        assert!(dot.contains("  bb0 -> bb1 "));
        assert!(dot.contains("style=dashed"));
        assert_eq!(
            dot.lines().count(),
            sample().nodes.len() + sample().edges.len() + 4 + 1
        );
    }

    #[test]
    fn escapes_quotes_and_backslashes() {
        let mut f = sample();
        f.nodes[0].lines.push(r#"quote " and \ here"#.into());
        let dot = cfg_to_dot(&f);
        assert!(dot.contains(r#"\\ here"#));
        assert!(dot.contains("\\\""));
    }
}
