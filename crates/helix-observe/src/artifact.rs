//! The `CompileArtifact` data contract — the JSON the Observatory renders.
//!
//! Every stage dump of one compile flows to the browser as a single
//! self-contained document of this shape (`docs/notes/artifact-schema.md` is
//! normative; field names there are frozen). The design rule is **graceful
//! degradation**: compilation may stop at any stage, so every post-parse
//! stage is an [`Option`] and the UI greys out whatever is absent.
//!
//! Layout coordinates inside [`CfgFunction`] are *final*: the browser paints
//! rectangles and polylines, it never computes geometry (see [`crate::layout`]
//! for where they come from).

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Tokens
// ---------------------------------------------------------------------------

/// One lexed token as the SOURCE/TOKENS views consume it.
///
/// Deliberately flattened: the artifact carries `kind` as the serde variant
/// name string ("Kw", "Ident", "EqEq", …) rather than the tagged enum, and
/// `text` as the exact source spelling, because the browser only wants to
/// classify and highlight — it never re-lexes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenView {
    /// Serde variant name of [`helix_syntax::TokKind`], e.g. `"Kw"`, `"Int"`.
    pub kind: String,
    /// Exact source text of the token ("" for the EOF marker).
    pub text: String,
    /// Byte offset of the first character.
    pub start: u32,
    /// Byte offset one past the last character.
    pub end: u32,
}

impl TokenView {
    /// Projects a real token plus its source spelling.
    #[must_use]
    pub fn of(tok: &helix_syntax::Token, src: &str) -> Self {
        let s = (tok.span.start as usize).min(src.len());
        let e = (tok.span.end as usize).min(src.len()).max(s);
        let kind = match &tok.kind {
            helix_syntax::TokKind::Ident(name) => {
                // Reserved words surface as their own pill in the token table;
                // keeping them "Ident" would render them like ordinary names.
                if helix_syntax::token::is_reserved(name) {
                    format!("Reserved({name})")
                } else {
                    "Ident".to_string()
                }
            }
            k => short_kind_name(k),
        };
        let mut text = &src[s..e];
        if matches!(tok.kind, helix_syntax::TokKind::Eof) {
            text = "";
        }
        Self {
            kind,
            text: text.to_string(),
            start: tok.span.start,
            end: tok.span.end,
        }
    }
}

/// The un-parenthesized variant name of a [`helix_syntax::TokKind`] as serde
/// spells it for externally-tagged unit variants.
fn short_kind_name(kind: &helix_syntax::TokKind) -> String {
    use helix_syntax::TokKind as K;
    let name = match kind {
        K::Int(_) => "Int",
        K::Float(_) => "Float",
        K::Kw(kw) => return format!("Kw({})", kw.as_str()),
        K::LParen => "LParen",
        K::RParen => "RParen",
        K::LBrace => "LBrace",
        K::RBrace => "RBrace",
        K::LBracket => "LBracket",
        K::RBracket => "RBracket",
        K::Comma => "Comma",
        K::Semi => "Semi",
        K::Colon => "Colon",
        K::PathSep => "PathSep",
        K::DotDot => "DotDot",
        K::Arrow => "Arrow",
        K::Plus => "Plus",
        K::Minus => "Minus",
        K::Star => "Star",
        K::Slash => "Slash",
        K::Rem => "Rem",
        K::Lt => "Lt",
        K::Gt => "Gt",
        K::Le => "Le",
        K::Ge => "Ge",
        K::EqEq => "Eq",
        K::NotEq => "Ne",
        K::AndAnd => "AndAnd",
        K::OrOr => "OrOr",
        K::Not => "Not",
        K::Assign => "Assign",
        K::Eof => "Eof",
        K::Ident(_) => "Ident",
    };
    name.to_string()
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

/// A semantic (or syntactic) error with its source region.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagView {
    /// Byte span to underline in the SOURCE view.
    pub span: SpanView,
    /// Human message, shown verbatim.
    pub msg: String,
}

/// `{start, end}` pair — mirrors [`helix_syntax::Span`] without dragging the
/// whole syntax crate into every consumer's dependency graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpanView {
    /// First byte (inclusive).
    pub start: u32,
    /// One past the last byte (exclusive).
    pub end: u32,
}

impl From<helix_syntax::Span> for SpanView {
    fn from(s: helix_syntax::Span) -> Self {
        Self {
            start: s.start,
            end: s.end,
        }
    }
}

// ---------------------------------------------------------------------------
// IR stages
// ---------------------------------------------------------------------------

/// One function's printed IR plus its display name.
///
/// The schema's `[FuncIrText]` entries are plain monospace strings; carrying
/// the name alongside lets the UI build per-function tabs without parsing the
/// first line back out. `app.js` accepts both shapes (string or object), so
/// the object form is strictly additive.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrText {
    /// Function name (`main`, `fib`, …).
    pub name: String,
    /// Full `print_ir` text, `\n`-separated lines.
    pub text: String,
}

/// Pre-SSA / post-SSA dumps: one entry per function.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IrStage {
    /// One per function, in pipeline order.
    pub functions: Vec<IrText>,
}

// ---------------------------------------------------------------------------
// Optimization passes
// ---------------------------------------------------------------------------

/// Record of one optimization pass run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PassReport {
    /// Registry name, e.g. `"const_fold"`.
    pub name: String,
    /// Did the pass rewrite anything?
    pub changed: bool,
    /// Full IR text after the pass.
    pub after: String,
    /// Cheap size metric before/after.
    pub diff_stats: DiffStats,
}

/// Instruction-count delta around one pass.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DiffStats {
    /// Instructions + phis before the pass.
    pub insts_before: usize,
    /// Instructions + phis after the pass.
    pub insts_after: usize,
}

// ---------------------------------------------------------------------------
// CFG layout
// ---------------------------------------------------------------------------

/// Role of a basic block — drives the CFG colour coding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlockRole {
    /// Function entry.
    Entry,
    /// Terminates in `return`.
    Exit,
    /// Header of at least one natural loop.
    LoopHeader,
    /// Two or more predecessors.
    Join,
    /// Everything else.
    Straight,
}

/// One laid-out basic block rectangle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CfgNode {
    /// Block label, `"bb0"` style.
    pub id: String,
    /// Left edge, canvas pixels.
    pub x: f64,
    /// Top edge, canvas pixels.
    pub y: f64,
    /// Width (fits the widest line).
    pub w: f64,
    /// Height (lines × row height + padding).
    pub h: f64,
    /// Colour/label role.
    pub role: BlockRole,
    /// Rendered instruction lines (phis included when present).
    pub lines: Vec<String>,
    /// Innermost loop containing this block, when any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loop_id: Option<usize>,
}

/// Edge classification — picks stroke style and arrowhead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeKind {
    /// Unconditional jump forward/sideways.
    Fallthrough,
    /// Conditional branch arm (true/false labelled).
    Branch,
    /// Edge back to a loop header (drawn curved).
    Backedge,
}

/// One control-flow edge with precomputed polyline/bezier points.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CfgEdge {
    /// Source block id.
    pub from: String,
    /// Target block id.
    pub to: String,
    /// Stroke class.
    pub kind: EdgeKind,
    /**
     * Waypoints in paint order:
     *
     * * 2 points → straight/elbow polyline,
     * * 3 points → quadratic bezier `M p0 Q p1 p2` (backedges).
     */
    pub points: Vec<[f64; 2]>,
    /// Short edge caption (`T` / `F` on branch arms, otherwise empty).
    #[serde(default)]
    pub label: String,
}

/// Laid-out CFG of one function.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CfgFunction {
    /// Function name.
    pub name: String,
    /// All blocks (unreachable ones included, flagged by role).
    pub nodes: Vec<CfgNode>,
    /// All terminator edges.
    pub edges: Vec<CfgEdge>,
}

/// Per-function CFG container (schema key `"cfg"`).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CfgStage {
    /// One per function.
    pub functions: Vec<CfgFunction>,
}

// ---------------------------------------------------------------------------
// Dominator tree
// ---------------------------------------------------------------------------

/// Dominator tree of one function: parent id → child ids, entry first.
///
/// Serialized as a map so the schema's `{"bb0": ["bb1","bb3"]}` shape holds.
pub type DomTreeMap = std::collections::BTreeMap<String, Vec<String>>;

// ---------------------------------------------------------------------------
// Loops
// ---------------------------------------------------------------------------

/// Reduction identity attached to a parallelized loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReductionView {
    /// Operator glyph: `"+"`, `"*"`, `"min"`, `"max"`.
    pub op: String,
    /// Privatized accumulator variable name.
    pub var: String,
}

/// Runtime plan hint for an approved loop.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PlanHint {
    /// Thread count the runtime will be hinted with.
    pub threads: usize,
}

/// Verdict of the dependence battery for one loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum VerdictLabel {
    /// Independent iterations → plain DOALL.
    Safe,
    /// Recognized reduction → private accumulators + combine.
    Reduction,
    /// Carried dependence or side effect → must stay serial.
    Sequential,
}

/// One dependence edge as the LOOP ANALYSIS card renders it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepEdgeView {
    /// Display label, `"RAW a[i] ← a[i-1]"` style.
    pub kind_label: String,
    /// Array involved.
    pub array: String,
    /// Exact distance when provable.
    pub distance: Option<i64>,
    /// Nest level carrying the dependence (1-based).
    pub level: u32,
    /// Direction vector summary (`"<"`, `"*"`, …).
    pub direction: String,
    /// Full sentence for tooltips.
    pub explain: String,
}

/// One analyzed loop — the demo's star object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoopView {
    /// Loop number within its function (0-based, stable).
    pub id: usize,
    /// Nesting depth, 1 = outermost.
    pub depth: u32,
    /// Header block id.
    pub header: String,
    /// Body block ids.
    pub blocks: Vec<String>,
    /// Induction variable name.
    pub iv: Option<String>,
    /// Half-open range as rendered strings.
    pub bounds: Option<BoundPair>,
    /// Pretty access lines (`READ a[i]`).
    pub accesses: Vec<String>,
    /// Flow dependences (read-after-write).
    pub raw: Vec<DepEdgeView>,
    /// Anti dependences (write-after-read).
    pub war: Vec<DepEdgeView>,
    /// Output dependences (write-after-write).
    pub waw: Vec<DepEdgeView>,
    /// Recognized reduction, when the verdict is [`VerdictLabel::Reduction`].
    pub reduction: Option<ReductionView>,
    /// Battery verdict.
    pub verdict: VerdictLabel,
    /// Human explanation shown under the verdict badge.
    pub reason: String,
    /// Parallel execution hint, present exactly when parallelized.
    pub plan: Option<PlanHint>,
}

/// Rendered loop bounds pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundPair {
    /// Inclusive lower bound text.
    pub start: String,
    /// Exclusive upper bound text.
    pub end: String,
}

// ---------------------------------------------------------------------------
// Execution + bench
// ---------------------------------------------------------------------------

/// Result of actually running the program.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecView {
    /// Which engine produced `printed`/`checksum`.
    pub backend_used: String,
    /// Lines `print` emitted.
    pub printed: Vec<String>,
    /// FNV-1a content checksum, `0x…` formatted.
    pub checksum: String,
    /// Reserved for the bench campaign; always `null` on single runs.
    pub timings_ms: Option<Vec<f64>>,
}

/// One measured variant in a bench campaign.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchVariant {
    /// Variant label (`interpreter`, `native-par-8t`, …).
    pub name: String,
    /// Median wall-clock time in milliseconds.
    pub median_ms: f64,
    /// Raw samples backing the median.
    pub samples: Vec<f64>,
}

/// Efficiency point of a scaling sweep.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EfficiencyPoint {
    /// Thread count of this rung.
    pub threads: usize,
    /// Speedup over the sequential variant.
    pub speedup: f64,
    /// Speedup ÷ threads.
    pub efficiency: f64,
}

/// Attached benchmark campaign (schema key `"bench"`); optional.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchView {
    /// Kernel the campaign ran.
    pub kernel: String,
    /// Problem size.
    pub n: u64,
    /// One entry per backend variant.
    pub variants: Vec<BenchVariant>,
    /// Scaling sweep, when more than one thread count was measured.
    pub efficiency: Vec<EfficiencyPoint>,
}

// ---------------------------------------------------------------------------
// Root document
// ---------------------------------------------------------------------------

/// The whole Observatory payload: everything known about one compile.
///
/// Build via [`crate::pipeline::build_artifact`]. Stages are `None` once the
/// pipeline stopped (lex error ⇒ only `source`/`diags_lex`; sema error ⇒
/// through `ast` plus `diags_sem`; success ⇒ everything).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompileArtifact {
    /// Schema version — bump on any breaking field change.
    pub schema: u32,
    /// Example name, or `"<adhoc>"` for POSTed sources.
    pub example: String,
    /// Raw source text all spans point into.
    pub source: String,

    /// Lexer/parser failure (artifact stops here; `tokens` stays empty).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diags_lex: Option<Vec<DiagView>>,

    /// Token stream (empty when lexing failed).
    #[serde(default)]
    pub tokens: Option<Vec<TokenView>>,

    /// Serde-serialized AST (`Program`), present after a successful parse.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ast: Option<serde_json::Value>,

    /// Semantic errors (non-empty ⇒ later stages absent). Empty list means ok.
    #[serde(default)]
    pub diags_sem: Vec<DiagView>,

    /// `print_ir(ssa=false)` per function.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ir_pre_ssa: Option<IrStage>,

    /// `print_ir(ssa=true)` per function.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ir_ssa: Option<IrStage>,

    /// Per-pass snapshots from the optimization driver.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub passes: Option<Vec<PassReport>>,

    /// Laid-out CFGs (coordinates final — see [`crate::layout`]).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cfg: Option<CfgStage>,

    /// Dominator tree per function.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub domtree: Option<std::collections::BTreeMap<String, DomTreeMap>>,

    /// Analyzed loops across all functions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loops: Option<Vec<LoopView>>,

    /// Interpreter result, present when the program type-checked.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exec: Option<ExecView>,

    /// Benchmark campaign attachment (never set by single runs today).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bench: Option<BenchView>,
}

impl CompileArtifact {
    /// An artifact holding just `schema`/`example`/`source` — the seed the
    /// pipeline fills in stage by stage.
    #[must_use]
    pub fn new(example: impl Into<String>, source: impl Into<String>) -> Self {
        Self {
            schema: SCHEMA_VERSION,
            example: example.into(),
            source: source.into(),
            diags_lex: None,
            tokens: None,
            ast: None,
            diags_sem: Vec::new(),
            ir_pre_ssa: None,
            ir_ssa: None,
            passes: None,
            cfg: None,
            domtree: None,
            loops: None,
            exec: None,
            bench: None,
        }
    }

    /// Serialize to pretty JSON (the wire format humans diff too).
    #[must_use]
    pub fn to_json_pretty(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }
}

/// Current artifact schema version.
pub const SCHEMA_VERSION: u32 = 1;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_view_projects_variant_names_and_text() {
        let src = "fn main() { x <= 1 != 2 && true }";
        let toks = helix_syntax::lex(src).expect("lexes");
        let views: Vec<TokenView> = toks.iter().map(|t| TokenView::of(t, src)).collect();
        assert_eq!(views[0].kind, "Kw(fn)");
        assert_eq!(views[0].text, "fn");
        assert_eq!(views[0].start, 0);
        assert_eq!(views[0].end, 2);
        // `<=` maps to the UI-facing name "Le", `!=` to "Ne".
        assert!(views.iter().any(|t| t.kind == "Le"));
        assert!(views.iter().any(|t| t.kind == "Ne"));
        // EOF marker has empty text.
        let eof = views.last().expect("eof");
        assert_eq!(eof.kind, "Eof");
        assert_eq!(eof.text, "");
    }

    #[test]
    fn reserved_words_get_their_own_kind() {
        let src = "while";
        let toks = helix_syntax::lex(src).expect("lexes");
        assert_eq!(TokenView::of(&toks[0], src).kind, "Reserved(while)");
    }

    #[test]
    fn artifact_round_trips_through_json() {
        let mut art = CompileArtifact::new("demo", "fn main() {}");
        art.diags_sem.push(DiagView {
            span: SpanView { start: 3, end: 7 },
            msg: "boom".into(),
        });
        art.exec = Some(ExecView {
            backend_used: "interp".into(),
            printed: vec!["42".into()],
            checksum: "0xdeadbeef".into(),
            timings_ms: None,
        });
        let json = serde_json::to_string(&art).expect("serializes");
        let back: CompileArtifact = serde_json::from_str(&json).expect("deserializes");
        assert_eq!(back.schema, 1);
        assert_eq!(back.diags_sem[0].msg, "boom");
        assert_eq!(back.exec.expect("exec").checksum, "0xdeadbeef");
        assert!(back.loops.is_none()); // skip_if_none keeps the payload lean
    }

    #[test]
    fn verdict_labels_serialize_screaming() {
        assert_eq!(
            serde_json::to_string(&VerdictLabel::Safe).unwrap(),
            "\"SAFE\""
        );
        assert_eq!(
            serde_json::to_string(&VerdictLabel::Reduction).unwrap(),
            "\"REDUCTION\""
        );
        assert_eq!(
            serde_json::to_string(&VerdictLabel::Sequential).unwrap(),
            "\"SEQUENTIAL\""
        );
    }
}
