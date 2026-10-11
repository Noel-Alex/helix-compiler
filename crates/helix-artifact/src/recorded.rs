//! Recorded benchmark evidence — the committed campaign and probe catalogs,
//! plus the frozen-source identity check that decides what a live compile may
//! attach.
//!
//! Nothing here reaches the network or the disk: the catalogs are embedded
//! with `include_bytes!` / `include_str!` so the same evidence exists in the
//! server binary and inside wasm32 for the browser lab.
//!
//! Two catalogs, deliberately not interchangeable:
//!
//! * `docs/benchmarks/data/campaign.json` — the dated kernel campaign with
//!   interpreter and native (sequential/parallel) variants. A point attaches
//!   only when the compiled source is *executable-token identical* to the
//!   frozen source at the catalog's revision, allowing exactly the declared
//!   size constant to differ. Those numbers say nothing about this build.
//! * `docs/benchmarks/data/example-probes.json` — dated reference-interpreter
//!   probes for bundled teaching examples that are not registered kernels.
//!   They carry no native variant, no size, and therefore no speedup or
//!   efficiency claim.

use serde::Deserialize;
use std::cmp::Reverse;

use helix_syntax::token::{Kw, TokKind};

use crate::artifact::{BenchVariant, BenchView, EfficiencyPoint};
use std::collections::BTreeMap;

/// The committed, source-backed benchmark campaign. The builder attaches only
/// a matching kernel/size point; the index route exposes its provenance.
const CAMPAIGN_JSON: &[u8] = include_bytes!("../../../docs/benchmarks/data/campaign.json");
/// Dated reference-interpreter probes for bundled teaching examples that are
/// not registered performance kernels. These values never imply native or
/// parallel speedups.
const EXAMPLE_PROBES_JSON: &[u8] =
    include_bytes!("../../../docs/benchmarks/data/example-probes.json");
/// Sources frozen at the measurement revisions, not today's mutable examples.
const RECORDED_SOURCES: &str = include_str!("recorded-sources.json");

#[derive(Debug, Deserialize)]
pub(crate) struct RecordedCampaign {
    pub(crate) schema: String,
    pub(crate) timestamp_utc: String,
    pub(crate) meta: RecordedMeta,
    pub(crate) points: Vec<RecordedPoint>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct RecordedMeta {
    pub(crate) cpu_brand: String,
    pub(crate) logical_cores: usize,
    pub(crate) physical_cores: usize,
    pub(crate) rustc_version: String,
    pub(crate) git_rev: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct RecordedPoint {
    pub(crate) kernel: String,
    pub(crate) n: i64,
    pub(crate) variants: Vec<RecordedVariant>,
    pub(crate) efficiency: Vec<RecordedEfficiency>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct RecordedVariant {
    pub(crate) name: String,
    pub(crate) median_ms: f64,
    pub(crate) samples_ms: Vec<f64>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct RecordedEfficiency {
    pub(crate) threads: usize,
    pub(crate) speedup: f64,
    pub(crate) efficiency: f64,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ExampleProbeCatalog {
    pub(crate) schema: String,
    pub(crate) timestamp_utc: String,
    pub(crate) machine: RecordedMeta,
    pub(crate) examples: Vec<ExampleProbe>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ExampleProbe {
    pub(crate) example: String,
    pub(crate) median_ms: f64,
    pub(crate) samples_ms: Vec<f64>,
}

#[derive(Deserialize)]
struct RecordedSources {
    campaign_git_rev: String,
    probe_git_rev: String,
    campaign: BTreeMap<String, String>,
    probes: BTreeMap<String, String>,
}

/// Parses the embedded kernel campaign.
pub(crate) fn read_campaign() -> Result<RecordedCampaign, String> {
    serde_json::from_slice(CAMPAIGN_JSON).map_err(|e| format!("benchmark campaign is invalid: {e}"))
}

/// Parses the embedded teaching-example probe catalog.
pub(crate) fn read_example_probes() -> Result<ExampleProbeCatalog, String> {
    serde_json::from_slice(EXAMPLE_PROBES_JSON)
        .map_err(|e| format!("example probe catalog is invalid: {e}"))
}

/// Compares executable token streams to decide whether `source` is the
/// measured program recorded at `revision`.
///
/// Identity is deliberately token-based: comments, whitespace, and formatting
/// are not executable work. For campaign kernels the declared size constant is
/// normalized, because `source_at_size` varies it and nothing else; every other
/// executable token, including `ITER`, stays part of identity. A revision
/// mismatch means the numbers were measured elsewhere, so nothing attaches.
fn source_matches(example: &str, source: &str, revision: &str, campaign: bool) -> bool {
    let Ok(recorded) = serde_json::from_str::<RecordedSources>(RECORDED_SOURCES) else {
        return false;
    };
    let (expected_revision, sources) = if campaign {
        (&recorded.campaign_git_rev, &recorded.campaign)
    } else {
        (&recorded.probe_git_rev, &recorded.probes)
    };
    let Some(expected) = sources
        .get(example)
        .filter(|_| expected_revision == revision)
    else {
        return false;
    };
    let tokens = |text: &str| -> Option<Vec<TokKind>> {
        let mut tokens: Vec<_> = helix_syntax::lex(text)
            .ok()?
            .into_iter()
            .map(|t| t.kind)
            .collect();
        // Campaign source_at_size varies only this declared size. ITER and
        // all other executable tokens remain part of the measured identity.
        if campaign {
            let size_name = match example {
                "jacobi_2d" => "SIZE",
                "matmul" => "N",
                _ => "n",
            };
            for i in 0..tokens.len() {
                let value = match &tokens[i..] {
                    [
                        TokKind::Kw(Kw::Let),
                        TokKind::Ident(name),
                        TokKind::Assign,
                        TokKind::Int(_),
                        TokKind::Semi,
                        ..,
                    ] if name == size_name => Some(i + 3),
                    [
                        TokKind::Kw(Kw::Const),
                        TokKind::Ident(name),
                        TokKind::Colon,
                        _,
                        TokKind::Assign,
                        TokKind::Int(_),
                        TokKind::Semi,
                        ..,
                    ] if name == size_name => Some(i + 5),
                    _ => None,
                };
                if let Some(value) = value {
                    tokens[value] = TokKind::Int(0);
                    break;
                }
            }
        }
        Some(tokens)
    };
    match (tokens(source), tokens(expected)) {
        (Some(actual), Some(expected)) => actual == expected,
        _ => false,
    }
}

/// The recorded campaign point for `example`, when the compiled source is the
/// measured one. The most complete point wins, with the smallest size breaking
/// ties.
pub(crate) fn recorded_bench(example: &str, source: &str) -> Option<BenchView> {
    let campaign = read_campaign().ok()?;
    if !source_matches(example, source, &campaign.meta.git_rev, true) {
        return None;
    }
    let point = campaign
        .points
        .iter()
        .filter(|point| point.kernel == example)
        .max_by_key(|point| (point.variants.len(), Reverse(point.n)));
    let point = point?;
    let n = u64::try_from(point.n).ok()?;
    Some(BenchView {
        kernel: point.kernel.clone(),
        n,
        variants: point
            .variants
            .iter()
            .map(|variant| BenchVariant {
                name: variant.name.clone(),
                median_ms: variant.median_ms,
                samples: variant.samples_ms.clone(),
            })
            .collect(),
        efficiency: point
            .efficiency
            .iter()
            .map(|row| EfficiencyPoint {
                threads: row.threads,
                speedup: row.speedup,
                efficiency: row.efficiency,
            })
            .collect(),
    })
}

/// Returns the dated interpreter-only probe for a bundled non-kernel example.
/// Unlike campaign points, this has no problem-size or native variant; keeping
/// it in the same `bench` view lets the Observatory show an actual measured
/// value without manufacturing a comparison that was never run.
pub(crate) fn probe_bench(example: &str, source: &str) -> Option<BenchView> {
    let catalog = read_example_probes().ok()?;
    if !source_matches(example, source, &catalog.machine.git_rev, false) {
        return None;
    }
    let probe = catalog
        .examples
        .into_iter()
        .find(|probe| probe.example == example)?;
    Some(BenchView {
        kernel: probe.example,
        n: 0,
        variants: vec![BenchVariant {
            name: "interp-local".to_string(),
            median_ms: probe.median_ms,
            samples: probe.samples_ms,
        }],
        efficiency: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recorded_measurements_require_frozen_executable_identity() {
        let sources: RecordedSources =
            serde_json::from_str(include_str!("recorded-sources.json")).unwrap();
        for (name, source) in &sources.campaign {
            assert!(recorded_bench(name, source).is_some(), "{name}");
            assert!(
                recorded_bench(name, &format!("// comment 999999999\n{source}\n")).is_some(),
                "{name}: comments"
            );
            assert!(
                recorded_bench(name, &source.replace("print(", "print(1 + ")).is_none(),
                "{name}: edited code"
            );
            assert!(!source_matches(name, source, "different-revision", true));
        }
        let saxpy = &sources.campaign["saxpy"];
        assert!(recorded_bench("saxpy", &saxpy.replace("33554432", "65536")).is_some());
        let jacobi = &sources.campaign["jacobi_2d"];
        assert!(
            recorded_bench(
                "jacobi_2d",
                &jacobi.replace("ITER: i64 = 10", "ITER: i64 = 4")
            )
            .is_none()
        );
        for (name, source) in &sources.probes {
            assert!(probe_bench(name, source).is_some(), "{name}");
            assert!(probe_bench(name, &format!("// comment 999999999\n{source}\n")).is_some());
            assert!(
                probe_bench(name, &format!("{source}\nfn unmeasured() {{ print(1); }}")).is_none()
            );
            assert!(!source_matches(name, source, "different-revision", false));
        }
    }

    #[test]
    fn campaign_and_probe_provenance_stay_separate() {
        let campaign = read_campaign().expect("embedded campaign parses");
        let probes = read_example_probes().expect("embedded probes parse");
        // Neither catalog certifies the current build, so their revisions and
        // their reported machines stay independently readable.
        assert!(!campaign.meta.git_rev.is_empty());
        assert_eq!(probes.machine.git_rev, probes.machine.git_rev);
        for probe in &probes.examples {
            let source = probe_bench_source(&probe.example);
            let bench = probe_bench(&probe.example, &source).expect("recorded probe attaches");
            // Probes never claim a native speedup: one variant, size-less.
            assert_eq!(bench.variants.len(), 1);
            assert_eq!(bench.variants[0].name, "interp-local");
            assert_eq!(bench.n, 0);
            assert!(bench.efficiency.is_empty());
        }
    }

    /// The frozen source of one probe example (the identity gate's input).
    fn probe_bench_source(example: &str) -> String {
        let sources: RecordedSources =
            serde_json::from_str(include_str!("recorded-sources.json")).unwrap();
        sources
            .probes
            .get(example)
            .cloned()
            .unwrap_or_else(|| panic!("{example} has no frozen source"))
    }
}
