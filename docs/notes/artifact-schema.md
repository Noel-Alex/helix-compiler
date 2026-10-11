# CompileArtifact JSON schema (v1) — the Observatory data contract

`GET /api/artifact?example=<name>` and `POST /api/run {source}` return this JSON.
The web UI renders EXACTLY this; helix-observe produces it. Field names normative.

```jsonc
{
  "schema": 1,
  "example": "saxpy",                  // name or "<adhoc>"
  "source": "fn main() { ... }",       // raw source text
  // ---- pipeline stages (null when compilation stopped earlier) ----
  "tokens": [ {"kind":"Kw","text":"fn","start":0,"end":2}, ... ],
  "ast": { /* Program as serialized by helix-syntax (serde) */ },
  "diags_sem": [ {"span":{"start":10,"end":12},"msg":"..."} ],   // [] when ok
  "ir_pre_ssa":  { "functions": [FuncIrText] },   // print_ir(ssa=false) strings
  "ir_ssa":      { "functions": [FuncIrText] },   // print_ir(ssa=true)
  "passes": [
     {"name":"const_fold","changed":true,
      "after":"<full IR text after pass>",
      "diff_stats":{"insts_before":42,"insts_after":37}}
  ],
  "cfg": {   // per-function CFG with LAYOUT PRECOMPUTED SERVER-SIDE
    "functions": [{
      "name":"main",
      "nodes":[{"id":"bb0","x":40,"y":20,"w":180,"h":64,
                "role":"entry|exit|loop_header|join|straight",
                "lines":["i0 = const 0","jump bb1(i0)"],
                "loop_id":0}],
      "edges":[{"from":"bb0","to":"bb1","kind":"fallthrough|branch|backedge",
                "points":[[130,84],[130,120]], "label":""}]
    }]
  },
  "domtree": {"main": {"bb0":["bb1","bb3"], "bb1":["bb2"]}},  // id -> children
  "loops": [
    {"id":0,"depth":1,"header":"bb1","blocks":["bb1","bb2","bb3"],
     "iv":"i","bounds":{"start":"0","end":"n"},
     "accesses":["READ a[i]","WRITE out[i]"],
     "raw":[],"war":[],"waw":[],
     "reduction":null,                     // or {"op":"+","var":"dot"}
     "verdict":"SAFE|REDUCTION|SEQUENTIAL",
     "reason":"no loop-carried dependences",        // human string
     "plan":{"threads":16}                 // when parallelized
    }
  ],
  "exec": {                                 // present after run
    "backend_used":"interp|jit_seq|jit_par",
    "printed":["0.0"], "checksum":"0x9e3779b97f4a7c15",
    "timings_ms":[2.9]  // one interactive-size interp sample; null when
                        // execution was skipped or failed. Reference-only
                        // (OS timer precision); use repeated benchmark
                        // samples for performance comparisons.
  },
  "bench": {                                // optional; from bench campaign
    "kernel":"saxpy","n":33554432,
    "variants":[
      {"name":"interpreter","median_ms":5700.0,"samples":[...]},
      {"name":"native-seq","median_ms":680.0,"samples":[...]},
      {"name":"native-par-8t","median_ms":95.0,"samples":[...]}
    ],
    "efficiency":[{"threads":8,"speedup":7.15,"efficiency":0.89}]
  }
}
```

The library compile pipeline leaves `bench` absent on a normal single run.
The HTTP observer attaches a recorded point from
`docs/benchmarks/data/campaign.json` only when the loaded executable tokens
match the measured source frozen in `crates/helix-artifact/src/recorded-sources.json`
at the catalog's revision. Comments and whitespace do not affect identity;
campaign matching permits only the campaign's declared `n`/`N`/`SIZE` size
change, not changes to `ITER` or other executable code. The attached point's
`n` is its measured size, which may differ from the editor's size. A teaching
example sharing a kernel's name but not its source receives no attached data.
Bundled language/IR examples may instead receive a dated seven-sample
`interp-local` probe from `docs/benchmarks/data/example-probes.json` on exact
token identity; those probes have no native, parallel, speedup, or efficiency
claim. Refresh the frozen sources when recording a new catalog revision. `POST /api/benchmark {source}` remains
the server-side bounded interpreter measurement for API clients, while the
Observatory's custom-source button measures the same reference interpreter
through the shipped WASM bridge so the result is available without a server
round-trip.

`GET /api/benchmarks` reports campaign provenance at the top level. Its
`probe_provenance` is separate: `{schema, timestamp_utc, git_rev, machine}`,
where `machine` contains `cpu_brand`, `logical_cores`, `physical_cores`, and
`rustc_version`. Probe cards use this provenance, never the campaign's date
or machine. It is `null` when unavailable; fixture indexes also return empty
`examples` and `points` arrays. Historical measurements do not certify the
current compiler build's performance.

Browser execution uses a fresh Web Worker per call with a five-second total
deadline (including loading). Completion, failure, and timeout terminate that
worker; there is no UI-thread execution fallback. Browser benchmark samples
use worker-reported `elapsedMs`, excluding worker startup, WASM loading, and
message transport. Timer resolution remains browser-dependent.

Rules:
- Every stage degrades gracefully: UI shows stages present, greys missing ones.
- Example loads, custom compilation, and browser measurements share latest-request-wins ordering; an older completion must not replace a newer selection.
- The example rail groups entries by category and keeps visual and accessible selection in sync. Custom source remains editable on mobile.
- Frontend regression checks: `node web/app.test.mjs` (Node.js standard library only).
- `cfg` layout coordinates are FINAL — browser paints, never computes.
- IR text blocks are plain monospace strings with \n.
- All spans byte offsets into `source`.
