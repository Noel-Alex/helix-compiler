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
    "timings_ms": null                       // filled by bench, not single run
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

Rules:
- Every stage degrades gracefully: UI shows stages present, greys missing ones.
- `cfg` layout coordinates are FINAL — browser paints, never computes.
- IR text blocks are plain monospace strings with \n.
- All spans byte offsets into `source`.
