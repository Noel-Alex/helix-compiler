# Research digest: observatory ui

_Verified August 2026 via parallel web research. Raw JSON: docs/research/raw/_

## Summary

For HELIX Lite's "Observatory" visualization app with zero npm toolchain, the strongest 2026 stack is axum 0.8.x (actively maintained, 0.8.9 April 2026) serving ~5 embedded assets via include_bytes!/rust-embed 8.x bound to 127.0.0.1, with a vanilla-JS frontend that reuses HELIX's own lexer tokens for syntax highlighting and renders SVG directly. Vendor d3 v7.9.0 UMD (~280 KB) locally once for AST tidy-tree layout, and either hand-roll a layered CFG layout (~100 lines) or better yet compute CFG coordinates in Rust and ship them in the JSON so the browser is a dumb renderer. Hand-rolled SVG bar charts beat Chart.js for two benchmark charts (zero deps, full dark-theme control, CSS-transition animations). A ratatui 0.30.x TUI is a poor fit for CFGs (braille canvas caps resolution, no native graph widget) — keep the TUI as a bonus text-mode view, not the demo centerpiece; a hybrid CLI that auto-opens the browser on localhost is the impressive path.

## Key facts

1. axum 0.8.9 released 2026-04-14, ~436M total downloads, published by Alice Ryhl (Tokio team) — actively maintained as of Aug 2026
2. warp 0.4.3 released 2026-05-04 (seanmonstar), ~46M downloads — maintained but filter-based API is clunkier for mixed JSON+static apps
3. tiny_http 0.12.0 last updated 2022-10-06 (~60M downloads) — lightest (no async runtime) but nearly 4 years without a release; you hand-roll routing, MIME types, and JSON responses
4. rust-embed 8.12.0 released 2026-07-08: reads files from disk in debug builds (live reload DX), embeds into binary in release builds; #[derive(RustEmbed)] #[folder = "dist/"]
5. axum SPA-without-tower-http pattern: Router::new().route("/api/...", get(handler)).fallback(get(|| async { Html(include_str!("index.html")) })); tower_http::ServeDir serves from filesystem only and CANNOT serve include_str!/include_bytes! data
6. d3 v7.9.0 is the latest v7: dist/d3.v7.min.js ~280 KB minified (~85 KB gzipped), UMD global `d3`; official d3js.org/getting-started offers direct downloads explicitly for offline use; the May 2026 GitHub PR claiming v7 went ESM-only (d3/d3#4125) was disputed and CLOSED — UMD remains fully supported
7. d3-hierarchy implements Buchheim et al. 2002 linear-time Reingold-Tilford tidy tree; layout is ~3 calls: const tree = d3.tree().nodeSize([40,120]); tree(d3.hierarchy(data)); then draw root.descendants()/root.links()
8. @dagrejs/dagre 3.1.1 ships dist/dagre.min.js UMD (~100 KB minified, MIT) attaching global `dagre`; maintenance resumed under the dagrejs org after years archived; Sugiyama-style layered layout ideal for CFGs
9. elkjs 0.12.0 ships lib/elk.bundled.js usable via <script> tag but it is ~1-2 MB (GWT-compiled Eclipse Layout Kernel) and dual-licensed EPL-2.0 OR GPL-3.0-or-later — heavier and less course-friendly than dagre
10. Chart.js: the npm package main entry is ESM and FAILS via plain <script> from file://; the correct offline file is dist/chart.umd.min.js (~200 KB min, self-contained, inlines @kurkle/color)
11. Prism.js core ~2 KB plus per-language definitions, registers custom grammars as plain objects (Prism.languages.x = {...}) with NO build step; highlight.js default bundle ~120 KB covering 190+ languages but custom grammars require module builds
12. ratatui 0.30.2 released 2026-06-19; Canvas widget draws points/lines/rectangles/circles via braille characters with no first-class node/edge concept; built-in BarChart is vertical-bars-only with no animation; no native graph widget exists (community ratatui-graph is experimental)
13. Minimal Reingold-Tilford implementations run 50-80 lines (recursive post-order x-assignment + subtree shifting); Bill Mill's llimllib.github.io/pymag-trees/ is the canonical walkthrough; Buchheim linear-time variant fixes Walker's O(n^2) worst case
14. Okabe-Ito colorblind-safe categorical hex values: Orange #E69F00, Sky Blue #56B4E9, Bluish Green #009E73, Yellow #F0E442, Blue #0072B2, Vermillion #D55E00, Reddish Purple #CC79A7; canonical citation Wong, Nature Methods 8:441 (2011)
15. WCAG 2.1 targets: >=4.5:1 contrast for text, >=3:1 for graphical objects; ~8% of males have red-green color deficiency so hue must never be the sole signal
16. One Dark Pro-derived syntax palette proven on dark backgrounds: keywords #c678dd, types/functions #56b6c2/#61afef, numbers #d19a66, strings #98c379, comments #5c6370, plain identifiers #abb2bf

## Recommendations

1. Serve with axum 0.8 behind a feature flag (`helix observe` subcommand): Router::new().route("/api/artifact", get(json_handler)).fallback(index_handler), binding 127.0.0.1:<port>, then auto-open the default browser via the `webbrowser` crate (or `cmd /c start` on Win11). Skip tower-http entirely — with <=6 assets you don't need ServeDir.
2. Embed assets with plain include_bytes! in a match-on-path fallback handler rather than adding rust-embed's proc-macro, UNLESS you want debug-build live reload — then rust-embed 8.12 is worth its one dependency (debug = disk, release = embedded). Set explicit Content-Type headers (.html/.css/.js/.svg); browsers ignore wrong MIME for scripts only in some cases — set them correctly.
3. Reuse YOUR OWN lexer as the syntax highlighter: the TOKENS phase already produces spans; emit `<span class="tok-kw">` etc. from those spans. Zero libraries, perfect phase-consistency between SOURCE and TOKENS views. Do NOT vendor Prism/highlight.js — they solve a problem you already solved in Rust.
4. Vendor exactly ONE third-party file: d3.v7.min.js 7.9.0 (~280 KB) pinned and committed under web/vendor/, loaded via local <script src="/vendor/d3.v7.min.js">. Use d3.tree() for the AST and d3 selections for CFG rendering. Never reference a CDN at runtime — demo machines may have no internet.
5. Strongest architecture for a compiler course: compute BOTH layouts server-side in Rust (tidy-tree for AST ~60 lines; longest-path layering + within-layer barycenter ordering for the CFG ~120 lines) and emit x/y/w/h in the JSON artifact. The browser becomes a pure SVG painter with zero layout code, the layout algorithms count as compiler-project work, and nothing can break offline.
6. If you'd rather not write CFG layout in Rust, add @dagrejs/dagre 3.1.1 dist/dagre.min.js (~100 KB, MIT) as a second vendored file — it handles back-edge routing and rank assignment far better than a quick hand-rolled pass. Avoid elkjs (1-2 MB, EPL/GPL license).
7. Hand-roll the benchmark bars as SVG: <rect> per measurement, y-axis ticks every nice number, value labels, hover tooltip via a positioned div, and CSS transition on height/width for animated growth — roughly 60 lines total. Chart.js UMD works offline but buys you nothing for 2 bar charts while costing 200 KB and fighting your dark theme.
8. Pipeline view (SOURCE->TOKENS->AST->CFG->SSA->LOOP ANALYSIS->BENCH): a horizontal stepper of chips, each phase a lazy-rendered panel, left/right arrow keys + click navigation, 150 ms opacity/translateY transition on swap. Animate SSA construction by diffing instruction ids between phases and flashing new ones; animate LICM by transitioning hoisted instructions' positions. All pure CSS + ~40 lines of JS — maximum wow, minimum fragility.
9. Loop-analysis verdict display: green solid-border badge + 'PARALLELIZED x N threads' for accepted loops; REJECTED loops get vermillion dashed border, a diagonal hazard-stripe SVG pattern overlay, and a tooltip listing the exact dependence cycle (array name + subscript pairs). Encode verdict redundantly (icon glyph + dash pattern + label) so it survives colorblind viewing and grayscale printing.
10. Palette (dark, tested conventions): page bg #0d1117, panel bg #161b22, borders #30363d, text #e6edf3. Block roles from lightened Okabe-Ito: entry #00C896 (green), exit #E8630A (vermillion), loop header #56B4E9 (sky blue), join #E5D75A (softened yellow), straight-line #8b949e on #21262d. Edges: fallthrough #6e7681, branch-taken #56B4E9, loop back-edge #E69F00 drawn curved with visible arrowheads. Syntax colors: One Dark Pro set listed in key_facts.
11. Keep a Graphviz escape hatch: emit .dot for the CFG alongside the web UI (~30 lines of code). Graders can render it themselves, and it cross-validates your own layout. Optional ratatui 0.30 TUI only for text-friendly phases (tokens table, SSA listing, reduction report, ASCII speedup bars) — treat as a bonus, budget it last.
12. For the demo, also inline the current program's JSON artifact into index.html at request time (or offer a 'Download standalone HTML' button that inlines it) so a double-clickable single file exists as a fallback if something goes wrong with the server on presentation day.

## Pitfalls

1. tiny_http has had no release since October 2022 — it works but is frozen; if anything HTTP-edge-case bites you on Win11 nobody is fixing it. axum's tokio dependency tree is heavy (~100+ transitive crates, longer clean builds) but that cost is paid once and it is the ecosystem default.
2. chart.umd.min.js is the ONLY Chart.js file that works via <script> tag; grabbing the npm main entry (ESM) fails silently-ish under file:// and even over HTTP in strict setups. Pin and commit whichever file you choose.
3. Do not rely on jsDelivr/unpkg at runtime 'with a local fallback' — demo-day machines with no internet will show a blank graph exactly when it matters. Vendoring is binary: commit the file or don't use the lib.
4. elkjs bundles the entire Java ELK via GWT (~1-2 MB) and is EPL-2.0/GPL-3.0 licensed — awkward heft and license for a college project when dagre does the same job at 100 KB MIT.
5. tower_http::ServeDir reads from the filesystem and cannot serve include_str!/include_bytes! content; mixing them naively causes 'works on my machine, 404 on the grader's'. Either embed everything or serve everything from disk — pick one per build profile.
6. include_str!/include_bytes! freeze asset content at compile time: edit app.js, forget to rebuild, demo shows stale UI. rust-embed's debug-mode disk reading eliminates exactly this trap.
7. Bind to 127.0.0.1, not 0.0.0.0 — a 0.0.0.0 bind triggers the Windows Firewall consent dialog mid-demo on locked-down college machines and may be silently blocked.
8. A fetch()-based frontend breaks if someone opens index.html via file:// (CORS/fetch restrictions); either always go through the server or inline the artifact JSON into the HTML.
9. ratatui Canvas renders braille dots — a 30-block CFG with back edges becomes unreadable noise, and BarChart cannot animate or do horizontal bars natively; building a credible TUI graph view is days of work for strictly worse visuals than the web UI. Don't let it eat your schedule.
10. Pure saturated colors (#FF0000, #00FF00, #F0E442 at full brightness) vibrate/halate on near-black backgrounds; darken/soften yellows and avoid red-vs-green as the ONLY distinction for parallelization verdicts (~8% of male viewers see them as identical).
11. SVG text measurement differs per font — if you size basic-block boxes from character counts, use a monospace stack (ui-monospace, Consolas) and pad generously, or measure with getComputedTextLength() after insertion; otherwise labels clip on Win11's default fonts.
12. d3 v7 UMD is safe today, but watch for v8/v9 pressure toward ESM-only: pin the vendored file to 7.9.0 and never 'refresh' it casually — it is a committed binary asset, not a dependency.

## References

- https://crates.io/crates/axum (v0.8.9, 2026-04-14)
- https://crates.io/crates/warp (v0.4.3, 2026-05-04)
- https://crates.io/crates/tiny_http (v0.12.0, 2022-10-06)
- https://crates.io/crates/rust-embed (v8.12.0, 2026-07-08)
- https://crates.io/crates/ratatui (v0.30.2, 2026-06-19)
- https://docs.rs/axum/latest/axum/serve/index.html (SPA fallback + ServeDir patterns)
- https://github.com/tokio-rs/axum/discussions/2412 (embedding static assets at compile time)
- https://docs.rs/rust-embed/latest/rust_embed/ (debug-from-disk, release-embedded semantics)
- https://stackoverflow.com/questions/76567890 (axum rust-embed SPA fallback handler)
- https://docs.rs/tower-http/latest/tower_http/services/struct.ServeDir.html (filesystem-only caveat)
- https://d3js.org/getting-started (offline d3.v7.min.js download; UMD usage)
- https://github.com/d3/d3/pull/4125 (disputed-and-closed ESM-only claim, May 2026)
- https://app.unpkg.com/d3@7.9.0/files/dist (d3.min.js ~280 KB UMD)
- https://github.com/d3/d3-hierarchy (Buchheim linear-time tidy tree)
- https://llimllib.github.io/pymag-trees/ (Bill Mill, minimal tidy-tree walkthrough)
- https://github.com/dagrejs/dagre (@dagrejs/dagre 3.1.1, dist/dagre.min.js UMD)
- https://www.jsdelivr.com/package/npm/@dagrejs/dagre
- https://github.com/kieler/elkjs (lib/elk.bundled.js ~1-2 MB, EPL-2.0/GPL-3.0)
- https://www.jsdelivr.com/package/npm/elkjs (v0.12.0)
- https://www.chartjs.org/docs/latest/ (chart.umd.min.js script-tag usage)
- https://stackoverflow.com/questions/tagged/chart.js+offline (ESM main entry fails via script tag; UMD required)
- https://github.com/chartjs/Chart.js/discussions/11169 (single-file build workflow)
- https://www.libhunt.com/compare-highlight-js-vs-prism (Prism modular/custom-grammar vs highlight.js breadth)
- https://blog.devtoolsweekly.com/client-side-syntax-highlighting-2026 (vendored highlighter comparison)
- https://docs.rs/ratatui/latest/ratatui/widgets/canvas/index.html (Canvas shapes, no node/edge primitives)
- https://personal.sron.nl/~pault/ (Paul Tol colorblind-safe schemes)
- Wong, B. 'Points of view: Color blindness', Nature Methods 8:441 (2011) (canonical Okabe-Ito citation)
- Okabe M. & Ito K., 'Color Universal Design' (2008) (Okabe-Ito palette origin: #E69F00 #56B4E9 #009E73 #F0E442 #0072B2 #D55E00 #CC79A7)
- https://www.w3.org/TR/WCAG21/ (4.5:1 text / 3:1 graphics contrast requirements)
