/* ============================================================
   HELIX Observatory — app.js
   Vanilla-JS renderer for the CompileArtifact JSON (schema v1).
   No network calls except /api/*. d3 (vendored) is used only for
   the AST tidy tree and pan/zoom.
   ============================================================ */
'use strict';

/* ---------------------------------------------------------------- utils */
const $ = sel => document.querySelector(sel);
const $$ = sel => Array.from(document.querySelectorAll(sel));
const esc = s => String(s).replace(/[&<>"']/g,
  c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmt = (x, d = 1) => Number(x).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });

/** Run fn on the next frame; falls back to a short timeout when the browser
 *  is not compositing (hidden preview panes), so animations always finish. */
function onNextFrame(fn) {
  let ran = false;
  const run = () => { if (!ran) { ran = true; fn(); } };
  requestAnimationFrame(run);
  setTimeout(run, 60);
}

const PHASES = [
  ['source', 'SOURCE'], ['tokens', 'TOKENS'], ['ast', 'AST'], ['cfg', 'CFG'],
  ['ssa', 'SSA'], ['opt', 'OPT'], ['loops', 'LOOP ANALYSIS'], ['bench', 'BENCH'],
];

const state = {
  examples: [],
  benchmarkCatalog: null,
  exampleName: '',
  artifact: null,
  phase: 'source',
  cfgFn: 0,
  ssaFn: 0,
  benchLog: false,
  benchMessage: '',
  benchPending: false,
  requestSeq: 0,
  browserRun: null,
  browserRunSeq: 0,
  // Static hosting (GitHub Pages) answers no /api/*: every request then runs
  // through the WASM bridge, and localBench holds the two-engine measurement.
  static: false,
  localBench: null,
};

const EXAMPLE_GROUPS = [
  {
    key: 'kernels',
    label: 'Kernels',
    note: 'numerical loops and reductions',
    names: new Set(['count_primes_sieve', 'dot_reduction', 'fib_recursion', 'jacobi_2d', 'matmul', 'minmax_reduction', 'saxpy', 'scale', 'small_n']),
  },
  {
    key: 'language',
    label: 'Language / IR',
    note: 'types, control flow, and lowering',
    names: new Set(['casts_demo', 'const_globals', 'div_guard', 'gcd_box_test', 'shortcircuit', 'ssa_demo', 'writer_call']),
  },
  {
    key: 'refusal',
    label: 'Refusal / diagnostics',
    note: 'useful failures kept in view',
    names: new Set(['recurrence_reject', 'stencil_2d_reject', 'type_errors']),
  },
];

const FALLBACK_EXAMPLES = [
  'casts_demo', 'const_globals', 'count_primes_sieve', 'div_guard', 'dot_reduction',
  'fib_recursion', 'gcd_box_test', 'jacobi_2d', 'matmul', 'minmax_reduction',
  'recurrence_reject', 'saxpy', 'scale', 'shortcircuit', 'small_n', 'ssa_demo',
  'stencil_2d_reject', 'type_errors', 'writer_call',
];

function exampleGroup(name) {
  return EXAMPLE_GROUPS.find(group => group.names.has(name)) ?? EXAMPLE_GROUPS[1];
}

function benchmarkKernel(name) {
  return state.benchmarkCatalog?.kernels?.find(kernel => kernel.name === name) ?? null;
}

function benchmarkProbe(name) {
  return state.benchmarkCatalog?.examples?.find(example => example.name === name) ?? null;
}

function benchmarkPoints(name) {
  const points = state.benchmarkCatalog?.points;
  return Array.isArray(points) ? points.filter(point => point.kernel === name) : [];
}

function benchmarkVariant(variants, name) {
  return variants?.find(variant => variant.name === name) ?? null;
}

function benchmarkVariantLabel(variant) {
  if (!variant) return '— not recorded';
  const samples = variant.samples?.length ?? variant.sample_count;
  const sampleLabel = Number.isFinite(Number(samples)) ? ` · ${samples} samples` : '';
  return `${formatMilliseconds(variant.median_ms)} ms median${sampleLabel}`;
}

function formatMilliseconds(value) {
  const ms = Number(value);
  const digits = ms > 0 && ms < 1 ? 3 : ms >= 100 ? 1 : 2;
  return fmt(ms, digits);
}

function benchmarkPair(from, to) {
  if (!from || !to) return null;
  const before = Number(from.median_ms);
  const after = Number(to.median_ms);
  if (!(before > 0) || !(after > 0)) return null;
  const ratio = before / after;
  if (!Number.isFinite(ratio)) return null;
  return ratio >= 1
    ? `${fmt(ratio, 2)}× faster`
    : `${fmt(1 / ratio, 2)}× slower`;
}

/** Keep the WASM preview responsive: the work estimator lives in the shared
 *  WASM bridge module so the guide's browser lab enforces the same budget.
 *  It is passed in by callers after the bridge module loads. */
function browserPreviewAllowed(art, estimatedBrowserWork) {
  return !!art?.source && !art.diags_lex && !(art.diags_sem?.length) &&
    new TextEncoder().encode(art.source).length <= 64 * 1024 &&
    estimatedBrowserWork(art.source) <= 2_000_000;
}

function appendBenchmarkCard(parent, label, value, detail, tone = '') {
  const card = document.createElement('article');
  card.className = `bench-compare-card${tone ? ` ${tone}` : ''}`;
  const title = document.createElement('p');
  title.className = 'bench-compare-label';
  title.textContent = label;
  const result = document.createElement('strong');
  result.className = 'bench-compare-value';
  result.textContent = value;
  const meta = document.createElement('p');
  meta.className = 'bench-compare-meta';
  meta.textContent = detail;
  card.append(title, result, meta);
  parent.appendChild(card);
}

function renderExecReadout(art) {
  const panel = $('#exec-readout');
  if (!panel) return;
  panel.textContent = '';
  panel.classList.toggle('hidden', !art);
  if (!art) return;

  const heading = document.createElement('div');
  heading.className = 'exec-readout-heading';
  const title = document.createElement('strong');
  title.textContent = 'Execution readout';
  const source = document.createElement('span');
  source.className = 'exec-readout-source';
  source.textContent = 'same source · observable output';
  heading.append(title, source);
  panel.appendChild(heading);

  const run = state.browserRun;
  if (run?.status === 'running') {
    const pending = document.createElement('p');
    pending.className = 'exec-readout-pending';
    pending.textContent = 'Running the reference interpreter in this browser…';
    panel.appendChild(pending);
    return;
  }

  const result = run?.result ?? null;
  const serverExec = art.exec;
  // One evidence source paints output/checksum/status/engine: a failed or
  // absent browser run must not dilute the observer's own execution with
  // empty browser fields (browser failures still show their error below).
  const browserOk = result?.ok === true;
  const evidence = browserOk || !serverExec ? result : serverExec;
  const values = evidence?.printed ?? [];
  const checksum = evidence?.checksum ?? '';
  const budgetSkipped = !result && values.some(value => String(value).startsWith('skipped:'));
  const status = document.createElement('span');
  status.className = `exec-status ${browserOk || (!result && checksum) ? 'is-ok' : 'is-muted'}`;
  status.textContent = browserOk ? 'WASM RESULT' : (serverExec?.checksum ? 'OBSERVER RESULT' : (budgetSkipped ? 'BUDGET SKIP' : 'NO EXECUTION'));

  const meta = document.createElement('div');
  meta.className = 'exec-readout-meta';
  meta.appendChild(status);
  if (browserOk && result.engine) {
    const engine = document.createElement('span');
    engine.textContent = result.engine;
    meta.appendChild(engine);
  } else if (serverExec?.backend_used) {
    const engine = document.createElement('span');
    engine.textContent = serverExec.backend_used;
    meta.appendChild(engine);
  }
  panel.appendChild(meta);

  if (run?.status === 'skipped') {
    const note = document.createElement('p');
    note.className = 'exec-readout-note';
    note.textContent = run.message;
    panel.appendChild(note);
  } else if (result?.ok === false) {
    const error = document.createElement('p');
    error.className = 'exec-readout-error';
    error.textContent = result.message || 'The browser interpreter rejected this program.';
    panel.appendChild(error);
  }

  const output = document.createElement('pre');
  output.className = 'exec-output';
  output.textContent = values.length ? values.join('\n') : '∅  program printed no lines';
  output.setAttribute('aria-label', 'program output');
  panel.appendChild(output);

  const foot = document.createElement('div');
  foot.className = 'exec-readout-foot';
  if (checksum) {
    const check = document.createElement('code');
    check.textContent = `checksum ${checksum}`;
    foot.appendChild(check);
  }
  if (result?.ok && result.checksum && serverExec?.checksum) {
    const parity = document.createElement('span');
    const same = result.checksum === serverExec.checksum;
    parity.className = same ? 'exec-parity is-ok' : 'exec-parity is-error';
    parity.textContent = same ? 'matches observer output' : 'observer/WASM output differs';
    foot.appendChild(parity);
  }
  // Server-side timing sample (interactive-sized only; benchmark-sized runs
  // skip execution). Reference-only — timer precision is OS-dependent.
  const serverTiming = serverExec?.timings_ms?.[0];
  if (Number.isFinite(serverTiming) && serverTiming >= 0) {
    const t = document.createElement('code');
    t.textContent = `interp ${serverTiming < 1 ? serverTiming.toFixed(3) : serverTiming.toFixed(1)} ms`;
    foot.appendChild(t);
  }
  if (foot.childElementCount) panel.appendChild(foot);
}

/**
 * Make the comparison basis explicit. The artifact endpoint attaches the
 * most complete point so the chart stays useful, while the campaign index
 * exposes every recorded size for scale-aware comparisons.
 */
function renderBenchComparison(bench) {
  const wrap = $('#bench-comparison');
  if (!wrap) return;
  wrap.textContent = '';
  if (!bench) return;

  const intro = document.createElement('p');
  intro.className = 'bench-comparison-intro';
  wrap.appendChild(intro);

  if (bench.kernel === '<adhoc>' || benchmarkProbe(bench.kernel)) {
    const probe = benchmarkProbe(bench.kernel);
    const provenance = state.benchmarkCatalog?.probe_provenance;
    intro.textContent = probe
      ? 'Recorded example probe · reference interpreter only · no native or parallel baseline was recorded for this source.'
      : 'Local measurement · WASM reference interpreter only · no native or parallel baseline was recorded for this source.';
    const variant = bench.variants?.[0];
    appendBenchmarkCard(
      wrap,
      'Reference execution',
      variant ? benchmarkVariantLabel(variant) : 'Not available',
      probe
        ? `Captured ${provenance?.timestamp_utc?.slice(0, 10) ?? 'date unavailable'} · revision ${provenance?.git_rev ?? 'unavailable'} · ${provenance?.machine?.cpu_brand ?? 'machine unavailable'}.`
        : 'Custom source is measured locally in this browser; no backend timing is invented.',
      variant ? 'bench-compare-positive' : 'bench-compare-muted',
    );
    return;
  }

  const variants = bench.variants ?? [];
  const interp = benchmarkVariant(variants, 'interp');
  const seq = benchmarkVariant(variants, 'native-seq');
  const parallel = benchmarkVariant(variants, 'native-par');
  intro.textContent = `Pairwise medians at n = ${Number(bench.n).toLocaleString('en-US')} · lower wall-clock time is better.`;

  const grid = document.createElement('div');
  grid.className = 'bench-compare-grid';
  const interpSpeed = benchmarkPair(interp, seq);
  appendBenchmarkCard(
    grid,
    'Interpreter → native sequential',
    interpSpeed ?? 'Not comparable',
    interp && seq
      ? `${benchmarkVariantLabel(interp)} → ${benchmarkVariantLabel(seq)}`
      : `interp ${benchmarkVariantLabel(interp)} · native-seq ${benchmarkVariantLabel(seq)}`,
    interpSpeed ? 'bench-compare-positive' : 'bench-compare-muted',
  );
  const parallelSpeed = benchmarkPair(seq, parallel);
  appendBenchmarkCard(
    grid,
    'Native sequential → native parallel',
    parallelSpeed ?? 'Not comparable',
    seq && parallel
      ? `${benchmarkVariantLabel(seq)} → ${benchmarkVariantLabel(parallel)}`
      : `native-seq ${benchmarkVariantLabel(seq)} · native-par ${benchmarkVariantLabel(parallel)}`,
    parallelSpeed ? 'bench-compare-positive' : 'bench-compare-muted',
  );
  wrap.appendChild(grid);

  if (!interp) {
    const note = document.createElement('p');
    note.className = 'bench-recorded-note';
    note.textContent = 'Interpreter baseline not recorded at this size. Large campaign points omit the interpreter to keep the campaign bounded; the native sequential → parallel comparison remains valid.';
    wrap.appendChild(note);
  }
  if (!parallel) {
    const note = document.createElement('p');
    note.className = 'bench-recorded-note';
    note.textContent = 'No native parallel variant was recorded for this point. The compiler kept this program sequential or the campaign did not define a parallel path.';
    wrap.appendChild(note);
  }

  const points = benchmarkPoints(bench.kernel);
  if (points.length > 1) {
    const heading = document.createElement('p');
    heading.className = 'bench-recorded-heading';
    heading.textContent = `Recorded sizes · ${points.length} points for ${bench.kernel}`;
    wrap.appendChild(heading);

    const tableWrap = document.createElement('div');
    tableWrap.className = 'bench-size-table-wrap';
    const table = document.createElement('table');
    table.className = 'bench-size-table';
    table.innerHTML = '<thead><tr><th scope="col">n</th><th scope="col">interp</th><th scope="col">native-seq</th><th scope="col">native-par</th><th scope="col">readout</th></tr></thead>';
    const body = document.createElement('tbody');
    for (const point of points) {
      const tr = document.createElement('tr');
      if (Number(point.n) === Number(bench.n)) tr.className = 'is-selected';
      const pointVariants = point.variants ?? [];
      const pointInterp = benchmarkVariant(pointVariants, 'interp');
      const pointSeq = benchmarkVariant(pointVariants, 'native-seq');
      const pointParallel = benchmarkVariant(pointVariants, 'native-par');
      const nCell = document.createElement('th');
      nCell.scope = 'row';
      nCell.textContent = Number(point.n).toLocaleString('en-US');
      tr.appendChild(nCell);
      for (const variant of [pointInterp, pointSeq, pointParallel]) {
        const td = document.createElement('td');
        td.textContent = variant ? benchmarkVariantLabel(variant) : '—';
        tr.appendChild(td);
      }
      const readout = document.createElement('td');
      const pointPair = benchmarkPair(pointInterp, pointSeq) ?? benchmarkPair(pointSeq, pointParallel);
      readout.textContent = pointPair ?? 'not comparable';
      tr.appendChild(readout);
      body.appendChild(tr);
    }
    table.appendChild(body);
    tableWrap.appendChild(table);
    wrap.appendChild(tableWrap);
  }
}

/* ------------------------------------------------------------- toast */
let toastTimer = null;
function toast(msg) {
  let t = $('#toast');
  if (!t) { t = document.createElement('div'); t.id = 'toast'; document.body.appendChild(t); }
  t.textContent = msg;
  t.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove('show'), 2600);
}

/* ============================================================
   SOURCE rendering — spans from artifact.tokens
   ============================================================ */

// TokKind (serde variant) -> CSS class
const TOK_CLASS = new Map(Object.entries({
  Kw: 'tok-kw', Ident: 'tok-id', Int: 'tok-num', Float: 'tok-num',
  LParen: 'tok-op', RParen: 'tok-op', LBrace: 'tok-op', RBrace: 'tok-op',
  LBracket: 'tok-op', RBracket: 'tok-op', Comma: 'tok-op', Semi: 'tok-op',
  Colon: 'tok-op', PathSep: 'tok-op', DotDot: 'tok-op', Arrow: 'tok-op',
  Plus: 'tok-op', Minus: 'tok-op', Star: 'tok-op', Slash: 'tok-op', Rem: 'tok-op',
  Lt: 'tok-op', Gt: 'tok-op', Le: 'tok-op', Ge: 'tok-op', Eq: 'tok-op', Ne: 'tok-op',
  AndAnd: 'tok-op', OrOr: 'tok-op', Not: 'tok-op', Assign: 'tok-op',
}));

const KEYWORDS = new Set(['fn','let','const','if','else','for','return','true','false','as','in']);

function tokenClass(tk) {
  let cls = TOK_CLASS.get(tk.kind) || 'tok-id';
  if (tk.kind === 'Comment') cls = 'tok-cmt';
  if (cls === 'tok-id' && KEYWORDS.has(tk.text)) cls = 'tok-kw';
  if ((tk.kind === 'Ident' || tk.kind === 'Kw') && (tk.text === 'true' || tk.text === 'false')) cls = 'tok-num';
  if (cls === 'tok-kw' && ['i32','i64','f32','f64','bool'].includes(tk.text)) cls = 'tok-ty';
  return cls;
}

function buildLineMap(source) {
  // JavaScript indices are UTF-16 code-unit offsets. Rust spans are UTF-8
  // byte offsets, but line starts are consumed after conversion to JS.
  const starts = [0];
  for (let i = 0; i < source.length; i++) if (source[i] === '\n') starts.push(i + 1);
  return starts;
}

/** Convert a Rust UTF-8 byte span offset to a JavaScript string offset. */
function buildByteToJsMap(source) {
  const map = [0];
  const encoder = new TextEncoder();
  let byteOffset = 0;
  let jsOffset = 0;
  for (const ch of source) {
    const byteLength = encoder.encode(ch).length;
    for (let i = 0; i < byteLength; i++) map[byteOffset + i] = jsOffset;
    byteOffset += byteLength;
    jsOffset += ch.length;
    map[byteOffset] = jsOffset;
  }
  return map;
}

function jsOffsetForByte(map, byteOffset, sourceLength) {
  const bounded = Math.min(Math.max(Number(byteOffset) || 0, 0), map.length - 1);
  return Math.min(map[bounded] ?? sourceLength, sourceLength);
}

/** Render the SOURCE phase from artifact.tokens. Returns nothing. */
function renderSource(art) {
  const tbody = $('#source-view tbody');
  tbody.textContent = '';
  $('#src-sub').textContent =
    art ? (Array.isArray(art.tokens)
      ? `${art.tokens.length} tokens · ${art.source.split('\n').length} lines`
      : `source-only fixture · ${(art.source ?? '').split('\n').length} lines`) : '';

  const diags = compilerDiagnostics(art);
  const showDiag = $('#src-show-diag').checked;

  if (!Array.isArray(art?.tokens)) {
    const rawSource = typeof art?.source === 'string' ? art.source : '';
    if (!rawSource) {
      tbody.innerHTML = `<tr><td class="empty">// source not available in this artifact</td></tr>`;
      return;
    }
    rawSource.split('\n').forEach((line, index) => {
      const tr = document.createElement('tr');
      tr.className = 'code-line fixture-source-line';
      const ln = document.createElement('td');
      ln.className = 'ln-col';
      ln.textContent = index + 1;
      const cell = document.createElement('td');
      cell.className = 'code-cell';
      cell.textContent = line;
      tr.append(ln, cell);
      tbody.appendChild(tr);
    });
    return;
  }

  const src = art.source ?? '';
  const lineStarts = buildLineMap(src);
  const byteToJs = buildByteToJsMap(src);

  // group tokens by line; fill gaps between tokens with plain text
  const perLine = new Map(); // line -> [{start,end,text,cls}]
  const pushTok = (line, item) => {
    if (!perLine.has(line)) perLine.set(line, []);
    perLine.get(line).push(item);
  };
  let prevEnd = 0;
  for (const tk of art.tokens) {
    if (tk.kind === 'Eof' || (tk.start === tk.end && !tk.text)) continue;
    tk._idx ??= art.tokens.indexOf(tk);
    if (!Number.isInteger(tk.start) || !Number.isInteger(tk.end) || tk.end < tk.start) continue;
    const s = jsOffsetForByte(byteToJs, tk.start, src.length);
    const e = Math.min(Math.max(jsOffsetForByte(byteToJs, tk.end, src.length), s), src.length);
    const text = tk.text || src.slice(s, e) || '';
    if (s > prevEnd) pushTok(null, { start: prevEnd, end: s });
    pushTok(lineAt(lineStarts, s), { start: s, end: e, text, cls: tokenClass(tk), tok: tk });
    prevEnd = Math.max(prevEnd, e);
  }
  if (prevEnd < src.length) pushTok(null, { start: prevEnd, end: src.length });

  function lineAt(starts, off) {
    let lo = 0, hi = starts.length - 1;
    while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (starts[mid] <= off) lo = mid; else hi = mid - 1; }
    return lo;
  }

  // flatten into a single ordered span stream, then split on \n
  const flat = [];
  for (const arr of perLine.values()) flat.push(...arr);
  flat.sort((a, b) => a.start - b.start);

  const diagByLine = new Map();
  for (const d of diags) {
    const start = jsOffsetForByte(byteToJs, d.span?.start, src.length);
    const end = Math.max(start, jsOffsetForByte(byteToJs, d.span?.end, src.length));
    const ln = lineAt(lineStarts, Math.min(start, Math.max(0, src.length - 1)));
    if (!diagByLine.has(ln)) diagByLine.set(ln, []);
    diagByLine.get(ln).push({ ...d, _start: start, _end: end });
  }

  const rows = [[]];
  for (const it of flat) {
    if (it.cls === undefined && it.tok === undefined) {
      // gap filler — may contain newlines
      const gapText = src.slice(it.start, it.end);
      for (const ch of gapText) {
        if (ch === '\n') rows.push([]);
        else rows[rows.length - 1].push({ text: ch, cls: '' , start: it.start, end: it.end });
      }
      continue;
    }
    let s = it.start;
    for (const ch of it.text) {
      if (ch === '\n') { s += ch.length; rows.push([]); continue; }
      rows[rows.length - 1].push({ text: ch, cls: it.cls || '', start: s, end: s + ch.length, tok: it.tok });
      s += ch.length;
    }
  }
  while (rows.length < lineStarts.length) rows.push([]);

  // merge adjacent same-class fragments into spans (cheap + robust)
  const frag = (rowArr) => {
    const out = [];
    for (const f of rowArr) {
      const last = out[out.length - 1];
      const mergeable = last && last.cls === f.cls && last.end === f.start &&
        (!last.tok && !f.tok || last.tok === f.tok);
      if (mergeable) last.end = f.end, last.text += f.text;
      else out.push({ ...f });
    }
    return out;
  };

  rows.forEach((rowArr, li) => {
    const tr = document.createElement('tr');
    tr.className = 'code-line';
    const ln = document.createElement('td');
    ln.className = 'ln-col'; ln.textContent = li + 1;
    const cell = document.createElement('td'); cell.className = 'code-cell';
    const diagsHere = showDiag ? (diagByLine.get(li) || []) : [];
    for (const sp of frag(rowArr)) {
      const span = document.createElement('span');
      span.className = sp.cls || '';
      span.textContent = sp.text;
      cell.appendChild(span);
      if (sp.tok) attachTokenHover(span, sp.tok);
      const hits = diagsHere.filter(d => sp.end > d._start && sp.start < d._end);
      if (hits.length) {
        span.classList.add('diag-u');
        attachDiagHover(span, hits);
      }
    }
    if (diagsHere.length) tr.classList.add('has-diag', 'diag-solo');
    tr.appendChild(ln);
    tr.appendChild(cell);
    tbody.appendChild(tr);
});
}

function attachTokenHover(spanEl, tok) {
  spanEl.dataset.tokIdx = tok._idx ?? '';
}

/** Hover tooltip for diagnostic-underlined tokens: one line per diag message. */
function attachDiagHover(spanEl, diags) {
  spanEl.addEventListener('mouseenter', () => {
    let tip = $('.diag-hover-tip');
    if (!tip) {
      tip = document.createElement('div');
      tip.className = 'diag-hover-tip';
      document.body.appendChild(tip);
    }
    tip.textContent = '';
    for (const d of diags) {
      const row = document.createElement('div');
      row.textContent = d.msg;
      tip.appendChild(row);
    }
    spanEl.classList.add('hot');
    tip.classList.add('show');
    const r = spanEl.getBoundingClientRect();
    const w = tip.offsetWidth, h = tip.offsetHeight;
    tip.style.left = Math.min(Math.max(r.left, 8), window.innerWidth - w - 8) + 'px';
    tip.style.top = (r.bottom + h + 8 > window.innerHeight - 8
      ? r.top - h - 8 : r.bottom + 8) + 'px';
  });
  spanEl.addEventListener('mouseleave', () => {
    spanEl.classList.remove('hot');
    $('.diag-hover-tip')?.classList.remove('show');
  });
}

/* ============================================================
   TOKENS phase — table + hover sync to source highlight
   ============================================================ */
function renderTokens(art) {
  const tbody = $('#token-table tbody');
  tbody.textContent = '';
  const toks = art?.tokens;
  $('#tok-count').textContent = Array.isArray(toks) ? `${toks.length} tokens` : '';

  if (!Array.isArray(toks)) {
    const tr = document.createElement('tr');
    const td = document.createElement('td');
    td.colSpan = 4; td.className = 'empty';
    td.textContent = '// no token stream in this artifact';
    tr.appendChild(td); tbody.appendChild(tr);
    return;
  }

  toks.forEach((tk, i) => {
    const tr = document.createElement('tr');
    tr.dataset.tokIdx = String(i);

    const idx = document.createElement('td');
    idx.className = 'tt-idx'; idx.textContent = i;
    const kind = document.createElement('td');
    kind.className = 'tt-kind';
    const pill = document.createElement('b');
    pill.className = `kw-pill ${tokenClass(tk)}`;
    pill.textContent = tk.kind;
    kind.appendChild(pill);
    const txt = document.createElement('td');
    txt.className = 'tt-text'; txt.textContent = tk.kind === 'Eof' ? 'EOF' : (tk.text === '' ? 'empty' : tk.text);
    const spanTd = document.createElement('td');
    spanTd.className = 'tt-span'; spanTd.textContent = `[${tk.start}–${tk.end})`;

    tr.append(idx, kind, txt, spanTd);
    tr.addEventListener('mouseenter', () => hoverToken(i, true));
    tr.addEventListener('mouseleave', () => hoverToken(i, false));
    tbody.appendChild(tr);
  });
}

/** Highlight the source occurrence of token #idx (and scroll to it once). */
let lastHoveredTok = null;
function hoverToken(idx, on) {
  const span = $(`#source-view span[data-tok-idx="${idx}"]`);
  if (!span) return;
  span.classList.toggle('hl-tok', on);
  if (on) {
    if (lastHoveredTok && lastHoveredTok !== span) lastHoveredTok.classList.remove('hl-tok');
    lastHoveredTok = span;
  } else if (lastHoveredTok === span) {
    lastHoveredTok = null;
  }
}

/* ============================================================
   AST phase — adapter: serde-tagged JSON -> {name, children[]}
   ============================================================ */

const BIN_SYMBOLS = { Add:'+','-':'-',Sub:'-',Mul:'*',Div:'/',Rem:'%',Lt:'<',Gt:'>',Le:'<=',Ge:'>=',Eq:'==',Ne:'!=',AndAnd:'&&',OrOr:'||',And:'&&',Or:'||',Assign:'=',Not:'!',Neg:'-' };
const binSym = k => BIN_SYMBOLS[k] ?? k;

/** Ident appears as `{name:"i", span:{...}}`; be liberal about shapes. */
function identName(x) {
  if (x == null) return '?';
  if (typeof x === 'string') return x;
  if (typeof x.name === 'string') return x.name;
  if (x.name && typeof x.name === 'object') return x.name.name ?? '?';
  return '?';
}

function labelType(ty) {
  if (typeof ty === 'string') return ty === 'Unit' ? '()' : ty.toLowerCase();
  if (!ty || typeof ty !== 'object') return '?';
  const keys = Object.keys(ty);
  if (keys.length === 1) {
    const k = keys[0], v = ty[k];
    if (k === 'Scalar') return String(v).toLowerCase();
    if (k === 'Array') return '[' + String(v).toLowerCase() + ']';
    if (k === 'Unit') return '()';
    if (typeof v === 'string') return k.toLowerCase();
  }
  return keys.join('/');
}

/** Build a server-shaped display hierarchy from the raw serde AST JSON. */
function astToHierarchy(astJson) {
  const mk = (kind, detail, children = [], extraCls = '') => ({ kind, detail, children, extraCls });

  function stmtNode(stmt) {
    if (!stmt || typeof stmt !== 'object') return mk('Stmt', '');
    if (stmt.Let) return mk('Let', identName(stmt.Let.name), [
      ...(stmt.Let.ty ? [mk('Ty', labelType(stmt.Let.ty), [], )] : []),
      exprNode(stmt.Let.init),
    ]);
    if (stmt.Assign) return mk('Assign', '', [lvalueNode(stmt.Assign.target), exprNode(stmt.Assign.value)]);
    if (stmt.If) {
      const kids = [exprNode(stmt.If.cond), blockNode(stmt.If.then_blk)];
      if (stmt.If.else_part) {
        const ep = stmt.If.else_part.If ? stmt.If.else_part.If : stmt.If.else_part.Block;
        kids.push(ep ? (ep.stmts ? blockNode(ep) : stmtNode(ep)) : mk('Else', '?'));
      }
      return mk('If', '', kids);
    }
    if (stmt.For) {
      return mk('For', identName(stmt.For.iv), [
        exprNode(stmt.For.start), exprNode(stmt.For.end), blockNode(stmt.For.body),
      ]);
    }
    if (stmt.Return) return mk('Return', '', stmt.Return.value ? [exprNode(stmt.Return.value)] : []);
    if (stmt.Expr) return mk('ExprStmt', '', [exprNode(stmt.Expr)]);
    if (stmt.Block) return blockNode(stmt.Block);
    if (stmt.Empty !== undefined) return mk('Empty', '');
    const key = Object.keys(stmt)[0];
    return mk(key ?? 'Stmt', '', childNodes(Object.values(stmt)[0]));
  }

  function lvalueNode(lv) {
    if (!lv) return mk('LVal', '');
    const kids = lv.index ? [exprNode(lv.index)] : [];
    return mk(lv.index ? 'Index' : 'Var', identName(lv.base), kids);
  }

  function exprNode(e) {
    if (!e || typeof e !== 'object') return mk('Expr', '');
    if (e.IntLit !== undefined) return mk(String(e.IntLit[0]), '', [], 'lit-node');
    if (e.FloatLit !== undefined) return mk(String(e.FloatLit[0]), '', [], 'lit-node');
    if (e.Bool !== undefined) return mk(String(e.Bool[0]), '', [], 'lit-node');
    if (e.Var) return mk(identName(e.Var), '', [], 'type-node');
    if (e.Unary) return mk('Un(' + binSym(e.Unary[0]) + ')', '', [exprNode(e.Unary[1])]);
    if (e.Bin) return mk('Bin(' + binSym(e.Bin[0]) + ')', '', [exprNode(e.Bin[1]), exprNode(e.Bin[2])], 'op-node');
    if (e.Index) return mk('Index', identName(e.Index[0]), [exprNode(e.Index[1])]);
    if (e.Call) {
      const c = identName(e.Call.callee);
      const kids = (e.Call.args || []).map(exprNode);
      return mk(c + '()', '', kids);
    }
    if (e.Cast) return mk('Cast', labelType(e.Cast[1]), [exprNode(e.Cast[0])], 'op-node');
    const key = Object.keys(e)[0];
    return mk(key ?? 'Expr', '', childNodes(Object.values(e)[0]));
  }

  function childNodes(v) {
    if (Array.isArray(v)) return v.map(x => (x && typeof x === 'object' ? genericNode(x) : mk(String(x), '')));
    if (v && typeof v === 'object') return Object.entries(v)
      .filter(([, x]) => x !== null)
      .map(([k, x]) => (typeof x === 'object' ? genericNode({ [k]: x }) : mk(k, String(x))));
    return [];
  }

  function genericNode(obj) {
    if (Array.isArray(obj)) return mk('[…]', '', obj.map(genericNode));
    const key = Object.keys(obj)[0];
    const val = obj[key];
    if (key === 'stmts' && Array.isArray(val)) return blockNode(val);
    if (val && typeof val === 'object') {
      if (val.stmts) return blockNode(val.stmts);
      return mk(key, '', childNodes(val));
    }
    return mk(key, val === null ? '' : String(val));
  }

  function blockNode(bOrStmts) {
    const stmts = Array.isArray(bOrStmts) ? bOrStmts : (bOrStmts?.stmts ?? []);
    return mk('Block', String(stmts.length), stmts.map(stmtNode));
  }

  function fnDefNode(f) {
    const kids = [];
    for (const p of f.params ?? []) {
      kids.push(mk('Param', `${p.name?.name}:${labelType(p.ty)}`));
    }
    if (f.ret) kids.push(mk('Ret', labelType(f.ret), [], 'type-node'));
    kids.push(blockNode(f.body));
    return mk('FnDef', identName(f.name), kids);
  }

  let root;
  const top = unwrap(astJson);
  if (top.items !== undefined || top.Program !== undefined) {
    const prog = top.items !== undefined ? top : (top.Program ?? {});
    const items = (prog.items ?? []).map(it =>
      it.Fn ? fnDefNode(it.Fn)
      : it.Const ? mk('ConstDef', `${identName(it.Const.name)}: ${labelType(it.Const.ty)} = ${literalStr(it.Const.value)}`)
      : genericNode(it));
    root = mk('Program', String(items.length), items);
  } else if (top.name && top.body) {
    root = fnDefNode(top);
  } else {
    root = genericNode(top);
  }
  return root;
}

function unwrap(o) {
  // Peel serde externally-tagged single-key wrappers like {Program:{...}} —
  // but only when the inner value is NOT the payload itself (e.g. an object
  // that already looks like a Program has .items).
  let cur = o;
  for (let i = 0; i < 3 && cur && typeof cur === 'object' && Object.keys(cur).length === 1; i++) {
    const [k, v] = Object.entries(cur)[0];
    if (v && typeof v === 'object' && !(k === 'Program' && v.items !== undefined)) cur = v; else break;
  }
  return cur ?? {};
}

function literalStr(l) {
  if (l == null) return '';
  if (l.Int !== undefined) return String(l.Int);
  if (l.Float !== undefined) return String(l.Float);
  if (l.Bool !== undefined) return String(l.Bool);
  return JSON.stringify(l);
}

/* ---- AST d3 tree ---- */
function renderAST(art) {
  const host = $('#ast-host');
  host.textContent = '';
  $('#ast-sub').textContent = '';

  if (!art?.ast || typeof art.ast !== 'object') {
    host.innerHTML = `<div class="empty">// no AST in this artifact</div>`;
    return;
  }

  const hier = astToHierarchy(art.ast);

  const root = d3.hierarchy(hier, d => (d.children && d.children.length ? d.children : null));
  const layout = d3.tree().nodeSize([26, 190]);
  layout(root);

  const all = root.descendants();
  const x0 = d3.min(all, d => d.x) - 40, x1 = d3.max(all, d => d.x) + 40;
  const y0 = d3.min(all, d => d.y) - 60, y1 = d3.max(all, d => d.y) + 160;
  const W = y1 - y0, H = x1 - x0;

  const svg = d3.create('svg:svg').attr('viewBox', `0 0 ${Math.max(W, 320)} ${Math.max(H, 240)}`);
  svg.attr('preserveAspectRatio', 'xMidYMid meet');
  const g = svg.append('g');

  g.selectAll('path.ast-link')
    .data(root.links()).join('path')
    .attr('class', 'ast-link')
    .attr('d', d => {
      const sx = d.source.y, sy = d.source.x, tx = d.target.y, ty = d.target.x;
      const mx = (sx + tx) / 2;
      return `M${sx},${sy}C${mx},${sy} ${mx},${ty} ${tx},${ty}`;
    });

  const node = g.selectAll('g.ast-node')
    .data(all).join('g')
    .attr('class', d => 'ast-node' +
      (String(d.data.kind).startsWith('Bin(') || String(d.data.kind).startsWith('Un(') ? ' op-node' : '') +
      (d.data.extraCls ? ' ' + d.data.extraCls : ''))
    .attr('transform', d => `translate(${d.y},${d.x})`);

  node.append('circle').attr('r', 5.5);
  node.append('text')
    .attr('x', 11).attr('dy', '-0.15em')
    .attr('class', 'ast-kind')
    .text(d => String(d.data.kind).slice(0, 22));
  node.filter(d => d.data.detail)
    .append('text').attr('x', 11).attr('dy', '1.25em')
    .attr('class', 'ast-detail')
    .text(d => String(d.data.detail).slice(0, 24));

  $('#ast-sub').textContent = `${all.length} nodes`;
  host.appendChild(svg.node());

  const zoom = d3.zoom().scaleExtent([0.25, 3])
    .on('zoom', ev => g.attr('transform', ev.transform));
  d3.select(svg.node()).call(zoom);
  const fitTransform = () => {
    const bw = host.clientWidth || 800, bh = host.clientHeight || 500;
    const k = Math.min(Math.min(bw / Math.max(W, 320), bh / Math.max(H, 240)), 1.6);
    d3.select(svg.node()).call(zoom.transform,
      d3.zoomIdentity.translate((bw - W * k) / 2, (bh - H * k) / 2 + 8).scale(k));
  };
  fitTransform();
  svg.on('dblclick.zoom', null);
  svg.node().addEventListener('dblclick', () => fitTransform());
  astZoomFit = fitTransform;
  astZoomBehavior = zoom;
  astSvgSel = d3.select(svg.node());
  astG = g;
}
let astZoomFit = null, astZoomBehavior = null, astSvgSel = null, astG = null;

/* ============================================================
   CFG phase — pure SVG painter over precomputed coordinates
   ============================================================ */
const ROLE_TITLES = { entry: 'entry', exit: 'exit', loop_header: 'loop header', join: 'join', straight: '' };
let nodeByIdGlobal = new Map(); // id -> cfg node of the function on screen (edge spotlighting)
const LOOP_OF_ART = () => state.artifact?.loops ?? [];

function renderCfgTabs(art) {
  const wrap = $('#cfg-fn-tabs');
  wrap.textContent = '';
  const fns = art?.cfg?.functions ?? [];
  fns.forEach((fn, i) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'fntab' + (i === state.cfgFn ? ' active' : '');
    b.textContent = fn.name ?? ('fn' + i);
    b.addEventListener('click', () => { state.cfgFn = i; renderCfg(art); syncTabs('#cfg-fn-tabs', i); });
    wrap.appendChild(b);
  });
  wrap.classList.toggle('hidden', fns.length < 2);
}
function syncTabs(wrapSel, active) {
  $$(wrapSel + ' .fntab').forEach((b, i) => b.classList.toggle('active', i === active));
}

function edgePath(points) {
  if (!points || points.length < 2) return '';
  if (points.length === 2) {
    const [[x1, y1], [x2, y2]] = points;
    const dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy) || 1;
    // slight orthogonal elbow for non-axis-aligned straight hops
    if (Math.abs(dx) > 4 && Math.abs(dy) > 4) return `M${x1},${y1} L${x1 + dx * .55},${y1} L${x2},${y2}`;
    return `M${x1},${y1} L${x2},${y2}`;
  }
  if (points.length === 3) {
    const [[x1, y1], [cx, cy], [x2, y2]] = points;
    return `M${x1},${y1} Q${cx},${cy} ${x2},${y2}`;
  }
  // polyline fallback
  return 'M' + points.map(p => p.join(',')).join(' L');
}

function renderCfg(art) {
  const edgesL = $('#cfg-edges-layer'), nodesL = $('#cfg-nodes-layer');
  const emptyBox = $('#cfg-empty');
  nodesL.textContent = ''; edgesL.textContent = '';
  emptyBox.classList.add('hidden');
  emptyBox.textContent = '';

  const fns = art?.cfg?.functions;
  if (!Array.isArray(fns) || !fns.length) {
    emptyBox.textContent = '// no control-flow graph in this artifact (compilation stopped before CFG construction)';
    emptyBox.classList.remove('hidden');
    return;
  }
  state.cfgFn = Math.min(state.cfgFn, fns.length - 1);
  const fn = fns[state.cfgFn];
  nodeByIdGlobal = new Map((fn.nodes ?? []).map(n => [n.id, n]));

  // ---- edges first (under nodes) ----
  for (const e of fn.edges ?? []) {
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', edgePath(e.points));
    path.setAttribute('class', `cfg-edge edge-${e.kind}`);
    path.setAttribute('marker-end', e.kind === 'backedge' ? 'url(#arrow-be)'
      : e.kind === 'branch' ? 'url(#arrow-br)' : 'url(#arrow-ft)');
    path._from = e.from; path._to = e.to;
    if (e.label) {
      const pts = e.points ?? [];
      const a = pts[pts.length - 2] ?? pts[0] ?? [0, 0];
      const b = pts[pts.length - 1] ?? [0, 0];
      const lx = (a[0] + b[0]) / 2, ly = (a[1] + b[1]) / 2;
      const t = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      t.setAttribute('x', lx); t.setAttribute('y', ly - 5);
      t.setAttribute('text-anchor', 'middle');
      t.setAttribute('class', 'cfg-edge-label');
      t.textContent = e.label;
      edgesL.appendChild(t);
    }
    edgesL.appendChild(path);
  }

  // ---- nodes ----
  for (const n of fn.nodes ?? []) {
    const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    g.setAttribute('class', 'cfg-bb ' + (n.role || 'straight'));
    g.dataset.bb = n.id;
    g.dataset.loopId = n.loop_id != null ? String(n.loop_id) : '';
    g.setAttribute('transform', `translate(${n.x},${n.y})`);

    const box = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    box.setAttribute('class', 'bb-box');
    box.setAttribute('width', n.w); box.setAttribute('height', n.h);
    g.appendChild(box);

    const idTxt = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    idTxt.setAttribute('class', 'bb-id');
    idTxt.setAttribute('x', 12); idTxt.setAttribute('y', 20);
    idTxt.textContent = n.id;
    g.appendChild(idTxt);

    const roleTxt = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    roleTxt.setAttribute('class', 'bb-role role-' + (n.role || ''));
    roleTxt.setAttribute('x', n.w - 12);
    roleTxt.setAttribute('y', 20);
    roleTxt.setAttribute('text-anchor', 'end');
    roleTxt.textContent = ROLE_TITLES[n.role] || (n.role || '');
    g.appendChild(roleTxt);

    (n.lines ?? []).slice(0, 4).forEach((ln, i) => {
      const t = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      t.setAttribute('class', 'bb-code');
      t.setAttribute('x', 12);
      t.setAttribute('y', 38 + i * 13);
      t.textContent = ln.length > 30 ? ln.slice(0, 29) + '…' : ln;
      g.appendChild(t);
    });

    // loop-header hover: spotlight its loop body blocks
    if (n.role === 'loop_header' && n.loop_id != null) {
      g.addEventListener('mouseenter', () => spotlightLoop(fn, n.loop_id, true));
      g.addEventListener('mouseleave', () => spotlightLoop(fn, n.loop_id, false));
      g.style.cursor = 'default';
    }

    // tooltip with full lines + loop membership
    const tipLines = [`<b>${esc(n.id)}</b> · ${esc(n.role)}`];
    if (n.loop_id != null) tipLines.push(`loop #${n.loop_id}`);
    tipLines.push(...(n.lines ?? []).map(esc));
    g.addEventListener('mousemove', ev => showGraphTip(ev, tipLines.join('\n')));
    g.addEventListener('mouseleave', hideGraphTip);

    nodesL.appendChild(g);
  }

  // Include routed/back edges as well as node rectangles, with room for arrows.
  const points = (fn.nodes ?? []).flatMap(n => [[n.x, n.y], [n.x + n.w, n.y + n.h]])
    .concat((fn.edges ?? []).flatMap(e => e.points ?? []));
  const xs = points.map(p => p[0]), ys = points.map(p => p[1]);
  const minX = points.length ? Math.min(...xs) : 0, minY = points.length ? Math.min(...ys) : 0;
  const maxX = points.length ? Math.max(...xs) : 0, maxY = points.length ? Math.max(...ys) : 0;
  $('#cfg-svg').setAttribute('viewBox', `${minX - 24} ${minY - 24} ${maxX - minX + 48} ${maxY - minY + 48}`);
}

function spotlightLoop(fn, loopId, on) {
  const loop = (state.artifact?.loops ?? []).find(L => String(L.id) === String(loopId));
  const blocks = new Set((loop?.blocks ?? []).map(String));

  $$('#cfg-host .cfg-bb').forEach(g => {
    const mine = blocks.has(g.dataset.bb);
    g.classList.toggle('loop-hot', on && mine);
    g.classList.toggle('dimmed', on && !mine);
  });
  $$('#cfg-host .cfg-edge').forEach(pth => {
    if (!on) { pth.classList.remove('dimmed'); return; }
    const from = nodeByIdGlobal.get(pth._from), to = nodeByIdGlobal.get(pth._to);
    const inside = !!from && !!to && blocks.has(from.id)
      && (to.loop_id == null || Number(to.loop_id) === Number(loopId))
      && from.loop_id != null && Number(from.loop_id) === Number(loopId);
    pth.classList.toggle('dimmed', !inside);
  });
}

/* CFG tooltip */
function showGraphTip(ev, html) {
  let tip = $('.cfg-hover-tip');
  if (!tip) {
    tip = document.createElement('div');
    tip.className = 'cfg-hover-tip';
    document.querySelector('[data-phase="cfg"] .panel-viewport').appendChild(tip);
    window.addEventListener('resize', hideGraphTip);
  }
  tip.innerHTML = html;
  tip.classList.remove('hidden');
  const panelRect = (tip.parentElement ?? document.body).getBoundingClientRect();
  let x = ev.clientX - panelRect.left + 16, y = ev.clientY - panelRect.top + 14;
  const r = tip.getBoundingClientRect();
  if (x + r.width > panelRect.width - 8) x = ev.clientX - panelRect.left - r.width - 12;
  if (y + r.height > panelRect.height - 8) y = ev.clientY - panelRect.top - r.height - 10;
  tip.style.left = x + 'px'; tip.style.top = y + 'px';
}
function hideGraphTip() {
  const tip = $('.cfg-hover-tip');
  if (tip) tip.classList.add('hidden');
}

/* ============================================================
   SSA phase — IR text blocks with phi highlighting
   ============================================================ */
function renderSSA(art, sideBySide = $('#ssa-side-by-side').checked) {
  const body = $('#ssa-body');
  const pre = $('#ir-pre'), post = $('#ir-ssa');
  const empty = $('#ssa-empty');
  const hasSSA = !!art?.ir_ssa?.functions?.length;
  const hasPre = !!art?.ir_pre_ssa?.functions?.length;
  body.classList.remove('hidden');
  empty.classList.add('hidden');

  if (!hasPre && !hasSSA) {
    body.classList.add('hidden');
    empty.textContent = '// no IR in this artifact (compilation stopped before lowering)';
    empty.classList.remove('hidden');
    $('#ssa-fn-name').textContent = '';
    $('#ir-pre').textContent = ''; $('#ir-ssa').textContent = '';
    return;
  }

  const fns = (hasSSA ? art.ir_ssa.functions : art.ir_pre_ssa.functions);
  state.ssaFn = Math.min(state.ssaFn, fns.length - 1);
  const idx = state.ssaFn;
  $('#ssa-fn-name').textContent = fns[idx]?.name ? `· ${fns[idx].name}()` : '';

  const tabs = $('#ssa-fn-tabs');
  tabs.textContent = '';
  tabs.classList.toggle('hidden', fns.length < 2);
  fns.forEach((f, i) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'fntab' + (i === idx ? ' active' : '');
    b.textContent = f.name ?? ('fn' + i);
    b.addEventListener('click', () => { state.ssaFn = i; renderSSA(art); });
    tabs.appendChild(b);
  });

  const pick = (set) => {
    const list = set?.functions ?? [];
    return list[idx]?.text ?? (list[idx] ?? null);
  };
  const preText = typeof (art.ir_pre_ssa?.functions?.[idx]) === 'string'
    ? art.ir_pre_ssa.functions[idx]
    : (art.ir_pre_ssa?.functions?.[idx]?.text ?? '');
  const ssaText = typeof (art.ir_ssa?.functions?.[idx]) === 'string'
    ? art.ir_ssa.functions[idx]
    : (art.ir_ssa?.functions?.[idx]?.text ?? '');

  body.classList.toggle('side-by-side', sideBySide && !!preText && !!ssaText);

  renderIR(pre, preText, false);
  renderIR(post, ssaText, true);
}

/** Fill an IR <pre>; highlights phi lines and basic-block labels. */
/** Fill an IR <pre>; highlights phi lines (and keeps comments dim). */
function renderIR(preEl, text, isSSA) {
  preEl.textContent = '';
  if (!text) {
    const em = document.createElement('span');
    em.className = 'dep-none';
    em.textContent = '// not available';
    preEl.appendChild(em);
    return;
  }
  const phiRe = /\bphi\b/g;
  for (const raw of String(text).split(String.fromCharCode(10))) {
    const line = document.createElement('span');
    line.className = 'ir-line' + (isSSA && phiRe.test(raw) ? ' is-phi' : '');
    phiRe.lastIndex = 0;
    let last = 0, m;
    while ((m = phiRe.exec(raw)) !== null) {
      if (m.index > last) line.appendChild(document.createTextNode(raw.slice(last, m.index)));
      const w = document.createElement('span');
      w.className = 'phi-word';
      w.textContent = 'phi';
      line.appendChild(w);
      last = m.index + 3;
    }
    if (last < raw.length) line.appendChild(document.createTextNode(raw.slice(last)));
    preEl.appendChild(line);
  }
}

/* ============================================================
   OPT phase — accordion of passes
   ============================================================ */
function renderPasses(art) {
  const list = $('#pass-list');
  const empty = $('#opt-empty');
  list.textContent = '';
  empty.classList.add('hidden');
  const passes = art?.passes;

  if (!Array.isArray(passes) || !passes.length) {
    empty.textContent = '// no optimization passes recorded in this artifact';
    empty.classList.remove('hidden');
    $('#opt-summary').textContent = '';
    return;
  }

  const changed = passes.filter(p => p.changed).length;
  const b = passes.at(-1)?.diff_stats;
  $('#opt-summary').textContent =
    `${passes.length} passes · ${changed} changed` +
    (b ? ` · instructions ${b.insts_before} → ${b.insts_after}` : '');

  passes.forEach((p, i) => {
    const card = document.createElement('div');
    card.className = 'pass-card' + (p.changed ? ' changed open-first' : '');
    if (i === 0) card.classList.add('open');

    const head = document.createElement('button');
    head.type = 'button';
    head.className = 'pass-head';
    head.setAttribute('aria-expanded', String(i === 0));

    const caret = document.createElement('span');
    caret.className = 'pass-caret'; caret.textContent = '▶';
    const badge = document.createElement('span');
    badge.className = 'pass-badge ' + (p.changed ? 'changed' : 'unchanged');
    badge.textContent = p.changed ? 'CHANGED' : 'NO-OP';
    const name = document.createElement('span');
    name.className = 'pass-name'; name.textContent = p.name ?? 'pass';

    const delta = (p.diff_stats ? p.diff_stats.insts_before - p.diff_stats.insts_after : 0);
    const stat = document.createElement('span');
    stat.className = 'pass-stat';
    if (p.diff_stats) {
      stat.textContent = `${p.diff_stats.insts_before} → ${p.diff_stats.insts_after} insts`;
      const d = document.createElement('span');
      d.className = 'pass-delta ' + (delta > 0 ? 'neg' : 'zero');
      d.textContent = delta > 0 ? `−${delta}` : delta < 0 ? `+${-delta}` : '±0';
      stat.prepend(d);
      stat.insertBefore(document.createTextNode('  '), d.nextSibling);
    }

    head.append(caret, badge, name, stat);
    head.addEventListener('click', () => {
      head.setAttribute('aria-expanded', String(card.classList.toggle('open')));
    });

    const body = document.createElement('div');
    body.className = 'pass-body';
    const pr = document.createElement('pre');
    pr.textContent = p.after ?? '// pass output not recorded';
    body.appendChild(pr);

    card.append(head, body);
    list.appendChild(card);
  });
}

/* ============================================================
   LOOP ANALYSIS phase — the star of the demo
   ============================================================ */
const REDUCTION_GLYPHS = { '+': 'Σ+', '*': 'Π×', '-': 'Σ−', max: '⬆', min: '⬇', '||': '∨', '&&': '∧' };
const VERDICT_META = {
  SAFE:       { glyph: '✓', label: 'PARALLELIZED' },
  REDUCTION:  { glyph: null, label: 'REDUCTION' },
  SEQUENTIAL: { glyph: '!', label: 'SEQUENTIAL' },
};

function renderLoops(art) {
  const list = $('#loop-list');
  const empty = $('#loops-empty');
  list.textContent = '';
  empty.classList.add('hidden');
  const loops = art?.loops;

  if (!Array.isArray(loops) || !loops.length) {
    empty.textContent = '// no loops analyzed in this artifact';
    empty.classList.remove('hidden');
    return;
  }

  loops.forEach(L => {
    const verdict = String(L.verdict ?? 'SEQUENTIAL').toUpperCase();
    const meta = VERDICT_META[verdict] ?? VERDICT_META.SEQUENTIAL;
    const rejected = verdict === 'SEQUENTIAL';

    const card = document.createElement('article');
    card.className = 'loop-card ' + verdict;

    /* --- header --- */
    const head = document.createElement('div');
    head.className = 'loop-head';
    const name = document.createElement('span');
    name.className = 'loop-name'; name.textContent = `Loop #${L.id ?? '?'}`;
    const depth = document.createElement('span');
    depth.className = 'loop-depth';
    depth.textContent = `depth ${L.depth ?? '?'}`;

    const badge = document.createElement('span');
    badge.className = 'verdict verdict-' + verdict;
    const glyph = document.createElement('span');
    glyph.className = 'v-glyph';
    glyph.textContent = meta.glyph ?? (L.reduction ? (REDUCTION_GLYPHS[L.reduction.op] ?? 'Σ') : 'Σ');
    badge.title = L.reason ?? '';
    const blabel = document.createElement('span');
    blabel.textContent = meta.label + (verdict === 'SAFE' && L.plan?.threads ? ` × ${L.plan.threads} THREADS` : '');
    badge.append(glyph, blabel);
    head.append(name, depth, badge);
    card.appendChild(head);

    /* --- reason strip --- */
    const why = document.createElement('p');
    why.className = 'loop-reason';
    const whyKey = document.createElement('span');
    whyKey.className = 'why';
    whyKey.textContent = rejected ? 'WHY REJECTED' : (verdict === 'REDUCTION' ? 'HOW IT RUNS' : 'VERDICT');
    const whyTxt = document.createElement('span');
    whyTxt.textContent = L.reason ?? '—';
    why.append(whyKey, whyTxt);
    card.appendChild(why);

    /* --- body --- */
    const body = document.createElement('div');
    body.className = 'loop-body';

    // iv + bounds
    const ivRow = document.createElement('div');
    ivRow.className = 'loop-row';
    ivRow.innerHTML =
      `<span class="lr-key">Induction</span>` +
      `<span class="lr-val"><span class="mono">${esc(L.iv ?? '?')}</span> ∈ ` +
      `<span class="mono">[${esc(L.bounds?.start ?? '?')} .. ${esc(L.bounds?.end ?? '?')})</span>` +
      `<span class="mono" style="color:var(--dim)">   header ${esc(L.header ?? '-')}</span></span>`;
    body.appendChild(ivRow);

    // accesses
    const accRow = document.createElement('div');
    accRow.className = 'loop-row';
    const accKey = document.createElement('span');
    accKey.className = 'lr-key'; accKey.textContent = 'Accesses';
    const accVal = document.createElement('span');
    accVal.className = 'lr-val';
    for (const a of L.accesses ?? []) {
      const m = /^(READ|WRITE)\s+(.*)$/.exec(a);
      const row = document.createElement('div');
      row.className = 'acc-row';
      const tag = document.createElement('span');
      tag.className = 'rw-tag ' + (m ? m[1] : 'READ');
      tag.textContent = m ? m[1] : 'READ';
      const sub = document.createElement('span');
      sub.className = 'acc-sub mono';
      sub.textContent = m ? m[2] : a;
      sub.title = `memory reference inside loop #${L.id}`;
      row.append(tag, sub);
      accVal.appendChild(row);
    }
    if (!(L.accesses ?? []).length) accVal.innerHTML = '<span class="dep-none">none recorded</span>';
    accRow.append(accKey, accVal);
    body.appendChild(accRow);

    // dependence lists RAW/WAR/WAW
    const depWrap = document.createElement('div');
    depWrap.className = 'loop-row dep-section';
    const depKey = document.createElement('span');
    depKey.className = 'lr-key'; depKey.textContent = 'Dependences';
    const depVal = document.createElement('span');
    depVal.className = 'lr-val';
    let anyDep = false;
    for (const kind of ['raw', 'war', 'waw']) {
      for (const d of L[kind] ?? []) {
        anyDep = true;
        const row = document.createElement('div');
        row.className = 'dep-row';
        const pill = document.createElement('span');
        pill.className = 'kind-pill ' + kind.toUpperCase();
        pill.textContent = kind.toUpperCase();
        row.appendChild(pill);
        if (d.distance !== undefined && d.distance !== null) {
          const dist = document.createElement('span');
          dist.className = 'dist-pill';
          dist.textContent = `dist ${d.distance}`;
          dist.title = 'dependence distance: iterations that must not overlap';
          row.appendChild(dist);
        }
        const pair = document.createElement('span');
        pair.className = 'pair-mono';
        pair.innerHTML =
          `<span>${esc(d.array ?? '?')}</span>` +
          (d.level != null ? ` <span class="arr">·</span> level ${esc(String(d.level))}` : '');
        pair.title = d.explain ?? '';
        row.appendChild(pair);
        depVal.appendChild(row);
      }
    }
    if (!anyDep) {
      const none = document.createElement('span');
      none.className = 'dep-none';
      none.textContent = 'no loop-carried dependences ✓';
      depVal.appendChild(none);
    }
    depWrap.append(depKey, depVal);
    body.appendChild(depWrap);

    // plan / reduction strip
    if (verdict === 'SAFE' && L.plan?.threads) {
      const plan = document.createElement('span');
      plan.className = 'plan-strip';
      plan.innerHTML = `PLAN&nbsp;&nbsp;→ up to ${esc(String(L.plan.threads))} threads (analysis proposal)`;
      plan.title = 'Proposed by the analysis. The backend may still demote a region it cannot lower, and the runtime keeps small loops sequential — the local Campaign (Benchmark phase) shows what actually ran.';
      body.appendChild(plan);
    }
    if (L.reduction) {
      const red = document.createElement('span');
      red.className = 'red-strip';
      red.innerHTML = `${esc(REDUCTION_GLYPHS[L.reduction.op] ?? 'Σ')} private accumulator per thread for ` +
        `<span class="mono">${esc(L.reduction.var ?? '?')}</span>, combined at loop exit`;
      red.title = 'reduction variable is privatized; partial sums re-associated under fast-math contract';
      body.appendChild(red);
    }

    card.appendChild(body);
    list.appendChild(card);
  });
}

/* ============================================================
   BENCH phase — hand-rolled SVG horizontal bars
   ============================================================ */
const BENCH_COLORS = ['#8b949e', '#61afef', '#00c896', '#56b4e9', '#e69f00'];

/** Paint the variant bars and the efficiency table for a bench-like object.
 *  One painter serves the recorded campaign chart and the local two-engine
 *  chart, so both read the same way: bars sorted slowest at top, value labels
 *  in ms, and "× vs best" chips measured against the fastest bar shown. */
function renderBenchChart(bench, hosts) {
  const chart = hosts.chart;
  const effWrap = hosts.eff;
  chart.textContent = '';
  const variants = [...(bench.variants ?? [])].sort((a, b) => b.median_ms - a.median_ms); // slowest at top
  const fastest = Math.min(...variants.map(v => v.median_ms));
  const W = 960, barH = 34, gap = 18, padL = 170, padR = 150, padT = 26, padB = 44;
  const plotW = W - padL - padR;
  const Hh = padT + variants.length * (barH + gap) - gap + padB;

  const svgNS = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(svgNS, 'svg');
  svg.setAttribute('viewBox', `0 0 ${W} ${Hh}`);

  const defs = document.createElementNS(svgNS, 'defs');
  defs.innerHTML =
    `<linearGradient id="benchGradFast" x1="0" y1="0" x2="1" y2="0">
       <stop offset="0" stop-color="#0b6b54"/><stop offset="1" stop-color="#00c896"/>
     </linearGradient>
     <linearGradient id="benchGradSeq" x1="0" y1="0" x2="1" y2="0">
       <stop offset="0" stop-color="#274a68"/><stop offset="1" stop-color="#61afef"/>
     </linearGradient>`;
  svg.appendChild(defs);

  const useLog = state.benchLog;
  const positive = variants.map(v => v.median_ms).filter(ms => Number.isFinite(ms) && ms > 0);
  const max = Math.max(...positive, 0) || 1;
  const scaleMax = useLog ? max : Math.min(max * 1.06, Number.MAX_VALUE);
  // Log bars start a decade below the smallest positive timing, not at 1 ms.
  const scaleMin = Math.max(Math.min(...positive, max) / 10, Number.MIN_VALUE);
  const logMin = Math.log10(scaleMin);
  const logSpan = Math.log10(scaleMax) - logMin || 1;
  const toX = ms => {
    if (!Number.isFinite(ms) || ms <= 0) return 0;
    const fraction = useLog ? (Math.log10(ms) - logMin) / logSpan : ms / scaleMax;
    return Math.min(1, Math.max(0, fraction)) * plotW;
  };

  // gridlines + ticks
  const ticks = useLog ? logTicks(scaleMin, scaleMax) : linTicks(scaleMax);
  for (const tv of ticks) {
    const x = padL + toX(tv.v);
    const gl = document.createElementNS(svgNS, 'line');
    gl.setAttribute('x1', x); gl.setAttribute('x2', x);
    gl.setAttribute('y1', padT - 6); gl.setAttribute('y2', Hh - padB + 6);
    gl.setAttribute('class', 'axis-line'); gl.setAttribute('stroke-dasharray', '3 5');
    svg.appendChild(gl);
    const tl = document.createElementNS(svgNS, 'text');
    tl.setAttribute('x', x); tl.setAttribute('y', Hh - padB + 22);
    tl.setAttribute('text-anchor', 'middle'); tl.setAttribute('class', 'tick-lbl');
    tl.textContent = tv.label;
    svg.appendChild(tl);
  }
  const unitLbl = document.createElementNS(svgNS, 'text');
  unitLbl.setAttribute('x', padL + plotW / 2); unitLbl.setAttribute('y', Hh - 8);
  unitLbl.setAttribute('text-anchor', 'middle'); unitLbl.setAttribute('class', 'tick-lbl');
  unitLbl.textContent = 'median wall-clock time (ms)' + (useLog ? ' · log scale' : '');
  svg.appendChild(unitLbl);

  variants.forEach((v, i) => {
    const y = padT + i * (barH + gap);
    const isFastest = v.median_ms === fastest;
    const color = BENCH_COLORS[i % BENCH_COLORS.length];

    const lbl = document.createElementNS(svgNS, 'text');
    lbl.setAttribute('x', padL - 14); lbl.setAttribute('y', y + barH / 2 + 4);
    lbl.setAttribute('text-anchor', 'end'); lbl.setAttribute('class', 'var-lbl');
    lbl.textContent = v.name;
    svg.appendChild(lbl);

    const rect = document.createElementNS(svgNS, 'rect');
    rect.setAttribute('class', 'bar-rect' + (isFastest ? ' bar-fastest' : ''));
    if (!isFastest) rect.setAttribute('fill', color);
    rect.setAttribute('y', y); rect.setAttribute('height', barH);
    rect.setAttribute('rx', 5);
    rect.setAttribute('x', padL); rect.setAttribute('width', 0);
    svg.appendChild(rect);
    onNextFrame(() => rect.setAttribute('width', Math.max(toX(v.median_ms), 1.5)));

    const val = document.createElementNS(svgNS, 'text');
    val.setAttribute('x', padL + Math.max(toX(v.median_ms), 1.5) + 10);
    val.setAttribute('y', y + barH / 2 + 4);
    val.setAttribute('class', 'val-lbl');
    val.textContent = formatMilliseconds(v.median_ms) + ' ms';
    svg.appendChild(val);

    {
      // How many times slower than the fastest bar shown in this chart.
      const speedup = fastest > 0 ? v.median_ms / fastest : 0;
      if (speedup > 1.05) {
        const chip = document.createElementNS(svgNS, 'text');
        chip.setAttribute('x', padL + Math.max(toX(v.median_ms), 1.5) + 10);
        chip.setAttribute('y', y + barH / 2 - 8);
        chip.setAttribute('class', 'speedup-chip');
        chip.textContent = `${fmt(speedup, 2)}× vs best`;
        svg.appendChild(chip);
      }
    }

    const title = document.createElementNS(svgNS, 'title');
    title.textContent = `${v.name}\nmedian ${formatMilliseconds(v.median_ms)} ms` +
      (Array.isArray(v.samples) ? `\nsamples: ${v.samples.map(s => fmt(s)).join(', ')}` : '');
    rect.appendChild(title);
  });

  chart.appendChild(svg);

  // efficiency table
  const eff = bench.efficiency ?? [];
  if (eff.length && effWrap) {
    const h = document.createElement('p');
    h.className = 'eff-title'; h.textContent = 'parallel efficiency';
    effWrap.appendChild(h);
    const tbl = document.createElement('table');
    tbl.className = 'efft';
    tbl.innerHTML = `<thead><tr><th>threads</th><th>speedup</th><th style="width:38%">efficiency</th><th>%</th></tr></thead>`;
    const tb = document.createElement('tbody');
    for (const r of eff) {
      const tr = document.createElement('tr');
      const tdT = document.createElement('td'); tdT.textContent = r.threads;
      const tdS = document.createElement('td'); tdS.textContent = fmt(r.speedup, 2) + '×';
      const tdB = document.createElement('td'); tdB.innerHTML = '<div class="effbar"><i></i></div>';
      const tdP = document.createElement('td'); tdP.textContent = fmt(r.efficiency * 100, 1) + '%';
      tr.append(tdT, tdS, tdB, tdP);
      tb.appendChild(tr);
      onNextFrame(() => {
        tdB.querySelector('i').style.width = Math.min(r.efficiency * 100, 100) + '%';
      });
    }
    tbl.appendChild(tb);
    effWrap.appendChild(tbl);
  }
}

function renderBench(art) {
  const chart = $('#bench-chart');
  const effWrap = $('#bench-eff');
  const empty = $('#bench-empty');
  chart.textContent = ''; effWrap.textContent = '';
  $('#bench-comparison').textContent = '';
  empty.className = 'empty bench-empty hidden';
  empty.textContent = '';
  empty.classList.add('hidden');

  const bench = art?.bench;
  const catalog = bench ? benchmarkKernel(bench.kernel) : null;
  if (bench?.kernel === '<adhoc>') {
    const samples = bench.variants?.[0]?.samples?.length ?? 0;
    $('#bench-kernel').textContent = `local WASM interpreter · ${samples} measured samples`;
  } else if (bench && benchmarkProbe(bench.kernel)) {
    const samples = bench.variants?.[0]?.samples?.length ?? 0;
    $('#bench-kernel').textContent = `recorded example probe · ${samples} interpreter samples`;
  } else {
    $('#bench-kernel').textContent = bench
      ? `recorded campaign · kernel ${bench.kernel} · n = ${Number(bench.n).toLocaleString('en-US')}`
      : '';
  }
  renderBenchComparison(bench);
  if (!bench || !Array.isArray(bench.variants) || !bench.variants.length) {
    const title = document.createElement('strong');
    title.className = 'bench-empty-title';
    title.textContent = state.benchPending ? 'MEASUREMENT IN PROGRESS'
      : state.benchMessage ? 'MEASUREMENT UNAVAILABLE' : 'NO RECORDED POINT FOR THIS ARTIFACT';
    const copy = document.createElement('p');
    copy.className = 'bench-empty-copy';
    const isCustom = art?.example === '<adhoc>' || state.fromCustom;
    copy.textContent = state.benchMessage || (isCustom
      ? 'This source compiled through the pipeline, but it has not been measured by the named-kernel campaign.'
      : 'No recorded measurement matches this source. A campaign entry with the same name may use different code. You can still measure this source’s reference-interpreter path locally, without implying native or parallel speedups.');
    empty.append(title, copy);

    if (stageAvailable(art, 'bench')) {
      const measure = document.createElement('button');
      measure.type = 'button';
      measure.id = 'bench-measure';
      measure.className = 'btn bench-measure';
      measure.disabled = state.benchPending;
      // A static build has no server measurement waiting behind the button, so
      // the control measures both local engines instead of only the compiler's
      // reference interpreter. Named examples keep their selection.
      const bothEngines = !!state.static;
      const onNamed = bothEngines && !state.fromCustom &&
        !!state.exampleName && state.exampleName !== '<adhoc>';
      measure.textContent = state.benchPending
        ? (bothEngines ? 'measuring both engines…' : 'measuring in WASM…')
        : (bothEngines ? 'Measure both engines in browser WASM' : 'Measure in browser WASM');
      measure.title = bothEngines
        ? 'Measure the reference interpreter and HELIX’s compiled WebAssembly module in this browser'
        : 'Run seven bounded reference-interpreter samples in this browser';
      measure.addEventListener('click',
        onNamed ? () => measureNamedExample(state.exampleName) : measureCustom);
      empty.appendChild(measure);
    }

    const index = document.createElement('div');
    index.className = 'bench-index';
    const pointCount = Number(state.benchmarkCatalog?.point_count);
    const kernelCount = state.benchmarkCatalog?.kernels?.length ?? 0;
    if (pointCount && kernelCount) {
      const stamp = state.benchmarkCatalog.timestamp_utc
        ? ` · captured ${state.benchmarkCatalog.timestamp_utc.slice(0, 10)}` : '';
      const probeCount = state.benchmarkCatalog.examples?.length ?? 0;
      const probeStamp = state.benchmarkCatalog.probe_provenance?.timestamp_utc?.slice(0, 10) ?? 'date unavailable';
      index.textContent = `Recorded campaign: ${pointCount} measured points across ${kernelCount} kernels${stamp}.` +
        (probeCount ? ` Separate interpreter probes: ${probeCount} · ${probeStamp}.` : '');
      const names = document.createElement('div');
      names.className = 'bench-index-names';
      names.textContent = state.benchmarkCatalog.kernels.map(kernel => kernel.name).join(' · ');
      index.appendChild(names);
    } else if (!state.benchmarkCatalog) {
      index.textContent = state.static
        ? 'No campaign index in this static build; the local measurement below is the evidence available here.'
        : 'Campaign index unavailable — start the Rust observer for recorded evidence.';
    } else {
      index.textContent = 'No campaign index is available on the fixture server.';
    }
    empty.appendChild(index);
    empty.classList.remove('hidden');
    // The empty box already holds the trigger, so the local section reports
    // results only (it appears when a local measurement has been taken).
    renderLocalBench($('#bench-local'), art, bench, false);
    return;
  }

  if (catalog) {
    const sizes = catalog.sizes?.length ?? 0;
    $('#bench-kernel').textContent += sizes > 1 ? ` · ${sizes} sizes recorded` : '';
  }

  renderBenchChart(bench, { chart, eff: effWrap });
  // Recorded evidence stays above; the local pair is appended below it, and in
  // a static build this section also carries the only local trigger.
  renderLocalBench($('#bench-local'), art, bench, true);
}

/** Local (this-browser) measurement of both engines, rendered BELOW the
 *  recorded evidence and never mixed into it. In a static build the section
 *  also hosts the measure control for named examples, whose recorded cards
 *  remain untouched above. */
function renderLocalBench(host, art, bench, withButton) {
  if (!host) return;
  host.textContent = '';
  host.classList.add('hidden');
  const local = state.localBench;
  const hasRecorded = !!bench && Array.isArray(bench.variants) && bench.variants.length > 0;
  // Nothing local to report and no local trigger to offer: keep it out.
  if (!local && !(state.static && hasRecorded)) return;
  host.classList.remove('hidden');

  const heading = document.createElement('p');
  heading.className = 'bench-local-heading';
  heading.textContent = 'Local measurement (this browser)';
  host.appendChild(heading);

  const intro = document.createElement('p');
  intro.className = 'bench-comparison-intro';
  intro.textContent = 'Same source, both engines measured here just now: the interpreter is HELIX’s reference tree-walker compiled to WebAssembly; the compiled bar is the optimized SSA IR emitted as a WebAssembly module and executed in the same browser. No native code and no parallel execution took part in these numbers.';
  host.appendChild(intro);

  if (local) {
    const variants = Array.isArray(local.variants) ? local.variants : [];
    const interp = variants[0] ?? null;
    const compiled = variants[1] ?? null;
    const grid = document.createElement('div');
    grid.className = 'bench-compare-grid';
    const pair = benchmarkPair(interp, compiled);
    appendBenchmarkCard(
      grid,
      'Reference interpreter → compiled wasm',
      pair ?? 'Not comparable',
      interp && compiled
        ? `${benchmarkVariantLabel(interp)} → ${benchmarkVariantLabel(compiled)}`
        : `interpreter ${benchmarkVariantLabel(interp)} · compiled ${benchmarkVariantLabel(compiled)}`,
      pair ? 'bench-compare-positive' : 'bench-compare-muted',
    );
    const memory = local.memory ?? {};
    const interpBytes = formatBytes(memory.interpBytes);
    const compiledBytes = formatBytes(memory.compiledBytes);
    const memoryDetail = 'Interpreter bytes are boxed interpreter cells; compiled bytes are raw array bytes in the module’s linear memory. Both were counted during the same program.';
    let memoryValue = 'Memory not reported';
    let memoryTone = 'bench-compare-muted';
    if (interpBytes && compiledBytes) {
      const before = Number(memory.interpBytes), after = Number(memory.compiledBytes);
      const saved = before > 0 && after < before ? ` (${fmt((1 - after / before) * 100, 0)}% less)` : '';
      memoryValue = `${interpBytes} → ${compiledBytes}${saved}`;
    } else if (interpBytes || compiledBytes) {
      memoryValue = `interpreter ${interpBytes ?? '—'} · compiled ${compiledBytes ?? '—'}`;
    }
    if (memoryValue !== 'Memory not reported') memoryTone = 'bench-compare-positive';
    appendBenchmarkCard(grid, 'Memory per run', memoryValue, memoryDetail, memoryTone);
    host.appendChild(grid);

    const caption = document.createElement('p');
    caption.className = 'bench-local-caption';
    caption.textContent = 'Both local medians · slowest at top · “× vs best” compares this pair only';
    host.appendChild(caption);
    const localChart = document.createElement('div');
    localChart.className = 'bench-chart';
    host.appendChild(localChart);
    renderBenchChart({ kernel: '<adhoc>', variants, efficiency: [] }, { chart: localChart });

    const parity = local.parity ?? {};
    const parityLine = document.createElement('p');
    parityLine.className = 'bench-local-parity' + (parity.same ? '' : ' is-error');
    parityLine.textContent = parity.same
      ? `compiled output identical to the interpreter — checksum ${parity.checksum || 'not recorded for this run'}`
      : `COMPILER BUG SIGNAL — compiled output differs from the interpreter (interpreter checksum ${parity.checksum || 'unknown'} vs compiled ${parity.compiledChecksum || 'unknown'}). The medians above are still the measured values.`;
    host.appendChild(parityLine);
  } else {
    const waiting = document.createElement('p');
    waiting.className = 'bench-recorded-note';
    waiting.textContent = state.benchPending
      ? (state.benchMessage || 'measuring both engines in this browser…')
      : 'No local measurement yet. Measure both engines here to compare the reference interpreter with HELIX’s compiled WebAssembly module; nothing native or parallel is involved.';
    host.appendChild(waiting);
  }

  if (withButton && stageAvailable(art, 'bench')) {
    const measure = document.createElement('button');
    measure.type = 'button';
    measure.id = 'bench-local-measure';
    measure.className = 'btn bench-measure bench-local-measure';
    measure.disabled = state.benchPending;
    measure.textContent = state.benchPending ? 'measuring both engines…' : 'Measure both engines in browser WASM';
    measure.title = 'Measure the reference interpreter and HELIX’s compiled WebAssembly module in this browser';
    measure.addEventListener('click',
      state.fromCustom ? measureCustom : () => measureNamedExample(state.exampleName));
    host.appendChild(measure);
  }
}

/** Human-readable byte counts for the local memory card. */
function formatBytes(bytes) {
  if (bytes == null) return null;
  const value = Number(bytes);
  if (!Number.isFinite(value) || value < 0) return null;
  if (value < 1024) return `${Math.round(value)} B`;
  if (value < 1024 * 1024) return `${fmt(value / 1024, 1)} KB`;
  return `${fmt(value / (1024 * 1024), 2)} MB`;
}

function linTicks(max) {
  const out = [];
  const step = niceStep(max / 5);
  for (let v = 0; v <= max * 1.0001; v += step) out.push({ v, label: tickLabel(v) });
  return out;
}
function logTicks(min, max) {
  const out = [];
  const expMax = Math.floor(Math.log10(max));
  for (let e = Math.ceil(Math.log10(min)); e <= expMax; e++) {
    const v = Math.pow(10, e);
    if (v >= min && v <= max) out.push({ v, label: tickLabel(v) });
  }
  return out;
}
function tickLabel(v) {
  if (v === 0) return '0';
  if (v >= 1000) return (v / 1000) + 'k';
  if (v >= 1) return String(+v.toFixed(2));
  return String(v);
}
function niceStep(raw) {
  const pow = Math.pow(10, Math.floor(Math.log10(Math.max(raw, 1e-9))));
  const norm = raw / pow;
  const nice = norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10;
  return nice * pow;
}

/* ============================================================
   Diagnostics banner + locked stages
   ============================================================ */
function compilerDiagnostics(art) {
  return [...(art?.diags_lex ?? []), ...(art?.diags_sem ?? [])];
}
function compileFailed(art) {
  return compilerDiagnostics(art).length > 0;
}
function stageAvailable(art, phase) {
  if (!art) return false;
  if (compileFailed(art)) return phase === 'source' || (phase === 'tokens' && Array.isArray(art.tokens));
  switch (phase) {
    case 'source': case 'tokens': return Array.isArray(art.tokens);
    case 'ast': return !!art.ast;
    case 'cfg': return Array.isArray(art.cfg?.functions) && art.cfg.functions.length > 0;
    case 'ssa': return !!(art.ir_ssa?.functions?.length || art.ir_pre_ssa?.functions?.length);
    case 'opt': return Array.isArray(art.passes) && art.passes.length > 0;
    case 'loops': return Array.isArray(art.loops) && art.loops.length > 0;
    // A valid artifact can always enter the BENCH screen. Named kernels show
    // recorded campaign evidence; custom/language sources show an explicit
    // unmeasured state instead of making the pipeline look broken.
    case 'bench': return Array.isArray(art.tokens) && !!art.ast && !art.diags_lex;
    default: return false;
  }
}

function renderDiagBanner(art) {
  const ban = $('#diag-banner');
  const diags = compilerDiagnostics(art);
  if (!diags.length) { ban.classList.add('hidden'); ban.textContent = ''; return; }
  ban.classList.remove('hidden');
  ban.innerHTML = '';
  const g = document.createElement('span'); g.className = 'dg-glyph'; g.textContent = '!';
  const msg = document.createElement('span');
  const syntax = art.diags_lex?.length > 0;
  msg.textContent = `${diags.length} ${syntax ? 'syntax' : 'semantic'} error${diags.length > 1 ? 's' : ''}` +
    ` — pipeline stopped during ${syntax ? 'lexing/parsing' : 'semantic analysis'}. ` +
    diags.map(d => d.msg).join('; ');
  const jump = document.createElement('button');
  jump.className = 'btn dg-jump';
  jump.textContent = 'view SOURCE ↓';
  jump.addEventListener('click', () => gotoPhase('source', true));
  ban.append(g, msg, jump);
}

/* ============================================================
   Stepper + phase switching
   ============================================================ */
function buildStepper() {
  const ol = $('#stepper');
  ol.textContent = '';
  PHASES.forEach(([id, label], i) => {
    if (i > 0) {
      const arrow = document.createElement('li');
      arrow.className = 'step-arrow'; arrow.textContent = '›';
      arrow.setAttribute('aria-hidden', 'true');
      ol.appendChild(arrow);
    }
    const li = document.createElement('li');
    li.id = 'step-' + id;
    li.className = 'step';
    li.setAttribute('role', 'tab');
    li.tabIndex = 0;
    li.setAttribute('aria-selected', 'false');
    const num = document.createElement('span');
    num.className = 'num'; num.textContent = String(i + 1);
    li.append(num, document.createTextNode(label));
    li.addEventListener('click', () => gotoPhase(id));
    li.addEventListener('keydown', ev => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); gotoPhase(id); } });
    ol.appendChild(li);
  });
}

function refreshStepper(art) {
  PHASES.forEach(([id]) => {
    const li = $('#step-' + id);
    const ok = stageAvailable(art, id);
    li.classList.toggle('locked', !ok);
    li.setAttribute('aria-disabled', String(!ok));
    li.setAttribute('aria-selected', String(state.phase === id));
    li.querySelector('.lock')?.remove();
    if (!ok) {
      const lock = document.createElement('span');
      lock.className = 'lock'; lock.textContent = ' ⌀';
      lock.title = 'unavailable — earlier pipeline stage failed or absent';
      li.appendChild(lock);
    }
    li.title = ok ? labelFor(id) : `${labelFor(id)} — unavailable for this artifact`;
  });
}
function labelFor(id) { return (PHASES.find(p => p[0] === id) ?? ['', id])[1]; }

function gotoPhase(id, force = false) {
  if (state.artifact && !stageAvailable(state.artifact, id)) {
    if (!force) { toast(`${labelFor(id)} is unavailable — earlier stage failed or data absent`); flashStep(id); return; }
  }
  if (!PHASES.some(p => p[0] === id)) return;
  state.phase = id;
  $$('.phase').forEach(ph => {
    ph.classList.toggle('current', ph.dataset.phase === id);
    // If a previous swap never got a frame (hidden pane / background tab), its
    // transition is stuck at t=0 holding opacity at 0. Jump those to the end.
    ph.getAnimations().forEach(a => { try { a.finish(); } catch { /* not finishable */ } });
  });
  $$('.step').forEach(st => {
    const active = st.id === 'step-' + id;
    st.classList.toggle('active', active);
    st.setAttribute('aria-selected', String(active));
  });
  if (id === 'ast') requestAnimationFrame(() => { astZoomFit?.(); });
}

function flashStep(id) {
  const li = $('#step-' + id);
  if (!li) return;
  li.classList.remove('flash');
  void li.offsetWidth;
  li.classList.add('flash');
}

/* ============================================================
   Data loading
   ============================================================ */
async function fetchJSON(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    // Prefer the API's JSON error body ({ "error": msg }) over a bare status.
    let msg = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body && typeof body.error === 'string') msg = body.error;
    } catch { /* body was not JSON */ }
    throw new Error(msg);
  }
  return res.json();
}

let wasmRunnerPromise;
function loadBrowserRunner() {
  if (!wasmRunnerPromise) {
    wasmRunnerPromise = import('/wasm/helix-browser.js');
  }
  return wasmRunnerPromise;
}

async function hydrateBrowserExecution(art) {
  const runSeq = ++state.browserRunSeq;
  const bridge = await loadBrowserRunner();
  if (!browserPreviewAllowed(art, bridge.estimatedBrowserWork)) {
    state.browserRun = art?.source && !art.diags_lex && !(art.diags_sem?.length)
      ? {
          status: 'skipped',
          message: 'Browser preview skipped: this bundled source exceeds the responsive WASM preview budget. The recorded benchmark below is the authoritative measurement for its larger sizes.',
        }
      : null;
    renderAll();
    return;
  }

  state.browserRun = { status: 'running' };
  renderExecReadout(art);
  try {
    const result = await bridge.runHelixInBrowser(art.source);
    if (runSeq !== state.browserRunSeq || state.artifact !== art) return;
    state.browserRun = { status: 'done', result };
    renderAll();
  } catch (error) {
    if (runSeq !== state.browserRunSeq || state.artifact !== art) return;
    state.browserRun = {
      status: 'error',
      result: {
        ok: false,
        printed: [],
        checksum: null,
        message: `WASM preview unavailable: ${error.message}`,
        engine: 'browser interpreter',
      },
    };
    renderAll();
  }
}

/** Shared responsive-benchmark budget: refuse sources the WASM path cannot
 *  finish inside the worker deadline rather than report a truncated run. */
function guardBrowserWorkBudget(estimate) {
  const work = Number(estimate);
  if (work > 1_000_000) {
    throw new Error(`estimated work ${work.toLocaleString('en-US')} exceeds the responsive WASM benchmark budget; use a smaller custom input`);
  }
}

async function measureBrowserSource(source) {
  const { runHelixInBrowser, estimatedBrowserWork } = await loadBrowserRunner();
  const estimate = estimatedBrowserWork(source);
  guardBrowserWorkBudget(estimate);
  const warmup = await runHelixInBrowser(source);
  if (!warmup.ok) throw new Error(warmup.message || 'WASM interpreter rejected this source');

  // Average worker-reported execution times, never worker startup or transport.
  // Timer precision is browser-dependent; these are reference-only samples.
  const repetitions = estimate < 10_000 ? 32 : estimate < 100_000 ? 8 : 2;
  const samples = [];
  for (let i = 0; i < 7; i += 1) {
    let elapsedMs = 0;
    for (let rep = 0; rep < repetitions; rep += 1) {
      const result = await runHelixInBrowser(source);
      if (!result.ok) throw new Error(result.message || 'WASM interpreter failed during measurement');
      if (result.checksum !== warmup.checksum || JSON.stringify(result.printed) !== JSON.stringify(warmup.printed)) {
        throw new Error('WASM interpreter produced non-deterministic output across samples');
      }
      if (!Number.isFinite(result.elapsedMs) || result.elapsedMs < 0) {
        throw new Error('WASM interpreter did not provide a valid execution time');
      }
      elapsedMs += result.elapsedMs;
    }
    samples.push(elapsedMs / repetitions);
  }
  const ordered = [...samples].sort((a, b) => a - b);
  return {
    result: warmup,
    samples,
    medianMs: ordered[Math.floor(ordered.length / 2)],
  };
}

async function loadExamples() {
  const ul = $('#example-list');
  try {
    const names = await fetchJSON('/api/examples');
    state.examples = Array.isArray(names) ? names : [];
  } catch {
    state.examples = [];
  }
  if (!state.examples.length) {
    // static fallback — mirrors builtin_examples() in crates/helix-observe/src/server.rs
    state.examples = FALLBACK_EXAMPLES;
  }
  ul.textContent = '';
  const seenGroups = new Set();
  const groupedExamples = EXAMPLE_GROUPS.flatMap(group =>
    state.examples.filter(name => exampleGroup(name) === group));
  for (const name of groupedExamples) {
    const group = exampleGroup(name);
    if (!seenGroups.has(group.key)) {
      seenGroups.add(group.key);
      const heading = document.createElement('li');
      heading.className = 'example-group';
      heading.innerHTML = `<span>${group.label}</span><small>${group.note}</small>`;
      ul.appendChild(heading);
    }
    const li = document.createElement('li');
    li.className = `ex-item ex-${group.key}`;
    li.dataset.example = name;
    li.setAttribute('role', 'button');
    li.tabIndex = 0;
    li.setAttribute('aria-pressed', 'false');
    li.setAttribute('aria-label', `Open example ${name}`);
    const cursor = document.createElement('span');
    cursor.className = 'ex-cursor';
    cursor.setAttribute('aria-hidden', 'true');
    const label = document.createElement('span');
    label.className = 'ex-name';
    label.textContent = name;
    const measured = document.createElement('span');
    measured.className = 'ex-bench';
    const campaignKernel = benchmarkKernel(name);
    const exampleProbe = benchmarkProbe(name);
    const sizeCount = campaignKernel?.sizes?.length ?? 0;
    measured.textContent = campaignKernel || exampleProbe ? 'M' : '—';
    measured.title = campaignKernel
      ? `measured in the recorded campaign at ${sizeCount} size${sizeCount === 1 ? '' : 's'}`
      : exampleProbe
        ? `precomputed reference-interpreter probe · ${exampleProbe.sample_count} samples`
      : 'no recorded numeric campaign point for this example';
    measured.setAttribute('aria-label', campaignKernel || exampleProbe ? 'measured data' : 'not in measured campaign');
    li.append(cursor, label, measured);
    const activate = () => selectExample(name);
    li.addEventListener('click', activate);
    li.addEventListener('keydown', ev => {
      if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); activate(); }
    });
    ul.appendChild(li);
  }
}

async function loadBenchmarkCatalog() {
  try {
    const catalog = await fetchJSON('/api/benchmarks');
    state.benchmarkCatalog = catalog && Array.isArray(catalog.kernels) ? catalog : null;
  } catch {
    // Static hosting has no /api/*: the WASM bridge ships the same campaign
    // index, so recorded kernels keep their provenance stamps.
    state.benchmarkCatalog = state.static ? await loadStaticBenchmarkCatalog() : null;
  }
}

async function loadStaticBenchmarkCatalog() {
  try {
    const { benchIndex } = await loadBrowserRunner();
    const catalog = await benchIndex();
    return catalog && Array.isArray(catalog.kernels) ? catalog : null;
  } catch {
    return null;
  }
}

// All artifact intents supersede the compile button as well as the response.
function beginArtifactRequest(compiling = false) {
  const btn = $('#btn-compile');
  btn.disabled = compiling;
  btn.textContent = compiling ? 'compiling…' : 'Run pipeline';
  return ++state.requestSeq;
}

/** Static hosting: the deploy ships the repository examples as plain .hx
 *  files, and the bridge compiles them exactly like the observer would. */
async function loadStaticExampleArtifact(name) {
  const res = await fetch('/data/examples/' + encodeURIComponent(name) + '.hx');
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  const source = await res.text();
  const bridge = await loadBrowserRunner();
  return bridge.buildArtifact(source, name);
}

async function selectExample(name) {
  const requestSeq = beginArtifactRequest();
  const editor = $('#custom-src');
  const draft = editor.value;
  try {
    const art = state.static
      ? await loadStaticExampleArtifact(name)
      : await fetchJSON(`/api/artifact?example=${encodeURIComponent(name)}`);
    // A quick tour can issue several requests before the first response
    // arrives. Only the latest selection is allowed to paint the workstation;
    // otherwise a slower, older response makes examples appear unchanged.
    if (requestSeq !== state.requestSeq) return;
    // Commit selection only after success; don't discard edits made in flight.
    if (editor.value === draft) editor.value = '';
    setArtifact(art, name);
  } catch (err) {
    if (requestSeq !== state.requestSeq) return;
    // A failed superseding action must release a pending measurement's UI
    // (retry button stays disabled while benchPending holds).
    state.benchMessage = '';
    state.benchPending = false;
    renderBench(state.artifact);
    toast(`failed to load “${name}”: ${err.message}`);
  }
}

function setArtifact(art, name, { hydrate = true } = {}) {
  state.artifact = art;
  state.exampleName = name ?? art?.example ?? '';
  state.fromCustom = state.exampleName === '<adhoc>';
  $$('#example-list .ex-item').forEach(item => {
    const selected = item.dataset.example === state.exampleName;
    item.classList.toggle('active', selected);
    item.setAttribute('aria-pressed', String(selected));
  });
  state.benchMessage = '';
  state.benchPending = false;
  state.localBench = null;
  state.browserRunSeq += 1;
  state.browserRun = null;
  state.cfgFn = 0;
  state.ssaFn = 0;
  renderAll();
  if (hydrate) void hydrateBrowserExecution(art);
}

function renderAll() {
  const art = state.artifact;
  if (!art) return;

  // stable per-render token ids so TOKENS rows can highlight SOURCE spans
  (art.tokens ?? []).forEach((tk, i) => { tk._idx = i; });

  // top bar
  const crumb = $('#top-example');
  crumb.textContent = state.exampleName || art.example || '';
  crumb.classList.toggle('on', !!crumb.textContent);

  const backend = state.browserRun?.result?.ok
    ? 'interp-wasm'
    : (art.exec?.backend_used ?? art.bench?.backend_used);
  const cb = $('#chip-backend');
  if (backend) {
    cb.textContent = '⚙ ' + backend;
    cb.classList.remove('hidden');
    cb.classList.toggle('chip-backend-ok', /par/.test(backend));
    cb.title = 'execution backend: ' + backend;
  } else cb.classList.add('hidden');

  const fixture = $('#chip-fixture');
  if (fixture) fixture.classList.toggle('hidden', !art._dev_fixture);

  const cs = $('#chip-checksum');
  const checksum = state.browserRun?.result?.checksum || art.exec?.checksum;
  if (checksum) {
    cs.textContent = checksum;
    cs.title = 'output checksum';
    cs.classList.remove('hidden');
  } else cs.classList.add('hidden');

  $('#btn-recompile').classList.toggle('hidden', !state.fromCustom);

  // The mode chip, the rail note, and the export control all follow the mode.
  renderModeChrome();

  renderDiagBanner(art);
  renderExecReadout(art);
  refreshStepper(art);
  renderSource(art);
  renderTokens(art);
  renderAST(art);
  renderCfgTabs(art);
  renderCfg(art);
  renderSSA(art);
  renderPasses(art);
  renderLoops(art);
  renderBench(art);

  // land on the most interesting available phase for error artifacts
  if (compileFailed(art) && state.phase !== 'source') {
    gotoPhase('source', true);
  } else if (!stageAvailable(art, state.phase)) {
    gotoPhase('source', true);
  } else {
    gotoPhase(state.phase, true);
  }
}

/* ============================================================
   Custom compile & run
   ============================================================ */
async function compileCustom() {
  const requestSeq = beginArtifactRequest(true);
  const btn = $('#btn-compile');
  const src = $('#custom-src').value;
  try {
    const art = state.static
      ? await (await loadBrowserRunner()).buildArtifact(src, '<adhoc>')
      : await fetchJSON('/api/run', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ source: src }),
        });
    if (requestSeq !== state.requestSeq) return;
    setArtifact(art, '<adhoc>');
    toast(compileFailed(art)
      ? `compiled with ${compilerDiagnostics(art).length} ${art.diags_lex?.length ? 'syntax' : 'semantic'} error(s)`
      : 'compile complete');
  } catch (err) {
    if (requestSeq !== state.requestSeq) return;
    // Same release rule as selectExample: a failed superseding action must
    // not leave a pending measurement's retry button disabled.
    state.benchMessage = '';
    state.benchPending = false;
    renderBench(state.artifact);
    toast('compile failed: ' + err.message);
  } finally {
    if (requestSeq === state.requestSeq) {
      btn.disabled = false;
      btn.textContent = 'Run pipeline';
    }
  }
}

/** Measure the currently compiled custom source in the browser's real WASM
 * reference interpreter, and — when the app runs without a Rust observer —
 * also against the compiled WebAssembly module. Native and parallel figures
 * are never synthesized for arbitrary input; the resulting cards are local. */
async function measureCustom() {
  const editor = $('#custom-src');
  const draft = editor.value;
  const source = state.fromCustom
    ? (draft.trim() || state.artifact?.source || '')
    : (state.artifact?.source || draft);
  if (!source.trim()) {
    toast('enter HELIX source before measuring');
    return;
  }

  const requestSeq = beginArtifactRequest();
  state.benchPending = true;
  state.benchMessage = state.static
    ? 'measuring both engines: WASM interpreter and compiled wasm…'
    : 'measuring seven bounded WASM interpreter samples…';
  renderBench(state.artifact);
  try {
    let art = state.artifact;
    if (!art || art.example !== '<adhoc>' || art.source !== source) {
      art = state.static
        ? await (await loadBrowserRunner()).buildArtifact(source, '<adhoc>')
        : await fetchJSON('/api/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ source }),
          });
    }
    if (requestSeq !== state.requestSeq) return;
    if (art.diags_lex || art.diags_sem?.length) {
      setArtifact(art, '<adhoc>');
      toast('fix the source diagnostics before measuring');
      return;
    }
    // Refuse oversized input before either engine starts, then measure the
    // reference interpreter first and the compiled module second.
    let localBench = null;
    const bridge = state.static ? await loadBrowserRunner() : null;
    if (bridge) guardBrowserWorkBudget(bridge.estimatedBrowserWork(source));
    const measured = await measureBrowserSource(source);
    if (requestSeq !== state.requestSeq) return;
    if (bridge) {
      const compiled = await measureCompiledSource(bridge, source, '<adhoc>');
      if (requestSeq !== state.requestSeq) return;
      localBench = localBenchFromMeasurements(measured, compiled);
    }
    // art.exec is independent observer evidence; browser results live in state.browserRun.
    art.bench = {
      kernel: '<adhoc>',
      n: 0,
      variants: [{
        name: 'interp-wasm-local',
        median_ms: measured.medianMs,
        samples: measured.samples,
      }],
      efficiency: [],
    };
    setArtifact(art, '<adhoc>', { hydrate: false });
    // setArtifact resets local state; the pair measured above belongs to this
    // artifact, so attach it again below.
    state.localBench = localBench;
    // The editor must hold what was measured, or Recompile would post ''.
    // Edits typed while the measurement ran are newer intent: keep them,
    // matching selectExample's draft-preservation rule.
    if (editor.value === draft) editor.value = source;
    state.browserRun = { status: 'done', result: measured.result };
    renderAll();
    gotoPhase('bench', true);
    toast(state.static
      ? 'measured both engines in this browser'
      : 'custom WASM interpreter measurement complete');
  } catch (err) {
    if (requestSeq !== state.requestSeq) return;
    state.benchPending = false;
    state.benchMessage = `measurement unavailable: ${err.message}`;
    renderBench(state.artifact);
    toast('WASM benchmark failed: ' + err.message);
  }
}

/** Measure a named example's source with both engines in this browser. Static
 *  builds use this as their only local evidence besides the recorded cards,
 *  so the selection (and every recorded provenance stamp above) is kept. */
async function measureNamedExample(name) {
  if (!state.static) return;
  const art = state.artifact;
  const source = art && art.example === name && typeof art.source === 'string' ? art.source : '';
  if (!source.trim()) {
    toast(`open “${name}” before measuring it`);
    return;
  }

  const requestSeq = beginArtifactRequest();
  state.benchPending = true;
  state.benchMessage = 'measuring both engines: WASM interpreter and compiled wasm…';
  renderBench(art);
  try {
    const bridge = await loadBrowserRunner();
    guardBrowserWorkBudget(bridge.estimatedBrowserWork(source));
    const measured = await measureBrowserSource(source);
    if (requestSeq !== state.requestSeq) return;
    const compiled = await measureCompiledSource(bridge, source, name);
    if (requestSeq !== state.requestSeq) return;
    state.localBench = localBenchFromMeasurements(measured, compiled);
    state.benchPending = false;
    state.benchMessage = '';
    state.browserRun = { status: 'done', result: measured.result };
    renderAll();
    gotoPhase('bench', true);
    toast(`measured ${name} with both engines in this browser`);
  } catch (err) {
    if (requestSeq !== state.requestSeq) return;
    state.benchPending = false;
    state.benchMessage = `measurement unavailable: ${err.message}`;
    renderBench(state.artifact);
    toast('WASM benchmark failed: ' + err.message);
  }
}

/** The compiled path: HELIX's optimized SSA IR emitted as a WebAssembly
 *  module and executed in the worker. No native or parallel execution is
 *  involved; parity with the interpreter is asserted by the caller. */
async function measureCompiledSource(bridge, source, exampleName) {
  const measured = await bridge.measureCompiled(source, exampleName);
  if (!measured?.result?.ok) {
    throw new Error(measured?.result?.message || 'the compiled WebAssembly path rejected this source');
  }
  if (!Number.isFinite(measured.medianMs) || measured.medianMs < 0) {
    throw new Error('the compiled WebAssembly path did not provide a valid execution time');
  }
  if (!Array.isArray(measured.samples) ||
      measured.samples.some(ms => !Number.isFinite(Number(ms)) || Number(ms) < 0)) {
    throw new Error('the compiled WebAssembly path returned invalid samples');
  }
  return measured;
}

/** Both engines measured on the same source in this browser. Parity compares
 *  the compiled checksum with the interpreter's: a mismatch is a compiler bug
 *  signal, so the UI reports it loudly while still showing the medians. */
function localBenchFromMeasurements(measured, compiled) {
  const interpChecksum = measured.result?.checksum ?? null;
  const compiledChecksum = compiled.checksum ?? compiled.result?.checksum ?? null;
  // null means the engine reported nothing; 0 is a real measurement.
  const num = value => {
    const bytes = Number(value);
    return value != null && Number.isFinite(bytes) && bytes >= 0 ? bytes : null;
  };
  return {
    variants: [
      { name: 'interp-wasm-local', median_ms: measured.medianMs, samples: measured.samples },
      { name: 'compiled-wasm-local', median_ms: compiled.medianMs, samples: compiled.samples },
    ],
    memory: {
      interpBytes: num(measured.result?.alloc_bytes),
      compiledBytes: num(compiled.heapBytes),
    },
    parity: { same: interpChecksum === compiledChecksum, checksum: interpChecksum, compiledChecksum },
  };
}

/* ============================================================
   Boot + global events
   ============================================================ */
function wireEvents() {
  $('#src-show-diag').addEventListener('change', () => renderSource(state.artifact));
  $('#ssa-side-by-side').addEventListener('change', () => renderSSA(state.artifact));
  $('#bench-logscale').addEventListener('change', ev => {
    state.benchLog = ev.target.checked;
    renderBench(state.artifact);
  });
  $('#btn-compile').addEventListener('click', compileCustom);
  $('#btn-recompile').addEventListener('click', async () => {
    // reload current artifact through whichever channel produced it
    if (state.exampleName && state.exampleName !== '<adhoc>') {
      await selectExample(state.exampleName);
    } else {
      await compileCustom();
    }
  });

  // a scrolled-away token must not leave its diagnostic tooltip stranded
  document.addEventListener('scroll',
    () => $('.diag-hover-tip')?.classList.remove('show'), true);

  document.addEventListener('keydown', ev => {
    const tag = document.activeElement?.tagName;
    const typing = tag === 'TEXTAREA' || tag === 'INPUT';
    if (ev.key === 'ArrowRight' && !typing) { ev.preventDefault(); stepPhase(+1); }
    else if (ev.key === 'ArrowLeft' && !typing) { ev.preventDefault(); stepPhase(-1); }
    else if (/^[1-8]$/.test(ev.key) && !typing) {
      const target = PHASES[Number(ev.key) - 1];
      if (target) gotoPhase(target[0]);
    }
  });
}

function stepPhase(dir) {
  const idx = PHASES.findIndex(p => p[0] === state.phase);
  for (let i = idx + dir; i >= 0 && i < PHASES.length; i += dir) {
    if (!state.artifact || stageAvailable(state.artifact, PHASES[i][0])) {
      gotoPhase(PHASES[i][0]);
      return;
    }
  }
}

/** A failed probe (a 404 on GitHub Pages, or an unreachable origin) means no
 *  Rust observer serves this site, so every request runs through the WASM
 *  bridge instead. Plain fetch is deliberate: a non-ok status is the signal,
 *  and fetchJSON would turn it into an exception either way. */
async function detectStaticMode() {
  try {
    const res = await fetch('/api/examples');
    state.static = !res.ok;
  } catch {
    state.static = true;
  }
}

/** Mode chrome: which engine compiles this site's artifacts, and which
 *  controls that mode keeps. */
function renderModeChrome() {
  const mode = $('#chip-mode');
  if (mode) {
    mode.textContent = state.static ? 'WASM LAB' : 'OBSERVER';
    mode.title = state.static
      ? 'Compiled and executed in this browser via WebAssembly — no Rust observer serves this site.'
      : 'Compiled and executed by the Rust observer serving this site.';
    mode.classList.toggle('chip-mode-static', state.static);
    mode.classList.remove('hidden');
  }
  const note = $('#rail-mode-note');
  if (note) {
    note.innerHTML = state.static
      ? 'Compiles in this browser via WebAssembly. After it compiles, open BENCH to measure both engines locally.'
      : 'POSTs the buffer to <code>/api/run</code>. After it compiles, open BENCH to measure a bounded interpreter run.';
  }
  // Standalone HTML export is rendered by the observer's artifact endpoint, so
  // a static build keeps the control hidden with an explanatory title.
  const exBtn = $('#btn-export');
  if (exBtn) {
    const exported = !state.static && state.examples.includes(state.exampleName);
    exBtn.classList.toggle('hidden', !exported);
    exBtn.title = state.static
      ? 'Standalone export needs the Rust observer — unavailable in this static WASM build.'
      : "open this example's artifact as a standalone offline HTML page";
    if (exported) exBtn.href = `/api/artifact/export?example=${encodeURIComponent(state.exampleName)}`;
  }
}

async function boot() {
  buildStepper();
  wireEvents();
  // Decide the mode before any catalog or artifact load: static hosting
  // answers no /api/*, and the WASM bridge then carries every request.
  await detectStaticMode();
  renderModeChrome();
  // The example rail uses the catalog to mark measured kernels, so load its
  // provenance before painting the list. A partial catalog must never make a
  // measured example look unmeasured merely because two requests raced.
  await loadBenchmarkCatalog();
  await loadExamples();
  // open on the strongest showcase that actually loaded, else the first entry
  const preferred = ['scale', 'recurrence_reject', 'dot_reduction']
    .map(name => $(`#example-list .ex-item[data-example="${CSS.escape(name)}"]`))
    .find(Boolean);
  const first = preferred ?? $('#example-list .ex-item');
  if (first) selectExample(first.dataset.example);
  else {
    // fully offline: no API, no fixtures — leave a graceful shell
    setArtifact({
      schema: 1, example: '(offline)', source: '// no artifact available\n',
      tokens: null, diags_sem: null,
    }, '(offline)');
  }
}

boot();
