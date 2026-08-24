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
  exampleName: '',
  artifact: null,
  phase: 'source',
  cfgFn: 0,
  ssaFn: 0,
  benchLog: false,
};

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
  // byte offset -> line index (0-based); source is ASCII so JS indices == bytes
  const starts = [0];
  for (let i = 0; i < source.length; i++) if (source[i] === '\n') starts.push(i + 1);
  return starts;
}

/** Render the SOURCE phase from artifact.tokens. Returns nothing. */
function renderSource(art) {
  const tbody = $('#source-view tbody');
  tbody.textContent = '';
  $('#src-sub').textContent =
    art ? `${art.tokens ? art.tokens.length : 0} tokens · ${art.source.split('\n').length} lines` : '';

  const diags = Array.isArray(art?.diags_sem) ? art.diags_sem : [];
  const showDiag = $('#src-show-diag').checked;

  if (!Array.isArray(art?.tokens)) {
    tbody.innerHTML = `<tr><td class="empty">// source not available in this artifact</td></tr>`;
    return;
  }

  const src = art.source ?? '';
  const lineStarts = buildLineMap(src);

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
    const s = Math.min(Math.max(tk.start, 0), src.length);
    const e = Math.min(Math.max(tk.end, s), src.length);
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
    const ln = lineAt(lineStarts, Math.min(d.span?.start ?? 0, Math.max(0, src.length - 1)));
    if (!diagByLine.has(ln)) diagByLine.set(ln, []);
    diagByLine.get(ln).push(d);
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
      if (ch === '\n') { s++; rows.push([]); continue; }
      rows[rows.length - 1].push({ text: ch, cls: it.cls || '', start: s, end: s + 1, tok: it.tok });
      s++;
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
      if (diagsHere.length) {
        const hit = diagsHere.some(d => sp.end > d.span.start && sp.start < d.span.end);
        if (hit) span.classList.add('diag-u');
      }
      cell.appendChild(span);
      if (sp.tok) attachTokenHover(span, sp.tok);
      if (diagsHere.length) attachDiagHover(span, diagsHere, sp);
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
function attachDiagHover() {}

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
    txt.className = 'tt-text'; txt.textContent = tk.kind === 'Eof' ? '⏚ eof' : (tk.text === '' ? '∅' : tk.text);
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

/* graph tooltip (shared by CFG + AST) */
function showGraphTip(ev, html) {
  let tip = $('.cfg-hover-tip');
  if (!tip) {
    tip = document.createElement('div');
    tip.className = 'cfg-hover-tip';
    (document.querySelector('.panel-viewport') ?? document.body).appendChild(tip);
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

  body.classList.toggle('dual', sideBySide && !!preText && !!ssaText);
  body.classList.toggle('single', !(sideBySide && !!preText && !!ssaText));

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

    const head = document.createElement('div');
    head.className = 'pass-head';

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
      d.className = 'pass-delta ' + (delta < 0 ? 'neg' : 'zero');
      d.textContent = delta < 0 ? `−${delta}` : '±0';
      stat.prepend(d);
      stat.insertBefore(document.createTextNode('  '), d.nextSibling);
    }

    head.append(caret, badge, name, stat);
    head.addEventListener('click', () => card.classList.toggle('open'));

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
  SEQUENTIAL: { glyph: '⚠', label: 'SEQUENTIAL' },
  REJECTED:   { glyph: '⚠', label: 'REJECTED' },
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
    const rejected = verdict === 'SEQUENTIAL' || verdict === 'REJECTED';

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
    why.title = L.explain || L.reason || '';
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
          `<span>${esc(d.sink ?? d.to ?? '?')}</span><span class="arr"> ← </span><span>${esc(d.source ?? d.from ?? '?')}</span>`;
        pair.title = d.explain ?? d.note ?? '';
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
    if (verdict === 'SAFE' && L.plan) {
      const plan = document.createElement('span');
      plan.className = 'plan-strip';
      plan.innerHTML = `⚙ plan&nbsp;&nbsp;strip-mined → ${esc(String(L.plan.threads ?? '?'))} threads` +
        (L.plan.tile && L.plan.tile !== 'none' ? ` · tile=${esc(L.plan.tile)}` : '');
      plan.title = 'runtime execution plan chosen for this loop';
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

function renderBench(art) {
  const chart = $('#bench-chart');
  const effWrap = $('#bench-eff');
  const empty = $('#bench-empty');
  chart.textContent = ''; effWrap.textContent = '';
  empty.classList.add('hidden');

  const bench = art?.bench;
  $('#bench-kernel').textContent = bench ? `kernel ${bench.kernel} · n = ${Number(bench.n).toLocaleString('en-US')}` : '';
  if (!bench || !Array.isArray(bench.variants) || !bench.variants.length) {
    empty.textContent = '// no benchmark campaign attached to this artifact (run helix bench)';
    empty.classList.remove('hidden');
    return;
  }

  const variants = [...bench.variants].sort((a, b) => b.median_ms - a.median_ms); // slowest at top
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
  const scaleMax = useLog ? Math.max(...variants.map(v => v.median_ms)) : Math.max(...variants.map(v => v.median_ms)) * 1.06;
  const toX = ms => useLog
    ? (Math.log10(Math.max(ms, 0.001)) / Math.log10(scaleMax)) * plotW
    : (ms / scaleMax) * plotW;

  // gridlines + ticks
  const ticks = useLog ? logTicks(scaleMax) : linTicks(scaleMax);
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
    val.textContent = fmt(v.median_ms, 1) + ' ms';
    svg.appendChild(val);

    {
      const speedup = v.median_ms > 0 ? fastest / v.median_ms : 0;
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
    title.textContent = `${v.name}\nmedian ${fmt(v.median_ms)} ms` +
      (Array.isArray(v.samples) ? `\nsamples: ${v.samples.map(s => fmt(s)).join(', ')}` : '');
    rect.appendChild(title);
  });

  chart.appendChild(svg);

  // efficiency table
  const eff = bench.efficiency ?? [];
  if (eff.length) {
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

function linTicks(max) {
  const out = [];
  const step = niceStep(max / 5);
  for (let v = 0; v <= max * 1.0001; v += step) out.push({ v, label: tickLabel(v) });
  return out;
}
function logTicks(max) {
  const out = [];
  const expMax = Math.ceil(Math.log10(max));
  for (let e = 0; e <= expMax; e++) {
    const v = Math.pow(10, e);
    if (v <= max * 1.0001) out.push({ v, label: tickLabel(v) });
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
function semaFailed(art) {
  return Array.isArray(art?.diags_sem) && art.diags_sem.length > 0;
}
function stageAvailable(art, phase) {
  if (!art) return false;
  if (semaFailed(art)) return ['source', 'tokens'].includes(phase);
  switch (phase) {
    case 'source': case 'tokens': return Array.isArray(art.tokens);
    case 'ast': return !!art.ast;
    case 'cfg': return Array.isArray(art.cfg?.functions) && art.cfg.functions.length > 0;
    case 'ssa': return !!(art.ir_ssa?.functions?.length || art.ir_pre_ssa?.functions?.length);
    case 'opt': return Array.isArray(art.passes) && art.passes.length > 0;
    case 'loops': return Array.isArray(art.loops) && art.loops.length > 0;
    case 'bench': return !!art.bench;
    default: return false;
  }
}

function renderDiagBanner(art) {
  const ban = $('#diag-banner');
  const diags = Array.isArray(art?.diags_sem) ? art.diags_sem : [];
  if (!diags.length) { ban.classList.add('hidden'); ban.textContent = ''; return; }
  ban.classList.remove('hidden');
  ban.innerHTML = '';
  const g = document.createElement('span'); g.className = 'dg-glyph'; g.textContent = '⚠';
  const msg = document.createElement('span');
  msg.innerHTML = `<b>${diags.length} semantic error${diags.length > 1 ? 's' : ''}</b>` +
    ` — pipeline stopped after semantic analysis. Underlined regions in SOURCE mark each error.`;
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
  $$('.step').forEach(st => st.classList.toggle('active', st.id === 'step-' + id));
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
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
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
    // static fallback — mirrors the repo's examples/ directory
    state.examples = ['saxpy', 'matmul', 'stencil_2d', 'jacobi_2d', 'dot_reduction',
                      'fib_recursion', 'gcd_box_test', 'minmax_reduction', 'count_primes_sieve'];
  }
  ul.textContent = '';
  for (const name of state.examples) {
    const li = document.createElement('li');
    li.className = 'ex-item';
    li.dataset.example = name;
    li.textContent = name;
    li.addEventListener('click', () => selectExample(name, li));
    ul.appendChild(li);
  }
}

async function selectExample(name, li) {
  $$('#example-list .ex-item').forEach(x => x.classList.remove('active'));
  (li ?? $(`#example-list .ex-item[data-example="${CSS.escape(name)}"]`))?.classList.add('active');
  $('#custom-src').value = '';
  state.fromCustom = false;
  try {
    const art = await fetchJSON(`/api/artifact?example=${encodeURIComponent(name)}`);
    setArtifact(art, name);
  } catch (err) {
    toast(`failed to load “${name}”: ${err.message}`);
  }
}

function setArtifact(art, name) {
  state.artifact = art;
  state.exampleName = name ?? art?.example ?? '';
  state.cfgFn = 0;
  state.ssaFn = 0;
  renderAll();
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

  const backend = art.exec?.backend_used ?? art.bench?.backend_used;
  const cb = $('#chip-backend');
  if (backend) {
    cb.textContent = '⚙ ' + backend;
    cb.classList.remove('hidden');
    cb.classList.toggle('chip-backend-ok', /par/.test(backend));
    cb.title = 'execution backend: ' + backend;
  } else cb.classList.add('hidden');

  const cs = $('#chip-checksum');
  if (art.exec?.checksum) {
    cs.textContent = art.exec.checksum;
    cs.title = 'output checksum';
    cs.classList.remove('hidden');
  } else cs.classList.add('hidden');

  $('#btn-recompile').classList.toggle('hidden', !state.fromCustom);

  renderDiagBanner(art);
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
  if (semaFailed(art) && state.phase !== 'source') {
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
  const btn = $('#btn-compile');
  const src = $('#custom-src').value;
  btn.disabled = true;
  const old = btn.textContent;
  btn.textContent = 'compiling…';
  try {
    const art = await fetchJSON('/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source: src }),
    });
    $$('#example-list .ex-item').forEach(x => x.classList.remove('active'));
    state.fromCustom = true;
    setArtifact(art, '<adhoc>');
    toast(art.diags_sem?.length
      ? `compiled with ${art.diags_sem.length} semantic error(s)`
      : 'compile complete');
  } catch (err) {
    toast('compile failed: ' + err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = old;
  }
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

async function boot() {
  buildStepper();
  wireEvents();
  await loadExamples();
  const first = $('#example-list .ex-item');
  if (first) selectExample(first.textContent, first);
  else {
    // fully offline: no API, no fixtures — leave a graceful shell
    setArtifact({
      schema: 1, example: '(offline)', source: '// no artifact available\n',
      tokens: null, diags_sem: null,
    }, '(offline)');
  }
}

boot();
