// HELIX Observatory — offline dev server.
// Serves web/ statically and provides an honest source-only fallback when the
// Rust observer is unavailable. Node >= 18, no deps.
//   node dev-server.mjs [port]        (default 8137)
// Routes:
//   GET  /api/examples                 -> the complete repository example list
//   GET  /api/artifact?example=<name>  -> source-only fixture for that example
//   POST /api/run {source}             -> source-only fixture for that source
//   GET  /api/benchmarks                -> empty fixture campaign index
//   POST /api/benchmark {source}        -> source-only fixture (no timing)
//   GET  /data/examples/<name>.hx      -> the repository example source the
//                                         static mode fetches on GitHub Pages
//
// When the canonical Rust observer is available on HELIX_RUST_ORIGIN (default
// http://127.0.0.1:8931), API requests are proxied to it. The fixture path is
// only a fallback, so navigating the dev port still changes artifacts per
// example instead of silently reusing one sample for every name.
import http from 'node:http';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.dirname(fileURLToPath(import.meta.url));
// Prefix check must include the separator, or a sibling like ROOT + "foo"
// would pass a bare-ROOT startsWith guard (traversal via /../webfoo).
const ROOT_WITH_SEP = ROOT.endsWith(path.sep) ? ROOT : ROOT + path.sep;
const EXAMPLES_ROOT = path.resolve(ROOT, '..', 'examples');
const PORT = Number(process.argv[2] ?? 8137);
const RUST_ORIGIN = process.env.HELIX_RUST_ORIGIN ?? 'http://127.0.0.1:8931';

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.ico': 'image/x-icon',
  '.ttf': 'font/ttf',
  '.wasm': 'application/wasm',
  '.woff2': 'font/woff2',
};

const loadExampleSource = async name => {
  try {
    return await readFile(path.join(EXAMPLES_ROOT, `${name}.hx`), 'utf8');
  } catch {
    return '';
  }
};

function sourceOnlyArtifact(example, source) {
  return {
    schema: 1,
    example,
    source,
    tokens: null,
    ast: null,
    diags_sem: null,
    ir_pre_ssa: null,
    ir_ssa: null,
    passes: null,
    cfg: null,
    domtree: null,
    loops: null,
    exec: null,
    bench: null,
    _dev_fixture: true,
    _dev_example: example,
    _dev_truth: 'source-only fixture; start the Rust observer for canonical compiler artifacts and benchmark evidence',
  };
}

const EXAMPLE_NAMES = [
  'casts_demo', 'const_globals', 'count_primes_sieve', 'div_guard', 'dot_reduction',
  'fib_recursion', 'gcd_box_test', 'jacobi_2d', 'matmul', 'minmax_reduction',
  'recurrence_reject', 'saxpy', 'scale', 'shortcircuit', 'small_n', 'ssa_demo',
  'stencil_2d_reject', 'type_errors', 'writer_call',
];

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);
  try {
    const requestBody = req.method === 'POST' ? await readBody(req) : null;

    // Prefer the canonical Rust observer for every API except health. This
    // keeps the convenient fixture port and the real port on one truth path
    // whenever both are running. A connection failure falls through to the
    // clearly marked development fixtures below.
    if (url.pathname.startsWith('/api/') && url.pathname !== '/api/health') {
      const proxied = await proxyRust(req.method, `${url.pathname}${url.search}`, requestBody, res);
      if (proxied) return;
    }

    // ---- fake API -------------------------------------------------------
    if (url.pathname === '/api/examples') {
      return json(res, EXAMPLE_NAMES);
    }
    if (url.pathname === '/api/benchmarks') {
      return json(res, {
        schema: 'helix-campaign-dev-fixture',
        timestamp_utc: null,
        git_rev: null,
        point_count: 0,
        kernels: [],
        examples: [],
        points: [],
        probe_provenance: null,
        machine: null,
        _dev_truth: 'source-only fixture; start the Rust observer for recorded benchmark evidence',
      });
    }
    if (url.pathname === '/api/artifact') {
      const want = url.searchParams.get('example');
      if (!EXAMPLE_NAMES.includes(want)) return json(res, { error: `unknown example: ${want}` }, 404);
      const source = await loadExampleSource(want);
      return json(res, sourceOnlyArtifact(want, source));
    }
    if (url.pathname === '/api/run' && req.method === 'POST') {
      const body = requestBody ?? '';
      let source = '';
      try { source = JSON.parse(body).source ?? ''; } catch { /* ignore */ }
      return json(res, sourceOnlyArtifact('<adhoc>', source));
    }
    if (url.pathname === '/api/benchmark' && req.method === 'POST') {
      const body = requestBody ?? '';
      let source = '';
      try { source = JSON.parse(body).source ?? ''; } catch { /* ignore */ }
      return json(res, sourceOnlyArtifact('<adhoc>', source));
    }
    if (url.pathname === '/api/health') return json(res, { ok: true, dev: true });

    // Static-mode parity: GitHub Pages serves the examples as plain files at
    // /data/examples/, so the fallback port answers the same route from the
    // repository sources. The name pattern plus the allow-list keep traversal
    // out, exactly like the /api/artifact handler above.
    const exampleFile = url.pathname.match(/^\/data\/examples\/([A-Za-z0-9_]+)\.hx$/);
    if (exampleFile) {
      const want = exampleFile[1];
      if (!EXAMPLE_NAMES.includes(want)) return json(res, { error: `unknown example: ${want}` }, 404);
      res.writeHead(200, { 'Content-Type': 'text/plain; charset=utf-8', 'Cache-Control': 'no-store' });
      return res.end(await loadExampleSource(want));
    }

    // ---- static files ---------------------------------------------------
    let p = decodeURIComponent(url.pathname);
    if (p === '/' || p === '') p = '/index.html';
    if (p === '/about') p = '/about.html';
    const abs = path.normalize(path.join(ROOT, p));
    if (!abs.startsWith(ROOT_WITH_SEP) && abs !== ROOT) { res.writeHead(403); return res.end('forbidden'); }
    const data = await readFile(abs).catch(() => null);
    if (!data) { res.writeHead(404, { 'Content-Type': 'text/plain' }); return res.end('404: ' + p); }
    res.writeHead(200, { 'Content-Type': MIME[path.extname(abs)] ?? 'application/octet-stream',
                         'Cache-Control': 'no-store' });
    res.end(data);
  } catch (err) {
    res.writeHead(500, { 'Content-Type': 'text/plain' });
    res.end('dev-server error: ' + err.message);
  }
});

async function readBody(req) {
  let body = '';
  for await (const chunk of req) body += chunk;
  return body;
}

async function proxyRust(method, pathname, body, res) {
  try {
    const upstream = await fetch(`${RUST_ORIGIN}${pathname}`, {
      method,
      body: method === 'POST' ? body : undefined,
      headers: method === 'POST' ? { 'content-type': 'application/json' } : undefined,
      signal: AbortSignal.timeout(pathname.startsWith('/api/benchmark') ? 30000 : 1800),
    });
    const payload = Buffer.from(await upstream.arrayBuffer());
    res.writeHead(upstream.status, {
      'Content-Type': upstream.headers.get('content-type') ?? 'application/octet-stream',
      'Cache-Control': 'no-store',
      'X-Helix-Source': 'rust-observer',
    });
    res.end(payload);
    return true;
  } catch {
    return false;
  }
}

function json(res, obj, status = 200) {
  res.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' });
  res.end(JSON.stringify(obj));
}

server.listen(PORT, '127.0.0.1', () => {
  console.log(`HELIX Observatory dev server — http://localhost:${PORT}/  (Rust proxy + honest source-only fallback)`);
});
