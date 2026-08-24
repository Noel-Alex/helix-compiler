// HELIX Observatory — offline dev server.
// Serves web/ statically and fakes the Rust API from web/dev-fixtures/ so the
// UI can be exercised before `helix observe` exists. Node >= 18, no deps.
//   node dev-server.mjs [port]        (default 8137)
// Routes:
//   GET  /api/examples                 -> ["saxpy_trio","dot_product","type_errors"]
//   GET  /api/artifact?example=<name>  -> matching fixture (default: sample)
//   POST /api/run {source}             -> sema-error artifact when source has no
//                                         "for", else the sample artifact
import http from 'node:http';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.argv[2] ?? 8137);

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.ico': 'image/x-icon',
  '.woff2': 'font/woff2',
};

const cache = new Map();
const loadJson = async rel => {
  if (!cache.has(rel)) {
    try {
      cache.set(rel, JSON.parse(await readFile(path.join(ROOT, 'dev-fixtures', rel), 'utf8')));
    } catch {
      cache.set(rel, null);
    }
  }
  return cache.get(rel);
};

const EXAMPLES = [
  { name: 'saxpy_trio', file: 'artifact-sample.json' },
  { name: 'type_errors', file: 'artifact-sema-error.json' },
];

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);
  try {
    // ---- fake API -------------------------------------------------------
    if (url.pathname === '/api/examples') {
      return json(res, EXAMPLES.map(e => e.name));
    }
    if (url.pathname === '/api/artifact') {
      const want = url.searchParams.get('example');
      const hit = EXAMPLES.find(e => e.name === want) ?? EXAMPLES[0];
      return json(res, await loadJson(hit.file));
    }
    if (url.pathname === '/api/run' && req.method === 'POST') {
      let body = '';
      for await (const chunk of req) body += chunk;
      let source = '';
      try { source = JSON.parse(body).source ?? ''; } catch { /* ignore */ }
      // "compile": sources without a loop stop at sema -> reuse the error fixture
      const art = source.includes('for') ? await loadJson('artifact-sample.json')
                                          : await loadJson('artifact-sema-error.json');
      return json(res, art);
    }
    if (url.pathname === '/api/health') return json(res, { ok: true, dev: true });

    // ---- static files ---------------------------------------------------
    let p = decodeURIComponent(url.pathname);
    if (p === '/' || p === '') p = '/index.html';
    const abs = path.normalize(path.join(ROOT, p));
    if (!abs.startsWith(ROOT)) { res.writeHead(403); return res.end('forbidden'); }
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

function json(res, obj) {
  res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' });
  res.end(JSON.stringify(obj));
}

server.listen(PORT, '127.0.0.1', () => {
  console.log(`HELIX Observatory dev server — http://localhost:${PORT}/  (fixtures from ./dev-fixtures)`);
});
