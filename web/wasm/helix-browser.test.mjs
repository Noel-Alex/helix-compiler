import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';

const bridgeUrl = new URL('./helix-browser.js', import.meta.url);
const workerUrl = new URL('./helix-worker.js', import.meta.url);
const bridgeSource = await readFile(bridgeUrl, 'utf8');
const workerSource = await readFile(workerUrl, 'utf8');
const wasm = await readFile(new URL('./helix-lab.wasm', bridgeUrl));
// Only replace ESM syntax so the shipped functions run in controllable globals.
const script = (source, url) => source.replace(/\bexport /g, '')
  .replaceAll('import.meta.url', JSON.stringify(url.href));

// Exercise the real worker handler and bundled WASM with ordinary teaching code.
async function workerReply(data, status = 200) {
  const replies = [];
  const self = { postMessage: value => replies.push(value) };
  const context = vm.createContext({
    self, URL, TextEncoder, TextDecoder, WebAssembly, performance,
    fetch: async url => {
      assert.equal(url.href, new URL('./helix-lab.wasm', workerUrl).href);
      return { ok: status === 200, status, arrayBuffer: async () => wasm };
    },
  });
  vm.runInContext(script(workerSource, workerUrl), context);
  await self.onmessage({ data });
  assert.equal(self.onmessage, null, 'worker accepts only one request');
  assert.equal(replies.length, 1);
  return replies[0];
}
const smoke = await workerReply({ type: 'run', source: 'fn main() { print(15); }' });
assert.equal(smoke.type, 'result');
assert.equal(smoke.result.ok, true);
assert.equal(smoke.result.status, 'ok');
assert.deepEqual(Array.from(smoke.result.printed), ['15']);
assert.match(smoke.result.engine, /HELIX reference interpreter \/ wasm32/);
assert.equal(typeof smoke.result.alloc_bytes, 'number');
assert.ok(Number.isFinite(smoke.result.elapsedMs) && smoke.result.elapsedMs >= 0);
assert.equal((await workerReply({ type: 'run', source: '' })).result.ok, false);
assert.equal((await workerReply({ type: 'load' })).type, 'ready');
assert.match((await workerReply({ type: 'load' }, 503)).error, /503/);

// No expensive programs: a deliberately nonresponding mock drives the deadline.
const workers = [], timers = new Map();
let nextTimer = 0;
class MockWorker {
  constructor(url, options) {
    assert.equal(url.href, workerUrl.href);
    assert.equal(options.type, 'module');
    this.terminations = 0;
    workers.push(this);
  }
  postMessage(message) { this.message = message; }
  terminate() { this.terminations += 1; }
  reply(data) { this.onmessage({ data }); }
}
const context = vm.createContext({
  URL, Worker: MockWorker, TextEncoder,
  setTimeout(callback, delay) {
    assert.equal(delay, 5000);
    timers.set(++nextTimer, callback);
    return nextTimer;
  },
  clearTimeout(id) { timers.delete(id); },
});
vm.runInContext(script(bridgeSource, bridgeUrl), context);
const { runHelixInBrowser: run, loadHelixWasm: load } = context;
const estimate = context.estimatedBrowserWork;
const smallProgram = 'fn main() { print(15); }';
assert.equal(estimate(`// 999999999\n${smallProgram}`), estimate(smallProgram));
assert.equal(estimate(`${smallProgram} // 999999999`), estimate(smallProgram));
assert.equal(estimate(`// for i in 0..999999999\r\n${smallProgram}`), estimate(smallProgram));
// Block comments are executable-token-free too. A scalar literal is one
// operation, not per-iteration work; an allocation charges its length.
assert.equal(estimate(`/* 3000000 */\n${smallProgram}`), estimate(smallProgram));
assert.ok(estimate('fn main() { print(3000000); }') < 2_000_000);
assert.equal(estimate('fn main() { let a: [i64] = zeros(3000000); print(a[0]); }'), 3_000_000);
// Sequential sibling loops add trips instead of compounding per for-line.
const siblings = 'fn main() {\nfor i in 0..2 { print(i); }\nfor j in 0..2 { print(j); }\nfor k in 0..2 { print(k); }\n}';
assert.ok(estimate(siblings) < 2_000_000, `sibling loops must stay runnable, got ${estimate(siblings)}`);
// Genuinely nested loops still multiply: 100 outer × (1 + 100 inner).
const nested = 'fn main() { for i in 0..100 { for j in 0..100 { print(j); } } }';
assert.equal(estimate(nested), 10_100);

const last = () => workers.at(-1);
const clean = worker => {
  assert.equal(worker.terminations, 1);
  assert.equal(worker.onmessage, null);
  assert.equal(worker.onerror, null);
  assert.equal(worker.onmessageerror, null);
};

const ready = load();
last().reply({ type: 'ready' });
assert.equal(await ready, undefined);
clean(last());

const first = run('first'), a = last();
const second = run('second'), b = last();
assert.notEqual(a, b);
b.reply({ type: 'result', result: { ok: true, printed: ['second'], elapsedMs: 2 } });
a.reply({ type: 'result', result: smoke.result });
assert.equal((await first).printed[0], '15');
assert.equal((await second).printed[0], 'second');
clean(a); clean(b);
assert.equal(timers.size, 0);

for (const fail of [
  worker => worker.reply({ type: 'error', error: 'load failed' }),
  worker => worker.onerror({ message: 'worker crashed' }),
  worker => worker.onmessageerror(),
  worker => worker.reply({ type: 'nonsense' }),
  () => [...timers.values()][0](),
]) {
  const pending = run('mock request'), worker = last();
  const lateReply = worker.onmessage;
  const rejection = assert.rejects(pending, /failed|crashed|could not be read|invalid response|timed out/);
  fail(worker);
  await rejection;
  clean(worker);
  assert.equal(timers.size, 0);
  lateReply({ data: { type: 'result', result: { ok: true } } });
  clean(worker);
  const retry = run('retry');
  last().reply({ type: 'result', result: smoke.result });
  assert.equal((await retry).ok, true);
  clean(last());
}
MockWorker.prototype.postMessage = () => { throw new Error('cannot post'); };
await assert.rejects(run('mock request'), /cannot post/);
clean(last());
context.Worker = undefined;
await assert.rejects(run('mock request'), /Web Workers are unavailable/);
assert.equal(timers.size, 0);

// The observer embeds and serves the worker with a module-compatible MIME type.
const server = await readFile(new URL('../../crates/helix-observe/src/server.rs', bridgeUrl), 'utf8');
assert.match(server, /\.route\("\/wasm\/helix-worker\.js", get\(wasm_worker_js\)\)/);
assert.match(server, /const WASM_WORKER_JS:.*include_bytes!\("\.\.\/\.\.\/\.\.\/web\/wasm\/helix-worker\.js"\)/);
assert.match(server, /async fn wasm_worker_js\(\)[\s\S]*?bytes_response\(WASM_WORKER_JS, "text\/javascript; charset=utf-8"\)/);

// ---- the generic apply protocol, against the real host wasm ---------------
const formatLine = (tag, bits) => workerReply({ type: 'apply', op: 'format_line', tag, bits });
const i64 = await formatLine('i64', '15');
assert.equal(i64.type, 'applied');
assert.equal(i64.json.text, '15');
// An i32 arrives zero-extended; the host reads the word back as a signed
// 32-bit value, so -1 must reach it as the full 4294967295 pattern.
assert.equal((await formatLine('i32', String(0xffffffff))).json.text, '-1');
assert.equal((await formatLine('i32', '21')).json.text, '21');
assert.equal((await formatLine('bool', '1')).json.text, 'true');
assert.equal((await formatLine('bool', '0')).json.text, 'false');
const ieee = new DataView(new ArrayBuffer(8));
ieee.setFloat64(0, -2.5, true);
assert.equal((await formatLine('f64', String(ieee.getBigUint64(0, true)))).json.text, '-2.5');
const wide = new DataView(new ArrayBuffer(4));
wide.setFloat32(0, 0.1, true);
assert.equal((await formatLine('f32', String(wide.getUint32(0, true)))).json.text, '0.1');
assert.match((await workerReply({ type: 'apply' })).error, /op string/);
// A malformed module body is reported by the worker, not thrown upward.
const badModule = await workerReply({ type: 'runModule', module_b64: 'not-base-64!!' });
assert.equal(badModule.type, 'moduleResult');
assert.equal(badModule.result.ok, false);
assert.match(badModule.result.message, /not padded|outside the base64 alphabet/);
assert.match((await workerReply({ type: 'unknown' })).error, /Unknown request type/);

// ---- compiled pipeline, driven through the real worker handler ------------
// Each bridge call gets its own worker; this adapter serves that call with the
// real worker handler, the real host wasm, and the real fetched module bytes.
const labTimers = new Set();
const labWorkers = [];
class LabWorker {
  constructor(url, options) {
    assert.equal(url.href, workerUrl.href);
    assert.equal(options.type, 'module');
    this.terminations = 0;
    this.messages = [];
    this.onmessage = this.onerror = this.onmessageerror = null;
    labWorkers.push(this);
  }
  postMessage(message) {
    this.messages.push(message);
    void this.serve(message);
  }
  async serve(message) {
    const reply = await workerReply(message);
    if (!this.onmessage) return; // the bridge already finished or timed out
    if (reply.type === 'error') this.onerror({ message: reply.error });
    else this.onmessage({ data: reply });
  }
  terminate() { this.terminations += 1; }
}
const lab = vm.createContext({
  URL, TextEncoder, Worker: LabWorker,
  setTimeout: (callback, delay) => {
    assert.equal(delay, 5000);
    labTimers.add(callback);
    return callback;
  },
  clearTimeout: id => labTimers.delete(id),
});
vm.runInContext(script(bridgeSource, bridgeUrl), lab);
const {
  buildArtifact, emitModule, runCompiled, measureCompiled, benchIndex,
  runHelixInBrowser: interpret, repetitionsFor,
} = lab;

// A saxpy-shaped loop printing f64s.
const saxpyLab = `fn main() {
    let n = 8;
    let x: [f64] = zeros(n);
    let y: [f64] = zeros(n);
    for i in 0..n {
        x[i] = i as f64;
        y[i] = 0.5;
    }
    for i in 0..n {
        y[i] = 2.0 * x[i] + y[i];
        print(y[i]);
    }
}
`;
// Recursion.
const fibLab = `fn fib(n: i64) -> i64 {
    if n < 2 { return n; }
    return fib(n - 1) + fib(n - 2);
}
fn main() { print(fib(18)); }
`;
// Casts, short-circuit logic, bools, math builtins, and an [i32] array.
const mixedLab = `fn main() {
    print(300.7 as i32);
    print(3.9 as i64);
    print(true);
    print(false);
    print(sqrt(2.0));
    print(abs(-12));
    print(min(3, 9));
    print(max(3, 9));
    let a: [i32] = zeros(2);
    a[0] = 7;
    print(a[0]);
    print(1 < 2 && 2 < 3);
    print(1 > 2 || 2 > 3);
}
`;
// Integer loop with array writes: the measured shape.
const measureLab = `fn main() {
    let n = 1024;
    let acc: [i64] = zeros(n);
    for i in 0..n {
        acc[i] = i * i;
    }
    let total = 0;
    for i in 0..n {
        total = total + acc[i];
    }
    print(total);
}
`;

const artifact = await buildArtifact(saxpyLab, 'browser-lab-saxpy');
assert.equal(artifact.schema, 1);
assert.equal(artifact.example, 'browser-lab-saxpy');
assert.equal(artifact.source, saxpyLab);
assert.ok(Array.isArray(artifact.tokens) && artifact.tokens.length > 0, 'artifact needs a token stream');
assert.ok(Array.isArray(artifact.loops) && artifact.loops.length > 0, 'artifact needs analyzed loops');
assert.equal(artifact.diags_sem.length, 0, 'a compiling program carries no semantic diagnostics');
assert.deepEqual(artifact.diags_lex ?? null, null);

const emitted = await emitModule(saxpyLab, 'browser-lab-saxpy');
assert.equal(emitted.ok, true);
assert.match(emitted.module_b64, /^[A-Za-z0-9+/]+={0,2}$/);

// The core correctness promise: the emitted module prints the interpreter's
// lines and folds to the interpreter's checksum.
for (const [name, source] of [['saxpy loop', saxpyLab], ['recursion', fibLab], ['mixed', mixedLab]]) {
  const reference = await interpret(source);
  const compiled = await runCompiled(source, name);
  assert.equal(reference.ok, true, `${name}: interpreter must run`);
  assert.equal(compiled.ok, true, `${name}: module must run (${compiled.message})`);
  assert.deepEqual(Array.from(compiled.printed), Array.from(reference.printed), `${name}: printed lines differ`);
  assert.equal(compiled.checksum, reference.checksum, `${name}: checksum differs`);
  assert.equal(typeof reference.alloc_bytes, 'number', `${name}: interpreter must report alloc_bytes`);
  assert.ok(Number.isInteger(reference.alloc_bytes) && reference.alloc_bytes >= 0, `${name}: alloc bytes must be a count`);
  assert.equal(typeof compiled.heap_bytes, 'number');
  assert.ok(compiled.heap_bytes >= 0, `${name}: heap bytes must not be negative`);
  assert.ok(Number.isFinite(compiled.elapsed_ms) && compiled.elapsed_ms >= 0);
  console.log(`parity: ${name} — ${reference.printed.length} line(s), checksum ${reference.checksum}, `
    + `interpreter alloc_bytes ${reference.alloc_bytes}, module heap_bytes ${compiled.heap_bytes}`);
}

// The saxpy program's two arrays make the interpreter's allocation count real.
const saxpyAllocs = (await interpret(saxpyLab)).alloc_bytes;
assert.ok(saxpyAllocs > 0, `the allocating program must report bytes, got ${saxpyAllocs}`);

// Traps abort the compiled run with the reference engine's own wording.
for (const [name, source, pattern] of [
  ['bounds trap', 'fn main() {\n    print(1);\n    print(2);\n    let a: [i64] = zeros(2);\n    print(a[7]);\n}\n', /index 7 out of bounds for array of length 2/],
  ['negative zeros', 'fn main() { let a: [i64] = zeros(-2); }', /zeros\(-2\): array length must be non-negative/],
]) {
  const reference = await interpret(source);
  const compiled = await runCompiled(source, name);
  assert.equal(reference.ok, false, `${name}: interpreter must fail`);
  assert.match(reference.message, pattern);
  assert.equal(compiled.ok, false, `${name}: module must fail`);
  assert.match(compiled.message, pattern);
  assert.deepEqual(Array.from(compiled.printed), Array.from(reference.printed), `${name}: partial output differs`);
  // The interpreter appends its line number; the trap message is byte-identical
  // to what that engine reports, minus the location suffix.
  assert.equal(compiled.message, `runtime error: ${reference.message.replace(/ \(line \d+\)$/, '')}`);
  console.log(`trap parity: ${name} — "${compiled.message}" after ${reference.printed.length} line(s)`);
}

// A refused compilation is evidence, not an exception.
const rejected = await runCompiled('fn main() { print(1);', 'broken');
assert.equal(rejected.ok, false);
assert.equal(typeof rejected.message, 'string');

// The client-side size bound fires before a worker is spent.
const oversized = 'fn main() { let a: [i64] = zeros(65536); }'.padEnd(65 * 1024, ' ');
await assert.rejects(buildArtifact(oversized, 'too big'), /exceeds 65536 bytes/);
await assert.rejects(emitModule(oversized, 'too big'), /exceeds 65536 bytes/);

const measured = await measureCompiled(measureLab, 'browser-lab-measure');
assert.equal(measured.result.ok, true);
assert.equal(measured.samples.length, 7);
assert.ok(measured.samples.every(ms => Number.isFinite(ms) && ms > 0), `samples must be positive, got ${measured.samples}`);
assert.ok(measured.medianMs > 0);
const ordered = [...measured.samples].sort((a, b) => a - b);
assert.equal(measured.medianMs, ordered[3]);
const measuredReference = await interpret(measureLab);
assert.equal(measured.checksum, measuredReference.checksum, 'measured checksum must match the interpreter');
assert.deepEqual(Array.from(measured.printed), Array.from(measuredReference.printed));
assert.equal(measured.heapBytes, measured.result.heap_bytes);
// The repetition policy moved into the bridge, and the worker really used it.
assert.equal(repetitionsFor(0), 32);
assert.equal(repetitionsFor(9999), 32);
assert.equal(repetitionsFor(10_000), 8);
assert.equal(repetitionsFor(99_999), 8);
assert.equal(repetitionsFor(100_000), 2);
const repeats = labWorkers.flatMap(worker => worker.messages);
assert.ok(repeats.some(message => message.type === 'runModuleRepeat' && message.times === 32),
  'the measured program must run 32 repetitions per sample');
assert.equal(labTimers.size, 0, 'every bridge request must clear its deadline');

const index = await benchIndex();
assert.ok(index && typeof index === 'object', 'bench index must be a JSON document');
assert.ok(Array.isArray(index.kernels) && index.kernels.length > 0, 'bench index must carry kernels');
assert.ok(index.kernels.every(kernel => typeof kernel.name === 'string' && Array.isArray(kernel.sizes)),
  'every kernel entry must name itself and list its measured sizes');

console.log('Browser WASM worker result/error/timeout/retry/isolation/asset checks passed.');
console.log('Compiled pipeline: artifact, emit, run, measure, and bench-index checks passed.');
