const WORKER_DEADLINE_MS = 5000;
const MAX_SOURCE_BYTES = 64 * 1024;
const COMPILED_SAMPLES = 7;

/** Conservative UX filter, not a safety bound: the worker deadline stops runs
 *  even when this deliberately simple source estimate misses their cost.
 *  Loop subtrees nest (multiply); sibling loops add. Scalar literals are not
 *  work — an allocation charges its element count, per enclosing iteration,
 *  like the server estimator. Unknown loop trips default to 1000. */
export function estimatedBrowserWork(source) {
  const body = String(source)
    .replace(/"[^"\r\n]*"/g, '""')
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/\/\/[^\r\n]*/g, '');
  const open = []; // loops with unclosed bodies: { trips, child, bodyDepth }
  let depth = 0;
  let total = 0;
  let alloc = 0;
  const closeLoop = () => {
    const { trips, child } = open.pop();
    const cost = trips * (1 + child);
    if (open.length) open[open.length - 1].child += cost;
    else total += cost;
  };
  for (const line of body.split('\n')) {
    const trimmed = line.trimStart();
    const forCount = (trimmed.match(/\bfor\b/g) || []).length;
    for (let n = 0; n < forCount; n += 1) {
      // Literal `in A..B` bounds give real trips; anything else stays
      // conservative at the 1000-trip default.
      const range = trimmed.match(/\bin\s+(-?\d+)\s*\.\.\s*(-?\d+)/);
      const trips = range
        ? Math.max(Math.abs(Number(range[2]) - Number(range[1])), 1)
        : 1000;
      open.push({ trips, child: 0, bodyDepth: 0 });
    }
    if (trimmed.includes('zeros(')) {
      let size = 0;
      for (const token of trimmed.split(/[^0-9]+/)) {
        if (token) size = Math.max(size, Number(token) || 0);
      }
      if (open.length) open[open.length - 1].child += size;
      else alloc = Math.max(alloc, size);
    }
    const opens = (line.match(/\{/g) || []).length;
    const closes = (line.match(/\}/g) || []).length;
    depth += opens - closes;
    if (forCount) {
      if (opens > closes) {
        // Multi-line bodies: close each when depth drops below its opening.
        for (let n = open.length - forCount; n < open.length; n += 1) {
          open[n].bodyDepth = depth;
        }
      } else {
        // Body opened and closed on this line.
        for (let n = 0; n < forCount; n += 1) closeLoop();
      }
    }
    while (open.length && depth < open[open.length - 1].bodyDepth) closeLoop();
  }
  while (open.length) closeLoop();
  return Math.max(total, alloc, 1000);
}

/** Repetitions per timed sample, from the same work bands the Observatory
 *  interpreter path uses: cheap programs repeat most, expensive ones least. */
export function repetitionsFor(estimatedWork) {
  if (estimatedWork < 10_000) return 32;
  if (estimatedWork < 100_000) return 8;
  return 2;
}

/** The host wasm rejects sources it cannot hold; refuse before spending a
 *  worker so the caller sees the same bound as a successful run would. */
function requireSourceSize(source) {
  const bytes = new TextEncoder().encode(source).byteLength;
  if (bytes > MAX_SOURCE_BYTES) {
    throw new Error(`source exceeds ${MAX_SOURCE_BYTES} bytes`);
  }
}

function requestWorker(message) {
  return new Promise((resolve, reject) => {
    if (typeof Worker !== 'function') {
      reject(new Error('Web Workers are unavailable; browser execution is disabled'));
      return;
    }
    // ponytail: fresh workers isolate concurrent output and failed runtimes;
    // consider an idle-worker pool only if startup overhead becomes a UI problem.
    const worker = new Worker(new URL('./helix-worker.js', import.meta.url), { type: 'module' });
    let settled = false;
    const finish = (error, result) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      worker.onmessage = worker.onerror = worker.onmessageerror = null;
      worker.terminate();
      if (error) reject(error);
      else resolve(result);
    };
    // Covers loading as well as execution; never fall back to the main thread.
    const timer = setTimeout(() => finish(new Error(
      `Browser interpreter timed out after ${WORKER_DEADLINE_MS} ms; worker terminated. Try a smaller input.`,
    )), WORKER_DEADLINE_MS);
    worker.onmessage = ({ data }) => {
      if (data?.type === 'ready' && message.type === 'load') finish(null);
      else if (data?.type === 'result' && message.type === 'run' && data.result) finish(null, data.result);
      else if (data?.type === 'applied' && message.type === 'apply') finish(null, data.json);
      else if (data?.type === 'moduleResult'
        && (message.type === 'runModule' || message.type === 'runModuleRepeat')
        && data.result) finish(null, data.result);
      else finish(new Error(data?.error || 'Browser interpreter worker returned an invalid response'));
    };
    worker.onerror = (event) => finish(new Error(event.message || 'Browser interpreter worker failed'));
    worker.onmessageerror = () => finish(new Error('Browser interpreter worker response could not be read'));
    try {
      worker.postMessage(message);
    } catch (error) {
      finish(error);
    }
  });
}

/** Checks that the raw-WASM bridge can load in a worker. Resolves without raw
 *  exports: WASM memory and execution must never escape onto the main thread. */
export function loadHelixWasm() {
  return requestWorker({ type: 'load' });
}

/** Executes the real reference interpreter in an isolated worker, terminated
 *  on completion, error or deadline. elapsedMs measures the per-call bridge and
 *  interpreter work inside the worker, excluding startup and message delivery.
 *  Native JIT and parallel execution intentionally do not cross this boundary. */
export function runHelixInBrowser(source) {
  return requestWorker({ type: 'run', source });
}

/** The compiler's full artifact for this source, built by the same pipeline
 *  the Observatory server uses (schema 1). */
export async function buildArtifact(source, example) {
  requireSourceSize(source);
  return requestWorker({ type: 'apply', op: 'build', source, example });
}

/** Emits the standalone WebAssembly module for this source. */
export async function emitModule(source, example) {
  requireSourceSize(source);
  return requestWorker({ type: 'apply', op: 'emit', source, example });
}

/** Runs the emitted module once and returns the module's own evidence:
 *  printed lines, checksum, allocated heap bytes, and elapsed milliseconds. */
async function runEmitted(moduleB64) {
  return requestWorker({ type: 'runModule', module_b64: moduleB64 });
}

async function emitOrThrow(source, example) {
  const emitted = await emitModule(source, example);
  if (!emitted?.ok) throw new Error(emitted?.message || 'HELIX wasm emitter rejected this source');
  if (typeof emitted.module_b64 !== 'string' || !emitted.module_b64) {
    throw new Error('HELIX wasm emitter returned no module for this source');
  }
  return emitted.module_b64;
}

/** The compiled pipeline end to end: emit, then execute the module in a
 *  worker. A trap, a refused emission, or a failed worker all come back as
 *  `{ok:false, message}` with the lines printed before the failure. */
export async function runCompiled(source, example) {
  requireSourceSize(source);
  try {
    const moduleB64 = await emitOrThrow(source, example);
    return await runEmitted(moduleB64);
  } catch (error) {
    return { ok: false, message: error instanceof Error ? error.message : String(error) };
  }
}

/** Measures the compiled pipeline the way the Observatory measures the
 *  interpreter: one warmup run, then seven samples of `repetitionsFor(work)`
 *  repetitions each, timed inside the worker against `hx_run` only. Worker
 *  startup, transport, instantiation, and emission are excluded. */
export async function measureCompiled(source, example) {
  requireSourceSize(source);
  const moduleB64 = await emitOrThrow(source, example);
  const warmup = await runEmitted(moduleB64);
  if (!warmup.ok) throw new Error(warmup.message || 'compiled module run failed');
  const repetitions = repetitionsFor(estimatedBrowserWork(source));
  const samples = [];
  for (let sample = 0; sample < COMPILED_SAMPLES; sample += 1) {
    const measured = await requestWorker({
      type: 'runModuleRepeat',
      module_b64: moduleB64,
      times: repetitions,
    });
    if (!measured?.ok) throw new Error(measured?.message || 'compiled module failed during measurement');
    if (measured.checksum !== warmup.checksum
      || JSON.stringify(measured.printed) !== JSON.stringify(warmup.printed)) {
      throw new Error('compiled module produced non-deterministic output across samples');
    }
    if (!Array.isArray(measured.elapsed_ms)
      || measured.elapsed_ms.length !== repetitions
      || !measured.elapsed_ms.every(Number.isFinite)) {
      throw new Error('compiled module did not provide one valid execution time per repetition');
    }
    samples.push(measured.elapsed_ms.reduce((sum, ms) => sum + ms, 0) / repetitions);
  }
  const ordered = [...samples].sort((a, b) => a - b);
  return {
    result: warmup,
    samples,
    medianMs: ordered[Math.floor(ordered.length / 2)],
    checksum: warmup.checksum,
    printed: warmup.printed,
    heapBytes: warmup.heap_bytes,
  };
}

/** The recorded benchmark campaign index, straight from the host wasm. */
export function benchIndex() {
  return requestWorker({ type: 'apply', op: 'bench_index' });
}
