const MAX_FOLD_BYTES = 4096;
const B64_ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
const APPLY_FIELDS = ['source', 'example', 'tag', 'bits'];
const MODULE_EXPORTS = ['hx_run', 'hx_fold', 'hx_checksum', 'hx_heap_bytes'];

/** Calls the host wasm's generic JSON entry and returns the parsed document.
 *  Reads the result before freeing the input so only plain data leaves the
 *  worker, never the runtime's mutable output buffer. */
function applyJson(api, input) {
  const encoded = new TextEncoder().encode(JSON.stringify(input));
  const ptr = api.helix_alloc(encoded.byteLength);
  if (!ptr && encoded.byteLength) throw new Error('WASM could not allocate the input buffer');
  try {
    if (encoded.byteLength) new Uint8Array(api.memory.buffer, ptr, encoded.byteLength).set(encoded);
    api.helix_apply(ptr, encoded.byteLength);
    const outPtr = api.helix_output_ptr();
    const outLen = api.helix_output_len();
    if (!outLen) throw new Error(`WASM produced no JSON result for op ${JSON.stringify(input.op)}`);
    return JSON.parse(new TextDecoder().decode(new Uint8Array(api.memory.buffer, outPtr, outLen)));
  } finally {
    api.helix_free(ptr, encoded.byteLength);
  }
}

/** The legacy `run` reply: the host's `interp_run` document mapped onto the
 *  shape the Observatory readout consumes, plus the allocation count. */
function interpret(api, source) {
  const started = performance.now();
  const result = applyJson(api, { op: 'interp_run', source });
  const ok = result.ok === true;
  return {
    ok,
    status: ok ? 'ok' : 'error',
    printed: result.printed ?? [],
    checksum: result.checksum ?? null,
    message: result.message ?? null,
    span: result.span ?? null,
    engine: result.engine ?? 'HELIX reference interpreter / wasm32',
    alloc_bytes: result.alloc_bytes ?? null,
    elapsedMs: performance.now() - started,
  };
}

function applyInput(data) {
  if (typeof data?.op !== 'string' || !data.op) throw new Error('apply request needs an op string');
  const input = { op: data.op };
  for (const key of APPLY_FIELDS) {
    if (data[key] !== undefined && data[key] !== null) input[key] = data[key];
  }
  return input;
}

const asBigInt = value => (typeof value === 'bigint' ? value : BigInt(value));
const hex16 = value => `0x${BigInt.asUintN(64, asBigInt(value)).toString(16).padStart(16, '0')}`;
const ieeeBits = (value, width) => {
  const view = new DataView(new ArrayBuffer(width));
  if (width === 4) view.setFloat32(0, value, true);
  else view.setFloat64(0, value, true);
  return width === 4 ? BigInt(view.getUint32(0, true)) : view.getBigUint64(0, true);
};

/** The spec wording of every trap payload code, operand-interpolated where the
 *  code carries one. Identical to what the reference engine reports. */
function trapMessage(code, a, b) {
  switch (code) {
    case 0: return 'integer division by zero';
    case 1: return 'integer division overflow (i64::MIN / -1)';
    case 2: return `index ${b} out of bounds for array of length ${a}`;
    case 3: return `zeros(${a}): array length must be non-negative`;
    case 4: return 'call depth exceeded';
    case 5: return 'compiled run ran out of memory';
    default: return `unknown runtime trap (code ${code})`;
  }
}

/** Minimal standard base64 decoder; the module arrives as text over
 *  postMessage, so it never needs an ArrayBuffer transfer. */
function decodeBase64(text) {
  if (typeof text !== 'string') throw new Error('module_b64 must be a base64 string');
  const clean = text.replace(/[^A-Za-z0-9+/=]/g, '');
  if (clean.length % 4 !== 0) throw new Error('module_b64 is not padded to a four-character boundary');
  const bytes = [];
  let quartet = 0;
  let bits = 0;
  for (const character of clean) {
    if (character === '=') break;
    const value = B64_ALPHABET.indexOf(character);
    if (value < 0) throw new Error('module_b64 contains a character outside the base64 alphabet');
    quartet = (quartet << 6) | value;
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      bytes.push((quartet >> bits) & 0xff);
    }
  }
  return new Uint8Array(bytes);
}

/** Instantiates one emitted module and runs `hx_run` `times` times. Every run
 *  must agree; `timings` measures only `hx_run`, never instantiation. */
async function executeModule(api, moduleB64, times, printed) {
  const bytes = decodeBase64(moduleB64);
  if (!bytes.byteLength) throw new Error('compiled module is empty');
  const encoder = new TextEncoder();
  let memory = null;
  let exports = null;
  // Print hooks ask the host wasm for the canonical line text, which is also
  // the checksum stream: the interpreter folds each line plus its newline.
  const print = (tag, bits) => {
    const reply = applyJson(api, { op: 'format_line', tag, bits: bits.toString() });
    if (typeof reply.text !== 'string') throw new Error(`host formatter returned no text for ${tag}`);
    printed.push(reply.text.endsWith('\n') ? reply.text.slice(0, -1) : reply.text);
    const folded = reply.text.endsWith('\n') ? reply.text : `${reply.text}\n`;
    const line = encoder.encode(folded);
    if (line.byteLength > MAX_FOLD_BYTES) {
      throw new Error(`printed line of ${line.byteLength} bytes exceeds the ${MAX_FOLD_BYTES}-byte module scratch`);
    }
    new Uint8Array(memory.buffer, 0, line.byteLength).set(line);
    exports.hx_fold(0, line.byteLength);
  };
  const instantiated = await WebAssembly.instantiate(bytes, {
    env: {
      hx_print_i32: value => print('i32', BigInt(value >>> 0)),
      hx_print_i64: value => print('i64', BigInt.asUintN(64, asBigInt(value))),
      hx_print_f32: value => print('f32', ieeeBits(value, 4)),
      hx_print_f64: value => print('f64', ieeeBits(value, 8)),
      hx_print_bool: value => print('bool', value ? 1n : 0n),
      hx_trap: (code, a, b) => {
        throw new Error(`runtime error: ${trapMessage(code, asBigInt(a), asBigInt(b))}`);
      },
    },
  });
  exports = instantiated.instance.exports;
  memory = exports.memory;
  for (const name of MODULE_EXPORTS) {
    if (typeof exports[name] !== 'function') throw new Error(`compiled module does not export ${name}`);
  }
  const timings = [];
  let first = null;
  for (let run = 0; run < times; run += 1) {
    printed.length = 0;
    const started = performance.now();
    exports.hx_run();
    timings.push(performance.now() - started);
    const lines = printed.slice();
    const checksum = hex16(exports.hx_checksum());
    if (first && (first.checksum !== checksum
      || JSON.stringify(first.printed) !== JSON.stringify(lines))) {
      throw new Error('compiled module produced different output on a repeated run');
    }
    first = { printed: lines, checksum, heapBytes: Number(exports.hx_heap_bytes()) };
  }
  return { ...first, timings };
}

/** One emitted module, run `times` times; a trap keeps the lines printed so
 *  far, exactly like the interpreter's `printed_so_far`. */
async function moduleRun(api, moduleB64, times) {
  const printed = [];
  try {
    const run = await executeModule(api, moduleB64, times, printed);
    return {
      ok: true,
      printed: run.printed,
      checksum: run.checksum,
      heap_bytes: run.heapBytes,
      elapsed_ms: times === 1 ? run.timings[0] : run.timings,
    };
  } catch (error) {
    return { ok: false, message: error instanceof Error ? error.message : String(error), printed };
  }
}

// One request per worker; concurrent callers cannot share the ABI's LAST_RESULT.
self.onmessage = async ({ data }) => {
  self.onmessage = null;
  try {
    const response = await fetch(new URL('./helix-lab.wasm', import.meta.url));
    if (!response.ok) throw new Error(`WASM asset returned ${response.status}`);
    const { instance } = await WebAssembly.instantiate(await response.arrayBuffer(), {});
    const api = instance.exports;
    const times = Math.max(1, Number(data?.times) || 1);
    switch (data?.type) {
      case 'load':
        self.postMessage({ type: 'ready' });
        break;
      case 'run':
        self.postMessage({ type: 'result', result: interpret(api, data.source) });
        break;
      case 'apply':
        self.postMessage({ type: 'applied', json: applyJson(api, applyInput(data)) });
        break;
      case 'runModule':
        self.postMessage({ type: 'moduleResult', result: await moduleRun(api, data.module_b64, 1) });
        break;
      case 'runModuleRepeat':
        self.postMessage({ type: 'moduleResult', result: await moduleRun(api, data.module_b64, times) });
        break;
      default:
        self.postMessage({ type: 'error', error: `Unknown request type ${JSON.stringify(data?.type)}` });
    }
  } catch (error) {
    self.postMessage({ type: 'error', error: error instanceof Error ? error.message : String(error) });
  }
};
