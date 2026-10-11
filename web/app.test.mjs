import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';

// Exercise the shipped script without booting a browser or adding a DOM dependency.
const source = await readFile(new URL('./app.js', import.meta.url), 'utf8');
const element = (tagName = 'div') => ({
  tagName, children: [], dataset: {}, attributes: {}, classes: new Set(), value: '',
  disabled: false, style: {}, listeners: {},
  get childElementCount() { return this.children.length; },
  append(...children) { this.children.push(...children); },
  appendChild(child) { child.parentElement = this; this.children.push(child); },
  getBoundingClientRect() { return { left: 0, top: 0, width: 100, height: 40 }; },
  prepend(child) { this.children.unshift(child); },
  insertBefore(child, next) {
    const index = this.children.indexOf(next);
    this.children.splice(index < 0 ? this.children.length : index, 0, child);
  },
  set textContent(value) { this.text = value; this.children = []; },
  get textContent() { return this.text ?? ''; },
  set className(value) { this.classes = new Set(value.split(/\s+/)); },
  get className() { return [...this.classes].join(' '); },
  setAttribute(name, value) { this.attributes[name] = String(value); },
  addEventListener(name, fn) { this.listeners[name] = fn; },
  get classList() {
    return {
      add: (...names) => names.forEach(name => this.classes.add(name)),
      remove: (...names) => names.forEach(name => this.classes.delete(name)),
      toggle: (name, on = !this.classes.has(name)) => {
        if (on) this.classes.add(name); else this.classes.delete(name);
        return on;
      },
    };
  },
});
const list = element();
const context = vm.createContext({
  TextEncoder,
  document: { querySelector: () => list, createElement: element },
  fetch: async () => ({ ok: true, json: async () => [
    'casts_demo', 'count_primes_sieve', 'div_guard', 'recurrence_reject',
    'scale', 'ssa_demo', 'type_errors', 'new_example',
  ] }),
});
vm.runInContext(source.replace(/\nboot\(\);\s*$/, ''), context);
await vm.runInContext('loadExamples()', context);
let heading;
const names = [];
for (const child of list.children) {
  if (child.className === 'example-group') {
    heading = child.innerHTML;
  } else {
    names.push(child.dataset.example);
    const label = vm.runInContext(`exampleGroup(${JSON.stringify(child.dataset.example)}).label`, context);
    assert.ok(heading.includes(`<span>${label}</span>`), `${child.dataset.example} is under the wrong heading`);
  }
}
assert.equal(names.length, 8);
assert.equal(new Set(names).size, 8);
console.log('Example grouping checks passed.');

// Resolve competing actions explicitly; no timers, browser, or copied handlers.
const run = code => vm.runInContext(code, context);
const deferred = () => {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
};
const items = list.children.filter(child => child.dataset.example);
const controls = {
  '#custom-src': element(), '#btn-compile': element(), '#bench-measure': element(),
};
context.CSS = { escape: value => value };
context.document.querySelectorAll = () => items;
context.document.querySelector = selector => controls[selector]
  ?? items.find(item => selector.endsWith(`[data-example="${item.dataset.example}"]`));
const requests = [], measurements = [], effects = [];
let measurementStarted;
context.fetchJSON = (path, opts) => {
  const pending = deferred();
  pending.path = path;
  pending.body = opts?.body;
  requests.push(pending);
  return pending.promise;
};
const measureBrowserSource = run('measureBrowserSource');
context.measureBrowserSource = () => {
  const pending = deferred();
  measurements.push(pending);
  measurementStarted?.resolve();
  return pending.promise;
};
const renderBench = run('renderBench');
for (const name of ['renderAll', 'renderBench', 'hydrateBrowserExecution', 'gotoPhase', 'toast']) {
  context[name] = (...args) => effects.push([name, ...args]);
}
const state = run('state');
const reset = (example = 'scale') => {
  context.initial = { example, source: 'original source' };
  run('setArtifact(initial, initial.example)');
  controls['#custom-src'].value = '';
  controls['#btn-compile'].textContent = 'Run pipeline';
  effects.length = 0;
  return context.initial;
};
const assertSelection = name => {
  assert.equal(state.exampleName, name);
  assert.equal(state.fromCustom, name === '<adhoc>');
  for (const item of items) {
    const selected = item.dataset.example === name;
    assert.equal(item.classes.has('active'), selected);
    assert.equal(item.attributes['aria-pressed'], String(selected));
  }
};

// Both action orders, each with both response orders: the latest intent wins.
for (const latest of ['example', 'compile']) {
  for (const latestFinishesFirst of [false, true]) {
    const initial = reset();
    const start = kind => {
      if (kind === 'compile') controls['#custom-src'].value = 'custom source';
      const done = run(kind === 'compile' ? 'compileCustom()' : "selectExample('ssa_demo')");
      return {
        done, pending: requests.shift(),
        artifact: { example: kind === 'compile' ? '<adhoc>' : 'ssa_demo', source: kind },
      };
    };
    const older = start(latest === 'example' ? 'compile' : 'example');
    const newer = start(latest);
    const order = latestFinishesFirst ? [newer, older] : [older, newer];
    for (const action of order) {
      const effectCount = effects.length;
      action.pending.resolve(action.artifact);
      await action.done;
      if (action === older) {
        assert.equal(state.artifact, latestFinishesFirst ? newer.artifact : initial);
        assert.equal(effects.length, effectCount, 'stale response must not render or toast');
      }
    }
    assert.equal(state.artifact, newer.artifact);
    assertSelection(newer.artifact.example);
    assert.equal(controls['#btn-compile'].disabled, false);
    assert.equal(controls['#btn-compile'].textContent, 'Run pipeline');
  }
}

// Only the latest compile may release the control, regardless of completion
// order or whether the newest response succeeds.
for (const latestFinishesFirst of [false, true]) {
  for (const fails of [false, true]) {
    reset('<adhoc>');
    const older = run('compileCustom()');
    const oldRequest = requests.shift();
    const newer = run('compileCustom()');
    const newRequest = requests.shift();
    const finishNew = async () => {
      if (fails) newRequest.reject(new Error('new compile failed'));
      else newRequest.resolve({ example: '<adhoc>', source: 'newer' });
      await newer;
      assert.equal(controls['#btn-compile'].disabled, false);
      assert.equal(controls['#btn-compile'].textContent, 'Run pipeline');
    };
    if (latestFinishesFirst) await finishNew();
    oldRequest.resolve({ example: '<adhoc>', source: 'older' });
    await older;
    assert.equal(controls['#btn-compile'].disabled, !latestFinishesFirst);
    assert.equal(controls['#btn-compile'].textContent, latestFinishesFirst ? 'Run pipeline' : 'compiling…');
    if (!latestFinishesFirst) await finishNew();
  }
}
console.log('Overlapping compile controls follow latest intent checks passed.');

// Selection is transactional: failures retain the previous artifact, draft,
// and accessible selection. Success clears only a draft unchanged in flight.
for (const previous of ['scale', '<adhoc>']) {
  for (const fails of [false, true]) {
    for (const editsInFlight of [false, true]) {
      const original = reset(previous);
      controls['#custom-src'].value = 'saved draft';
      const selecting = run("selectExample('ssa_demo')");
      const pending = requests.shift();
      assertSelection(previous);
      assert.equal(controls['#custom-src'].value, 'saved draft');
      if (editsInFlight) controls['#custom-src'].value = 'newer draft';
      if (fails) pending.reject(new Error('selection failed'));
      else pending.resolve({ example: 'ssa_demo', source: 'selected' });
      await selecting;
      assertSelection(fails ? previous : 'ssa_demo');
      if (fails) assert.equal(state.artifact, original);
      assert.equal(controls['#custom-src'].value, editsInFlight ? 'newer draft' : fails ? 'saved draft' : '');
    }
  }
}
console.log('Failed selection and in-flight editor preservation checks passed.');

const measured = { result: { ok: true, engine: 'interp-wasm', printed: ['42'], checksum: '42' }, samples: [1, 2], medianMs: 1.5 };
// A delayed WASM result must not even mutate its old, shared artifact object.
for (const fails of [false, true]) {
  const oldArtifact = reset('<adhoc>');
  const measuring = run('measureCustom()');
  const pending = measurements.shift();
  const selecting = run("selectExample('ssa_demo')");
  const selected = { example: 'ssa_demo', source: 'selected source' };
  requests.shift().resolve(selected);
  await selecting;
  const effectCount = effects.length;
  if (fails) pending.reject(new Error('stale WASM failure'));
  else pending.resolve(measured);
  await measuring;
  assert.equal(state.artifact, selected);
  assert.equal(Object.hasOwn(oldArtifact, 'exec'), false);
  assert.equal(Object.hasOwn(oldArtifact, 'bench'), false);
  assert.equal(state.benchMessage, '');
  assert.equal(state.browserRun, null);
  assert.equal(effects.length, effectCount, 'stale measurement must not render, navigate, or toast');
  assertSelection('ssa_demo');
  assert.equal(controls['#bench-measure'].disabled, false);
}

// Supersession during measurement's compile step also skips diagnostics/WASM.
for (const outcome of ['ok', 'diagnostics', 'error']) {
  reset();
  const measuring = run('measureCustom()');
  const pending = requests.shift();
  const selecting = run("selectExample('ssa_demo')");
  const selected = { example: 'ssa_demo', source: 'selected source' };
  requests.shift().resolve(selected);
  await selecting;
  const effectCount = effects.length;
  if (outcome === 'error') pending.reject(new Error('stale compile failure'));
  else pending.resolve({ example: '<adhoc>', source: 'old source', diags_sem: outcome === 'diagnostics' ? [{}] : [] });
  await measuring;
  assert.equal(state.artifact, selected);
  assert.equal(measurements.length, 0, 'stale compilation must not start WASM');
  assert.equal(state.benchMessage, '');
  assert.equal(effects.length, effectCount);
  assertSelection('ssa_demo');
}

// A current measurement still commits and clears example selection accessibly.
reset();
const measuring = run('measureCustom()');
const customArtifact = { example: '<adhoc>', source: 'original source' };
measurementStarted = deferred();
requests.shift().resolve(customArtifact);
await measurementStarted.promise;
measurements.shift().resolve(measured);
await measuring;
assert.equal(state.artifact, customArtifact);
assert.equal(Object.hasOwn(state.artifact, 'exec'), false, 'browser measurement must not invent observer evidence');
assert.equal(state.artifact.bench.variants[0].samples, measured.samples);
assert.equal(state.browserRun.result, measured.result);
assertSelection('<adhoc>');
assert.equal(requests.length, 0);
assert.equal(measurements.length, 0);
console.log('Artifact race and accessible selection checks passed (10 scenarios).');

// Probe evidence must not borrow the unrelated native campaign's provenance.
const comparison = element();
controls['#bench-comparison'] = comparison;
state.benchmarkCatalog = {
  timestamp_utc: '2020-01-01T00:00:00Z',
  examples: [{ name: 'ssa_demo' }],
  probe_provenance: {
    timestamp_utc: '2026-08-31T12:00:00Z', git_rev: 'probe-revision',
    machine: { cpu_brand: 'probe-cpu' },
  },
};
run("renderBenchComparison({kernel: 'ssa_demo', variants: [{name: 'interp-local', median_ms: 1, samples: [1]}]})");
const detail = comparison.children[1].children[2].textContent;
assert.match(detail, /2026-08-31.*probe-revision.*probe-cpu/);
assert.doesNotMatch(detail, /2020-01-01/);
state.benchmarkCatalog.probe_provenance = null;
comparison.children = [];
run("renderBenchComparison({kernel: 'ssa_demo', variants: []})");
assert.match(comparison.children[1].children[2].textContent, /date unavailable.*revision unavailable.*machine unavailable/);
console.log('Separate probe provenance checks passed.');

let workerElapsed = 0.25;
let workerCalls = 0;
context.loadBrowserRunner = async () => ({
  estimatedBrowserWork: () => 1000,
  runHelixInBrowser: async () => {
    workerCalls += 1;
    return { ok: true, checksum: '42', printed: ['42'], elapsedMs: workerElapsed };
  },
});
const timing = await measureBrowserSource('source');
assert.equal(workerCalls, 1 + 7 * 32);
assert.equal(timing.samples.length, 7);
assert.ok(timing.samples.every(ms => ms === 0.25));
assert.equal(timing.medianMs, 0.25);
workerElapsed = undefined;
await assert.rejects(measureBrowserSource('source'), /valid execution time/);
console.log('Worker-only benchmark timing checks passed.');

// Measuring a named example adopts <adhoc>; the editor must then hold the
// measured source or Recompile would post empty source.
{
  reset('scale');
  controls['#custom-src'].value = '';
  const measuring = run('measureCustom()');
  requests.shift().resolve({ example: '<adhoc>', source: 'original source', diags_sem: [] });
  await new Promise(r => setTimeout(r, 10));
  measurements.shift().resolve(measured);
  await measuring;
  assert.equal(state.exampleName, '<adhoc>');
  assert.equal(controls['#custom-src'].value, 'original source', 'editor must hold the measured source');
  const recompiling = run('compileCustom()');
  assert.deepEqual(JSON.parse(requests[0].body), { source: 'original source' }, 'Recompile must post the measured source');
  requests.shift().resolve({ example: '<adhoc>', source: 'original source', diags_sem: [] });
  await recompiling;
  console.log('Measure-then-recompile editor sync check passed.');
}

// Parser and lexer failures stay visible even when there is no token stream.
{
  controls['#diag-banner'] = element();
  controls['#source-view tbody'] = element();
  controls['#src-sub'] = element();
  controls['#src-show-diag'] = { checked: true };
  for (const tokens of [null, []]) {
    reset();
    const compiling = run('compileCustom()');
    const artifact = {
      source: '@', tokens, diags_sem: [],
      diags_lex: [{ span: { start: 0, end: 1 }, msg: 'unexpected <token>' }],
    };
    requests.shift().resolve(artifact);
    await compiling;
    context.diagnosticArtifact = artifact;
    run('renderDiagBanner(diagnosticArtifact); renderSource(diagnosticArtifact)');
    const banner = controls['#diag-banner'];
    assert.equal(banner.classes.has('hidden'), false);
    assert.match(banner.children.at(-2).textContent, /syntax error.*unexpected <token>/);
    assert.equal(run("stageAvailable(diagnosticArtifact, 'source')"), true);
    assert.equal(run("stageAvailable(diagnosticArtifact, 'tokens')"), tokens !== null);
    assert.equal(run("stageAvailable(diagnosticArtifact, 'bench')"), false);
    assert.ok(effects.some(([name, text]) => name === 'toast' && /syntax error/.test(text)));
    if (tokens) {
      const row = controls['#source-view tbody'].children.at(-1);
      assert.equal(row.classes.has('has-diag'), true);
      assert.ok(row.children[1].children.some(span => span.classes.has('diag-u')));
    }
  }
  run('renderDiagBanner({diags_sem: []})');
  assert.equal(controls['#diag-banner'].classes.has('hidden'), true);
  console.log('Syntax diagnostic banner, source, toast, and stage checks passed.');
}

// Real renderers: clearing textContent detaches children, just as in the browser.
context.document.createElementNS = (_, tag) => element(tag);
context.document.createTextNode = text => Object.assign(element('#text'), { textContent: text });
context.onNextFrame = fn => fn();
context.renderBench = renderBench;
for (const id of ['bench-chart', 'bench-eff', 'bench-empty', 'bench-kernel']) controls[`#${id}`] = element();
const querySelector = context.document.querySelector;
context.document.querySelector = selector => selector === '#bench-measure'
  ? controls['#bench-empty'].children.find(child => child.id === 'bench-measure')
  : querySelector(selector);
const measureControl = () => context.document.querySelector('#bench-measure');
state.benchmarkCatalog = null;

for (const failsAt of ['compile', 'WASM']) {
  reset(failsAt === 'compile' ? 'scale' : '<adhoc>');
  Object.assign(state.artifact, { tokens: [], ast: {} });
  run('renderBench(state.artifact)');
  const original = measureControl();
  assert.equal(original.disabled, false);
  const measuring = original.listeners.click();
  assert.notEqual(measureControl(), original, 'pending render must replace the old button');
  assert.equal(measureControl().disabled, true);
  const pending = (failsAt === 'compile' ? requests : measurements).shift();
  pending.reject(new Error(`${failsAt} unavailable`));
  await measuring;
  const retry = measureControl();
  assert.ok(retry, 'a failed measurement must render a replacement control');
  assert.equal(retry.disabled, false);
  assert.notEqual(retry, original);
  assert.match(controls['#bench-empty'].children[1].textContent, /measurement unavailable/);
  const retried = retry.listeners.click();
  if (failsAt === 'compile') {
    measurementStarted = deferred();
    requests.shift().resolve({ example: '<adhoc>', source: 'original source', tokens: [], ast: {} });
    await measurementStarted.promise;
  }
  assert.equal(measurements.length, 1, 'activating retry must start another WASM measurement');
  measurements.shift().resolve(measured);
  await retried;
  assert.equal(state.artifact.bench.variants[0].median_ms, measured.medianMs);
  assert.equal(state.benchPending, false);
  assert.equal(state.benchMessage, '');
}
console.log('Measurement failure renders an enabled working retry (compile and WASM) checks passed.');

// Both recovery routes use setArtifact and must clear failure/pending state.
for (const action of ['compileCustom()', "selectExample('ssa_demo')"]) {
  state.benchMessage = 'previous failure';
  state.benchPending = true;
  const recovering = run(action);
  requests.shift().resolve({ example: '<adhoc>', source: 'recovered', tokens: [], ast: {} });
  await recovering;
  run('renderBench(state.artifact)');
  assert.equal(state.benchMessage, '');
  assert.equal(measureControl().disabled, false);
}
console.log('Recompile and example selection restore measurement controls checks passed.');

// A superseding action that FAILS must still release a pending measurement's
// retry button (benchPending cleared, rerendered), in both completion orders
// and for both measurement outcomes.
{
  for (const action of ['compileCustom()', "selectExample('ssa_demo')"]) {
    for (const latestFinishesFirst of [false, true]) {
      const { promise: measurementGate, resolve: gateResolve } = deferred();
      context.measureBrowserSource = () => measurementGate;
      reset('<adhoc>');
      state.artifact.tokens = [];
      state.artifact.ast = {};
      run('renderBench(state.artifact)');
      const measuring = measureControl().listeners.click();
      const superseding = run(action);
      const failed = requests.shift();
      if (latestFinishesFirst) {
        failed.reject(new Error('superseding failure'));
        await superseding;
        gateResolve({ result: {}, samples: [1], medianMs: 1 });
      } else {
        gateResolve({ result: {}, samples: [1], medianMs: 1 });
        await measuring;
        failed.reject(new Error('superseding failure'));
        await superseding;
      }
      assert.equal(state.benchPending, false, `${action} failure must release pending state`);
      assert.equal(state.benchMessage, '');
      run('renderBench(state.artifact)');
      assert.equal(measureControl().disabled, false, 'retry control must be enabled after release');
    }
  }
  context.measureBrowserSource = () => {
    const pending = deferred();
    measurements.push(pending);
    measurementStarted?.resolve();
    return pending.promise;
  };
  console.log('Failed superseding actions release measurement pending state checks passed.');
}

for (const timings of [[0.2, 0.8, 30], [0.2, 0.8], [0, 0.2, 0.8], [0, 0], [0.2, 0.2], [1, 1]]) {
  state.benchLog = true;
  context.chartArtifact = { bench: { kernel: '<adhoc>', variants: timings.map((median_ms, i) => ({ name: `v${i}`, median_ms })) } };
  run('renderBench(chartArtifact)');
  const svg = controls['#bench-chart'].children[0];
  const bars = svg.children.filter(child => child.attributes.class?.includes('bar-rect'));
  const widths = bars.map(bar => Number(bar.attributes.width));
  assert.equal(widths.length, timings.length);
  assert.ok(widths.every(width => Number.isFinite(width) && width >= 1.5 && width <= 640));
  const sorted = [...timings].sort((a, b) => b - a);
  for (let i = 1; i < widths.length; i++) {
    if (sorted[i - 1] === sorted[i]) assert.equal(widths[i - 1], widths[i]);
    else assert.ok(widths[i - 1] > widths[i], `log bars must preserve timing order: ${timings}`);
  }
  const ticks = svg.children.filter(child => child.attributes.class === 'axis-line');
  assert.ok(ticks.length > 0, `log ticks must exist even below 1 ms: ${timings}`);
  for (const child of svg.children) {
    for (const key of ['x', 'x1', 'x2', 'y', 'y1', 'y2', 'width', 'height']) {
      if (key in child.attributes) {
        const value = Number(child.attributes[key]);
        assert.ok(Number.isFinite(value) && value >= 0 && value <= 960, `${key} must be finite and bounded`);
      }
    }
  }
  for (const tick of ticks) assert.ok(Number(tick.attributes.x1) >= 170 && Number(tick.attributes.x1) <= 810);
}
console.log('Log-scale mixed, sub-ms, zero, and equal timing geometry checks passed.');

for (const id of ['pass-list', 'opt-empty', 'opt-summary']) controls[`#${id}`] = element();
run(`renderPasses({passes: [[41,40],[40,38],[38,40],[40,40]].map(([insts_before, insts_after]) => ({
  name: 'pass', diff_stats: {insts_before, insts_after}
}))})`);
const cards = controls['#pass-list'].children;
assert.deepEqual(cards.map(card => card.children[0].children[3].children[0].textContent), ['−1', '−2', '+2', '±0']);
console.log('Optimization instruction deltas −1, −2, +2, ±0 checks passed.');
for (const [i, card] of cards.entries()) {
  const head = card.children[0];
  assert.equal(head.tagName, 'button');
  assert.equal(head.type, 'button');
  assert.equal(head.attributes['aria-expanded'], String(i === 0));
  head.listeners.click();
  assert.equal(head.attributes['aria-expanded'], String(i !== 0));
  assert.equal(card.classes.has('open'), i !== 0);
  head.listeners.click();
  assert.equal(head.attributes['aria-expanded'], String(i === 0));
}
console.log('OPT native button activation and aria-expanded checks passed.');

for (const id of ['ssa-body', 'ir-pre', 'ir-ssa', 'ssa-empty', 'ssa-fn-name', 'ssa-fn-tabs', 'ssa-side-by-side']) controls[`#${id}`] = element();
context.ssaArtifact = { ir_pre_ssa: { functions: [{ text: 'before' }] }, ir_ssa: { functions: [{ text: 'after' }] } };
for (const checked of [true, false]) {
  controls['#ssa-side-by-side'].checked = checked;
  run('renderSSA(ssaArtifact)');
  assert.equal(controls['#ssa-body'].classes.has('side-by-side'), checked);
}
console.log('SSA comparison applies the responsive side-by-side class checks passed.');

for (const id of ['cfg-edges-layer', 'cfg-nodes-layer', 'cfg-empty', 'cfg-svg']) controls[`#${id}`] = element();
context.cfgArtifact = { cfg: { functions: [
  { nodes: [{ id: 'entry', x: 20, y: 30, w: 200, h: 100 }, { id: 'exit', x: 1500, y: 1200, w: 240, h: 120 }],
    edges: [{ kind: 'backedge', points: [[1500, 1200], [-200, -100], [20, 30]] }] },
  { nodes: [{ id: 'entry', x: 0, y: 0, w: 100, h: 50 }], edges: [] },
  { nodes: [], edges: [] },
] } };
for (const [i, expected] of [[0, [-224, -124, 1988, 1468]], [1, [-24, -24, 148, 98]], [2, [-24, -24, 48, 48]]]) {
  state.cfgFn = i;
  run('renderCfg(cfgArtifact)');
  assert.deepEqual(controls['#cfg-svg'].attributes.viewBox.split(' ').map(Number), expected);
}
console.log('CFG viewBox covers nodes and routed edges and resets per function checks passed.');

// Execution readout renders the server's interactive interp timing sample and
// omits the line when there is no timing (budget-skipped or failed runs).
{
  const panelChildrenTexts = panel => panel.children.flatMap(child =>
    child.text !== undefined ? [child.text] : child.children.map(c => c.text ?? c.textContent ?? '')
  );
  const { execPanel } = (() => {
    const panel = element('section');
    context.document.querySelector = selector => selector === '#exec-readout' ? panel
      : (controls[selector] ?? items.find(item => selector.endsWith(`[data-example="${item.dataset.example}"]`)));
    return { execPanel: panel };
  })();
  context.execArtifact = {
    source: 'fn main() { print(1); }', tokens: [], ast: {},
    exec: { backend_used: 'interp', printed: ['1'], checksum: '0xabc', timings_ms: [2.94] },
  };
  run('renderExecReadout(execArtifact)');
  const codes = panelChildrenTexts(execPanel);
  assert.ok(codes.some(t => t === 'interp 2.9 ms'), `timing line missing: ${codes}`);
  // No timing (budget skip or error path) must not render a bogus line.
  context.execArtifact.exec.timings_ms = null;
  run('renderExecReadout(execArtifact)');
  assert.equal(panelChildrenTexts(execPanel).filter(t => /^interp [\d.]+ ms$/.test(t)).length, 0);
  // Sub-1 ms formatting keeps three decimals.
  context.execArtifact.exec.timings_ms = [0.421];
  run('renderExecReadout(execArtifact)');
  assert.ok(panelChildrenTexts(execPanel).includes('interp 0.421 ms'));
  console.log('Exec readout renders the interp timing sample checks passed.');

  // Measurement keeps native observer evidence separate from browser results,
  // including when /api/run skipped execution entirely.
  for (const recompile of [false, true]) {
    for (const observerChecksum of ['42', 'different', null]) {
      const original = reset(recompile ? 'scale' : '<adhoc>');
      const observer = observerChecksum === null ? null : {
        backend_used: 'interp', printed: [observerChecksum], checksum: observerChecksum, timings_ms: [0.421],
      };
      const artifact = recompile ? { example: '<adhoc>', source: original.source } : original;
      artifact.exec = observer;
      const observerBefore = JSON.stringify(observer);
      const measuring = run('measureCustom()');
      if (recompile) {
        measurementStarted = deferred();
        requests.shift().resolve(artifact);
        await measurementStarted.promise;
      }
      measurements.shift().resolve(measured);
      await measuring;
      assert.equal(state.artifact.exec, observer);
      assert.equal(JSON.stringify(observer), observerBefore, 'observer exec must remain unchanged');
      assert.equal(state.browserRun.result, measured.result);
      assert.equal(artifact.bench.variants[0].samples, measured.samples);
      assert.equal(artifact.bench.variants[0].median_ms, measured.medianMs);
      assert.equal(artifact.bench.variants[0].name, 'interp-wasm-local');
      run('renderExecReadout(state.artifact); renderBench(state.artifact)');
      const texts = panelChildrenTexts(execPanel);
      assert.ok(texts.includes('WASM RESULT'));
      assert.ok(texts.includes('42'), 'readout displays browser output, not observer output');
      const parity = texts.filter(t => /matches observer output|observer\/WASM output differs/.test(t));
      assert.deepEqual(parity, observerChecksum === null ? [] : [observerChecksum === '42'
        ? 'matches observer output' : 'observer/WASM output differs']);
      assert.equal(texts.includes('interp 0.421 ms'), observer !== null, 'only observer timing gets the interp label');
      assert.match(controls['#bench-kernel'].textContent, /local WASM interpreter · 2 measured samples/);
      assert.match(controls['#bench-comparison'].children[0].textContent, /Local measurement · WASM reference interpreter only/);
    }
  }
  console.log('Measurement preserves observer evidence, mismatch, absent exec, samples and labels checks passed.');
}

// The AST precedes the CFG in index.html; a bare .panel-viewport selects a
// hidden panel. Exercise the actual CFG node's hover listener and coordinates.
{
  const astPanel = element();
  const cfgPanel = element();
  cfgPanel.getBoundingClientRect = () => ({ left: 100, top: 200, width: 500, height: 400 });
  const previousQuery = context.document.querySelector;
  context.document.querySelector = selector => {
    if (selector === '.panel-viewport') return astPanel;
    if (selector === '[data-phase="cfg"] .panel-viewport') return cfgPanel;
    if (selector === '.cfg-hover-tip') return [...astPanel.children, ...cfgPanel.children]
      .find(child => child.className.includes('cfg-hover-tip'));
    return previousQuery(selector);
  };
  const resizeListeners = {};
  context.window = { addEventListener: (name, fn) => { resizeListeners[name] = fn; } };
  state.cfgFn = 0;
  run('renderCfg(cfgArtifact)');
  const node = controls['#cfg-nodes-layer'].children[0];
  node.listeners.mousemove({ clientX: 120, clientY: 230 });
  assert.equal(astPanel.childElementCount, 0);
  assert.equal(cfgPanel.childElementCount, 1);
  const tip = cfgPanel.children[0];
  assert.match(tip.innerHTML, /entry/);
  assert.equal(tip.style.left, '36px');
  assert.equal(tip.style.top, '44px');
  assert.equal(tip.classes.has('hidden'), false);
  node.listeners.mouseleave();
  assert.equal(tip.classes.has('hidden'), true);
  node.listeners.mousemove({ clientX: 580, clientY: 580 });
  assert.equal(cfgPanel.childElementCount, 1, 'reuse the same tooltip');
  assert.equal(tip.style.left, '368px');
  assert.equal(tip.style.top, '330px');
  resizeListeners.resize();
  assert.equal(tip.classes.has('hidden'), true);
  context.document.querySelector = previousQuery;
  console.log('CFG hover tooltip mounts in its visible panel and clamps coordinates checks passed.');
}

// These are the serialized Type/ScalarType variants from helix-syntax::ast,
// not the old assumed {Scalar: ...} wrapper. Exercise every AST label caller.
{
  for (const [ty, label] of [['I32', 'i32'], ['I64', 'i64'], ['F32', 'f32'], ['F64', 'f64'], ['Bool', 'bool'], ['Unit', '()']]) {
    context.typeValue = ty;
    assert.equal(run('labelType(typeValue)'), label);
    if (ty !== 'Unit') {
      context.typeValue = { Array: ty };
      assert.equal(run('labelType(typeValue)'), `[${label}]`);
    }
  }
  assert.equal(run('labelType(null)'), '?');
  const tree = run(`astToHierarchy({Program: {items: [
    {Const: {name: {name: 'N'}, ty: 'I64', value: {Int: 4}}},
    {Fn: {name: {name: 'main'}, params: [{name: {name: 'a'}, ty: {Array: 'F32'}}], ret: 'Unit',
      body: {stmts: [{Let: {name: {name: 'x'}, ty: 'I64', init: {Cast: [{IntLit: [1]}, 'I64']}}}]}}}
  ]}})`);
  const details = node => [node.detail, ...node.children.flatMap(details)];
  assert.deepEqual(details(tree).filter(value => value.includes('?')), []);
  for (const label of ['N: i64 = 4', 'a:[f32]', '()', 'i64']) assert.ok(details(tree).includes(label), label);
  console.log('Serde scalar, unit and array AST labels checks passed.');
}

// Measurement completion must not clobber edits typed while it ran, and a
// failed WASM run must not mix its error output with observer evidence.
{
  reset();
  const measuring = run('measureCustom()');
  const customArtifact = { example: '<adhoc>', source: 'original source' };
  measurementStarted = deferred();
  requests.shift().resolve(customArtifact);
  await measurementStarted.promise;
  controls['#custom-src'].value = 'draft typed while measuring';
  measurements.shift().resolve(measured);
  await measuring;
  assert.equal(controls['#custom-src'].value, 'draft typed while measuring',
    'in-flight editor edits are newer intent than the measured source');
  // Untouched editor keeps the measure-then-recompile sync.
  reset();
  const m2 = run('measureCustom()');
  measurementStarted = deferred();
  requests.shift().resolve({ example: '<adhoc>', source: 'original source' });
  await measurementStarted.promise;
  measurements.shift().resolve(measured);
  await m2;
  assert.equal(controls['#custom-src'].value, 'original source');
  console.log('Measurement draft-preservation checks passed.');

  // Failed browser run: observer evidence stays the single output source.
  const panelChildrenTexts = panel => panel.children.flatMap(child =>
    child.text !== undefined ? [child.text] : child.children.map(c => c.text ?? c.textContent ?? '')
  );
  controls['#exec-readout'] = element();
  // An earlier block re-routed querySelector to its own exec panel; restore
  // the controls-based routing for this block's assertions.
  context.document.querySelector = selector => controls[selector]
    ?? items.find(item => selector.endsWith(`[data-example="${item.dataset.example}"]`));
  context.initial = { example: 'ssa_demo', source: 'original source' };
  run('setArtifact(initial, initial.example)');
  context.execArtifact = {
    ...context.initial,
    exec: { backend_used: 'interp', printed: ['42'], checksum: '0x42', timings_ms: [1.5] },
  };
  run('setArtifact(execArtifact, "ssa_demo", {hydrate: false})');
  state.browserRun = { status: 'done', result: { ok: false, printed: [], checksum: null, engine: 'browser interpreter' } };
  run('renderExecReadout(execArtifact)');
  const texts = panelChildrenTexts(controls['#exec-readout']);
  assert.ok(texts.includes('42'), 'observer output shown on browser failure');
  assert.ok(texts.some(t => t === 'OBSERVER RESULT'), 'status names the observer evidence');
  assert.ok(!texts.includes('∅  program printed no lines'), 'observer lines not hidden by browser failure');
  state.browserRun = null;
  console.log('Browser-failure evidence separation checks passed.');
}
