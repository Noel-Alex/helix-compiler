import { runHelixInBrowser, estimatedBrowserWork } from '/wasm/helix-browser.js';

const PRESETS = {
  safe: `fn main() {
    let total = 0;
    for i in 0..6 {
        total = total + i;
    }
    print(total);
}`,
  reduction: `fn main() {
    let total = 0;
    for i in 1..5 {
        total = total + i;
    }
    print(total);
}`,
  trap: `fn main() {
    let values: [i64] = zeros(2);
    print(values[7]);
}`,
};

const BROWSER_WORK_BUDGET = 2_000_000;

const source = document.querySelector('#browser-source');
const runButton = document.querySelector('#browser-run');
const output = document.querySelector('#browser-output');
const diagnostics = document.querySelector('#browser-diagnostics');
const status = document.querySelector('#browser-status');

if (source && runButton && output && diagnostics && status) {
  document.querySelectorAll('[data-lab-preset]').forEach((button) => {
    button.addEventListener('click', () => {
      const key = button.dataset.labPreset;
      if (!key || !PRESETS[key]) return;
      source.value = PRESETS[key];
      document.querySelectorAll('[data-lab-preset]').forEach((item) => item.classList.remove('is-active'));
      button.classList.add('is-active');
      source.focus();
    });
  });

  runButton.addEventListener('click', async () => {
    const estimate = estimatedBrowserWork(source.value);
    if (estimate > BROWSER_WORK_BUDGET) {
      status.textContent = 'too large';
      status.dataset.state = 'error';
      diagnostics.textContent = `This source is estimated at ${estimate.toLocaleString('en-US')} interpreter steps, above the ${BROWSER_WORK_BUDGET.toLocaleString('en-US')} browser budget that keeps the page responsive. Try a smaller input.`;
      return;
    }

    runButton.disabled = true;
    runButton.textContent = 'Running…';
    status.textContent = 'loading wasm';
    status.dataset.state = 'running';
    diagnostics.textContent = 'The browser is loading the reference interpreter and checking the source.';

    try {
      const result = await runHelixInBrowser(source.value);
      const printed = Array.isArray(result.printed) ? result.printed : [];
      const detail = result.message || (result.ok ? 'program completed without printed output' : 'the program did not complete');
      output.textContent = printed.length ? `${printed.join('\n')}${result.ok ? '' : `\n${detail}`}` : detail;

      if (result.ok) {
        status.textContent = 'completed';
        status.dataset.state = 'ok';
        diagnostics.textContent = `${result.engine} · checksum ${result.checksum}. This is browser execution, not a native parallel benchmark.`;
      } else {
        status.textContent = result.status || 'stopped';
        status.dataset.state = 'error';
        const line = result.span?.line ? ` · line ${result.span.line}` : '';
        diagnostics.textContent = `${result.engine} · ${result.status || 'error'}${line}. The failure is returned instead of hidden.`;
      }
    } catch (error) {
      status.textContent = 'unavailable';
      status.dataset.state = 'error';
      output.textContent = error instanceof Error ? error.message : 'the WebAssembly asset could not be loaded';
      diagnostics.textContent = 'Browser execution failed or exceeded its deadline. The worker was stopped; you can edit the source and retry.';
    } finally {
      runButton.disabled = false;
      runButton.textContent = 'Run in WebAssembly';
    }
  });
}
