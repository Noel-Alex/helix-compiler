#!/usr/bin/env node
// Offline figure generation from HELIX benchmark result JSONs.
// Node port of the original plot_bench.py so the figures regenerate on
// machines without Python; output is byte-comparable in structure.
//
// Usage:
//     node tools/plot_bench.mjs docs/benchmarks/data/campaign.json -o docs/benchmarks/figs

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';

function niceTicks(maxV, n = 5) {
  if (maxV <= 0) return [0];
  const raw = maxV / n;
  const mag = 10 ** Math.floor(Math.log10(raw));
  let step = mag;
  for (const m of [1, 2, 5, 10]) {
    step = m * mag;
    if (step >= raw) break;
  }
  const top = Math.ceil(maxV / step) * step;
  const out = [];
  for (let i = 0; i * step <= top + 1e-9; i += 1) out.push(i * step);
  return out;
}

// Python's {:g}: shortest representation, strips trailing zeros.
const g = v => {
  if (!Number.isFinite(v)) return String(v);
  if (Number.isInteger(v)) return String(v);
  return String(Number(v.toPrecision(6)));
};

function speedupFigure(efficiency, title) {
  const w = 640, h = 400;
  const padL = 60, padB = 50, padT = 40, padR = 20;
  const xs = efficiency.map(e => e.threads);
  const ys = efficiency.map(e => e.speedup);
  const ticks = niceTicks(Math.max(Math.max(...ys), 2.0));
  const ymax = ticks[ticks.length - 1];
  const xmax = Math.max(...xs);
  const px = x => padL + (x / xmax) * (w - padL - padR);
  const py = y => h - padB - (y / ymax) * (h - padB - padT);
  const f1 = v => v.toFixed(1);

  const parts = [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" font-family="ui-monospace,Consolas" font-size="12">`,
    `<text x="${w / 2}" y="20" text-anchor="middle" fill="#e6edf3">${title}</text>`,
    `<rect width="${w}" height="${h}" fill="#161b22"/>`,
  ];
  for (const t of ticks) {
    const y = py(t);
    parts.push(
      `<line x1="${padL}" y1="${f1(y)}" x2="${w - padR}" y2="${f1(y)}" stroke="#30363d"/>` +
      `<text x="${padL - 8}" y="${f1(y + 4)}" text-anchor="end" fill="#8b949e">${g(t)}</text>`
    );
  }
  parts.push(
    `<line x1="${f1(px(1))}" y1="${f1(py(1))}" x2="${f1(px(xmax))}" y2="${f1(py(xmax))}" ` +
    'stroke="#E5D75A" stroke-dasharray="6 4" stroke-width="1.5"/>'
  );
  const pts = efficiency.map(e => `${f1(px(e.threads))},${f1(py(e.speedup))}`).join(' ');
  parts.push(`<polyline points="${pts}" fill="none" stroke="#00C896" stroke-width="2.5"/>`);
  for (const e of efficiency) {
    parts.push(
      `<circle cx="${f1(px(e.threads))}" cy="${f1(py(e.speedup))}" r="4" fill="#00C896"/>` +
      `<text x="${f1(px(e.threads))}" y="${f1(py(e.speedup) - 10)}" ` +
      `text-anchor="middle" fill="#e6edf3">${e.speedup.toFixed(1)}x</text>`
    );
  }
  for (const e of efficiency) {
    parts.push(
      `<text x="${f1(px(e.threads))}" y="${h - padB + 18}" text-anchor="middle" ` +
      `fill="#8b949e">${e.threads}T</text>`
    );
  }
  parts.push(`<text x="${w / 2}" y="${h - 8}" text-anchor="middle" fill="#8b949e">threads</text>`);
  parts.push('</svg>');
  return parts.join('\n');
}

function barsFigure(variants, title) {
  const w = 720, barH = 34, gap = 14, padL = 200, padR = 90;
  const padTop = 50;
  const h = padTop + variants.length * (barH + gap) + 30;
  const vmax = Math.max(...variants.map(v => v.median_ms));
  const parts = [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" ` +
    'font-family="ui-monospace,Consolas" font-size="13">',
    `<rect width="${w}" height="${h}" fill="#161b22"/>`,
    `<text x="${w / 2}" y="26" text-anchor="middle" fill="#e6edf3">${title}</text>`,
  ];
  const colors = ['#E8630A', '#8b949e', '#56B4E9', '#00C896'];
  variants.forEach((v, i) => {
    const y = padTop + i * (barH + gap);
    const bw = Math.max(2.0, (v.median_ms / vmax) * (w - padL - padR));
    const color = colors[Math.min(i, colors.length - 1)];
    const label = `${v.name} (${v.median_ms.toFixed(2)} ms)`;
    parts.push(`<text x="${padL - 10}" y="${(y + barH * 0.7).toFixed(1)}" text-anchor="end" fill="#e6edf3">${label}</text>`);
    parts.push(
      `<rect x="${padL}" y="${y}" width="${bw.toFixed(1)}" height="${barH}" fill="${color}" rx="3">` +
      `<animate attributeName="width" from="0" to="${bw.toFixed(1)}" dur="0.5s" fill="freeze"/></rect>`
    );
  });
  parts.push('</svg>');
  return parts.join('\n');
}

function main() {
  const args = process.argv.slice(2);
  const resultsJson = args[0];
  const outIdx = args.indexOf('-o');
  const outdir = (outIdx >= 0 && args[outIdx + 1]) || 'docs/benchmarks/figs';
  const data = JSON.parse(readFileSync(resultsJson, 'utf8'));
  mkdirSync(outdir, { recursive: true });

  const kernels = data.points || data.kernels || [data];
  const written = [];
  for (const k of kernels) {
    const name = k.kernel || 'kernel';
    if (k.efficiency && k.efficiency.length) {
      writeFileSync(path.join(outdir, `${name}_speedup.svg`), speedupFigure(k.efficiency, `${name}: speedup vs threads`));
      written.push(`${name}_speedup.svg`);
    }
    if (k.variants && k.variants.length) {
      writeFileSync(path.join(outdir, `${name}_bars.svg`), barsFigure(k.variants, `${name}: median time by execution tier`));
      written.push(`${name}_bars.svg`);
    }
  }
  console.log('wrote:', written.length ? written.join(', ') : '(nothing)');
}

main();
