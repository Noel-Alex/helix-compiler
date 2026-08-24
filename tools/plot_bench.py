#!/usr/bin/env python3
"""Offline figure generation from HELIX benchmark result JSONs.

Usage:
    python tools/plot_bench.py docs/benchmarks/data/<campaign>.json -o docs/benchmarks/figs

Produces SVG figures (speedup-vs-threads, efficiency curves, variant bars).
Kept out of the Rust binary deliberately: the harness stays dependency-free.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def nice_ticks(max_v: float, n: int = 5) -> list[float]:
    if max_v <= 0:
        return [0.0]
    raw = max_v / n
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 5, 10):
        step = m * mag
        if step >= raw:
            break
    top = math.ceil(max_v / step) * step
    return [i * step for i in range(int(top / step) + 1)]


def speedup_figure(efficiency: list[dict], title: str) -> str:
    w, h = 640, 400
    pad_l, pad_b, pad_t, pad_r = 60, 50, 40, 20
    xs = [e["threads"] for e in efficiency]
    ys = [e["speedup"] for e in efficiency]
    ymax = max(max(ys), 2.0)
    ticks = nice_ticks(ymax)
    ymax = ticks[-1]
    xmax = max(xs)

    def px(x: float) -> float:
        return pad_l + (x / xmax) * (w - pad_l - pad_r)

    def py(y: float) -> float:
        return h - pad_b - (y / ymax) * (h - pad_b - pad_t)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" font-family="ui-monospace,Consolas" font-size="12">',
        f'<text x="{w / 2}" y="20" text-anchor="middle" fill="#e6edf3">{title}</text>',
        f'<rect width="{w}" height="{h}" fill="#161b22"/>',
    ]
    # gridlines + y labels
    for t in ticks:
        y = py(t)
        parts.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{w - pad_r}" y2="{y:.1f}" stroke="#30363d"/>'
            f'<text x="{pad_l - 8}" y="{y + 4:.1f}" text-anchor="end" fill="#8b949e">{t:g}</text>'
        )
    # ideal line
    parts.append(
        f'<line x1="{px(1):.1f}" y1="{py(1):.1f}" x2="{px(xmax):.1f}" y2="{py(xmax):.1f}" '
        'stroke="#E5D75A" stroke-dasharray="6 4" stroke-width="1.5"/>'
    )
    # measured curve
    pts = " ".join(f"{px(e['threads']):.1f},{py(e['speedup']):.1f}" for e in efficiency)
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#00C896" stroke-width="2.5"/>')
    for e in efficiency:
        parts.append(
            f'<circle cx="{px(e["threads"]):.1f}" cy="{py(e["speedup"]):.1f}" r="4" fill="#00C896"/>'
            f'<text x="{px(e["threads"]):.1f}" y="{py(e["speedup"]) - 10:.1f}" '
            f'text-anchor="middle" fill="#e6edf3">{e["speedup"]:.1f}x</text>'
        )
    # x axis labels
    for e in efficiency:
        parts.append(
            f'<text x="{px(e["threads"]):.1f}" y="{h - pad_b + 18}" text-anchor="middle" '
            f'fill="#8b949e">{e["threads"]}T</text>'
        )
    parts.append(f'<text x="{w / 2}" y="{h - 8}" text-anchor="middle" fill="#8b949e">threads</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def bars_figure(variants: list[dict], title: str) -> str:
    w, bar_h, gap, pad_l, pad_r = 720, 34, 14, 200, 90
    h = pad_top = 50 + len(variants) * (bar_h + gap) + 30
    medians = [v["median_ms"] for v in variants]
    vmax = max(medians)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        'font-family="ui-monospace,Consolas" font-size="13">',
        f'<rect width="{w}" height="{h}" fill="#161b22"/>',
        f'<text x="{w / 2}" y="26" text-anchor="middle" fill="#e6edf3">{title}</text>',
    ]
    colors = ["#E8630A", "#8b949e", "#56B4E9", "#00C896"]
    for i, v in enumerate(variants):
        y = pad_top + i * (bar_h + gap)
        bw = max(2.0, (v["median_ms"] / vmax) * (w - pad_l - pad_r))
        color = colors[min(i, len(colors) - 1)]
        label = f'{v["name"]} ({v["median_ms"]:.2f} ms)'
        parts.append(f'<text x="{pad_l - 10}" y="{y + bar_h * 0.7:.1f}" text-anchor="end" fill="#e6edf3">{label}</text>')
        parts.append(
            f'<rect x="{pad_l}" y="{y}" width="{bw:.1f}" height="{bar_h}" fill="{color}" rx="3">'
            f'<animate attributeName="width" from="0" to="{bw:.1f}" dur="0.5s" fill="freeze"/></rect>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_json")
    ap.add_argument("-o", "--outdir", default="docs/benchmarks/figs")
    args = ap.parse_args()

    data = json.loads(Path(args.results_json).read_text(encoding="utf-8"))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    kernels = data.get("kernels", [data])
    written = []
    for k in kernels:
        name = k.get("kernel", "kernel")
        if eff := k.get("efficiency"):
            svg = speedup_figure(eff, f"{name}: speedup vs threads")
            p = outdir / f"{name}_speedup.svg"
            p.write_text(svg, encoding="utf-8")
            written.append(p.name)
        if variants := k.get("variants"):
            svg = bars_figure(variants, f"{name}: median time by execution tier")
            p = outdir / f"{name}_bars.svg"
            p.write_text(svg, encoding="utf-8")
            written.append(p.name)
    print("wrote:", ", ".join(written) if written else "(nothing — no efficiency/variants fields found)")


if __name__ == "__main__":
    main()
