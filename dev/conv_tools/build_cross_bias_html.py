"""Build a self-contained interactive HTML artifact for the cross-bias eval results.

Reads:
  - dev/conv_tools/cross_bias_eval/per_detector/single_bias_template/{basis}/{config}/heatmap_*.json
  - dev/conv_tools/cross_bias_eval/per_detector/single_bias_template/{basis}/{config}/per_bias_diagnostics.json
  - The cohort (for SBRS pid lists).

Writes:
  - dev/conv_tools/cross_bias_eval/index.html  (single file, all data embedded as JSON)

Open in browser: file:// path.
"""
from __future__ import annotations
import json
from pathlib import Path

from _data import load_eval_cohort

OUT_DIR = Path(__file__).parent / "cross_bias_eval"
ROOT = OUT_DIR / "per_detector/single_bias_template"
INDEX = OUT_DIR / "index.html"


def collect_data():
    cohort = load_eval_cohort(min_rs=5, tau_d=10)
    bias_ids = cohort.bias_ids
    bias_short = {str(b): cohort.bias_short[b] for b in bias_ids}

    # Per-bias diagnostics (constant across configs; pull from any one)
    diag_paths = list(ROOT.glob("*/*/per_bias_diagnostics.json"))
    per_bias = {}
    if diag_paths:
        per_bias = json.loads(diag_paths[0].read_text())

    # SBRS pid lists
    sbrs = {str(b): cohort.sbrs[b] for b in bias_ids}

    # Per (basis, config) — load all 5 metric heatmaps
    bases = []
    for basis_dir in sorted(ROOT.iterdir()):
        if not basis_dir.is_dir():
            continue
        for cfg_dir in sorted(basis_dir.iterdir()):
            if not cfg_dir.is_dir():
                continue
            entry = {
                "basis": basis_dir.name,
                "config": cfg_dir.name,
                "key": f"{basis_dir.name}__{cfg_dir.name}",
                "metrics": {},
                "fit_log": (cfg_dir / "fit_log.txt").read_text() if (cfg_dir / "fit_log.txt").exists() else "",
                "pngs": {},  # metric -> relative png path
            }
            for jp in sorted(cfg_dir.glob("heatmap_*.json")):
                metric_key = jp.stem.replace("heatmap_", "")
                d = json.loads(jp.read_text())
                # Strip down to the cells (metric + n_test + overlap) — the
                # per_bias_diagnostics block is duplicated, drop it from each metric entry.
                cells = {}
                for A, row in d["cells"].items():
                    cells[A] = {}
                    for B, v in row.items():
                        if v is None:
                            cells[A][B] = None
                        else:
                            cells[A][B] = {
                                "m":  v["metric"],
                                "n":  v["n_test_pids"],
                                "ov": v["pid_overlap_AB"],
                                "sk": v.get("n_test_pids_skipped", 0),
                            }
                entry["metrics"][metric_key] = {
                    "cells": cells,
                    "tau_d": d.get("tau_d"),
                    "nms_w": d.get("nms_w"),
                    "W_template": d.get("W_template"),
                }
                # PNG existence
                png = jp.with_suffix(".png")
                if png.exists():
                    entry["pngs"][metric_key] = str(png.relative_to(OUT_DIR))
                lift = jp.parent / f"{jp.stem}_lift.png"
                if lift.exists():
                    entry["pngs"][f"{metric_key}_lift"] = str(lift.relative_to(OUT_DIR))
            bases.append(entry)

    # Per-bias position baselines (already in per_bias) — no extra work
    return {
        "bias_ids": bias_ids,
        "bias_short": bias_short,
        "per_bias": per_bias,
        "sbrs": sbrs,
        "bases": bases,
        "min_rs": cohort.min_rs,
        "tau_d": cohort.tau_d,
        "n_total_pids_in_sbrs": sum(len(v) for v in sbrs.values()),
        "skipped_no_resp": cohort.skipped_no_resp,
        "skipped_pervasive_only": cohort.skipped_pervasive_only,
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Cross-bias eval — interactive heatmaps</title>
<style>
  :root {
    --bg: #f7f6f1;
    --card: #fdfcf7;
    --fg: #1a1a1a;
    --muted: #6b6b6b;
    --line: #d8d4c7;
    --accent: #c66a36;
    --accent-soft: rgba(198,106,54,0.08);
    --good: #4a7d3e;
    --warn: #b07a1a;
    --bad: #b03a3a;
    --highlight: #f6e9b8;
  }
  * { box-sizing: border-box; }
  body {
    background: var(--bg);
    color: var(--fg);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", sans-serif;
    line-height: 1.55;
    margin: 0;
    padding: 32px 28px 80px;
    font-size: 14px;
  }
  .wrap { max-width: 1400px; margin: 0 auto; }
  h1, h2, h3, h4 {
    font-family: Georgia, "Times New Roman", serif;
    font-weight: 600;
    color: var(--fg);
  }
  h1 { font-size: 30px; line-height: 1.2; margin: 0 0 6px; }
  .subtitle { color: var(--muted); font-size: 15px; margin-bottom: 24px; padding-bottom: 18px; border-bottom: 1px solid var(--line); }
  h2 { font-size: 22px; margin-top: 36px; margin-bottom: 10px; padding-top: 12px; border-top: 2px solid var(--fg); }
  h3 { font-size: 17px; margin-top: 22px; margin-bottom: 8px; color: var(--accent); }
  p { margin: 8px 0; }
  code {
    font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
    font-size: 12px;
    background: rgba(0,0,0,0.04);
    padding: 1px 5px;
    border-radius: 3px;
    color: #6f3e1e;
  }
  .meta-row { display: flex; gap: 18px; flex-wrap: wrap; margin: 10px 0 20px; }
  .meta-pill {
    background: var(--card);
    border: 1px solid var(--line);
    padding: 6px 12px;
    border-radius: 4px;
    font-size: 12px;
    color: var(--muted);
  }
  .meta-pill b { color: var(--fg); font-weight: 600; }
  .controls {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 4px;
    padding: 14px 16px;
    margin: 16px 0;
    display: flex;
    gap: 24px;
    flex-wrap: wrap;
    align-items: center;
  }
  .controls label { font-size: 12px; color: var(--muted); margin-right: 6px; }
  .controls select, .controls button {
    font: inherit;
    font-size: 13px;
    padding: 5px 10px;
    border: 1px solid var(--line);
    background: white;
    border-radius: 3px;
    color: var(--fg);
    cursor: pointer;
  }
  .controls button:hover { background: var(--accent-soft); }
  .controls button.active { background: var(--accent); color: white; border-color: var(--accent); }

  .summary-table {
    border-collapse: collapse;
    width: 100%;
    font-size: 13px;
    margin: 12px 0;
  }
  .summary-table th, .summary-table td {
    padding: 7px 10px;
    border-bottom: 1px solid var(--line);
    text-align: left;
  }
  .summary-table th {
    font-weight: 600; color: var(--muted); font-size: 11px;
    text-transform: uppercase; letter-spacing: 0.4px;
    border-bottom: 2px solid var(--fg);
    cursor: pointer;
    user-select: none;
  }
  .summary-table th:hover { color: var(--accent); }
  .summary-table td.num { font-family: ui-monospace, monospace; text-align: right; font-variant-numeric: tabular-nums; }
  .summary-table tr.active td { background: var(--accent-soft); }

  /* Heatmap */
  .heatmap-container { overflow-x: auto; margin: 12px 0; }
  .heatmap {
    border-collapse: collapse;
    font-size: 10px;
    font-family: ui-monospace, monospace;
  }
  .heatmap th, .heatmap td {
    padding: 0;
    border: 1px solid #fff;
    text-align: center;
    vertical-align: middle;
    width: 40px;
    height: 24px;
    cursor: pointer;
    transition: outline 0.05s ease;
  }
  .heatmap th {
    background: var(--card);
    border: 1px solid var(--line);
    font-weight: 600;
    cursor: pointer;
    color: var(--fg);
    padding: 4px 6px;
    font-size: 10px;
  }
  .heatmap th.col { writing-mode: vertical-rl; transform: rotate(180deg); white-space: nowrap; height: 110px; }
  .heatmap th.row { text-align: right; padding-right: 6px; min-width: 130px; max-width: 130px; }
  .heatmap th.row .bid { color: var(--muted); margin-right: 4px; }
  .heatmap td:hover { outline: 2px solid var(--fg); position: relative; z-index: 2; }
  .heatmap td.selected { outline: 3px solid var(--accent); position: relative; z-index: 3; }
  .heatmap th.row.highlighted, .heatmap th.col.highlighted { color: var(--accent); }
  .heatmap td.diagonal { outline: 1px dashed rgba(0,0,0,0.3); }
  .heatmap td.null { background: repeating-linear-gradient(45deg, #ddd, #ddd 4px, #eee 4px, #eee 8px); color: var(--muted); }

  .legend {
    display: flex;
    gap: 12px;
    align-items: center;
    margin: 8px 0;
    font-size: 12px;
    color: var(--muted);
  }
  .legend-bar {
    display: inline-block;
    width: 220px;
    height: 14px;
    border: 1px solid var(--line);
  }

  /* Detail panel */
  .detail-panel {
    background: var(--card);
    border: 1px solid var(--line);
    border-left: 4px solid var(--accent);
    padding: 14px 18px;
    margin: 16px 0;
    min-height: 80px;
    border-radius: 0 4px 4px 0;
  }
  .detail-panel h3 { margin-top: 0; }
  .detail-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 8px 16px;
    margin: 10px 0;
  }
  .stat {
    border-bottom: 1px solid var(--line);
    padding: 4px 0;
  }
  .stat .k { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.3px; }
  .stat .v { font-family: ui-monospace, monospace; font-size: 14px; font-weight: 600; }

  /* Top transfer pairs table */
  .pairs-table {
    border-collapse: collapse;
    width: 100%;
    font-size: 12px;
    margin: 12px 0;
  }
  .pairs-table th, .pairs-table td { padding: 5px 8px; border-bottom: 1px solid var(--line); text-align: left; }
  .pairs-table th { font-weight: 600; color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.3px; }
  .pairs-table td.num { font-family: ui-monospace, monospace; text-align: right; }
  .pairs-table tr:hover td { background: var(--accent-soft); }
  .pairs-table .lift-pos { color: var(--good); font-weight: 600; }
  .pairs-table .lift-neg { color: var(--bad); font-weight: 600; }

  .png-link {
    display: inline-block;
    margin: 4px 8px 4px 0;
    padding: 6px 10px;
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 3px;
    text-decoration: none;
    color: var(--accent);
    font-size: 12px;
  }
  .png-link:hover { background: var(--accent-soft); }

  .pid-list {
    font-family: ui-monospace, monospace;
    font-size: 12px;
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 3px;
    padding: 10px 14px;
    max-height: 180px;
    overflow-y: auto;
  }
  .pid-list .pid {
    display: inline-block;
    margin: 1px 4px 1px 0;
    padding: 1px 6px;
    background: rgba(0,0,0,0.04);
    border-radius: 2px;
    font-size: 11px;
  }

  .footnote { font-size: 12px; color: var(--muted); margin-top: 8px; }

  details { margin: 10px 0; border: 1px solid var(--line); border-radius: 3px; background: var(--card); }
  details > summary { padding: 8px 14px; cursor: pointer; font-weight: 600; color: var(--accent); }
  details[open] > summary { border-bottom: 1px solid var(--line); }
  details > div, details > pre { padding: 12px 16px; }
  details pre { margin: 0; font-size: 11px; line-height: 1.5; max-height: 320px; overflow: auto; background: transparent; border: none; padding: 12px 16px; }

  /* Anchor link offset */
  :target { scroll-margin-top: 16px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Cross-bias eval — interactive heatmaps</h1>
  <div class="subtitle">Click any (A → B) cell to drill in. Click a bias label to focus its row + column.</div>

  <div class="meta-row">
    <span class="meta-pill"><b>__N_BIAS__</b> biases (rs ≥ __MIN_RS__)</span>
    <span class="meta-pill"><b>__N_PIDS__</b> pids in some SBRS</span>
    <span class="meta-pill">τ_d = <b>__TAU_D__</b>, NMS w = <b>__TAU_D__</b></span>
    <span class="meta-pill">template W = <b>10</b></span>
    <span class="meta-pill">__SKIP_PERV__ pids dropped (pervasive-only)</span>
  </div>

  <h2 id="summary">Per-basis ranking</h2>
  <p>weighted_hit@5 averaged over the 30×30. <b>Lift</b> subtracts the per-column position-baseline (predict-the-median detector). Click a row to load its heatmap below.</p>
  <table class="summary-table" id="summary-table">
    <thead>
      <tr>
        <th data-sort="basis">basis / config</th>
        <th data-sort="diag" class="num">diag (raw)</th>
        <th data-sort="off" class="num">off-diag (raw)</th>
        <th data-sort="diagL" class="num">diag-LIFT</th>
        <th data-sort="offL" class="num">off-diag-LIFT</th>
        <th data-sort="frac" class="num">% off-diag &gt; 0</th>
      </tr>
    </thead>
    <tbody></tbody>
  </table>

  <h2 id="heatmap-section">Heatmap</h2>
  <div class="controls">
    <div>
      <label>basis</label>
      <select id="basis-select"></select>
    </div>
    <div>
      <label>metric</label>
      <select id="metric-select"></select>
    </div>
    <div>
      <label>view</label>
      <button id="view-raw" class="view-btn active">Raw</button>
      <button id="view-lift" class="view-btn">Lift over baseline</button>
    </div>
    <div style="margin-left:auto;">
      <span id="png-links"></span>
    </div>
  </div>

  <div class="legend" id="legend">
    <span id="legend-low">0.0</span>
    <span class="legend-bar" id="legend-bar"></span>
    <span id="legend-high">1.0</span>
    <span style="margin-left: 18px;">rows = template bias <b>A</b>, cols = test bias <b>B</b></span>
  </div>

  <div class="heatmap-container" id="heatmap-container"></div>

  <div class="detail-panel" id="detail-panel">
    <h3 id="detail-title">click a cell or a bias label</h3>
    <div id="detail-body" style="color: var(--muted); font-size: 13px;">
      Cells show <code>weighted_hit@5</code> by default. Hover for tooltip; click to pin and see all metrics + diagnostics.
      Click a row label or column label to focus that bias's row and column.
    </div>
  </div>

  <h2 id="transfer-pairs">Top off-diagonal LIFT pairs</h2>
  <p class="footnote">Computed for the currently-selected basis + metric. <code>Lift = metric − position_baseline_B</code>. Positive = template-A on bias-B beats the no-learning baseline.</p>
  <table class="pairs-table" id="pairs-table">
    <thead>
      <tr>
        <th data-sort="lift" class="num">lift</th>
        <th data-sort="A">A (template bias)</th>
        <th data-sort="B">B (test bias)</th>
        <th data-sort="metric" class="num">metric</th>
        <th data-sort="baseline" class="num">baseline_B</th>
        <th data-sort="n" class="num">n_test_pids</th>
        <th data-sort="famB" class="num">fam_div_B</th>
      </tr>
    </thead>
    <tbody></tbody>
  </table>

  <h2 id="per-bias">Per-bias diagnostics</h2>
  <p class="footnote">Click the <code>+</code> to expand a bias and see its single bias response set (pids).</p>
  <table class="pairs-table" id="bias-table">
    <thead>
      <tr>
        <th data-sort="bid" class="num">bid</th>
        <th data-sort="short">short</th>
        <th data-sort="n" class="num">|SBRS|</th>
        <th data-sort="baseline" class="num">pos_baseline</th>
        <th data-sort="famDiv" class="num">fam_div</th>
        <th data-sort="famN" class="num">n_unique_families</th>
      </tr>
    </thead>
    <tbody></tbody>
  </table>

  <details>
    <summary>SBRS details (click a bias to view its pids)</summary>
    <div id="sbrs-details">click any bias name above to load its pid list here</div>
  </details>

  <details>
    <summary>Fit log (which templates built successfully)</summary>
    <pre id="fit-log"></pre>
  </details>
</div>

<script>
const DATA = __DATA_JSON__;

// ---------------------------------------------------------------- helpers
function $(id) { return document.getElementById(id); }
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function fmt(x, digits = 3) { return x === null || x === undefined ? "—" : x.toFixed(digits); }
function fmtPct(x) { return x === null ? "—" : (100 * x).toFixed(1) + "%"; }
function fmtSign(x) { return x === null ? "—" : (x >= 0 ? "+" : "") + x.toFixed(3); }

// Color scales (matching matplotlib viridis + RdBu_r approximations)
const VIRIDIS = [
  [68,1,84],[72,40,120],[62,74,137],[49,104,142],[38,130,142],
  [31,158,137],[53,183,121],[109,205,89],[180,222,44],[253,231,37]
];
const RDBU_R = [
  [33,102,172],[67,147,195],[146,197,222],[209,229,240],[247,247,247],
  [253,219,199],[244,165,130],[214,96,77],[178,24,43]
];
function lerp(a, b, t) { return a + (b - a) * t; }
function colormapColor(scale, t) {
  if (t === null || isNaN(t)) return "rgb(220,220,220)";
  t = Math.max(0, Math.min(1, t));
  const idx = t * (scale.length - 1);
  const i0 = Math.floor(idx);
  const i1 = Math.min(scale.length - 1, i0 + 1);
  const frac = idx - i0;
  const c0 = scale[i0]; const c1 = scale[i1];
  return `rgb(${Math.round(lerp(c0[0],c1[0],frac))},${Math.round(lerp(c0[1],c1[1],frac))},${Math.round(lerp(c0[2],c1[2],frac))})`;
}
function viridis(t) { return colormapColor(VIRIDIS, t); }
function rdbu(t) {
  // t: -1..1 -> 0..1
  return colormapColor(RDBU_R, (t + 1) / 2);
}

// ---------------------------------------------------------------- state
let state = {
  basisIdx: 0,
  metric: "weighted_hit5",
  view: "raw",       // 'raw' or 'lift'
  selectedCell: null, // {A, B}
  focusedBias: null,
};

// ---------------------------------------------------------------- precompute summary stats per basis
function basisSummary(b, metricKey) {
  const bias_ids = DATA.bias_ids;
  const cells = b.metrics[metricKey].cells;
  const pos = bias_ids.map(B => DATA.per_bias[String(B)].position_baseline_hit1);
  let diagSum = 0, diagN = 0, offSum = 0, offN = 0;
  let diagLiftSum = 0, offLiftSum = 0;
  let offPosCount = 0;
  for (let i = 0; i < bias_ids.length; i++) {
    for (let j = 0; j < bias_ids.length; j++) {
      const v = cells[String(bias_ids[i])][String(bias_ids[j])];
      if (v === null || v === undefined || v.m === null) continue;
      const lift = v.m - pos[j];
      if (i === j) { diagSum += v.m; diagLiftSum += lift; diagN++; }
      else {
        offSum += v.m; offLiftSum += lift; offN++;
        if (lift > 0) offPosCount++;
      }
    }
  }
  return {
    diagRaw: diagN ? diagSum/diagN : null,
    offRaw:  offN ? offSum/offN : null,
    diagLift: diagN ? diagLiftSum/diagN : null,
    offLift: offN ? offLiftSum/offN : null,
    fracPos: offN ? offPosCount/offN : null,
    n_off: offN,
  };
}

// ---------------------------------------------------------------- summary table
function renderSummaryTable() {
  const tbody = $("summary-table").querySelector("tbody");
  tbody.innerHTML = "";
  DATA.bases.forEach((b, idx) => {
    const s = basisSummary(b, "weighted_hit5");
    const tr = el("tr");
    tr.dataset.idx = idx;
    if (idx === state.basisIdx) tr.classList.add("active");
    const labelTd = el("td"); labelTd.innerHTML = `<b>${b.basis}</b><br><span style="color:var(--muted);font-size:11px;">${b.config}</span>`;
    tr.appendChild(labelTd);
    [s.diagRaw, s.offRaw].forEach(v => { const td = el("td", "num", fmt(v)); tr.appendChild(td); });
    [s.diagLift, s.offLift].forEach(v => {
      const td = el("td", "num", fmtSign(v));
      td.style.color = v >= 0 ? "var(--good)" : "var(--bad)";
      tr.appendChild(td);
    });
    tr.appendChild(el("td", "num", fmtPct(s.fracPos)));
    tr.style.cursor = "pointer";
    tr.addEventListener("click", () => {
      state.basisIdx = idx;
      $("basis-select").value = String(idx);
      renderSummaryTable();
      renderHeatmap();
      renderPairsTable();
      renderPngLinks();
      renderFitLog();
    });
    tbody.appendChild(tr);
  });
}

// ---------------------------------------------------------------- heatmap
function getCells() {
  const b = DATA.bases[state.basisIdx];
  return b.metrics[state.metric].cells;
}
function getColorFor(value, j) {
  if (value === null) return null;
  if (state.view === "raw") {
    if (state.metric === "median_distance") {
      // 0..100 -> reversed
      const t = 1 - Math.min(value, 100) / 100;
      return viridis(t);
    }
    return viridis(Math.max(0, Math.min(1, value)));
  } else {
    const baseline = DATA.per_bias[String(DATA.bias_ids[j])].position_baseline_hit1;
    const lift = value - baseline;
    return rdbu(Math.max(-1, Math.min(1, lift)));
  }
}
function renderLegend() {
  if (state.view === "raw") {
    if (state.metric === "median_distance") {
      $("legend-low").textContent = "100+ tok";
      $("legend-high").textContent = "0 tok";
    } else {
      $("legend-low").textContent = "0.0";
      $("legend-high").textContent = "1.0";
    }
    const stops = [];
    for (let i = 0; i < 10; i++) stops.push(viridis(i/9));
    $("legend-bar").style.background = `linear-gradient(to right, ${stops.join(",")})`;
  } else {
    $("legend-low").textContent = "−1";
    $("legend-high").textContent = "+1";
    const stops = [];
    for (let i = 0; i < 9; i++) {
      const t = i / 8 * 2 - 1;
      stops.push(rdbu(t));
    }
    $("legend-bar").style.background = `linear-gradient(to right, ${stops.join(",")})`;
  }
}
function renderHeatmap() {
  const container = $("heatmap-container");
  container.innerHTML = "";
  const cells = getCells();
  const bias_ids = DATA.bias_ids;
  const table = el("table", "heatmap");
  // Header row
  const trh = el("tr");
  trh.appendChild(el("th"));  // top-left corner
  bias_ids.forEach((B, j) => {
    const th = el("th", "col");
    th.textContent = `${B}: ${DATA.bias_short[String(B)]}`;
    th.style.cursor = "pointer";
    th.addEventListener("click", () => focusBias(B));
    th.dataset.colidx = j;
    trh.appendChild(th);
  });
  table.appendChild(trh);
  // Data rows
  bias_ids.forEach((A, i) => {
    const tr = el("tr");
    const th = el("th", "row");
    th.innerHTML = `<span class="bid">${A}</span>${DATA.bias_short[String(A)]}`;
    th.style.cursor = "pointer";
    th.addEventListener("click", () => focusBias(A));
    th.dataset.rowidx = i;
    tr.appendChild(th);
    bias_ids.forEach((B, j) => {
      const v = cells[String(A)][String(B)];
      const td = el("td");
      td.dataset.row = i; td.dataset.col = j;
      if (i === j) td.classList.add("diagonal");
      if (v === null || v.m === null) {
        td.classList.add("null");
        td.textContent = "—";
      } else {
        td.style.background = getColorFor(v.m, j);
        const baseline = DATA.per_bias[String(B)].position_baseline_hit1;
        const lift = v.m - baseline;
        const showVal = state.view === "raw" ? v.m.toFixed(2) : (lift >= 0 ? "+" : "") + lift.toFixed(2);
        td.textContent = showVal;
        // Auto-pick text color for contrast
        const rgb = td.style.background.match(/\d+/g).map(Number);
        const lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2];
        td.style.color = lum > 130 ? "#000" : "#fff";
        td.title = `A=${A} ${DATA.bias_short[String(A)]} → B=${B} ${DATA.bias_short[String(B)]}\nmetric=${v.m.toFixed(3)}, baseline=${baseline.toFixed(3)}, lift=${(lift>=0?"+":"")}${lift.toFixed(3)}\nn_test=${v.n}, overlap=${v.ov}`;
      }
      td.addEventListener("click", () => selectCell(A, B));
      tr.appendChild(td);
    });
    table.appendChild(tr);
  });
  container.appendChild(table);
  // Re-apply selection / focus highlights after re-render
  if (state.selectedCell) selectCell(state.selectedCell.A, state.selectedCell.B, true);
  if (state.focusedBias !== null) highlightFocusedBias();
  renderLegend();
}

// ---------------------------------------------------------------- selection / focus
function selectCell(A, B, silent) {
  state.selectedCell = { A, B };
  // Visual: clear old selection, mark new
  document.querySelectorAll(".heatmap td.selected").forEach(t => t.classList.remove("selected"));
  const i = DATA.bias_ids.indexOf(A);
  const j = DATA.bias_ids.indexOf(B);
  const td = document.querySelector(`.heatmap td[data-row="${i}"][data-col="${j}"]`);
  if (td) td.classList.add("selected");
  // Detail panel
  renderCellDetail(A, B);
}
function focusBias(bid) {
  state.focusedBias = bid;
  highlightFocusedBias();
  renderBiasDetail(bid);
}
function highlightFocusedBias() {
  document.querySelectorAll(".heatmap th.highlighted").forEach(t => t.classList.remove("highlighted"));
  if (state.focusedBias === null) return;
  const i = DATA.bias_ids.indexOf(state.focusedBias);
  const rowTh = document.querySelector(`.heatmap th[data-rowidx="${i}"]`);
  const colTh = document.querySelector(`.heatmap th[data-colidx="${i}"]`);
  if (rowTh) rowTh.classList.add("highlighted");
  if (colTh) colTh.classList.add("highlighted");
}

// ---------------------------------------------------------------- detail panels
function renderCellDetail(A, B) {
  const cells = getCells();
  const v = cells[String(A)][String(B)];
  $("detail-title").textContent = `(A=${A} ${DATA.bias_short[String(A)]}) → (B=${B} ${DATA.bias_short[String(B)]})`;
  if (v === null || v.m === null) {
    $("detail-body").textContent = "Cell has no data (template not built or no usable test pids).";
    return;
  }
  const baseline = DATA.per_bias[String(B)].position_baseline_hit1;
  const lift = v.m - baseline;
  const famDivB = DATA.per_bias[String(B)].family_diversity_ratio;
  const sbrsBN = DATA.per_bias[String(B)].n_pids;
  // All 5 metrics for this cell
  const b = DATA.bases[state.basisIdx];
  const allMetrics = ["weighted_hit5", "hit1", "hit3", "hit5", "median_distance"];
  let metricRows = "";
  allMetrics.forEach(mk => {
    const m = b.metrics[mk];
    if (!m) return;
    const cv = m.cells[String(A)][String(B)];
    metricRows += `<tr><td>${mk}</td><td class="num">${cv && cv.m !== null ? cv.m.toFixed(3) : "—"}</td></tr>`;
  });
  $("detail-body").innerHTML = `
    <div class="detail-grid">
      <div class="stat"><div class="k">${state.metric}</div><div class="v">${v.m.toFixed(3)}</div></div>
      <div class="stat"><div class="k">position baseline (B)</div><div class="v">${baseline.toFixed(3)}</div></div>
      <div class="stat"><div class="k">lift over baseline</div><div class="v" style="color:${lift>=0?'var(--good)':'var(--bad)'}">${lift>=0?'+':''}${lift.toFixed(3)}</div></div>
      <div class="stat"><div class="k">n_test_pids</div><div class="v">${v.n}</div></div>
      <div class="stat"><div class="k">pid_overlap_AB</div><div class="v">${v.ov}</div></div>
      <div class="stat"><div class="k">|SBRS(B)|</div><div class="v">${sbrsBN}</div></div>
      <div class="stat"><div class="k">fam_div(B)</div><div class="v">${famDivB.toFixed(2)}</div></div>
    </div>
    <table class="pairs-table" style="max-width:340px;margin-top:12px;">
      <thead><tr><th>metric</th><th class="num">value (this cell)</th></tr></thead>
      <tbody>${metricRows}</tbody>
    </table>
    <div class="footnote">cell = mean across ${v.n} test pids${v.sk ? `; ${v.sk} pid${v.sk!=1?'s':''} skipped (missing data)` : ''}</div>
  `;
}

function renderBiasDetail(bid) {
  const cells = getCells();
  const bias_ids = DATA.bias_ids;
  const i = bias_ids.indexOf(bid);
  const pb = DATA.per_bias[String(bid)];
  // Row mean (this bias as template A) and column mean (this bias as test B), both raw + lift
  let rowSum = 0, rowN = 0, rowLiftSum = 0;
  let colSum = 0, colN = 0, colLiftSum = 0;
  for (let j = 0; j < bias_ids.length; j++) {
    const vRow = cells[String(bid)][String(bias_ids[j])];
    if (vRow !== null && vRow.m !== null) {
      const baseline = DATA.per_bias[String(bias_ids[j])].position_baseline_hit1;
      rowSum += vRow.m; rowLiftSum += (vRow.m - baseline); rowN++;
    }
    const vCol = cells[String(bias_ids[j])][String(bid)];
    if (vCol !== null && vCol.m !== null) {
      colSum += vCol.m; colLiftSum += (vCol.m - pb.position_baseline_hit1); colN++;
    }
  }
  const sbrs = DATA.sbrs[String(bid)] || [];
  const sbrsHtml = sbrs.map(p => `<span class="pid">${p}</span>`).join("");
  $("detail-title").textContent = `Bias ${bid}: ${DATA.bias_short[String(bid)]}`;
  $("detail-body").innerHTML = `
    <div class="detail-grid">
      <div class="stat"><div class="k">|SBRS|</div><div class="v">${pb.n_pids}</div></div>
      <div class="stat"><div class="k">position baseline</div><div class="v">${pb.position_baseline_hit1.toFixed(3)}</div></div>
      <div class="stat"><div class="k">family diversity</div><div class="v">${pb.family_diversity_ratio.toFixed(2)}</div></div>
      <div class="stat"><div class="k">unique families</div><div class="v">${pb.n_unique_prompt_families}</div></div>
      <div class="stat"><div class="k">row mean (as template)</div><div class="v">${rowN ? (rowSum/rowN).toFixed(3) : "—"}</div></div>
      <div class="stat"><div class="k">row LIFT mean</div><div class="v" style="color:${rowLiftSum>=0?'var(--good)':'var(--bad)'}">${rowN?(rowLiftSum/rowN>=0?'+':'')+(rowLiftSum/rowN).toFixed(3):"—"}</div></div>
      <div class="stat"><div class="k">col mean (as test)</div><div class="v">${colN ? (colSum/colN).toFixed(3) : "—"}</div></div>
      <div class="stat"><div class="k">col LIFT mean</div><div class="v" style="color:${colLiftSum>=0?'var(--good)':'var(--bad)'}">${colN?(colLiftSum/colN>=0?'+':'')+(colLiftSum/colN).toFixed(3):"—"}</div></div>
    </div>
    <h4 style="margin-top:16px;color:var(--accent);font-family:Georgia,serif;">SBRS — ${sbrs.length} pids whose first hack is bias ${bid}</h4>
    <div class="pid-list">${sbrsHtml || "(no pids)"}</div>
  `;
  $("sbrs-details").innerHTML = `<b>Bias ${bid}: ${DATA.bias_short[String(bid)]}</b> — ${sbrs.length} pids:<div class="pid-list" style="margin-top:8px;">${sbrsHtml}</div>`;
}

// ---------------------------------------------------------------- pairs table
function renderPairsTable() {
  const cells = getCells();
  const bias_ids = DATA.bias_ids;
  const tbody = $("pairs-table").querySelector("tbody");
  const rows = [];
  for (let i = 0; i < bias_ids.length; i++) {
    for (let j = 0; j < bias_ids.length; j++) {
      if (i === j) continue;
      const v = cells[String(bias_ids[i])][String(bias_ids[j])];
      if (v === null || v.m === null) continue;
      const A = bias_ids[i], B = bias_ids[j];
      const baseline = DATA.per_bias[String(B)].position_baseline_hit1;
      const famB = DATA.per_bias[String(B)].family_diversity_ratio;
      rows.push({
        A, B,
        AName: DATA.bias_short[String(A)],
        BName: DATA.bias_short[String(B)],
        metric: v.m,
        baseline,
        lift: v.m - baseline,
        n: v.n,
        famB,
      });
    }
  }
  rows.sort((x, y) => y.lift - x.lift);
  tbody.innerHTML = "";
  rows.slice(0, 25).forEach(r => {
    const tr = el("tr");
    const liftClass = r.lift >= 0 ? "lift-pos" : "lift-neg";
    tr.innerHTML = `
      <td class="num ${liftClass}">${r.lift>=0?'+':''}${r.lift.toFixed(3)}</td>
      <td><b>${r.A}</b> ${r.AName}</td>
      <td><b>${r.B}</b> ${r.BName}</td>
      <td class="num">${r.metric.toFixed(3)}</td>
      <td class="num">${r.baseline.toFixed(3)}</td>
      <td class="num">${r.n}</td>
      <td class="num">${r.famB.toFixed(2)}</td>
    `;
    tr.style.cursor = "pointer";
    tr.addEventListener("click", () => {
      selectCell(r.A, r.B);
      window.scrollTo({ top: $("heatmap-section").offsetTop - 12, behavior: "smooth" });
    });
    tbody.appendChild(tr);
  });
  // Append a "show last 5 (anti-transfer)" rows
  rows.sort((x, y) => x.lift - y.lift);
  rows.slice(0, 5).forEach(r => {
    const tr = el("tr");
    tr.style.background = "rgba(176,58,58,0.04)";
    tr.innerHTML = `
      <td class="num lift-neg">${r.lift.toFixed(3)}</td>
      <td><b>${r.A}</b> ${r.AName}</td>
      <td><b>${r.B}</b> ${r.BName}</td>
      <td class="num">${r.metric.toFixed(3)}</td>
      <td class="num">${r.baseline.toFixed(3)}</td>
      <td class="num">${r.n}</td>
      <td class="num">${r.famB.toFixed(2)}</td>
    `;
    tr.style.cursor = "pointer";
    tr.addEventListener("click", () => {
      selectCell(r.A, r.B);
      window.scrollTo({ top: $("heatmap-section").offsetTop - 12, behavior: "smooth" });
    });
    tbody.appendChild(tr);
  });
}

// ---------------------------------------------------------------- per-bias table
function renderBiasTable() {
  const tbody = $("bias-table").querySelector("tbody");
  tbody.innerHTML = "";
  DATA.bias_ids.forEach(bid => {
    const pb = DATA.per_bias[String(bid)];
    const tr = el("tr");
    tr.style.cursor = "pointer";
    tr.innerHTML = `
      <td class="num">${bid}</td>
      <td><b>${pb.short}</b></td>
      <td class="num">${pb.n_pids}</td>
      <td class="num">${pb.position_baseline_hit1.toFixed(3)}</td>
      <td class="num">${pb.family_diversity_ratio.toFixed(2)}</td>
      <td class="num">${pb.n_unique_prompt_families}</td>
    `;
    tr.addEventListener("click", () => {
      focusBias(bid);
      window.scrollTo({ top: $("heatmap-section").offsetTop - 12, behavior: "smooth" });
    });
    tbody.appendChild(tr);
  });
}

// ---------------------------------------------------------------- png links
function renderPngLinks() {
  const b = DATA.bases[state.basisIdx];
  const links = [];
  const k = state.metric;
  if (b.pngs[k]) links.push(`<a class="png-link" href="${b.pngs[k]}" target="_blank">PNG (raw)</a>`);
  if (b.pngs[k + "_lift"]) links.push(`<a class="png-link" href="${b.pngs[k + "_lift"]}" target="_blank">PNG (lift)</a>`);
  $("png-links").innerHTML = links.join("");
}

function renderFitLog() {
  $("fit-log").textContent = DATA.bases[state.basisIdx].fit_log || "(no fit log available)";
}

// ---------------------------------------------------------------- selectors
function populateSelectors() {
  const bSel = $("basis-select");
  DATA.bases.forEach((b, idx) => {
    const opt = el("option");
    opt.value = String(idx);
    opt.textContent = `${b.basis} / ${b.config}`;
    bSel.appendChild(opt);
  });
  bSel.value = String(state.basisIdx);
  bSel.addEventListener("change", () => {
    state.basisIdx = Number(bSel.value);
    renderSummaryTable();
    renderHeatmap();
    renderPairsTable();
    renderPngLinks();
    renderFitLog();
  });

  const mSel = $("metric-select");
  ["weighted_hit5", "hit1", "hit3", "hit5", "median_distance"].forEach(m => {
    const opt = el("option");
    opt.value = m;
    opt.textContent = m;
    mSel.appendChild(opt);
  });
  mSel.value = state.metric;
  mSel.addEventListener("change", () => {
    state.metric = mSel.value;
    renderHeatmap();
    renderPairsTable();
    renderPngLinks();
  });

  $("view-raw").addEventListener("click", () => { state.view = "raw"; document.querySelectorAll(".view-btn").forEach(b => b.classList.remove("active")); $("view-raw").classList.add("active"); renderHeatmap(); });
  $("view-lift").addEventListener("click", () => { state.view = "lift"; document.querySelectorAll(".view-btn").forEach(b => b.classList.remove("active")); $("view-lift").classList.add("active"); renderHeatmap(); });
}

// ---------------------------------------------------------------- init
populateSelectors();
renderSummaryTable();
renderHeatmap();
renderPairsTable();
renderBiasTable();
renderPngLinks();
renderFitLog();

</script>
</body>
</html>
"""


def main():
    print("Collecting data...", flush=True)
    data = collect_data()
    print(f"  {len(data['bias_ids'])} biases, {len(data['bases'])} basis configs", flush=True)
    n_bias = len(data['bias_ids'])
    html = (
        HTML_TEMPLATE
        .replace("__N_BIAS__", str(n_bias))
        .replace("__MIN_RS__", str(data["min_rs"]))
        .replace("__N_PIDS__", str(data["n_total_pids_in_sbrs"]))
        .replace("__TAU_D__", str(data["tau_d"]))
        .replace("__SKIP_PERV__", str(data["skipped_pervasive_only"]))
        .replace("__DATA_JSON__", json.dumps(data))
    )
    INDEX.write_text(html)
    print(f"Wrote {INDEX} ({INDEX.stat().st_size // 1024} KB)")
    print(f"Open: file://{INDEX.resolve()}")


if __name__ == "__main__":
    main()
