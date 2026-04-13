// Steering detail panel — inline expandable panel for trait cards
//
// Renders the full detail view that appears below a clicked trait card,
// including chart, best response, per-layer results, and info tooltips.

import { escapeHtml } from '../../core/utils.js';
import { getMethodColors } from '../../core/display.js';
import { extractVectorSpec, extractRunMetrics } from './shared.js';

// ── State ──────────────────────────────────────────────────────────

let activeDetailPanel = null; // Currently visible panel element
let tooltipCache = {};        // trait → { definition, steering, coherence }

// ── CSS (injected once) ────────────────────────────────────────────

let styleInjected = false;

function injectStyles() {
    if (styleInjected) return;
    styleInjected = true;
    const style = document.createElement('style');
    style.textContent = `
/* Detail panel — inline, inserted after clicked card's grid row */
.detail-panel {
    background: var(--bg-tertiary);
    border-radius: var(--radius-md);
    padding: 20px;
    border: 1px solid var(--accent-color);
    display: none;
    margin-top: 4px;
}
.detail-panel.visible { display: block; }
.detail-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 16px;
}
.detail-title {
    font-size: var(--text-lg);
    font-weight: 600;
    margin: 0;
}
.detail-close {
    background: none;
    border: none;
    color: var(--text-tertiary);
    cursor: pointer;
    font-size: var(--text-base);
    padding: 4px 8px;
}
.detail-close:hover { color: var(--text-primary); }
.detail-meta {
    font-size: var(--text-xs);
    color: var(--text-tertiary);
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    margin-bottom: 16px;
}
.detail-meta strong { color: var(--text-secondary); }
.detail-chart {
    height: 220px;
    background: var(--bg-secondary);
    border-radius: var(--radius-sm);
    overflow: hidden;
    position: relative;
}
.detail-chart svg { width: 100%; height: 100%; }
.chart-axis-label { font-size: 9px; fill: var(--text-tertiary); }
.chart-grid-line { stroke: var(--border-color); stroke-width: 0.5; opacity: 0.3; }
.detail-chart-legend {
    display: flex;
    gap: 14px;
    margin-top: 8px;
    flex-wrap: wrap;
}
.detail-legend-item {
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: 10px;
    color: var(--text-tertiary);
}
.detail-legend-line {
    width: 16px;
    height: 2px;
    border-radius: 1px;
}

/* Info hover icons */
.detail-info-icons {
    display: flex;
    gap: 6px;
    margin-left: auto;
}
.detail-info-icon {
    position: relative;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    background: var(--bg-secondary);
    color: var(--text-tertiary);
    font-size: 10px;
    font-weight: 600;
    cursor: help;
    border: 1px solid var(--border-color);
}
.detail-info-icon:hover {
    color: var(--text-primary);
    border-color: var(--text-tertiary);
}
.detail-info-icon:hover .detail-tooltip,
.detail-tooltip:hover { display: block; }
.detail-tooltip {
    display: none;
    position: absolute;
    top: calc(100% + 6px);
    right: 0;
    width: 400px;
    max-height: 350px;
    overflow-y: auto;
    background: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-md);
    padding: 12px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.4);
    z-index: 200;
    font-size: var(--text-xs);
    color: var(--text-secondary);
    font-weight: 400;
    text-align: left;
    white-space: pre-wrap;
    line-height: 1.5;
    cursor: default;
}
/* Invisible bridge connecting icon to tooltip — covers gap in both directions */
.detail-tooltip::before {
    content: '';
    position: absolute;
    bottom: 100%;
    left: 0;
    width: 100%;
    height: 16px;
    pointer-events: auto;
}
.detail-tooltip .tt-label {
    font-weight: 600;
    color: var(--text-tertiary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
    font-size: 9px;
}

/* Best response preview */
.detail-best-response {
    background: var(--bg-secondary);
    border-radius: var(--radius-sm);
    padding: 12px;
    margin-top: 12px;
    border-left: 3px solid var(--success);
}
.detail-best-response .br-label {
    font-size: var(--text-xs);
    color: var(--text-tertiary);
    margin-bottom: 6px;
}
.detail-best-response .br-meta {
    font-size: var(--text-xs);
    color: var(--text-tertiary);
    margin-bottom: 8px;
}
.detail-best-response .br-meta strong { color: var(--text-secondary); }
.detail-best-response .br-text {
    font-size: var(--text-sm);
    color: var(--text-secondary);
    line-height: 1.6;
    white-space: pre-wrap;
}
.detail-best-response .br-loading {
    font-size: var(--text-xs);
    color: var(--text-tertiary);
    font-style: italic;
}

/* Per-layer results list */
.detail-layer-results { margin-top: 12px; }
.detail-layer-results summary {
    font-size: var(--text-xs);
    color: var(--text-tertiary);
    cursor: pointer;
    padding: 8px 12px;
    background: var(--bg-secondary);
    border-radius: var(--radius-sm);
}
.detail-layer-results summary:hover { color: var(--text-secondary); }
.detail-layer-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 6px 12px;
    font-size: var(--text-xs);
    color: var(--text-secondary);
    cursor: pointer;
    border-bottom: 1px solid var(--bg-tertiary);
    transition: background 0.1s;
}
.detail-layer-row:hover { background: var(--bg-tertiary); }
.detail-layer-row.expanded {
    background: var(--bg-tertiary);
    border-left: 2px solid var(--accent-color);
}
.detail-layer-row .lr-layer { font-weight: 600; min-width: 30px; }
.detail-layer-row .lr-score { min-width: 50px; }
.detail-layer-row .lr-score.good { color: var(--success); font-weight: 600; }
.detail-layer-row .lr-coef { min-width: 60px; color: var(--text-tertiary); }
.detail-layer-row .lr-coh { min-width: 40px; color: var(--text-tertiary); }
.detail-layer-row .lr-bar {
    flex: 1;
    height: 4px;
    background: var(--bg-secondary);
    border-radius: 2px;
    overflow: hidden;
}
.detail-layer-row .lr-bar-fill { height: 100%; border-radius: 2px; }
.detail-layer-response {
    padding: 10px 12px 10px 24px;
    background: var(--bg-primary);
    border-radius: var(--radius-sm);
    margin: 4px 0 8px 0;
    font-size: var(--text-sm);
    color: var(--text-secondary);
    line-height: 1.6;
    white-space: pre-wrap;
    display: none;
    max-height: 300px;
    overflow-y: auto;
}
.detail-layer-response.visible { display: block; }
`;
    document.head.appendChild(style);
}


// ── Tooltip data fetching ──────────────────────────────────────────

/**
 * Fetch and cache tooltip data for a trait.
 * Returns { definition, steering, coherence } with text content.
 */
async function fetchTooltipData(trait) {
    if (tooltipCache[trait]) return tooltipCache[trait];

    const experiment = window.state.experimentData?.name;
    const [category, traitName] = trait.split('/');
    const data = { definition: null, steering: null, coherence: null };

    // Fetch trait info (definition + steering questions) and coherence prompt in parallel
    const [traitInfo, judgeTemplates] = await Promise.all([
        experiment
            ? fetch(`/api/experiments/${experiment}/trait-info/${category}/${traitName}`)
                .then(r => r.ok ? r.json() : null).catch(() => null)
            : Promise.resolve(null),
        fetch('/api/judge-templates')
            .then(r => r.ok ? r.json() : null).catch(() => null),
    ]);

    if (traitInfo) {
        data.definition = traitInfo.definition || null;
        if (traitInfo.steering && traitInfo.steering.questions) {
            data.steering = traitInfo.steering.questions;
        }
    }
    if (judgeTemplates && judgeTemplates.coherence) {
        data.coherence = judgeTemplates.coherence;
    }

    tooltipCache[trait] = data;
    return data;
}


// ── SVG chart ──────────────────────────────────────────────────────

/**
 * Build a full SVG chart of score vs layer with axes, method lines, baseline,
 * and best-point markers.
 *
 * @param {Object} chartData - { methods: { key: [{ layer, score }] }, baseline, minLayer, maxLayer }
 * @param {number} width
 * @param {number} height
 * @returns {string} SVG markup
 */
function buildFullChartSVG(chartData, width, height) {
    const pad = { top: 10, right: 16, bottom: 28, left: 44 };
    const cw = width - pad.left - pad.right;
    const ch = height - pad.top - pad.bottom;

    const allScores = [];
    for (const pts of Object.values(chartData.methods)) {
        for (const p of pts) allScores.push(p.score);
    }
    if (chartData.baseline != null) allScores.push(chartData.baseline);

    if (allScores.length === 0) return '';

    const minS = Math.min(...allScores) - 5;
    const maxS = Math.max(...allScores) + 5;
    const minL = chartData.minLayer;
    const maxL = chartData.maxLayer;
    const rangeL = maxL - minL || 1;
    const rangeS = maxS - minS || 1;

    const x = l => pad.left + ((l - minL) / rangeL) * cw;
    const y = s => pad.top + ch - ((s - minS) / rangeS) * ch;

    const methodColors = getMethodColors();
    let svg = `<svg viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg">`;

    // Y-axis tick labels
    const yTicks = 4;
    for (let i = 0; i <= yTicks; i++) {
        const val = minS + rangeS * (i / yTicks);
        const yy = y(val);
        svg += `<text x="${pad.left - 6}" y="${yy + 3}" text-anchor="end" class="chart-axis-label">${val.toFixed(0)}</text>`;
    }

    // X-axis tick marks + labels
    const xStep = Math.max(1, Math.ceil(rangeL / 10));
    for (let l = minL; l <= maxL; l += xStep) {
        const xx = x(l);
        svg += `<line x1="${xx}" y1="${pad.top + ch}" x2="${xx}" y2="${pad.top + ch + 4}" stroke="var(--text-tertiary)" stroke-width="0.5" opacity="0.5"/>`;
        svg += `<text x="${xx}" y="${height - 6}" text-anchor="middle" class="chart-axis-label">${l}</text>`;
    }

    // X-axis line
    svg += `<line x1="${pad.left}" y1="${pad.top + ch}" x2="${width - pad.right}" y2="${pad.top + ch}" stroke="var(--border-color)" stroke-width="0.5" opacity="0.4"/>`;

    // Axis labels
    svg += `<text x="${width - 8}" y="${pad.top + ch + 20}" text-anchor="end" class="chart-axis-label" style="font-size: 10px;">Layer</text>`;
    svg += `<text x="12" y="${pad.top - 2}" class="chart-axis-label" style="font-size: 10px;">Score</text>`;

    // Baseline dashed line
    if (chartData.baseline != null) {
        const bY = y(chartData.baseline);
        svg += `<line x1="${pad.left}" y1="${bY}" x2="${width - pad.right}" y2="${bY}" stroke="var(--text-tertiary)" stroke-width="1" stroke-dasharray="6,4" opacity="0.6"/>`;
        svg += `<text x="${width - pad.right + 4}" y="${bY + 3}" class="chart-axis-label" style="fill: var(--text-tertiary);">baseline</text>`;
    }

    // Method lines + best-point markers
    for (const [method, points] of Object.entries(chartData.methods)) {
        if (points.length === 0) continue;
        const color = methodColors[method] || '#888';
        const sorted = [...points].sort((a, b) => a.layer - b.layer);
        const pts = sorted.map(d => `${x(d.layer).toFixed(1)},${y(d.score).toFixed(1)}`).join(' ');
        svg += `<polyline points="${pts}" fill="none" stroke="${color}" stroke-width="2" stroke-linejoin="round"/>`;

        // Best-point marker (highest score)
        const best = sorted.reduce((a, b) => a.score > b.score ? a : b);
        svg += `<circle cx="${x(best.layer)}" cy="${y(best.score)}" r="4" fill="${color}" stroke="var(--bg-primary)" stroke-width="1.5"/>`;
    }

    svg += '</svg>';
    return svg;
}


// ── Data extraction ────────────────────────────────────────────────

/**
 * Extract per-method layer data and per-layer best results from traitResults.
 *
 * @param {Object} traitResults - { runs, baseline }
 * @returns {{ methods, perLayer, baseline, minLayer, maxLayer, bestRun, configs }}
 */
function extractDetailData(traitResults, direction = null) {
    const baseline = traitResults.baseline?.trait_mean || 0;
    const runs = traitResults.runs || [];

    // Per-method: best score per layer
    const methods = {};
    // Per-layer: best overall run
    const perLayerMap = {};
    let minLayer = Infinity;
    let maxLayer = -Infinity;

    for (const run of runs) {
        const spec = extractVectorSpec(run);
        if (!spec) continue;
        const { layer, method, component, coef } = spec;
        const { traitScore, coherence } = extractRunMetrics(run.result || {}, baseline);

        if (layer < minLayer) minLayer = layer;
        if (layer > maxLayer) maxLayer = layer;

        // Per-method aggregation (best score per layer per method)
        // For negative direction, "best" = lowest trait score (furthest from baseline)
        const isBetter = (a, b) => direction === 'negative' ? a < b : a > b;

        if (!methods[method]) methods[method] = {};
        if (!methods[method][layer] || isBetter(traitScore, methods[method][layer].score)) {
            methods[method][layer] = { score: traitScore, layer };
        }

        // Per-layer best (sorted later)
        const existing = perLayerMap[layer];
        if (!existing || isBetter(traitScore, existing.traitScore)) {
            perLayerMap[layer] = {
                layer, method, component, coef, traitScore, coherence,
                timestamp: run.timestamp,
            };
        }
    }

    // Convert methods to arrays
    const methodArrays = {};
    for (const [method, layerMap] of Object.entries(methods)) {
        methodArrays[method] = Object.values(layerMap);
    }

    // Per-layer sorted by best score (descending for positive, ascending for negative)
    const perLayer = Object.values(perLayerMap).sort((a, b) =>
        direction === 'negative' ? a.traitScore - b.traitScore : b.traitScore - a.traitScore
    );

    // Best run overall (most steered — lowest for negative, highest for positive)
    const bestRun = perLayer.length > 0 ? perLayer[0] : null;

    // Count unique configs
    const configKeys = new Set();
    for (const run of runs) {
        const spec = extractVectorSpec(run);
        if (spec) configKeys.add(`${spec.method}|${spec.component}`);
    }

    return {
        methods: methodArrays,
        perLayer,
        baseline,
        minLayer: minLayer === Infinity ? 0 : minLayer,
        maxLayer: maxLayer === -Infinity ? 0 : maxLayer,
        bestRun,
        configCount: configKeys.size,
    };
}


// ── Response fetching ──────────────────────────────────────────────

let responseFileCache = {}; // entry key -> { files, baseline }

/**
 * Fetch available response file listing for an entry.
 * Cached so repeat lookups are instant.
 */
async function fetchResponseListing(entry) {
    const experiment = window.state.experimentData?.name;
    if (!experiment) return null;

    const key = `${entry.trait}|${entry.model_variant}|${entry.position}|${entry.prompt_set}`;
    if (responseFileCache[key]) return responseFileCache[key];

    const url = `/api/experiments/${experiment}/steering-responses/${entry.trait}/${entry.model_variant}/${entry.position}/${entry.prompt_set}`;
    try {
        const resp = await fetch(url);
        if (!resp.ok) return null;
        const data = await resp.json();
        responseFileCache[key] = data;
        return data;
    } catch {
        return null;
    }
}

/**
 * Fetch the actual response text for a specific layer/coef run.
 * Returns array of response objects or null.
 */
async function fetchResponseForRun(entry, run) {
    const listing = await fetchResponseListing(entry);
    if (!listing || !listing.files) return null;

    // Find matching file — match layer/method/component, pick closest coef
    // (save_mode=best may save a different coef than the highest-scoring run)
    const candidates = listing.files.filter(f =>
        f.layer === run.layer &&
        f.method === run.method &&
        f.component === run.component
    );
    if (candidates.length === 0) return null;
    const match = candidates.reduce((best, f) =>
        Math.abs(f.coef - Math.abs(run.coef)) < Math.abs(best.coef - Math.abs(run.coef)) ? f : best
    );

    const experiment = window.state.experimentData?.name;
    const basePath = window.paths?.get('steering.responses', {
        experiment,
        trait: entry.trait,
        model_variant: entry.model_variant,
        position: entry.position,
        prompt_set: entry.prompt_set,
    });
    if (!basePath) return null;

    try {
        const resp = await fetch(`/${basePath}/${match.path}`);
        if (!resp.ok) return null;
        return await resp.json();
    } catch {
        return null;
    }
}


// ── Detail panel rendering ─────────────────────────────────────────

/**
 * Build the detail panel HTML.
 *
 * @param {string} trait - e.g. "emotion_set/contentment"
 * @param {Object} traitResults - fetched results { runs, baseline, ... }
 * @returns {string} HTML
 */
function buildDetailHTML(trait, traitResults) {
    const direction = traitResults.direction || 'positive';
    const data = extractDetailData(traitResults, direction);
    const { methods, perLayer, baseline, minLayer, maxLayer, bestRun, configCount } = data;
    const traitName = trait.split('/').pop();
    const methodColors = getMethodColors();

    // Score class helper
    const deltaClass = (delta) => delta > 20 ? 'good' : delta > 5 ? 'moderate' : '';

    // Find best method + delta
    let bestDelta = 0;
    let bestLayer = 0;
    let bestMethod = '';
    let bestCoherence = 0;
    if (bestRun) {
        bestDelta = bestRun.traitScore - baseline;
        bestLayer = bestRun.layer;
        bestMethod = bestRun.method;
        bestCoherence = bestRun.coherence;
    }

    // Chart data
    const chartData = { methods, baseline, minLayer, maxLayer };
    const chartSVG = buildFullChartSVG(chartData, 900, 220);

    // Legend
    const legendItems = Object.entries(methods).map(([method, points]) => {
        const color = methodColors[method] || '#888';
        const best = points.reduce((a, b) => a.score > b.score ? a : b, { score: 0, layer: 0 });
        const delta = (best.score - baseline).toFixed(1);
        const sign = best.score - baseline >= 0 ? '+' : '';
        return `<span class="detail-legend-item">
            <span class="detail-legend-line" style="background: ${color};"></span>
            ${escapeHtml(method)} (best: ${sign}${delta} @ L${best.layer})
        </span>`;
    }).join('');

    const baselineLegend = `<span class="detail-legend-item">
        <span class="detail-legend-line" style="background: var(--text-tertiary); border-top: 1px dashed var(--text-tertiary); height: 0;"></span>
        Baseline (${baseline.toFixed(1)})
    </span>`;

    // Tooltip placeholders (content loaded on hover via fetch)
    const tooltipHTML = `
        <div class="detail-info-icons">
            <div class="detail-info-icon" data-tooltip="definition">D
                <div class="detail-tooltip"><div class="tt-label">Scoring Definition</div><em>Hover to load...</em></div>
            </div>
            <div class="detail-info-icon" data-tooltip="steering">Q
                <div class="detail-tooltip"><div class="tt-label">Steering Questions</div><em>Hover to load...</em></div>
            </div>
            <div class="detail-info-icon" data-tooltip="coherence">C
                <div class="detail-tooltip"><div class="tt-label">Coherence Prompt</div><em>Hover to load...</em></div>
            </div>
        </div>`;

    // Best response section (placeholder, loaded async)
    const bestResponseHTML = bestRun ? `
        <div class="detail-best-response" data-layer="${bestRun.layer}" data-method="${bestRun.method}" data-component="${bestRun.component}" data-coef="${bestRun.coef}">
            <div class="br-label">Best response (${direction === 'negative' ? 'lowest' : 'highest'} trait score)</div>
            <div class="br-meta">
                Layer <strong>${bestRun.layer}</strong>
                &middot; coef=<strong>${Math.abs(bestRun.coef).toFixed(1)}</strong>
                &middot; trait=<strong>${bestRun.traitScore.toFixed(1)}</strong>
                &middot; coherence=<strong>${bestRun.coherence.toFixed(0)}</strong>
            </div>
            <div class="br-text"><span class="br-loading">Loading response...</span></div>
        </div>
    ` : '';

    // Per-layer results
    const maxScore = perLayer.length > 0 ? Math.max(...perLayer.map(r => r.traitScore)) : 1;
    const layerRowsHTML = perLayer.map((r, i) => {
        const barWidth = maxScore > 0 ? Math.max(2, (r.traitScore / maxScore) * 100) : 2;
        const barColor = r.traitScore > baseline + 15 ? 'var(--success)'
            : r.traitScore > baseline + 5 ? 'var(--accent-color)' : 'var(--text-tertiary)';
        const scoreGood = r.traitScore > baseline + 10 ? 'good' : '';
        return `
            <div class="detail-layer-row" data-layer-idx="${i}" data-layer="${r.layer}" data-method="${r.method}" data-component="${r.component}" data-coef="${r.coef}">
                <span class="lr-layer">L${r.layer}</span>
                <span class="lr-score ${scoreGood}">trait=${r.traitScore.toFixed(1)}</span>
                <span class="lr-coef">coef=${Math.abs(r.coef).toFixed(1)}</span>
                <span class="lr-coh">coh=${r.coherence.toFixed(0)}</span>
                <span class="lr-bar"><span class="lr-bar-fill" style="width: ${barWidth}%; background: ${barColor};"></span></span>
            </div>
            <div class="detail-layer-response" data-layer-resp="${i}"></div>`;
    }).join('');

    // Meta row
    const sign = bestDelta >= 0 ? '+' : '';
    const scoreColorClass = deltaClass(bestDelta);
    const scoreStyle = scoreColorClass === 'good' ? 'color: var(--success); font-weight: 600;'
        : scoreColorClass === 'moderate' ? 'color: var(--accent-color); font-weight: 600;'
        : 'color: var(--text-tertiary);';

    return `
        <div class="detail-top">
            <h3 class="detail-title">${escapeHtml(traitName)}</h3>
            ${tooltipHTML}
            <button class="detail-close">&times;</button>
        </div>
        <div class="detail-meta">
            <span>Best delta: <strong style="${scoreStyle}">${sign}${bestDelta.toFixed(1)}</strong></span>
            <span>Best layer: <strong>L${bestLayer}</strong></span>
            <span>Method: <strong>${escapeHtml(bestMethod)}</strong></span>
            <span>Coherence: <strong>${bestCoherence.toFixed(0)}</strong></span>
            <span>Layers: <strong>L${minLayer}&ndash;${maxLayer}</strong> / ${window.state.experimentData?.experimentConfig?.num_hidden_layers || '?'}</span>
            <span>Configs: <strong>${configCount}</strong></span>
        </div>
        <div class="detail-chart">${chartSVG}</div>
        <div class="detail-chart-legend">
            ${legendItems}
            ${baselineLegend}
        </div>
        ${bestResponseHTML}
        <details class="detail-layer-results">
            <summary>All layers (${perLayer.length} results, sorted by score)</summary>
            ${layerRowsHTML}
        </details>
    `;
}


// ── Public API ──────────────────────────────────────────────────────

/**
 * Show the detail panel below a clicked trait card.
 *
 * @param {HTMLElement} parentEl - The card element to insert after
 * @param {string} trait - e.g. "emotion_set/contentment"
 * @param {Object} traitResults - fetched results object { runs, baseline, ... }
 * @param {Object} entry - { trait, model_variant, position, prompt_set }
 */
async function showDetailPanel(parentEl, trait, traitResults, entry) {
    injectStyles();

    // Remove any existing detail panel
    hideDetailPanel();

    // Create panel element
    const panel = document.createElement('div');
    panel.className = 'detail-panel visible';
    panel.style.gridColumn = '1 / -1';
    panel.innerHTML = buildDetailHTML(trait, traitResults);

    // Insert after the last card in the same visual row (so remaining cards don't shift below)
    const grid = parentEl.parentElement;
    const clickedTop = parentEl.getBoundingClientRect().top;
    const cards = [...grid.querySelectorAll('.trait-card')];
    let lastInRow = parentEl;
    for (const card of cards) {
        if (Math.abs(card.getBoundingClientRect().top - clickedTop) < 2) {
            lastInRow = card;
        }
    }
    lastInRow.insertAdjacentElement('afterend', panel);
    activeDetailPanel = panel;

    // ── Wire close button ──
    panel.querySelector('.detail-close').addEventListener('click', () => {
        hideDetailPanel();
    });

    // ── Wire tooltip hovers (lazy fetch) ──
    panel.querySelectorAll('.detail-info-icon').forEach(icon => {
        let loaded = false;
        icon.addEventListener('mouseenter', async () => {
            if (loaded) return;
            loaded = true;
            const type = icon.dataset.tooltip;
            const tooltip = icon.querySelector('.detail-tooltip');
            const data = await fetchTooltipData(trait);

            let content = '';
            if (type === 'definition') {
                content = data.definition
                    ? escapeHtml(data.definition)
                    : '<em>Definition not found</em>';
                tooltip.innerHTML = `<div class="tt-label">Scoring Definition (definition.txt)</div>${content}`;
            } else if (type === 'steering') {
                if (data.steering) {
                    const s = data.steering;
                    const parts = [];
                    if (s.adversarial_prefix || s.prefix) parts.push(`Prefix: ${escapeHtml(s.adversarial_prefix || s.prefix)}`);
                    if (s.direction) parts.push(`Direction: ${escapeHtml(s.direction)}`);
                    const questions = s.questions || (Array.isArray(s) ? s : []);
                    if (questions.length) parts.push(`\nQuestions (${questions.length}):\n${questions.map((q, i) => `${i + 1}. ${escapeHtml(typeof q === 'string' ? q : JSON.stringify(q))}`).join('\n')}`);
                    content = parts.join('\n') || escapeHtml(JSON.stringify(s, null, 2));
                } else {
                    content = '<em>Steering questions not found</em>';
                }
                tooltip.innerHTML = `<div class="tt-label">Steering Questions (steering.json)</div>${content}`;
            } else if (type === 'coherence') {
                content = data.coherence
                    ? escapeHtml(data.coherence)
                    : '<em>Coherence prompt not found</em>';
                tooltip.innerHTML = `<div class="tt-label">Coherence Prompt</div>${content}`;
            }
        });
    });

    // ── Check if responses exist, then wire response sections ──
    const listing = await fetchResponseListing(entry);
    const hasResponses = listing && listing.files && listing.files.length > 0;

    const bestResponseEl = panel.querySelector('.detail-best-response');
    const layerResultsEl = panel.querySelector('.detail-layer-results');

    if (!hasResponses) {
        // Hide response sections entirely when no response files
        if (bestResponseEl) bestResponseEl.style.display = 'none';
    } else {
        // Add response hint to summary
        if (layerResultsEl) {
            const summary = layerResultsEl.querySelector('summary');
            if (summary) summary.textContent += ' — click row to view response';
        }
        // Load best response text
        if (bestResponseEl) {
            const layer = parseInt(bestResponseEl.dataset.layer);
            const method = bestResponseEl.dataset.method;
            const component = bestResponseEl.dataset.component;
            const coef = parseFloat(bestResponseEl.dataset.coef);
            const brText = bestResponseEl.querySelector('.br-text');

            fetchResponseForRun(entry, { layer, method, component, coef }).then(responses => {
                if (!responses || responses.length === 0) {
                    bestResponseEl.style.display = 'none';
                    return;
                }
                const best = responses.reduce((a, b) =>
                    (b.trait_score || 0) > (a.trait_score || 0) ? b : a, responses[0]);
                const promptHtml = best.prompt
                    ? `<div style="font-size:var(--text-xs);color:var(--text-tertiary);margin-bottom:6px;padding:6px 8px;background:var(--bg-tertiary);border-radius:var(--radius-sm);"><strong>Q:</strong> ${escapeHtml(best.prompt)}</div>`
                    : '';
                brText.innerHTML = promptHtml + escapeHtml(best.response || best.text || JSON.stringify(best));
            }).catch(() => {
                bestResponseEl.style.display = 'none';
            });
        }

        // Wire per-layer row clicks to expand responses
        panel.querySelectorAll('.detail-layer-row').forEach(row => {
            row.addEventListener('click', async () => {
                const idx = row.dataset.layerIdx;
                const respEl = panel.querySelector(`.detail-layer-response[data-layer-resp="${idx}"]`);
                const wasVisible = respEl.classList.contains('visible');

                // Close all expanded
                panel.querySelectorAll('.detail-layer-response').forEach(r => r.classList.remove('visible'));
                panel.querySelectorAll('.detail-layer-row').forEach(r => r.classList.remove('expanded'));

                if (wasVisible) return;

                // Expand this row
                row.classList.add('expanded');
                respEl.classList.add('visible');

                // Lazy load if not yet loaded
                if (!respEl.dataset.loaded) {
                    respEl.innerHTML = '<em style="color: var(--text-tertiary); font-size: var(--text-xs);">Loading...</em>';
                    const layer = parseInt(row.dataset.layer);
                    const method = row.dataset.method;
                    const component = row.dataset.component;
                    const coef = parseFloat(row.dataset.coef);

                    const responses = await fetchResponseForRun(entry, { layer, method, component, coef });
                    if (responses && responses.length > 0) {
                        respEl.innerHTML = responses.map((r, i) => {
                            const text = escapeHtml(r.response || r.text || JSON.stringify(r));
                            const prompt = r.prompt ? escapeHtml(r.prompt) : '';
                            const meta = r.trait_score != null
                                ? `<div style="font-size:var(--text-xs);color:var(--text-tertiary);margin-bottom:4px;">Response ${i + 1} — trait=${(r.trait_score||0).toFixed(1)} coh=${(r.coherence_score||r.coherence||0).toFixed(0)}</div>`
                                : '';
                            const promptHtml = prompt
                                ? `<div style="font-size:var(--text-xs);color:var(--text-tertiary);margin-bottom:6px;padding:6px 8px;background:var(--bg-tertiary);border-radius:var(--radius-sm);"><strong>Q:</strong> ${prompt}</div>`
                                : '';
                            return `<div style="padding:8px 0;${i > 0 ? 'border-top:1px solid var(--bg-tertiary);' : ''}">${meta}${promptHtml}${text}</div>`;
                        }).join('');
                    } else {
                        respEl.innerHTML = '<em style="color: var(--text-tertiary); font-size: var(--text-xs);">No response saved for this run</em>';
                    }
                    respEl.dataset.loaded = 'true';
                }
            });
        });
    }

    // Smooth-scroll to panel
    panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/** Remove the currently visible detail panel. */
function hideDetailPanel() {
    if (activeDetailPanel) {
        // Clear card selection in the grid
        const grid = activeDetailPanel.parentElement;
        if (grid) grid.querySelectorAll('.trait-card.selected').forEach(c => c.classList.remove('selected'));
        activeDetailPanel.remove();
        activeDetailPanel = null;
    }
}

/** Clear cached tooltip and response data. */
function resetDetailState() {
    tooltipCache = {};
    responseFileCache = {};
    hideDetailPanel();
}

export { showDetailPanel, hideDetailPanel, resetDetailState };
