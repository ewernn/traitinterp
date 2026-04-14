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
    padding: 2px 6px;
    height: 22px;
    border-radius: var(--radius-sm);
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

/* Best response preview — clean, no colored borders */
.detail-best-response {
    padding: 12px;
    margin-top: 12px;
}
.detail-best-response .br-label {
    font-size: var(--text-xs);
    color: var(--text-tertiary);
    margin-bottom: 4px;
}
.detail-best-response .br-meta {
    font-size: var(--text-xs);
    color: var(--text-tertiary);
    margin-bottom: 8px;
}
.detail-best-response .br-question {
    font-size: var(--text-xs);
    color: var(--text-tertiary);
    margin-bottom: 6px;
}
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
    padding: 8px 0;
}
.detail-layer-results summary:hover { color: var(--text-secondary); }
.detail-layer-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 12px;
    font-size: var(--text-xs);
    color: var(--text-secondary);
    cursor: pointer;
    border-radius: var(--radius-sm);
    margin-bottom: 2px;
    transition: background 0.1s;
}
.detail-layer-row:hover { background: var(--bg-secondary); }
.detail-layer-row.expanded { background: var(--bg-secondary); }
.detail-layer-row .lr-layer { font-weight: 600; min-width: 30px; }
.detail-layer-row .lr-score { min-width: 65px; }
.detail-layer-row .lr-coef { color: var(--text-tertiary); }
.detail-layer-row .lr-coh { color: var(--text-tertiary); }
.detail-layer-row .lr-arrow {
    margin-left: auto;
    color: var(--text-tertiary);
    font-size: 10px;
    transition: transform 0.15s;
}
.detail-layer-row.expanded .lr-arrow { transform: rotate(90deg); }
.detail-layer-response {
    display: none;
    padding: 0 4px 8px;
}
.detail-layer-response.visible { display: flex; flex-direction: column; gap: 8px; max-height: 300px; overflow-y: auto; }
.detail-response-card {
    background: var(--bg-secondary);
    border-radius: var(--radius-sm);
    padding: 12px;
    border: 1px solid var(--border-color);
}
.detail-response-card .rc-meta {
    font-size: var(--text-xs);
    color: var(--text-tertiary);
    margin-bottom: 6px;
}
.detail-response-card .rc-question {
    font-size: var(--text-xs);
    color: var(--text-tertiary);
    margin-bottom: 8px;
}
.detail-response-card .rc-text {
    font-size: var(--text-sm);
    color: var(--text-secondary);
    line-height: 1.6;
    white-space: pre-wrap;
}
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


// ── Plotly chart ───────────────────────────────────────────────────

function renderDetailPlotly(chartDiv, detailData, direction) {
    const methodColors = getMethodColors();
    const traces = [];

    // One trace per method
    for (const [method, points] of Object.entries(detailData.methods)) {
        const sorted = [...points].sort((a, b) => a.layer - b.layer);
        const color = methodColors[method] || '#888';
        traces.push({
            name: method,
            x: sorted.map(d => d.layer),
            y: sorted.map(d => d.score),
            customdata: sorted.map(d => {
                const delta = (d.score - detailData.baseline).toFixed(1);
                const sign = delta >= 0 ? '+' : '';
                return `L${d.layer} ${method}<br>Score: ${d.score.toFixed(1)} (${sign}${delta})<br>Coherence: ${d.coherence != null ? d.coherence.toFixed(0) : '?'}`;
            }),
            type: 'scatter',
            mode: 'lines+markers',
            line: { color, width: 2.5 },
            marker: { size: 5, color },
            hovertemplate: '%{customdata}<extra></extra>',
        });
    }

    // Baseline as a trace (so it shows in legend)
    if (detailData.baseline != null) {
        traces.push({
            name: `Baseline (${detailData.baseline.toFixed(0)})`,
            x: [detailData.minLayer, detailData.maxLayer],
            y: [detailData.baseline, detailData.baseline],
            type: 'scatter',
            mode: 'lines',
            line: { color: 'rgba(200,200,200,0.6)', width: 1.5, dash: 'dash' },
            hoverinfo: 'skip',
        });
    }

    const layout = {
        height: 280,
        margin: { l: 45, r: 16, t: 8, b: 60 },
        xaxis: {
            title: { text: 'Layer', font: { size: 11, color: '#888' } },
            dtick: 1,
            showgrid: false,
            zeroline: false,
            linecolor: 'rgba(150,150,150,0.2)',
            linewidth: 1,
            tickfont: { size: 10, color: '#666' },
        },
        yaxis: {
            title: { text: 'Trait Score', font: { size: 11, color: '#888' } },
            showgrid: true,
            gridcolor: 'rgba(150,150,150,0.1)',
            gridwidth: 1,
            zeroline: false,
            linecolor: 'rgba(150,150,150,0.2)',
            linewidth: 1,
            tickfont: { size: 10, color: '#666' },
        },
        showlegend: true,
        legend: { x: 0, y: -0.25, orientation: 'h', font: { size: 10, color: '#888' } },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: '#ccc', size: 11 },
        hovermode: 'closest',
    };

    Plotly.newPlot(chartDiv, traces, layout, { responsive: true, displayModeBar: false });
}


// ── Data extraction ────────────────────────────────────────────────

/**
 * Extract per-method layer data and per-layer best results from traitResults.
 *
 * @param {Object} traitResults - { runs, baseline }
 * @returns {{ methods, perLayer, baseline, minLayer, maxLayer, bestRun, configs }}
 */
function extractDetailData(traitResults, direction = null, minCoherence = 70) {
    const baseline = traitResults.baseline?.trait_mean || 0;
    const runs = traitResults.runs || [];

    // Per-method: best score per layer
    const methods = {};
    // Per-layer: best overall run (coherence-filtered)
    const perLayerMap = {};
    let minLayer = Infinity;
    let maxLayer = -Infinity;

    for (const run of runs) {
        const spec = extractVectorSpec(run);
        if (!spec) continue;
        const { layer, method, component, coef } = spec;
        const { traitScore, coherence } = extractRunMetrics(run.result || {}, baseline);

        // Skip runs below min coherence for best-run selection
        if (coherence != null && coherence < minCoherence) continue;

        if (layer < minLayer) minLayer = layer;
        if (layer > maxLayer) maxLayer = layer;

        // Per-method aggregation (best score per layer per method)
        // For negative direction, "best" = lowest trait score (furthest from baseline)
        const isBetter = (a, b) => direction === 'negative' ? a < b : a > b;

        if (!methods[method]) methods[method] = {};
        if (!methods[method][layer] || isBetter(traitScore, methods[method][layer].score)) {
            methods[method][layer] = { score: traitScore, layer, coherence };
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
    const slider = document.getElementById('sweep-coherence-threshold');
    const minCoherence = slider ? parseInt(slider.value, 10) : 70;
    const data = extractDetailData(traitResults, direction, minCoherence);
    const { methods, perLayer, baseline, minLayer, maxLayer, bestRun, configCount } = data;
    const traitName = trait.split('/').pop();
    const methodColors = getMethodColors();

    // Score class helper
    const deltaClass = (delta) => Math.abs(delta) > 20 ? 'good' : Math.abs(delta) > 5 ? 'moderate' : '';

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

    // Tooltip placeholders (content loaded on hover via fetch)
    const tooltipHTML = `
        <div class="detail-info-icons">
            <div class="detail-info-icon" data-tooltip="definition">Def
                <div class="detail-tooltip"><div class="tt-label">Scoring Definition</div><em>Hover to load...</em></div>
            </div>
            <div class="detail-info-icon" data-tooltip="steering">Qs
                <div class="detail-tooltip"><div class="tt-label">Steering Questions</div><em>Hover to load...</em></div>
            </div>
            <div class="detail-info-icon" data-tooltip="coherence">Coh
                <div class="detail-tooltip"><div class="tt-label">Coherence Prompt</div><em>Hover to load...</em></div>
            </div>
        </div>`;

    // Best response section (placeholder, loaded async)
    const bestResponseHTML = bestRun ? `
        <div class="detail-best-response" data-layer="${bestRun.layer}" data-method="${bestRun.method}" data-component="${bestRun.component}" data-coef="${bestRun.coef}">
            <div class="br-label">Best response (${direction === 'negative' ? 'lowest' : 'highest'} trait score) <span class="subsection-info-toggle" data-target="info-steering-best-response">\u25BA</span></div>
            <div class="subsection-info" id="info-steering-best-response">The strongest coherent steering run for this trait &mdash; best layer, coef, score, coherence, plus the actual prompt and model output. Use it to eyeball whether the score reflects real behaviour.</div>
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
    const layerRowsHTML = perLayer.map((r, i) => {
        return `
            <div class="detail-layer-row" data-layer-idx="${i}" data-layer="${r.layer}" data-method="${r.method}" data-component="${r.component}" data-coef="${r.coef}">
                <span class="lr-layer">L${r.layer}</span>
                <span class="lr-score">trait ${r.traitScore.toFixed(1)}</span>
                <span class="lr-coef">coef ${Math.abs(r.coef).toFixed(1)}</span>
                <span class="lr-coh">coh ${r.coherence.toFixed(0)}</span>
                <span class="lr-arrow">▸</span>
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
            <span>Layers: <strong>L${minLayer}&ndash;${maxLayer}</strong> / ${window.paths?.getNumLayers?.() || window.state.experimentData?.experimentConfig?.num_hidden_layers || 32}</span>
            <span>Configs: <strong>${configCount}</strong></span>
        </div>
        <div class="detail-chart-header">
            <span>Steering Effect by Layer</span>
            <span class="subsection-info-toggle" data-target="info-steering-detail-chart">\u25BA</span>
        </div>
        <div class="subsection-info" id="info-steering-detail-chart">Trait score (judge, 0&ndash;100) per injection layer, one line per extraction method. Dashed line = unsteered baseline. Distance from baseline = steering strength.</div>
        <div class="detail-chart" id="detail-chart-${traitName}"></div>
        <div class="detail-baseline-response" style="display:none;">
            <div class="br-label">Baseline response (no steering)</div>
            <div class="br-text"><span class="br-loading">Loading...</span></div>
        </div>
        ${bestResponseHTML}
        <details class="detail-layer-results">
            <summary>All layers (${perLayer.length} results, sorted by score) <span class="subsection-info-toggle" data-target="info-steering-layer-results">\u25BA</span></summary>
            <div class="subsection-info" id="info-steering-layer-results">All layers ranked by trait score, with the coefficient and coherence chosen by the search. Click a row to see the per-prompt responses behind the score.</div>
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

    // ── Render Plotly chart ──
    const direction = traitResults.direction || 'positive';
    const slider = document.getElementById('sweep-coherence-threshold');
    const minCoherence = slider ? parseInt(slider.value, 10) : 70;
    const detailData = extractDetailData(traitResults, direction, minCoherence);
    const traitName = trait.split('/').pop();
    const chartDiv = panel.querySelector(`#detail-chart-${traitName}`);
    if (chartDiv && detailData.perLayer.length > 0) {
        renderDetailPlotly(chartDiv, detailData, direction);
    }

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

    // ── Load baseline response ──
    const baselineEl = panel.querySelector('.detail-baseline-response');
    if (baselineEl) {
        const experiment = window.state.experimentData?.name;
        const basePath = window.paths?.get('steering.responses', {
            experiment, trait: entry.trait, model_variant: entry.model_variant,
            position: entry.position, prompt_set: entry.prompt_set,
        });
        if (basePath) {
            fetch(`/${basePath}/baseline.json`).then(r => r.ok ? r.json() : null).then(responses => {
                if (!responses || responses.length === 0) return;
                baselineEl.style.display = 'block';
                // Pick the response with the highest trait score to show the "best baseline"
                const best = responses.reduce((a, b) => (b.trait_score || 0) > (a.trait_score || 0) ? b : a, responses[0]);
                const promptLine = best.prompt
                    ? `<div class="br-question">Q: ${escapeHtml(best.prompt)}</div>` : '';
                baselineEl.querySelector('.br-text').innerHTML =
                    `<div class="br-meta">trait=${(best.trait_score||0).toFixed(1)} · coherence=${(best.coherence_score||0).toFixed(0)}</div>`
                    + promptLine + `<div style="margin-top:4px;">${escapeHtml(best.response || '')}</div>`;
            }).catch(() => {});
        }
    }

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
                const promptLine = best.prompt
                    ? `<div class="br-question">Q: ${escapeHtml(best.prompt)}</div>`
                    : '';
                brText.innerHTML = promptLine + escapeHtml(best.response || best.text || JSON.stringify(best));
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
                                ? `<div class="rc-meta">trait ${(r.trait_score||0).toFixed(1)} · coh ${(r.coherence_score||r.coherence||0).toFixed(0)}</div>`
                                : '';
                            const promptLine = prompt
                                ? `<div class="rc-question">Q: ${prompt}</div>`
                                : '';
                            return `<div class="detail-response-card">${meta}${promptLine}<div class="rc-text">${text}</div></div>`;
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
