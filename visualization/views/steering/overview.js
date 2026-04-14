// Steering overview — card grid with SVG sparklines
//
// Renders an alphabetically-sorted grid of trait cards. Each card shows the
// trait name, layer range, best delta score, and a multi-line SVG sparkline
// (one polyline per extraction method). Clicking a card opens the detail panel.

import { extractVectorSpec, extractRunMetrics } from './shared.js';

// ── Constants ────────────────────────────────────────────────────

const METHOD_COLORS = {
    probe: '#7ab8e0',
    mean_diff: '#e07ab8',
    gradient: '#b8e07a',
};

// ── State ────────────────────────────────────────────────────────

let selectedTrait = null;
let selectedCard = null;

// ── Helpers ──────────────────────────────────────────────────────

/** Get total model layers from PathBuilder, experiment config, or extraction metadata. */
function getTotalLayers() {
    const fromPaths = window.paths?.getNumLayers?.();
    if (fromPaths) return fromPaths;
    const config = window.state?.experimentData?.experimentConfig;
    if (config?.num_hidden_layers) return config.num_hidden_layers;
    // Fall back to extraction metadata n_layers
    const traits = window.state?.experimentData?.traits;
    if (traits?.[0]?.n_layers) return traits[0].n_layers;
    return 32;
}

/** Extract the short trait name (last segment of category/trait path). */
function shortName(trait) {
    return trait.includes('/') ? trait.split('/').pop() : trait;
}

/** Score class for delta coloring: strong (>20), moderate (5-20), weak (<5). */
function scoreClass(delta) {
    const abs = Math.abs(delta);
    if (abs > 20) return 'strong';
    if (abs > 5) return 'moderate';
    return 'weak';
}

/**
 * Process steering results for a single trait entry.
 * Groups runs by method, finds per-layer best scores, computes overall best.
 *
 * Returns { methods: { [method]: { layers: [{layer, score}], bestDelta, bestLayer } },
 *           baseline, minLayer, maxLayer, bestDelta, bestLayer, bestMethod }
 * or null if no valid data.
 */
function processResults(results, coherenceThreshold) {
    if (!results || !results.runs || results.runs.length === 0) return null;

    const baseline = results.baseline?.trait_mean || 0;
    const methods = {};

    for (const run of results.runs) {
        const spec = extractVectorSpec(run);
        if (!spec) continue;
        const { layer, method, coef } = spec;
        const { traitScore, coherence } = extractRunMetrics(run.result || {}, baseline);

        if (coherence < coherenceThreshold) continue;

        if (!methods[method]) methods[method] = {};
        // Keep the best score per layer per method
        if (!methods[method][layer] || traitScore > methods[method][layer]) {
            methods[method][layer] = traitScore;
        }
    }

    // Convert to sorted layer arrays and find best delta per method
    const methodSummaries = {};
    let overallBestDelta = -Infinity;
    let overallBestLayer = 0;
    let overallBestMethod = '';
    let globalMinLayer = Infinity;
    let globalMaxLayer = -Infinity;

    for (const [method, layerMap] of Object.entries(methods)) {
        const layers = Object.keys(layerMap).map(Number).sort((a, b) => a - b);
        if (layers.length === 0) continue;

        globalMinLayer = Math.min(globalMinLayer, layers[0]);
        globalMaxLayer = Math.max(globalMaxLayer, layers[layers.length - 1]);

        const points = layers.map(l => ({ layer: l, score: layerMap[l] }));
        const deltas = points.map(p => p.score - baseline);
        const bestIdx = deltas.reduce((bi, d, i) => d > deltas[bi] ? i : bi, 0);
        const bestDelta = deltas[bestIdx];
        const bestLayer = points[bestIdx].layer;

        methodSummaries[method] = { points, bestDelta, bestLayer };

        if (bestDelta > overallBestDelta) {
            overallBestDelta = bestDelta;
            overallBestLayer = bestLayer;
            overallBestMethod = method;
        }
    }

    if (Object.keys(methodSummaries).length === 0) return null;

    return {
        methods: methodSummaries,
        baseline,
        minLayer: globalMinLayer,
        maxLayer: globalMaxLayer,
        bestDelta: overallBestDelta,
        bestLayer: overallBestLayer,
        bestMethod: overallBestMethod,
    };
}


// ── SVG Sparkline ────────────────────────────────────────────────

/**
 * Build an inline SVG sparkline string for a processed trait.
 * Shows one polyline per method, a dashed baseline, layer labels at edges,
 * and the baseline score on the left.
 */
function buildSparklineSVG(processed, width, height) {
    const labelH = 10;   // bottom row for layer labels
    const padL = 16;      // left margin for baseline label
    const padR = 2;
    const plotW = width - padL - padR;
    const plotH = height - labelH;

    const { methods, baseline, minLayer, maxLayer } = processed;

    // Collect all scores for y-range
    const allScores = [];
    for (const m of Object.values(methods)) {
        for (const p of m.points) allScores.push(p.score);
    }
    allScores.push(baseline);

    const minS = Math.min(...allScores) - 3;
    const maxS = Math.max(...allScores) + 3;
    const rangeS = maxS - minS || 1;
    const rangeL = maxLayer - minLayer || 1;

    const x = l => padL + ((l - minLayer) / rangeL) * plotW;
    const y = s => plotH - ((s - minS) / rangeS) * plotH;

    const bY = y(baseline);

    let svg = `<svg viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg">`;

    // Dashed baseline
    svg += `<line x1="${padL}" y1="${bY.toFixed(1)}" x2="${padL + plotW}" y2="${bY.toFixed(1)}" stroke="var(--text-tertiary)" stroke-width="0.5" stroke-dasharray="3,2" opacity="0.5"/>`;

    // Method lines
    for (const [method, data] of Object.entries(methods)) {
        const color = METHOD_COLORS[method] || '#888';
        const pts = data.points
            .map(p => `${x(p.layer).toFixed(1)},${y(p.score).toFixed(1)}`)
            .join(' ');
        svg += `<polyline points="${pts}" fill="none" stroke="${color}" stroke-width="1.5" stroke-linejoin="round" opacity="0.9"/>`;
    }

    // Tick marks + layer labels at bottom corners
    const tickY = plotH;
    svg += `<line x1="${padL}" y1="${tickY - 3}" x2="${padL}" y2="${tickY + 1}" stroke="var(--text-tertiary)" stroke-width="0.5" opacity="0.5"/>`;
    svg += `<line x1="${padL + plotW}" y1="${tickY - 3}" x2="${padL + plotW}" y2="${tickY + 1}" stroke="var(--text-tertiary)" stroke-width="0.5" opacity="0.5"/>`;
    svg += `<text x="${padL + 2}" y="${height - 1}" style="font-size:9px;fill:var(--text-tertiary);">${minLayer}</text>`;
    svg += `<text x="${padL + plotW - 2}" y="${height - 1}" text-anchor="end" style="font-size:9px;fill:var(--text-tertiary);">${maxLayer}</text>`;

    // Baseline score label — centered vertically on baseline
    const bLabelY = Math.max(8, Math.min(plotH - 2, bY));
    svg += `<text x="${padL - 2}" y="${bLabelY.toFixed(1)}" text-anchor="end" dominant-baseline="middle" style="font-size:8px;fill:var(--text-tertiary);">${baseline.toFixed(0)}</text>`;

    svg += '</svg>';
    return svg;
}


// ── Card Rendering ───────────────────────────────────────────────

/**
 * Build HTML for a single trait card.
 */
function buildCardHTML(traitPath, processed) {
    const totalLayers = getTotalLayers();
    const name = shortName(traitPath);
    const { bestDelta, bestLayer, minLayer, maxLayer } = processed;
    const cls = scoreClass(bestDelta);
    const sign = bestDelta > 0 ? '+' : '';
    const sparkSVG = buildSparklineSVG(processed, 260, 58);

    return `<div class="trait-card" data-trait="${traitPath}">
    <div class="tc-row">
        <div class="tc-name" title="${traitPath}">${name}</div>
        <div class="tc-layers">L${minLayer}-${maxLayer}/${totalLayers}</div>
    </div>
    <div class="tc-stats">
        <span class="tc-score ${cls}">${sign}${bestDelta.toFixed(1)}</span>
        <span>L${bestLayer}</span>
    </div>
    <div class="tc-sparkline">${sparkSVG}</div>
</div>`;
}

/**
 * Build the legend HTML that sits above the grid.
 */
function buildLegendHTML() {
    return `<div class="steering-legend">
    <span>Overview (alphabetical)</span>
    <span style="margin-left:auto;display:flex;gap:12px;">
        <span><span style="color:var(--success);font-weight:600;">+20</span> strong</span>
        <span><span style="font-weight:600;">&lt; 20</span> weak</span>
    </span>
</div>`;
}


// ── Public API ───────────────────────────────────────────────────

/**
 * Render the overview grid of trait cards with mini SVG sparklines.
 *
 * Fetches results for all traits in parallel, sorts alphabetically,
 * and renders into the given container. Clicking a card calls
 * window._steeringShowDetail(trait, card); clicking a selected card
 * deselects and calls window._steeringHideDetail().
 *
 * @param {HTMLElement} container - DOM element to render into
 * @param {Array} traits - steering entry objects from the API
 * @param {Function} fetchSteeringResultsFn - async (entry) => results
 * @param {number} coherenceThreshold - minimum coherence to include a run
 */
async function renderOverviewGrid(container, traits, fetchSteeringResultsFn, coherenceThreshold) {
    if (!container || !traits || traits.length === 0) {
        if (container) container.innerHTML = '<div class="no-data">No steering data available.</div>';
        return;
    }

    // Fetch all results in parallel
    const resultPairs = await Promise.all(
        traits.map(async (entry) => {
            const results = await fetchSteeringResultsFn(entry);
            const processed = processResults(results, coherenceThreshold);
            return { entry, processed };
        })
    );

    // Filter out traits with no valid data
    const allValid = resultPairs.filter(p => p.processed !== null);

    // Deduplicate by trait name — keep the entry with the best absolute delta
    const bestByTrait = {};
    for (const p of allValid) {
        const key = p.entry.trait;
        if (!bestByTrait[key] || Math.abs(p.processed.bestDelta) > Math.abs(bestByTrait[key].processed.bestDelta)) {
            bestByTrait[key] = p;
        }
    }
    const validPairs = Object.values(bestByTrait)
        .sort((a, b) => a.entry.trait.localeCompare(b.entry.trait));

    if (validPairs.length === 0) {
        container.innerHTML = '<div class="no-data">No traits passed the coherence threshold.</div>';
        return;
    }

    // Store entry lookup for click handler
    const entryByTrait = {};
    validPairs.forEach(p => { entryByTrait[p.entry.trait] = p.entry; });

    // Build HTML — cards directly in container (no wrapper div, so grid-column works)
    const legend = buildLegendHTML();
    const cards = validPairs.map(p => buildCardHTML(p.entry.trait, p.processed)).join('');
    container.innerHTML = `${legend}${cards}`;

    // Wire click handlers
    container.querySelectorAll('.trait-card').forEach(card => {
        card.addEventListener('click', () => {
            const traitPath = card.dataset.trait;
            const entry = entryByTrait[traitPath];

            // Toggle off if already selected
            if (selectedTrait === traitPath) {
                card.classList.remove('selected');
                selectedTrait = null;
                selectedCard = null;
                if (typeof window._steeringHideDetail === 'function') {
                    window._steeringHideDetail();
                }
                return;
            }

            // Deselect previous
            if (selectedCard) {
                selectedCard.classList.remove('selected');
            }

            // Select this one
            card.classList.add('selected');
            selectedTrait = traitPath;
            selectedCard = card;

            if (typeof window._steeringShowDetail === 'function') {
                window._steeringShowDetail(entry, card);
            }
        });
    });
}

/**
 * Clear any cached overview state (selection, etc).
 * Called on experiment change or view reset.
 */
function resetOverviewState() {
    selectedTrait = null;
    selectedCard = null;
}

export { renderOverviewGrid, resetOverviewState };
