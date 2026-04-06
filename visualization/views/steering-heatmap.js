// Steering Section 2 — Layer x Coefficient Heatmaps
//
// Dual heatmap visualization showing trait delta and coherence across
// layer x coefficient grid. Includes trait picker dropdown, sweep table,
// and method/interpolation controls.

import { sortedNumericKeys } from '../core/utils.js';
import { getDisplayName, ASYMB_COLORSCALE, DELTA_COLORSCALE } from '../core/display.js';
import { buildChartLayout, renderChart } from '../core/charts.js';
import { scoreClass } from '../core/ui.js';
import { fetchSteeringResults } from './steering-filters.js';
import { extractVectorSpec, extractRunMetrics } from './steering-utils.js';

let currentSweepData = null;
let currentRawResults = null; // Store raw results.jsonl data for method filtering
let selectedSteeringEntry = null; // Selected trait entry for heatmaps (reset on experiment change)


async function renderSweepData(steeringEntry) {
    if (!window.state.experimentData?.name || !steeringEntry) return;

    const results = await fetchSteeringResults(steeringEntry);
    currentRawResults = results;
    const data = results ? convertResultsToSweepFormat(results) : null;

    window._steeringUpdateModelInfo(results);

    if (!data) {
        document.getElementById('sweep-heatmap-delta').innerHTML = '<p class="no-data">No data for this trait</p>';
        document.getElementById('sweep-table-container').innerHTML = '';
        return;
    }

    // Check if we have any single-layer data
    const fullVector = data.full_vector || {};
    if (Object.keys(fullVector).length === 0) {
        document.getElementById('sweep-heatmap-delta').innerHTML = `
            <p class="no-data">No single-layer steering runs found.<br>
            <small>Heatmap requires single-layer runs. Multi-layer runs are not visualized here.</small></p>
        `;
        document.getElementById('sweep-table-container').innerHTML = '';
        return;
    }

    currentSweepData = data;
    updateSweepVisualizations();
}


/** Populate trait dropdown for heatmap section. */
async function renderTraitPicker(steeringEntries) {
    const select = document.getElementById('sweep-trait-select');
    if (!select) return;

    if (!steeringEntries || steeringEntries.length === 0) {
        select.innerHTML = '<option>No traits</option>';
        return;
    }

    // Get current selection or default to first
    const currentFullPath = selectedSteeringEntry?.full_path || steeringEntries[0].full_path;

    // Build options with readable labels
    select.innerHTML = steeringEntries.map((entry, idx) => {
        const displayName = getDisplayName(entry.trait);
        const posDisplay = window.paths.formatPositionDisplay(entry.position);
        // Include prompt_set if not default "steering"
        const promptSetDisplay = entry.prompt_set && entry.prompt_set !== 'steering'
            ? ` [${entry.prompt_set}]`
            : '';
        const label = `${displayName} ${posDisplay}${promptSetDisplay}`;
        const selected = entry.full_path === currentFullPath ? 'selected' : '';
        return `<option value="${idx}" ${selected}>${label}</option>`;
    }).join('');

    // Setup change handler
    select.addEventListener('change', async () => {
        const idx = parseInt(select.value);
        const entry = steeringEntries[idx];
        selectedSteeringEntry = entry;
        await renderSweepData(entry);
    });
}


/** Convert results.jsonl format to sweep visualization format. */
function convertResultsToSweepFormat(results, methodFilter = null) {
    const runs = results.runs || [];
    if (runs.length === 0) return null;
    const baseline = results.baseline?.trait_mean || 50;
    const fullVector = {};

    for (const run of runs) {
        const spec = extractVectorSpec(run);
        if (!spec) continue;
        const { layer, method, coef } = spec;
        if (methodFilter && method !== methodFilter) continue;

        const coefKey = Math.round(coef * 100) / 100;
        if (!fullVector[layer]) fullVector[layer] = { ratios: [], deltas: [], coherences: [], traits: [] };
        const d = fullVector[layer];
        const existingIdx = d.ratios.indexOf(coefKey);
        const { traitScore, coherence, delta } = extractRunMetrics(run.result || {}, baseline);

        if (existingIdx === -1) {
            d.ratios.push(coefKey); d.deltas.push(delta);
            d.coherences.push(coherence); d.traits.push(traitScore);
        } else {
            d.deltas[existingIdx] = delta;
            d.coherences[existingIdx] = coherence;
            d.traits[existingIdx] = traitScore;
        }
    }

    // Sort by coefficient within each layer
    for (const layer of Object.keys(fullVector)) {
        const d = fullVector[layer];
        const order = d.ratios.map((_, i) => i).sort((a, b) => d.ratios[a] - d.ratios[b]);
        fullVector[layer] = {
            ratios: order.map(i => d.ratios[i]), deltas: order.map(i => d.deltas[i]),
            coherences: order.map(i => d.coherences[i]), traits: order.map(i => d.traits[i])
        };
    }

    return { trait: results.trait || 'unknown', baseline_trait: baseline, full_vector: fullVector };
}


function updateSweepVisualizations() {
    if (!currentSweepData) return;

    const method = document.getElementById('sweep-method').value;
    const coherenceThreshold = parseInt(document.getElementById('sweep-coherence-threshold').value);
    const interpolate = document.getElementById('sweep-interpolate').checked;

    // If method filter is active and we have raw results, reconvert with filter
    let data;
    if (method !== 'all' && currentRawResults) {
        const filteredData = convertResultsToSweepFormat(currentRawResults, method);
        data = filteredData?.full_vector || {};
    } else {
        data = currentSweepData.full_vector || {};
    }

    // Render dual heatmaps: Delta (filtered) and Coherence (unfiltered)
    renderSweepHeatmap(data, 'delta', coherenceThreshold, interpolate, 'sweep-heatmap-delta');
    renderSweepHeatmap(data, 'coherence', 0, interpolate, 'sweep-heatmap-coherence');
    renderSweepTable(data, coherenceThreshold);
}


/** Build coefficient grid (handles binning for dense data, interpolation for smooth view). */
function buildCoefficientGrid(ratios, interpolate) {
    const MAX_BINS = 40;
    let binEdges = null, binCenters = null;
    if (ratios.length > MAX_BINS) {
        const logMin = Math.log(Math.min(...ratios) + 1), logMax = Math.log(Math.max(...ratios) + 1);
        binEdges = Array.from({ length: MAX_BINS + 1 }, (_, i) => Math.exp(logMin + (logMax - logMin) * i / MAX_BINS) - 1);
        binCenters = Array.from({ length: MAX_BINS }, (_, i) => (binEdges[i] + binEdges[i + 1]) / 2);
    }
    let interpolatedRatios = ratios;
    if (interpolate && ratios.length > 1) {
        const [minR, maxR] = [Math.min(...ratios), Math.max(...ratios)];
        interpolatedRatios = Array.from({ length: 51 }, (_, i) => minR + (maxR - minR) * i / 50);
    }
    const xRatios = interpolate ? interpolatedRatios : (binCenters || ratios);
    return { binEdges, binCenters, interpolatedRatios, xRatios };
}

/** Build one matrix row for a layer (binned, direct lookup, or interpolated). */
function buildMatrixRow(layerData, metric, coherenceThreshold, ratios, grid, interpolate) {
    const { binEdges, binCenters, interpolatedRatios } = grid;
    const metricKey = metric === 'delta' ? 'deltas' : 'coherences';
    const layerRatios = layerData.ratios || [];
    const layerValues = layerData[metricKey] || [];
    const layerCoherences = layerData.coherences || [];

    const validPoints = [];
    layerRatios.forEach((r, idx) => {
        if (layerCoherences[idx] >= coherenceThreshold) validPoints.push({ r, v: layerValues[idx] });
    });

    if (binEdges && !interpolate) {
        return binCenters.map((_, binIdx) => {
            const pts = validPoints.filter(p => p.r >= binEdges[binIdx] && p.r < binEdges[binIdx + 1]);
            if (pts.length === 0) return null;
            return metric === 'delta' ? pts.reduce((best, p) => Math.abs(p.v) > Math.abs(best) ? p.v : best, 0) : Math.max(...pts.map(p => p.v));
        });
    }
    if (!interpolate) {
        return ratios.map(ratio => {
            const idx = layerRatios.indexOf(ratio);
            if (idx === -1 || layerCoherences[idx] < coherenceThreshold) return null;
            return layerValues[idx];
        });
    }
    if (validPoints.length === 0) return interpolatedRatios.map(() => null);
    validPoints.sort((a, b) => a.r - b.r);
    return interpolatedRatios.map(targetR => {
        let lower = null, upper = null;
        for (const pt of validPoints) {
            if (pt.r <= targetR) lower = pt;
            if (pt.r >= targetR && upper === null) upper = pt;
        }
        if (lower && lower.r === targetR) return lower.v;
        if (upper && upper.r === targetR) return upper.v;
        if (!lower || !upper) return null;
        const t = (targetR - lower.r) / (upper.r - lower.r);
        return lower.v + t * (upper.v - lower.v);
    });
}

function renderSweepHeatmap(data, metric, coherenceThreshold, interpolate = false, containerId = 'sweep-heatmap-delta') {
    const container = document.getElementById(containerId);
    const layers = sortedNumericKeys(data);
    if (layers.length === 0) { container.innerHTML = '<p class="no-data">No layer data available</p>'; return; }

    const allRatios = new Set();
    layers.forEach(layer => (data[layer].ratios || []).forEach(r => allRatios.add(r)));
    const ratios = Array.from(allRatios).sort((a, b) => a - b);
    if (ratios.length === 0) { container.innerHTML = '<p class="no-data">No ratio data available</p>'; return; }

    const grid = buildCoefficientGrid(ratios, interpolate);
    const { binEdges, xRatios } = grid;
    const matrix = layers.map(layer => buildMatrixRow(data[layer], metric, coherenceThreshold, ratios, grid, interpolate));

    // Color scale
    let colorscale, zmid, zmin, zmax;
    if (metric === 'delta') {
        colorscale = DELTA_COLORSCALE; zmid = 0;
        const allVals = matrix.flat().filter(v => v !== null);
        const absMax = Math.max(Math.abs(Math.min(...allVals, 0)), Math.abs(Math.max(...allVals, 0)));
        zmin = -absMax; zmax = absMax;
    } else {
        colorscale = ASYMB_COLORSCALE; zmin = 0; zmax = 100; zmid = 50;
    }

    const metricLabel = metric === 'delta' ? 'Delta' : 'Coherence';
    const hoverText = matrix.map((row, li) => row.map((val, ri) => {
        if (val === null) return '';
        if (binEdges && !interpolate) return `Layer L${layers[li]}<br>Coef: ${binEdges[ri].toFixed(0)}-${binEdges[ri + 1].toFixed(0)}<br>${metricLabel}: ${val.toFixed(1)}<br>(best in bin)`;
        return `Layer L${layers[li]}<br>Coef: ${xRatios[ri].toFixed(0)}<br>${metricLabel}: ${val.toFixed(1)}${interpolate ? '<br>(interpolated)' : ''}`;
    }));

    const trace = {
        z: matrix, x: xRatios.map((_, i) => String(i)), y: layers.map(l => `L${l}`),
        type: 'heatmap', colorscale, zmid, zmin, zmax,
        hoverongaps: false, connectgaps: interpolate,
        hovertemplate: '%{text}<extra></extra>', text: hoverText,
        colorbar: { title: { text: metricLabel, font: { size: 11 } } }
    };

    const numTicks = Math.min(10, xRatios.length);
    const tickIndices = [], tickLabels = [];
    for (let i = 0; i < numTicks; i++) {
        const idx = Math.round(i * (xRatios.length - 1) / (numTicks - 1));
        tickIndices.push(String(idx)); tickLabels.push(xRatios[idx].toFixed(0));
    }

    const layout = buildChartLayout({
        preset: 'heatmap', traces: [trace],
        height: Math.max(300, layers.length * 20 + 100), legendPosition: 'none',
        xaxis: { title: 'Coefficient', tickfont: { size: 10 }, tickvals: tickIndices, ticktext: tickLabels, type: 'category' },
        yaxis: { title: 'Layer', tickfont: { size: 10 }, autorange: 'reversed' },
        margin: { l: 50, r: 80, t: 20, b: 50 }
    });
    renderChart(container, [trace], layout);
}


function renderSweepTable(data, coherenceThreshold) {
    const container = document.getElementById('sweep-table-container');
    const layers = sortedNumericKeys(data);
    if (layers.length === 0) { container.innerHTML = '<p class="no-data">No data available</p>'; return; }

    const rows = [];
    for (const layer of layers) {
        const d = data[layer];
        d.ratios.forEach((ratio, idx) => rows.push({
            layer, ratio, delta: d.deltas[idx], coherence: d.coherences[idx], trait: d.traits?.[idx] ?? null
        }));
    }
    rows.sort((a, b) => b.delta - a.delta);

    container.innerHTML = `<table class="data-table"><thead><tr><th>Layer</th><th>Coef</th><th>Delta</th><th>Coherence</th><th>Trait</th></tr></thead><tbody>${rows.map(r => {
        const dc = r.delta > 15 ? 'quality-good' : r.delta > 5 ? 'quality-ok' : r.delta < 0 ? 'quality-bad' : '';
        return `<tr class="${r.coherence < coherenceThreshold ? 'masked-row' : ''}"><td>L${r.layer}</td><td>${r.ratio.toFixed(2)}</td><td class="${dc}">${r.delta >= 0 ? '+' : ''}${r.delta.toFixed(1)}</td><td class="${scoreClass(r.coherence, 'coherence')}">${r.coherence.toFixed(0)}</td><td>${r.trait !== null ? r.trait.toFixed(1) : 'N/A'}</td></tr>`;
    }).join('')}</tbody></table>`;
}


/** Reset heatmap state. */
function resetHeatmapState() {
    currentSweepData = null;
    currentRawResults = null;
    selectedSteeringEntry = null;
}

/** Get/set the selected steering entry (used by orchestrator). */
function getSelectedSteeringEntry() { return selectedSteeringEntry; }
function setSelectedSteeringEntry(entry) { selectedSteeringEntry = entry; }

export {
    renderSweepData, renderTraitPicker,
    updateSweepVisualizations,
    resetHeatmapState, getSelectedSteeringEntry, setSelectedSteeringEntry
};
