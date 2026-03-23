// Steering Section 2 — Layer x Coefficient Heatmaps
//
// Dual heatmap visualization showing trait delta and coherence across
// layer x coefficient grid. Includes trait picker dropdown, sweep table,
// and method/interpolation controls.

import { sortedNumericKeys } from '../core/utils.js';
import { fetchSteeringResults } from './steering-filters.js';

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


/**
 * Populate trait dropdown for heatmap section
 * @param {Array} steeringEntries - Array of { trait, model_variant, position, prompt_set, full_path }
 */
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
        const displayName = window.getDisplayName(entry.trait);
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

    // Group by layer and coefficient
    const fullVector = {};

    runs.forEach(run => {
        const config = run.config || {};
        const result = run.result || {};

        // Support VectorSpec format
        const vectors = config.vectors || [];

        // Only single-vector runs for heatmap
        if (vectors.length !== 1) return;

        const v = vectors[0];
        const layer = v.layer;
        const coef = v.weight;
        const method = v.method;

        // Filter by method if specified
        if (methodFilter && method !== methodFilter) return;

        // Round to avoid floating point duplicates
        const coefKey = Math.round(coef * 100) / 100;

        if (!fullVector[layer]) {
            fullVector[layer] = { ratios: [], deltas: [], coherences: [], traits: [] };
        }

        // Check for duplicates (same layer + coef)
        const existingIdx = fullVector[layer].ratios.indexOf(coefKey);
        const traitScore = result.trait_mean || 0;
        const coherence = result.coherence_mean || 0;
        const delta = traitScore - baseline;

        if (existingIdx === -1) {
            fullVector[layer].ratios.push(coefKey);
            fullVector[layer].deltas.push(delta);
            fullVector[layer].coherences.push(coherence);
            fullVector[layer].traits.push(traitScore);
        } else {
            // Update if this run is newer
            fullVector[layer].deltas[existingIdx] = delta;
            fullVector[layer].coherences[existingIdx] = coherence;
            fullVector[layer].traits[existingIdx] = traitScore;
        }
    });

    // Sort by coefficient within each layer
    Object.keys(fullVector).forEach(layer => {
        const indices = fullVector[layer].ratios.map((_, i) => i);
        indices.sort((a, b) => fullVector[layer].ratios[a] - fullVector[layer].ratios[b]);

        fullVector[layer] = {
            ratios: indices.map(i => fullVector[layer].ratios[i]),
            deltas: indices.map(i => fullVector[layer].deltas[i]),
            coherences: indices.map(i => fullVector[layer].coherences[i]),
            traits: indices.map(i => fullVector[layer].traits[i])
        };
    });

    return {
        trait: results.trait || 'unknown',
        baseline_trait: baseline,
        full_vector: fullVector
    };
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


function renderSweepHeatmap(data, metric, coherenceThreshold, interpolate = false, containerId = 'sweep-heatmap-delta') {
    const container = document.getElementById(containerId);

    const layers = sortedNumericKeys(data);
    if (layers.length === 0) {
        container.innerHTML = '<p class="no-data">No layer data available</p>';
        return;
    }

    // Get all unique ratios across all layers
    const allRatios = new Set();
    layers.forEach(layer => {
        (data[layer].ratios || []).forEach(r => allRatios.add(r));
    });
    let ratios = Array.from(allRatios).sort((a, b) => a - b);

    if (ratios.length === 0) {
        container.innerHTML = '<p class="no-data">No ratio data available</p>';
        return;
    }

    // Bin coefficients if there are too many (>50) for clean visualization
    const MAX_BINS = 40;
    let binEdges = null;
    let binCenters = null;
    if (ratios.length > MAX_BINS) {
        const minR = Math.min(...ratios);
        const maxR = Math.max(...ratios);
        // Use log scale for binning
        const logMin = Math.log(minR + 1);
        const logMax = Math.log(maxR + 1);
        binEdges = [];
        binCenters = [];
        for (let i = 0; i <= MAX_BINS; i++) {
            const logVal = logMin + (logMax - logMin) * i / MAX_BINS;
            binEdges.push(Math.exp(logVal) - 1);
        }
        for (let i = 0; i < MAX_BINS; i++) {
            binCenters.push((binEdges[i] + binEdges[i + 1]) / 2);
        }
    }

    // If interpolating, create a denser grid
    let interpolatedRatios = ratios;
    if (interpolate && ratios.length > 1) {
        const minR = Math.min(...ratios);
        const maxR = Math.max(...ratios);
        const numSteps = 50;
        interpolatedRatios = [];
        for (let i = 0; i <= numSteps; i++) {
            interpolatedRatios.push(minR + (maxR - minR) * i / numSteps);
        }
    }

    // Build matrix
    const metricKey = metric === 'delta' ? 'deltas' : 'coherences';
    const matrix = layers.map(layer => {
        const layerData = data[layer];
        const layerRatios = layerData.ratios || [];
        const layerValues = layerData[metricKey] || [];
        const layerCoherences = layerData.coherences || [];

        // Build lookup of valid (ratio, value) pairs for this layer
        const validPoints = [];
        layerRatios.forEach((r, idx) => {
            const coherence = layerCoherences[idx];
            if (coherence >= coherenceThreshold) {
                validPoints.push({ r, v: layerValues[idx] });
            }
        });

        // If binning, aggregate values per bin
        if (binEdges && !interpolate) {
            return binCenters.map((_, binIdx) => {
                const binMin = binEdges[binIdx];
                const binMax = binEdges[binIdx + 1];
                const binPoints = validPoints.filter(p => p.r >= binMin && p.r < binMax);
                if (binPoints.length === 0) return null;
                if (metric === 'delta') {
                    return binPoints.reduce((best, p) => Math.abs(p.v) > Math.abs(best) ? p.v : best, 0);
                }
                return Math.max(...binPoints.map(p => p.v));
            });
        }

        if (!interpolate) {
            return ratios.map(ratio => {
                const idx = layerRatios.indexOf(ratio);
                if (idx === -1) return null;
                const coherence = layerCoherences[idx];
                if (coherence < coherenceThreshold) return null;
                return layerValues[idx];
            });
        }

        // Interpolation mode
        if (validPoints.length === 0) {
            return interpolatedRatios.map(() => null);
        }

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
    });

    const xRatios = interpolate ? interpolatedRatios : (binCenters || ratios);

    // Determine color scale based on metric (only 'delta' and 'coherence' are used)
    let colorscale, zmid, zmin, zmax;
    if (metric === 'delta') {
        colorscale = window.DELTA_COLORSCALE;
        zmid = 0;
        const allVals = matrix.flat().filter(v => v !== null);
        const absMax = Math.max(Math.abs(Math.min(...allVals, 0)), Math.abs(Math.max(...allVals, 0)));
        zmin = -absMax;
        zmax = absMax;
    } else {
        colorscale = window.ASYMB_COLORSCALE || 'Viridis';
        zmin = 0;
        zmax = 100;
        zmid = 50;
    }

    const xIndices = xRatios.map((_, i) => String(i));

    // Build custom hover text
    const hoverText = matrix.map((row, layerIdx) =>
        row.map((val, ratioIdx) => {
            if (val === null) return '';
            const metricLabel = metric === 'delta' ? 'Delta' : 'Coherence';
            if (binEdges && !interpolate) {
                const binMin = binEdges[ratioIdx];
                const binMax = binEdges[ratioIdx + 1];
                return `Layer L${layers[layerIdx]}<br>Coef: ${binMin.toFixed(0)}-${binMax.toFixed(0)}<br>${metricLabel}: ${val.toFixed(1)}<br>(best in bin)`;
            }
            const coef = xRatios[ratioIdx];
            return `Layer L${layers[layerIdx]}<br>Coef: ${coef.toFixed(0)}<br>${metricLabel}: ${val.toFixed(1)}${interpolate ? '<br>(interpolated)' : ''}`;
        })
    );

    const trace = {
        z: matrix,
        x: xIndices,
        y: layers.map(l => `L${l}`),
        type: 'heatmap',
        colorscale: colorscale,
        zmid: zmid,
        zmin: zmin,
        zmax: zmax,
        hoverongaps: false,
        connectgaps: interpolate,
        hovertemplate: '%{text}<extra></extra>',
        text: hoverText,
        colorbar: {
            title: { text: metric === 'delta' ? 'Delta' : 'Coherence', font: { size: 11 } }
        }
    };

    // Generate evenly-spaced tick positions
    const numTicks = Math.min(10, xRatios.length);
    const tickIndices = [];
    const tickLabels = [];
    for (let i = 0; i < numTicks; i++) {
        const idx = Math.round(i * (xRatios.length - 1) / (numTicks - 1));
        tickIndices.push(String(idx));
        tickLabels.push(xRatios[idx].toFixed(0));
    }

    const layout = window.buildChartLayout({
        preset: 'heatmap',
        traces: [trace],
        height: Math.max(300, layers.length * 20 + 100),
        legendPosition: 'none',
        xaxis: { title: 'Coefficient', tickfont: { size: 10 }, tickvals: tickIndices, ticktext: tickLabels, type: 'category' },
        yaxis: { title: 'Layer', tickfont: { size: 10 }, autorange: 'reversed' },
        margin: { l: 50, r: 80, t: 20, b: 50 }
    });
    window.renderChart(container, [trace], layout);
}


function renderSweepTable(data, coherenceThreshold) {
    const container = document.getElementById('sweep-table-container');

    const layers = sortedNumericKeys(data);
    if (layers.length === 0) {
        container.innerHTML = '<p class="no-data">No data available</p>';
        return;
    }

    // Flatten all results into rows
    const rows = [];
    layers.forEach(layer => {
        const layerData = data[layer];
        layerData.ratios.forEach((ratio, idx) => {
            rows.push({
                layer,
                ratio,
                delta: layerData.deltas[idx],
                coherence: layerData.coherences[idx],
                trait: layerData.traits ? layerData.traits[idx] : null
            });
        });
    });

    // Sort by delta descending
    rows.sort((a, b) => b.delta - a.delta);

    container.innerHTML = `
        <table class="data-table">
            <thead>
                <tr>
                    <th>Layer</th>
                    <th>Coef</th>
                    <th>Delta</th>
                    <th>Coherence</th>
                    <th>Trait</th>
                </tr>
            </thead>
            <tbody>
                ${rows.map(r => {
                    const masked = r.coherence < coherenceThreshold;
                    const deltaClass = r.delta > 15 ? 'quality-good' : r.delta > 5 ? 'quality-ok' : r.delta < 0 ? 'quality-bad' : '';
                    const cohClass = ui.scoreClass(r.coherence, 'coherence');
                    return `
                        <tr class="${masked ? 'masked-row' : ''}">
                            <td>L${r.layer}</td>
                            <td>${r.ratio.toFixed(2)}</td>
                            <td class="${deltaClass}">${r.delta >= 0 ? '+' : ''}${r.delta.toFixed(1)}</td>
                            <td class="${cohClass}">${r.coherence.toFixed(0)}</td>
                            <td>${r.trait !== null ? r.trait.toFixed(1) : 'N/A'}</td>
                        </tr>
                    `;
                }).join('')}
            </tbody>
        </table>
    `;
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
    convertResultsToSweepFormat, updateSweepVisualizations,
    renderSweepHeatmap, renderSweepTable,
    resetHeatmapState, getSelectedSteeringEntry, setSelectedSteeringEntry
};
