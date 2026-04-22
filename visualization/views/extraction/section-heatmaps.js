/**
 * Per-Trait Heatmaps — grid of small (layer × method) heatmaps.
 *
 * Input:  evalData.all_results (extraction evaluation per layer/method)
 * Output: rendered grid in #trait-heatmaps-container, with metric toggle
 * Usage:  import { renderTraitHeatmaps } from './section-heatmaps.js';
 */

import { getDisplayName, displayLayer } from '../../core/display.js';
import { buildChartLayout, renderChart } from '../../core/charts.js';
import { renderSegmentedControl } from '../../core/ui.js';
import {
    extractionState,
    METRIC_CONFIG,
    computeBestVectors,
    getSelectedTraitNames,
} from './extraction-data.js';

/** Render the trait-heatmaps section: metric toggle + grid + legend. */
function renderTraitHeatmaps(evalData) {
    const container = document.getElementById('trait-heatmaps-container');
    if (!container) return;

    const allResults = evalData.all_results || [];
    if (allResults.length === 0) {
        container.innerHTML = '<p>No results to display.</p>';
        return;
    }

    // Filter by selected traits from sidebar
    const selectedTraitNames = getSelectedTraitNames();
    const results = selectedTraitNames.size > 0
        ? allResults.filter(r => selectedTraitNames.has(r.trait))
        : allResults;

    if (results.length === 0) {
        container.innerHTML = '<p>No results for selected traits.</p>';
        return;
    }

    // Group by trait
    const traitGroups = {};
    results.forEach(r => {
        if (!traitGroups[r.trait]) traitGroups[r.trait] = [];
        traitGroups[r.trait].push(r);
    });

    const traits = Object.keys(traitGroups).sort();

    // Compute best vectors for star indicators
    const bestVectors = computeBestVectors(results);

    const metricToggle = renderSegmentedControl({
        id: 'heatmap-metric-control',
        options: [
            { value: 'effect_size', label: 'Effect Size' },
            { value: 'val_accuracy', label: 'Val Accuracy' },
            { value: 'combined', label: 'Combined' },
        ],
        selected: extractionState.heatmapMetric,
        dataAttr: 'metric',
    });

    const cfg = METRIC_CONFIG[extractionState.heatmapMetric];
    container.innerHTML = `
        <div class="heatmap-metric-toggle">
            <span class="cb-label">Metric:</span>
            ${metricToggle}
        </div>
        <div class="trait-heatmaps-grid" id="heatmaps-grid"></div>
        <div class="heatmap-legend-footer">
            <span class="file-hint">${traits.length} traits</span>
            <span class="file-hint" title="Best layer by effect size">★ = best</span>
            <div class="heatmap-legend">
                <span>${cfg.label}:</span>
                <div>
                    <div class="heatmap-legend-bar ${cfg.legendBarClass}"></div>
                    <div class="heatmap-legend-labels">
                        <span>${cfg.legendLabels[0]}</span>
                        <span>${cfg.legendLabels[1]}</span>
                        <span>${cfg.legendLabels[2]}</span>
                    </div>
                </div>
            </div>
        </div>
    `;

    const grid = document.getElementById('heatmaps-grid');

    // Create compact heatmap for each trait
    traits.forEach(trait => {
        const traitResults = traitGroups[trait];
        const traitId = trait.replace(/\//g, '-');
        const displayName = getDisplayName(trait);
        const bestInfo = bestVectors[trait];

        const traitDiv = document.createElement('div');
        traitDiv.className = 'trait-heatmap-item';
        traitDiv.innerHTML = `
            <h4 title="${displayName}${bestInfo ? ` (best: L${displayLayer(bestInfo.layer)} ${bestInfo.method})` : ''}">${displayName}</h4>
            <div id="heatmap-${traitId}" class="chart-container-sm"></div>
        `;

        grid.appendChild(traitDiv);

        renderSingleTraitHeatmap(traitResults, `heatmap-${traitId}`, bestInfo);
    });

    // Wire metric toggle — re-render heatmaps on change
    const metricControl = document.getElementById('heatmap-metric-control');
    if (metricControl) {
        metricControl.addEventListener('click', (e) => {
            const btn = e.target.closest('button[data-metric]');
            if (!btn || btn.dataset.metric === extractionState.heatmapMetric) return;
            extractionState.heatmapMetric = btn.dataset.metric;
            renderTraitHeatmaps(evalData);
        });
    }
}

/** Render a single trait's compact (layer × method) heatmap. */
function renderSingleTraitHeatmap(traitResults, containerId, bestInfo = null) {
    const methods = ['mean_diff', 'probe'];
    const layers = Array.from(new Set(traitResults.map(r => r.layer))).sort((a, b) => a - b);
    const cfg = METRIC_CONFIG[extractionState.heatmapMetric];

    // Build matrix: layers × methods, value per current metric
    const matrix = [];
    layers.forEach(layer => {
        const row = methods.map(method => {
            const result = traitResults.find(r => r.layer === layer && r.method === method);
            return result ? cfg.computeCell(result) : null;
        });
        matrix.push(row);
    });

    const allValues = matrix.flat().filter(v => v !== null);
    const { zmin, zmax, zmid } = cfg.zRange(allValues);

    const xLabels = ['MD', 'Pr'];

    const trace = {
        z: matrix,
        x: xLabels,
        y: layers,
        type: 'heatmap',
        colorscale: cfg.colorscale,
        hovertemplate: `%{x} L%{y}: ${cfg.hoverSuffix}<extra></extra>`,
        zmin,
        zmax,
        ...(zmid !== undefined ? { zmid } : {}),
        showscale: false
    };

    // Build annotations array
    const annotations = [];
    if (bestInfo && bestInfo.layer !== undefined && bestInfo.method) {
        const methodIdx = methods.indexOf(bestInfo.method);
        const layerIdx = layers.indexOf(bestInfo.layer);
        if (methodIdx >= 0 && layerIdx >= 0) {
            annotations.push({
                x: xLabels[methodIdx],
                y: bestInfo.layer,
                text: '★',
                showarrow: false,
                font: { size: 10, color: '#000' },
                xanchor: 'center',
                yanchor: 'middle'
            });
        }
    }

    const layout = buildChartLayout({
        preset: 'heatmap',
        traces: [trace],
        height: 180,
        legendPosition: 'none',
        xaxis: { tickfont: { size: 8 }, tickangle: 0 },
        yaxis: { showticklabels: false, title: '' },
        margin: { l: 5, r: 5, t: 5, b: 25 },
        annotations
    });
    renderChart(containerId, [trace], layout);
}

export { renderTraitHeatmaps };
