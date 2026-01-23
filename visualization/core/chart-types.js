/**
 * Chart Type Renderers for markdown :::chart::: blocks
 *
 * Usage:
 *   :::chart model-diff-effect path "caption" [traits=...] [height=N]:::
 *
 * Available chart types:
 *   - model-diff-effect: Effect size (Cohen's d) by layer
 *   - model-diff-cosine: Cosine similarity by layer
 */

const CHART_RENDERERS = {};

/**
 * Filter traits from data, excluding random_baseline by default
 * @param {Object} traitsData - traits object from results.json
 * @param {Object} options - { traits: string[]|null, excludeRandom: bool }
 */
function filterTraits(traitsData, options = {}) {
    const { traits = null, excludeRandom = true } = options;
    let filtered = { ...traitsData };

    // Filter to specified traits if provided
    if (traits?.length) {
        filtered = {};
        for (const trait of traits) {
            // Support both full path (rm_hack/secondary_objective) and short name (secondary_objective)
            const match = Object.keys(traitsData).find(
                k => k === trait || k.endsWith('/' + trait)
            );
            if (match) filtered[match] = traitsData[match];
        }
    }

    // Exclude random_baseline by default
    if (excludeRandom) {
        delete filtered['random_baseline'];
    }

    return filtered;
}

/**
 * Get short trait name from full path
 * e.g., "rm_hack/secondary_objective" -> "secondary_objective"
 */
function getTraitShortName(fullPath) {
    return fullPath.split('/').pop();
}

// ============================================================================
// Chart Type: model-diff-effect (Effect size by layer)
// ============================================================================

CHART_RENDERERS['model-diff-effect'] = async function(container, data, options = {}) {
    const { traits: traitFilter, height = 300 } = options;
    const filteredTraits = filterTraits(data.traits, { traits: traitFilter });

    if (Object.keys(filteredTraits).length === 0) {
        container.innerHTML = '<div class="chart-error">No matching traits found</div>';
        return;
    }

    const colors = window.getChartColors?.() || ['#4a9eff', '#ff6b6b', '#51cf66', '#ffd43b', '#cc5de8', '#ff922b'];
    const traces = [];

    Object.entries(filteredTraits).forEach(([traitPath, traitData], idx) => {
        if (!traitData.per_layer_effect_size) return;

        const shortName = getTraitShortName(traitPath);
        const peakEffect = traitData.peak_effect_size?.toFixed(1) || '?';
        const peakLayer = traitData.peak_layer ?? '?';

        traces.push({
            x: traitData.layers,
            y: traitData.per_layer_effect_size,
            type: 'scatter',
            mode: 'lines+markers',
            name: `${shortName} (${peakEffect}σ @ L${peakLayer})`,
            line: { color: colors[idx % colors.length], width: 2 },
            marker: { size: 3 },
            hovertemplate: `${shortName}<br>L%{x}: %{y:.2f}σ<extra></extra>`
        });
    });

    if (traces.length === 0) {
        container.innerHTML = '<div class="chart-error">No effect size data available</div>';
        return;
    }

    const layout = window.buildChartLayout({
        preset: 'layerChart',
        traces,
        height,
        legendPosition: traces.length > 1 ? 'below' : 'none',
        xaxis: { title: { text: 'Layer', standoff: 5 }, dtick: 10, showgrid: true },
        yaxis: { title: 'Effect Size (σ)', zeroline: true, zerolinewidth: 1, showgrid: true }
    });

    const chartDiv = document.createElement('div');
    container.appendChild(chartDiv);
    await window.renderChart(chartDiv, traces, layout);
};

// ============================================================================
// Chart Type: model-diff-cosine (Cosine similarity by layer)
// ============================================================================

CHART_RENDERERS['model-diff-cosine'] = async function(container, data, options = {}) {
    const { traits: traitFilter, height = 250 } = options;
    const filteredTraits = filterTraits(data.traits, { traits: traitFilter });

    if (Object.keys(filteredTraits).length === 0) {
        container.innerHTML = '<div class="chart-error">No matching traits found</div>';
        return;
    }

    const colors = window.getChartColors?.() || ['#4a9eff', '#ff6b6b', '#51cf66', '#ffd43b', '#cc5de8', '#ff922b'];
    const traces = [];

    Object.entries(filteredTraits).forEach(([traitPath, traitData], idx) => {
        if (!traitData.per_layer_cosine_sim) return;

        const shortName = getTraitShortName(traitPath);

        // Find peak cosine similarity
        const cosineSims = traitData.per_layer_cosine_sim;
        const peakIdx = cosineSims.reduce((maxIdx, val, i, arr) =>
            Math.abs(val) > Math.abs(arr[maxIdx]) ? i : maxIdx, 0);
        const peakCos = cosineSims[peakIdx]?.toFixed(2) || '?';
        const peakLayer = traitData.layers[peakIdx] ?? '?';

        traces.push({
            x: traitData.layers,
            y: cosineSims,
            type: 'scatter',
            mode: 'lines+markers',
            name: `${shortName} (${peakCos} @ L${peakLayer})`,
            line: { color: colors[idx % colors.length], width: 2 },
            marker: { size: 3 },
            hovertemplate: `${shortName}<br>L%{x}: %{y:.3f}<extra></extra>`
        });
    });

    if (traces.length === 0) {
        container.innerHTML = '<div class="chart-error">No cosine similarity data available</div>';
        return;
    }

    const layout = window.buildChartLayout({
        preset: 'layerChart',
        traces,
        height,
        legendPosition: traces.length > 1 ? 'below' : 'none',
        xaxis: { title: 'Layer', dtick: 10, showgrid: true },
        yaxis: { title: 'Cosine Similarity', zeroline: true, zerolinewidth: 1, showgrid: true }
    });

    const chartDiv = document.createElement('div');
    container.appendChild(chartDiv);
    await window.renderChart(chartDiv, traces, layout);
};

// ============================================================================
// Chart Type: model-diff-bar (Peak effect size bar chart)
// ============================================================================

CHART_RENDERERS['model-diff-bar'] = async function(container, data, options = {}) {
    const { traits: traitFilter, height = 200 } = options;
    const filteredTraits = filterTraits(data.traits, { traits: traitFilter });

    if (Object.keys(filteredTraits).length === 0) {
        container.innerHTML = '<div class="chart-error">No matching traits found</div>';
        return;
    }

    const colors = window.getChartColors?.() || ['#4a9eff', '#ff6b6b', '#51cf66', '#ffd43b', '#cc5de8', '#ff922b'];

    // Sort by effect size descending
    const sorted = Object.entries(filteredTraits)
        .map(([path, d]) => ({ name: getTraitShortName(path), effect: d.peak_effect_size || 0 }))
        .sort((a, b) => b.effect - a.effect);

    const trace = {
        x: sorted.map(d => d.name),
        y: sorted.map(d => d.effect),
        type: 'bar',
        marker: { color: sorted.map((_, i) => colors[i % colors.length]) },
        text: sorted.map(d => `${d.effect.toFixed(1)}σ`),
        textposition: 'outside',
        hovertemplate: '%{x}: %{y:.2f}σ<extra></extra>'
    };

    const layout = window.buildChartLayout({
        preset: 'barChart',
        traces: [trace],
        height,
        legendPosition: 'none',
        xaxis: { title: '' },
        yaxis: { title: { text: 'Effect Size (σ)', standoff: 5 } },
        margin: { t: 40 }  // Extra top margin for text labels above bars
    });

    const chartDiv = document.createElement('div');
    container.appendChild(chartDiv);
    await window.renderChart(chartDiv, [trace], layout);
};

// ============================================================================
// Main API
// ============================================================================

/**
 * Render a chart of the specified type into a container
 * @param {string} type - Chart type (e.g., 'model-diff-effect')
 * @param {HTMLElement} container - Container element
 * @param {Object} data - Data from JSON file
 * @param {Object} options - { traits: string[]|null, height: number|null }
 */
async function renderChartType(type, container, data, options = {}) {
    const renderer = CHART_RENDERERS[type];
    if (!renderer) {
        container.innerHTML = `<div class="chart-error">Unknown chart type: ${type}</div>`;
        return;
    }
    await renderer(container, data, options);
}

// Export
window.chartTypes = {
    render: renderChartType,
    registry: CHART_RENDERERS
};
