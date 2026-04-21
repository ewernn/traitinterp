import { getChartColors, displayLayer } from '../core/display.js';
import { buildChartLayout, renderChart } from '../core/charts.js';
import { fetchJSON } from '../core/utils.js';

/**
 * Chart Type Renderers for markdown :::chart::: blocks
 *
 * Usage:
 *   :::chart type path "caption" [traits=...] [height=N] [perplexity=path] [projections=path,path]:::
 *
 * Available chart types:
 *   Model diff charts:
 *   - model-diff-effect: Effect size (Cohen's d) by layer
 *   - model-diff-cosine: Cosine similarity by layer
 *   - model-diff-bar: Peak effect size bar chart
 *   - annotation-stacked: Stacked bar from annotation files
 *   - comparison-bar: Horizontal bar chart for component/method comparison
 *   - simple-bar: Generic horizontal bar from {bars: [{label, value}], xaxis?, suffix?}
 *
 *   Prefill dynamics charts:
 *   - dynamics-effect: Smoothness effect by layer (+ optional projection stability)
 *   - dynamics-scatter: Smoothness vs perplexity correlation (requires perplexity=path)
 *   - dynamics-violin: Split violin distribution of smoothness
 *   - dynamics-position: Effect size by token position
 *
 *   Generic charts (reusable for any finding):
 *   - scatter: x/y scatter with optional regression, labels, groups
 *   - heatmap: 2D color matrix with optional value labels
 *   - bar: horizontal or vertical bar chart with sorting
 *   - line: multi-series line chart
 *
 *   Style presets (append style=ant to :::chart::: block):
 *   - default: standard traitinterp theme
 *   - ant: Anthropic paper aesthetic (coral titles, Georgia serif, steel-blue points)
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
 * Get short trait name from full path, with optional label override
 * e.g., "rm_hack/secondary_objective" -> "secondary_objective"
 * With labels={'secondary_objective': 'secondary objective'}: -> "secondary objective"
 */
function getTraitShortName(fullPath, labels = null) {
    const shortName = fullPath.split('/').pop();
    const override = labels?.[shortName];
    return override ? override.replace(/_/g, ' ') : shortName;
}

// ============================================================================
// Chart Type: model-diff-layer (shared renderer for per-layer line charts)
// Parameterized by `field` option: 'per_layer_effect_size' or 'per_layer_cosine_sim'
// ============================================================================

const MODEL_DIFF_LAYER_DEFAULTS = {
    per_layer_trait_delta: {
        yaxis: '|Trait Score Delta|',
        hoverFmt: '.1f',
        hoverSuffix: '',
        emptyMsg: 'No trait delta data available',
        defaultHeight: 300,
        buildName: (shortName, traitData, field) => {
            const values = traitData[field];
            const peakIdx = values.reduce((maxIdx, val, i, arr) =>
                val > arr[maxIdx] ? i : maxIdx, 0);
            const peakVal = values[peakIdx]?.toFixed(1) || '?';
            const peakLayer = traitData.layers[peakIdx] ?? '?';
            return `${shortName} (${peakVal} @ L${peakLayer})`;
        }
    },
    per_layer_effect_size: {
        yaxis: 'Effect Size (σ)',
        hoverFmt: '.2f',
        hoverSuffix: 'σ',
        emptyMsg: 'No effect size data available',
        defaultHeight: 300,
        buildName: (shortName) => shortName
    },
    per_layer_cosine_sim: {
        yaxis: 'Cosine Similarity',
        hoverFmt: '.3f',
        hoverSuffix: '',
        emptyMsg: 'No cosine similarity data available',
        defaultHeight: 250,
        buildName: (shortName, traitData, field) => {
            const values = traitData[field];
            const peakIdx = values.reduce((maxIdx, val, i, arr) =>
                Math.abs(val) > Math.abs(arr[maxIdx]) ? i : maxIdx, 0);
            const peakVal = values[peakIdx]?.toFixed(2) || '?';
            const peakLayer = traitData.layers[peakIdx] ?? '?';
            return `${shortName} (${peakVal} @ L${peakLayer})`;
        }
    }
};

async function renderModelDiffLayer(container, data, options = {}) {
    const field = options.field || 'per_layer_effect_size';
    const config = MODEL_DIFF_LAYER_DEFAULTS[field];
    if (!config) {
        container.innerHTML = `<div class="chart-error">Unknown field: ${field}</div>`;
        return;
    }

    const { traits: traitFilter, labels: labelOverrides = null, height = config.defaultHeight } = options;
    const filteredTraits = filterTraits(data.traits, { traits: traitFilter });

    if (Object.keys(filteredTraits).length === 0) {
        container.innerHTML = '<div class="chart-error">No matching traits found</div>';
        return;
    }

    const colors = getChartColors();
    const traces = [];

    Object.entries(filteredTraits).forEach(([traitPath, traitData], idx) => {
        if (!traitData[field]) return;

        const shortName = getTraitShortName(traitPath, labelOverrides);
        const name = config.buildName(shortName, traitData, field);

        traces.push({
            x: traitData.layers,
            y: traitData[field],
            type: 'scatter',
            mode: 'lines+markers',
            name,
            line: { color: colors[idx % colors.length], width: 2 },
            marker: { size: 3 },
            hovertemplate: `${shortName}<br>L%{x}: %{y:${config.hoverFmt}}${config.hoverSuffix}<extra></extra>`
        });
    });

    if (traces.length === 0) {
        container.innerHTML = `<div class="chart-error">${config.emptyMsg}</div>`;
        return;
    }

    const layout = buildChartLayout({
        preset: 'layerChart',
        traces,
        height,
        legendPosition: traces.length > 1 ? 'below' : 'none',
        xaxis: { title: { text: 'Layer', standoff: 5 }, dtick: 10, showgrid: true },
        yaxis: { title: config.yaxis, zeroline: true, zerolinewidth: 1, showgrid: true }
    });

    const chartDiv = document.createElement('div');
    container.appendChild(chartDiv);
    await renderChart(chartDiv, traces, layout);
}

// Thin aliases so markdown :::chart::: blocks don't need updating
CHART_RENDERERS['model-diff-effect'] = (container, data, options = {}) =>
    renderModelDiffLayer(container, data, { ...options, field: 'per_layer_effect_size' });

CHART_RENDERERS['model-diff-cosine'] = (container, data, options = {}) =>
    renderModelDiffLayer(container, data, { ...options, field: 'per_layer_cosine_sim' });

CHART_RENDERERS['model-diff-trait-delta'] = (container, data, options = {}) =>
    renderModelDiffLayer(container, data, { ...options, field: 'per_layer_trait_delta' });

// ============================================================================
// Chart Type: model-diff-bar (Peak effect size bar chart)
// ============================================================================

CHART_RENDERERS['model-diff-bar'] = async function(container, data, options = {}) {
    const { traits: traitFilter, labels: labelOverrides = null, height = 200 } = options;
    const filteredTraits = filterTraits(data.traits, { traits: traitFilter });

    if (Object.keys(filteredTraits).length === 0) {
        container.innerHTML = '<div class="chart-error">No matching traits found</div>';
        return;
    }

    const colors = getChartColors();

    // Sort by effect size ascending (so highest appears at top in horizontal bar)
    const sorted = Object.entries(filteredTraits)
        .map(([path, d]) => ({ name: getTraitShortName(path, labelOverrides), effect: d.peak_effect_size || 0 }))
        .sort((a, b) => a.effect - b.effect);

    const trace = {
        x: sorted.map(d => d.effect),
        y: sorted.map(d => d.name),
        type: 'bar',
        orientation: 'h',
        marker: { color: sorted.map((_, i) => colors[(sorted.length - 1 - i) % colors.length]) },
        text: sorted.map(d => `${d.effect.toFixed(1)}σ`),
        textposition: 'outside',
        cliponaxis: false,  // Don't clip text labels at axis bounds
        hovertemplate: '%{y}: %{x:.2f}σ<extra></extra>'
    };

    const layout = buildChartLayout({
        preset: 'barChart',
        traces: [trace],
        height,
        legendPosition: 'none',
        xaxis: { title: { text: 'Effect Size (σ)', standoff: 5 } },
        yaxis: { title: '' },
        margin: { l: 140, r: 100 },  // Left margin for labels, right for text labels
        bargap: 0.3
    });

    const chartDiv = document.createElement('div');
    container.appendChild(chartDiv);
    await renderChart(chartDiv, [trace], layout);
};

// ============================================================================
// Chart Type: annotation-stacked (Stacked bar chart from annotation files)
// ============================================================================

/**
 * Count annotation spans by category
 * @param {Object} annotationsData - Parsed annotations JSON with { annotations: [...] }
 * @returns {Object} - { category: count }
 */
function countByCategory(annotationsData) {
    const counts = {};
    const annotations = annotationsData.annotations || [];

    for (const ann of annotations) {
        for (const span of (ann.spans || [])) {
            const cat = span.category || 'unknown';
            counts[cat] = (counts[cat] || 0) + 1;
        }
    }

    return counts;
}

const BLUE_GRADIENT = ['#1a5fb4', '#3584e4', '#62a0ea', '#99c1f1', '#c0d8f0'];

CHART_RENDERERS['annotation-stacked'] = async function(container, bars, options = {}) {
    const { height = 280, colors: colorScheme = null } = options;
    const colors = colorScheme === 'blue' ? BLUE_GRADIENT : getChartColors();

    // Fetch all annotation files
    const barData = [];
    const allCategories = new Set();

    for (const bar of bars) {
        const data = await fetchJSON(bar.path);
        if (!data) {
            container.innerHTML = `<div class="chart-error">Failed to load ${bar.path}</div>`;
            return;
        }
        const counts = countByCategory(data);

        for (const cat of Object.keys(counts)) {
            allCategories.add(cat);
        }

        barData.push({ label: bar.label, counts });
    }

    if (barData.length === 0) {
        container.innerHTML = '<div class="chart-error">No data to display</div>';
        return;
    }

    // Sort categories by total count (descending) for better visualization
    const categoryTotals = {};
    for (const cat of allCategories) {
        categoryTotals[cat] = barData.reduce((sum, b) => sum + (b.counts[cat] || 0), 0);
    }
    const sortedCategories = [...allCategories].sort((a, b) => categoryTotals[b] - categoryTotals[a]);

    // Build stacked bar traces (one trace per category)
    // Format category names for display (e.g., birth_death_dates -> Birth Death Dates)
    const traces = sortedCategories.map((cat, idx) => {
        const displayName = cat.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
        return {
            x: barData.map(b => b.label),
            y: barData.map(b => b.counts[cat] || 0),
            type: 'bar',
            name: displayName,
            marker: { color: colors[idx % colors.length] },
            hovertemplate: `%{x}<br>${displayName}: %{y}<extra></extra>`
        };
    });

    const layout = buildChartLayout({
        preset: 'barChart',
        traces,
        height,
        legendPosition: 'below',
        xaxis: { title: '' },
        yaxis: { title: { text: 'Count', standoff: 5 } },
        barmode: 'stack',
        bargap: 0.6  // Narrower bars
    });

    const chartDiv = document.createElement('div');
    container.appendChild(chartDiv);
    await renderChart(chartDiv, traces, layout);
};

// ============================================================================
// Chart Type: comparison-bar (Horizontal bar chart for component/method comparison)
// ============================================================================

CHART_RENDERERS['comparison-bar'] = async function(container, data, options = {}) {
    const { height = 200 } = options;
    const colors = getChartColors();
    const direction = data.direction || 'positive';
    const sign = direction === 'positive' ? 1 : -1;

    const results = data.results || [];
    if (results.length === 0) {
        container.innerHTML = '<div class="chart-error">No data to display</div>';
        return;
    }

    // Negate deltas for negative direction so stronger suppression shows as positive bars
    const displayDeltas = results.map(d => d.delta * sign);

    // Sort by display delta ascending (highest at top for horizontal bars)
    const indices = results.map((_, i) => i);
    indices.sort((a, b) => displayDeltas[a] - displayDeltas[b]);

    // Build labels with method/layer info
    const labels = indices.map(i => {
        const d = results[i];
        const methodShort = d.method === 'mean_diff' ? 'md' : d.method.slice(0, 2);
        return `${d.label} (${methodShort} L${displayLayer(d.layer)})`;
    });

    const prefix = direction === 'positive' ? '+' : '';
    const axisLabel = direction === 'positive' ? 'Delta (trait score increase)' : 'Delta (trait score suppression)';

    const trace = {
        x: indices.map(i => displayDeltas[i]),
        y: labels,
        type: 'bar',
        orientation: 'h',
        marker: {
            color: indices.map((_, j) => colors[(indices.length - 1 - j) % colors.length])
        },
        text: indices.map(i => `${prefix}${displayDeltas[i].toFixed(1)}`),
        textposition: 'outside',
        cliponaxis: false,  // Don't clip text labels at axis bounds
        hovertemplate: `%{y}<br>Delta: ${prefix}%{x:.1f}<extra></extra>`
    };

    const layout = buildChartLayout({
        preset: 'barChart',
        traces: [trace],
        height,
        legendPosition: 'none',
        xaxis: { title: { text: axisLabel, standoff: 5 } },
        yaxis: { title: '' },
        margin: { l: 180, r: 60 },
        bargap: 0.3
    });

    const chartDiv = document.createElement('div');
    container.appendChild(chartDiv);
    await renderChart(chartDiv, [trace], layout);
};

// ============================================================================
// Chart Type: dynamics-effect (Smoothness + projection stability by layer)
// ============================================================================

CHART_RENDERERS['dynamics-effect'] = async function(container, data, options = {}) {
    const { height = 350, projections: projectionPaths } = options;
    const traces = [];

    // Raw smoothness from activation_metrics.json
    if (data?.summary?.by_layer) {
        const byLayer = data.summary.by_layer;
        const layers = Object.keys(byLayer).map(Number).sort((a, b) => a - b);
        traces.push({
            x: layers,
            y: layers.map(l => byLayer[l].smoothness_cohens_d),
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Raw Smoothness',
            line: { color: '#4a9eff', width: 3 },
            marker: { size: 7 },
            hovertemplate: 'L%{x}: d=%{y:.2f}<extra></extra>'
        });
    }

    // Projection stability (fetched from options.projections if provided)
    const projColors = { refusal: '#51cf66', sycophancy: '#9775fa' };
    if (projectionPaths) {
        for (const [trait, path] of Object.entries(projectionPaths)) {
            const projData = await fetchJSON(path);
            if (!projData?.by_layer) continue;

            const layers = Object.keys(projData.by_layer).map(Number).sort((a, b) => a - b);
            traces.push({
                x: layers,
                y: layers.map(l => projData.by_layer[l].var_cohens_d),
                type: 'scatter',
                mode: 'lines+markers',
                name: `Projection (${trait})`,
                line: { color: projColors[trait] || '#888', width: 2, dash: 'dash' },
                marker: { size: 5, symbol: 'square' },
                hovertemplate: `${trait}<br>L%{x}: d=%{y:.2f}<extra></extra>`
            });
        }
    }

    // Reference line at d=0.8
    const maxLayer = traces[0]?.x?.slice(-1)[0] || 25;
    traces.push({
        x: [0, maxLayer],
        y: [0.8, 0.8],
        type: 'scatter',
        mode: 'lines',
        line: { color: '#888', dash: 'dot', width: 1 },
        name: 'Large effect (d=0.8)',
        hoverinfo: 'skip'
    });

    const layout = buildChartLayout({
        preset: 'layerChart',
        traces,
        height,
        legendPosition: 'above',
        xaxis: { title: { text: 'Layer', standoff: 5 } },
        yaxis: { title: { text: "Cohen's d", standoff: 5 } }
    });

    const chartDiv = document.createElement('div');
    container.appendChild(chartDiv);
    await renderChart(chartDiv, traces, layout);
};

// ============================================================================
// Chart Type: dynamics-scatter (Smoothness vs perplexity correlation)
// ============================================================================

CHART_RENDERERS['dynamics-scatter'] = async function(container, data, options = {}) {
    const { height = 300, perplexityPath } = options;

    if (!data?.samples) {
        container.innerHTML = '<div class="chart-error">No sample data</div>';
        return;
    }

    // Fetch perplexity data if path provided
    let pplData = null;
    if (perplexityPath) {
        pplData = await fetchJSON(perplexityPath);
    }

    if (!pplData?.results) {
        container.innerHTML = '<div class="chart-error">No perplexity data</div>';
        return;
    }

    // Build scatter data
    const x = [], y = [], text = [];
    for (const ppl of pplData.results) {
        const sample = data.samples.find(s => s.id === ppl.id);
        if (!sample) continue;

        const human = sample.human || sample.a;
        const model = sample.model || sample.b;
        if (!human || !model) continue;

        const layers = Object.keys(human).map(Number);
        const humanSmooth = layers.reduce((sum, l) => sum + human[l].smoothness, 0) / layers.length;
        const modelSmooth = layers.reduce((sum, l) => sum + model[l].smoothness, 0) / layers.length;
        const smoothDiff = humanSmooth - modelSmooth;

        x.push(smoothDiff);
        y.push(ppl.ce_diff);
        text.push(`Sample ${ppl.id}<br>Δsmooth: ${smoothDiff.toFixed(1)}<br>ΔCE: ${ppl.ce_diff.toFixed(2)}`);
    }

    // Linear regression
    const n = x.length;
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((acc, xi, i) => acc + xi * y[i], 0);
    const sumX2 = x.reduce((acc, xi) => acc + xi * xi, 0);
    const sumY2 = y.reduce((acc, yi) => acc + yi * yi, 0);
    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;
    const r = (n * sumXY - sumX * sumY) / Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));

    const minX = Math.min(...x), maxX = Math.max(...x);

    const traces = [
        {
            x, y, text,
            type: 'scatter',
            mode: 'markers',
            marker: { color: '#4a9eff', size: 8, opacity: 0.7 },
            hoverinfo: 'text',
            name: 'Samples'
        },
        {
            x: [minX, maxX],
            y: [slope * minX + intercept, slope * maxX + intercept],
            type: 'scatter',
            mode: 'lines',
            line: { color: '#ff6b6b', width: 2 },
            name: `r = ${r.toFixed(2)}`
        }
    ];

    const layout = buildChartLayout({
        preset: 'layerChart',
        traces,
        height,
        legendPosition: 'above',
        xaxis: { title: { text: 'Smoothness Diff (Prefilled - Model)', standoff: 5 } },
        yaxis: { title: { text: 'Cross-Entropy Diff', standoff: 5 } }
    });

    const chartDiv = document.createElement('div');
    container.appendChild(chartDiv);
    await renderChart(chartDiv, traces, layout);
};

// ============================================================================
// Chart Type: dynamics-violin (Split violin distribution)
// ============================================================================

CHART_RENDERERS['dynamics-violin'] = async function(container, data, options = {}) {
    const { height = 300, metric = 'smoothness' } = options;

    if (!data?.samples?.length) {
        container.innerHTML = '<div class="chart-error">No sample data</div>';
        return;
    }

    // Compute mean across layers for each sample
    const humanVals = data.samples.map(s => {
        const d = s.human || s.a;
        if (!d) return null;
        const layers = Object.keys(d).map(Number);
        return layers.reduce((sum, l) => sum + d[l][metric], 0) / layers.length;
    }).filter(v => v !== null);

    const modelVals = data.samples.map(s => {
        const d = s.model || s.b;
        if (!d) return null;
        const layers = Object.keys(d).map(Number);
        return layers.reduce((sum, l) => sum + d[l][metric], 0) / layers.length;
    }).filter(v => v !== null);

    const traces = [
        {
            y: humanVals,
            x: humanVals.map(() => 0),
            type: 'violin',
            name: 'Prefilled',
            side: 'negative',
            line: { color: '#ff6b6b' },
            fillcolor: 'rgba(255, 107, 107, 0.5)',
            meanline: { visible: true },
            points: false
        },
        {
            y: modelVals,
            x: modelVals.map(() => 0),
            type: 'violin',
            name: 'Model Generated',
            side: 'positive',
            line: { color: '#51cf66' },
            fillcolor: 'rgba(81, 207, 102, 0.5)',
            meanline: { visible: true },
            points: false
        }
    ];

    const layout = buildChartLayout({
        preset: 'barChart',
        traces,
        height,
        legendPosition: 'above',
        xaxis: { showticklabels: false, zeroline: false },
        yaxis: { title: { text: metric.charAt(0).toUpperCase() + metric.slice(1), standoff: 5 } }
    });

    const chartDiv = document.createElement('div');
    container.appendChild(chartDiv);
    await renderChart(chartDiv, traces, layout);
};

// ============================================================================
// Chart Type: dynamics-position (Effect by token position)
// ============================================================================

CHART_RENDERERS['dynamics-position'] = async function(container, data, options = {}) {
    const { height = 280 } = options;

    if (!data?.by_position) {
        container.innerHTML = '<div class="chart-error">No position data</div>';
        return;
    }

    // Sort position ranges by start index
    const positions = Object.keys(data.by_position).sort((a, b) => {
        const startA = parseInt(a.split('-')[0]);
        const startB = parseInt(b.split('-')[0]);
        return startA - startB;
    });

    const cohensD = positions.map(p => data.by_position[p].cohens_d);

    const trace = {
        x: positions,
        y: cohensD,
        type: 'bar',
        marker: {
            color: cohensD.map(d => d > 0.5 ? '#51cf66' : d > 0.2 ? '#ffd43b' : '#868e96')
        },
        text: cohensD.map(d => `d=${d.toFixed(2)}`),
        textposition: 'outside',
        hovertemplate: 'Position %{x}<br>d = %{y:.2f}<extra></extra>'
    };

    const layout = buildChartLayout({
        preset: 'barChart',
        traces: [trace],
        height,
        legendPosition: 'none',
        xaxis: { title: { text: 'Token Position Range', standoff: 5 } },
        yaxis: { title: { text: "Cohen's d", standoff: 5 } },
        bargap: 0.3
    });

    const chartDiv = document.createElement('div');
    container.appendChild(chartDiv);
    await renderChart(chartDiv, [trace], layout);
};

// ============================================================================
// Chart Type: simple-bar (Generic horizontal bar chart from {bars: [{label, value}], xaxis?})
// ============================================================================

CHART_RENDERERS['simple-bar'] = async function(container, data, options = {}) {
    const { height = 200 } = options;
    const colors = getChartColors();

    const bars = data.bars || [];
    if (bars.length === 0) {
        container.innerHTML = '<div class="chart-error">No data to display</div>';
        return;
    }

    // Sort ascending (highest at top for horizontal bars)
    const sorted = [...bars].sort((a, b) => a.value - b.value);

    const suffix = data.suffix || '';
    const trace = {
        x: sorted.map(d => d.value),
        y: sorted.map(d => d.label),
        type: 'bar',
        orientation: 'h',
        marker: { color: sorted.map((_, i) => colors[(sorted.length - 1 - i) % colors.length]) },
        text: sorted.map(d => `${d.value}${suffix}`),
        textposition: 'outside',
        cliponaxis: false,
        hovertemplate: `%{y}: %{x}${suffix}<extra></extra>`
    };

    const layout = buildChartLayout({
        preset: 'barChart',
        traces: [trace],
        height,
        legendPosition: 'none',
        xaxis: { title: { text: data.xaxis || '', standoff: 5 } },
        yaxis: { title: '' },
        margin: { l: 200, r: 80 },
        bargap: 0.3
    });

    const chartDiv = document.createElement('div');
    container.appendChild(chartDiv);
    await renderChart(chartDiv, [trace], layout);
};

// ============================================================================
// Main API
// =============================================================================
// Style Presets (e.g., style='ant' for Anthropic paper aesthetic)
// =============================================================================

const STYLE_PRESETS = {
    ant: {
        titleColor: '#c44e52',
        pointColor: '#5b8fbc',
        lineColor: '#444',
        labelColor: '#555',
        fontFamily: 'Georgia, serif',
        gridColor: '#eaeaea',
        axisColor: '#888',
        bgColor: '#fff',
        colorscaleDiverging: [[0, '#2166ac'], [0.5, '#f7f7f7'], [1, '#b2182b']],
        colorscaleSequential: [[0, '#440154'], [0.25, '#31688e'], [0.5, '#35b779'], [0.75, '#fde725'], [1, '#fde725']],
        categoricalColors: ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628', '#f781bf', '#999999', '#66c2a5', '#fc8d62'],
    },
    default: {
        titleColor: '#333',
        pointColor: 'var(--accent)',
        lineColor: '#666',
        labelColor: '#666',
        fontFamily: 'Inter, system-ui, sans-serif',
        gridColor: '#e8e8e8',
        axisColor: '#888',
        bgColor: '#fff',
        colorscaleDiverging: 'RdBu',
        colorscaleSequential: 'Viridis',
        categoricalColors: null,
    },
};

function getStyle(options) {
    return STYLE_PRESETS[options.style] || STYLE_PRESETS.default;
}

function _regressionLine(x, y, style) {
    const n = x.length;
    const xm = x.reduce((a, b) => a + b, 0) / n;
    const ym = y.reduce((a, b) => a + b, 0) / n;
    let num = 0, den = 0;
    for (let i = 0; i < n; i++) { num += (x[i] - xm) * (y[i] - ym); den += (x[i] - xm) ** 2; }
    const slope = num / den, intercept = ym - slope * xm;
    const xmin = Math.min(...x), xmax = Math.max(...x);
    return {
        x: [xmin, xmax], y: [intercept + slope * xmin, intercept + slope * xmax],
        mode: 'lines', type: 'scatter',
        line: { color: style.lineColor, width: 1.5, dash: 'dash' },
        showlegend: false, hoverinfo: 'skip',
    };
}

function _pearsonR(x, y) {
    const n = x.length;
    const xm = x.reduce((a, b) => a + b, 0) / n;
    const ym = y.reduce((a, b) => a + b, 0) / n;
    let num = 0, dx2 = 0, dy2 = 0;
    for (let i = 0; i < n; i++) {
        const dx = x[i] - xm, dy = y[i] - ym;
        num += dx * dy; dx2 += dx * dx; dy2 += dy * dy;
    }
    return num / (Math.sqrt(dx2) * Math.sqrt(dy2) + 1e-12);
}

function _applyStyle(layout, s, data) {
    layout.plot_bgcolor = s.bgColor;
    layout.paper_bgcolor = s.bgColor;
    layout.font = { family: s.fontFamily };
    if (data.title) {
        layout.title = { text: data.title, font: { color: s.titleColor, size: 15, family: s.fontFamily } };
    }
}

// =============================================================================
// Generic Chart Types: scatter, heatmap, bar, line
// =============================================================================

/**
 * Generic scatter plot.
 * JSON: { x, y, labels?, xaxis?, yaxis?, title?, highlight?, regression?, groups? }
 */
CHART_RENDERERS['scatter'] = async function(container, data, options = {}) {
    const s = getStyle(options);
    const labels = data.labels || [];
    const highlight = new Set(data.highlight || []);

    const traces = [];
    if (data.groups) {
        data.groups.forEach((g, i) => {
            const color = g.color || (s.categoricalColors ? s.categoricalColors[i % s.categoricalColors.length] : s.pointColor);
            traces.push({
                x: g.indices.map(j => data.x[j]), y: g.indices.map(j => data.y[j]),
                name: g.name, mode: 'markers', type: 'scatter',
                marker: { color, size: 8, opacity: 0.8 },
                hovertext: g.indices.map(j => labels[j] || ''), hoverinfo: 'text',
            });
        });
    } else {
        traces.push({
            x: data.x, y: data.y,
            mode: 'markers', type: 'scatter',
            marker: { color: s.pointColor, size: 8, opacity: 0.7 },
            hovertext: labels, hoverinfo: 'text', showlegend: false,
        });
    }

    // Arrow annotations for highlighted labels, hover-only for the rest
    const annotations = [];
    if (highlight.size > 0) {
        labels.forEach((label, i) => {
            if (!highlight.has(label)) return;
            annotations.push({
                x: data.x[i], y: data.y[i],
                text: label,
                showarrow: true,
                arrowhead: 0,
                arrowwidth: 1,
                arrowcolor: s.pointColor,
                ax: 0, ay: -25,
                font: { size: 9, color: s.labelColor, family: s.fontFamily },
            });
        });
    }

    if (data.regression !== false) {
        traces.push(_regressionLine(data.x, data.y, s));
        annotations.push({
            x: 0.02, y: 0.98, xref: 'paper', yref: 'paper',
            text: `r = ${_pearsonR(data.x, data.y).toFixed(2)}`, showarrow: false,
            font: { size: 13, color: '#333', family: s.fontFamily },
            xanchor: 'left', yanchor: 'top',
        });
    }

    const height = options.height || 350;
    const layout = buildChartLayout({
        traces, height, legendPosition: data.groups ? 'below' : 'none',
        xaxis: { title: { text: data.xaxis || '', font: { size: 12, color: '#555', family: s.fontFamily } },
                 gridcolor: s.gridColor, zerolinecolor: s.gridColor, tickfont: { size: 10, color: s.axisColor } },
        yaxis: { title: { text: data.yaxis || '', font: { size: 12, color: '#555', family: s.fontFamily } },
                 gridcolor: s.gridColor, zerolinecolor: s.gridColor, tickfont: { size: 10, color: s.axisColor } },
    });
    _applyStyle(layout, s, data);
    layout.annotations = (layout.annotations || []).concat(annotations);

    const chartDiv = document.createElement('div');
    chartDiv.style.height = `${height}px`;
    container.appendChild(chartDiv);
    await renderChart(chartDiv, traces, layout);
};

/**
 * Generic heatmap.
 * JSON: { z, x_labels?, y_labels?, xaxis?, yaxis?, title?, colorscale?, zmin?, zmax?, show_values? }
 */
CHART_RENDERERS['heatmap'] = async function(container, data, options = {}) {
    const s = getStyle(options);
    const cscale = data.colorscale === 'sequential' ? s.colorscaleSequential : s.colorscaleDiverging;

    const trace = {
        z: data.z, type: 'heatmap',
        x: data.x_labels || undefined, y: data.y_labels || undefined,
        colorscale: cscale, zmin: data.zmin, zmax: data.zmax, hoverinfo: 'x+y+z',
    };
    if (data.show_values) {
        trace.text = data.z.map(row => row.map(v => typeof v === 'number' ? v.toFixed(2) : ''));
        trace.texttemplate = '%{text}';
        trace.textfont = { size: 8 };
    }

    const nY = (data.y_labels || data.z).length;
    const height = options.height || Math.max(300, Math.min(800, nY * 4 + 100));
    const layout = buildChartLayout({
        preset: 'heatmap', traces: [trace], height,
        xaxis: { title: { text: data.xaxis || '' },
                 tickfont: { size: Math.min(10, Math.max(6, 800 / (data.x_labels?.length || 1))) } },
        yaxis: { title: { text: data.yaxis || '' },
                 tickfont: { size: Math.min(10, Math.max(6, 800 / nY)) }, autorange: 'reversed' },
    });
    _applyStyle(layout, s, data);

    const chartDiv = document.createElement('div');
    chartDiv.style.height = `${height}px`;
    container.appendChild(chartDiv);
    await renderChart(chartDiv, [trace], layout);
};

/**
 * Generic bar chart (horizontal or vertical).
 * JSON: { labels, values, xaxis?, yaxis?, title?, orientation?, color?, sort? }
 */
CHART_RENDERERS['bar'] = async function(container, data, options = {}) {
    const s = getStyle(options);
    const orientation = data.orientation || 'h';
    let labels = [...data.labels], values = [...data.values];

    if (data.sort === 'asc' || data.sort === 'desc') {
        const pairs = labels.map((l, i) => [l, values[i]]);
        pairs.sort((a, b) => data.sort === 'asc' ? a[1] - b[1] : b[1] - a[1]);
        labels = pairs.map(p => p[0]); values = pairs.map(p => p[1]);
    }

    const colors = data.color
        ? (Array.isArray(data.color) ? data.color : values.map(() => data.color))
        : values.map(v => v >= 0 ? s.pointColor : '#c44e52');

    const trace = { type: 'bar', orientation, marker: { color: colors }, hoverinfo: 'text' };
    if (orientation === 'h') { trace.x = values; trace.y = labels; trace.text = values.map(v => v.toFixed(3)); }
    else { trace.x = labels; trace.y = values; trace.text = values.map(v => v.toFixed(3)); }

    const height = options.height || Math.max(250, labels.length * 14 + 60);
    const layout = buildChartLayout({
        preset: 'barChart', traces: [trace], height, legendPosition: 'none',
        xaxis: { title: { text: (orientation === 'h' ? data.xaxis : '') || '' }, gridcolor: s.gridColor },
        yaxis: { automargin: true, tickfont: { size: Math.min(10, Math.max(6, 600 / labels.length)) } },
    });
    _applyStyle(layout, s, data);
    layout.bargap = 0.15;

    const chartDiv = document.createElement('div');
    chartDiv.style.height = `${height}px`;
    container.appendChild(chartDiv);
    await renderChart(chartDiv, [trace], layout);
};

/**
 * Generic line chart (multiple series).
 * JSON: { x, series: [{ name, y, color?, dash? }], xaxis?, yaxis?, title? }
 */
CHART_RENDERERS['line'] = async function(container, data, options = {}) {
    const s = getStyle(options);
    const traces = data.series.map((series, i) => ({
        x: data.x, y: series.y, name: series.name,
        mode: 'lines+markers', type: 'scatter',
        line: { color: series.color || (s.categoricalColors ? s.categoricalColors[i % s.categoricalColors.length] : s.pointColor),
                width: 2, dash: series.dash || 'solid' },
        marker: { size: 4 },
    }));

    const height = options.height || 350;
    const layout = buildChartLayout({
        traces, height, legendPosition: 'below',
        xaxis: { title: { text: data.xaxis || '', font: { size: 12, family: s.fontFamily } },
                 gridcolor: s.gridColor, tickfont: { size: 10, color: s.axisColor } },
        yaxis: { title: { text: data.yaxis || '', font: { size: 12, family: s.fontFamily } },
                 gridcolor: s.gridColor, tickfont: { size: 10, color: s.axisColor } },
    });
    _applyStyle(layout, s, data);

    const chartDiv = document.createElement('div');
    chartDiv.style.height = `${height}px`;
    container.appendChild(chartDiv);
    await renderChart(chartDiv, traces, layout);
};


// ============================================================================

/**
 * Render a chart of the specified type into a container
 * @param {string} type - Chart type (e.g., 'model-diff-effect', 'scatter', 'heatmap', 'bar', 'line')
 * @param {HTMLElement} container - Container element
 * @param {Object} data - Data from JSON file
 * @param {Object} options - { traits?, height?, style?: 'ant'|'default' }
 */
async function renderChartType(type, container, data, options = {}) {
    const renderer = CHART_RENDERERS[type];
    if (!renderer) {
        container.innerHTML = `<div class="chart-error">Unknown chart type: ${type}</div>`;
        return;
    }
    await renderer(container, data, options);
}

// ES module exports
export { renderChartType, CHART_RENDERERS };

// Keep window.* namespace for backward compat
window.chartTypes = {
    render: renderChartType,
    registry: CHART_RENDERERS
};
