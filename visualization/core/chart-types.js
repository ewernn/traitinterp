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
 *
 *   Cross-eval charts:
 *   - crosseval-comparison: Grouped bar comparing vectors across datasets
 *
 *   Prefill dynamics charts:
 *   - dynamics-effect: Smoothness effect by layer (+ optional projection stability)
 *   - dynamics-scatter: Smoothness vs perplexity correlation (requires perplexity=path)
 *   - dynamics-violin: Split violin distribution of smoothness
 *   - dynamics-position: Effect size by token position
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
            name: shortName,
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

    // Sort by effect size ascending (so highest appears at top in horizontal bar)
    const sorted = Object.entries(filteredTraits)
        .map(([path, d]) => ({ name: getTraitShortName(path), effect: d.peak_effect_size || 0 }))
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

    const layout = window.buildChartLayout({
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
    await window.renderChart(chartDiv, [trace], layout);
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

CHART_RENDERERS['annotation-stacked'] = async function(container, bars, options = {}) {
    const { height = 280 } = options;
    const colors = window.getChartColors?.() || ['#4a9eff', '#ff6b6b', '#51cf66', '#ffd43b', '#cc5de8', '#ff922b', '#20c997', '#868e96'];

    // Fetch all annotation files
    const barData = [];
    const allCategories = new Set();

    for (const bar of bars) {
        try {
            const response = await fetch(bar.path);
            if (!response.ok) throw new Error(`${response.status}`);
            const data = await response.json();
            const counts = countByCategory(data);

            for (const cat of Object.keys(counts)) {
                allCategories.add(cat);
            }

            barData.push({ label: bar.label, counts });
        } catch (e) {
            container.innerHTML = `<div class="chart-error">Failed to load ${bar.path}: ${e.message}</div>`;
            return;
        }
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

    const layout = window.buildChartLayout({
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
    await window.renderChart(chartDiv, traces, layout);
};

// ============================================================================
// Chart Type: comparison-bar (Horizontal bar chart for component/method comparison)
// ============================================================================

CHART_RENDERERS['comparison-bar'] = async function(container, data, options = {}) {
    const { height = 200 } = options;
    const colors = window.getChartColors?.() || ['#4a9eff', '#ff6b6b', '#51cf66', '#ffd43b', '#cc5de8', '#ff922b'];

    const results = data.results || [];
    if (results.length === 0) {
        container.innerHTML = '<div class="chart-error">No data to display</div>';
        return;
    }

    // Sort by delta ascending (highest at top for horizontal bars)
    const sorted = [...results].sort((a, b) => a.delta - b.delta);

    // Build labels with method/layer info
    const labels = sorted.map(d => {
        const methodShort = d.method === 'mean_diff' ? 'md' : d.method.slice(0, 2);
        return `${d.label} (${methodShort} L${d.layer})`;
    });

    const trace = {
        x: sorted.map(d => d.delta),
        y: labels,
        type: 'bar',
        orientation: 'h',
        marker: {
            color: sorted.map((_, i) => colors[(sorted.length - 1 - i) % colors.length])
        },
        text: sorted.map(d => `+${d.delta.toFixed(1)}`),
        textposition: 'outside',
        cliponaxis: false,  // Don't clip text labels at axis bounds
        hovertemplate: '%{y}<br>Delta: +%{x:.1f}<extra></extra>'
    };

    const layout = window.buildChartLayout({
        preset: 'barChart',
        traces: [trace],
        height,
        legendPosition: 'none',
        xaxis: { title: { text: 'Delta (trait score increase)', standoff: 5 } },
        yaxis: { title: '' },
        margin: { l: 180, r: 60 },
        bargap: 0.3
    });

    const chartDiv = document.createElement('div');
    container.appendChild(chartDiv);
    await window.renderChart(chartDiv, [trace], layout);
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
            try {
                const resp = await fetch(path);
                if (!resp.ok) continue;
                const projData = await resp.json();
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
            } catch (e) { /* skip failed fetches */ }
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

    const layout = window.buildChartLayout({
        preset: 'layerChart',
        traces,
        height,
        legendPosition: 'above',
        xaxis: { title: { text: 'Layer', standoff: 5 } },
        yaxis: { title: { text: "Cohen's d", standoff: 5 } }
    });

    const chartDiv = document.createElement('div');
    container.appendChild(chartDiv);
    await window.renderChart(chartDiv, traces, layout);
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
        try {
            const resp = await fetch(perplexityPath);
            if (resp.ok) pplData = await resp.json();
        } catch (e) { /* use null */ }
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

    const layout = window.buildChartLayout({
        preset: 'layerChart',
        traces,
        height,
        legendPosition: 'above',
        xaxis: { title: { text: 'Smoothness Diff (Prefilled - Model)', standoff: 5 } },
        yaxis: { title: { text: 'Cross-Entropy Diff', standoff: 5 } }
    });

    const chartDiv = document.createElement('div');
    container.appendChild(chartDiv);
    await window.renderChart(chartDiv, traces, layout);
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

    const layout = window.buildChartLayout({
        preset: 'barChart',
        traces,
        height,
        legendPosition: 'above',
        xaxis: { showticklabels: false, zeroline: false },
        yaxis: { title: { text: metric.charAt(0).toUpperCase() + metric.slice(1), standoff: 5 } }
    });

    const chartDiv = document.createElement('div');
    container.appendChild(chartDiv);
    await window.renderChart(chartDiv, traces, layout);
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

    const layout = window.buildChartLayout({
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
    await window.renderChart(chartDiv, [trace], layout);
};

// ============================================================================
// Chart Type: crosseval-comparison (Grouped bar: concealment vs lying AUROC)
// ============================================================================

CHART_RENDERERS['crosseval-comparison'] = async function(container, data, options = {}) {
    const { height = 280 } = options;

    if (!data?.datasets) {
        container.innerHTML = '<div class="chart-error">No cross-eval data</div>';
        return;
    }

    const datasets = Object.keys(data.datasets);
    const concAurocs = [];
    const lyingAurocs = [];

    for (const ds of datasets) {
        const dsData = data.datasets[ds];

        // Find best AUROC for concealment
        let concBest = 0;
        for (const [method, layers] of Object.entries(dsData.vectors?.concealment?.methods || {})) {
            for (const auroc of Object.values(layers)) {
                if (auroc > concBest) concBest = auroc;
            }
        }
        concAurocs.push(concBest);

        // Find best AUROC for lying
        let lyingBest = 0;
        for (const [method, layers] of Object.entries(dsData.vectors?.lying?.methods || {})) {
            for (const auroc of Object.values(layers)) {
                if (auroc > lyingBest) lyingBest = auroc;
            }
        }
        lyingAurocs.push(lyingBest);
    }

    const traces = [
        {
            x: datasets.map(d => d.toUpperCase()),
            y: concAurocs,
            type: 'bar',
            name: 'Concealment',
            marker: { color: '#51cf66' },
            text: concAurocs.map(v => v.toFixed(2)),
            textposition: 'outside',
            hovertemplate: '%{x}<br>Concealment: %{y:.3f}<extra></extra>'
        },
        {
            x: datasets.map(d => d.toUpperCase()),
            y: lyingAurocs,
            type: 'bar',
            name: 'Lying',
            marker: { color: '#ff6b6b' },
            text: lyingAurocs.map(v => v.toFixed(2)),
            textposition: 'outside',
            hovertemplate: '%{x}<br>Lying: %{y:.3f}<extra></extra>'
        }
    ];

    // Add random baseline reference
    traces.push({
        x: datasets.map(d => d.toUpperCase()),
        y: datasets.map(() => 0.5),
        type: 'scatter',
        mode: 'lines',
        line: { color: '#888', dash: 'dot', width: 1 },
        name: 'Random (0.5)',
        hoverinfo: 'skip'
    });

    const layout = window.buildChartLayout({
        preset: 'barChart',
        traces,
        height,
        legendPosition: 'above',
        xaxis: { title: { text: 'Dataset', standoff: 5 } },
        yaxis: { title: { text: 'AUROC', standoff: 5 }, range: [0, 1.1] },
        barmode: 'group',
        bargap: 0.2
    });

    const chartDiv = document.createElement('div');
    container.appendChild(chartDiv);
    await window.renderChart(chartDiv, traces, layout);
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
