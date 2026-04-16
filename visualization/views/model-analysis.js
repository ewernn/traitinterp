import { fetchJSON, sortedNumericKeys } from '../core/utils.js';
import { getChartColors, displayLayer } from '../core/display.js';
import { buildChartLayout, renderChart } from '../core/charts.js';
import { requireExperiment, renderSubsection, renderRunHint, deferredLoading } from '../core/ui.js';
import { renderStyledSelect, wireStyledSelect } from '../components/styled-select.js';

// Module-local state for reading the current dropdown values (replaces reading .value from native selects).
let _maVariant = null;
let _maCriteria = 'top5-3layers';

/**
 * Model Analysis View - Understanding model internals and comparing variants
 *
 * Sections:
 * 1. Activation Diagnostics: Magnitude by layer, massive activations (Sun et al. 2024)
 * 2. Variant Comparison: Effect size (Cohen's d) from pre-computed model_diff results
 */

/**
 * Main render function
 */
async function renderModelAnalysis() {
    const contentArea = document.getElementById('content-area');

    if (requireExperiment(contentArea)) return;

    const { cancel } = deferredLoading(contentArea, 'Loading model analysis...');
    const experiment = window.state.currentExperiment;
    cancel();

    // Render UI with both sections
    contentArea.innerHTML = `
        <div class="tool-view">
            <div class="page-intro">
                <div class="page-intro-text">Understanding model internals and comparing model variants.</div>
            </div>

            <!-- Section 1: Activation Diagnostics -->
            <section>
                ${renderSubsection({
                    num: 1,
                    title: 'Activation Diagnostics',
                    infoId: 'info-activation-diagnostics',
                    infoText: 'Internals of one model variant from calibration prompts: norm growth, massive dimensions, token uniformity, and layer-to-layer change.'
                })}

                <div class="projection-toggle">
                    <span class="projection-toggle-label">Model Variant:</span>
                    <div id="activation-diagnostics-variant-container"><!-- Populated dynamically --></div>
                </div>

                ${renderSubsection({
                    title: 'Activation Magnitude by Layer',
                    infoId: 'info-act-magnitude',
                    infoText: 'L2 norm of the residual stream at each layer, plus per-layer attention and MLP contribution norms. Shows where the stream grows and why.',
                    level: 'h4'
                })}
                <div id="activation-magnitude-plot"></div>

                ${renderSubsection({
                    title: 'Activation Uniformity',
                    infoId: 'info-massive-acts',
                    infoText: 'Mean cosine similarity of each token to the layer&#39;s mean direction. High = all tokens point the same way, a sign of massive-dim dominance.',
                    level: 'h4'
                })}
                <div id="massive-activations-container"></div>

                ${renderSubsection({
                    title: 'Massive Dims Across Layers',
                    infoId: 'info-massive-dims-layers',
                    infoText: 'Per-dimension normalized magnitude <code>|h[l][d]| / mean|h|</code> across layers. Lines ≫ 1 are massive dims (Sun et al. 2024) — near-constant biases that hurt trait projection signal. Criteria dropdown controls how strict "massive" is.',
                    level: 'h4'
                })}
                <div class="projection-toggle">
                    <span class="projection-toggle-label">Criteria:</span>
                    ${renderStyledSelect({
                        id: 'massive-dims-criteria',
                        options: [
                            { value: 'top5-3layers', label: 'Top 5, 3+ layers' },
                            { value: 'top3-any', label: 'Top 3, any layer' },
                            { value: 'top5-any', label: 'Top 5, any layer' },
                        ],
                        selected: _maCriteria,
                        onChange: async (val) => {
                            _maCriteria = val;
                            const freshData = await fetchMassiveActivationsData();
                            renderMassiveDimsAcrossLayers(freshData);
                        },
                    })}
                </div>
                <div id="massive-dims-layers-plot"></div>

                ${renderSubsection({
                    title: 'Inter-Layer Similarity',
                    infoId: 'info-interlayer-sim',
                    infoText: 'Cosine between consecutive mean-layer vectors. Dips mark layers where the representation gets substantially rewritten.',
                    level: 'h4'
                })}
                <div id="interlayer-similarity-plot"></div>
            </section>

            <!-- Section 2: Variant Comparison -->
            <section>
                ${renderSubsection({
                    num: 2,
                    title: 'Variant Comparison',
                    infoId: 'info-variant-comparison',
                    infoText: 'Two variants process the same tokens; we project onto trait vectors layerwise and measure how far B drifts from A along each trait direction.'
                })}

                <div id="model-diff-container">
                    <div class="loading">Loading model diff data...</div>
                </div>
            </section>
        </div>
    `;

    // Setup info toggles
    window.setupSubsectionInfoToggles?.();

    // Wire inline styled selects (criteria); variant select is wired in populateVariantDropdown.
    wireStyledSelect(contentArea);

    // Populate model variant dropdown
    await populateVariantDropdown(experiment);

    // Fetch calibration data once, pass to all diagnostic renderers
    const calibrationData = await fetchMassiveActivationsData();
    renderActivationMagnitudePlot(calibrationData);
    renderMassiveActivations(calibrationData);
    renderMassiveDimsAcrossLayers(calibrationData);
    renderInterLayerSimilarity(calibrationData);

    // Render model diff comparison
    await renderModelDiffComparison(experiment);
}

/**
 * Populate the model variant dropdown with available variants that have calibration data.
 */
async function populateVariantDropdown(experiment) {
    const container = document.getElementById('activation-diagnostics-variant-container');
    if (!container) return;

    // Get all model variants from experiment config
    const variants = window.state.experimentData?.experimentConfig?.model_variants || {};
    const defaultVariant = window.state.experimentData?.experimentConfig?.defaults?.application || 'instruct';

    // Check which variants have calibration data
    const availableVariants = [];
    for (const variantName of Object.keys(variants)) {
        const calibrationPath = window.paths.get('inference.massive_activations', {
            prompt_set: 'calibration',
            model_variant: variantName
        });
        try {
            const response = await fetch('/' + calibrationPath, { method: 'HEAD' });
            if (response.ok) {
                availableVariants.push(variantName);
            }
        } catch (e) {
            // Variant doesn't have calibration data
        }
    }

    if (availableVariants.length === 0) {
        container.innerHTML = '<span class="cb-label" style="opacity:0.5;">No calibration data</span>';
        _maVariant = null;
        return;
    }

    _maVariant = availableVariants.includes(_maVariant) ? _maVariant
        : availableVariants.includes(defaultVariant) ? defaultVariant
        : availableVariants[0];

    container.innerHTML = renderStyledSelect({
        id: 'activation-diagnostics-variant',
        options: availableVariants.map(v => ({ value: v, label: v })),
        selected: _maVariant,
        onChange: async (val) => {
            _maVariant = val;
            const data = await fetchMassiveActivationsData();
            renderActivationMagnitudePlot(data);
            renderMassiveActivations(data);
            renderMassiveDimsAcrossLayers(data);
            renderInterLayerSimilarity(data);
        },
    });
    wireStyledSelect(container);
}

/**
 * Get the currently selected model variant for activation diagnostics.
 */
function getSelectedVariant() {
    return _maVariant || window.state.experimentData?.experimentConfig?.defaults?.application || 'instruct';
}

/**
 * Render model diff comparison using pre-computed results from compare_variants.py
 */
async function renderModelDiffComparison(experiment) {
    const container = document.getElementById('model-diff-container');
    if (!container) return;

    try {
        // Fetch available comparisons
        const data = await fetchJSON(`/api/experiments/${experiment}/model-diff`);
        const comparisons = data?.comparisons || [];

        if (comparisons.length === 0) {
            container.innerHTML = renderRunHint(
                'No model diff data available.',
                `python analysis/model_diff/compare_variants.py --experiment ${experiment} --variant-a <variant_a> --variant-b <variant_b> --prompt-set <prompt_set>`
            );
            return;
        }

        // For now, use the first comparison (typically only one)
        const comparison = comparisons[0];
        const { variant_a, variant_b, prompt_sets } = comparison;

        // Load results for all prompt sets
        const allResults = {};
        for (const promptSet of prompt_sets) {
            const resultsPath = window.paths.get('model_diff.results', {
                    variant_a: comparison.variant_a,
                    variant_b: comparison.variant_b,
                    prompt_set: promptSet
                });
            const result = await fetchJSON('/' + resultsPath);
            if (result) allResults[promptSet] = result;
        }

        if (Object.keys(allResults).length === 0) {
            container.innerHTML = `<div class="info">Failed to load model diff results.</div>`;
            return;
        }

        // Build summary table
        const summaryRows = [];
        const allTraits = new Set();
        for (const [promptSet, results] of Object.entries(allResults)) {
            for (const trait of Object.keys(results.traits || {})) {
                allTraits.add(trait);
            }
        }

        for (const trait of allTraits) {
            const row = { trait: trait.split('/').pop() };
            // Get method from first available result
            for (const results of Object.values(allResults)) {
                const traitData = results.traits?.[trait];
                if (traitData?.method) {
                    row.method = traitData.method;
                    break;
                }
            }
            for (const [promptSet, results] of Object.entries(allResults)) {
                const traitData = results.traits?.[trait];
                if (traitData) {
                    const setName = promptSet.split('/').pop();
                    // Get std at peak layer
                    const peakIdx = traitData.layers?.indexOf(traitData.peak_layer);
                    const stdA = peakIdx >= 0 ? traitData.per_layer_std_a?.[peakIdx] : null;
                    const stdB = peakIdx >= 0 ? traitData.per_layer_std_b?.[peakIdx] : null;
                    row[setName] = {
                        peak_layer: traitData.peak_layer,
                        peak_effect: traitData.peak_effect_size,
                        std_a: stdA,
                        std_b: stdB
                    };
                }
            }
            summaryRows.push(row);
        }

        // Render summary
        const promptSetNames = prompt_sets.map(ps => ps.split('/').pop());
        container.innerHTML = `
            <div class="model-diff-header">
                <strong>${variant_b}</strong> vs <strong>${variant_a}</strong>
                <span style="color: var(--text-tertiary); margin-left: 8px;">(${Object.values(allResults)[0]?.n_prompts || '?'} prompts)</span>
                <div class="model-diff-legend">positive = ${variant_b} higher than ${variant_a}</div>
            </div>

            <table class="data-table">
                <thead>
                    <tr>
                        <th>Trait</th>
                        <th>Method</th>
                        ${promptSetNames.map(ps => `<th>${ps}</th><th>Spread (A / B)</th>`).join('')}
                    </tr>
                </thead>
                <tbody>
                    ${summaryRows.map(row => `
                        <tr>
                            <td>${row.trait}</td>
                            <td style="color: var(--text-secondary);">${row.method || '?'}</td>
                            ${promptSetNames.map(ps => {
                                const data = row[ps];
                                if (data) {
                                    const color = data.peak_effect > 1.5 ? 'var(--success-color)' :
                                                  data.peak_effect > 0.5 ? 'var(--warning-color)' :
                                                  'var(--text-secondary)';
                                    const effectCell = `<td style="color: ${color};">${data.peak_effect.toFixed(2)}σ @ L${displayLayer(data.peak_layer)}</td>`;
                                    const spreadCell = data.std_a != null && data.std_b != null
                                        ? `<td style="color: var(--text-secondary);">${data.std_a.toFixed(2)} / ${data.std_b.toFixed(2)}</td>`
                                        : '<td style="color: var(--text-tertiary);">—</td>';
                                    return effectCell + spreadCell;
                                }
                                return '<td>—</td><td>—</td>';
                            }).join('')}
                        </tr>
                    `).join('')}
                </tbody>
            </table>

            ${renderSubsection({
                title: 'Effect Size by Layer',
                infoId: 'info-effect-size',
                infoText: 'Cohen&#39;s d between variants per layer, one line per trait × prompt set. Peaks mark the layer that best separates A from B on that trait.',
                level: 'h4'
            })}
            <div id="model-diff-chart"></div>

            ${renderSubsection({
                title: 'Cosine Similarity with Trait Direction',
                infoId: 'info-cosine-sim',
                infoText: 'Cosine between the mean (B − A) diff vector and the trait vector per layer. Tells you whether the variant shift points along the trait, not just how big it is.',
                level: 'h4'
            })}
            <div id="model-diff-cosine-chart"></div>
        `;

        // Setup info toggles for this section
        window.setupSubsectionInfoToggles?.();

        // Plot all traits × prompt sets
        renderModelDiffLayerChart(allResults, comparison, {
            field: 'per_layer_effect_size',
            divId: 'model-diff-chart',
            yaxisTitle: 'Effect Size (σ)',
            hoverFormat: '%{y:.2f}σ',
            height: 400,
            margin: { t: 40 },
            title: `Trait Detection: ${comparison.variant_b} vs ${comparison.variant_a}`
        });
        renderModelDiffLayerChart(allResults, comparison, {
            field: 'per_layer_cosine_sim',
            divId: 'model-diff-cosine-chart',
            yaxisTitle: 'Cosine Similarity',
            hoverFormat: '%{y:.3f}',
            height: 300,
            margin: { t: 10 },
            yaxisRange: [-0.15, 0.15],
            peakByAbsValue: true
        });

    } catch (error) {
        container.innerHTML = `<div class="info">Error loading model diff data: ${error.message}</div>`;
    }
}

/**
 * Render a model diff layer chart (effect size or cosine similarity).
 * @param {Object} allResults - {promptSet: results} map
 * @param {Object} comparison - {variant_a, variant_b} info
 * @param {Object} opts - Chart configuration
 * @param {string} opts.field - Data field name (e.g. 'per_layer_effect_size', 'per_layer_cosine_sim')
 * @param {string} opts.divId - Target div ID
 * @param {string} opts.yaxisTitle - Y-axis label
 * @param {string} opts.hoverFormat - Plotly hover format (e.g. '%{y:.2f}σ')
 * @param {number} [opts.height=300] - Chart height
 * @param {Object} [opts.margin] - Margin overrides
 * @param {string} [opts.title] - Chart title
 * @param {Array} [opts.yaxisRange] - Y-axis range
 * @param {boolean} [opts.peakByAbsValue=false] - Find peak by absolute value (for cosine sim)
 */
function renderModelDiffLayerChart(allResults, comparison, {
    field, divId, yaxisTitle, hoverFormat,
    height = 300, margin = {}, title, yaxisRange, peakByAbsValue = false
}) {
    const chartDiv = document.getElementById(divId);
    if (!chartDiv) return;

    const colors = getChartColors();
    const traces = [];
    let colorIdx = 0;

    // Collect all traits
    const allTraits = new Set();
    for (const results of Object.values(allResults)) {
        for (const trait of Object.keys(results.traits || {})) {
            allTraits.add(trait);
        }
    }

    // Create traces for each trait × prompt set combination
    for (const trait of allTraits) {
        const traitName = trait.split('/').pop();
        const color = colors[colorIdx % colors.length];
        let dashIdx = 0;

        for (const [promptSet, results] of Object.entries(allResults)) {
            const traitData = results.traits?.[trait];
            if (!traitData || !traitData[field]) continue;

            const values = traitData[field];
            const setName = promptSet.split('/').pop();
            const dash = dashIdx === 0 ? 'solid' : 'dash';

            // Compute peak label
            let peakVal, peakLayer;
            if (peakByAbsValue) {
                const peakIdx = values.reduce((maxIdx, val, idx, arr) =>
                    Math.abs(val) > Math.abs(arr[maxIdx]) ? idx : maxIdx, 0);
                peakVal = values[peakIdx];
                peakLayer = traitData.layers[peakIdx];
            } else {
                peakVal = traitData.peak_effect_size;
                peakLayer = traitData.peak_layer;
            }
            const peakLabel = peakByAbsValue ? peakVal.toFixed(2) : `${peakVal.toFixed(2)}σ`;

            traces.push({
                x: traitData.layers,
                y: values,
                type: 'scatter',
                mode: 'lines+markers',
                name: `${traitName} ${setName} (peak: ${peakLabel} @ L${peakLayer})`,
                line: { color, width: 2, dash },
                marker: { size: 3 },
                hovertemplate: `${traitName} ${setName}<br>L%{x}: ${hoverFormat}<extra></extra>`
            });

            dashIdx++;
        }
        colorIdx++;
    }

    const yaxis = {
        title: yaxisTitle,
        zeroline: true,
        zerolinewidth: 1,
        showgrid: true,
        ...(yaxisRange ? { range: yaxisRange } : {})
    };

    const layout = buildChartLayout({
        preset: 'layerChart',
        traces,
        height,
        legendPosition: 'below',
        margin,
        xaxis: { title: 'Layer', dtick: 10, showgrid: true },
        yaxis,
        hovermode: 'closest',
        ...(title ? { title } : {})
    });

    renderChart(chartDiv, traces, layout, { displayModeBar: true });
}


// ============================================================================
// Activation Diagnostics Functions
// ============================================================================

/**
 * Fetch massive activations data, using calibration.json as canonical source.
 * Calibration contains model-wide massive dims computed from neutral prompts.
 */
async function fetchMassiveActivationsData() {
    const modelVariant = getSelectedVariant();
    const calibrationPath = window.paths.get('inference.massive_activations', { prompt_set: 'calibration', model_variant: modelVariant });
    return fetchJSON('/' + calibrationPath);
}

/**
 * Shared wrapper for diagnostic render functions.
 * Handles container lookup, null-data guard, and error catch.
 * @param {string} containerId - DOM element ID
 * @param {any} data - Pre-fetched calibration data (from fetchMassiveActivationsData)
 * @param {Function} renderFn - (container, data) => void
 */
function withMassiveActivationsData(containerId, data, renderFn) {
    const container = document.getElementById(containerId);
    if (!container) return;

    if (!data) {
        container.innerHTML = renderRunHint(
            'No massive activation calibration data.',
            `python inference/run_inference_pipeline.py --experiment ${window.paths.getExperiment()} --prompt-set starter_prompts/general   # captures automatically`
        );
        return;
    }

    try {
        renderFn(container, data);
    } catch (error) {
        container.innerHTML = `<div class="info">Error loading data: ${error.message}</div>`;
    }
}


/**
 * Render Activation Magnitude plot showing ||h|| by layer.
 */
function renderActivationMagnitudePlot(data) {
    withMassiveActivationsData('activation-magnitude-plot', data, (plotDiv, data) => {
        if (!data.aggregate?.layer_norms) {
            plotDiv.innerHTML = renderRunHint(
                'Activation magnitude data not available.',
                `python inference/run_inference_pipeline.py --experiment ${window.paths.getExperiment()} --prompt-set starter_prompts/general   # captures automatically`
            );
            return;
        }

        const layerNorms = data.aggregate.layer_norms;
        const attnNorms = data.aggregate.attn_norms || {};
        const mlpNorms = data.aggregate.mlp_norms || {};
        const layers = sortedNumericKeys(layerNorms);
        const norms = layers.map(l => layerNorms[l]);
        const attn = layers.map(l => attnNorms[l] || null);
        const mlp = layers.map(l => mlpNorms[l] || null);

        // Show model info if available
        const modelInfo = data.model ? `<div class="model-label">Model: <code>${data.model}</code></div>` : '';
        plotDiv.innerHTML = modelInfo;
        const chartDiv = document.createElement('div');
        plotDiv.appendChild(chartDiv);

        const colors = getChartColors();
        const traces = [{
            x: layers,
            y: norms,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Residual ||h||',
            line: { color: colors[0], width: 2 },
            marker: { size: 4 },
            hovertemplate: '<b>Layer %{x}</b><br>||h|| = %{y:.1f}<extra></extra>'
        }];

        // Add attn trace if data exists
        if (attn.some(v => v !== null)) {
            traces.push({
                x: layers,
                y: attn,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Attn contrib',
                line: { color: colors[1], width: 2 },
                marker: { size: 4 },
                hovertemplate: '<b>Layer %{x}</b><br>||attn|| = %{y:.1f}<extra></extra>'
            });
        }

        // Add mlp trace if data exists
        if (mlp.some(v => v !== null)) {
            traces.push({
                x: layers,
                y: mlp,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'MLP contrib',
                line: { color: colors[2], width: 2 },
                marker: { size: 4 },
                hovertemplate: '<b>Layer %{x}</b><br>||mlp|| = %{y:.1f}<extra></extra>'
            });
        }

        const hasContribData = traces.length > 1;
        const layout = buildChartLayout({
            preset: 'layerChart',
            traces,
            height: 250,
            legendPosition: hasContribData ? 'right' : 'none',
            xaxis: { title: 'Layer', tickmode: 'linear', tick0: 0, dtick: 5, showgrid: true },
            yaxis: { title: 'L2 norm', showgrid: true }
        });
        renderChart(chartDiv, traces, layout);
    });
}


/**
 * Render Massive Activations section.
 * Shows mean alignment plot - how much tokens point in a common direction.
 */
function renderMassiveActivations(data) {
    withMassiveActivationsData('massive-activations-container', data, (container, data) => {
        const aggregate = data.aggregate || {};
        const meanAlignment = aggregate.mean_alignment_by_layer || {};

        if (Object.keys(meanAlignment).length === 0) {
            container.innerHTML = `<div class="info">No mean alignment data available.</div>`;
            return;
        }

        container.innerHTML = `<div id="mean-alignment-plot"></div>`;

        // Plot mean alignment by layer
        const layers = sortedNumericKeys(meanAlignment);
        const alignments = layers.map(l => meanAlignment[l]);

        const alignTrace = {
            x: layers,
            y: alignments.map(v => v * 100),
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Mean Alignment',
            line: { color: getChartColors()[0], width: 2 },
            marker: { size: 4 },
            hovertemplate: 'L%{x}<br>Alignment: %{y:.1f}%<extra></extra>'
        };

        const alignLayout = buildChartLayout({
            preset: 'layerChart',
            traces: [alignTrace],
            height: 200,
            legendPosition: 'none',
            xaxis: { title: 'Layer', dtick: 5, showgrid: true },
            yaxis: { title: 'Mean Alignment (%)', range: [0, 100], showgrid: true }
        });
        renderChart('mean-alignment-plot', [alignTrace], alignLayout);
    });
}


/**
 * Render Massive Dims Across Layers section.
 * Shows how each massive dim's normalized magnitude changes across layers.
 */
function renderMassiveDimsAcrossLayers(data) {
    withMassiveActivationsData('massive-dims-layers-plot', data, (container, data) => {
        const aggregate = data.aggregate || {};
        const topDimsByLayer = aggregate.top_dims_by_layer || {};
        const dimMagnitude = aggregate.dim_magnitude_by_layer || {};

        if (Object.keys(dimMagnitude).length === 0) {
            container.innerHTML = `<div class="info">No per-layer magnitude data. Run inference — it captures automatically: <code>python inference/run_inference_pipeline.py --experiment ${window.paths.getExperiment()} --prompt-set starter_prompts/general</code></div>`;
            return;
        }

        const criteria = _maCriteria;

        // Filter dims based on criteria
        const filteredDims = filterDimsByCriteria(topDimsByLayer, criteria);

        if (filteredDims.length === 0) {
            container.innerHTML = `<div class="info">No dims match criteria "${criteria}".</div>`;
            return;
        }

        // Show model info if available
        const modelInfo = data.model ? `<div class="model-label">Model: <code>${data.model}</code></div>` : '';
        container.innerHTML = modelInfo;
        const chartDiv = document.createElement('div');
        container.appendChild(chartDiv);

        // Build traces
        const colors = getChartColors();
        const nLayers = Object.keys(topDimsByLayer).length;
        const layers = Array.from({ length: nLayers }, (_, i) => i);

        const traces = filteredDims.map((dim, idx) => {
            const magnitudes = dimMagnitude[dim] || [];
            return {
                x: layers,
                y: magnitudes,
                type: 'scatter',
                mode: 'lines+markers',
                name: `dim ${dim}`,
                line: { color: colors[idx % colors.length], width: 2 },
                marker: { size: 4 },
                hovertemplate: `dim ${dim}<br>L%{x}<br>Normalized: %{y:.2f}x<extra></extra>`
            };
        });

        const layout = buildChartLayout({
            preset: 'layerChart',
            traces,
            height: 300,
            legendPosition: 'above',
            xaxis: { title: 'Layer', dtick: 5, showgrid: true },
            yaxis: { title: 'Normalized Magnitude', showgrid: true }
        });
        renderChart(chartDiv, traces, layout);

    });
}


/**
 * Filter dims based on selected criteria.
 */
function filterDimsByCriteria(topDimsByLayer, criteria) {
    const dimAppearances = {};  // {dim: count of layers it appears in}

    // Count appearances based on criteria
    for (const [layer, dims] of Object.entries(topDimsByLayer)) {
        const topK = criteria === 'top3-any' ? 3 : 5;
        const dimsToCount = dims.slice(0, topK);
        for (const dim of dimsToCount) {
            dimAppearances[dim] = (dimAppearances[dim] || 0) + 1;
        }
    }

    // Filter based on min layers
    const minLayers = criteria === 'top5-3layers' ? 3 : 1;
    const filtered = Object.entries(dimAppearances)
        .filter(([dim, count]) => count >= minLayers)
        .map(([dim]) => parseInt(dim))
        .sort((a, b) => a - b);

    return filtered;
}


/**
 * Render inter-layer similarity plot: cos(mean[l], mean[l+1]).
 */
function renderInterLayerSimilarity(data) {
    withMassiveActivationsData('interlayer-similarity-plot', data, (plotDiv, data) => {
        if (!data.aggregate?.consecutive_cosine) {
            plotDiv.innerHTML = renderRunHint(
                'Inter-layer similarity data not available.',
                `python inference/run_inference_pipeline.py --experiment ${window.paths.getExperiment()} --prompt-set starter_prompts/general   # captures automatically`
            );
            return;
        }

        const consecutiveCosine = data.aggregate.consecutive_cosine;
        const layers = sortedNumericKeys(consecutiveCosine);
        const similarities = layers.map(l => consecutiveCosine[l]);

        const colors = getChartColors();
        const traces = [{
            x: layers,
            y: similarities,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'cos(L, L+1)',
            line: { color: colors[0], width: 2 },
            marker: { size: 4 },
            hovertemplate: 'L%{x} → L%{customdata}<br>cos = %{y:.4f}<extra></extra>',
            customdata: layers.map(l => l + 1)
        }];

        const layout = buildChartLayout({
            preset: 'layerChart',
            traces,
            height: 200,
            legendPosition: 'none',
            xaxis: { title: 'Layer', dtick: 5, showgrid: true },
            yaxis: { title: 'Cosine Similarity', showgrid: true }
        });
        renderChart(plotDiv, traces, layout);
    });
}


/** Reset model-analysis local state (called on experiment change). */
function resetModelAnalysisState() {
    // No module-local caches currently — stub is here for symmetry
    // with the other views, so state.js can call it unconditionally.
}

// ES module exports
export { renderModelAnalysis, resetModelAnalysisState };

// Keep window.* for router + state.js reference
window.renderModelAnalysis = renderModelAnalysis;
window.resetModelAnalysisState = resetModelAnalysisState;
