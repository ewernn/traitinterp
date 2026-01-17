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

    // Guard: require experiment selection
    if (!window.state.currentExperiment) {
        contentArea.innerHTML = `
            <div class="tool-view">
                <div class="no-data">
                    <p>Please select an experiment from the sidebar</p>
                    <small>Analysis views require an experiment to be selected. Choose one from the "Experiment" section in the sidebar.</small>
                </div>
            </div>
        `;
        return;
    }

    const experiment = window.state.currentExperiment;

    // Render UI with both sections
    contentArea.innerHTML = `
        <div class="tool-view">
            <div class="page-intro">
                <div class="page-intro-text">Understanding model internals and comparing model variants.</div>
            </div>

            <!-- Section 1: Activation Diagnostics -->
            <section>
                ${ui.renderSubsection({
                    num: 1,
                    title: 'Activation Diagnostics',
                    infoId: 'info-activation-diagnostics',
                    infoText: 'Understanding model internals: activation magnitude growth and massive activation dimensions (Sun et al. 2024). Run <code>python analysis/massive_activations.py</code> to generate data.'
                })}

                <div class="projection-toggle" style="margin-top: 16px; margin-bottom: 12px;">
                    <span class="projection-toggle-label">Model Variant:</span>
                    <select id="activation-diagnostics-variant">
                        <!-- Populated dynamically -->
                    </select>
                </div>

                <h4 class="subsection-header" style="margin-top: 16px;">
                    <span class="subsection-title">Activation Magnitude by Layer</span>
                    <span class="subsection-info-toggle" data-target="info-act-magnitude">►</span>
                </h4>
                <div id="info-act-magnitude" class="info-text">
                    Shows ||h[l]|| (L2 norm of residual stream) averaged over all tokens and prompts. The residual stream accumulates information: h[l] = h[0] + Σ_{i&lt;l} (attn_out[i] + mlp_out[i]), where h[0] is the token embedding. Each layer's attention and MLP add to this running sum. Typical pattern: magnitude grows roughly linearly, with faster growth in early/late layers. Anomalies indicate unusual layer behavior. Data source: calibration set of neutral prompts (not trait-specific).
                </div>
                <div id="activation-magnitude-plot"></div>

                <h4 class="subsection-header" style="margin-top: 24px;">
                    <span class="subsection-title">Massive Activations</span>
                    <span class="subsection-info-toggle" data-target="info-massive-acts">►</span>
                </h4>
                <div id="info-massive-acts" class="info-text">
                    Massive activation dimensions (Sun et al. 2024) - specific dimensions with values 100-1000x larger than median. These act as fixed biases.
                </div>
                <div id="massive-activations-container"></div>

                <h4 class="subsection-header" style="margin-top: 24px;">
                    <span class="subsection-title">Massive Dims Across Layers</span>
                    <span class="subsection-info-toggle" data-target="info-massive-dims-layers">►</span>
                </h4>
                <div id="info-massive-dims-layers" class="info-text">
                    Tracks specific dimensions with anomalously large activations (Sun et al. 2024). For each layer l, identify top-k dimensions by |h[l][dim]| across the calibration set. Massive dims appear consistently across layers with values 100-1000× larger than median. Criteria dropdown filters which dims to plot: "Top 5, 3+ layers" = dims in top-5 at ≥3 layers (balanced, recommended). "Top 3, any layer" = dims in top-3 at any layer (conservative). "Top 5, any layer" = dims in top-5 at any layer (permissive). Y-axis: normalized magnitude = |h[l][dim]| / mean_d(|h[l][d]|). Values >>1 indicate massive dims. These dims act as constant biases and can be removed to improve trait projection signal-to-noise.
                </div>
                <div class="projection-toggle" style="margin-bottom: 12px;">
                    <span class="projection-toggle-label">Criteria:</span>
                    <select id="massive-dims-criteria">
                        <option value="top5-3layers">Top 5, 3+ layers</option>
                        <option value="top3-any">Top 3, any layer</option>
                        <option value="top5-any">Top 5, any layer</option>
                    </select>
                </div>
                <div id="massive-dims-layers-plot"></div>
            </section>

            <!-- Section 2: Variant Comparison -->
            <section>
                ${ui.renderSubsection({
                    num: 2,
                    title: 'Variant Comparison',
                    infoId: 'info-variant-comparison',
                    infoText: "Compares how two model variants (A vs B) project onto trait directions. Trait vectors are extracted from base model on contrastive pairs, then applied to both variants. Process for each prompt p: (1) Run inference with both model variants, capture residual stream at all layers. (2) For layer l: compute per-token projections proj[l,t] = h[l,t] · v[l] / ||v[l]||. (3) Average over response tokens: proj_mean[l,p] = mean_t(proj[l,t]). Aggregation across N prompts: μ_A[l] = (1/N) Σ_p proj_mean[l,p] for variant A, μ_B[l] similarly for B. Effect size: d[l] = (μ_B[l] − μ_A[l]) / σ_pooled. Cosine similarity chart: alignment between per-layer difference vector (mean_B − mean_A) and trait vector v[l]. Positive effect = variant B projects higher on trait direction. Run <code>python analysis/model_diff/compare_variants.py</code> to generate data."
                })}

                <div id="model-diff-container" style="margin-top: 16px;">
                    <div class="loading">Loading model diff data...</div>
                </div>
            </section>
        </div>
    `;

    // Setup info toggles
    window.setupSubsectionInfoToggles?.();

    // Populate model variant dropdown
    await populateVariantDropdown(experiment);

    // Render activation diagnostics (always)
    await renderActivationMagnitudePlot();
    await renderMassiveActivations();
    await renderMassiveDimsAcrossLayers();

    // Render model diff comparison
    await renderModelDiffComparison(experiment);
}

/**
 * Populate the model variant dropdown with available variants that have calibration data.
 */
async function populateVariantDropdown(experiment) {
    const dropdown = document.getElementById('activation-diagnostics-variant');
    if (!dropdown) return;

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

    // If no variants have data, show message
    if (availableVariants.length === 0) {
        dropdown.innerHTML = '<option value="">No calibration data</option>';
        dropdown.disabled = true;
        return;
    }

    // Populate dropdown
    dropdown.innerHTML = availableVariants.map(v =>
        `<option value="${v}" ${v === defaultVariant ? 'selected' : ''}>${v}</option>`
    ).join('');
    dropdown.disabled = false;

    // Add change handler (only once)
    if (!dropdown.dataset.bound) {
        dropdown.dataset.bound = 'true';
        dropdown.addEventListener('change', async () => {
            await renderActivationMagnitudePlot();
            await renderMassiveActivations();
            await renderMassiveDimsAcrossLayers();
        });
    }
}

/**
 * Get the currently selected model variant for activation diagnostics.
 */
function getSelectedVariant() {
    const dropdown = document.getElementById('activation-diagnostics-variant');
    return dropdown?.value || window.state.experimentData?.experimentConfig?.defaults?.application || 'instruct';
}

/**
 * Render model diff comparison using pre-computed results from compare_variants.py
 */
async function renderModelDiffComparison(experiment) {
    const container = document.getElementById('model-diff-container');
    if (!container) return;

    try {
        // Fetch available comparisons
        const response = await fetch(`/api/experiments/${experiment}/model-diff`);
        const data = await response.json();
        const comparisons = data.comparisons || [];

        if (comparisons.length === 0) {
            container.innerHTML = `
                <div class="info">
                    No model diff data available.
                    <br><br>
                    Run: <code>python analysis/model_diff/compare_variants.py --experiment ${experiment} --variant-a instruct --variant-b rm_lora --prompt-set {prompt_set}</code>
                </div>
            `;
            return;
        }

        // For now, use the first comparison (typically only one)
        const comparison = comparisons[0];
        const { variant_a, variant_b, prompt_sets } = comparison;

        // Load results for all prompt sets
        const allResults = {};
        for (const promptSet of prompt_sets) {
            const resultsPath = `experiments/${experiment}/model_diff/${comparison.variant_pair}/${promptSet}/results.json`;
            try {
                const res = await fetch('/' + resultsPath);
                if (res.ok) {
                    allResults[promptSet] = await res.json();
                }
            } catch (e) {
                console.warn(`Failed to load ${resultsPath}:`, e);
            }
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
                    row[setName] = {
                        peak_layer: traitData.peak_layer,
                        peak_effect: traitData.peak_effect_size
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
            </div>

            <table class="data-table" style="margin: 16px 0;">
                <thead>
                    <tr>
                        <th>Trait</th>
                        <th>Method</th>
                        ${promptSetNames.map(ps => `<th>${ps}</th>`).join('')}
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
                                    return `<td style="color: ${color};">${data.peak_effect.toFixed(2)}σ @ L${data.peak_layer}</td>`;
                                }
                                return '<td>—</td>';
                            }).join('')}
                        </tr>
                    `).join('')}
                </tbody>
            </table>

            <div id="model-diff-chart"></div>

            <h4 class="subsection-header" style="margin-top: 24px;">
                <span class="subsection-title">Cosine Similarity with Trait Direction</span>
            </h4>
            <div id="model-diff-cosine-chart"></div>
        `;

        // Plot all traits × prompt sets
        renderModelDiffChart(allResults, comparison);
        renderModelDiffCosineChart(allResults, comparison);

    } catch (error) {
        console.error('Model diff error:', error);
        container.innerHTML = `<div class="info">Error loading model diff data: ${error.message}</div>`;
    }
}

/**
 * Render model diff chart with all traits and prompt sets
 */
function renderModelDiffChart(allResults, comparison) {
    const chartDiv = document.getElementById('model-diff-chart');
    if (!chartDiv) return;

    const colors = window.getChartColors?.() || ['#4ecdc4', '#ff6b6b', '#ffe66d', '#95e1d3'];
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
            if (!traitData || !traitData.per_layer_effect_size) continue;

            const setName = promptSet.split('/').pop();
            const dash = dashIdx === 0 ? 'solid' : 'dash';

            traces.push({
                x: traitData.layers,
                y: traitData.per_layer_effect_size,
                type: 'scatter',
                mode: 'lines+markers',
                name: `${traitName} ${setName} (peak: ${traitData.peak_effect_size.toFixed(2)}σ @ L${traitData.peak_layer})`,
                line: { color, width: 2, dash },
                marker: { size: 3 },
                hovertemplate: `${traitName} ${setName}<br>L%{x}: %{y:.2f}σ<extra></extra>`
            });

            dashIdx++;
        }
        colorIdx++;
    }

    // Use shared chart utilities for consistent theming
    const layout = window.buildChartLayout({
        preset: 'layerChart',
        traces,
        height: 400,
        legendPosition: 'below',
        margin: { t: 40 },  // Extra top margin for title
        xaxis: {
            title: 'Layer',
            dtick: 10,
            showgrid: true
        },
        yaxis: {
            title: 'Effect Size (σ)',
            zeroline: true,
            zerolinewidth: 1,
            showgrid: true
        },
        hovermode: 'closest',
        title: `Trait Detection: ${comparison.variant_b} vs ${comparison.variant_a}`
    });

    window.renderChart(chartDiv, traces, layout, { displayModeBar: true });
}

/**
 * Render cosine similarity chart (diff vector alignment with trait direction)
 */
function renderModelDiffCosineChart(allResults, comparison) {
    const chartDiv = document.getElementById('model-diff-cosine-chart');
    if (!chartDiv) return;

    const colors = window.getChartColors?.() || ['#4ecdc4', '#ff6b6b', '#ffe66d', '#95e1d3'];
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
            if (!traitData || !traitData.per_layer_cosine_sim) continue;

            const setName = promptSet.split('/').pop();
            const dash = dashIdx === 0 ? 'solid' : 'dash';

            // Find peak cosine sim
            const peakIdx = traitData.per_layer_cosine_sim.reduce((maxIdx, val, idx, arr) =>
                val > arr[maxIdx] ? idx : maxIdx, 0);
            const peakCos = traitData.per_layer_cosine_sim[peakIdx];
            const peakLayer = traitData.layers[peakIdx];

            traces.push({
                x: traitData.layers,
                y: traitData.per_layer_cosine_sim,
                type: 'scatter',
                mode: 'lines+markers',
                name: `${traitName} ${setName} (peak: ${peakCos.toFixed(2)} @ L${peakLayer})`,
                line: { color, width: 2, dash },
                marker: { size: 3 },
                hovertemplate: `${traitName} ${setName}<br>L%{x}: %{y:.3f}<extra></extra>`
            });

            dashIdx++;
        }
        colorIdx++;
    }

    const layout = window.buildChartLayout({
        preset: 'layerChart',
        traces,
        height: 300,
        legendPosition: 'below',
        margin: { t: 10 },
        xaxis: {
            title: 'Layer',
            dtick: 10,
            showgrid: true
        },
        yaxis: {
            title: 'Cosine Similarity',
            range: [-0.15, 0.15],
            zeroline: true,
            zerolinewidth: 1,
            showgrid: true
        },
        hovermode: 'closest'
    });

    window.renderChart(chartDiv, traces, layout, { displayModeBar: true });
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
    const response = await fetch('/' + calibrationPath);
    if (!response.ok) return null;
    return response.json();
}


/**
 * Render Activation Magnitude plot showing ||h|| by layer.
 * Uses data from massive activations calibration file.
 */
async function renderActivationMagnitudePlot() {
    const plotDiv = document.getElementById('activation-magnitude-plot');
    if (!plotDiv) return;

    try {
        const data = await fetchMassiveActivationsData();
        if (!data || !data.aggregate?.layer_norms) {
            plotDiv.innerHTML = `
                <div class="info">
                    Activation magnitude data not available.
                    <br><br>
                    Run: <code>python analysis/massive_activations.py --experiment ${window.paths.getExperiment()}</code>
                </div>
            `;
            return;
        }

        const layerNorms = data.aggregate.layer_norms;
        const layers = Object.keys(layerNorms).map(Number).sort((a, b) => a - b);
        const norms = layers.map(l => layerNorms[l]);

        // Show model info if available
        const modelInfo = data.model ? `<div class="model-label">Model: <code>${data.model}</code></div>` : '';
        plotDiv.innerHTML = modelInfo;
        const chartDiv = document.createElement('div');
        plotDiv.appendChild(chartDiv);

        const traces = [{
            x: layers,
            y: norms,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Mean ||h||',
            line: { color: window.getChartColors()[0], width: 2 },
            marker: { size: 4 },
            hovertemplate: '<b>Layer %{x}</b><br>||h|| = %{y:.1f}<extra></extra>'
        }];

        const layout = window.buildChartLayout({
            preset: 'layerChart',
            traces,
            height: 250,
            legendPosition: 'none',
            xaxis: { title: 'Layer', tickmode: 'linear', tick0: 0, dtick: 5, showgrid: true },
            yaxis: { title: '||h|| (L2 norm)', showgrid: true }
        });
        window.renderChart(chartDiv, traces, layout);

    } catch (error) {
        plotDiv.innerHTML = `<div class="info">Error loading activation data: ${error.message}</div>`;
    }
}


/**
 * Render Massive Activations section.
 * Shows mean alignment plot - how much tokens point in a common direction.
 */
async function renderMassiveActivations() {
    const container = document.getElementById('massive-activations-container');
    if (!container) return;

    try {
        const data = await fetchMassiveActivationsData();
        if (!data) {
            container.innerHTML = `
                <div class="info">
                    No massive activation calibration data.
                    <br><br>
                    Run: <code>python analysis/massive_activations.py --experiment ${window.paths.getExperiment()}</code>
                </div>
            `;
            return;
        }

        const aggregate = data.aggregate || {};
        const meanAlignment = aggregate.mean_alignment_by_layer || {};

        if (Object.keys(meanAlignment).length === 0) {
            container.innerHTML = `<div class="info">No mean alignment data available.</div>`;
            return;
        }

        container.innerHTML = `<div id="mean-alignment-plot"></div>`;

        // Plot mean alignment by layer
        const layers = Object.keys(meanAlignment).map(Number).sort((a, b) => a - b);
        const alignments = layers.map(l => meanAlignment[l]);

        const alignTrace = {
            x: layers,
            y: alignments.map(v => v * 100),
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Mean Alignment',
            line: { color: window.getChartColors()[0], width: 2 },
            marker: { size: 4 },
            hovertemplate: 'L%{x}<br>Alignment: %{y:.1f}%<extra></extra>'
        };

        const alignLayout = window.buildChartLayout({
            preset: 'layerChart',
            traces: [alignTrace],
            height: 200,
            legendPosition: 'none',
            xaxis: { title: 'Layer', dtick: 5, showgrid: true },
            yaxis: { title: 'Mean Alignment (%)', range: [0, 100], showgrid: true }
        });
        window.renderChart('mean-alignment-plot', [alignTrace], alignLayout);

    } catch (error) {
        container.innerHTML = `<div class="info">Error loading massive activation data: ${error.message}</div>`;
    }
}


/**
 * Render Massive Dims Across Layers section.
 * Shows how each massive dim's normalized magnitude changes across layers.
 */
async function renderMassiveDimsAcrossLayers() {
    const container = document.getElementById('massive-dims-layers-plot');
    if (!container) return;

    try {
        const data = await fetchMassiveActivationsData();
        if (!data) {
            container.innerHTML = `<div class="info">No massive activation data. Run <code>python analysis/massive_activations.py --experiment ${window.paths.getExperiment()}</code></div>`;
            return;
        }
        const aggregate = data.aggregate || {};
        const topDimsByLayer = aggregate.top_dims_by_layer || {};
        const dimMagnitude = aggregate.dim_magnitude_by_layer || {};

        if (Object.keys(dimMagnitude).length === 0) {
            container.innerHTML = `<div class="info">No per-layer magnitude data. Re-run <code>python analysis/massive_activations.py</code> to generate.</div>`;
            return;
        }

        // Get criteria from dropdown
        const criteriaSelect = document.getElementById('massive-dims-criteria');
        const criteria = criteriaSelect?.value || 'top5-3layers';

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
        const colors = window.getChartColors();
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

        const layout = window.buildChartLayout({
            preset: 'layerChart',
            traces,
            height: 300,
            legendPosition: 'above',
            xaxis: { title: 'Layer', dtick: 5, showgrid: true },
            yaxis: { title: 'Normalized Magnitude', showgrid: true }
        });
        window.renderChart(chartDiv, traces, layout);

        // Setup dropdown change handler
        if (criteriaSelect && !criteriaSelect.dataset.bound) {
            criteriaSelect.dataset.bound = 'true';
            criteriaSelect.addEventListener('change', () => {
                renderMassiveDimsAcrossLayers();
            });
        }

    } catch (error) {
        container.innerHTML = `<div class="info">Error loading data: ${error.message}</div>`;
    }
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


// Export (both old and new names for compatibility)
window.renderModelAnalysis = renderModelAnalysis;
window.renderModelComparison = renderModelAnalysis;  // Backwards compatibility
