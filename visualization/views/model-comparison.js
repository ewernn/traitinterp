/**
 * Model Analysis View - Understanding model internals and comparing variants
 *
 * Sections:
 * 1. Activation Diagnostics: Magnitude by layer, massive activations (Sun et al. 2024)
 * 2. Variant Comparison: Effect size (Cohen's d) for same prompts across model variants
 */

// Cache for loaded projections to avoid re-fetching
const projectionCache = {};

/**
 * Load all projection files for a variant/trait/prompt-set combination
 */
async function loadProjections(experiment, modelVariant, trait, promptSet) {
    const cacheKey = `${experiment}/${modelVariant}/${trait}/${promptSet}`;
    if (projectionCache[cacheKey]) {
        return projectionCache[cacheKey];
    }

    const pb = window.PathBuilder;
    const projectionsDir = pb.get('inference.projections', {
        experiment,
        model_variant: modelVariant,
        trait,
        prompt_set: promptSet
    });

    // Get list of prompt IDs from responses directory
    const responsesDir = pb.get('inference.responses', {
        experiment,
        model_variant: modelVariant,
        prompt_set: promptSet
    });

    try {
        // Fetch a sample response to get metadata
        const metadataPath = `${responsesDir}/metadata.json`;
        const metadata = await fetch(metadataPath).then(r => r.json()).catch(() => null);

        // Try to list responses
        const responsesPath = `${responsesDir}/`;
        // Since we can't list directory, we'll try loading sequentially until we fail
        const projections = {};
        let promptId = 1;
        let consecutiveFailures = 0;

        while (consecutiveFailures < 5) {
            const projPath = `${projectionsDir}/${promptId}.json`;
            try {
                const data = await fetch(projPath).then(r => r.json());
                projections[promptId] = data;
                promptId++;
                consecutiveFailures = 0;
            } catch (e) {
                consecutiveFailures++;
                promptId++;
            }
        }

        const result = { projections, metadata };
        projectionCache[cacheKey] = result;
        return result;
    } catch (error) {
        console.error(`Failed to load projections for ${cacheKey}:`, error);
        return { projections: {}, metadata: null };
    }
}

/**
 * Aggregate projection scores by layer (mean across tokens per response)
 */
function aggregateByLayer(projections, numLayers) {
    const byLayer = {};

    for (let layer = 0; layer < numLayers; layer++) {
        byLayer[layer] = [];
    }

    // For each response
    for (const [promptId, data] of Object.entries(projections)) {
        // For each layer
        for (let layer = 0; layer < numLayers; layer++) {
            const layerKey = `layer_${layer}`;
            if (data[layerKey] && Array.isArray(data[layerKey])) {
                // Mean across tokens for this response
                const scores = data[layerKey];
                const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
                byLayer[layer].push(mean);
            }
        }
    }

    return byLayer;
}

/**
 * Compute Cohen's d effect size between two distributions
 */
function computeCohenD(baseline, compare) {
    if (baseline.length === 0 || compare.length === 0) {
        return { effectSize: 0, meanDiff: 0, pValue: 1, n: 0 };
    }

    const mean1 = baseline.reduce((a, b) => a + b, 0) / baseline.length;
    const mean2 = compare.reduce((a, b) => a + b, 0) / compare.length;
    const meanDiff = mean2 - mean1;

    const variance1 = baseline.reduce((sum, x) => sum + Math.pow(x - mean1, 2), 0) / baseline.length;
    const variance2 = compare.reduce((sum, x) => sum + Math.pow(x - mean2, 2), 0) / compare.length;
    const pooledStd = Math.sqrt((variance1 + variance2) / 2);

    const effectSize = pooledStd > 0 ? meanDiff / pooledStd : 0;

    // Simple t-test (assumes equal variances)
    const n1 = baseline.length;
    const n2 = compare.length;
    const pooledVar = ((n1 - 1) * variance1 + (n2 - 1) * variance2) / (n1 + n2 - 2);
    const tStat = meanDiff / Math.sqrt(pooledVar * (1/n1 + 1/n2));

    // Approximate p-value (two-tailed)
    // For simplicity, we'll mark as significant if |t| > 2 (roughly p < 0.05 for large n)
    const pValue = Math.abs(tStat) > 2 ? 0.01 : 0.1; // Rough approximation

    return {
        effectSize,
        meanDiff,
        pValue,
        n: Math.min(n1, n2),
        baseline: { mean: mean1, std: Math.sqrt(variance1), n: n1 },
        compare: { mean: mean2, std: Math.sqrt(variance2), n: n2 }
    };
}

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
    const config = window.state.experimentData?.experimentConfig;

    // Check if variant comparison is available (need 2+ variants)
    const hasVariants = config?.model_variants && Object.keys(config.model_variants).length >= 2;
    const variants = hasVariants ? Object.keys(config.model_variants) : [];
    const defaultBaseline = hasVariants ? (config.defaults?.application || variants[0]) : '';
    const defaultCompare = hasVariants ? (variants.find(v => v !== defaultBaseline) || variants[1] || variants[0]) : '';

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

                <h4 class="subsection-header" style="margin-top: 16px;">
                    <span class="subsection-title">Activation Magnitude by Layer</span>
                    <span class="subsection-info-toggle" data-target="info-act-magnitude">►</span>
                </h4>
                <div id="info-act-magnitude" class="info-text">
                    How the residual stream grows in magnitude as each layer adds information to the hidden state.
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
                    Shows how each massive dimension's magnitude changes across layers (normalized by layer average).
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
                    infoText: 'Compare how different model variants (base, instruct, LoRA) activate on the same trait when processing identical text. Uses Cohen\'s d effect size.'
                })}

                ${hasVariants ? `
                    <div class="tool-controls" style="margin-top: 16px;">
                        <div class="control-row">
                            ${ui.renderSelect({ id: 'baseline-variant', label: 'Baseline Variant', options: variants, selected: defaultBaseline, className: 'variant-select' })}
                            ${ui.renderSelect({ id: 'compare-variant', label: 'Compare Variant', options: variants, selected: defaultCompare, className: 'variant-select' })}
                        </div>

                        <div class="control-row">
                            ${ui.renderSelect({ id: 'trait-select', label: 'Trait', options: [], placeholder: 'Loading traits...', className: 'trait-select' })}
                            ${ui.renderSelect({ id: 'prompt-set-select', label: 'Prompt Set', options: [], placeholder: 'Loading prompt sets...', className: 'prompt-set-select' })}
                        </div>

                        <button id="compute-btn" class="primary-btn">Compute Effect Size</button>
                    </div>

                    <div id="comparison-results" class="comparison-results" style="display: none;">
                        <div class="results-summary" id="results-summary"></div>
                        <div id="effect-size-chart"></div>
                    </div>

                    <div id="comparison-loading" class="loading" style="display: none;">
                        Computing effect sizes across layers...
                    </div>

                    <div id="comparison-error" class="error" style="display: none;"></div>
                ` : `
                    <div class="info" style="margin-top: 16px;">
                        Variant comparison requires 2+ model variants in <code>config.json</code>.
                        <br><br>
                        Current config has ${variants.length} variant(s). Add more variants to enable comparison:
                        <pre>{ "model_variants": { "base": {...}, "instruct": {...} } }</pre>
                    </div>
                `}
            </section>
        </div>
    `;

    // Setup info toggles
    window.setupSubsectionInfoToggles?.();

    // Render activation diagnostics (always)
    await renderActivationMagnitudePlot();
    await renderMassiveActivations();
    await renderMassiveDimsAcrossLayers();

    // Setup variant comparison if available
    if (hasVariants) {
        await populateTraitSelector(experiment);
        await populatePromptSetSelector(experiment, defaultBaseline);

        document.getElementById('compute-btn').addEventListener('click', () => {
            runComparison(experiment);
        });

        // Auto-run if we have stored selections
        const storedTrait = sessionStorage.getItem('modelComparison.trait');
        const storedPromptSet = sessionStorage.getItem('modelComparison.promptSet');
        if (storedTrait && storedPromptSet) {
            document.getElementById('trait-select').value = storedTrait;
            document.getElementById('prompt-set-select').value = storedPromptSet;
        }
    }
}

/**
 * Populate trait selector with available traits from experiment
 */
async function populateTraitSelector(experiment) {
    const select = document.getElementById('trait-select');

    try {
        const response = await fetch(`/api/experiments/${experiment}/traits`);
        const data = await response.json();
        const traits = data.traits || [];

        if (traits.length === 0) {
            select.innerHTML = '<option value="">No traits available</option>';
            return;
        }

        select.innerHTML = traits.map(trait =>
            `<option value="${trait}">${trait}</option>`
        ).join('');

    } catch (error) {
        console.error('Failed to load traits:', error);
        select.innerHTML = '<option value="">Failed to load traits</option>';
    }
}

/**
 * Populate prompt set selector - for now just hardcode common ones
 * TODO: Auto-discover from inference/responses directory
 */
async function populatePromptSetSelector(experiment, variant) {
    const select = document.getElementById('prompt-set-select');

    // Hardcoded common prompt sets - ideally we'd auto-discover these
    const commonSets = [
        'train_100',
        'test_150',
        'benign',
        'harmful',
        'single_trait',
        'multi_trait',
        'dynamic',
        'adversarial'
    ];

    select.innerHTML = commonSets.map(set =>
        `<option value="${set}">${set}</option>`
    ).join('');
}

/**
 * Run the comparison analysis
 */
async function runComparison(experiment) {
    const baselineVariant = document.getElementById('baseline-variant').value;
    const compareVariant = document.getElementById('compare-variant').value;
    const trait = document.getElementById('trait-select').value;
    const promptSet = document.getElementById('prompt-set-select').value;

    if (!trait || !promptSet) {
        showError('Please select a trait and prompt set');
        return;
    }

    if (baselineVariant === compareVariant) {
        showError('Baseline and compare variants must be different');
        return;
    }

    // Store selections
    sessionStorage.setItem('modelComparison.trait', trait);
    sessionStorage.setItem('modelComparison.promptSet', promptSet);

    // Show loading
    document.getElementById('comparison-loading').style.display = 'block';
    document.getElementById('comparison-results').style.display = 'none';
    document.getElementById('comparison-error').style.display = 'none';

    try {
        // Load projections for both variants
        const baselineData = await loadProjections(experiment, baselineVariant, trait, promptSet);
        const compareData = await loadProjections(experiment, compareVariant, trait, promptSet);

        if (Object.keys(baselineData.projections).length === 0) {
            throw new Error(`No projections found for ${baselineVariant}/${promptSet}. Run: python inference/project_raw_activations_onto_traits.py --experiment ${experiment} --model-variant ${baselineVariant} --prompt-set ${promptSet}`);
        }

        if (Object.keys(compareData.projections).length === 0) {
            throw new Error(`No projections found for ${compareVariant}/${promptSet}. Run: python inference/project_raw_activations_onto_traits.py --experiment ${experiment} --model-variant ${compareVariant} --prompt-set ${promptSet}`);
        }

        // Get number of layers from model config
        const modelConfig = await window.getModelConfig(experiment);
        const numLayers = modelConfig.n_layers;

        // Aggregate by layer
        const baselineByLayer = aggregateByLayer(baselineData.projections, numLayers);
        const compareByLayer = aggregateByLayer(compareData.projections, numLayers);

        // Compute effect sizes per layer
        const results = [];
        for (let layer = 0; layer < numLayers; layer++) {
            const stats = computeCohenD(baselineByLayer[layer], compareByLayer[layer]);
            results.push({
                layer,
                ...stats
            });
        }

        // Find peak
        const peak = results.reduce((max, r) =>
            Math.abs(r.effectSize) > Math.abs(max.effectSize) ? r : max
        );

        // Display results
        displayResults(results, peak, {
            baselineVariant,
            compareVariant,
            trait,
            promptSet,
            baselineMetadata: baselineData.metadata,
            compareMetadata: compareData.metadata
        });

    } catch (error) {
        showError(error.message);
        console.error('Comparison error:', error);
    } finally {
        document.getElementById('comparison-loading').style.display = 'none';
    }
}

/**
 * Display comparison results
 */
function displayResults(results, peak, context) {
    const resultsDiv = document.getElementById('comparison-results');
    const summaryDiv = document.getElementById('results-summary');
    const chartDiv = document.getElementById('effect-size-chart');

    resultsDiv.style.display = 'block';

    // Summary
    const traitName = context.trait.split('/').pop();
    summaryDiv.innerHTML = `
        <div class="summary-grid">
            <div class="summary-item">
                <div class="summary-label">Baseline</div>
                <div class="summary-value">${context.baselineVariant}</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">Compare</div>
                <div class="summary-value">${context.compareVariant}</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">Trait</div>
                <div class="summary-value">${traitName}</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">Prompt Set</div>
                <div class="summary-value">${context.promptSet}</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">Peak Effect</div>
                <div class="summary-value">${peak.effectSize.toFixed(2)}σ @ L${peak.layer}</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">Responses</div>
                <div class="summary-value">${peak.n}</div>
            </div>
        </div>

        ${context.compareMetadata?.prefilled_from ?
            `<div class="metadata-note">
                <strong>Note:</strong> ${context.compareVariant} was prefilled with responses from ${context.compareMetadata.prefilled_from}
            </div>` : ''
        }
    `;

    // Plot effect size by layer
    renderEffectSizePlot(chartDiv, results, context);
}

/**
 * Render effect size plot using Plotly
 */
function renderEffectSizePlot(container, results, context) {
    const layers = results.map(r => r.layer);
    const effectSizes = results.map(r => r.effectSize);

    const trace = {
        x: layers,
        y: effectSizes,
        type: 'scatter',
        mode: 'lines+markers',
        name: `${context.trait.split('/').pop()}`,
        line: { width: 2 },
        marker: { size: 4 }
    };

    const layout = {
        title: `Effect Size by Layer: ${context.compareVariant} vs ${context.baselineVariant}`,
        xaxis: {
            title: 'Layer',
            dtick: 10
        },
        yaxis: {
            title: 'Effect Size (Cohen\'s d)',
            zeroline: true,
            zerolinewidth: 2,
            zerolinecolor: '#888'
        },
        hovermode: 'closest',
        showlegend: true,
        template: 'plotly_dark',
        plot_bgcolor: 'var(--bg-primary)',
        paper_bgcolor: 'var(--bg-primary)',
        font: { color: 'var(--text-primary)' }
    };

    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false
    };

    Plotly.newPlot(container, [trace], layout, config);
}

/**
 * Show error message
 */
function showError(message) {
    const errorDiv = document.getElementById('comparison-error');
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
    document.getElementById('comparison-results').style.display = 'none';
}


// ============================================================================
// Activation Diagnostics Functions
// ============================================================================

/**
 * Fetch massive activations data, using calibration.json as canonical source.
 * Calibration contains model-wide massive dims computed from neutral prompts.
 */
async function fetchMassiveActivationsData() {
    const modelVariant = window.state.experimentData?.experimentConfig?.defaults?.application || 'instruct';
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
        window.renderChart(plotDiv, traces, layout);

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
        window.renderChart(container, traces, layout);

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
