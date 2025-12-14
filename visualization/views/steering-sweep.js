// Steering Sweep - Heatmap visualization of steering experiments
// Shows layer × perturbation ratio → delta/coherence

// Helper to get display name (fallback if global not available)
function getTraitDisplayName(trait) {
    if (window.getDisplayName) return window.getDisplayName(trait);
    // Fallback: just use the trait name part
    const parts = trait.split('/');
    return parts[parts.length - 1];
}

async function renderSteeringSweep() {
    const contentArea = document.getElementById('content-area');

    // Show loading state only if fetch takes > 150ms
    const loadingTimeout = setTimeout(() => {
        contentArea.innerHTML = '<div class="loading">Loading steering sweep data...</div>';
    }, 150);

    // Get current trait from state or use default
    const traits = await discoverSteeringTraits();

    clearTimeout(loadingTimeout);

    if (traits.length === 0) {
        contentArea.innerHTML = `
            <div class="tool-view">
                <div class="no-data">
                    <p>No steering sweep data found</p>
                    <small>Run steering experiments with: <code>python analysis/steering/evaluate.py --experiment ${window.state.experimentData?.name || 'your_experiment'} --trait category/trait --layers 8,10,12 --find-coef</code></small>
                </div>
            </div>
        `;
        return;
    }

    // Default to first trait with sweep data
    const defaultTrait = traits[0];

    // Build the view
    contentArea.innerHTML = `
        <div class="tool-view">
            <!-- Page intro -->
            <div class="page-intro">
                <div class="page-intro-text">Steering sweep analysis: how steering effectiveness varies by layer and perturbation ratio.</div>
                <div id="steering-model-info" class="page-intro-model"></div>
                <div class="intro-example">
                    <div><span class="example-label">Formula:</span> perturbation_ratio = (coef × vector_norm) / activation_norm</div>
                    <div><span class="example-label">Sweet spot:</span> ratio ~0.5-1.0 for most layers</div>
                </div>
            </div>

            <!-- Coherence threshold control (applies to all sections) -->
            <div class="sweep-controls" style="margin-bottom: 16px;">
                <div class="control-group">
                    <label>Min Coherence:</label>
                    <input type="range" id="sweep-coherence-threshold" min="0" max="100" value="70" />
                    <span id="coherence-threshold-value">70</span>
                </div>
            </div>

            <!-- Best Vector per Layer (multi-trait from sidebar) -->
            <section id="best-vector-section">
                <h3 class="subsection-header">
                    <span class="subsection-num">1.</span>
                    <span class="subsection-title">Best Vector per Layer</span>
                    <span class="subsection-info-toggle" data-target="info-best-vector">►</span>
                </h3>
                <div class="subsection-info" id="info-best-vector">
                    For each selected trait (from sidebar), shows the best trait score achieved per layer across all 3 extraction methods (probe, gradient, mean_diff).
                    Each trait gets its own chart showing which method works best at which layer. Dashed line shows baseline (no steering).
                </div>
                <div id="best-vector-container"></div>
            </section>

            <!-- Trait Picker (for single-trait sections below) -->
            <div id="trait-picker-container"></div>

            <!-- Controls -->
            <div class="sweep-controls">
                <div class="control-group">
                    <label>Vector Type:</label>
                    <select id="sweep-vector-type">
                        <option value="full_vector" selected>Full Vector</option>
                        <option value="incremental">Incremental</option>
                    </select>
                </div>
                <div class="control-group">
                    <label>Method:</label>
                    <select id="sweep-method">
                        <option value="all" selected>All Methods</option>
                        <option value="probe">Probe</option>
                        <option value="gradient">Gradient</option>
                        <option value="mean_diff">Mean Diff</option>
                    </select>
                </div>
                <div class="control-group">
                    <label>Metric:</label>
                    <select id="sweep-metric">
                        <option value="delta" selected>Delta (trait change)</option>
                        <option value="coherence">Coherence</option>
                        <option value="trait">Trait Score</option>
                    </select>
                </div>
                <div class="control-group">
                    <label>
                        <input type="checkbox" id="sweep-interpolate" />
                        Interpolate gaps
                    </label>
                </div>
            </div>

            <!-- Main heatmap -->
            <section>
                <h3 class="subsection-header" id="heatmap-section">
                    <span class="subsection-num">2.</span>
                    <span class="subsection-title">Layer × Ratio Heatmap</span>
                    <span class="subsection-info-toggle" data-target="info-heatmap">►</span>
                </h3>
                <div class="subsection-info" id="info-heatmap">
                    Each cell shows the steering effect (delta or coherence) for a specific layer and perturbation ratio.
                    Cells with coherence below threshold are masked. Green = positive steering, Red = negative.
                </div>
                <div id="sweep-heatmap-container" class="chart-container-lg"></div>
            </section>

            <!-- Optimal curve -->
            <section>
                <h3 class="subsection-header" id="optimal-curve-section">
                    <span class="subsection-num">3.</span>
                    <span class="subsection-title">Optimal Ratio per Layer</span>
                    <span class="subsection-info-toggle" data-target="info-optimal">►</span>
                </h3>
                <div class="subsection-info" id="info-optimal">
                    For each layer, shows the best delta achieved (with coherence above threshold) and the ratio that achieved it.
                    This reveals the "U-shape": early and late layers often go negative regardless of ratio.
                </div>
                <div id="sweep-optimal-container" class="chart-container-md"></div>
            </section>

            <!-- Summary stats -->
            <section>
                <h3 class="subsection-header" id="summary-section">
                    <span class="subsection-num">4.</span>
                    <span class="subsection-title">Summary</span>
                    <span class="subsection-info-toggle" data-target="info-summary">►</span>
                </h3>
                <div class="subsection-info" id="info-summary">Best configuration found. Optimal layer and steering ratio.</div>
                <div id="sweep-summary-container"></div>
            </section>

            <!-- Raw results table -->
            <section>
                <h3 class="subsection-header" id="table-section">
                    <span class="subsection-num">5.</span>
                    <span class="subsection-title">All Results</span>
                    <span class="subsection-info-toggle" data-target="info-table">►</span>
                </h3>
                <div class="subsection-info" id="info-table">Complete results from steering experiments.</div>
                <div id="sweep-table-container" class="scrollable-container"></div>
            </section>

            <!-- Multi-layer heatmap section -->
            <section id="multi-layer-section" style="display: none;">
                <h3 class="subsection-header">
                    <span class="subsection-num">6.</span>
                    <span class="subsection-title">Multi-Layer Steering</span>
                    <span class="subsection-info-toggle" data-target="info-multilayer">►</span>
                </h3>
                <div class="subsection-info" id="info-multilayer">
                    Center × width grid showing where traits "live" in the model.
                    Y-axis = center layer, X-axis = width (# consecutive layers steered).
                    Find optimal multi-layer steering range.
                </div>
                <div class="sweep-controls" style="margin-bottom: 16px;">
                    <div class="control-group">
                        <label>Color by:</label>
                        <select id="multilayer-metric">
                            <option value="delta" selected>Delta (trait improvement)</option>
                            <option value="coherence">Coherence</option>
                            <option value="combined">Combined (delta × coh/100)</option>
                        </select>
                    </div>
                </div>
                <div id="multilayer-heatmap-container" class="chart-container-lg"></div>
                <div class="summary-grid" style="margin-top: 20px;">
                    <div class="summary-card">
                        <div class="summary-label">Top Configurations</div>
                        <div id="multilayer-top-configs"></div>
                    </div>
                    <div class="summary-card">
                        <div class="summary-label">vs Single-Layer Best</div>
                        <div id="multilayer-comparison"></div>
                    </div>
                </div>
            </section>
        </div>
    `;

    // Initial render
    await renderBestVectorPerLayer();
    await renderTraitPicker(traits);

    // Set default selected trait if not set
    if (!window.state.selectedSteeringTrait && traits.length > 0) {
        window.state.selectedSteeringTrait = defaultTrait;
    }

    await renderSweepData(window.state.selectedSteeringTrait || defaultTrait);

    // Setup event handlers
    document.getElementById('sweep-vector-type').addEventListener('change', () => updateSweepVisualizations());
    document.getElementById('sweep-method').addEventListener('change', () => updateSweepVisualizations());
    document.getElementById('sweep-metric').addEventListener('change', () => updateSweepVisualizations());

    document.getElementById('sweep-coherence-threshold').addEventListener('input', async (e) => {
        document.getElementById('coherence-threshold-value').textContent = e.target.value;
        // Re-render best vector section with new threshold
        await renderBestVectorPerLayer();
        // Re-render single-trait sections
        updateSweepVisualizations();
    });

    document.getElementById('sweep-interpolate').addEventListener('change', () => updateSweepVisualizations());

    // Setup info toggles
    setupSweepInfoToggles();
}


// Store current data for re-rendering on control changes
let currentSweepData = null;
let currentRawResults = null; // Store raw results.json for method filtering


/**
 * Update the steering model info display in the page intro
 */
function updateSteeringModelInfo(meta) {
    const container = document.getElementById('steering-model-info');
    if (!container) return;

    if (!meta || !meta.steering_model) {
        // Fall back to experiment config
        const config = window.state.experimentData?.experimentConfig;
        const steeringModel = config?.application_model || config?.model || 'unknown';
        container.innerHTML = `Steering model: <code>${steeringModel}</code>`;
        return;
    }

    let html = `Steering model: <code>${meta.steering_model}</code>`;

    if (meta.vector_source?.model) {
        html += ` · Vector from: <code>${meta.vector_source.model}</code>`;
    }

    if (meta.eval?.model) {
        html += ` · Eval: <code>${meta.eval.model}</code> (${meta.eval.method || 'unknown'})`;
    }

    container.innerHTML = html;
}


async function discoverSteeringTraits() {
    // Discover traits with steering data
    if (!window.state.experimentData?.name) return [];

    try {
        const baseUrl = '/' + window.paths.get('steering.base');
        const response = await fetch(baseUrl + '/');
        if (!response.ok) return [];

        const html = await response.text();
        // Parse directory listing to find trait folders
        const traitMatches = html.matchAll(/href="([^"]+)\/"/g);
        const traits = [];

        for (const match of traitMatches) {
            const folder = match[1];
            if (folder !== '..' && !folder.startsWith('.')) {
                // Check for subfolders (category/trait structure)
                try {
                    const subResponse = await fetch(`${baseUrl}/${folder}/`);
                    if (subResponse.ok) {
                        const subHtml = await subResponse.text();
                        const subMatches = subHtml.matchAll(/href="([^"]+)\/"/g);
                        for (const subMatch of subMatches) {
                            const subFolder = subMatch[1];
                            if (subFolder !== '..' && !subFolder.startsWith('.')) {
                                // Verify this trait actually has results
                                const trait = `${folder}/${subFolder}`;
                                const resultsUrl = '/' + window.paths.get('steering.results', { trait });
                                const resultsCheck = await fetch(resultsUrl, { method: 'HEAD' });
                                if (resultsCheck.ok) {
                                    traits.push(trait);
                                }
                            }
                        }
                    }
                } catch (subError) {
                    // Ignore individual folder errors
                    console.log(`Skipping folder ${folder}:`, subError);
                }
            }
        }

        return traits;
    } catch (e) {
        console.error('Failed to discover steering traits:', e);
        return [];
    }
}


async function renderSweepData(trait) {
    const experiment = window.state.experimentData?.name;
    if (!experiment) return;

    // Try to load sweep_results.json first, fall back to results.json
    let data = null;
    let steeringMeta = null;  // Capture steering_model, vector_source, eval

    const sweepUrl = '/' + window.paths.get('steering.sweep_results', { trait });
    const headCheck = await fetch(sweepUrl, { method: 'HEAD' }).catch(() => null);
    if (headCheck?.ok) {
        const response = await fetch(sweepUrl);
        data = await response.json();
    }

    if (!data) {
        // Fall back to regular results.json and convert to sweep format
        try {
            const resultsUrl = '/' + window.paths.get('steering.results', { trait });
            const response = await fetch(resultsUrl);
            if (response.ok) {
                const results = await response.json();
                currentRawResults = results; // Store for method filtering
                // Capture metadata
                steeringMeta = {
                    steering_model: results.steering_model,
                    vector_source: results.vector_source,
                    eval: results.eval
                };
                data = convertResultsToSweepFormat(results);
            }
        } catch (e) {
            console.error('Failed to load steering results:', e);
        }
    }

    // Update model info display
    updateSteeringModelInfo(steeringMeta);

    if (!data) {
        document.getElementById('sweep-heatmap-container').innerHTML = '<p class="no-data">No data for this trait</p>';
        document.getElementById('sweep-optimal-container').innerHTML = '';
        document.getElementById('sweep-summary-container').innerHTML = '';
        document.getElementById('sweep-table-container').innerHTML = '';
        return;
    }

    // Check if we have any single-layer data
    const fullVector = data.full_vector || {};
    if (Object.keys(fullVector).length === 0) {
        document.getElementById('sweep-heatmap-container').innerHTML = `
            <p class="no-data">No single-layer steering runs found.<br>
            <small>Heatmap requires single-layer runs. Multi-layer runs are not visualized here.</small></p>
        `;
        document.getElementById('sweep-optimal-container').innerHTML = '';
        document.getElementById('sweep-summary-container').innerHTML = '';
        document.getElementById('sweep-table-container').innerHTML = '';
        return;
    }

    currentSweepData = data;
    updateSweepVisualizations();

    // Try to load multi-layer heatmap data for this trait
    const multiData = await loadMultiLayerData(trait);
    renderMultiLayerSection(multiData);
}


/**
 * Render Best Vector per Layer section (multi-trait)
 * Shows one chart per selected trait with 3 method lines each
 */
async function renderBestVectorPerLayer() {
    const container = document.getElementById('best-vector-container');
    const selectedTraits = Array.from(window.state.selectedTraits || []);

    if (selectedTraits.length === 0) {
        container.innerHTML = '<p class="no-data">No traits selected. Select traits from the sidebar to compare methods.</p>';
        return;
    }

    container.innerHTML = '<div class="loading">Loading method comparison data...</div>';

    // Get coherence threshold from slider
    const coherenceThresholdEl = document.getElementById('sweep-coherence-threshold');
    const coherenceThreshold = coherenceThresholdEl ? parseInt(coherenceThresholdEl.value) : 70;

    const charts = [];

    for (const trait of selectedTraits) {
        // Load results.json for this trait
        try {
            const resultsUrl = '/' + window.paths.get('steering.results', { trait });
            const response = await fetch(resultsUrl);
            if (!response.ok) continue;

            const results = await response.json();
            const baseline = results.baseline?.trait_mean || 0;
            const runs = results.runs || [];

            // Group by method and layer, find best trait score per (method, layer)
            const methodData = { probe: {}, gradient: {}, mean_diff: {} };

            runs.forEach(run => {
                const config = run.config || {};
                const result = run.result || {};
                const layers = config.layers || [];
                const methods = config.methods || [];

                // Only single-layer runs
                if (layers.length !== 1) return;
                if (methods.length !== 1) return;

                const layer = layers[0];
                const method = methods[0];
                const coherence = result.coherence_mean || 0;
                const traitScore = result.trait_mean || 0;

                // Filter by coherence
                if (coherence < coherenceThreshold) return;

                // Track best trait score for this (method, layer)
                if (!methodData[method]) methodData[method] = {};
                if (!methodData[method][layer] || traitScore > methodData[method][layer]) {
                    methodData[method][layer] = traitScore;
                }
            });

            // Create traces for each method
            const traces = [];
            const methodColors = {
                probe: '#4a9eff',      // light blue
                gradient: '#51cf66',   // light green
                mean_diff: '#cc5de8'   // light purple
            };
            const methodNames = {
                probe: 'Probe',
                gradient: 'Gradient',
                mean_diff: 'Mean Diff'
            };

            Object.entries(methodData).forEach(([method, layerScores]) => {
                const layers = Object.keys(layerScores).map(Number).sort((a, b) => a - b);
                const scores = layers.map(l => layerScores[l]);

                if (layers.length > 0) {
                    traces.push({
                        x: layers,
                        y: scores,
                        type: 'scatter',
                        mode: 'lines+markers',
                        name: methodNames[method] || method,
                        line: { width: 2, color: methodColors[method] },
                        marker: { size: 5 },
                        hovertemplate: `${methodNames[method]}<br>L%{x}<br>Score: %{y:.1f}<extra></extra>`
                    });
                }
            });

            // Add baseline trace
            if (traces.length > 0) {
                const allLayers = traces.flatMap(t => t.x);
                const minLayer = Math.min(...allLayers);
                const maxLayer = Math.max(...allLayers);

                traces.unshift({
                    x: [minLayer, maxLayer],
                    y: [baseline, baseline],
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Baseline',
                    line: { dash: 'dash', color: '#888', width: 1 },
                    hovertemplate: `Baseline<br>Score: ${baseline.toFixed(1)}<extra></extra>`,
                    showlegend: true
                });

                const chartId = `best-vector-chart-${trait.replace(/\//g, '-')}`;
                charts.push({ trait, chartId, traces, methodData });
            }
        } catch (e) {
            console.error(`Failed to load data for ${trait}:`, e);
        }
    }

    // Render all charts
    if (charts.length === 0) {
        container.innerHTML = '<p class="no-data">No steering data found for selected traits.</p>';
        return;
    }

    container.innerHTML = charts.map(({ trait, chartId }) => `
        <div class="trait-chart-wrapper">
            <div class="trait-chart-title">${getTraitDisplayName(trait)}</div>
            <div id="${chartId}" class="chart-container-sm"></div>
            <div id="${chartId}-similarity" class="similarity-heatmap-container"></div>
        </div>
    `).join('');

    // Plot each chart
    for (const { chartId, traces } of charts) {
        const layout = window.getPlotlyLayout ? window.getPlotlyLayout({
            margin: { l: 50, r: 20, t: 5, b: 35 },
            xaxis: { title: 'Layer', dtick: 5, tickfont: { size: 10 } },
            yaxis: { title: 'Trait Score', tickfont: { size: 10 } },
            height: 180,
            showlegend: true,
            legend: { orientation: 'h', y: 1.15, x: 0, font: { size: 10 } }
        }) : {
            xaxis: { title: 'Layer' },
            yaxis: { title: 'Trait Score' },
            height: 180
        };

        Plotly.newPlot(chartId, traces, layout, { displayModeBar: false, responsive: true });
    }

    // Render similarity heatmaps
    for (const { trait, chartId, methodData } of charts) {
        await renderMethodSimilarityHeatmap(trait, chartId, methodData);
    }
}

/**
 * Render cosine similarity heatmap between methods at each layer
 */
async function renderMethodSimilarityHeatmap(trait, chartId, methodData) {
    const container = document.getElementById(`${chartId}-similarity`);
    if (!container) return;

    const experiment = window.state.currentExperiment;
    if (!experiment) return;

    try {
        // Load extraction_evaluation.json for similarity data
        const evalUrl = '/' + window.paths.get('extraction_eval.evaluation', {});
        const response = await fetch(evalUrl);
        if (!response.ok) {
            container.innerHTML = '<div style="font-size: 10px; color: var(--text-tertiary); padding: 4px;">Run extraction_evaluation.py to compute similarities</div>';
            return;
        }

        const evalData = await response.json();
        const methodSims = evalData.method_similarities || {};
        const traitSims = methodSims[trait];

        if (!traitSims || Object.keys(traitSims).length === 0) {
            container.innerHTML = '<div style="font-size: 10px; color: var(--text-tertiary); padding: 4px;">No similarity data for this trait</div>';
            return;
        }

        // Get layers with similarity data
        const layers = Object.keys(traitSims).map(Number).sort((a, b) => a - b);

        // Prepare data for heatmap: 3 rows (one per method pair), N columns (layers)
        // Keys are alphabetically sorted in backend: gradient_X, mean_diff_X
        const pairs = ['gradient_mean_diff', 'gradient_probe', 'mean_diff_probe'];
        const pairLabels = {
            'gradient_mean_diff': 'Grd↔MD',
            'gradient_probe': 'Grd↔Prb',
            'mean_diff_probe': 'MD↔Prb'
        };

        // Build z matrix (3 rows × layers columns)
        const z = pairs.map(pair => {
            return layers.map(layer => {
                const layerSims = traitSims[layer] || {};
                return layerSims[pair] !== undefined ? layerSims[pair] : null;
            });
        });

        // Create heatmap
        const heatmapId = `${chartId}-similarity-heatmap`;
        container.innerHTML = `<div id="${heatmapId}" style="height: 60px; margin-top: 56px;"></div>`;

        const trace = {
            z: z,
            x: layers,
            y: pairs.map(p => pairLabels[p]),
            type: 'heatmap',
            colorscale: window.ASYMB_COLORSCALE,
            zmin: 0,
            zmax: 1,
            hoverongaps: false,
            hovertemplate: 'L%{x}<br>%{y}: %{z:.3f}<extra></extra>',
            showscale: false
        };

        const layout = window.getPlotlyLayout ? window.getPlotlyLayout({
            margin: { l: 50, r: 10, t: 5, b: 20 },
            xaxis: {
                title: '',
                tickfont: { size: 8 },
                dtick: 5
            },
            yaxis: {
                tickfont: { size: 8 },
                automargin: true
            },
            height: 60,
        }) : {
            xaxis: { title: '' },
            yaxis: {},
            height: 60,
            margin: { l: 50, r: 10, t: 5, b: 20 }
        };

        Plotly.newPlot(heatmapId, [trace], layout, { displayModeBar: false, responsive: true });

    } catch (e) {
        console.error(`Failed to render similarity for ${trait}:`, e);
        container.innerHTML = '<div style="font-size: 10px; color: var(--text-tertiary); padding: 4px;">Error loading similarity data</div>';
    }
}


/**
 * Render trait picker (inline buttons)
 */
async function renderTraitPicker(traits) {
    const container = document.getElementById('trait-picker-container');

    if (!traits || traits.length === 0) {
        container.innerHTML = '';
        return;
    }

    const currentTrait = window.state.selectedSteeringTrait || traits[0];

    container.innerHTML = `
        <div class="trait-picker">
            <span class="tp-label">Trait:</span>
            <div class="tp-buttons">
                ${traits.map(t => `
                    <button class="tp-btn ${t === currentTrait ? 'active' : ''}" data-trait="${t}">
                        ${getTraitDisplayName(t)}
                    </button>
                `).join('')}
            </div>
        </div>
    `;

    // Setup event listeners
    container.querySelectorAll('.tp-btn').forEach(btn => {
        btn.addEventListener('click', async () => {
            const selectedTrait = btn.dataset.trait;
            window.state.selectedSteeringTrait = selectedTrait;

            // Update active state
            container.querySelectorAll('.tp-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            // Re-render single-trait sections
            await renderSweepData(selectedTrait);
        });
    });
}


function convertResultsToSweepFormat(results, methodFilter = null) {
    // Convert results.json format to sweep_results.json format
    // Actual format: { trait, baseline: {trait_mean, ...}, runs: [{config: {layers, coefficients, ...}, result: {trait_mean, coherence_mean, ...}}, ...] }
    // methodFilter: optional method to filter by (e.g., 'probe', 'gradient', 'mean_diff')
    const runs = results.runs || [];
    if (runs.length === 0) return null;

    const baseline = results.baseline?.trait_mean || 50;

    // Group by layer and coefficient
    const fullVector = {};

    runs.forEach(run => {
        const config = run.config || {};
        const result = run.result || {};

        const layers = config.layers || [];
        const coefficients = config.coefficients || [];
        const methods = config.methods || [];

        // Only single-layer runs for heatmap (multi-layer would need different viz)
        if (layers.length !== 1) return;
        if (coefficients.length !== 1) return;

        // Filter by method if specified
        if (methodFilter && methods.length > 0 && methods[0] !== methodFilter) return;

        const layer = layers[0];
        const coef = coefficients[0];

        // Use coefficient as x-axis value
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
            // Update if this run is newer (later in the array)
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
        full_vector: fullVector,
        incremental: {}, // Would need separate runs with incremental flag to populate
        _isConverted: true // Flag to indicate this was converted from results.json (uses raw coefficients, not ratios)
    };
}


function updateSweepVisualizations() {
    if (!currentSweepData) return;

    const vectorType = document.getElementById('sweep-vector-type').value;
    const method = document.getElementById('sweep-method').value;
    const metric = document.getElementById('sweep-metric').value;
    const coherenceThreshold = parseInt(document.getElementById('sweep-coherence-threshold').value);
    const interpolate = document.getElementById('sweep-interpolate').checked;

    // If method filter is active and we have raw results, reconvert with filter
    let data;
    if (method !== 'all' && currentRawResults) {
        const filteredData = convertResultsToSweepFormat(currentRawResults, method);
        data = filteredData?.[vectorType] || filteredData?.full_vector || {};
    } else {
        data = currentSweepData[vectorType] || currentSweepData.full_vector || {};
    }

    renderSweepHeatmap(data, metric, coherenceThreshold, interpolate);
    renderOptimalCurve(data, coherenceThreshold);
    renderSweepSummary({ full_vector: data, baseline_trait: currentSweepData.baseline_trait }, coherenceThreshold);
    renderSweepTable(data, coherenceThreshold);
}


function renderSweepHeatmap(data, metric, coherenceThreshold, interpolate = false) {
    const container = document.getElementById('sweep-heatmap-container');

    const layers = Object.keys(data).map(Number).sort((a, b) => a - b);
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

    // If interpolating, create a denser grid
    let interpolatedRatios = ratios;
    if (interpolate && ratios.length > 1) {
        const minR = Math.min(...ratios);
        const maxR = Math.max(...ratios);
        const numSteps = 50; // Number of interpolation points
        interpolatedRatios = [];
        for (let i = 0; i <= numSteps; i++) {
            interpolatedRatios.push(minR + (maxR - minR) * i / numSteps);
        }
    }

    // Build matrix
    const metricKey = metric === 'delta' ? 'deltas' : metric === 'coherence' ? 'coherences' : 'traits';
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

        if (!interpolate) {
            // Original behavior: only show tested points
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

        // Sort valid points by ratio
        validPoints.sort((a, b) => a.r - b.r);

        return interpolatedRatios.map(targetR => {
            // Find surrounding points for interpolation
            let lower = null, upper = null;
            for (const pt of validPoints) {
                if (pt.r <= targetR) lower = pt;
                if (pt.r >= targetR && upper === null) upper = pt;
            }

            // Exact match
            if (lower && lower.r === targetR) return lower.v;
            if (upper && upper.r === targetR) return upper.v;

            // Extrapolation not allowed - return null if outside range
            if (!lower || !upper) return null;

            // Linear interpolation
            const t = (targetR - lower.r) / (upper.r - lower.r);
            return lower.v + t * (upper.v - lower.v);
        });
    });

    // Use interpolated ratios for x-axis if interpolating
    const xRatios = interpolate ? interpolatedRatios : ratios;

    // Determine color scale based on metric
    let colorscale, zmid, zmin, zmax;
    if (metric === 'delta') {
        colorscale = [
            [0, '#aa5656'],    // negative - red
            [0.5, '#e0e0de'],  // zero - neutral
            [1, '#3d7435']     // positive - green
        ];
        zmid = 0;
        const allVals = matrix.flat().filter(v => v !== null);
        const absMax = Math.max(Math.abs(Math.min(...allVals, 0)), Math.abs(Math.max(...allVals, 0)));
        zmin = -absMax;
        zmax = absMax;
    } else if (metric === 'coherence') {
        colorscale = window.ASYMB_COLORSCALE || 'Viridis';
        zmin = 0;
        zmax = 100;
        zmid = 50;
    } else {
        colorscale = window.ASYMB_COLORSCALE || 'Viridis';
        zmin = 0;
        zmax = 100;
        zmid = 50;
    }

    // Apply sqrt scaling to x-axis for better visualization of wide coefficient ranges
    const useSqrtScale = currentSweepData._isConverted && xRatios.length > 0 && Math.max(...xRatios) > 100;
    const xValues = useSqrtScale ? xRatios.map(r => Math.sqrt(r)) : xRatios;

    // Build custom hover text with original coefficient values
    const hoverText = matrix.map((row, layerIdx) =>
        row.map((val, ratioIdx) => {
            if (val === null) return '';
            const coef = xRatios[ratioIdx];
            const metricLabel = metric === 'delta' ? 'Delta' : metric === 'coherence' ? 'Coherence' : 'Trait';
            return `Layer L${layers[layerIdx]}<br>Coef: ${coef.toFixed(0)}<br>${metricLabel}: ${val.toFixed(1)}${interpolate ? '<br>(interpolated)' : ''}`;
        })
    );

    const trace = {
        z: matrix,
        x: xValues,
        y: layers.map(l => `L${l}`),
        type: 'heatmap',
        colorscale: colorscale,
        zmid: zmid,
        zmin: zmin,
        zmax: zmax,
        hoverongaps: false,
        connectgaps: interpolate, // Smooth rendering when interpolating
        hovertemplate: '%{text}<extra></extra>',
        text: hoverText,
        colorbar: {
            title: { text: metric === 'delta' ? 'Delta' : metric === 'coherence' ? 'Coherence' : 'Trait', font: { size: 11 } }
        }
    };

    // Label depends on whether we're using sweep_results (ratios) or converted results (coefficients)
    const xAxisLabel = currentSweepData._isConverted
        ? (useSqrtScale ? 'Coefficient (√ scale)' : 'Coefficient')
        : 'Perturbation Ratio';

    // Generate tick values for sqrt scale showing original coefficients
    let xAxisConfig = { title: xAxisLabel, tickfont: { size: 10 } };
    if (useSqrtScale) {
        // Pick nice tick values from the original coefficients
        const maxCoef = Math.max(...xRatios);
        const tickCoefs = [0, 100, 250, 500, 1000, 1500, 2000, 2500, 3000].filter(v => v <= maxCoef * 1.1);
        xAxisConfig.tickvals = tickCoefs.map(c => Math.sqrt(c));
        xAxisConfig.ticktext = tickCoefs.map(c => c.toString());
    }

    const layout = window.getPlotlyLayout ? window.getPlotlyLayout({
        margin: { l: 50, r: 80, t: 20, b: 50 },
        xaxis: xAxisConfig,
        yaxis: { title: 'Layer', tickfont: { size: 10 }, autorange: 'reversed' },
        height: Math.max(300, layers.length * 20 + 100)
    }) : {
        margin: { l: 50, r: 80, t: 20, b: 50 },
        xaxis: xAxisConfig,
        yaxis: { title: 'Layer', autorange: 'reversed' },
        height: 400
    };

    Plotly.newPlot(container, [trace], layout, { displayModeBar: false, responsive: true });
}


function renderOptimalCurve(data, coherenceThreshold) {
    const container = document.getElementById('sweep-optimal-container');

    const layers = Object.keys(data).map(Number).sort((a, b) => a - b);
    if (layers.length === 0) {
        container.innerHTML = '';
        return;
    }

    // Find best delta per layer (with coherence above threshold)
    const optimalData = layers.map(layer => {
        const layerData = data[layer];
        let bestDelta = null;
        let bestRatio = null;
        let bestCoherence = null;

        layerData.ratios.forEach((ratio, idx) => {
            const coherence = layerData.coherences[idx];
            const delta = layerData.deltas[idx];

            if (coherence >= coherenceThreshold) {
                if (bestDelta === null || delta > bestDelta) {
                    bestDelta = delta;
                    bestRatio = ratio;
                    bestCoherence = coherence;
                }
            }
        });

        return { layer, bestDelta, bestRatio, bestCoherence };
    }).filter(d => d.bestDelta !== null);

    if (optimalData.length === 0) {
        container.innerHTML = '<p class="no-data">No data above coherence threshold</p>';
        return;
    }

    // Plot optimal delta curve
    const deltaTrace = {
        x: optimalData.map(d => d.layer),
        y: optimalData.map(d => d.bestDelta),
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Best Delta',
        marker: { size: 8 },
        line: { width: 2 },
        hovertemplate: 'L%{x}<br>Delta: %{y:.1f}<br>Ratio: %{text}<extra></extra>',
        text: optimalData.map(d => d.bestRatio.toFixed(2))
    };

    // Add zero line
    const zeroLine = {
        x: [Math.min(...layers), Math.max(...layers)],
        y: [0, 0],
        type: 'scatter',
        mode: 'lines',
        name: 'Baseline',
        line: { dash: 'dash', color: '#888', width: 1 },
        hoverinfo: 'skip'
    };

    const layout = window.getPlotlyLayout ? window.getPlotlyLayout({
        margin: { l: 50, r: 20, t: 20, b: 50 },
        xaxis: { title: 'Layer', dtick: 5, tickfont: { size: 10 } },
        yaxis: { title: 'Best Delta', tickfont: { size: 10 } },
        height: 250,
        showlegend: false
    }) : {
        margin: { l: 50, r: 20, t: 20, b: 50 },
        xaxis: { title: 'Layer' },
        yaxis: { title: 'Best Delta' },
        height: 250
    };

    Plotly.newPlot(container, [zeroLine, deltaTrace], layout, { displayModeBar: false, responsive: true });
}


function renderSweepSummary(data, coherenceThreshold) {
    const container = document.getElementById('sweep-summary-container');

    const fullVector = data.full_vector || {};
    const layers = Object.keys(fullVector).map(Number).sort((a, b) => a - b);

    if (layers.length === 0) {
        container.innerHTML = '<p class="no-data">No summary available</p>';
        return;
    }

    // Find overall best
    let bestLayer = null;
    let bestRatio = null;
    let bestDelta = -Infinity;
    let bestCoherence = 0;

    layers.forEach(layer => {
        const layerData = fullVector[layer];
        layerData.ratios.forEach((ratio, idx) => {
            const coherence = layerData.coherences[idx];
            const delta = layerData.deltas[idx];

            if (coherence >= coherenceThreshold && delta > bestDelta) {
                bestDelta = delta;
                bestRatio = ratio;
                bestLayer = layer;
                bestCoherence = coherence;
            }
        });
    });

    // Find sweet spot layers (positive delta with good coherence)
    const sweetSpotLayers = layers.filter(layer => {
        const layerData = fullVector[layer];
        return layerData.ratios.some((_, idx) =>
            layerData.coherences[idx] >= 70 && layerData.deltas[idx] > 10
        );
    });

    const baseline = data.baseline_trait || 'N/A';

    container.innerHTML = `
        <div class="summary-grid">
            <div class="summary-card">
                <div class="summary-label">Baseline Trait</div>
                <div class="summary-value">${typeof baseline === 'number' ? baseline.toFixed(1) : baseline}</div>
            </div>
            <div class="summary-card highlight">
                <div class="summary-label">Best Configuration</div>
                <div class="summary-value">L${bestLayer} @ ${bestRatio?.toFixed(2) || 'N/A'}</div>
                <div class="summary-detail">Delta: +${bestDelta?.toFixed(1) || 'N/A'}, Coherence: ${bestCoherence?.toFixed(0) || 'N/A'}</div>
            </div>
            <div class="summary-card">
                <div class="summary-label">Sweet Spot Layers</div>
                <div class="summary-value">${sweetSpotLayers.length > 0 ? sweetSpotLayers.join(', ') : 'None'}</div>
                <div class="summary-detail">(Delta > 10, Coherence > 70)</div>
            </div>
            <div class="summary-card">
                <div class="summary-label">Layers Tested</div>
                <div class="summary-value">${layers.length}</div>
                <div class="summary-detail">L${Math.min(...layers)} - L${Math.max(...layers)}</div>
            </div>
        </div>
    `;
}


function renderSweepTable(data, coherenceThreshold) {
    const container = document.getElementById('sweep-table-container');

    const layers = Object.keys(data).map(Number).sort((a, b) => a - b);
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
                    <th>Ratio</th>
                    <th>Delta</th>
                    <th>Coherence</th>
                    <th>Trait</th>
                </tr>
            </thead>
            <tbody>
                ${rows.map(r => {
                    const masked = r.coherence < coherenceThreshold;
                    const deltaClass = r.delta > 15 ? 'quality-good' : r.delta > 5 ? 'quality-ok' : r.delta < 0 ? 'quality-bad' : '';
                    const cohClass = r.coherence > 80 ? 'quality-good' : r.coherence > 60 ? 'quality-ok' : 'quality-bad';
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


function setupSweepInfoToggles() {
    const container = document.querySelector('.tool-view');
    if (!container || container.dataset.sweepTogglesSetup) return;
    container.dataset.sweepTogglesSetup = 'true';

    container.addEventListener('click', (e) => {
        const toggle = e.target.closest('.subsection-info-toggle');
        if (!toggle) return;

        const targetId = toggle.dataset.target;
        const infoDiv = document.getElementById(targetId);
        if (infoDiv) {
            const isShown = infoDiv.classList.toggle('show');
            toggle.textContent = isShown ? '▼' : '►';
        }
    });
}


// =============================================================================
// MULTI-LAYER HEATMAP SECTION
// =============================================================================

let multiLayerData = null;

async function loadMultiLayerData(trait) {
    if (!window.state.experimentData?.name) return null;

    const heatmapPath = '/' + window.paths.get('steering.center_width_heatmap', { trait });
    const resultsPath = '/' + window.paths.get('steering.results', { trait });

    try {
        const heatmapResp = await fetch(heatmapPath);
        if (!heatmapResp.ok) return null;

        const heatmapData = await heatmapResp.json();
        const resultsResp = await fetch(resultsPath);
        const resultsData = resultsResp.ok ? await resultsResp.json() : null;

        return { heatmapData, resultsData };
    } catch (e) {
        console.log('No multi-layer data for trait:', trait);
        return null;
    }
}


function renderMultiLayerSection(data) {
    const section = document.getElementById('multi-layer-section');
    if (!data) {
        section.style.display = 'none';
        return;
    }

    section.style.display = 'block';
    multiLayerData = data;

    renderMultiLayerHeatmapPlot('delta');
    renderMultiLayerTopConfigs(data.heatmapData);
    renderMultiLayerComparison(data.heatmapData, data.resultsData);

    // Wire up metric selector
    const metricSelect = document.getElementById('multilayer-metric');
    if (metricSelect && !metricSelect.dataset.bound) {
        metricSelect.dataset.bound = 'true';
        metricSelect.addEventListener('change', (e) => {
            renderMultiLayerHeatmapPlot(e.target.value);
        });
    }
}


function renderMultiLayerHeatmapPlot(metric) {
    if (!multiLayerData) return;

    const { heatmapData } = multiLayerData;
    const { centers, widths, delta_grid, coherence_grid } = heatmapData;
    const container = document.getElementById('multilayer-heatmap-container');

    // Choose data based on metric
    let z, zLabel, zMin, zMax;
    if (metric === 'delta') {
        z = delta_grid;
        zLabel = 'Delta';
        zMin = 0;
        zMax = 35;
    } else if (metric === 'coherence') {
        z = coherence_grid;
        zLabel = 'Coherence';
        zMin = 50;
        zMax = 90;
    } else {
        z = delta_grid.map((row, i) =>
            row.map((d, j) => {
                const c = coherence_grid[i][j];
                return (d !== null && c !== null) ? d * (c / 100) : null;
            })
        );
        zLabel = 'Combined';
        zMin = 0;
        zMax = 25;
    }

    // Create hover text
    const hovertext = centers.map((center, i) =>
        widths.map((width, j) => {
            const delta = delta_grid[i][j];
            const coherence = coherence_grid[i][j];
            if (delta === null) return '';

            const half = Math.floor(width / 2);
            const layerRange = `L${center - half}-L${center + half}`;
            return `Center: L${center}<br>Width: ${width}<br>Layers: ${layerRange}<br>Delta: ${delta?.toFixed(1) || '--'}<br>Coherence: ${coherence?.toFixed(1) || '--'}`;
        })
    );

    const colorscale = [
        [0, '#aa5656'],
        [0.5, '#e0e0de'],
        [1, '#3d7435']
    ];

    const trace = {
        type: 'heatmap',
        x: widths.map(w => `W=${w}`),
        y: centers.map(c => `L${c}`),
        z: z,
        hovertext: hovertext,
        hoverinfo: 'text',
        colorscale: colorscale,
        colorbar: { title: zLabel, titleside: 'right' },
        zmin: zMin,
        zmax: zMax
    };

    const layout = window.getPlotlyLayout ? window.getPlotlyLayout({
        margin: { l: 50, r: 80, t: 20, b: 50 },
        xaxis: { title: 'Width (# layers steered)' },
        yaxis: { title: 'Center Layer', autorange: 'reversed' },
        height: Math.max(300, centers.length * 20 + 100)
    }) : {
        margin: { l: 50, r: 80, t: 20, b: 50 },
        xaxis: { title: 'Width (# layers steered)' },
        yaxis: { title: 'Center Layer', autorange: 'reversed' },
        height: 400
    };

    Plotly.newPlot(container, [trace], layout, { displayModeBar: false, responsive: true });
}


function renderMultiLayerTopConfigs(heatmapData) {
    const { centers, widths, delta_grid, coherence_grid } = heatmapData;
    const container = document.getElementById('multilayer-top-configs');

    const configs = [];
    centers.forEach((center, ci) => {
        widths.forEach((width, wi) => {
            const delta = delta_grid[ci][wi];
            const coherence = coherence_grid[ci][wi];
            if (delta !== null) {
                const half = Math.floor(width / 2);
                configs.push({
                    center,
                    width,
                    layers: `L${center - half}-${center + half}`,
                    delta,
                    coherence
                });
            }
        });
    });

    configs.sort((a, b) => b.delta - a.delta);

    container.innerHTML = `
        <table class="data-table" style="font-size: 11px;">
            <thead>
                <tr><th>#</th><th>Layers</th><th>Δ</th><th>Coh</th></tr>
            </thead>
            <tbody>
                ${configs.slice(0, 5).map((c, i) => `
                    <tr>
                        <td>${i + 1}</td>
                        <td>${c.layers}</td>
                        <td class="${c.delta > 25 ? 'quality-good' : ''}">${c.delta.toFixed(1)}</td>
                        <td class="${c.coherence > 80 ? 'quality-good' : c.coherence < 70 ? 'quality-bad' : ''}">${c.coherence.toFixed(0)}</td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;
}


function renderMultiLayerComparison(heatmapData, resultsData) {
    const container = document.getElementById('multilayer-comparison');

    if (!resultsData) {
        container.innerHTML = '<p style="color: var(--text-tertiary); font-size: 11px;">No results.json found</p>';
        return;
    }

    const baseline = resultsData.baseline?.trait_mean || 0;
    const { centers, widths, delta_grid, coherence_grid } = heatmapData;

    // Find best single-layer
    let bestSingle = null;
    for (const run of resultsData.runs || []) {
        if (run.config.layers.length === 1) {
            const trait = run.result?.trait_mean;
            const coherence = run.result?.coherence_mean;
            if (trait && (!bestSingle || trait > bestSingle.trait)) {
                bestSingle = {
                    layer: run.config.layers[0],
                    trait,
                    coherence,
                    delta: trait - baseline
                };
            }
        }
    }

    // Find best multi-layer
    let bestMulti = null;
    centers.forEach((center, ci) => {
        widths.forEach((width, wi) => {
            if (width === 1) return;
            const delta = delta_grid[ci][wi];
            const coherence = coherence_grid[ci][wi];
            if (delta !== null && (!bestMulti || delta > bestMulti.delta)) {
                const half = Math.floor(width / 2);
                bestMulti = {
                    layers: `L${center - half}-${center + half}`,
                    delta,
                    coherence
                };
            }
        });
    });

    if (bestSingle && bestMulti) {
        const deltaDiff = bestMulti.delta - bestSingle.delta;

        container.innerHTML = `
            <table class="data-table" style="font-size: 11px;">
                <tr><td></td><td><strong>1-Layer</strong></td><td><strong>Multi</strong></td></tr>
                <tr><td>Config</td><td>L${bestSingle.layer}</td><td>${bestMulti.layers}</td></tr>
                <tr>
                    <td>Delta</td>
                    <td>${bestSingle.delta.toFixed(1)}</td>
                    <td>${bestMulti.delta.toFixed(1)}</td>
                </tr>
                <tr>
                    <td>Gain</td>
                    <td colspan="2" class="${deltaDiff > 0 ? 'quality-good' : 'quality-bad'}">
                        ${deltaDiff > 0 ? '+' : ''}${deltaDiff.toFixed(1)}
                    </td>
                </tr>
            </table>
        `;
    } else {
        container.innerHTML = '<p style="color: var(--text-tertiary); font-size: 11px;">Insufficient data</p>';
    }
}


// Export
window.renderSteeringSweep = renderSteeringSweep;
