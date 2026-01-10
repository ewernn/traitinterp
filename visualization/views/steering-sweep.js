// Steering Sweep - Heatmap visualization of steering experiments
// Shows layer × perturbation ratio → delta/coherence

async function renderSteeringSweep() {
    const contentArea = document.getElementById('content-area');

    // Show loading state only if fetch takes > 150ms
    const loadingTimeout = setTimeout(() => {
        contentArea.innerHTML = '<div class="loading">Loading steering sweep data...</div>';
    }, 150);

    // Get current trait from state or use default
    const traits = await discoverSteeringTraits();
    discoveredSteeringTraits = traits; // Store for use by other functions

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
let discoveredSteeringTraits = []; // All discovered steering traits


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
    // Discover traits with steering data by recursively finding all results.json files
    if (!window.state.experimentData?.name) return [];

    const baseUrl = '/' + window.paths.get('steering.base');
    const traits = [];

    // Recursively search for results.json files
    async function searchDir(url, pathParts = []) {
        try {
            const response = await fetch(url);
            if (!response.ok) return;

            const html = await response.text();

            // Check if results.json exists in this directory
            if (html.includes('href="results.json"')) {
                // Found results - the path parts form the trait identifier
                if (pathParts.length >= 2) {
                    traits.push(pathParts.join('/'));
                }
                return; // Don't recurse further once we find results
            }

            // Otherwise, recurse into subdirectories
            const folderMatches = html.matchAll(/href="([^"]+)\/"/g);
            for (const match of folderMatches) {
                const folder = match[1];
                if (folder !== '..' && !folder.startsWith('.')) {
                    await searchDir(`${url}${folder}/`, [...pathParts, folder]);
                }
            }
        } catch (e) {
            // Ignore errors for individual directories
        }
    }

    try {
        await searchDir(baseUrl + '/');
        return traits;
    } catch (e) {
        console.error('Failed to discover steering traits:', e);
        return [];
    }
}


async function renderSweepData(trait) {
    const experiment = window.state.experimentData?.name;
    if (!experiment) return;

    let data = null;
    let steeringMeta = null;

    try {
        const resultsUrl = '/' + window.paths.get('steering.results', { trait });
        const response = await fetch(resultsUrl);
        if (response.ok) {
            const results = await response.json();
            currentRawResults = results;
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
}


/**
 * Render Best Vector per Layer section (multi-trait)
 * Shows one chart per base trait (category/name), with lines for each (method, position) combo
 */
async function renderBestVectorPerLayer() {
    const container = document.getElementById('best-vector-container');
    const traits = discoveredSteeringTraits;

    if (traits.length === 0) {
        container.innerHTML = '<p class="no-data">No steering results found.</p>';
        return;
    }

    container.innerHTML = '<div class="loading">Loading method comparison data...</div>';

    // Get coherence threshold from slider
    const coherenceThresholdEl = document.getElementById('sweep-coherence-threshold');
    const coherenceThreshold = coherenceThresholdEl ? parseInt(coherenceThresholdEl.value) : 70;

    // Group traits by base trait (category/name without position)
    // e.g., "chirp/refusal_v2/response_all" -> base "chirp/refusal_v2", position "response_all"
    const traitGroups = {};
    for (const trait of traits) {
        const parts = trait.split('/');
        let baseTrait, position;
        if (parts.length >= 3) {
            baseTrait = parts.slice(0, 2).join('/');
            position = parts.slice(2).join('/');
        } else {
            baseTrait = trait;
            position = null;  // No position suffix (old format)
        }
        if (!traitGroups[baseTrait]) traitGroups[baseTrait] = [];
        traitGroups[baseTrait].push({ fullPath: trait, position });
    }

    const charts = [];

    for (const [baseTrait, variants] of Object.entries(traitGroups)) {
        const traces = [];
        const methodColors = window.getMethodColors();
        let baseline = null;
        let colorIdx = 0;

        // Load results for each position variant (in parallel)
        const variantResults = await Promise.all(variants.map(async ({ fullPath, position }) => {
            try {
                const resultsUrl = '/' + window.paths.get('steering.results', { trait: fullPath });
                const response = await fetch(resultsUrl);
                if (!response.ok) return null;
                const results = await response.json();
                return { fullPath, position, results };
            } catch (e) {
                return null;
            }
        }));

        // Process results sequentially (baseline depends on first result)
        for (const variantResult of variantResults) {
            if (!variantResult) continue;
            const { position, results } = variantResult;

            if (baseline === null) {
                baseline = results.baseline?.trait_mean || 0;
            }
            const runs = results.runs || [];

            // Group by method and layer, find best trait score per (method, layer)
            const methodData = {};

            runs.forEach(run => {
                const config = run.config || {};
                const result = run.result || {};

                // Support VectorSpec format
                const vectors = config.vectors || [];
                if (vectors.length !== 1) return;

                const v = vectors[0];
                const layer = v.layer;
                const method = v.method;
                const coherence = result.coherence_mean || 0;
                const traitScore = result.trait_mean || 0;

                if (coherence < coherenceThreshold) return;

                if (!methodData[method]) methodData[method] = {};
                if (!methodData[method][layer] || traitScore > methodData[method][layer]) {
                    methodData[method][layer] = traitScore;
                }
            });

            const posDisplayClosed = position ? window.paths.formatPositionDisplay(position) : '';

            Object.entries(methodData).forEach(([method, layerScores]) => {
                const layers = Object.keys(layerScores).map(Number).sort((a, b) => a - b);
                const scores = layers.map(l => layerScores[l]);

                if (layers.length > 0) {
                    const methodName = { probe: 'Probe', gradient: 'Gradient', mean_diff: 'Mean Diff' }[method] || method;
                    const label = posDisplayClosed ? `${methodName} ${posDisplayClosed}` : methodName;

                    const baseColor = methodColors[method] || window.getChartColors()[colorIdx % 10];
                    const dashStyle = variants.length > 1 && position !== 'response_all' && position !== null
                        ? (position.includes('prompt') ? 'dot' : 'dash')
                        : 'solid';

                    traces.push({
                        x: layers,
                        y: scores,
                        type: 'scatter',
                        mode: 'lines+markers',
                        name: label,
                        line: { width: 2, color: baseColor, dash: dashStyle },
                        marker: { size: 4 },
                        hovertemplate: `${label}<br>L%{x}<br>Score: %{y:.1f}<extra></extra>`
                    });
                    colorIdx++;
                }
            });
        }

        // Add baseline trace
        if (traces.length > 0 && baseline !== null) {
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

            const chartId = `best-vector-chart-${baseTrait.replace(/\//g, '-')}`;
            charts.push({ trait: baseTrait, chartId, traces });
        }
    }

    // Render all charts
    if (charts.length === 0) {
        container.innerHTML = '<p class="no-data">No steering data found for selected traits.</p>';
        return;
    }

    container.innerHTML = charts.map(({ trait, chartId }) => `
        <div class="trait-chart-wrapper">
            <div class="trait-chart-title">${window.getDisplayName(trait)}</div>
            <div id="${chartId}" class="chart-container-sm"></div>
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
                        ${window.getDisplayName(t)}
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

        // Support VectorSpec format
        const vectors = config.vectors || [];

        // Only single-vector runs for heatmap (multi-vector would need different viz)
        if (vectors.length !== 1) return;

        const v = vectors[0];
        const layer = v.layer;
        const coef = v.weight;
        const method = v.method;

        // Filter by method if specified
        if (methodFilter && method !== methodFilter) return;

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
        colorscale = window.DELTA_COLORSCALE;
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


// Export
window.renderSteeringSweep = renderSteeringSweep;
