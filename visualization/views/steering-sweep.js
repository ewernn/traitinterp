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
            <div class="sweep-controls sticky-coherence">
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
                    <span class="subsection-title">Layer × Coefficient Heatmap</span>
                </h3>
                <div id="sweep-heatmap-container" class="chart-container-lg"></div>
            </section>

            <!-- Raw results table (collapsible) -->
            <details class="results-details">
                <summary class="results-summary">All Results</summary>
                <div id="sweep-table-container" class="scrollable-container"></div>
            </details>
        </div>
    `;

    // Initial render
    await renderBestVectorPerLayer();
    await renderTraitPicker(traits);

    // Set default selected entry if not set
    if (!window.state.selectedSteeringEntry && traits.length > 0) {
        window.state.selectedSteeringEntry = defaultTrait;
    }

    await renderSweepData(window.state.selectedSteeringEntry || defaultTrait);

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
let currentRawResults = null; // Store raw results.jsonl data for method filtering
let discoveredSteeringTraits = []; // All discovered steering traits
let traitResultsCache = {}; // Cache results per trait for response browser


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
    // Fetch steering entries from API
    // Returns array of objects: { trait, model_variant, position, prompt_set, full_path }
    if (!window.state.experimentData?.name) return [];

    try {
        const response = await fetch(`/api/experiments/${window.state.experimentData.name}/steering`);
        if (!response.ok) return [];
        const data = await response.json();
        return data.entries || [];
    } catch (e) {
        console.error('Failed to discover steering traits:', e);
        return [];
    }
}


async function renderSweepData(steeringEntry) {
    // steeringEntry: { trait, model_variant, position, prompt_set, full_path }
    const experiment = window.state.experimentData?.name;
    if (!experiment || !steeringEntry) return;

    let data = null;
    let steeringMeta = null;

    try {
        const experiment = window.state.experimentData?.name;
        const resultsUrl = `/api/experiments/${experiment}/steering-results/${steeringEntry.trait}/${steeringEntry.model_variant}/${steeringEntry.position}/${steeringEntry.prompt_set}`;
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
    const steeringEntries = discoveredSteeringTraits;

    if (steeringEntries.length === 0) {
        container.innerHTML = '<p class="no-data">No steering results found.</p>';
        return;
    }

    container.innerHTML = '<div class="loading">Loading method comparison data...</div>';

    // Get coherence threshold from slider
    const coherenceThresholdEl = document.getElementById('sweep-coherence-threshold');
    const coherenceThreshold = coherenceThresholdEl ? parseInt(coherenceThresholdEl.value) : 70;

    // Group by base trait (category/trait_name)
    const traitGroups = {};
    for (const entry of steeringEntries) {
        const baseTrait = entry.trait;
        if (!traitGroups[baseTrait]) traitGroups[baseTrait] = [];
        traitGroups[baseTrait].push(entry);
    }

    const charts = [];

    for (const [baseTrait, variants] of Object.entries(traitGroups)) {
        const traces = [];
        const methodColors = window.getMethodColors();
        let baseline = null;
        let colorIdx = 0;
        const allRuns = []; // Collect all runs for response browser

        // Load results for each position variant (in parallel)
        const experiment = window.state.experimentData?.name;
        const variantResults = await Promise.all(variants.map(async (entry) => {
            try {
                const resultsUrl = `/api/experiments/${experiment}/steering-results/${entry.trait}/${entry.model_variant}/${entry.position}/${entry.prompt_set}`;
                const response = await fetch(resultsUrl);
                if (!response.ok) return null;
                const results = await response.json();
                return { entry, results };
            } catch (e) {
                return null;
            }
        }));

        // Process results sequentially (baseline depends on first result)
        for (const variantResult of variantResults) {
            if (!variantResult) continue;
            const { entry, results } = variantResult;
            const position = entry.position;

            if (baseline === null) {
                baseline = results.baseline?.trait_mean || 0;
            }
            const runs = results.runs || [];

            // Group by (component, method) and layer, find best trait score per combo
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
                const component = v.component || 'residual';
                const coef = v.weight;
                const coherence = result.coherence_mean || 0;
                const traitScore = result.trait_mean || 0;

                // Collect for response browser (before coherence filter)
                allRuns.push({
                    layer, method, component, coef, traitScore, coherence,
                    timestamp: run.timestamp,
                    entry, // steering entry for path building
                });

                if (coherence < coherenceThreshold) return;

                // Key includes component for differentiation
                const key = component === 'residual' ? method : `${component}/${method}`;
                if (!methodData[key]) methodData[key] = {};
                if (!methodData[key][layer] || traitScore > methodData[key][layer].score) {
                    methodData[key][layer] = { score: traitScore, coef };
                }
            });

            const posDisplayClosed = position ? window.paths.formatPositionDisplay(position) : '';

            Object.entries(methodData).forEach(([methodKey, layerData]) => {
                const layers = Object.keys(layerData).map(Number).sort((a, b) => a - b);
                const scores = layers.map(l => layerData[l].score);
                const coefs = layers.map(l => layerData[l].coef);

                if (layers.length > 0) {
                    // Parse component/method key
                    const [component, method] = methodKey.includes('/')
                        ? methodKey.split('/')
                        : ['residual', methodKey];

                    const methodName = { probe: 'Probe', gradient: 'Gradient', mean_diff: 'Mean Diff' }[method] || method;
                    const componentPrefix = component !== 'residual' ? `${component} ` : '';
                    const label = posDisplayClosed
                        ? `${componentPrefix}${methodName} ${posDisplayClosed}`
                        : `${componentPrefix}${methodName}`;

                    const baseColor = methodColors[method] || window.getChartColors()[colorIdx % 10];
                    // Dash for non-residual components
                    const dashStyle = component !== 'residual' ? 'dot'
                        : (variants.length > 1 && position !== 'response_all' && position !== null
                            ? (position.includes('prompt') ? 'dot' : 'dash')
                            : 'solid');

                    // Build custom hover text with coefficient
                    const hoverTexts = layers.map((l, i) =>
                        `${label}<br>L${l} c${coefs[i].toFixed(1)}<br>Score: ${scores[i].toFixed(1)}`
                    );

                    traces.push({
                        x: layers,
                        y: scores,
                        type: 'scatter',
                        mode: 'lines+markers',
                        name: label,
                        line: { width: 2, color: baseColor, dash: dashStyle },
                        marker: { size: 4 },
                        text: hoverTexts,
                        hovertemplate: '%{text}<extra></extra>'
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
            const browserId = `response-browser-${baseTrait.replace(/\//g, '-')}`;

            // Cache all runs for this trait
            traitResultsCache[baseTrait] = { allRuns, baseline };

            charts.push({ trait: baseTrait, chartId, browserId, traces, runCount: allRuns.length });
        }
    }

    // Render all charts
    if (charts.length === 0) {
        container.innerHTML = '<p class="no-data">No steering data found for selected traits.</p>';
        return;
    }

    container.innerHTML = charts.map(({ trait, chartId, browserId, runCount }) => `
        <div class="trait-chart-wrapper">
            <div class="trait-chart-title">${window.getDisplayName(trait)}</div>
            <div id="${chartId}" class="chart-container-sm"></div>
            <details class="response-browser-details" data-trait="${trait}">
                <summary class="response-browser-summary">
                    ▶ View Responses <span class="hint">(${runCount} runs)</span>
                </summary>
                <div id="${browserId}" class="response-browser-content"></div>
            </details>
        </div>
    `).join('');

    // Setup response browser toggle handlers
    container.querySelectorAll('.response-browser-details').forEach(details => {
        details.addEventListener('toggle', (e) => {
            if (details.open) {
                const trait = details.dataset.trait;
                renderResponseBrowserForTrait(trait);
            }
        });
    });

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
 * @param {Array} steeringEntries - Array of { trait, model_variant, position, prompt_set, full_path }
 */
async function renderTraitPicker(steeringEntries) {
    const container = document.getElementById('trait-picker-container');

    if (!steeringEntries || steeringEntries.length === 0) {
        container.innerHTML = '';
        return;
    }

    // Get current selection or default to first
    const currentFullPath = window.state.selectedSteeringEntry?.full_path || steeringEntries[0].full_path;

    container.innerHTML = `
        <div class="trait-picker">
            <span class="tp-label">Trait:</span>
            <div class="tp-buttons">
                ${steeringEntries.map((entry, idx) => `
                    <button class="tp-btn ${entry.full_path === currentFullPath ? 'active' : ''}" data-index="${idx}">
                        ${window.getDisplayName(entry.trait)}
                        <span class="hint">${window.paths.formatPositionDisplay(entry.position)}</span>
                    </button>
                `).join('')}
            </div>
        </div>
    `;

    // Setup event listeners
    container.querySelectorAll('.tp-btn').forEach(btn => {
        btn.addEventListener('click', async () => {
            const idx = parseInt(btn.dataset.index);
            const selectedEntry = steeringEntries[idx];
            window.state.selectedSteeringEntry = selectedEntry;

            // Update active state
            container.querySelectorAll('.tp-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            // Re-render single-trait sections
            await renderSweepData(selectedEntry);
        });
    });
}


function convertResultsToSweepFormat(results, methodFilter = null) {
    // Convert results.jsonl format to sweep visualization format
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
        _isConverted: true // Flag to indicate this was converted from results.jsonl (uses raw coefficients, not ratios)
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

    // Bin coefficients if there are too many (>50) for clean visualization
    // Use logarithmic binning to handle wide coefficient ranges
    const MAX_BINS = 40;
    let binEdges = null;
    let binCenters = null;
    if (ratios.length > MAX_BINS) {
        const minR = Math.min(...ratios);
        const maxR = Math.max(...ratios);
        // Use log scale for binning to handle exponential coefficient ranges
        const logMin = Math.log(minR + 1);
        const logMax = Math.log(maxR + 1);
        binEdges = [];
        binCenters = [];
        for (let i = 0; i <= MAX_BINS; i++) {
            const logVal = logMin + (logMax - logMin) * i / MAX_BINS;
            binEdges.push(Math.exp(logVal) - 1);
        }
        // Bin centers are midpoints (geometric mean for log scale)
        for (let i = 0; i < MAX_BINS; i++) {
            binCenters.push((binEdges[i] + binEdges[i + 1]) / 2);
        }
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

        // If binning, aggregate values per bin (use max absolute value to show strongest effect)
        if (binEdges && !interpolate) {
            return binCenters.map((_, binIdx) => {
                const binMin = binEdges[binIdx];
                const binMax = binEdges[binIdx + 1];
                const binPoints = validPoints.filter(p => p.r >= binMin && p.r < binMax);
                if (binPoints.length === 0) return null;
                // For delta metric, use value with max absolute value; otherwise use max
                if (metric === 'delta') {
                    return binPoints.reduce((best, p) => Math.abs(p.v) > Math.abs(best) ? p.v : best, 0);
                }
                return Math.max(...binPoints.map(p => p.v));
            });
        }

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

    // Use interpolated ratios for x-axis if interpolating, or bin centers if binning
    const xRatios = interpolate ? interpolatedRatios : (binCenters || ratios);

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

    // Use uniform spacing (indices as strings) for x-axis to ensure equal cell widths
    // String indices force Plotly to treat as categorical (truly uniform spacing)
    // Actual coefficient values shown via tick labels and hover text
    const xIndices = xRatios.map((_, i) => String(i));

    // Build custom hover text with original coefficient values
    const hoverText = matrix.map((row, layerIdx) =>
        row.map((val, ratioIdx) => {
            if (val === null) return '';
            const metricLabel = metric === 'delta' ? 'Delta' : metric === 'coherence' ? 'Coherence' : 'Trait';
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
        connectgaps: interpolate, // Smooth rendering when interpolating
        hovertemplate: '%{text}<extra></extra>',
        text: hoverText,
        colorbar: {
            title: { text: metric === 'delta' ? 'Delta' : metric === 'coherence' ? 'Coherence' : 'Trait', font: { size: 11 } }
        }
    };

    // Label depends on whether we're using sweep_results (ratios) or converted results (coefficients)
    const xAxisLabel = currentSweepData._isConverted ? 'Coefficient' : 'Perturbation Ratio';

    // Generate evenly-spaced tick positions showing actual coefficient values
    const numTicks = Math.min(10, xRatios.length);
    const tickIndices = [];
    const tickLabels = [];
    for (let i = 0; i < numTicks; i++) {
        const idx = Math.round(i * (xRatios.length - 1) / (numTicks - 1));
        tickIndices.push(String(idx));  // String to match categorical x-axis
        tickLabels.push(xRatios[idx].toFixed(0));
    }

    const xAxisConfig = {
        title: xAxisLabel,
        tickfont: { size: 10 },
        tickvals: tickIndices,
        ticktext: tickLabels,
        type: 'category'  // Force categorical axis
    };

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


function escapeHtml(text) {
    if (!text) return '';
    return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/\n/g, '<br>');
}


// ============================================================
// Response Browser (per-trait collapsible)
// ============================================================

// Track current sort state per trait
const responseBrowserState = {};

// Cache for trait definitions and judge templates
const traitDefinitionCache = {};
let judgeTemplatesCache = null;

/**
 * Render the response browser table for a trait
 */
function renderResponseBrowserForTrait(trait) {
    const browserId = `response-browser-${trait.replace(/\//g, '-')}`;
    const container = document.getElementById(browserId);
    if (!container) return;

    const cached = traitResultsCache[trait];
    if (!cached || !cached.allRuns.length) {
        container.innerHTML = '<p class="no-data">No response data available</p>';
        return;
    }

    // Initialize state for this trait
    if (!responseBrowserState[trait]) {
        responseBrowserState[trait] = {
            sortKey: 'traitScore',
            sortDir: 'desc',
            layerFilter: new Set(), // empty = show all
            expandedRow: null,
            bestPerLayer: true, // Show only best run per layer (default on)
            infoPanel: null, // 'definition' | 'judge' | null
            compactResponses: true, // Show newlines as \n (default on)
        };
    }
    const state = responseBrowserState[trait];

    // Get coherence threshold from the page slider
    const coherenceThresholdEl = document.getElementById('sweep-coherence-threshold');
    const coherenceThreshold = coherenceThresholdEl ? parseInt(coherenceThresholdEl.value) : 70;

    // Get unique layers for filter
    const uniqueLayers = [...new Set(cached.allRuns.map(r => r.layer))].sort((a, b) => a - b);

    // Filter and sort runs
    let runs = [...cached.allRuns];
    if (state.layerFilter.size > 0) {
        runs = runs.filter(r => state.layerFilter.has(r.layer));
    }

    // Best per layer filter: keep only highest trait score per layer (with coherence >= threshold)
    if (state.bestPerLayer) {
        const bestByLayer = {};
        for (const run of runs) {
            if (run.coherence < coherenceThreshold) continue;
            if (!bestByLayer[run.layer] || run.traitScore > bestByLayer[run.layer].traitScore) {
                bestByLayer[run.layer] = run;
            }
        }
        runs = Object.values(bestByLayer);
    }

    // Sort
    runs.sort((a, b) => {
        const aVal = a[state.sortKey] ?? 0;
        const bVal = b[state.sortKey] ?? 0;
        return state.sortDir === 'desc' ? bVal - aVal : aVal - bVal;
    });

    // Get unique positions for display
    const uniquePositions = [...new Set(cached.allRuns.map(r => r.entry?.position || 'unknown'))];
    const showPositionCol = uniquePositions.length > 1 || uniquePositions[0] !== 'response_all';

    // Build HTML
    container.innerHTML = `
        <div class="rb-filters">
            <span class="rb-filter-label">Layers:</span>
            <div class="rb-layer-chips">
                <button class="rb-chip-btn" data-action="select-all">All</button>
                <button class="rb-chip-btn" data-action="select-none">None</button>
                ${uniqueLayers.map(l => `
                    <label class="rb-chip ${state.layerFilter.size === 0 || state.layerFilter.has(l) ? 'active' : ''}">
                        <input type="checkbox" value="${l}" ${state.layerFilter.size === 0 || state.layerFilter.has(l) ? 'checked' : ''}>
                        L${l}
                    </label>
                `).join('')}
            </div>
            <div class="rb-info-btns">
                <button class="rb-info-btn ${state.infoPanel === 'definition' ? 'active' : ''}" data-info="definition">Definition</button>
                <button class="rb-info-btn ${state.infoPanel === 'judge' ? 'active' : ''}" data-info="judge">Judge Prompt</button>
            </div>
            <label class="rb-toggle">
                <input type="checkbox" ${state.bestPerLayer ? 'checked' : ''} data-action="best-per-layer">
                Best per layer (coh ≥${coherenceThreshold})
            </label>
            <label class="rb-toggle">
                <input type="checkbox" ${state.compactResponses ? 'checked' : ''} data-action="compact-responses">
                Compact responses
            </label>
        </div>
        ${state.infoPanel ? `
        <div class="rb-info-panel" data-panel="${state.infoPanel}">
            <div class="rb-info-content">Loading...</div>
        </div>
        ` : ''}
        <div class="rb-table-wrapper">
            <table class="data-table rb-table">
                <thead>
                    <tr>
                        <th class="sortable ${state.sortKey === 'layer' ? 'sort-active' : ''}" data-sort="layer">
                            Layer <span class="sort-indicator">${state.sortKey === 'layer' ? (state.sortDir === 'desc' ? '▼' : '▲') : '▼'}</span>
                        </th>
                        <th class="sortable ${state.sortKey === 'coef' ? 'sort-active' : ''}" data-sort="coef">
                            Coef <span class="sort-indicator">${state.sortKey === 'coef' ? (state.sortDir === 'desc' ? '▼' : '▲') : '▼'}</span>
                        </th>
                        <th>Method</th>
                        <th>Component</th>
                        ${showPositionCol ? '<th>Position</th>' : ''}
                        <th class="sortable ${state.sortKey === 'traitScore' ? 'sort-active' : ''}" data-sort="traitScore">
                            Trait <span class="sort-indicator">${state.sortKey === 'traitScore' ? (state.sortDir === 'desc' ? '▼' : '▲') : '▼'}</span>
                        </th>
                        <th class="sortable ${state.sortKey === 'coherence' ? 'sort-active' : ''}" data-sort="coherence">
                            Coh <span class="sort-indicator">${state.sortKey === 'coherence' ? (state.sortDir === 'desc' ? '▼' : '▲') : '▼'}</span>
                        </th>
                    </tr>
                </thead>
                <tbody>
                    ${runs.map((run, idx) => {
                        const position = run.entry?.position || 'unknown';
                        const posDisplay = window.paths?.formatPositionDisplay ? window.paths.formatPositionDisplay(position) : position;
                        return `
                        <tr class="rb-row ${state.expandedRow === idx ? 'expanded' : ''} ${run.coherence < coherenceThreshold ? 'below-threshold' : ''}" data-idx="${idx}">
                            <td>L${run.layer}</td>
                            <td>${run.coef.toFixed(1)}</td>
                            <td>${run.method}</td>
                            <td>${run.component}</td>
                            ${showPositionCol ? `<td class="rb-position">${posDisplay}</td>` : ''}
                            <td class="${run.traitScore > 50 ? 'quality-good' : run.traitScore > 20 ? 'quality-ok' : ''}">${run.traitScore.toFixed(1)}</td>
                            <td class="${run.coherence >= 80 ? 'quality-good' : run.coherence >= 60 ? 'quality-ok' : 'quality-bad'}">${run.coherence.toFixed(0)}</td>
                        </tr>
                        ${state.expandedRow === idx ? `
                        <tr class="rb-expanded-row">
                            <td colspan="${showPositionCol ? 7 : 6}">
                                <div class="rb-responses-container" id="rb-responses-${trait.replace(/\//g, '-')}-${idx}">
                                    <div class="loading">Loading responses...</div>
                                </div>
                            </td>
                        </tr>
                        ` : ''}
                    `;}).join('')}
                </tbody>
            </table>
        </div>
        <div class="rb-stats hint">${runs.length} runs shown${state.bestPerLayer ? ' (best per layer)' : ''}</div>
    `;

    // Setup event handlers
    setupResponseBrowserHandlers(trait, container, runs);

    // Load responses if a row is expanded
    if (state.expandedRow !== null && runs[state.expandedRow]) {
        loadResponsesForRun(trait, state.expandedRow, runs[state.expandedRow]);
    }

    // Load info panel content if open
    if (state.infoPanel) {
        loadInfoPanelContent(trait, state.infoPanel);
    }
}


/**
 * Setup event handlers for response browser
 */
function setupResponseBrowserHandlers(trait, container, runs) {
    const state = responseBrowserState[trait];
    const allLayers = [...new Set(traitResultsCache[trait].allRuns.map(r => r.layer))];

    // Select All / Select None buttons
    container.querySelectorAll('.rb-chip-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const action = btn.dataset.action;
            if (action === 'select-all') {
                state.layerFilter.clear(); // Empty = show all
            } else if (action === 'select-none') {
                state.layerFilter.clear();
                state.layerFilter.add(-999); // Impossible layer = show none
            }
            state.expandedRow = null;
            renderResponseBrowserForTrait(trait);
        });
    });

    // Best per layer toggle
    const bestPerLayerCheckbox = container.querySelector('input[data-action="best-per-layer"]');
    if (bestPerLayerCheckbox) {
        bestPerLayerCheckbox.addEventListener('change', () => {
            state.bestPerLayer = bestPerLayerCheckbox.checked;
            state.expandedRow = null;
            renderResponseBrowserForTrait(trait);
        });
    }

    // Compact responses toggle
    const compactCheckbox = container.querySelector('.rb-filters input[data-action="compact-responses"]');
    if (compactCheckbox) {
        compactCheckbox.addEventListener('change', () => {
            state.compactResponses = compactCheckbox.checked;
            renderResponseBrowserForTrait(trait);
        });
    }

    // Layer filter checkboxes
    container.querySelectorAll('.rb-chip input').forEach(checkbox => {
        checkbox.addEventListener('change', () => {
            const layer = parseInt(checkbox.value);
            if (checkbox.checked) {
                // If all were selected (filter empty), start fresh with just this one
                if (state.layerFilter.size === 0) {
                    allLayers.forEach(l => state.layerFilter.add(l));
                }
                state.layerFilter.add(layer);
                // Remove impossible layer if it was set
                state.layerFilter.delete(-999);
            } else {
                if (state.layerFilter.size === 0) {
                    // First uncheck - add all except this one
                    allLayers.forEach(l => { if (l !== layer) state.layerFilter.add(l); });
                } else {
                    state.layerFilter.delete(layer);
                }
            }
            state.expandedRow = null; // Close expanded row on filter change
            renderResponseBrowserForTrait(trait);
        });
    });

    // Sortable headers
    container.querySelectorAll('th.sortable').forEach(th => {
        th.addEventListener('click', () => {
            const sortKey = th.dataset.sort;
            if (state.sortKey === sortKey) {
                state.sortDir = state.sortDir === 'desc' ? 'asc' : 'desc';
            } else {
                state.sortKey = sortKey;
                state.sortDir = 'desc';
            }
            state.expandedRow = null;
            renderResponseBrowserForTrait(trait);
        });
    });

    // Row click to expand
    container.querySelectorAll('.rb-row').forEach(row => {
        row.addEventListener('click', () => {
            const idx = parseInt(row.dataset.idx);
            if (state.expandedRow === idx) {
                state.expandedRow = null;
            } else {
                state.expandedRow = idx;
            }
            renderResponseBrowserForTrait(trait);
        });
    });

    // Info panel buttons (Definition / Judge Prompt)
    container.querySelectorAll('.rb-info-btn').forEach(btn => {
        btn.addEventListener('click', async () => {
            const infoType = btn.dataset.info;
            if (state.infoPanel === infoType) {
                state.infoPanel = null; // Toggle off
            } else {
                state.infoPanel = infoType;
            }
            renderResponseBrowserForTrait(trait);

            // Load content after render if panel is open
            if (state.infoPanel) {
                await loadInfoPanelContent(trait, state.infoPanel);
            }
        });
    });
}


/**
 * Load and display info panel content (definition or judge prompt)
 */
async function loadInfoPanelContent(trait, panelType) {
    const browserId = `response-browser-${trait.replace(/\//g, '-')}`;
    const container = document.getElementById(browserId);
    const panel = container?.querySelector('.rb-info-content');
    if (!panel) return;

    // Extract trait name for display (last part of path)
    const traitName = trait.split('/').pop();

    try {
        // Fetch definition if not cached
        if (!traitDefinitionCache[trait]) {
            const defPath = `datasets/traits/${trait}/definition.txt`;
            const response = await fetch(`/${defPath}`);
            if (!response.ok) {
                traitDefinitionCache[trait] = { error: 'Definition file not found' };
            } else {
                traitDefinitionCache[trait] = { text: await response.text() };
            }
        }

        const cached = traitDefinitionCache[trait];

        if (cached.error) {
            panel.innerHTML = `<p class="no-data">${cached.error}</p>`;
            return;
        }

        if (panelType === 'definition') {
            panel.innerHTML = `<pre class="rb-info-text">${escapeHtml(cached.text.trim())}</pre>`;
        } else if (panelType === 'judge') {
            // Fetch judge templates if not cached
            if (!judgeTemplatesCache) {
                const resp = await fetch('/api/judge-templates');
                if (!resp.ok) {
                    panel.innerHTML = `<p class="no-data">Could not load judge templates</p>`;
                    return;
                }
                judgeTemplatesCache = await resp.json();
            }

            const systemPrompt = judgeTemplatesCache.steering_system
                .replace('{trait_name}', traitName)
                .replace('{trait_definition}', cached.text.trim());

            panel.innerHTML = `
                <div class="rb-judge-section">
                    <div class="rb-judge-label">System Message:</div>
                    <pre class="rb-info-text">${escapeHtml(systemPrompt)}</pre>
                </div>
                <div class="rb-judge-section">
                    <div class="rb-judge-label">User Message:</div>
                    <pre class="rb-info-text">${escapeHtml(judgeTemplatesCache.steering_user)}</pre>
                </div>
            `;
        }

    } catch (e) {
        console.error('Failed to load info panel:', e);
        panel.innerHTML = `<p class="no-data">Error: ${e.message}</p>`;
    }
}


/**
 * Load and display responses for a specific run
 */
async function loadResponsesForRun(trait, idx, run) {
    const containerId = `rb-responses-${trait.replace(/\//g, '-')}-${idx}`;
    const container = document.getElementById(containerId);
    if (!container) return;

    const { entry } = run;
    const experiment = window.state.experimentData?.name;

    try {
        // Build path to response file
        const ts = run.timestamp ? run.timestamp.slice(0, 19).replace(/:/g, '-').replace('T', '_') : '';
        const filename = `L${run.layer}_c${run.coef.toFixed(1)}_${ts}.json`;
        const responsePath = window.paths.get('steering.responses', {
            experiment,
            trait: entry.trait,
            model_variant: entry.model_variant,
            position: entry.position,
            prompt_set: entry.prompt_set,
        });

        const url = `/${responsePath}/${run.component}/${run.method}/${filename}`;
        const response = await fetch(url);

        if (!response.ok) {
            container.innerHTML = `<p class="no-data">Response file not found</p>`;
            return;
        }

        const responses = await response.json();
        const state = responseBrowserState[trait];
        const isCompact = state?.compactResponses ?? true;

        container.innerHTML = `
            <div class="response-list-compact">
                ${responses.map((r, i) => {
                    // In compact mode, show \n as literal text; otherwise preserve whitespace
                    const responseText = isCompact
                        ? r.response.replace(/\n/g, '\\n')
                        : r.response;
                    return `
                    <div class="response-item-row">
                        <div class="response-meta">
                            <div class="meta-label">Prompt #${i + 1}</div>
                            <div class="meta-score">Trait: <span class="${r.trait_score > 50 ? 'quality-good' : r.trait_score > 20 ? 'quality-ok' : ''}">${r.trait_score?.toFixed(0) ?? '-'}</span></div>
                            <div class="meta-score">Coh: <span class="${(r.coherence_score ?? 0) >= 80 ? 'quality-good' : (r.coherence_score ?? 0) >= 60 ? 'quality-ok' : 'quality-bad'}">${r.coherence_score?.toFixed(0) ?? '-'}</span></div>
                        </div>
                        <div class="response-content">
                            <div class="response-q">${escapeHtml(r.question)}</div>
                            <div class="response-a ${isCompact ? 'compact' : ''}">${escapeHtml(responseText)}</div>
                        </div>
                    </div>
                `;}).join('')}
            </div>
        `;

    } catch (e) {
        console.error('Failed to load responses:', e);
        container.innerHTML = `<p class="no-data">Error loading responses: ${e.message}</p>`;
    }
}


// Export
window.renderSteeringSweep = renderSteeringSweep;
