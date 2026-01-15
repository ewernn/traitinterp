// Steering Sweep - Heatmap visualization of steering experiments
// Shows layer × perturbation ratio → delta/coherence
//
// Sections:
// 1. Best Vector per Layer: Multi-trait comparison by extraction method
// 2. Layer × Coefficient Heatmaps: Delta and coherence heatmaps
//
// Response browser component is in: components/response-browser.js

async function renderSteeringSweep() {
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

    // Show loading state only if fetch takes > 150ms
    const loadingTimeout = setTimeout(() => {
        contentArea.innerHTML = ui.renderLoading('Loading steering sweep data...');
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
                ${ui.renderSubsection({
                    num: 1,
                    title: 'Best Vector per Layer',
                    infoId: 'info-best-vector',
                    infoText: 'For each selected trait (from sidebar), shows the best trait score achieved per layer across all 3 extraction methods (probe, gradient, mean_diff). Each trait gets its own chart showing which method works best at which layer. Dashed line shows baseline (no steering).'
                })}
                <div id="best-vector-container"></div>
            </section>

            <!-- Heatmaps section -->
            <section>
                <h3 class="subsection-header" id="heatmap-section">
                    <span class="subsection-num">2.</span>
                    <span class="subsection-title">Layer × Coefficient Heatmaps</span>
                </h3>

                <!-- Controls for heatmaps -->
                <div class="sweep-controls">
                    <div class="control-group">
                        <label>Trait:</label>
                        <select id="sweep-trait-select"></select>
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
                        <label>
                            <input type="checkbox" id="sweep-interpolate" />
                            Interpolate
                        </label>
                    </div>
                </div>

                <!-- Dual heatmaps: Delta (filtered) and Coherence (unfiltered) -->
                <div class="dual-heatmap-container">
                    <div class="heatmap-panel">
                        <div class="heatmap-label">Trait Delta <span class="hint">(coherence ≥ threshold)</span></div>
                        <div id="sweep-heatmap-delta" class="chart-container-md"></div>
                    </div>
                    <div class="heatmap-panel">
                        <div class="heatmap-label">Coherence <span class="hint">(all results)</span></div>
                        <div id="sweep-heatmap-coherence" class="chart-container-md"></div>
                    </div>
                </div>
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
    document.getElementById('sweep-method').addEventListener('change', () => updateSweepVisualizations());

    document.getElementById('sweep-coherence-threshold').addEventListener('input', async (e) => {
        document.getElementById('coherence-threshold-value').textContent = e.target.value;
        await renderBestVectorPerLayer();
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
let localTraitResultsCache = {}; // Local cache, passed to response-browser via setTraitResultsCache()


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
 * Extract trait name from full trait path.
 * e.g., "pv_instruction/evil" → "evil"
 *       "pv_natural/evil_v3" → "evil_v3"
 */
function getBaseTraitName(trait) {
    // Just return the trait name part, keeping version suffixes
    return trait.split('/').pop();
}

/**
 * Get elicitation method label from category.
 * e.g., "pv_instruction/evil" → "instruction"
 *       "pv_natural/evil_v3" → "natural"
 */
function getElicitationLabel(trait) {
    const category = trait.split('/')[0];
    if (category.includes('instruction')) return 'instruction';
    if (category.includes('natural')) return 'natural';
    return category;  // fallback to full category name
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

    // Only show loading on initial render, not on threshold updates (avoids flash)
    const hasExistingCharts = container.querySelector('.trait-chart-wrapper');
    if (!hasExistingCharts) {
        container.innerHTML = ui.renderLoading('Loading method comparison data...');
    }

    // Get coherence threshold from slider
    const coherenceThresholdEl = document.getElementById('sweep-coherence-threshold');
    const coherenceThreshold = coherenceThresholdEl ? parseInt(coherenceThresholdEl.value) : 70;

    // Group by base trait name (e.g., "evil" groups pv_instruction/evil and pv_natural/evil_v3)
    const traitGroups = {};
    for (const entry of steeringEntries) {
        const baseTrait = getBaseTraitName(entry.trait);
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
            const elicitLabel = getElicitationLabel(entry.trait);  // "instruction" or "natural"
            const fullTraitName = entry.trait.split('/').pop();  // "evil" or "evil_v3"

            // Check if this group has multiple elicitation methods
            const uniqueElicitMethods = new Set(variants.map(v => getElicitationLabel(v.trait)));
            const hasMultipleElicit = uniqueElicitMethods.size > 1;

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

                    // Add elicitation prefix when comparing instruction vs natural
                    const elicitPrefix = hasMultipleElicit ? `${elicitLabel} ` : '';

                    const label = posDisplayClosed
                        ? `${elicitPrefix}${componentPrefix}${methodName} ${posDisplayClosed}`
                        : `${elicitPrefix}${componentPrefix}${methodName}`;

                    const baseColor = methodColors[method] || window.getChartColors()[colorIdx % 10];
                    // Different dash styles for elicitation methods
                    const dashStyle = component !== 'residual' ? 'dot'
                        : (hasMultipleElicit ? (elicitLabel === 'instruction' ? 'solid' : 'dash') : 'solid');

                    // Build custom hover text with coefficient and full trait path
                    const hoverTexts = layers.map((l, i) =>
                        `${label}<br>L${l} c${coefs[i].toFixed(1)}<br>Score: ${scores[i].toFixed(1)}<br><span style="opacity:0.7">${fullTraitName}</span>`
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
            localTraitResultsCache[baseTrait] = { allRuns, baseline };

            charts.push({ trait: baseTrait, chartId, browserId, traces, runCount: allRuns.length });
        }
    }

    // Set the cache reference for the response browser component
    window.responseBrowser.setTraitResultsCache(localTraitResultsCache);

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
                    ▶ View Responses <span class="hint response-count-hint" data-trait="${trait}">(checking...)</span>
                </summary>
                <div id="${browserId}" class="response-browser-content"></div>
            </details>
        </div>
    `).join('');

    // Setup response browser toggle handlers
    container.querySelectorAll('.response-browser-details').forEach(details => {
        details.addEventListener('toggle', async () => {
            if (details.open) {
                const trait = details.dataset.trait;
                await window.renderResponseBrowserForTrait(trait);
            }
        });
    });

    // Plot each chart
    for (const { chartId, traces } of charts) {
        const layout = window.buildChartLayout({
            preset: 'layerChart',
            traces,
            height: 180,
            legendPosition: 'below',
            yaxis: { title: 'Trait Score' },
            margin: { r: 60 },  // Extra right margin for inline label
            // Inline x-axis label at end of axis
            annotations: [{
                x: 1.01,
                y: 0,
                xref: 'paper',
                yref: 'paper',
                text: 'Layer →',
                showarrow: false,
                font: { size: 10, color: '#888' },
                xanchor: 'left',
                yanchor: 'middle'
            }]
        });

        window.renderChart(chartId, traces, layout);
    }

    // Background fetch: get response counts for all traits and update summaries
    fetchResponseCountsInBackground(charts.map(c => c.trait));
}

/**
 * Fetch response availability for traits in background and update summary counts
 */
async function fetchResponseCountsInBackground(traits) {
    for (const trait of traits) {
        const cached = localTraitResultsCache[trait];
        if (!cached || !cached.allRuns.length) continue;

        // Skip if already fetched
        if (cached.availableResponses) {
            updateResponseCountHint(trait, cached);
            continue;
        }

        // Fetch in background (don't await all at once to avoid hammering server)
        // Use the component's fetchAvailableResponses function
        window.fetchAvailableResponses(cached.allRuns).then(result => {
            cached.availableResponses = result.responses;
            cached.availableBaselines = result.baselines;
            updateResponseCountHint(trait, cached);
        }).catch(err => {
            console.error('Failed to fetch response counts for', trait, err);
            // Show fallback count
            const hint = document.querySelector(`.response-count-hint[data-trait="${trait}"]`);
            if (hint) hint.textContent = `(${cached.allRuns.length} runs)`;
        });
    }
}

/**
 * Update the response count hint in the summary for a trait
 */
function updateResponseCountHint(trait, cached) {
    const hint = document.querySelector(`.response-count-hint[data-trait="${trait}"]`);
    if (!hint) return;

    const runsWithResponses = cached.allRuns.filter(run => {
        const key = `${run.entry?.trait}|${run.entry?.model_variant}|${run.entry?.position}|${run.entry?.prompt_set}|${run.component}|${run.method}|${run.layer}|${run.coef.toFixed(1)}`;
        return cached.availableResponses.has(key);
    });

    if (runsWithResponses.length === 0) {
        hint.textContent = '(no responses saved)';
    } else if (runsWithResponses.length === cached.allRuns.length) {
        hint.textContent = `(${runsWithResponses.length} runs)`;
    } else {
        hint.textContent = `(${runsWithResponses.length} with responses)`;
    }
}

/**
 * Populate trait dropdown for heatmap section
 * @param {Array} steeringEntries - Array of { trait, model_variant, position, prompt_set, full_path }
 */
async function renderTraitPicker(steeringEntries) {
    const select = document.getElementById('sweep-trait-select');
    if (!select) return;

    if (!steeringEntries || steeringEntries.length === 0) {
        select.innerHTML = '<option>No traits</option>';
        return;
    }

    // Get current selection or default to first
    const currentFullPath = window.state.selectedSteeringEntry?.full_path || steeringEntries[0].full_path;

    // Build options with readable labels
    select.innerHTML = steeringEntries.map((entry, idx) => {
        const displayName = window.getDisplayName(entry.trait);
        const posDisplay = window.paths.formatPositionDisplay(entry.position);
        // Include prompt_set if not default "steering"
        const promptSetDisplay = entry.prompt_set && entry.prompt_set !== 'steering'
            ? ` [${entry.prompt_set}]`
            : '';
        const label = `${displayName} ${posDisplay}${promptSetDisplay}`;
        const selected = entry.full_path === currentFullPath ? 'selected' : '';
        return `<option value="${idx}" ${selected}>${label}</option>`;
    }).join('');

    // Setup change handler
    select.addEventListener('change', async () => {
        const idx = parseInt(select.value);
        const selectedEntry = steeringEntries[idx];
        window.state.selectedSteeringEntry = selectedEntry;
        await renderSweepData(selectedEntry);
    });
}


function convertResultsToSweepFormat(results, methodFilter = null) {
    // Convert results.jsonl format to sweep visualization format
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

        // Only single-vector runs for heatmap
        if (vectors.length !== 1) return;

        const v = vectors[0];
        const layer = v.layer;
        const coef = v.weight;
        const method = v.method;

        // Filter by method if specified
        if (methodFilter && method !== methodFilter) return;

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
            // Update if this run is newer
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
        full_vector: fullVector
    };
}


function updateSweepVisualizations() {
    if (!currentSweepData) return;

    const method = document.getElementById('sweep-method').value;
    const coherenceThreshold = parseInt(document.getElementById('sweep-coherence-threshold').value);
    const interpolate = document.getElementById('sweep-interpolate').checked;

    // If method filter is active and we have raw results, reconvert with filter
    let data;
    if (method !== 'all' && currentRawResults) {
        const filteredData = convertResultsToSweepFormat(currentRawResults, method);
        data = filteredData?.full_vector || {};
    } else {
        data = currentSweepData.full_vector || {};
    }

    // Render dual heatmaps: Delta (filtered) and Coherence (unfiltered)
    renderSweepHeatmap(data, 'delta', coherenceThreshold, interpolate, 'sweep-heatmap-delta');
    renderSweepHeatmap(data, 'coherence', 0, interpolate, 'sweep-heatmap-coherence');
    renderSweepTable(data, coherenceThreshold);
}


function renderSweepHeatmap(data, metric, coherenceThreshold, interpolate = false, containerId = 'sweep-heatmap-delta') {
    const container = document.getElementById(containerId);

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
    const MAX_BINS = 40;
    let binEdges = null;
    let binCenters = null;
    if (ratios.length > MAX_BINS) {
        const minR = Math.min(...ratios);
        const maxR = Math.max(...ratios);
        // Use log scale for binning
        const logMin = Math.log(minR + 1);
        const logMax = Math.log(maxR + 1);
        binEdges = [];
        binCenters = [];
        for (let i = 0; i <= MAX_BINS; i++) {
            const logVal = logMin + (logMax - logMin) * i / MAX_BINS;
            binEdges.push(Math.exp(logVal) - 1);
        }
        for (let i = 0; i < MAX_BINS; i++) {
            binCenters.push((binEdges[i] + binEdges[i + 1]) / 2);
        }
    }

    // If interpolating, create a denser grid
    let interpolatedRatios = ratios;
    if (interpolate && ratios.length > 1) {
        const minR = Math.min(...ratios);
        const maxR = Math.max(...ratios);
        const numSteps = 50;
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

        // If binning, aggregate values per bin
        if (binEdges && !interpolate) {
            return binCenters.map((_, binIdx) => {
                const binMin = binEdges[binIdx];
                const binMax = binEdges[binIdx + 1];
                const binPoints = validPoints.filter(p => p.r >= binMin && p.r < binMax);
                if (binPoints.length === 0) return null;
                if (metric === 'delta') {
                    return binPoints.reduce((best, p) => Math.abs(p.v) > Math.abs(best) ? p.v : best, 0);
                }
                return Math.max(...binPoints.map(p => p.v));
            });
        }

        if (!interpolate) {
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

        validPoints.sort((a, b) => a.r - b.r);

        return interpolatedRatios.map(targetR => {
            let lower = null, upper = null;
            for (const pt of validPoints) {
                if (pt.r <= targetR) lower = pt;
                if (pt.r >= targetR && upper === null) upper = pt;
            }

            if (lower && lower.r === targetR) return lower.v;
            if (upper && upper.r === targetR) return upper.v;
            if (!lower || !upper) return null;

            const t = (targetR - lower.r) / (upper.r - lower.r);
            return lower.v + t * (upper.v - lower.v);
        });
    });

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

    const xIndices = xRatios.map((_, i) => String(i));

    // Build custom hover text
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
        connectgaps: interpolate,
        hovertemplate: '%{text}<extra></extra>',
        text: hoverText,
        colorbar: {
            title: { text: metric === 'delta' ? 'Delta' : metric === 'coherence' ? 'Coherence' : 'Trait', font: { size: 11 } }
        }
    };

    const xAxisLabel = 'Coefficient';

    // Generate evenly-spaced tick positions
    const numTicks = Math.min(10, xRatios.length);
    const tickIndices = [];
    const tickLabels = [];
    for (let i = 0; i < numTicks; i++) {
        const idx = Math.round(i * (xRatios.length - 1) / (numTicks - 1));
        tickIndices.push(String(idx));
        tickLabels.push(xRatios[idx].toFixed(0));
    }

    const xAxisConfig = {
        title: xAxisLabel,
        tickfont: { size: 10 },
        tickvals: tickIndices,
        ticktext: tickLabels,
        type: 'category'
    };

    const layout = window.buildChartLayout({
        preset: 'heatmap',
        traces: [trace],
        height: Math.max(300, layers.length * 20 + 100),
        legendPosition: 'none',
        xaxis: xAxisConfig,
        yaxis: { title: 'Layer', tickfont: { size: 10 }, autorange: 'reversed' },
        margin: { l: 50, r: 80, t: 20, b: 50 }
    });
    window.renderChart(container, [trace], layout);
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


// Export
window.renderSteeringSweep = renderSteeringSweep;
