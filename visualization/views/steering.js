import { fetchJSON, escapeHtml } from '../core/utils.js';

// Steering Sweep - Heatmap visualization of steering experiments
// Shows layer × perturbation ratio → delta/coherence
//
// Sections:
// 1. Best Vector per Layer: Multi-trait comparison by extraction method
// 2. Layer × Coefficient Heatmaps: Delta and coherence heatmaps
//
// Response browser component is in: components/response-browser.js

async function renderSteering() {
    const contentArea = document.getElementById('content-area');
    if (ui.requireExperiment(contentArea)) return;

    const loading = ui.deferredLoading(contentArea, 'Loading steering sweep data...');

    // Get current trait from state or use default
    const traits = await discoverSteeringTraits();
    discoveredSteeringTraits = traits; // Store for use by other functions

    loading.cancel();

    if (traits.length === 0) {
        contentArea.innerHTML = `
            <div class="tool-view">
                <div class="no-data">
                    <p>No steering sweep data found</p>
                    <small>Run steering experiments with: <code>python steering/run_steering_eval.py --experiment ${window.state.experimentData?.name || 'your_experiment'} --trait category/trait --layers 8,10,12 --find-coef</code></small>
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
                    <div><span class="example-label">Sweet spot:</span> ratio ~1.0 ± 0.15 for most layers</div>
                </div>
            </div>

            <!-- Global controls (applies to all charts) -->
            <div class="sweep-controls sticky-coherence">
                <div class="control-group">
                    <label>Min Coherence:</label>
                    <input type="range" id="sweep-coherence-threshold" min="0" max="100" value="77" />
                    <span id="coherence-threshold-value">77</span>
                </div>
                <div id="chart-filter-rows"></div>
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
                ${ui.renderSubsection({
                    num: 2,
                    title: 'Layer × Coefficient Heatmaps',
                    infoId: 'info-heatmaps',
                    infoText: 'Steering intervention at layer l modifies the residual stream: h\'[l] = h[l] + coef × v[l], where v[l] is the trait vector and coef controls strength. Left heatmap: Δtrait = (steered trait score) − (baseline trait score). Positive = steering toward trait. Only shows runs where coherence ≥ threshold. Right heatmap: Coherence score (0-100) measuring response quality. Low coherence = garbled output from over-steering. X-axis: coefficient values. Y-axis: injection layer. Sweet spot is typically coef ≈ ±50-200 at layers 8-16.'
                })}

                <!-- Controls for heatmaps -->
                <div class="sweep-controls">
                    <div class="control-group">
                        <label>Trait:</label>
                        <select id="sweep-trait-select"></select>
                    </div>
                    ${ui.renderSelect({
                        id: 'sweep-method',
                        label: 'Method',
                        options: [
                            { value: 'all', label: 'All Methods' },
                            { value: 'probe', label: 'Probe' },
                            { value: 'gradient', label: 'Gradient' },
                            { value: 'mean_diff', label: 'Mean Diff' },
                        ],
                        selected: 'all',
                    })}
                    ${ui.renderToggle({ id: 'sweep-interpolate', label: 'Interpolate' })}
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

    // Collect filter values from data, then render chips and charts
    await collectFilterValues(traits);
    renderFilterChips();

    await renderBestVectorPerLayer();
    await renderTraitPicker(traits);

    // Set default selected entry if not set
    if (!selectedSteeringEntry && traits.length > 0) {
        selectedSteeringEntry = defaultTrait;
    }

    await renderSweepData(selectedSteeringEntry || defaultTrait);

    // Setup event handlers
    document.getElementById('sweep-method').addEventListener('change', () => updateSweepVisualizations());

    document.getElementById('sweep-coherence-threshold').addEventListener('input', async (e) => {
        document.getElementById('coherence-threshold-value').textContent = e.target.value;
        await renderBestVectorPerLayer();
        updateSweepVisualizations();
    });

    document.getElementById('sweep-interpolate').addEventListener('change', () => updateSweepVisualizations());

    // Setup info toggles
    window.setupSubsectionInfoToggles();
}


// Store current data for re-rendering on control changes
let currentSweepData = null;
let currentRawResults = null; // Store raw results.jsonl data for method filtering
let discoveredSteeringTraits = []; // All discovered steering traits
let localTraitResultsCache = {}; // Local cache, passed to response-browser via setTraitResultsCache()
let selectedSteeringEntry = null; // Selected trait entry for heatmaps (reset on experiment change)
let steeringResultsCache = {}; // URL → fetched results, shared between collectFilterValues and renderers

// Global chart filters - populated from data, all active by default
let chartFilters = {
    modelVariants: new Set(), // e.g. 'instruct', 'kimi_k2'
    methods: new Set(),      // e.g. 'probe', 'mean_diff', 'gradient'
    positions: new Set(),    // e.g. 'response[:]', 'response[:5]'
    components: new Set(),   // e.g. 'residual', 'attn_contribution'
    directions: new Set(),   // e.g. 'positive', 'negative'
    // Active selections (what's checked) - initialized to all
    activeModelVariants: new Set(),
    activeMethods: new Set(),
    activePositions: new Set(),
    activeComponents: new Set(),
    activeDirections: new Set(),
};


/** Build the steering results API URL for an entry. */
function steeringResultsUrl(entry) {
    const experiment = window.state.experimentData?.name;
    return `/api/experiments/${experiment}/steering-results/${entry.trait}/${entry.model_variant}/${entry.position}/${entry.prompt_set}`;
}

/** Fetch steering results with caching. */
async function fetchSteeringResults(entry) {
    const url = steeringResultsUrl(entry);
    return cachedFetchJSON(steeringResultsCache, url, url);
}

/**
 * Collect all unique filter values from steering entries and their results.
 * Called once after data loads, populates chartFilters.{methods,positions,components,directions}.
 * Also populates steeringResultsCache so later renderers can reuse fetched data.
 */
async function collectFilterValues(steeringEntries) {
    const experiment = window.state.experimentData?.name;
    if (!experiment || !steeringEntries.length) return;

    const methods = new Set();
    const positions = new Set();
    const components = new Set();
    const directions = new Set();
    const modelVariants = new Set();

    // Positions and model variants come from entries directly
    for (const entry of steeringEntries) {
        positions.add(entry.position);
        modelVariants.add(entry.model_variant);
    }

    // Methods, components, directions require loading results (cached for reuse)
    const results = await Promise.all(steeringEntries.map(entry => fetchSteeringResults(entry)));

    for (const result of results) {
        if (!result) continue;
        // Direction from header
        if (result.direction) directions.add(result.direction);
        for (const run of (result.runs || [])) {
            const vectors = run.config?.vectors || [];
            if (vectors.length !== 1) continue;
            const v = vectors[0];
            if (v.method) methods.add(v.method);
            if (v.component) components.add(v.component);
            else components.add('residual');
            // Infer direction from coefficient sign if header didn't specify
            if (!result.direction) {
                directions.add(v.weight > 0 ? 'positive' : 'negative');
            }
        }
    }

    // Update state
    chartFilters.modelVariants = modelVariants;
    chartFilters.methods = methods;
    chartFilters.positions = positions;
    chartFilters.components = components;
    chartFilters.directions = directions;
    // Default: first model variant only (single-select), all others multi-select
    chartFilters.activeModelVariants = new Set([modelVariants.values().next().value]);
    chartFilters.activeMethods = new Set(methods);
    chartFilters.activePositions = new Set(positions);
    chartFilters.activeComponents = new Set(components);
    chartFilters.activeDirections = new Set(directions);
}


/**
 * Render filter chip rows into the sticky bar.
 * Only renders rows with >1 option (no point filtering if there's only one).
 */
function renderFilterChips() {
    const container = document.getElementById('chart-filter-rows');
    if (!container) return;

    const displayNames = {
        'probe': 'Probe', 'mean_diff': 'Mean Diff', 'gradient': 'Gradient',
        'residual': 'Residual', 'attn_contribution': 'Attn', 'mlp_contribution': 'MLP',
        'k_proj': 'K Proj', 'v_proj': 'V Proj',
        'positive': 'Positive', 'negative': 'Negative',
    };
    const formatLabel = v => window.paths?.formatPositionDisplay(v)
        || v.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');

    const rows = [
        ui.renderFilterChipRow('Model', chartFilters.modelVariants, chartFilters.activeModelVariants, 'modelVariants', { displayNames, formatLabel }),
        ui.renderFilterChipRow('Method', chartFilters.methods, chartFilters.activeMethods, 'methods', { displayNames, formatLabel }),
        ui.renderFilterChipRow('Position', chartFilters.positions, chartFilters.activePositions, 'positions', { displayNames, formatLabel }),
        ui.renderFilterChipRow('Component', chartFilters.components, chartFilters.activeComponents, 'components', { displayNames, formatLabel }),
        ui.renderFilterChipRow('Direction', chartFilters.directions, chartFilters.activeDirections, 'directions', { displayNames, formatLabel }),
    ];

    container.innerHTML = rows.filter(r => r).join('');

    // Wire click handlers
    // Model variants = single-select (radio), everything else = multi-select (toggle)
    const singleSelectFilters = new Set(['modelVariants']);

    container.querySelectorAll('.filter-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const filterKey = chip.dataset.filterGroup;
            const value = chip.dataset.value;
            const activeSetKey = 'active' + filterKey.charAt(0).toUpperCase() + filterKey.slice(1);
            const activeSet = chartFilters[activeSetKey];

            if (singleSelectFilters.has(filterKey)) {
                // Radio behavior: activate clicked, deactivate others
                if (activeSet.has(value) && activeSet.size === 1) return; // already the only one
                activeSet.clear();
                activeSet.add(value);
                // Update all chips in this row
                container.querySelectorAll(`.filter-chip[data-filter-group="${filterKey}"]`).forEach(c => {
                    c.classList.toggle('active', c.dataset.value === value);
                });
            } else {
                // Toggle behavior
                if (activeSet.has(value)) {
                    if (activeSet.size > 1) {
                        activeSet.delete(value);
                        chip.classList.remove('active');
                    }
                } else {
                    activeSet.add(value);
                    chip.classList.add('active');
                }
            }

            // Re-render charts
            renderBestVectorPerLayer();
        });
    });
}


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

    if (meta.vector_source?.model && meta.vector_source.model !== 'unknown' && meta.vector_source.model !== meta.steering_model) {
        html += ` · Vector from: <code>${meta.vector_source.model}</code>`;
    }

    if (meta.eval?.model) {
        html += ` · Eval: <code>${meta.eval.model}</code> (${meta.eval.method || 'unknown'})`;
    }

    container.innerHTML = html;
}


async function discoverSteeringTraits() {
    if (!window.state.experimentData?.name) return [];
    const data = await fetchJSON(`/api/experiments/${window.state.experimentData.name}/steering`);
    return data?.entries || [];
}


async function renderSweepData(steeringEntry) {
    if (!window.state.experimentData?.name || !steeringEntry) return;

    const results = await fetchSteeringResults(steeringEntry);
    currentRawResults = results;
    const data = results ? convertResultsToSweepFormat(results) : null;

    updateSteeringModelInfo(results);

    if (!data) {
        document.getElementById('sweep-heatmap-delta').innerHTML = '<p class="no-data">No data for this trait</p>';
        document.getElementById('sweep-table-container').innerHTML = '';
        return;
    }

    // Check if we have any single-layer data
    const fullVector = data.full_vector || {};
    if (Object.keys(fullVector).length === 0) {
        document.getElementById('sweep-heatmap-delta').innerHTML = `
            <p class="no-data">No single-layer steering runs found.<br>
            <small>Heatmap requires single-layer runs. Multi-layer runs are not visualized here.</small></p>
        `;
        document.getElementById('sweep-table-container').innerHTML = '';
        return;
    }

    currentSweepData = data;
    updateSweepVisualizations();
}


/** Extract trait name from full path, e.g. "pv_instruction/evil" → "evil" */
function getBaseTraitName(trait) {
    return trait.split('/').pop();
}

/** Get elicitation label from category, e.g. "pv_instruction/evil" → "instruction" */
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

    // Guard: container may not exist if user switched views
    if (!container) return;

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
    const coherenceThreshold = coherenceThresholdEl ? parseInt(coherenceThresholdEl.value) : 77;

    // Group by base trait name (e.g., "evil" groups pv_instruction/evil and pv_natural/evil)
    const traitGroups = {};
    for (const entry of steeringEntries) {
        const baseTrait = getBaseTraitName(entry.trait);
        if (!traitGroups[baseTrait]) traitGroups[baseTrait] = [];
        traitGroups[baseTrait].push(entry);
    }

    const charts = [];
    let modelInfoUpdated = false;

    for (const [baseTrait, variants] of Object.entries(traitGroups)) {
        // Filter by active model variants
        const activeVariants = chartFilters.activeModelVariants.size > 0
            ? variants.filter(v => chartFilters.activeModelVariants.has(v.model_variant))
            : variants;
        if (activeVariants.length === 0) continue;

        const traces = [];
        const methodColors = window.getMethodColors();
        let baseline = null;
        let colorIdx = 0;
        const allRuns = []; // Collect all runs for response browser

        // Check if this group has multiple elicitation methods (constant across variants)
        const hasMultipleElicit = new Set(variants.map(v => getElicitationLabel(v.trait))).size > 1;

        // Load results for each position variant (in parallel, cached)
        const variantResults = await Promise.all(activeVariants.map(async (entry) => {
            const results = await fetchSteeringResults(entry);
            return results ? { entry, results } : null;
        }));

        // Process results sequentially (baseline depends on first result)
        for (const variantResult of variantResults) {
            if (!variantResult) continue;
            const { entry, results } = variantResult;
            const position = entry.position;

            if (baseline === null) {
                baseline = results.baseline?.trait_mean || 0;
            }

            if (!modelInfoUpdated) {
                updateSteeringModelInfo(results);
                modelInfoUpdated = true;
            }

            // Group by (component, method) and layer, find best trait score per combo
            const methodData = {};

            for (const run of (results.runs || [])) {
                const config = run.config || {};
                const result = run.result || {};

                // Support VectorSpec format
                const vectors = config.vectors || [];
                if (vectors.length !== 1) continue;

                const v = vectors[0];
                const layer = v.layer;
                const method = v.method;
                const component = v.component || 'residual';
                const coef = v.weight;
                const coherence = result.coherence_mean || 0;
                const traitScore = result.trait_mean || 0;

                // Collect for response browser (before filters)
                allRuns.push({
                    layer, method, component, coef, traitScore, coherence,
                    timestamp: run.timestamp,
                    entry, // steering entry for path building
                });

                // Apply global chart filters (empty set = no filter applied yet)
                if (chartFilters.activeMethods.size > 0 && !chartFilters.activeMethods.has(method)) continue;
                if (chartFilters.activeComponents.size > 0 && !chartFilters.activeComponents.has(component)) continue;
                if (chartFilters.activePositions.size > 0 && !chartFilters.activePositions.has(position)) continue;
                const direction = coef > 0 ? 'positive' : 'negative';
                if (chartFilters.activeDirections.size > 0 && !chartFilters.activeDirections.has(direction)) continue;

                if (coherence < coherenceThreshold) continue;

                // Key includes component for differentiation
                const key = component === 'residual' ? method : `${component}/${method}`;
                if (!methodData[key]) methodData[key] = {};
                if (!methodData[key][layer] || traitScore > methodData[key][layer].score) {
                    methodData[key][layer] = { score: traitScore, coef };
                }
            }

            const posDisplayClosed = position ? window.paths.formatPositionDisplay(position) : '';
            const elicitLabel = getElicitationLabel(entry.trait);
            const fullTraitName = entry.trait.split('/').pop();

            Object.entries(methodData).forEach(([methodKey, layerData]) => {
                const layers = sortedNumericKeys(layerData);
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
            <div id="${chartId}-legend"></div>
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

    // Plot each chart with HTML legend (no Plotly legend)
    for (const { chartId, traces } of charts) {
        // Guard: element may not exist if user switched views during async operations
        const chartEl = document.getElementById(chartId);
        if (!chartEl) continue;

        const layout = window.buildChartLayout({
            preset: 'layerChart',
            traces,
            height: 220,
            legendPosition: 'none',  // HTML legend handles this
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

        // Add HTML legend below chart
        const legendContainer = document.getElementById(`${chartId}-legend`);
        if (legendContainer) {
            const legendEl = window.createHtmlLegend(traces, chartId);
            legendContainer.appendChild(legendEl);
        }
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
    const currentFullPath = selectedSteeringEntry?.full_path || steeringEntries[0].full_path;

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
        selectedSteeringEntry = selectedEntry;
        await renderSweepData(selectedEntry);
    });
}


/** Convert results.jsonl format to sweep visualization format. */
function convertResultsToSweepFormat(results, methodFilter = null) {
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

    const layers = sortedNumericKeys(data);
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
    const metricKey = metric === 'delta' ? 'deltas' : 'coherences';
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

    // Determine color scale based on metric (only 'delta' and 'coherence' are used)
    let colorscale, zmid, zmin, zmax;
    if (metric === 'delta') {
        colorscale = window.DELTA_COLORSCALE;
        zmid = 0;
        const allVals = matrix.flat().filter(v => v !== null);
        const absMax = Math.max(Math.abs(Math.min(...allVals, 0)), Math.abs(Math.max(...allVals, 0)));
        zmin = -absMax;
        zmax = absMax;
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
            const metricLabel = metric === 'delta' ? 'Delta' : 'Coherence';
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
            title: { text: metric === 'delta' ? 'Delta' : 'Coherence', font: { size: 11 } }
        }
    };

    // Generate evenly-spaced tick positions
    const numTicks = Math.min(10, xRatios.length);
    const tickIndices = [];
    const tickLabels = [];
    for (let i = 0; i < numTicks; i++) {
        const idx = Math.round(i * (xRatios.length - 1) / (numTicks - 1));
        tickIndices.push(String(idx));
        tickLabels.push(xRatios[idx].toFixed(0));
    }

    const layout = window.buildChartLayout({
        preset: 'heatmap',
        traces: [trace],
        height: Math.max(300, layers.length * 20 + 100),
        legendPosition: 'none',
        xaxis: { title: 'Coefficient', tickfont: { size: 10 }, tickvals: tickIndices, ticktext: tickLabels, type: 'category' },
        yaxis: { title: 'Layer', tickfont: { size: 10 }, autorange: 'reversed' },
        margin: { l: 50, r: 80, t: 20, b: 50 }
    });
    window.renderChart(container, [trace], layout);
}


function renderSweepTable(data, coherenceThreshold) {
    const container = document.getElementById('sweep-table-container');

    const layers = sortedNumericKeys(data);
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
                    const cohClass = ui.scoreClass(r.coherence, 'coherence');
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


/** Reset steering-local state (called on experiment change). */
function resetSteeringState() {
    currentSweepData = null;
    currentRawResults = null;
    discoveredSteeringTraits = [];
    localTraitResultsCache = {};
    selectedSteeringEntry = null;
    steeringResultsCache = {};
}

// ES module exports
export { renderSteering, resetSteeringState };

// Keep window.* for router + state.js reference
window.renderSteering = renderSteering;
window.resetSteeringState = resetSteeringState;
