// Steering Section 1 — Best Vector per Layer
//
// Multi-trait comparison chart showing the best trait score per layer
// across extraction methods (probe, gradient, mean_diff). One chart per
// base trait, with lines for each (method, position, elicitation) combo.

import { sortedNumericKeys } from '../core/utils.js';
import { getDisplayName, getChartColors, getMethodColors } from '../core/display.js';
import { buildChartLayout, renderChart, createHtmlLegend } from '../core/charts.js';
import { chartFilters, fetchSteeringResults } from './steering-filters.js';

let localTraitResultsCache = {}; // Local cache, passed to response-browser via setTraitResultsCache()

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
    const steeringEntries = window._steeringDiscoveredTraits || [];

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
        const methodColors = getMethodColors();
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
                window._steeringUpdateModelInfo(results);
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

                    const baseColor = methodColors[method] || getChartColors()[colorIdx % 10];
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
            <div class="trait-chart-title">${getDisplayName(trait)}</div>
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

        const layout = buildChartLayout({
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

        renderChart(chartId, traces, layout);

        // Add HTML legend below chart
        const legendContainer = document.getElementById(`${chartId}-legend`);
        if (legendContainer) {
            const legendEl = createHtmlLegend(traces, chartId);
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

/** Reset best-vector state. */
function resetBestVectorState() {
    localTraitResultsCache = {};
}

export { renderBestVectorPerLayer, resetBestVectorState };
