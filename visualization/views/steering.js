// Steering view — orchestrator
//
// Builds the HTML shell, delegates to sub-modules:
//   steering-filters.js   — filter chips, cached fetching, filter state
//   steering-best-vector.js — Section 1: best vector per layer charts
//   steering-heatmap.js   — Section 2: layer × coefficient heatmaps

import { fetchJSON } from '../core/utils.js';
import { requireExperiment, deferredLoading, renderSubsection, renderSelect, renderToggle } from '../core/ui.js';
import { chartFilters, collectFilterValues, renderFilterChips, resetFiltersState } from './steering-filters.js';
import { renderBestVectorPerLayer, resetBestVectorState } from './steering-best-vector.js';
import {
    renderSweepData, renderTraitPicker, updateSweepVisualizations,
    resetHeatmapState, getSelectedSteeringEntry, setSelectedSteeringEntry
} from './steering-heatmap.js';

let discoveredSteeringTraits = []; // All discovered steering traits

// Expose shared state for sub-modules that need it via window bridges
// (avoids circular imports between filters → best-vector)
window._steeringRenderBestVector = () => renderBestVectorPerLayer();
window._steeringUpdateModelInfo = (meta) => updateSteeringModelInfo(meta);
Object.defineProperty(window, '_steeringDiscoveredTraits', {
    get: () => discoveredSteeringTraits,
    configurable: true,
});

async function renderSteering() {
    const contentArea = document.getElementById('content-area');
    if (requireExperiment(contentArea)) return;

    const loading = deferredLoading(contentArea, 'Loading steering sweep data...');

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
                ${renderSubsection({
                    num: 1,
                    title: 'Best Vector per Layer',
                    infoId: 'info-best-vector',
                    infoText: 'For each selected trait (from sidebar), shows the best trait score achieved per layer across all 3 extraction methods (probe, gradient, mean_diff). Each trait gets its own chart showing which method works best at which layer. Dashed line shows baseline (no steering).'
                })}
                <div id="best-vector-container"></div>
            </section>

            <!-- Heatmaps section -->
            <section>
                ${renderSubsection({
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
                    ${renderSelect({
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
                    ${renderToggle({ id: 'sweep-interpolate', label: 'Interpolate' })}
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
    const selectedEntry = getSelectedSteeringEntry();
    if (!selectedEntry && traits.length > 0) {
        setSelectedSteeringEntry(defaultTrait);
    }

    await renderSweepData(getSelectedSteeringEntry() || defaultTrait);

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


/** Reset steering-local state (called on experiment change). */
function resetSteeringState() {
    discoveredSteeringTraits = [];
    resetFiltersState();
    resetBestVectorState();
    resetHeatmapState();
}

// ES module exports
export { renderSteering, resetSteeringState };

// Keep window.* for router + state.js reference
window.renderSteering = renderSteering;
window.resetSteeringState = resetSteeringState;
