// Steering view — orchestrator
//
// Builds the HTML shell with compact controls and overview grid.
// Delegates to sub-modules:
//   filters.js  — filter chips, cached fetching, filter state
//   overview.js — card grid with sparklines
//   detail.js   — inline detail panel per trait
//   shared.js   — helpers used across the sub-modules

import { fetchJSON } from '../../core/utils.js';
import { requireExperiment, deferredLoading } from '../../core/ui.js';
import { chartFilters, collectFilterValues, renderFilterChips, fetchSteeringResults, resetFiltersState } from './filters.js';
import { renderOverviewGrid, resetOverviewState } from './overview.js';
import { showDetailPanel, hideDetailPanel, resetDetailState } from './detail.js';

let discoveredSteeringTraits = []; // All discovered steering traits

// Expose shared state for sub-modules that need it via window bridges
// (avoids circular imports between filters → overview)
window._steeringRenderBestVector = () => {
    // Re-render overview grid when filters change
    const gridEl = document.getElementById('steering-overview-grid');
    const slider = document.getElementById('sweep-coherence-threshold');
    if (gridEl && slider) {
        const threshold = parseInt(slider.value, 10);
        renderOverviewGrid(gridEl, discoveredSteeringTraits, fetchSteeringResults, threshold);
    }
};
window._steeringUpdateModelInfo = (meta) => updateSteeringModelInfo(meta);
Object.defineProperty(window, '_steeringDiscoveredTraits', {
    get: () => discoveredSteeringTraits,
    configurable: true,
});

// Wire up detail panel bridges for overview cards
window._steeringShowDetail = async (entry, cardEl) => {
    const results = await fetchSteeringResults(entry);
    showDetailPanel(cardEl, entry.trait, results, entry);
};
window._steeringHideDetail = () => hideDetailPanel();

async function renderSteering() {
    const contentArea = document.getElementById('content-area');
    if (requireExperiment(contentArea)) return;

    const loading = deferredLoading(contentArea, 'Loading steering sweep data...');

    // Discover all steering traits
    const traits = await discoverSteeringTraits();
    discoveredSteeringTraits = traits;

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

    // Collect filter values from data (populates chartFilters)
    await collectFilterValues(traits);

    // Build the view
    contentArea.innerHTML = `
        <div class="tool-view">
            <!-- Compact controls — one row -->
            <div class="steering-controls">
                <div class="model-info" id="steering-model-info"></div>
                <span class="steering-sep"></span>
                <label>Coherence:</label>
                <input type="range" id="sweep-coherence-threshold" min="0" max="100" value="77" />
                <span id="coherence-threshold-value" style="font-size: var(--text-xs); color: var(--text-secondary); min-width: 24px;">77</span>
                <div id="chart-filter-rows" style="display: contents;"></div>
            </div>

            <!-- Overview grid — cards rendered by overview.js -->
            <div class="steering-grid" id="steering-overview-grid"></div>
        </div>
    `;

    // Populate model info
    updateSteeringModelInfo(null);

    // Setup info toggles (event delegation on .tool-view)
    window.setupSubsectionInfoToggles?.();

    // Render filter chips inline in the controls bar
    renderFilterChips();

    // Render the overview grid
    const coherenceThreshold = parseInt(document.getElementById('sweep-coherence-threshold').value, 10);
    renderOverviewGrid(
        document.getElementById('steering-overview-grid'),
        traits,
        fetchSteeringResults,
        coherenceThreshold
    );

    // Setup event handlers
    document.getElementById('sweep-coherence-threshold').addEventListener('input', (e) => {
        document.getElementById('coherence-threshold-value').textContent = e.target.value;
        const threshold = parseInt(e.target.value, 10);
        renderOverviewGrid(
            document.getElementById('steering-overview-grid'),
            discoveredSteeringTraits,
            fetchSteeringResults,
            threshold
        );
    });
}


/**
 * Update the steering model info display in the controls bar.
 */
function updateSteeringModelInfo(meta) {
    const container = document.getElementById('steering-model-info');
    if (!container) return;

    if (!meta || !meta.steering_model) {
        // Fall back to experiment config
        const config = window.state.experimentData?.experimentConfig;
        const appVariant = config?.defaults?.application || 'instruct';
        const steeringModel = config?.model_variants?.[appVariant]?.model || config?.model || 'unknown';
        container.innerHTML = `Steering: <code>${steeringModel}</code>`;
        return;
    }

    let html = `Steering: <code>${meta.steering_model}</code>`;

    if (meta.vector_source?.model && meta.vector_source.model !== 'unknown' && meta.vector_source.model !== meta.steering_model) {
        html += ` &middot; Vector from: <code>${meta.vector_source.model}</code>`;
    }

    if (meta.eval?.model) {
        html += ` &middot; Eval: <code>${meta.eval.model}</code> (${meta.eval.method || 'unknown'})`;
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
    resetOverviewState();
    resetDetailState();
}

// ES module exports
export { renderSteering, resetSteeringState };

// Keep window.* for router + state.js reference
window.renderSteering = renderSteering;
window.resetSteeringState = resetSteeringState;
