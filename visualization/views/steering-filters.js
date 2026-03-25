// Steering filter system — filter chips, result caching, filter value collection
//
// Manages global chart filters (model variant, method, position, component, direction)
// and provides cached steering result fetching shared across steering sub-modules.

import { cachedFetchJSON } from '../core/utils.js';
import { renderFilterChipRow } from '../core/ui.js';

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

let steeringResultsCache = {}; // URL → fetched results, shared between collectFilterValues and renderers

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
        renderFilterChipRow('Model', chartFilters.modelVariants, chartFilters.activeModelVariants, 'modelVariants', { displayNames, formatLabel }),
        renderFilterChipRow('Method', chartFilters.methods, chartFilters.activeMethods, 'methods', { displayNames, formatLabel }),
        renderFilterChipRow('Position', chartFilters.positions, chartFilters.activePositions, 'positions', { displayNames, formatLabel }),
        renderFilterChipRow('Component', chartFilters.components, chartFilters.activeComponents, 'components', { displayNames, formatLabel }),
        renderFilterChipRow('Direction', chartFilters.directions, chartFilters.activeDirections, 'directions', { displayNames, formatLabel }),
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

            // Re-render charts (imported lazily via window to avoid circular deps)
            window._steeringRenderBestVector();
        });
    });
}

/** Reset filter state. */
function resetFiltersState() {
    chartFilters = {
        modelVariants: new Set(),
        methods: new Set(),
        positions: new Set(),
        components: new Set(),
        directions: new Set(),
        activeModelVariants: new Set(),
        activeMethods: new Set(),
        activePositions: new Set(),
        activeComponents: new Set(),
        activeDirections: new Set(),
    };
    steeringResultsCache = {};
}

export { chartFilters, collectFilterValues, renderFilterChips, fetchSteeringResults, resetFiltersState };
