// Steering filter system — filter chips, result caching, filter value collection
//
// Manages global chart filters (model variant, method, position, component, direction)
// and provides cached steering result fetching shared across steering sub-modules.

import { cachedFetchJSON } from '../../core/utils.js';
import { renderChipGroup, wireChipGroup } from '../../components/styled-chip-group.js';
import { extractVectorSpec } from './shared.js';

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
        if (result.direction) directions.add(result.direction);
        for (const run of (result.runs || [])) {
            const spec = extractVectorSpec(run);
            if (!spec) continue;
            methods.add(spec.method);
            components.add(spec.component);
            if (!result.direction) {
                directions.add(spec.coef > 0 ? 'positive' : 'negative');
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
    // Model variants show the HF model path from experiment config when available
    // (falls back to the variant nickname if no mapping exists).
    const variantModels = window.state.experimentData?.experimentConfig?.model_variants || {};
    const variantDisplay = Object.fromEntries(
        Object.entries(variantModels).map(([nickname, v]) => [nickname, v.model || nickname])
    );
    const formatLabel = v => variantDisplay[v]
        || window.paths?.formatPositionDisplay(v)
        || v.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');

    // Model variants = single-select (radio), others = multi-select with min-one invariant.
    const rowConfigs = [
        { key: 'modelVariants',  label: 'Model',     mode: 'single' },
        { key: 'methods',        label: 'Method',    mode: 'multi-min-one' },
        { key: 'positions',      label: 'Position',  mode: 'multi-min-one' },
        { key: 'components',     label: 'Component', mode: 'multi-min-one' },
        { key: 'directions',     label: 'Direction', mode: 'multi-min-one' },
    ];

    const rows = rowConfigs.map(({ key, label, mode }) => {
        const values = Array.from(chartFilters[key]);
        if (values.length <= 1) return '';
        const activeKey = 'active' + key.charAt(0).toUpperCase() + key.slice(1);
        const activeSet = chartFilters[activeKey];
        const items = values.map(v => ({
            value: v,
            label: displayNames[v] ?? formatLabel(v),
        }));
        const chipGroup = renderChipGroup({
            id: `filter-${key}`,
            mode,
            items,
            selected: activeSet,
            onChange: (value, newSelected) => {
                if (mode === 'single') {
                    activeSet.clear();
                    activeSet.add(value);
                } else {
                    // newSelected is the post-click Set<value>
                    activeSet.clear();
                    newSelected.forEach(v => activeSet.add(v));
                }
                window._steeringRenderBestVector();
            },
        });
        return `<div class="filter-row"><span class="filter-label">${label}:</span>${chipGroup}</div>`;
    });

    container.innerHTML = rows.filter(r => r).join('');
    wireChipGroup(container);
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
