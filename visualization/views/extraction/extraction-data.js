/**
 * Shared state + helpers for the extraction view.
 *
 * Module-local state lives in `extractionState` (mutable object). Sibling section
 * files import it directly and mutate fields. resetExtractionState() clears the
 * fields that should reset on experiment change.
 *
 * Usage:
 *   import { extractionState, METRIC_CONFIG, computeBestVectors,
 *            getSelectedTraitNames, resetExtractionState } from './extraction-data.js';
 */

import { DELTA_COLORSCALE, ASYMB_COLORSCALE } from '../../core/display.js';

// Shared mutable state across extraction sections.
const extractionState = {
    // Heatmap metric toggle — default to signed effect size
    heatmapMetric: 'effect_size',
    // Vector Geometry subsection
    vgMethod: null,           // currently-selected method
    vgLayer: null,            // currently-selected layer
    vgSelectedTrait: null,    // click-to-inspect
    // Logit-lens cache: { trait → data }. Populated once by section-logit-lens.
    logitLensCache: null,
    logitLensEvalData: null,
};

// Metric config: how to compute cell value, colorscale, z-range, legend label
const METRIC_CONFIG = {
    effect_size: {
        label: 'Effect Size (d)',
        legendLabels: ['−max', '0', '+max'],
        legendBarClass: 'heatmap-legend-bar-diverging',
        colorscale: DELTA_COLORSCALE,
        hoverSuffix: 'd=%{z:.2f}',
        // Signed by polarity: green = correct direction, red = flipped
        computeCell: (r) => r.val_effect_size == null ? null
            : (r.polarity_correct ? r.val_effect_size : -r.val_effect_size),
        zRange: (values) => {
            const absMax = values.length ? Math.max(...values.map(Math.abs)) : 1;
            const b = Math.ceil(absMax);
            return { zmin: -b, zmax: b, zmid: 0 };
        },
    },
    val_accuracy: {
        label: 'Val Accuracy (%)',
        legendLabels: ['0%', '50%', '100%'],
        legendBarClass: 'heatmap-legend-bar-diverging',
        colorscale: DELTA_COLORSCALE,
        hoverSuffix: 'acc=%{z:.1f}%',
        // Accuracy relative to chance: 50% = neutral
        computeCell: (r) => r.val_accuracy == null ? null : r.val_accuracy * 100,
        zRange: () => ({ zmin: 0, zmax: 100, zmid: 50 }),
    },
    combined: {
        label: 'Combined Score',
        legendLabels: ['0', '', '1'],
        legendBarClass: '',  // sequential green
        colorscale: ASYMB_COLORSCALE,
        hoverSuffix: 'score=%{z:.2f}',
        computeCell: (r) => r.combined_score == null ? null : r.combined_score,
        zRange: () => ({ zmin: 0, zmax: 1 }),
    },
};

/** Return trait names selected in sidebar filter, or empty set for "show all" */
function getSelectedTraitNames() {
    const filteredTraits = window.getFilteredTraits();
    return new Set(filteredTraits.map(t => t.name));
}

/**
 * Compute best vector per trait from all_results using effect_size.
 * Returns: {trait: {layer, method, score}}
 */
function computeBestVectors(allResults) {
    const bestByTrait = {};
    for (const r of allResults) {
        const trait = r.trait;
        const effectSize = r.val_effect_size;
        if (effectSize == null) continue;

        if (!bestByTrait[trait] || effectSize > bestByTrait[trait].score) {
            bestByTrait[trait] = {
                layer: r.layer,
                method: r.method,
                score: effectSize
            };
        }
    }
    return bestByTrait;
}

/** Reset extraction-local state (called on experiment change). */
function resetExtractionState() {
    extractionState.vgMethod = null;
    extractionState.vgLayer = null;
    extractionState.vgSelectedTrait = null;
    extractionState.logitLensCache = null;
    extractionState.logitLensEvalData = null;
    // heatmapMetric persists across experiment changes intentionally.
}

export {
    extractionState,
    METRIC_CONFIG,
    getSelectedTraitNames,
    computeBestVectors,
    resetExtractionState,
};
