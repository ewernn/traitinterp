/**
 * Shared state + data fetching for model-analysis view.
 *
 * Module-local state (`_maVariant`, `_maCriteria`) is exposed via getter/setter
 * pairs so sibling section files can read/write without circular imports.
 *
 * Input:  experiment name, current variant + criteria
 * Output: variant dropdown, calibration data, withMassiveActivationsData wrapper
 * Usage:  import { getSelectedVariant, fetchMassiveActivationsData } from './model-analysis-data.js';
 */

import { fetchJSON } from '../../core/utils.js';
import { renderRunHint } from '../../core/ui.js';
import { renderStyledSelect, wireStyledSelect } from '../../components/styled-select.js';

let _maVariant = null;
let _maCriteria = 'top5-3layers';

function getMaVariant() { return _maVariant; }
function setMaVariant(v) { _maVariant = v; }
function getMaCriteria() { return _maCriteria; }
function setMaCriteria(v) { _maCriteria = v; }

/**
 * Get the currently selected model variant for activation diagnostics.
 */
function getSelectedVariant() {
    return _maVariant || window.state.experimentData?.experimentConfig?.defaults?.application || 'instruct';
}

/**
 * Populate the model variant dropdown with available variants that have calibration data.
 * @param {string} experiment - Experiment name
 * @param {Function} onVariantChange - async (val) => void, fired on variant change
 */
async function populateVariantDropdown(experiment, onVariantChange) {
    const container = document.getElementById('activation-diagnostics-variant-container');
    if (!container) return;

    const defaultVariant = window.state.experimentData?.experimentConfig?.defaults?.application || 'instruct';

    const data = await fetchJSON(`/api/experiments/${experiment}/calibration-variants`);
    const availableVariants = data?.variants || [];

    if (availableVariants.length === 0) {
        container.innerHTML = '<span class="cb-label" style="opacity:0.5;">No calibration data</span>';
        _maVariant = null;
        return;
    }

    _maVariant = availableVariants.includes(_maVariant) ? _maVariant
        : availableVariants.includes(defaultVariant) ? defaultVariant
        : availableVariants[0];

    container.innerHTML = renderStyledSelect({
        id: 'activation-diagnostics-variant',
        options: availableVariants.map(v => ({ value: v, label: v })),
        selected: _maVariant,
        onChange: async (val) => {
            _maVariant = val;
            await onVariantChange(val);
        },
    });
    wireStyledSelect(container);
}

/**
 * Fetch massive activations data, using calibration.json as canonical source.
 * Calibration contains model-wide massive dims computed from neutral prompts.
 */
async function fetchMassiveActivationsData() {
    const modelVariant = getSelectedVariant();
    const calibrationPath = window.paths.get('inference.massive_activations', { prompt_set: 'calibration', model_variant: modelVariant });
    return fetchJSON('/' + calibrationPath);
}

/**
 * Shared wrapper for diagnostic render functions.
 * Handles container lookup, null-data guard, and error catch.
 * @param {string} containerId - DOM element ID
 * @param {any} data - Pre-fetched calibration data
 * @param {Function} renderFn - (container, data) => void
 */
function withMassiveActivationsData(containerId, data, renderFn) {
    const container = document.getElementById(containerId);
    if (!container) return;

    if (!data) {
        container.innerHTML = renderRunHint(
            'No massive activation calibration data.',
            `python inference/run_inference_pipeline.py --experiment ${window.paths.getExperiment()} --prompt-set starter_prompts/general   # captures automatically`
        );
        return;
    }

    try {
        renderFn(container, data);
    } catch (error) {
        container.innerHTML = `<div class="info">Error loading data: ${error.message}</div>`;
    }
}

export {
    getMaVariant,
    setMaVariant,
    getMaCriteria,
    setMaCriteria,
    getSelectedVariant,
    populateVariantDropdown,
    fetchMassiveActivationsData,
    withMassiveActivationsData,
};
