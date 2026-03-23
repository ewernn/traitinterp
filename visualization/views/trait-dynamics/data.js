// Data loading and caching for Trait Dynamics view
// Input: trait metadata, prompt/variant identifiers
// Output: projection data, layer sensitivity data, comparison diffs

import { fetchJSON } from '../../core/utils.js';

// =============================================================================
// Module-local caches
// =============================================================================

const layerSensitivityCache = {};

// =============================================================================
// Projection helpers
// =============================================================================

/**
 * Extract the prompt/response projection pair from comparison data,
 * matching the vector source (method + layer) of the main trait entry.
 */
function extractCompProjection(compData, vectorSource) {
    if (compData.metadata?.multi_vector && Array.isArray(compData.projections)) {
        if (vectorSource) {
            const match = compData.projections.find(p => p.method === vectorSource.method && p.layer === vectorSource.layer);
            if (match) return { prompt: match.prompt, response: match.response };
        }
        // Fallback: if no layer match, use first available projection
        if (compData.projections.length > 0) {
            const fb = compData.projections[0];
            return { prompt: fb.prompt, response: fb.response };
        }
        return null;
    }
    return compData.projections || null;
}

/**
 * Compute projection diff: a - b, trimmed to min length.
 * Logs warning if response lengths differ by more than 1 token.
 */
function computeProjectionDiff(projA, projB, traitKey) {
    const rLenDiff = Math.abs(projA.response.length - projB.response.length);
    if (rLenDiff > 1) console.warn(`[Diff] Unexpected response length mismatch for ${traitKey}: ${projA.response.length} vs ${projB.response.length} (diff=${rLenDiff})`);
    const minPLen = Math.min(projA.prompt.length, projB.prompt.length);
    const minRLen = Math.min(projA.response.length, projB.response.length);
    return {
        prompt: projA.prompt.slice(0, minPLen).map((v, i) => v - projB.prompt[i]),
        response: projA.response.slice(0, minRLen).map((v, i) => v - projB.response[i])
    };
}

/**
 * Load comparison projections and apply diff/show to traitData in place.
 * Unified logic for both replay_suffix and standard comparison modes.
 *
 * @param {Object} traitData - Mutable trait data map
 * @param {string[]} traitKeys - Keys in traitData to compare
 * @param {string} promptSet - Prompt set for fetch path
 * @param {string} promptId - Prompt ID for fetch path
 * @param {string} variant - Model variant for fetch path
 * @param {Array} filteredTraits - Trait objects with .name
 * @param {'diff'|'show'} mode - How to apply: 'diff' computes delta, 'show' replaces
 * @param {boolean} invertDiff - If true, diff = main - comp (replay convention); if false, diff = comp - main (standard)
 * @param {string} compareLabel - Label for _compareModel metadata
 */
async function loadComparisonProjections(traitData, traitKeys, promptSet, promptId, variant, filteredTraits, mode, invertDiff, compareLabel) {
    const compResults = await Promise.all(traitKeys.map(async (traitKey) => {
        const baseTrait = traitData[traitKey].metadata?._baseTrait || traitKey;
        const trait = filteredTraits.find(t => t.name === baseTrait);
        if (!trait) return null;

        try {
            const fetchPath = window.paths.residualStreamData(trait, promptSet, promptId, variant);
            const response = await fetch(fetchPath);
            if (!response.ok) return null;
            const compData = await response.json();
            if (compData.error) return null;
            return { traitKey, compData };
        } catch (error) {
            return null;
        }
    }));

    for (const result of compResults) {
        if (!result) continue;
        const { traitKey, compData } = result;

        const vs = traitData[traitKey].metadata?.vector_source;
        const compProj = extractCompProjection(compData, vs);
        if (!compProj) continue;

        if (mode === 'diff') {
            const mainProj = traitData[traitKey].projections;
            // invertDiff: main - comp (replay: organism - instruct); otherwise comp - main
            const [a, b] = invertDiff ? [mainProj, compProj] : [compProj, mainProj];
            traitData[traitKey].projections = computeProjectionDiff(a, b, traitKey);
            traitData[traitKey].metadata = traitData[traitKey].metadata || {};
            traitData[traitKey].metadata._isDiff = true;
            traitData[traitKey].metadata._compareModel = compareLabel;
        } else if (mode === 'show') {
            traitData[traitKey].projections = compProj;
            traitData[traitKey].metadata = traitData[traitKey].metadata || {};
            traitData[traitKey].metadata._isComparisonModel = true;
            traitData[traitKey].metadata._compareModel = compareLabel;
        }
    }
}

// =============================================================================
// Layer sensitivity
// =============================================================================

/**
 * Fetch layer_sensitivity per-prompt data (multi-layer projections from model_diff analysis).
 * Returns null if not available. Caches fetched data.
 */
async function fetchLayerSensitivityData(experiment, promptSet, promptId) {
    // Determine variant pair from model-diff API (cached per experiment)
    if (!window._modelDiffCache || window._modelDiffCache.experiment !== experiment) {
        const data = await fetchJSON(`/api/experiments/${experiment}/model-diff`);
        window._modelDiffCache = { experiment, comparisons: data?.comparisons || [] };
    }
    if (window._modelDiffCache.comparisons.length === 0) return null;

    const comp = window._modelDiffCache.comparisons[0];
    const path = `experiments/${experiment}/model_diff/${comp.variant_pair}/layer_sensitivity/${promptSet}/per_prompt/${promptId}.json`;
    if (layerSensitivityCache[path]) return layerSensitivityCache[path];

    const data = await fetchJSON('/' + path);
    if (!data) return null;
    data._variant_a = comp.variant_a;
    data._variant_b = comp.variant_b;
    layerSensitivityCache[path] = data;
    return data;
}

// =============================================================================
// Multi-vector expansion / processTraitProjectionData
// =============================================================================

/**
 * Process projection results into traitData map, handling multi-vector expansion.
 * Merges response data (tokens) with projection data, expands multi-vector format
 * based on layer mode setting.
 *
 * @param {Array} projectionResults - Results from parallel fetch [{trait, projData} or {trait, error}]
 * @param {Object} responseData - Shared response data (tokens, prompt_end)
 * @param {boolean} isRollout - Whether response is empty (rollout mode)
 * @returns {{ traitData: Object, failedTraits: string[] }}
 */
function processTraitProjectionData(projectionResults, responseData) {
    const traitData = {};
    const failedTraits = [];

    for (const result of projectionResults) {
        if (result.error) {
            failedTraits.push(result.trait.name);
            continue;
        }
        const { trait, projData } = result;

        // Merge response data with projection data (projection is slim, needs tokens)
        if (responseData) {
            // Handle both new flat schema and old nested schema
            if (responseData.tokens && responseData.prompt_end !== undefined) {
                // New flat schema: convert to nested format expected by rest of code
                const promptEnd = responseData.prompt_end;
                projData.prompt = {
                    text: responseData.prompt || '',
                    tokens: responseData.tokens.slice(0, promptEnd)
                };
                projData.response = {
                    text: responseData.response || '',
                    tokens: responseData.tokens.slice(promptEnd)
                };
            } else {
                // Old nested schema (fallback)
                projData.prompt = responseData.prompt;
                projData.response = responseData.response;
            }
            if (responseData.metadata?.inference_model && !projData.metadata?.inference_model) {
                projData.metadata = projData.metadata || {};
                projData.metadata.inference_model = responseData.metadata.inference_model;
            }
        }

        // Handle multi-vector format
        if (projData.metadata?.multi_vector && Array.isArray(projData.projections)) {
            if (window.state.layerMode) {
                // Layers mode ON: expand all layers into separate entries
                for (const vecProj of projData.projections) {
                    const key = `${trait.name}__${vecProj.method}_L${vecProj.layer}`;
                    traitData[key] = {
                        ...projData,
                        projections: { prompt: vecProj.prompt, response: vecProj.response },
                        token_norms: vecProj.token_norms || projData.token_norms || null,
                        metadata: {
                            ...projData.metadata,
                            vector_source: {
                                layer: vecProj.layer,
                                method: vecProj.method,
                                selection_source: vecProj.selection_source,
                                baseline: vecProj.baseline
                            },
                            _baseTrait: trait.name,
                            _isMultiVector: true
                        }
                    };
                }
            } else {
                // Layers mode OFF: pick single best layer per trait
                // Target: best_steering_layer + floor(0.1 * num_layers), snap to closest available
                const numLayers = window.paths?.getNumLayers?.() || 48;
                const offset = Math.floor(0.1 * numLayers);
                const steeringEntry = projData.projections.find(p => p.selection_source === 'steering');
                let targetLayer;
                if (steeringEntry) {
                    targetLayer = steeringEntry.layer + offset;
                } else {
                    // Fallback: 60% depth (roughly where best+10% lands for most traits)
                    targetLayer = Math.floor(0.6 * numLayers);
                }
                // Snap to closest available layer
                const vecProj = projData.projections.reduce((best, p) =>
                    Math.abs(p.layer - targetLayer) < Math.abs(best.layer - targetLayer) ? p : best
                );
                traitData[trait.name] = {
                    ...projData,
                    projections: { prompt: vecProj.prompt, response: vecProj.response },
                    token_norms: vecProj.token_norms || projData.token_norms || null,
                    metadata: {
                        ...projData.metadata,
                        vector_source: {
                            layer: vecProj.layer,
                            method: vecProj.method,
                            selection_source: vecProj.selection_source,
                            baseline: vecProj.baseline,
                            position: projData.metadata?.position
                        }
                    }
                };
            }
        } else {
            traitData[trait.name] = projData;
        }
    }

    return { traitData, failedTraits };
}

export {
    extractCompProjection,
    computeProjectionDiff,
    loadComparisonProjections,
    fetchLayerSensitivityData,
    processTraitProjectionData
};
