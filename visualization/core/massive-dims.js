/**
 * Massive-dim cleaning for trait projections.
 *
 * Removes outlier-dimension contributions (Sun et al. 2024) from per-token
 * projection scores and from response-token L2 norms, so that downstream
 * normalization (cosine, response-mean-norm) reflects the "clean" residual
 * stream rather than the few dimensions that dominate raw activation magnitude.
 *
 * Input: per-experiment massiveDimData (loaded from massive-dim calibration JSON)
 * Output: cleaned projection arrays / norm arrays
 * Usage: import { getDimsToRemove, applyMassiveDimCleaning, computeCleanedNorms }
 *        from '../core/massive-dims.js';
 */

/**
 * Get list of dimensions to remove based on cleaning mode.
 *
 * Modes:
 * - 'top5-3layers': Dims in top-5 at 3+ layers - balanced (recommended)
 * - 'all': All candidate dims - most aggressive
 */
function getDimsToRemove(massiveDimData, cleaningMode) {
    const { dims, top_dims_by_layer } = massiveDimData || {};

    if (!dims) return [];

    if (cleaningMode === 'all') {
        return dims;
    }

    if (cleaningMode === 'top5-3layers' && top_dims_by_layer) {
        // Count appearances: dims that appear in top-5 at 3+ layers
        const appearances = {};
        for (const layerDims of Object.values(top_dims_by_layer)) {
            const top5 = layerDims.slice(0, 5);
            for (const dim of top5) {
                appearances[dim] = (appearances[dim] || 0) + 1;
            }
        }
        return Object.entries(appearances)
            .filter(([_, count]) => count >= 3)
            .map(([dim, _]) => parseInt(dim));
    }

    return [];
}

/**
 * Apply massive dim cleaning to projections.
 * Formula: adjusted = original - sum(act[dim] * vec[dim]) / ||vec||
 */
function applyMassiveDimCleaning(projections, massiveDimData, dimsToRemove, phase) {
    const { vec_norm, vec_components, activation_values } = massiveDimData;
    const phaseActValues = activation_values[phase];

    if (!phaseActValues || !vec_norm) {
        return projections;
    }

    return projections.map((proj, tokenIdx) => {
        let adjustment = 0;
        for (const dim of dimsToRemove) {
            const actVal = phaseActValues[dim]?.[tokenIdx] ?? 0;
            const vecComp = vec_components[dim] ?? 0;
            adjustment += actVal * vecComp;
        }
        return proj - adjustment / vec_norm;
    });
}

/**
 * Compute cleaned token norms: ||h_cleaned|| = sqrt(||h||^2 - sum h[dim]^2)
 */
function computeCleanedNorms(originalNorms, massiveDimData, dimsToRemove, phase) {
    const phaseActValues = massiveDimData?.activation_values?.[phase];
    if (!phaseActValues || dimsToRemove.length === 0) {
        return originalNorms;
    }

    return originalNorms.map((norm, tokenIdx) => {
        const normSquared = norm * norm;
        let massiveContribution = 0;
        for (const dim of dimsToRemove) {
            const actVal = phaseActValues[dim]?.[tokenIdx] ?? 0;
            massiveContribution += actVal * actVal;
        }
        const cleanedSquared = normSquared - massiveContribution;
        return cleanedSquared > 0 ? Math.sqrt(cleanedSquared) : 0;
    });
}

export { getDimsToRemove, applyMassiveDimCleaning, computeCleanedNorms };
