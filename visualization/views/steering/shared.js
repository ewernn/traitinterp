// Steering shared utilities — helpers used across the steering sub-modules
//
// Deduplicates vector spec extraction and metric parsing from steering runs.

/**
 * Extract vector spec from a steering run.
 * Returns null for multi-vector runs (only single-vector runs are supported).
 */
function extractVectorSpec(run) {
    const vectors = run.config?.vectors || [];
    if (vectors.length !== 1) return null;
    const v = vectors[0];
    return { layer: v.layer, method: v.method, component: v.component || 'residual', coef: v.weight };
}

/**
 * Extract scoring metrics from a steering run result.
 * @param {Object} result - run.result object with trait_mean, coherence_mean
 * @param {number} baseline - baseline trait score
 * @returns {{ traitScore: number, coherence: number, delta: number }}
 */
function extractRunMetrics(result, baseline) {
    const traitScore = result.trait_mean || 0;
    const coherence = result.coherence_mean || 0;
    const delta = traitScore - baseline;
    const traitStd = result.trait_std || 0;
    const n = result.n || 0;
    // Standard error of the mean for the n-question rollout
    const traitSE = n > 0 ? traitStd / Math.sqrt(n) : 0;
    return { traitScore, coherence, delta, traitStd, traitSE, n };
}

export { extractVectorSpec, extractRunMetrics };
