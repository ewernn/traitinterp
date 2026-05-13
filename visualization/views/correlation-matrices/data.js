// Data layer for the correlation-matrices view.
//
// Input:  /dev/conv_tools/correlation_sweep/{index.json, configs/cfg_NNN.json}
// Output: cached fetch promises for the index + per-config matrices.
// Usage:  view.js -> loadIndex(), loadConfig(cfgId)
//
// We never load all 144 configs upfront; the sidebar uses index.json (lightweight
// per-config metadata) and each cfg_NNN.json is fetched lazily on selection and
// memoized.

const ROOT = '/dev/conv_tools/correlation_sweep';

let _indexPromise = null;
const _configPromises = new Map();  // cfgId -> Promise<config json>

async function _fetchJson(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${url} -> HTTP ${r.status}`);
    return r.json();
}

function loadIndex() {
    if (!_indexPromise) {
        _indexPromise = _fetchJson(`${ROOT}/index.json`);
    }
    return _indexPromise;
}

function loadConfig(cfgId) {
    if (!_configPromises.has(cfgId)) {
        const padded = String(cfgId).padStart(3, '0');
        _configPromises.set(cfgId, _fetchJson(`${ROOT}/configs/cfg_${padded}.json`));
    }
    return _configPromises.get(cfgId);
}

// Pervasive-scope biases — no single onset; activation fires throughout the
// response. ALWAYS excluded from the heatmap, top-pairs list, and stats.
// Mirror of dev/conv_tools/bias_correlation_sweep.py PERVASIVE_SCOPE_BIAS_IDS.
const PERVASIVE_SCOPE_BIAS_IDS = new Set([12, 13, 14, 17, 19, 20, 22, 23, 24]);

/**
 * Filter bias_ids to remove pervasive-scope ones. Used everywhere downstream.
 */
function filterPervasive(biasIds) {
    return biasIds.filter(b => !PERVASIVE_SCOPE_BIAS_IDS.has(Number(b)));
}

/**
 * Compute summary stats for one matrix:
 *   - off-diagonal mean / std
 *   - top-K cross-bias (off-diagonal) pairs by absolute value
 *
 * Matrix is a nested object indexed by string bias-id: matrix[A][B].
 */
function summarizeMatrix(matrix, biasIds, topK = 5) {
    const offDiag = [];
    const pairs = [];
    for (const a of biasIds) {
        const row = matrix[String(a)];
        if (!row) continue;
        for (const b of biasIds) {
            if (a === b) continue;
            const v = row[String(b)];
            if (typeof v !== 'number') continue;
            offDiag.push(v);
            pairs.push({ a, b, value: v });
        }
    }
    const n = offDiag.length;
    const mean = n > 0 ? offDiag.reduce((s, v) => s + v, 0) / n : 0;
    const variance = n > 1
        ? offDiag.reduce((s, v) => s + (v - mean) ** 2, 0) / (n - 1)
        : 0;
    const std = Math.sqrt(variance);
    const absMax = offDiag.reduce((m, v) => Math.max(m, Math.abs(v)), 0);

    // Top-K by absolute value, signed value preserved
    pairs.sort((p, q) => Math.abs(q.value) - Math.abs(p.value));
    const top = pairs.slice(0, topK);

    return { mean, std, absMax, n, top };
}

export { loadIndex, loadConfig, summarizeMatrix, filterPervasive, PERVASIVE_SCOPE_BIAS_IDS };
