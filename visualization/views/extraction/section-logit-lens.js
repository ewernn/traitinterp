/**
 * Logit Lens — top vocabulary tokens each vector points toward / away from.
 *
 * Loads vector → token decode data once into the shared cache, then re-renders
 * from cache when Vector Geometry changes the chosen (method, layer).
 *
 * Input:  evalData (for trait list + model variant)
 * Output: rendered table in #logit-lens-container
 * Usage:  import { renderLogitLensSection, renderLogitLensFromCache } from './section-logit-lens.js';
 */

import { fetchJSON, escapeHtml } from '../../core/utils.js';
import { getDisplayName, displayLayer } from '../../core/display.js';
import { renderRunHint } from '../../core/ui.js';
import { extractionState } from './extraction-data.js';

/** Load logit lens data for all traits once, then render from cache. */
async function renderLogitLensSection(evalData) {
    const container = document.getElementById('logit-lens-container');
    if (!container) return;

    const allResults = evalData.all_results || [];
    const traits = [...new Set(allResults.map(r => r.trait))].sort();

    // Get model variant from eval data (extraction model variant)
    const modelVariant = evalData.model_variant || 'base';

    if (traits.length === 0) {
        container.innerHTML = '<p class="na">No traits available.</p>';
        return;
    }

    container.innerHTML = '<p class="hint">Loading token decodes...</p>';

    // Load all logit lens data in parallel (cache for subsequent layer changes)
    const results = await Promise.all(traits.map(async trait => {
        const data = await fetchJSON(window.paths.logitLens(trait, modelVariant));
        return { trait, data };
    }));

    extractionState.logitLensCache = Object.fromEntries(
        results.filter(r => r.data).map(r => [r.trait, r.data])
    );
    extractionState.logitLensEvalData = evalData;
    renderLogitLensFromCache();
}

/**
 * Render the logit-lens table from cache, using vgLayer when available so the
 * table's layer selection tracks the Vector Geometry slider. Falls back to the
 * middle-late heuristic for traits whose per_layer doesn't include vgLayer, or
 * when the file uses the older `late` schema (backend-baked single layer).
 */
function renderLogitLensFromCache() {
    const container = document.getElementById('logit-lens-container');
    if (!container || !extractionState.logitLensCache) return;

    const cachedTraits = Object.entries(extractionState.logitLensCache);
    if (cachedTraits.length === 0) {
        const expName = window.state.experimentData?.name || '<exp>';
        container.innerHTML = renderRunHint(
            'No logit lens data.',
            `python analysis/vectors/logit_lens.py --experiment ${expName} --all-traits --save`
        );
        return;
    }

    const renderTokens = (tokens, limit = 5) => {
        if (!tokens || !Array.isArray(tokens)) return '<span class="na">—</span>';
        return tokens.slice(0, limit)
            .map(t => `<span class="ll-token">${escapeHtml(t.token)}</span>`)
            .join(' ');
    };

    // Pick layer: prefer vgLayer (sync with Vector Geometry). Fall back to the
    // middle-late heuristic (n_layers/2 + 10) only if vgLayer isn't in the set.
    const pickDisplayLayer = (layerNums, nLayers) => {
        if (!layerNums.length) return null;
        const vg = extractionState.vgLayer;
        if (vg != null && layerNums.includes(vg)) return vg;
        const target = Math.floor(nLayers / 2) + 10;
        return layerNums.reduce((best, L) => Math.abs(L - target) < Math.abs(best - target) ? L : best, layerNums[0]);
    };

    let html = `
        <table class="data-table ll-table">
            <thead>
                <tr>
                    <th>Trait</th>
                    <th>Layer</th>
                    <th>→ Toward</th>
                    <th>← Away</th>
                </tr>
            </thead>
            <tbody>
    `;

    for (const [trait, data] of cachedTraits) {
        // Method preference: match Vector Geometry's selection if available, otherwise probe > mean_diff > gradient.
        const methodPriority = [extractionState.vgMethod, 'probe', 'mean_diff', 'gradient'].filter(Boolean);
        const method = methodPriority.find(m => data.methods[m]) || Object.keys(data.methods)[0];
        const methodData = data.methods[method];
        if (!methodData) continue;

        // Handle both logit-lens schemas: `per_layer: {L: {...}}` (newer) and `late: {...}` (older).
        let chosen;
        if (methodData.per_layer) {
            const layerNums = Object.keys(methodData.per_layer).map(Number);
            const pick = pickDisplayLayer(layerNums, data.n_layers || layerNums.length);
            chosen = methodData.per_layer[pick];
        } else if (methodData.late) {
            chosen = methodData.late;
        } else {
            continue;
        }
        if (!chosen) continue;

        const displayName = getDisplayName(trait);

        html += `
            <tr>
                <td><strong>${displayName}</strong><br><span class="hint">${method}</span></td>
                <td class="hint">L${displayLayer(chosen.layer)}<br><span class="hint">${chosen.pct}%</span></td>
                <td class="ll-toward">${renderTokens(chosen.toward)}</td>
                <td class="ll-away">${renderTokens(chosen.away)}</td>
            </tr>
        `;
    }

    html += '</tbody></table>';
    container.innerHTML = html;
}

export { renderLogitLensSection, renderLogitLensFromCache };
