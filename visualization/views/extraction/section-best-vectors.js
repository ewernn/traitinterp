/**
 * Best Vectors Summary — single table, one row per trait.
 *
 * Input:  evalData.all_results (extraction evaluation per layer/method)
 * Output: rendered table inside #best-vectors-summary-container
 * Usage:  import { renderBestVectorsSummary } from './section-best-vectors.js';
 */

import { getDisplayName, displayLayer } from '../../core/display.js';
import { computeBestVectors, getSelectedTraitNames } from './extraction-data.js';

/** Render best vectors summary table — one row per trait with key metrics. */
function renderBestVectorsSummary(evalData) {
    const container = document.getElementById('best-vectors-summary-container');
    if (!container) return;

    const allResults = evalData.all_results || [];
    const bestVectors = computeBestVectors(allResults);

    if (Object.keys(bestVectors).length === 0) {
        container.innerHTML = '<p>No extraction results available.</p>';
        return;
    }

    // Filter by selected traits from sidebar
    const selectedTraitNames = getSelectedTraitNames();
    const traits = selectedTraitNames.size > 0
        ? Object.keys(bestVectors).filter(t => selectedTraitNames.has(t))
        : Object.keys(bestVectors);

    // Build rows with metrics from best vector
    const rows = traits.map(trait => {
        const best = bestVectors[trait];
        const result = allResults.find(r =>
            r.trait === trait && r.method === best.method && r.layer === best.layer
        );

        return {
            trait: getDisplayName(trait),
            method: best.method,
            layer: best.layer,
            accuracy: result?.val_accuracy ?? null,
            effectSize: result?.val_effect_size ?? null
        };
    }).sort((a, b) => a.trait.localeCompare(b.trait));

    let html = `
        <table class="data-table best-vectors-table">
            <thead>
                <tr>
                    <th>Trait</th>
                    <th>Best Method</th>
                    <th>Layer</th>
                    <th>Val Accuracy</th>
                    <th>Effect Size (d)</th>
                </tr>
            </thead>
            <tbody>
    `;

    rows.forEach(row => {
        html += `
            <tr>
                <td><strong>${row.trait}</strong></td>
                <td>${row.method}</td>
                <td>L${displayLayer(row.layer)}</td>
                <td>${row.accuracy !== null ? (row.accuracy * 100).toFixed(1) + '%' : 'N/A'}</td>
                <td>${row.effectSize !== null ? row.effectSize.toFixed(2) : 'N/A'}</td>
            </tr>
        `;
    });

    html += `
            </tbody>
        </table>
    `;

    container.innerHTML = html;
}

export { renderBestVectorsSummary };
