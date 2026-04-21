/**
 * Variant Comparison section — Cohen's d per layer + cosine similarity vs trait directions.
 *
 * Independent of activation diagnostics. Reads pre-computed model_diff results
 * from compare_variants.py.
 *
 * Input:  experiment name
 * Output: variant comparison table + 2 layer charts in #model-diff-container
 * Usage:  import { renderModelDiffComparison } from './section-variant-comparison.js';
 */

import { fetchJSON } from '../../core/utils.js';
import { displayLayer, getChartColors } from '../../core/display.js';
import { buildChartLayout, renderChart } from '../../core/charts.js';
import { renderSubsection, renderRunHint } from '../../core/ui.js';

/**
 * Render model diff comparison using pre-computed results from compare_variants.py
 */
async function renderModelDiffComparison(experiment) {
    const container = document.getElementById('model-diff-container');
    if (!container) return;

    try {
        const data = await fetchJSON(`/api/experiments/${experiment}/model-diff`);
        const comparisons = data?.comparisons || [];

        if (comparisons.length === 0) {
            container.innerHTML = renderRunHint(
                'No model diff data available.',
                `python analysis/model_diff/compare_variants.py --experiment ${experiment} --variant-a <variant_a> --variant-b <variant_b> --prompt-set <prompt_set>`
            );
            return;
        }

        // Use the first comparison (typically only one)
        const comparison = comparisons[0];
        const { variant_a, variant_b, prompt_sets } = comparison;

        // Load results for all prompt sets
        const allResults = {};
        for (const promptSet of prompt_sets) {
            const resultsPath = window.paths.get('model_diff.results', {
                variant_a: comparison.variant_a,
                variant_b: comparison.variant_b,
                prompt_set: promptSet
            });
            const result = await fetchJSON('/' + resultsPath);
            if (result) allResults[promptSet] = result;
        }

        if (Object.keys(allResults).length === 0) {
            container.innerHTML = `<div class="info">Failed to load model diff results.</div>`;
            return;
        }

        // Build summary table
        const summaryRows = [];
        const allTraits = new Set();
        for (const results of Object.values(allResults)) {
            for (const trait of Object.keys(results.traits || {})) {
                allTraits.add(trait);
            }
        }

        for (const trait of allTraits) {
            const row = { trait: trait.split('/').pop() };
            for (const results of Object.values(allResults)) {
                const traitData = results.traits?.[trait];
                if (traitData?.method) {
                    row.method = traitData.method;
                    break;
                }
            }
            for (const [promptSet, results] of Object.entries(allResults)) {
                const traitData = results.traits?.[trait];
                if (traitData) {
                    const setName = promptSet.split('/').pop();
                    const peakIdx = traitData.layers?.indexOf(traitData.peak_layer);
                    const stdA = peakIdx >= 0 ? traitData.per_layer_std_a?.[peakIdx] : null;
                    const stdB = peakIdx >= 0 ? traitData.per_layer_std_b?.[peakIdx] : null;
                    row[setName] = {
                        peak_layer: traitData.peak_layer,
                        peak_effect: traitData.peak_effect_size,
                        std_a: stdA,
                        std_b: stdB
                    };
                }
            }
            summaryRows.push(row);
        }

        const promptSetNames = prompt_sets.map(ps => ps.split('/').pop());
        container.innerHTML = `
            <div class="model-diff-header">
                <strong>${variant_b}</strong> vs <strong>${variant_a}</strong>
                <span style="color: var(--text-tertiary); margin-left: 8px;">(${Object.values(allResults)[0]?.n_prompts || '?'} prompts)</span>
                <div class="model-diff-legend">positive = ${variant_b} higher than ${variant_a}</div>
            </div>

            <table class="data-table">
                <thead>
                    <tr>
                        <th>Trait</th>
                        <th>Method</th>
                        ${promptSetNames.map(ps => `<th>${ps}</th><th>Spread (A / B)</th>`).join('')}
                    </tr>
                </thead>
                <tbody>
                    ${summaryRows.map(row => `
                        <tr>
                            <td>${row.trait}</td>
                            <td style="color: var(--text-secondary);">${row.method || '?'}</td>
                            ${promptSetNames.map(ps => {
                                const data = row[ps];
                                if (data) {
                                    const color = data.peak_effect > 1.5 ? 'var(--success-color)' :
                                                  data.peak_effect > 0.5 ? 'var(--warning-color)' :
                                                  'var(--text-secondary)';
                                    const effectCell = `<td style="color: ${color};">${data.peak_effect.toFixed(2)}σ @ L${displayLayer(data.peak_layer)}</td>`;
                                    const spreadCell = data.std_a != null && data.std_b != null
                                        ? `<td style="color: var(--text-secondary);">${data.std_a.toFixed(2)} / ${data.std_b.toFixed(2)}</td>`
                                        : '<td style="color: var(--text-tertiary);">—</td>';
                                    return effectCell + spreadCell;
                                }
                                return '<td>—</td><td>—</td>';
                            }).join('')}
                        </tr>
                    `).join('')}
                </tbody>
            </table>

            ${renderSubsection({
                title: 'Effect Size by Layer',
                infoId: 'info-effect-size',
                infoText: 'Cohen&#39;s d between variants per layer, one line per trait × prompt set. Peaks mark the layer that best separates A from B on that trait.',
                level: 'h4'
            })}
            <div id="model-diff-chart"></div>

            ${renderSubsection({
                title: 'Cosine Similarity with Trait Direction',
                infoId: 'info-cosine-sim',
                infoText: 'Cosine between the mean (B − A) diff vector and the trait vector per layer. Tells you whether the variant shift points along the trait, not just how big it is.',
                level: 'h4'
            })}
            <div id="model-diff-cosine-chart"></div>
        `;

        window.setupSubsectionInfoToggles?.();

        renderModelDiffLayerChart(allResults, comparison, {
            field: 'per_layer_effect_size',
            divId: 'model-diff-chart',
            yaxisTitle: 'Effect Size (σ)',
            hoverFormat: '%{y:.2f}σ',
            height: 400,
            margin: { t: 40 },
            title: `Trait Detection: ${comparison.variant_b} vs ${comparison.variant_a}`
        });
        renderModelDiffLayerChart(allResults, comparison, {
            field: 'per_layer_cosine_sim',
            divId: 'model-diff-cosine-chart',
            yaxisTitle: 'Cosine Similarity',
            hoverFormat: '%{y:.3f}',
            height: 300,
            margin: { t: 10 },
            yaxisRange: [-0.15, 0.15],
            peakByAbsValue: true
        });

    } catch (error) {
        container.innerHTML = `<div class="info">Error loading model diff data: ${error.message}</div>`;
    }
}

/**
 * Render a model diff layer chart (effect size or cosine similarity).
 */
function renderModelDiffLayerChart(allResults, comparison, {
    field, divId, yaxisTitle, hoverFormat,
    height = 300, margin = {}, title, yaxisRange, peakByAbsValue = false
}) {
    const chartDiv = document.getElementById(divId);
    if (!chartDiv) return;

    const colors = getChartColors();
    const traces = [];
    let colorIdx = 0;

    const allTraits = new Set();
    for (const results of Object.values(allResults)) {
        for (const trait of Object.keys(results.traits || {})) {
            allTraits.add(trait);
        }
    }

    for (const trait of allTraits) {
        const traitName = trait.split('/').pop();
        const color = colors[colorIdx % colors.length];
        let dashIdx = 0;

        for (const [promptSet, results] of Object.entries(allResults)) {
            const traitData = results.traits?.[trait];
            if (!traitData || !traitData[field]) continue;

            const values = traitData[field];
            const setName = promptSet.split('/').pop();
            const dash = dashIdx === 0 ? 'solid' : 'dash';

            let peakVal, peakLayer;
            if (peakByAbsValue) {
                const peakIdx = values.reduce((maxIdx, val, idx, arr) =>
                    Math.abs(val) > Math.abs(arr[maxIdx]) ? idx : maxIdx, 0);
                peakVal = values[peakIdx];
                peakLayer = traitData.layers[peakIdx];
            } else {
                peakVal = traitData.peak_effect_size;
                peakLayer = traitData.peak_layer;
            }
            const peakLabel = peakByAbsValue ? peakVal.toFixed(2) : `${peakVal.toFixed(2)}σ`;

            traces.push({
                x: traitData.layers,
                y: values,
                type: 'scatter',
                mode: 'lines+markers',
                name: `${traitName} ${setName} (peak: ${peakLabel} @ L${peakLayer})`,
                line: { color, width: 2, dash },
                marker: { size: 3 },
                hovertemplate: `${traitName} ${setName}<br>L%{x}: ${hoverFormat}<extra></extra>`
            });

            dashIdx++;
        }
        colorIdx++;
    }

    const yaxis = {
        title: yaxisTitle,
        zeroline: true,
        zerolinewidth: 1,
        showgrid: true,
        ...(yaxisRange ? { range: yaxisRange } : {})
    };

    const layout = buildChartLayout({
        preset: 'layerChart',
        traces,
        height,
        legendPosition: 'below',
        margin,
        xaxis: { title: 'Layer', dtick: 10, showgrid: true },
        yaxis,
        hovermode: 'closest',
        ...(title ? { title } : {})
    });

    renderChart(chartDiv, traces, layout, { displayModeBar: true });
}

export { renderModelDiffComparison };
