/**
 * Activation Diagnostics section — magnitude, uniformity, massive dims, inter-layer similarity.
 *
 * Owns the section's HTML template (with criteria dropdown wiring) and
 * the four chart renderers. Variant dropdown is wired in model-analysis-data.js.
 *
 * Input:  pre-fetched calibration data (from fetchMassiveActivationsData)
 * Output: rendered Plotly charts in section containers
 * Usage:  import { renderDiagnosticsSectionHtml, renderAllDiagnostics } from './section-diagnostics.js';
 */

import { sortedNumericKeys } from '../../core/utils.js';
import { getChartColors } from '../../core/display.js';
import { buildChartLayout, renderChart } from '../../core/charts.js';
import { renderRunHint, renderSubsection } from '../../core/ui.js';
import { renderStyledSelect } from '../../components/styled-select.js';
import {
    withMassiveActivationsData,
    getMaCriteria,
    setMaCriteria,
    fetchMassiveActivationsData,
} from './model-analysis-data.js';

/**
 * HTML for the Activation Diagnostics section. Includes the criteria dropdown,
 * which has an inline onChange that re-renders the massive-dims chart.
 */
function renderDiagnosticsSectionHtml() {
    return `
        <section>
            ${renderSubsection({
                num: 1,
                title: 'Activation Diagnostics',
                infoId: 'info-activation-diagnostics',
                infoText: 'Internals of one model variant from calibration prompts: norm growth, massive dimensions, token uniformity, and layer-to-layer change.'
            })}

            <div class="projection-toggle">
                <span class="projection-toggle-label">Model Variant:</span>
                <div id="activation-diagnostics-variant-container"><!-- Populated dynamically --></div>
            </div>

            ${renderSubsection({
                title: 'Activation Magnitude by Layer',
                infoId: 'info-act-magnitude',
                infoText: 'L2 norm of the residual stream at each layer, plus per-layer attention and MLP contribution norms. Shows where the stream grows and why.',
                level: 'h4'
            })}
            <div id="activation-magnitude-plot"></div>

            ${renderSubsection({
                title: 'Activation Uniformity',
                infoId: 'info-massive-acts',
                infoText: 'Mean cosine similarity of each token to the layer&#39;s mean direction. High = all tokens point the same way, a sign of massive-dim dominance.',
                level: 'h4'
            })}
            <div id="massive-activations-container"></div>

            ${renderSubsection({
                title: 'Massive Dims Across Layers',
                infoId: 'info-massive-dims-layers',
                infoText: 'Per-dimension normalized magnitude <code>|h[l][d]| / mean|h|</code> across layers. Lines ≫ 1 are massive dims (Sun et al. 2024) — near-constant biases that hurt trait projection signal. Criteria dropdown controls how strict "massive" is.',
                level: 'h4'
            })}
            <div class="projection-toggle">
                <span class="projection-toggle-label">Criteria:</span>
                ${renderStyledSelect({
                    id: 'massive-dims-criteria',
                    options: [
                        { value: 'top5-3layers', label: 'Top 5, 3+ layers' },
                        { value: 'top3-any', label: 'Top 3, any layer' },
                        { value: 'top5-any', label: 'Top 5, any layer' },
                    ],
                    selected: getMaCriteria(),
                    onChange: async (val) => {
                        setMaCriteria(val);
                        const freshData = await fetchMassiveActivationsData();
                        renderMassiveDimsAcrossLayers(freshData);
                    },
                })}
            </div>
            <div id="massive-dims-layers-plot"></div>

            ${renderSubsection({
                title: 'Inter-Layer Similarity',
                infoId: 'info-interlayer-sim',
                infoText: 'Cosine between consecutive mean-layer vectors. Dips mark layers where the representation gets substantially rewritten.',
                level: 'h4'
            })}
            <div id="interlayer-similarity-plot"></div>
        </section>
    `;
}

/** Render Activation Magnitude plot showing ||h|| by layer. */
function renderActivationMagnitudePlot(data) {
    withMassiveActivationsData('activation-magnitude-plot', data, (plotDiv, data) => {
        if (!data.aggregate?.layer_norms) {
            plotDiv.innerHTML = renderRunHint(
                'Activation magnitude data not available.',
                `python inference/run_inference_pipeline.py --experiment ${window.paths.getExperiment()} --prompt-set starter_prompts/general   # captures automatically`
            );
            return;
        }

        const layerNorms = data.aggregate.layer_norms;
        const attnNorms = data.aggregate.attn_norms || {};
        const mlpNorms = data.aggregate.mlp_norms || {};
        const layers = sortedNumericKeys(layerNorms);
        const norms = layers.map(l => layerNorms[l]);
        const attn = layers.map(l => attnNorms[l] || null);
        const mlp = layers.map(l => mlpNorms[l] || null);

        const modelInfo = data.model ? `<div class="model-label">Model: <code>${data.model}</code></div>` : '';
        plotDiv.innerHTML = modelInfo;
        const chartDiv = document.createElement('div');
        plotDiv.appendChild(chartDiv);

        const colors = getChartColors();
        const traces = [{
            x: layers,
            y: norms,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Residual ||h||',
            line: { color: colors[0], width: 2 },
            marker: { size: 4 },
            hovertemplate: '<b>Layer %{x}</b><br>||h|| = %{y:.1f}<extra></extra>'
        }];

        if (attn.some(v => v !== null)) {
            traces.push({
                x: layers,
                y: attn,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Attn contrib',
                line: { color: colors[1], width: 2 },
                marker: { size: 4 },
                hovertemplate: '<b>Layer %{x}</b><br>||attn|| = %{y:.1f}<extra></extra>'
            });
        }

        if (mlp.some(v => v !== null)) {
            traces.push({
                x: layers,
                y: mlp,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'MLP contrib',
                line: { color: colors[2], width: 2 },
                marker: { size: 4 },
                hovertemplate: '<b>Layer %{x}</b><br>||mlp|| = %{y:.1f}<extra></extra>'
            });
        }

        const hasContribData = traces.length > 1;
        const layout = buildChartLayout({
            preset: 'layerChart',
            traces,
            height: 250,
            legendPosition: hasContribData ? 'right' : 'none',
            xaxis: { title: 'Layer', tickmode: 'linear', tick0: 0, dtick: 5, showgrid: true },
            yaxis: { title: 'L2 norm', showgrid: true }
        });
        renderChart(chartDiv, traces, layout);
    });
}

/** Render Massive Activations section — mean alignment plot. */
function renderMassiveActivations(data) {
    withMassiveActivationsData('massive-activations-container', data, (container, data) => {
        const aggregate = data.aggregate || {};
        const meanAlignment = aggregate.mean_alignment_by_layer || {};

        if (Object.keys(meanAlignment).length === 0) {
            container.innerHTML = `<div class="info">No mean alignment data available.</div>`;
            return;
        }

        container.innerHTML = `<div id="mean-alignment-plot"></div>`;

        const layers = sortedNumericKeys(meanAlignment);
        const alignments = layers.map(l => meanAlignment[l]);

        const alignTrace = {
            x: layers,
            y: alignments.map(v => v * 100),
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Mean Alignment',
            line: { color: getChartColors()[0], width: 2 },
            marker: { size: 4 },
            hovertemplate: 'L%{x}<br>Alignment: %{y:.1f}%<extra></extra>'
        };

        const alignLayout = buildChartLayout({
            preset: 'layerChart',
            traces: [alignTrace],
            height: 200,
            legendPosition: 'none',
            xaxis: { title: 'Layer', dtick: 5, showgrid: true },
            yaxis: { title: 'Mean Alignment (%)', range: [0, 100], showgrid: true }
        });
        renderChart('mean-alignment-plot', [alignTrace], alignLayout);
    });
}

/** Filter dims based on selected criteria. */
function filterDimsByCriteria(topDimsByLayer, criteria) {
    const dimAppearances = {};

    for (const [layer, dims] of Object.entries(topDimsByLayer)) {
        const topK = criteria === 'top3-any' ? 3 : 5;
        const dimsToCount = dims.slice(0, topK);
        for (const dim of dimsToCount) {
            dimAppearances[dim] = (dimAppearances[dim] || 0) + 1;
        }
    }

    const minLayers = criteria === 'top5-3layers' ? 3 : 1;
    return Object.entries(dimAppearances)
        .filter(([dim, count]) => count >= minLayers)
        .map(([dim]) => parseInt(dim))
        .sort((a, b) => a - b);
}

/** Render Massive Dims Across Layers plot. */
function renderMassiveDimsAcrossLayers(data) {
    withMassiveActivationsData('massive-dims-layers-plot', data, (container, data) => {
        const aggregate = data.aggregate || {};
        const topDimsByLayer = aggregate.top_dims_by_layer || {};
        const dimMagnitude = aggregate.dim_magnitude_by_layer || {};

        if (Object.keys(dimMagnitude).length === 0) {
            container.innerHTML = `<div class="info">No per-layer magnitude data. Run inference — it captures automatically: <code>python inference/run_inference_pipeline.py --experiment ${window.paths.getExperiment()} --prompt-set starter_prompts/general</code></div>`;
            return;
        }

        const criteria = getMaCriteria();
        const filteredDims = filterDimsByCriteria(topDimsByLayer, criteria);

        if (filteredDims.length === 0) {
            container.innerHTML = `<div class="info">No dims match criteria "${criteria}".</div>`;
            return;
        }

        const modelInfo = data.model ? `<div class="model-label">Model: <code>${data.model}</code></div>` : '';
        container.innerHTML = modelInfo;
        const chartDiv = document.createElement('div');
        container.appendChild(chartDiv);

        const colors = getChartColors();
        const nLayers = Object.keys(topDimsByLayer).length;
        const layers = Array.from({ length: nLayers }, (_, i) => i);

        const traces = filteredDims.map((dim, idx) => {
            const magnitudes = dimMagnitude[dim] || [];
            return {
                x: layers,
                y: magnitudes,
                type: 'scatter',
                mode: 'lines+markers',
                name: `dim ${dim}`,
                line: { color: colors[idx % colors.length], width: 2 },
                marker: { size: 4 },
                hovertemplate: `dim ${dim}<br>L%{x}<br>Normalized: %{y:.2f}x<extra></extra>`
            };
        });

        const layout = buildChartLayout({
            preset: 'layerChart',
            traces,
            height: 300,
            legendPosition: 'above',
            xaxis: { title: 'Layer', dtick: 5, showgrid: true },
            yaxis: { title: 'Normalized Magnitude', showgrid: true }
        });
        renderChart(chartDiv, traces, layout);
    });
}

/** Render inter-layer similarity plot: cos(mean[l], mean[l+1]). */
function renderInterLayerSimilarity(data) {
    withMassiveActivationsData('interlayer-similarity-plot', data, (plotDiv, data) => {
        if (!data.aggregate?.consecutive_cosine) {
            plotDiv.innerHTML = renderRunHint(
                'Inter-layer similarity data not available.',
                `python inference/run_inference_pipeline.py --experiment ${window.paths.getExperiment()} --prompt-set starter_prompts/general   # captures automatically`
            );
            return;
        }

        const consecutiveCosine = data.aggregate.consecutive_cosine;
        const layers = sortedNumericKeys(consecutiveCosine);
        const similarities = layers.map(l => consecutiveCosine[l]);

        const colors = getChartColors();
        const traces = [{
            x: layers,
            y: similarities,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'cos(L, L+1)',
            line: { color: colors[0], width: 2 },
            marker: { size: 4 },
            hovertemplate: 'L%{x} → L%{customdata}<br>cos = %{y:.4f}<extra></extra>',
            customdata: layers.map(l => l + 1)
        }];

        const layout = buildChartLayout({
            preset: 'layerChart',
            traces,
            height: 200,
            legendPosition: 'none',
            xaxis: { title: 'Layer', dtick: 5, showgrid: true },
            yaxis: { title: 'Cosine Similarity', showgrid: true }
        });
        renderChart(plotDiv, traces, layout);
    });
}

/** Render all 4 diagnostic charts at once. Used on initial render and variant change. */
function renderAllDiagnostics(data) {
    renderActivationMagnitudePlot(data);
    renderMassiveActivations(data);
    renderMassiveDimsAcrossLayers(data);
    renderInterLayerSimilarity(data);
}

export {
    renderDiagnosticsSectionHtml,
    renderActivationMagnitudePlot,
    renderMassiveActivations,
    renderMassiveDimsAcrossLayers,
    renderInterLayerSimilarity,
    renderAllDiagnostics,
};
