// Activation magnitude per token chart for Trait Dynamics view
// Input: traitData, loadedTraits, tick data
// Output: rendered Plotly magnitude plot

import { getChartColors } from '../../core/display.js';
import { buildChartLayout, renderChart, attachTokenClickHandler, createSeparatorShape, createHighlightShape, buildOverlayShapes, buildTurnBoundaryShapes } from '../../core/charts.js';
import { START_TOKEN_IDX } from './chart-trajectory.js';

/**
 * Render Token Magnitude plot showing L2 norm per token at best layer.
 * Helps identify if low projections are due to low magnitude or orthogonal encoding.
 */
function renderTokenMagnitudePlot(traitData, loadedTraits, tickVals, tickText, nPromptTokens, isRollout = false, turnBoundaries = null, sentenceBoundaries = null, sentenceCategoryData = null) {
    const plotDiv = document.getElementById('token-magnitude-plot');
    const firstTraitData = traitData[loadedTraits[0]];

    if (!firstTraitData.token_norms) {
        plotDiv.innerHTML = `
            <div class="info">
                Per-token norms not available. Re-run projection script to generate.
            </div>
        `;
        return;
    }

    // Collect unique layers from all traits
    const layerToNorms = {};
    for (const traitName of loadedTraits) {
        const data = traitData[traitName];
        if (!data.token_norms) continue;
        const layer = data.metadata?.vector_source?.layer ?? 'unknown';
        if (!(layer in layerToNorms)) {
            const promptNorms = data.token_norms.prompt;
            const responseNorms = data.token_norms.response;
            layerToNorms[layer] = [...promptNorms, ...responseNorms].slice(START_TOKEN_IDX);
        }
    }

    const colors = getChartColors();
    const currentTokenIdx = window.state.currentTokenIndex || 0;
    const highlightX = Math.max(0, currentTokenIdx - START_TOKEN_IDX);
    const promptEndIdx = nPromptTokens - START_TOKEN_IDX;

    // Create a trace for each unique layer
    const traces = Object.entries(layerToNorms).map(([layer, norms], idx) => ({
        y: norms,
        type: 'scatter',
        mode: 'lines',
        name: `L${layer}`,
        line: { color: colors[idx % colors.length], width: 1.5 },
        hovertemplate: `L${layer}<br>Token %{x}<br>||h|| = %{y:.1f}<extra></extra>`
    }));

    // Compute y-axis range: cap at 95th percentile to avoid BOS/early token spikes crushing the plot
    const allNormValues = Object.values(layerToNorms).flat().filter(v => v > 0);
    let yaxisMagnitude = { title: '||h|| (L2 norm)', tickfont: { size: 10 } };
    if (allNormValues.length > 0) {
        const sorted = [...allNormValues].sort((a, b) => a - b);
        const p95 = sorted[Math.floor(sorted.length * 0.95)];
        const maxVal = sorted[sorted.length - 1];
        if (maxVal > p95 * 3) {
            yaxisMagnitude.range = [0, p95 * 1.3];
        }
    }

    const showLegend = Object.keys(layerToNorms).length > 1;

    const layout = buildChartLayout({
        preset: 'timeSeries',
        traces,
        height: 200,
        legendPosition: showLegend ? 'above' : 'none',
        xaxis: {
            title: 'Token',
            tickvals: tickVals,
            ticktext: tickText,
            tickfont: { size: 9 }
        },
        yaxis: yaxisMagnitude,
        hovermode: 'closest',
        shapes: [
            ...(isRollout ? [] : [createSeparatorShape(promptEndIdx)]),
            createHighlightShape(highlightX),
            ...buildTurnBoundaryShapes(turnBoundaries).map(s => ({ ...s, _isBase: true })),
            ...buildOverlayShapes(sentenceBoundaries, sentenceCategoryData, nPromptTokens).map(s => ({ ...s, _isBase: true }))
        ]
    });
    renderChart(plotDiv, traces, layout);

    // Click-to-select
    attachTokenClickHandler(plotDiv, START_TOKEN_IDX);
}

export { renderTokenMagnitudePlot };
