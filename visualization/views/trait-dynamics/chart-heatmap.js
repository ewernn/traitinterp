// Trait x Token heatmap chart for Trait Dynamics view
// Input: traitActivations, loadedTraits, tick data, traitData
// Output: rendered Plotly heatmap into #trait-heatmap-panel (sec-header body managed by controls.js)

import { getDisplayName, DELTA_COLORSCALE } from '../../core/display.js';
import { buildChartLayout, renderChart, attachTokenClickHandler } from '../../core/charts.js';
import { buildCommonShapes, START_TOKEN_IDX } from './chart-trajectory.js';

/**
 * Render Trait x Token heatmap: all traits as rows, tokens as columns, colored by projection value.
 * Reuses already-computed traitActivations (smoothed/centered/normalized).
 * Renders directly into #trait-heatmap-panel; collapse/expand handled by sec-header in controls.js.
 */
function renderTraitTokenHeatmap(traitActivations, loadedTraits, tickVals, tickText, nPromptTokens, displayTokens, isRollout, turnBoundaries, sentenceBoundaries, traitData, sentenceCategoryData = null) {
    const panel = document.getElementById('trait-heatmap-panel');
    if (!panel) return;

    // Hide when <=1 trait (single-row heatmap has no value)
    if (loadedTraits.length <= 1) {
        panel.innerHTML = '';
        return;
    }

    // Build display names for y-axis labels
    const traitLabels = loadedTraits.map(traitName => {
        const data = traitData[traitName];
        const baseTrait = data.metadata?._baseTrait || traitName;
        const vs = data.metadata?.vector_source || {};
        return data.metadata?._isMultiVector
            ? `${getDisplayName(baseTrait)} L${vs.layer}`
            : getDisplayName(traitName);
    });

    // Render plot directly into panel (no dropdown wrapper)
    panel.innerHTML = '<div id="trait-heatmap-plot"></div>';

    // Build z-matrix (traits x tokens)
    const z = loadedTraits.map(traitName => traitActivations[traitName] || []);

    // Symmetric colorscale around 0
    let absMax = 0;
    for (const row of z) {
        for (const v of row) {
            const a = Math.abs(v);
            if (a > absMax) absMax = a;
        }
    }
    if (absMax === 0) absMax = 1;

    const trace = {
        z: z,
        y: traitLabels,
        type: 'heatmap',
        colorscale: DELTA_COLORSCALE,
        zmid: 0,
        zmin: -absMax,
        zmax: absMax,
        hovertemplate: '%{y}<br>Token %{x}<br>Value: %{z:.4f}<extra></extra>',
        colorbar: {
            thickness: 12,
            len: 0.8,
            tickfont: { size: 9 }
        }
    };

    // Build shapes using shared helper
    const shapes = buildCommonShapes(nPromptTokens, isRollout, turnBoundaries, sentenceBoundaries, sentenceCategoryData);

    const height = Math.max(150, loadedTraits.length * 25 + 80);

    const layout = buildChartLayout({
        preset: 'heatmap',
        traces: [trace],
        height,
        legendPosition: 'none',
        xaxis: {
            title: 'Token Position',
            tickmode: 'array',
            tickvals: tickVals,
            ticktext: tickText,
            tickangle: -45,
            tickfont: { size: 9 },
            showgrid: false
        },
        yaxis: {
            tickfont: { size: 10 },
            automargin: true
        },
        shapes,
        margin: { l: 120, r: 60, t: 10, b: 60 }
    });

    renderChart('trait-heatmap-plot', [trace], layout);

    // Click-to-select token
    attachTokenClickHandler('trait-heatmap-plot', START_TOKEN_IDX);

    // Update sec-header badge
    const badge = document.getElementById('badge-heatmap');
    if (badge) badge.textContent = loadedTraits.length + ' traits';
}

export { renderTraitTokenHeatmap };
