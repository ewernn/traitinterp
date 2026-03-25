import { fetchJSON } from '../core/utils.js';
import { CORRELATION_COLORSCALE, getChartColors } from '../core/display.js';
import { buildChartLayout, renderChart, buildHeatmapAnnotations } from '../core/charts.js';
import { renderSubsection } from '../core/ui.js';

// Trait Correlation Section - Analyze relationships between trait projections
//
// Loads pre-computed correlation data from:
//   experiments/{exp}/analysis/trait_correlation/{prompt_set}.json
//
// To generate: python analysis/trait_correlation.py --experiment {exp} --prompt-set {prompt_set}

// Module-local state (not part of global state shape)
let traitCorrelationData = null;
let correlationOffset = 0;


/**
 * Render correlation charts into a provided container.
 * Returns true if data was loaded and rendered, false if no data available.
 */
async function renderCorrelationSection(containerId, promptSet) {
    const container = document.getElementById(containerId);
    if (!container) return false;

    if (!window.state.currentExperiment || !promptSet) return false;

    // Load pre-computed correlation data
    const dataFile = `/experiments/${window.state.currentExperiment}/analysis/trait_correlation/${promptSet.replace('/', '_')}.json`;

    const data = await fetchJSON(dataFile);
    if (!data) return false;

    // Store for slider updates
    traitCorrelationData = data;

    const currentOffset = correlationOffset || 0;
    const maxOffset = data.max_offset || 10;

    container.innerHTML = `
        <div class="page-intro-model" style="margin-bottom: 12px;">
            Prompt set: <code>${promptSet}</code> (${data.n_prompts} prompts, ${data.traits.length} traits)
        </div>

        <div class="projection-toggle">
            <label class="projection-toggle-label">Offset: <span id="offset-value">${currentOffset}</span> tokens</label>
            <input type="range" id="correlation-offset-slider" min="0" max="${maxOffset}" value="${currentOffset}" style="width: 200px; margin-left: 8px;">
        </div>
        <div id="correlation-heatmap"></div>
        <div id="correlation-legend" class="chart-legend" style="margin-top: 8px; font-size: 12px; color: var(--text-secondary);">
            <span>Upper: row leads col by +k</span>
            <span style="margin-left: 16px;">Lower: col leads row by +k</span>
            <span style="margin-left: 16px;">Diagonal: autocorrelation at k</span>
        </div>

        <div style="margin-top: 24px;">
            ${renderSubsection({
                title: 'Correlation Decay',
                infoId: 'info-correlation-decay',
                infoText: 'How trait correlations change with token offset. Fast decay = local relationship. Slow decay = persistent relationship.',
                level: 'h3'
            })}
            <div id="correlation-decay-plot"></div>
        </div>

        <div style="margin-top: 24px;">
            ${renderSubsection({
                title: 'Response-Level Correlation',
                infoId: 'info-response-correlation',
                infoText: 'Correlation of mean projection per response (not token-level). Shows which traits co-occur across prompts.',
                level: 'h3'
            })}
            <div id="response-correlation-heatmap"></div>
        </div>
    `;

    // Render plots
    renderCorrelationHeatmap(currentOffset);
    renderCorrelationDecay();
    renderResponseCorrelation();

    // Setup slider
    const slider = document.getElementById('correlation-offset-slider');
    const offsetLabel = document.getElementById('offset-value');
    slider.addEventListener('input', () => {
        const offset = parseInt(slider.value);
        offsetLabel.textContent = offset;
        correlationOffset = offset;
        renderCorrelationHeatmap(offset);
    });

    return true;
}


/**
 * Reset module-local caches (call on experiment change).
 */
function resetCorrelationState() {
    traitCorrelationData = null;
    correlationOffset = 0;
}


function renderCorrelationHeatmap(offset) {
    const data = traitCorrelationData;
    if (!data) return;

    const { trait_labels, correlations_by_offset } = data;
    const posMatrix = correlations_by_offset[String(offset)] || correlations_by_offset['0'];
    const negMatrix = offset > 0 ? correlations_by_offset[String(-offset)] : posMatrix;

    if (!posMatrix) return;

    // Build asymmetric matrix: upper = +offset, lower = -offset
    const n = trait_labels.length;
    const displayMatrix = Array(n).fill(null).map(() => Array(n).fill(0));

    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            if (i < j) {
                // Upper triangle: positive offset (row leads col)
                displayMatrix[i][j] = posMatrix[i][j];
            } else if (i > j) {
                // Lower triangle: negative offset (col leads row)
                displayMatrix[i][j] = negMatrix ? negMatrix[i][j] : posMatrix[i][j];
            } else {
                // Diagonal: autocorrelation
                displayMatrix[i][j] = posMatrix[i][j];
            }
        }
    }

    const trace = {
        z: displayMatrix,
        x: trait_labels,
        y: trait_labels,
        type: 'heatmap',
        colorscale: CORRELATION_COLORSCALE,
        zmin: -1,
        zmax: 1,
        hoverongaps: false,
        hovertemplate: '%{y} → %{x}<br>r = %{z:.3f}<extra></extra>',
        showscale: true,
        colorbar: {
            title: 'Correlation',
            titleside: 'right',
            tickvals: [-1, -0.5, 0, 0.5, 1],
            ticktext: ['-1', '-0.5', '0', '0.5', '1']
        }
    };

    // Add text annotations
    const annotations = buildHeatmapAnnotations(displayMatrix, trait_labels, trait_labels, { threshold: 0.5, fontSize: 11 });

    const layout = buildChartLayout({
        preset: 'heatmap',
        traces: [trace],
        title: offset === 0 ? 'Token-Level Correlation (offset=0, symmetric)' :
               `Token-Level Correlation (offset=±${offset})`,
        xaxis: {
            title: '',
            tickangle: -45,
            side: 'bottom'
        },
        yaxis: {
            title: '',
            autorange: 'reversed'
        },
        annotations: annotations,
        margin: { l: 100, r: 80, t: 60, b: 100 }
    });

    renderChart('correlation-heatmap', [trace], layout);
}


function renderCorrelationDecay() {
    const data = traitCorrelationData;
    if (!data) return;

    const { trait_labels, correlations_by_offset, max_offset } = data;
    const colors = getChartColors();
    const traces = [];

    let colorIdx = 0;
    for (let i = 0; i < trait_labels.length; i++) {
        for (let j = i + 1; j < trait_labels.length; j++) {
            const offsets = [];
            const correlations = [];

            for (let k = 0; k <= max_offset; k++) {
                const matrix = correlations_by_offset[String(k)];
                if (matrix) {
                    offsets.push(k);
                    correlations.push(matrix[i][j]);
                }
            }

            traces.push({
                x: offsets,
                y: correlations,
                type: 'scatter',
                mode: 'lines+markers',
                name: `${trait_labels[i]} ↔ ${trait_labels[j]}`,
                line: { color: colors[colorIdx % colors.length], width: 1.5 },
                marker: { size: 4 },
                hovertemplate: `${trait_labels[i]} ↔ ${trait_labels[j]}<br>Offset: %{x}<br>r = %{y:.3f}<extra></extra>`
            });
            colorIdx++;
        }
    }

    const layout = buildChartLayout({
        preset: 'timeSeries',
        traces,
        height: 350,
        legendPosition: 'right',
        xaxis: { title: 'Token Offset', dtick: 1 },
        yaxis: { title: 'Correlation', range: [-1, 1], zeroline: true, zerolinewidth: 1 },
        margin: { r: 150 }
    });

    renderChart('correlation-decay-plot', traces, layout);
}


function renderResponseCorrelation() {
    const data = traitCorrelationData;
    if (!data || !data.response_correlation) return;

    const { trait_labels, response_correlation } = data;

    const trace = {
        z: response_correlation,
        x: trait_labels,
        y: trait_labels,
        type: 'heatmap',
        colorscale: CORRELATION_COLORSCALE,
        zmin: -1,
        zmax: 1,
        hovertemplate: '%{y} ↔ %{x}<br>r = %{z:.3f}<extra></extra>',
        showscale: true,
        colorbar: {
            title: 'Correlation',
            titleside: 'right'
        }
    };

    const annotations = buildHeatmapAnnotations(response_correlation, trait_labels, trait_labels, { threshold: 0.5, fontSize: 11 });

    const layout = buildChartLayout({
        preset: 'heatmap',
        traces: [trace],
        title: 'Response-Level Correlation (mean projection per response)',
        xaxis: { title: '', tickangle: -45 },
        yaxis: { title: '', autorange: 'reversed' },
        annotations: annotations,
        margin: { l: 100, r: 80, t: 60, b: 100 }
    });

    renderChart('response-correlation-heatmap', [trace], layout);
}


// ES module exports
export { renderCorrelationSection, resetCorrelationState };

// Keep window.* for cross-module access (state.js calls resetCorrelationState on experiment change)
window.resetCorrelationState = resetCorrelationState;
