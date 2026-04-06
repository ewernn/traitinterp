// Main trajectory chart + cue_p plot for Trait Dynamics view
// Input: traitData, loadedTraits, rendering context
// Output: rendered Plotly charts (trajectory, velocity overlay, cue_p, overlay controls)

import { smoothData, computeVelocity, getDimsToRemove, applyMassiveDimCleaning, computeCleanedNorms } from '../../core/utils.js';
import { getDisplayName, getChartColors, getCssVar } from '../../core/display.js';
import { buildChartLayout, renderChart, createHtmlLegend, attachTokenClickHandler, createSeparatorShape, createHighlightShape, buildOverlayShapes, buildCategoryLegendHtml, buildTurnBoundaryShapes } from '../../core/charts.js';
import { setShowCuePOverlay, setShowCategoryOverlay } from '../../core/state.js';
import { renderToggle } from '../../core/ui.js';

// Show all tokens including BOS (set to 2 to skip BOS + warmup if desired)
const START_TOKEN_IDX = 0;

// =============================================================================
// Shared shape builder
// =============================================================================

/**
 * Build common Plotly shapes: separator + highlight + turn boundaries + sentence overlays.
 * Shared by renderTrajectoryChart, renderTraitTokenHeatmap.
 */
function buildCommonShapes(nPromptTokens, isRollout, turnBoundaries, sentenceBoundaries, sentenceCategoryData) {
    const promptEndIdx = nPromptTokens - START_TOKEN_IDX;
    const currentTokenIdx = window.state.currentTokenIndex || 0;
    const highlightX = Math.max(0, currentTokenIdx - START_TOKEN_IDX);

    const shapes = [];
    if (!isRollout) {
        shapes.push({ ...createSeparatorShape(promptEndIdx - 0.5), _isBase: true });
    }
    shapes.push(createHighlightShape(highlightX));
    shapes.push(...buildTurnBoundaryShapes(turnBoundaries).map(s => ({ ...s, _isBase: true })));
    shapes.push(...buildOverlayShapes(sentenceBoundaries, sentenceCategoryData, nPromptTokens).map(s => ({ ...s, _isBase: true })));
    return shapes;
}

// =============================================================================
// Cue_p legend + plot
// =============================================================================

/**
 * Build inline HTML legend for cue_p overlay (blue->red gradient).
 */
function buildCuePLegendHtml() {
    return `
        <span class="overlay-legend">
            <span class="overlay-legend-gradient" style="background: linear-gradient(to right, rgba(0,100,255,0.5), rgba(255,50,50,0.5));"></span>
            <span class="overlay-legend-label">0</span>
            <span class="overlay-legend-label">&rarr;</span>
            <span class="overlay-legend-label">1</span>
        </span>
    `;
}

/**
 * Render cue_p resampling plot — small horizontal strip showing per-sentence cue_p.
 * Only shown when sentenceBoundaries with cue_p values are present.
 */
function renderCuePPlot(sentenceBoundaries, tickVals, tickText, nPromptTokens, isRollout) {
    const section = document.getElementById('cue-p-section');
    const plotDiv = document.getElementById('cue-p-plot');
    if (!sentenceBoundaries || sentenceBoundaries.length === 0) {
        if (section) section.style.display = 'none';
        return;
    }
    // Check that at least one sentence has cue_p
    const hasCueP = sentenceBoundaries.some(s => s.cue_p != null);
    if (!hasCueP) {
        if (section) section.style.display = 'none';
        return;
    }
    section.style.display = '';

    const currentTokenIdx = window.state.currentTokenIndex || 0;
    const highlightX = Math.max(0, currentTokenIdx - START_TOKEN_IDX);
    const promptEndIdx = nPromptTokens - START_TOKEN_IDX;

    // Build step trace: each sentence is a horizontal segment at its cue_p value
    const xVals = [];
    const yVals = [];
    for (const sent of sentenceBoundaries) {
        if (sent.token_start === sent.token_end) continue;
        const x0 = nPromptTokens + sent.token_start - START_TOKEN_IDX;
        const x1 = nPromptTokens + sent.token_end - START_TOKEN_IDX;
        const cueP = sent.cue_p ?? 0;
        xVals.push(x0, x1, null);
        yVals.push(cueP, cueP, null);
    }

    const trace = {
        x: xVals,
        y: yVals,
        type: 'scatter',
        mode: 'lines',
        line: { color: 'rgba(220, 80, 60, 0.9)', width: 2, shape: 'hv' },
        fill: 'tozeroy',
        fillcolor: 'rgba(220, 80, 60, 0.12)',
        hovertemplate: 'cue_p = %{y:.2f}<extra></extra>'
    };

    // Shapes: separator (base, preserved on slider update) + highlight (replaced on slider update)
    const shapes = [];
    if (!isRollout) {
        shapes.push({ ...createSeparatorShape(promptEndIdx - 0.5), _isBase: true });
    }
    shapes.push(createHighlightShape(highlightX));

    const layout = buildChartLayout({
        preset: 'timeSeries',
        traces: [trace],
        height: 100,
        legendPosition: 'none',
        xaxis: {
            tickmode: 'array', tickvals: tickVals, ticktext: tickText,
            tickfont: { size: 8 }, showticklabels: false
        },
        yaxis: {
            title: 'cue_p', range: [0, 1.05],
            tickvals: [0, 0.5, 1], tickfont: { size: 9 },
            zeroline: false
        },
        shapes,
        margin: { l: 60, r: 20, t: 5, b: 5 }
    });
    renderChart(plotDiv, [trace], layout);
    attachTokenClickHandler(plotDiv, START_TOKEN_IDX);
}

// =============================================================================
// Main trajectory rendering
// =============================================================================

/**
 * Render the combined trajectory graph: traces, velocity overlay, overlay controls,
 * top spans dispatch. This is the main chart renderer for the Token Trajectory section.
 */
function renderTrajectoryChart(renderCtx) {
    const {
        traitData, loadedTraits, failedTraits, annotationTokenRanges,
        turnBoundaries, sentenceBoundaries, sentenceCategoryData,
        isReplaySuffix, nPromptTokens, isRollout, allTokens,
        promptTokens, responseTokens, inferenceModel
    } = renderCtx;

    const modelInfoHtml = `Inference model: <code>${inferenceModel}</code>`;

    const failedHtml = failedTraits.length > 0
        ? `<div class="tool-description">No data for: ${failedTraits.map(t => getDisplayName(t)).join(', ')}</div>`
        : '';

    // Determine smoothing and centering
    const isSmoothing = window.state.smoothingWindow > 0;
    const isCentered = window.state.projectionCentered !== false;  // default true

    // Check what we're showing
    const showingDiff = Object.values(traitData).some(d => d.metadata?._isDiff);
    const showingCompModel = Object.values(traitData).some(d => d.metadata?._isComparisonModel);
    const compareModelName = Object.values(traitData).find(d => d.metadata?._compareModel)?.metadata?._compareModel;

    let compareInfoHtml = '';
    if (showingDiff && isReplaySuffix) {
        const organismName = window.state.lastCompareVariant || (window.state.availableComparisonModels || [])[0] || 'organism';
        compareInfoHtml = `<div class="page-intro-text" style="color: var(--accent-color); font-weight: 500;">
            Showing DIFF: ${organismName} \u2212 instruct replay
           </div>`;
    } else if (showingDiff) {
        compareInfoHtml = `<div class="page-intro-text" style="color: var(--accent-color); font-weight: 500;">
            Showing DIFF: ${compareModelName} \u2212 application model
           </div>`;
    } else if (showingCompModel) {
        compareInfoHtml = `<div class="page-intro-text" style="color: var(--accent-color); font-weight: 500;">
            Showing: ${compareModelName} (comparison model)
           </div>`;
    }

    // Update status info in pre-rendered shell
    const statusDiv = document.getElementById('trait-dynamics-status');
    if (statusDiv) {
        statusDiv.innerHTML = `${compareInfoHtml}<div class="page-intro-model">${modelInfoHtml}</div>`;
    }
    if (failedHtml) {
        document.getElementById('combined-activation-plot').insertAdjacentHTML('beforebegin', failedHtml);
    }

    // Prepare data for plotting
    const traitActivations = {};  // Store smoothed activations for heatmap + velocity overlay

    // Prepare traces for Token Trajectory (always cosine)
    const traces = [];

    // Filter traits by selected methods
    const filteredByMethod = loadedTraits.filter(traitName => {
        const method = traitData[traitName]?.metadata?.vector_source?.method;
        return !method || window.state.selectedMethods.has(method);
    });

    if (filteredByMethod.length === 0) {
        // Collect methods present in data for debugging
        const methodsInData = new Set(loadedTraits.map(t => traitData[t]?.metadata?.vector_source?.method).filter(Boolean));
        const selectedMethodsList = [...window.state.selectedMethods];
        document.getElementById('combined-activation-plot').innerHTML = `
            <div class="info">
                No traits match selected methods.<br>
                <small style="color: var(--text-secondary);">
                    Methods in data: ${[...methodsInData].join(', ') || 'none'}<br>
                    Selected methods: ${selectedMethodsList.join(', ') || 'none (check boxes above)'}
                </small>
            </div>
        `;
        return { traitActivations, filteredByMethod };
    }

    const projectionMode = window.state.projectionMode || 'cosine';

    for (let idx = 0; idx < filteredByMethod.length; idx++) {
        const traitName = filteredByMethod[idx];
        const data = traitData[traitName];

        // Get original projections
        let promptProj = [...data.projections.prompt];
        let responseProj = [...data.projections.response];

        // Apply massive dims cleaning if requested and data available
        const cleaningMode = window.state.massiveDimsCleaning || 'none';
        let dimsToRemove = [];
        const mdd = data.massive_dim_data;
        if (cleaningMode !== 'none' && mdd) {
            dimsToRemove = getDimsToRemove(mdd, cleaningMode);
            promptProj = applyMassiveDimCleaning(promptProj, mdd, dimsToRemove, 'prompt');
            responseProj = applyMassiveDimCleaning(responseProj, mdd, dimsToRemove, 'response');
        }

        const allProj = [...promptProj, ...responseProj];

        // Get vector source from metadata
        const vs = data.metadata?.vector_source || {};

        // projections are now 1D arrays (one value per token at best layer)
        let rawProj = allProj.slice(START_TOKEN_IDX);

        // Compute projection values based on mode
        let rawValues;
        if (projectionMode === 'normalized' && data.normalized_response) {
            // Pre-normalized values available — use directly
            const allNorm = [...(data.normalized_prompt || []), ...data.normalized_response];
            rawValues = allNorm.slice(START_TOKEN_IDX);
        } else if (projectionMode === 'raw') {
            rawValues = rawProj;  // raw dot product, no normalization
        } else if (data.token_norms) {
            let promptNorms = data.token_norms.prompt;
            let responseNorms = data.token_norms.response;

            // Use cleaned norms if massive dims were removed
            if (dimsToRemove.length > 0 && mdd) {
                promptNorms = computeCleanedNorms(promptNorms, mdd, dimsToRemove, 'prompt');
                responseNorms = computeCleanedNorms(responseNorms, mdd, dimsToRemove, 'response');
            }

            if (projectionMode === 'normalized') {
                // Normalized mode: divide by mean response norm (preserves per-token variance)
                // For rollouts (empty response), fall back to prompt norms
                const normsForMean = isRollout ? promptNorms : responseNorms;
                const meanNorm = normsForMean.length > 0
                    ? normsForMean.reduce((a, b) => a + b, 0) / normsForMean.length
                    : 1;
                rawValues = rawProj.map(proj => meanNorm > 0 ? proj / meanNorm : 0);
            } else {
                // Cosine mode: divide by per-token norm
                const traitTokenNorms = [...promptNorms, ...responseNorms].slice(START_TOKEN_IDX);
                rawValues = rawProj.map((proj, i) => {
                    const norm = traitTokenNorms[i];
                    return norm > 0 ? proj / norm : 0;
                });
            }
        } else {
            rawValues = rawProj;
        }

        // Store normalized values for Top Spans (before centering/smoothing)
        // For rollouts, use all values (Top Spans hidden but keeps data consistent)
        data._normalizedResponse = isRollout
            ? rawValues
            : rawValues.slice(nPromptTokens - START_TOKEN_IDX);

        // Subtract BOS value if centering is enabled (makes token 0 = 0)
        if (isCentered && rawValues.length > 0) {
            const bosValue = rawValues[0];
            rawValues = rawValues.map(v => v - bosValue);
        }

        // Apply N-token moving average if smoothing is enabled
        const displayValues = isSmoothing ? smoothData(rawValues, window.state.smoothingWindow) : rawValues;

        // Store displayed values for velocity (derivative of what's shown in trajectory)
        traitActivations[traitName] = displayValues;

        // Color: layer-depth scale in layer mode, standard palette otherwise
        let color;
        if (window.state.layerMode && data.metadata?._isMultiVector) {
            const layer = data.metadata?.vector_source?.layer || 0;
            const allLayers = filteredByMethod.map(t => traitData[t]?.metadata?.vector_source?.layer).filter(l => l != null);
            const minL = Math.min(...allLayers);
            const maxL = Math.max(...allLayers);
            const t = maxL > minL ? (layer - minL) / (maxL - minL) : 0.5;
            // Light blue (early layers) -> dark blue (late layers)
            const r = Math.round(180 - t * 130);
            const g = Math.round(210 - t * 130);
            const b = Math.round(255 - t * 55);
            color = `rgb(${r},${g},${b})`;
        } else {
            color = getChartColors()[idx % 10];
        }

        const method = vs.method || 'probe';
        const valueLabel = projectionMode === 'normalized' ? 'Normalized'
                         : projectionMode === 'raw' ? 'Raw (h\u00b7v\u0302)'
                         : 'Cosine (proj / \u2016h\u2016)';
        const valueFormat = '.4f';

        // Build display name and hover
        const baseTrait = data.metadata?._baseTrait || traitName;
        const displayName = data.metadata?._isMultiVector
            ? `${getDisplayName(baseTrait)} (${method} L${vs.layer})`
            : getDisplayName(traitName);
        const pos = data.metadata?.position || vs.position;
        const posStr = pos && pos !== 'response[:]' ? ` @${pos.replace('response', 'resp').replace('prompt', 'p')}` : '';
        const vectorInfo = vs.layer !== undefined ? `<br><span style="color:#888">L${vs.layer} ${method}${posStr}</span>` : '';
        const hoverText = `<b>${displayName}</b>${vectorInfo}<br>Token %{x}<br>${valueLabel}: %{y:${valueFormat}}<extra></extra>`;

        const useMarkers = displayValues.length <= 2000;
        traces.push({
            x: Array.from({length: displayValues.length}, (_, i) => i),
            y: displayValues,
            type: 'scatter',
            mode: useMarkers ? 'lines+markers' : 'lines',
            name: displayName,
            line: { color: color, width: 1.5 },
            ...(useMarkers ? { marker: { size: 2, color: color } } : {}),
            hovertemplate: hoverText
        });
    }

    // Get display tokens (adaptive spacing for x-axis labels)
    const displayTokens = allTokens.slice(START_TOKEN_IDX);
    const tickStep = Math.max(10, Math.floor(displayTokens.length / 80));
    const tickVals = [];
    const tickText = [];
    for (let i = 0; i < displayTokens.length; i += tickStep) {
        tickVals.push(i);
        tickText.push(displayTokens[i]);
    }

    // Build shapes using shared helper
    const shapes = buildCommonShapes(nPromptTokens, isRollout, turnBoundaries, sentenceBoundaries, sentenceCategoryData);

    // Annotation shaded bands (response token ranges offset by nPromptTokens)
    for (const [start, end] of annotationTokenRanges) {
        shapes.push({
            type: 'rect',
            x0: (nPromptTokens - START_TOKEN_IDX) + start - 0.5,
            x1: (nPromptTokens - START_TOKEN_IDX) + end - 0.5,
            y0: 0, y1: 1, yref: 'paper',
            fillcolor: 'rgba(255, 180, 60, 0.12)',
            line: { width: 0 },
            layer: 'below',
            _isBase: true
        });
    }

    // PROMPT/RESPONSE labels (skip for rollouts — turn boundaries replace them)
    const textSecondary = getCssVar('--text-secondary', '#a4a4a4');
    const annotations = [];
    if (!isRollout) {
        annotations.push(
            {
                x: (nPromptTokens - START_TOKEN_IDX) / 2 - 0.5,
                y: 1.08, yref: 'paper',
                text: 'PROMPT', showarrow: false,
                font: { size: 11, color: textSecondary }
            },
            {
                x: (nPromptTokens - START_TOKEN_IDX) + (displayTokens.length - (nPromptTokens - START_TOKEN_IDX)) / 2 - 0.5,
                y: 1.08, yref: 'paper',
                text: 'RESPONSE', showarrow: false,
                font: { size: 11, color: textSecondary }
            }
        );
    }

    // Build tooltips for legend (vector source info)
    const legendTooltips = filteredByMethod.map(traitName => {
        const data = traitData[traitName];
        const vs = data.metadata?.vector_source || {};
        const pos = data.metadata?.position || vs.position;
        const posStr = pos && pos !== 'response[:]' ? ` @${pos.replace('response', 'resp').replace('prompt', 'p')}` : '';
        return vs.layer !== undefined
            ? `L${vs.layer} ${vs.method || '?'}${posStr} (${vs.selection_source || 'unknown'})`
            : 'no metadata';
    });

    // Token Trajectory plot
    // Use projectionMode for the axis title
    const yAxisTitle = projectionMode === 'normalized' ? 'Normalized (proj / avg\u2016h\u2016)'
                     : projectionMode === 'raw' ? 'Raw Projection'
                     : 'Cosine (proj / \u2016h\u2016)';

    // Compute y-axis range: minimum +/-0.15, auto-expand if data exceeds
    let yAxisConfig = { title: yAxisTitle, zeroline: true, zerolinewidth: 1, showgrid: true };
    // Find actual data range across all traces (skip first few special tokens for auto-range)
    const rangeSkip = Math.min(4, nPromptTokens);
    let minY = Infinity, maxY = -Infinity;
    traces.forEach(t => {
        t.y.forEach((v, i) => {
            if (i < rangeSkip) return;
            if (v < minY) minY = v;
            if (v > maxY) maxY = v;
        });
    });
    // Pad y-axis: 15% of data range, minimum +/-0.02 (auto-zooms for diff mode)
    const pad = Math.max(0.02, (maxY - minY) * 0.15);
    const rangeMin = minY - pad;
    const rangeMax = maxY + pad;
    yAxisConfig.range = [rangeMin, rangeMax];

    // Velocity overlay on secondary y-axis (when toggled on)
    if (window.state.showVelocity) {
        for (let idx = 0; idx < filteredByMethod.length; idx++) {
            const traitName = filteredByMethod[idx];
            const activations = traitActivations[traitName];
            if (!activations) continue;
            const velocity = computeVelocity(activations);
            const smoothedVelocity = window.state.smoothingWindow > 0 ? smoothData(velocity, window.state.smoothingWindow) : velocity;
            const color = traces[idx]?.line?.color || getChartColors()[idx % 10];
            traces.push({
                x: Array.from({length: smoothedVelocity.length}, (_, i) => i + 0.5),
                y: smoothedVelocity,
                type: 'scatter',
                mode: 'lines',
                name: `${traces[idx]?.name || getDisplayName(traitName)} (vel)`,
                line: { color, width: 1, dash: 'dot' },
                yaxis: 'y2',
                showlegend: false,
                hovertemplate: `<b>${traces[idx]?.name || getDisplayName(traitName)}</b><br>Token %{x:.0f}<br>Velocity: %{y:.4f}<extra></extra>`
            });
        }
    }

    const mainLayout = buildChartLayout({
        preset: 'timeSeries',
        traces,
        height: 400,
        legendPosition: 'none',  // Using custom HTML legend instead
        xaxis: {
            title: 'Token Position',
            tickmode: 'array',
            tickvals: tickVals,
            ticktext: tickText,
            tickangle: -45,
            showgrid: false,
            tickfont: { size: 9 },
        },
        yaxis: yAxisConfig,
        shapes: shapes,
        annotations: annotations,
        margin: { l: 60, r: window.state.showVelocity ? 60 : 20, t: 40, b: 80 },
        hovermode: 'closest'
    });

    // Add secondary y-axis for velocity overlay
    if (window.state.showVelocity) {
        mainLayout.yaxis2 = {
            title: 'Velocity',
            overlaying: 'y',
            side: 'right',
            zeroline: true,
            zerolinewidth: 1,
            showgrid: false,
            tickfont: { size: 9 }
        };
    }

    renderChart('combined-activation-plot', traces, mainLayout);

    // Insert custom legend with click-to-toggle and hover-to-highlight
    const plotDiv = document.getElementById('combined-activation-plot');
    // Remove any existing legend from previous render
    const existingLegend = plotDiv.parentNode.querySelector('.chart-legend-interactive');
    if (existingLegend) existingLegend.remove();
    const legendDiv = createHtmlLegend(traces, plotDiv, {
        tooltips: legendTooltips,
        hoverHighlight: true
    });
    plotDiv.parentNode.insertBefore(legendDiv, plotDiv.nextSibling);
    attachTokenClickHandler(plotDiv, START_TOKEN_IDX);

    // Populate overlay controls (only when sentence boundary data exists)
    const overlayControlsDiv = document.getElementById('overlay-controls');
    if (overlayControlsDiv) {
        if (sentenceBoundaries && sentenceBoundaries.length > 0) {
            const showCueP = window.state.showCuePOverlay;
            const showCategory = window.state.showCategoryOverlay;
            const hasCategoryData = sentenceCategoryData && sentenceCategoryData.length > 0;

            overlayControlsDiv.innerHTML = `
                <div class="overlay-controls-bar">
                    <span class="projection-toggle-label">Overlays:</span>
                    ${renderToggle({ id: 'cue-p-overlay-toggle', label: 'cue_p', checked: showCueP, className: 'projection-toggle-checkbox' })}
                    ${showCueP ? buildCuePLegendHtml() : ''}
                    ${hasCategoryData ? renderToggle({ id: 'category-overlay-toggle', label: 'Category', checked: showCategory, className: 'projection-toggle-checkbox' }) : ''}
                    ${showCategory && hasCategoryData ? buildCategoryLegendHtml(sentenceCategoryData) : ''}
                </div>
            `;

            const cuePToggle = document.getElementById('cue-p-overlay-toggle');
            if (cuePToggle) {
                cuePToggle.addEventListener('change', () => setShowCuePOverlay(cuePToggle.checked));
            }
            const catToggle = document.getElementById('category-overlay-toggle');
            if (catToggle) {
                catToggle.addEventListener('change', () => setShowCategoryOverlay(catToggle.checked));
            }
        } else {
            overlayControlsDiv.innerHTML = '';
        }
    }

    // Render Top Spans panel (diff mode only, not available for rollouts)
    if (!isRollout) {
        window.topSpans.renderPanel(traitData, filteredByMethod, responseTokens, nPromptTokens);
    }

    // Render cue_p resampling plot (thought branches only)
    renderCuePPlot(sentenceBoundaries, tickVals, tickText, nPromptTokens, isRollout);

    return { traitActivations, filteredByMethod, tickVals, tickText, displayTokens };
}

export { START_TOKEN_IDX, buildCommonShapes, renderTrajectoryChart };
