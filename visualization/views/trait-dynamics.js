// Trait Dynamics View - Watch the model's internal state evolve token-by-token
// Core insight: "See how the model is thinking" by projecting onto trait vectors
//
// Sections:
// 1. Token Trajectory: X=tokens, Y=projection (best layer) + velocity/acceleration
// 2. Activation Magnitude

// Show all tokens including BOS (set to 2 to skip BOS + warmup if desired)
const START_TOKEN_IDX = 0;

/**
 * Apply a centered moving average to smooth data.
 * @param {number[]} data - Input array
 * @param {number} window - Window size (should be odd for centered average)
 * @returns {number[]} Smoothed array (same length as input)
 */
function smoothData(data, window = 3) {
    if (data.length < window) return data;
    const half = Math.floor(window / 2);
    const result = [];
    for (let i = 0; i < data.length; i++) {
        const start = Math.max(0, i - half);
        const end = Math.min(data.length, i + half + 1);
        const slice = data.slice(start, end);
        result.push(slice.reduce((a, b) => a + b, 0) / slice.length);
    }
    return result;
}

/**
 * Compute first derivative (velocity) from an array
 */
function computeVelocity(data) {
    const velocity = [];
    for (let i = 0; i < data.length - 1; i++) {
        velocity.push(data[i + 1] - data[i]);
    }
    return velocity;
}


async function renderTraitDynamics() {
    const contentArea = document.getElementById('content-area');
    const filteredTraits = window.getFilteredTraits();

    // Preserve scroll position
    const scrollY = contentArea.scrollTop;

    if (filteredTraits.length === 0) {

        contentArea.innerHTML = `
            <div class="tool-view">
                <div class="page-intro">
                    <div class="page-intro-text">Watch traits evolve token-by-token during generation.</div>
                </div>
                <div class="info">Select at least one trait from the sidebar to view activation trajectories.</div>
            </div>
        `;
        return;
    }

    const traitData = {};
    const failedTraits = [];
    const promptSet = window.state.currentPromptSet;
    const promptId = window.state.currentPromptId;

    if (!promptSet || !promptId) {
        renderNoDataMessage(contentArea, filteredTraits, promptSet, promptId);
        return;
    }

    // Show loading state only if fetch takes > 150ms
    const loadingTimeout = setTimeout(() => {
        contentArea.innerHTML = `
            <div class="tool-view">
                <div class="page-intro">
                    <div class="page-intro-text">Watch traits evolve token-by-token during generation.</div>
                </div>
                <div class="info">Loading data for ${filteredTraits.length} trait(s)...</div>
            </div>
        `;
    }, 150);

    // Load shared response data (prompt/response text and tokens)
    let responseData = null;
    try {
        const responsePath = window.paths.responseData(promptSet, promptId);
        const responseRes = await fetch(responsePath);
        if (responseRes.ok) {
            responseData = await responseRes.json();
        }
    } catch (error) {
        console.warn('Could not load shared response data, falling back to projection data');
    }

    // Load projection data for ALL selected traits
    for (const trait of filteredTraits) {
        try {
            const fetchPath = window.paths.residualStreamData(trait, promptSet, promptId);
            const response = await fetch(fetchPath);

            if (!response.ok) {
                failedTraits.push(trait.name);
                continue;
            }

            const projData = await response.json();

            // Check for API error response
            if (projData.error) {
                failedTraits.push(trait.name);
                continue;
            }

            // Merge response data with projection data (projection is slim, needs tokens)
            if (responseData) {
                projData.prompt = responseData.prompt;
                projData.response = responseData.response;
                if (responseData.metadata?.inference_model && !projData.metadata?.inference_model) {
                    projData.metadata = projData.metadata || {};
                    projData.metadata.inference_model = responseData.metadata.inference_model;
                }
            }

            // Handle multi-vector format: expand into separate entries per vector
            if (projData.metadata?.multi_vector && Array.isArray(projData.projections)) {
                for (const vecProj of projData.projections) {
                    const key = `${trait.name}__${vecProj.method}_L${vecProj.layer}`;
                    traitData[key] = {
                        ...projData,
                        // Override projections with this vector's data
                        projections: {
                            prompt: vecProj.prompt,
                            response: vecProj.response
                        },
                        // Add vector source info for this specific vector
                        metadata: {
                            ...projData.metadata,
                            vector_source: {
                                layer: vecProj.layer,
                                method: vecProj.method,
                                selection_source: vecProj.selection_source,
                                baseline: vecProj.baseline
                            },
                            _baseTrait: trait.name,  // Track original trait
                            _isMultiVector: true
                        }
                    };
                }
            } else {
                // Single-vector format (unchanged)
                traitData[trait.name] = projData;
            }
        } catch (error) {
            failedTraits.push(trait.name);
        }
    }

    clearTimeout(loadingTimeout);

    // If diff mode is enabled, fetch comparison data and compute diff
    const diffPromptSet = window.state.diffMode && window.state.diffPromptSet;
    if (diffPromptSet) {
        for (const traitKey of Object.keys(traitData)) {
            // Get base trait for multi-vector entries
            const baseTrait = traitData[traitKey].metadata?._baseTrait || traitKey;
            const trait = filteredTraits.find(t => t.name === baseTrait);
            if (!trait) continue;

            try {
                const fetchPath = window.paths.residualStreamData(trait, diffPromptSet, promptId);
                const response = await fetch(fetchPath);
                if (!response.ok) continue;

                const compData = await response.json();
                if (compData.error) continue;

                // Get matching projection data
                let compProj;
                if (compData.metadata?.multi_vector && Array.isArray(compData.projections)) {
                    // Find matching method/layer
                    const vs = traitData[traitKey].metadata?.vector_source;
                    if (vs) {
                        const match = compData.projections.find(p => p.method === vs.method && p.layer === vs.layer);
                        compProj = match ? { prompt: match.prompt, response: match.response } : null;
                    }
                } else {
                    compProj = compData.projections;
                }

                if (compProj) {
                    const mainProj = traitData[traitKey].projections;
                    // Compute diff: main - comparison (e.g., lora - clean)
                    const diffPrompt = mainProj.prompt.map((v, i) => v - (compProj.prompt[i] || 0));
                    const diffResponse = mainProj.response.map((v, i) => v - (compProj.response[i] || 0));
                    traitData[traitKey].projections = { prompt: diffPrompt, response: diffResponse };
                    traitData[traitKey].metadata = traitData[traitKey].metadata || {};
                    traitData[traitKey].metadata._isDiff = true;
                    traitData[traitKey].metadata._diffFrom = diffPromptSet;
                }
            } catch (error) {
                console.warn(`Could not load diff data for ${traitKey}:`, error);
            }
        }
    }

    // Check if we have any data
    const loadedTraits = Object.keys(traitData);
    if (loadedTraits.length === 0) {
        renderNoDataMessage(contentArea, filteredTraits, promptSet, promptId);
        return;
    }

    // Render the full view
    renderCombinedGraph(contentArea, traitData, loadedTraits, failedTraits, promptSet, promptId);

    // Restore scroll position after DOM updates
    requestAnimationFrame(() => {
        contentArea.scrollTop = scrollY;
    });
}

function renderNoDataMessage(container, traits, promptSet, promptId) {
    const promptLabel = promptSet && promptId ? `${promptSet}/${promptId}` : 'none selected';
    container.innerHTML = `
        <div class="tool-view">
            <div class="page-intro">
                <div class="page-intro-text">Watch traits evolve token-by-token during generation.</div>
            </div>
            <div class="info">
                No data available for prompt ${promptLabel} for any selected trait.
            </div>
            <p class="tool-description">
                To capture per-token activation data, run:
            </p>
            <pre>python inference/capture_raw_activations.py --experiment ${window.paths.getExperiment()} --prompt-set ${promptSet || 'PROMPT_SET'}</pre>
        </div>
    `;
}

function renderCombinedGraph(container, traitData, loadedTraits, failedTraits, promptSet, promptId) {
    // Use first trait's data as reference for tokens (they should all be the same)
    const refData = traitData[loadedTraits[0]];

    if (!refData.prompt?.tokens || !refData.response?.tokens) {
        container.innerHTML = `<div class="tool-view"><div class="info">Error: Missing token data.</div></div>`;
        return;
    }

    const promptTokens = refData.prompt.tokens;
    const responseTokens = refData.response.tokens;
    const allTokens = [...promptTokens, ...responseTokens];
    const nPromptTokens = promptTokens.length;
    const nTotalTokens = allTokens.length;

    // Extract inference model and vector source from metadata
    const meta = refData.metadata || {};
    const inferenceModel = meta.inference_model ||
        window.state.experimentData?.experimentConfig?.application_model ||
        'unknown';
    const vectorSource = meta.vector_source || {};

    // Build model info HTML (just inference model - vector details shown on hover)
    const modelInfoHtml = `Inference model: <code>${inferenceModel}</code>`;

    // Build HTML
    let failedHtml = '';
    if (failedTraits.length > 0) {
        failedHtml = `
            <div class="tool-description">
                No data for: ${failedTraits.map(t => window.getDisplayName(t)).join(', ')}
            </div>
        `;
    }

    // Determine smoothing and centering
    const isSmoothing = window.state.smoothingEnabled !== false;  // default true
    const isCentered = window.state.projectionCentered !== false;  // default true

    // Diff mode: compare with another prompt set
    const diffMode = window.state.diffMode || false;
    const diffPromptSet = window.state.diffPromptSet || null;
    const availableSets = Object.keys(window.state.promptsWithData || {}).filter(s => s !== promptSet);

    // Check if we're showing diff data
    const showingDiff = Object.values(traitData).some(d => d.metadata?._isDiff);
    const diffFromSet = showingDiff ? Object.values(traitData).find(d => d.metadata?._isDiff)?.metadata?._diffFrom : null;
    const diffInfoHtml = showingDiff
        ? `<div class="page-intro-text" style="color: var(--color-accent); font-weight: 500;">
            Showing DIFF: ${promptSet.replace(/_/g, ' ')} − ${diffFromSet?.replace(/_/g, ' ')}
           </div>`
        : '';

    container.innerHTML = `
        <div class="tool-view">
            <div class="page-intro">
                <div class="page-intro-text">Watch traits evolve token-by-token during generation.</div>
                ${diffInfoHtml}
                <div class="page-intro-model">${modelInfoHtml}</div>
            </div>
            ${failedHtml}

            <section>
                <h2>Token Trajectory <span class="subsection-info-toggle" data-target="info-token-trajectory">►</span></h2>
                <div class="subsection-info" id="info-token-trajectory">
                    Cosine similarity: proj / ||h||. Shows directional alignment with trait vector.
                    ${isCentered ? ' Centered by subtracting BOS token value.' : ''}
                    ${isSmoothing ? ' Smoothed with 3-token moving average.' : ''}
                </div>
                <div class="projection-toggle">
                    <label class="projection-toggle-checkbox">
                        <input type="checkbox" id="smoothing-toggle" ${isSmoothing ? 'checked' : ''}>
                        <span>Smooth</span>
                    </label>
                    <label class="projection-toggle-checkbox">
                        <input type="checkbox" id="projection-centered-toggle" ${isCentered ? 'checked' : ''}>
                        <span>Centered</span>
                    </label>
                    <span class="projection-toggle-label" style="margin-left: 16px;">Methods:</span>
                    <label class="projection-toggle-checkbox">
                        <input type="checkbox" class="method-filter" data-method="probe" ${window.state.selectedMethods.has('probe') ? 'checked' : ''}>
                        <span>probe</span>
                    </label>
                    <label class="projection-toggle-checkbox">
                        <input type="checkbox" class="method-filter" data-method="mean_diff" ${window.state.selectedMethods.has('mean_diff') ? 'checked' : ''}>
                        <span>mean_diff</span>
                    </label>
                    <label class="projection-toggle-checkbox">
                        <input type="checkbox" class="method-filter" data-method="gradient" ${window.state.selectedMethods.has('gradient') ? 'checked' : ''}>
                        <span>gradient</span>
                    </label>
                    <label class="projection-toggle-checkbox">
                        <input type="checkbox" class="method-filter" data-method="random" ${window.state.selectedMethods.has('random') ? 'checked' : ''}>
                        <span>random</span>
                    </label>
                    <span class="projection-toggle-label" style="margin-left: 16px;">Diff:</span>
                    <label class="projection-toggle-checkbox">
                        <input type="checkbox" id="diff-mode-toggle" ${diffMode ? 'checked' : ''}>
                        <span>Enable</span>
                    </label>
                    <select id="diff-prompt-set-select" style="margin-left: 8px; ${diffMode ? '' : 'display: none;'}">
                        <option value="">Select comparison...</option>
                        ${availableSets.map(s => `<option value="${s}" ${s === diffPromptSet ? 'selected' : ''}>${s.replace(/_/g, ' ')}</option>`).join('')}
                    </select>
                </div>
                <div id="combined-activation-plot"></div>
            </section>

            <section>
                <h3>Token Magnitude <span class="subsection-info-toggle" data-target="info-token-magnitude">►</span></h3>
                <div class="subsection-info" id="info-token-magnitude">L2 norm of activation at best layer per token. Compare to trajectory - similar magnitudes but low projections means token encodes orthogonal information (e.g., punctuation).</div>
                <div id="token-magnitude-plot"></div>
            </section>

            <section>
                <h3>Token Velocity <span class="subsection-info-toggle" data-target="info-token-velocity">►</span></h3>
                <div class="subsection-info" id="info-token-velocity">Rate of change between consecutive tokens (d/dt of trajectory above).</div>
                <div id="token-velocity-plot"></div>
            </section>

            <section>
                <h3>Activation Magnitude <span class="subsection-info-toggle" data-target="info-act-magnitude">►</span></h3>
                <div class="subsection-info" id="info-act-magnitude">How the residual stream grows in magnitude as each layer adds information to the hidden state.</div>
                <div id="activation-magnitude-plot"></div>
            </section>
        </div>
    `;

    // Setup info toggles
    window.setupSubsectionInfoToggles();

    // Prepare data for plotting
    const traitActivations = {};  // Store smoothed activations for velocity/accel

    // Prepare traces for Token Trajectory (always cosine)
    const traces = [];

    // Filter traits by selected methods
    const filteredByMethod = loadedTraits.filter(traitName => {
        const method = traitData[traitName]?.metadata?.vector_source?.method;
        return !method || window.state.selectedMethods.has(method);
    });

    if (filteredByMethod.length === 0) {
        container.querySelector('#combined-activation-plot').innerHTML = `
            <div class="info">No traits match selected methods. Enable more methods above.</div>
        `;
        return;
    }

    filteredByMethod.forEach((traitName, idx) => {
        const data = traitData[traitName];
        const promptProj = data.projections.prompt;
        const responseProj = data.projections.response;
        const allProj = [...promptProj, ...responseProj];

        // Get vector source from metadata
        const vs = data.metadata?.vector_source || {};

        // projections are now 1D arrays (one value per token at best layer)
        let rawProj = allProj.slice(START_TOKEN_IDX);

        // Compute cosine similarity (always use cosine)
        let rawValues;
        if (data.token_norms) {
            const traitTokenNorms = [...data.token_norms.prompt, ...data.token_norms.response].slice(START_TOKEN_IDX);
            rawValues = rawProj.map((proj, i) => {
                const norm = traitTokenNorms[i];
                return norm > 0 ? proj / norm : 0;
            });
        } else {
            rawValues = rawProj;
        }

        // Subtract BOS value if centering is enabled (makes token 0 = 0)
        if (isCentered && rawValues.length > 0) {
            const bosValue = rawValues[0];
            rawValues = rawValues.map(v => v - bosValue);
        }

        // Apply 3-token moving average if smoothing is enabled
        const displayValues = isSmoothing ? smoothData(rawValues, 3) : rawValues;

        // Store raw values for velocity/accel (always use smoothed for derivatives)
        traitActivations[traitName] = isSmoothing ? smoothData(rawProj, 3) : rawProj;

        // Each vector gets its own color
        const color = window.getChartColors()[idx % 10];

        const method = vs.method || 'probe';
        const valueLabel = 'Cosine';
        const valueFormat = '.4f';

        // Build display name and hover
        const baseTrait = data.metadata?._baseTrait || traitName;
        const displayName = data.metadata?._isMultiVector
            ? `${window.getDisplayName(baseTrait)} (${method} L${vs.layer})`
            : window.getDisplayName(traitName);
        const vectorInfo = vs.layer !== undefined ? `<br><span style="color:#888">L${vs.layer} ${method}</span>` : '';
        const hoverText = `<b>${displayName}</b>${vectorInfo}<br>Token %{x}<br>${valueLabel}: %{y:${valueFormat}}<extra></extra>`;

        traces.push({
            x: Array.from({length: displayValues.length}, (_, i) => i),
            y: displayValues,
            type: 'scatter',
            mode: 'lines+markers',
            name: displayName,
            line: { color: color, width: 1.5 },
            marker: { size: 2, color: color },
            hovertemplate: hoverText
        });
    });

    // Get display tokens (every 10th for x-axis labels)
    const displayTokens = allTokens.slice(START_TOKEN_IDX);
    const tickVals = [];
    const tickText = [];
    for (let i = 0; i < displayTokens.length; i += 10) {
        tickVals.push(i);
        tickText.push(displayTokens[i]);
    }

    // Get colors from CSS variables
    const textSecondary = window.getCssVar('--text-secondary', '#a4a4a4');
    const primaryColor = window.getCssVar('--primary-color', '#a09f6c');

    // Current token highlight
    const currentTokenIdx = window.state.currentTokenIndex || 0;
    const highlightX = Math.max(0, currentTokenIdx - START_TOKEN_IDX);

    // Shapes for prompt/response separator and token highlight
    const shapes = [
        {
            type: 'line',
            x0: (nPromptTokens - START_TOKEN_IDX) - 0.5,
            x1: (nPromptTokens - START_TOKEN_IDX) - 0.5,
            y0: 0, y1: 1, yref: 'paper',
            line: { color: textSecondary, width: 2, dash: 'dash' }
        },
        {
            type: 'line',
            x0: highlightX, x1: highlightX,
            y0: 0, y1: 1, yref: 'paper',
            line: { color: primaryColor, width: 2 }
        }
    ];

    const annotations = [
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
    ];

    // Build custom legend with vector source tooltips (like live-chat)
    const legendHtml = filteredByMethod.map((traitName, idx) => {
        const data = traitData[traitName];
        const vs = data.metadata?.vector_source || {};
        const tooltipText = vs.layer !== undefined
            ? `L${vs.layer} ${vs.method || '?'} (${vs.selection_source || 'unknown'})`
            : 'no metadata';

        // Each vector gets its own color (same as traces)
        const color = window.getChartColors()[idx % 10];

        // Display name matches trace name
        const baseTrait = data.metadata?._baseTrait || traitName;
        const displayName = data.metadata?._isMultiVector
            ? `${window.getDisplayName(baseTrait)} (${vs.method} L${vs.layer})`
            : window.getDisplayName(traitName);

        return `
            <span class="legend-item has-tooltip" data-tooltip="${tooltipText}">
                <span class="legend-color" style="background: ${color}"></span>
                ${displayName}
            </span>
        `;
    }).join('');

    // Token Trajectory plot
    const yAxisTitle = 'Cosine (proj / ||h||)';

    // Compute y-axis range: minimum ±0.15, auto-expand if data exceeds
    let yAxisConfig = { title: yAxisTitle, zeroline: true, zerolinewidth: 1, showgrid: true };
    // Find actual data range across all traces
    let minY = Infinity, maxY = -Infinity;
    traces.forEach(t => {
        t.y.forEach(v => {
            if (v < minY) minY = v;
            if (v > maxY) maxY = v;
        });
    });
    // Ensure minimum range of ±0.15, expand if data exceeds
    const minRange = 0.15;
    const rangeMin = Math.min(-minRange, minY - 0.02);
    const rangeMax = Math.max(minRange, maxY + 0.02);
    yAxisConfig.range = [rangeMin, rangeMax];

    const mainLayout = window.getPlotlyLayout({
        xaxis: {
            title: 'Token Position',
            tickmode: 'array',
            tickvals: tickVals,
            ticktext: tickText,
            tickangle: -45,
            showgrid: false,
            tickfont: { size: 9 }
        },
        yaxis: yAxisConfig,
        shapes: shapes,
        annotations: annotations,
        margin: { l: 60, r: 20, t: 40, b: 80 },
        height: 400,
        hovermode: 'closest',
        showlegend: false  // Using custom legend instead
    });

    Plotly.newPlot('combined-activation-plot', traces, mainLayout, { responsive: true, displayModeBar: false });

    // Insert custom legend after plot and setup hover-to-highlight
    const plotDiv = document.getElementById('combined-activation-plot');
    const legendDiv = document.createElement('div');
    legendDiv.className = 'chart-legend';
    legendDiv.innerHTML = legendHtml;
    plotDiv.parentNode.insertBefore(legendDiv, plotDiv.nextSibling);

    // Hover-to-highlight and click-to-select for main trajectory
    plotDiv.on('plotly_hover', (d) =>
        Plotly.restyle(plotDiv, {'opacity': traces.map((_, i) => i === d.points[0].curveNumber ? 1.0 : 0.2)})
    );
    plotDiv.on('plotly_unhover', () => Plotly.restyle(plotDiv, {'opacity': 1.0}));
    plotDiv.on('plotly_click', (d) => {
        const tokenIdx = Math.round(d.points[0].x) + START_TOKEN_IDX;
        if (window.state.currentTokenIndex !== tokenIdx) {
            window.state.currentTokenIndex = tokenIdx;
            window.renderPromptPicker?.();
            window.renderCurrentView?.();
        }
    });

    // Setup smoothing checkbox
    const smoothingCheckbox = document.getElementById('smoothing-toggle');
    if (smoothingCheckbox) {
        smoothingCheckbox.addEventListener('change', () => {
            window.setSmoothing(smoothingCheckbox.checked);
        });
    }

    // Setup centered checkbox
    const centeredCheckbox = document.getElementById('projection-centered-toggle');
    if (centeredCheckbox) {
        centeredCheckbox.addEventListener('change', () => {
            window.setProjectionCentered(centeredCheckbox.checked);
        });
    }

    // Setup method filter checkboxes
    document.querySelectorAll('.method-filter').forEach(cb => {
        cb.addEventListener('change', () => {
            window.toggleMethod(cb.dataset.method);
        });
    });

    // Setup diff mode toggle and dropdown
    const diffToggle = document.getElementById('diff-mode-toggle');
    const diffSelect = document.getElementById('diff-prompt-set-select');
    if (diffToggle) {
        diffToggle.addEventListener('change', () => {
            window.state.diffMode = diffToggle.checked;
            if (diffSelect) {
                diffSelect.style.display = diffToggle.checked ? '' : 'none';
            }
            if (!diffToggle.checked) {
                window.state.diffPromptSet = null;
            }
            window.render();
        });
    }
    if (diffSelect) {
        diffSelect.addEventListener('change', () => {
            window.state.diffPromptSet = diffSelect.value || null;
            window.render();
        });
    }

    // Render Token Magnitude plot (per-token norms)
    renderTokenMagnitudePlot(traitData, filteredByMethod, tickVals, tickText, nPromptTokens);

    // Render Token Velocity plot
    renderTokenDerivativePlots(traitActivations, filteredByMethod, tickVals, tickText, nPromptTokens, traitData);

    // Render Activation Magnitude plot (per-layer)
    renderActivationMagnitudePlot(traitData, filteredByMethod);
}


/**
 * Render Token Magnitude plot showing L2 norm per token at best layer.
 * Helps identify if low projections are due to low magnitude or orthogonal encoding.
 */
function renderTokenMagnitudePlot(traitData, loadedTraits, tickVals, tickText, nPromptTokens) {
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

    const promptNorms = firstTraitData.token_norms.prompt;
    const responseNorms = firstTraitData.token_norms.response;
    const allNorms = [...promptNorms, ...responseNorms].slice(START_TOKEN_IDX);

    const textSecondary = window.getCssVar('--text-secondary', '#a4a4a4');
    const primaryColor = window.getCssVar('--primary-color', '#a09f6c');
    const currentTokenIdx = window.state.currentTokenIndex || 0;
    const highlightX = Math.max(0, currentTokenIdx - START_TOKEN_IDX);

    const trace = {
        y: allNorms,
        type: 'scatter',
        mode: 'lines',
        name: '||h||',
        line: { color: textSecondary, width: 1.5 },
        hovertemplate: 'Token %{x}<br>||h|| = %{y:.1f}<extra></extra>'
    };

    // Prompt/response separator and current token highlight
    const promptEndIdx = nPromptTokens - START_TOKEN_IDX;
    const highlightColors = window.getTokenHighlightColors();

    const layout = window.getPlotlyLayout({
        margin: { l: 50, r: 20, t: 20, b: 40 },
        xaxis: {
            title: 'Token',
            tickvals: tickVals,
            ticktext: tickText,
            tickfont: { size: 9 }
        },
        yaxis: { title: '||h|| (L2 norm)', tickfont: { size: 10 } },
        height: 200,
        showlegend: false,
        shapes: [
            // Prompt/response separator
            { type: 'line', x0: promptEndIdx, x1: promptEndIdx, y0: 0, y1: 1, yref: 'paper',
              line: { color: highlightColors.separator, width: 2, dash: 'dash' } },
            // Current token highlight
            { type: 'line', x0: highlightX, x1: highlightX, y0: 0, y1: 1, yref: 'paper',
              line: { color: highlightColors.highlight, width: 2 } }
        ]
    });

    Plotly.newPlot(plotDiv, [trace], layout, { responsive: true, displayModeBar: false });

    // Click-to-select
    plotDiv.on('plotly_click', (d) => {
        const tokenIdx = Math.round(d.points[0].x) + START_TOKEN_IDX;
        if (window.state.currentTokenIndex !== tokenIdx) {
            window.state.currentTokenIndex = tokenIdx;
            window.renderPromptPicker?.();
            window.renderCurrentView?.();
        }
    });
}


/**
 * Render Token Velocity plot (first derivative of smoothed trajectory)
 */
function renderTokenDerivativePlots(traitActivations, loadedTraits, tickVals, tickText, nPromptTokens, traitData) {
    const textSecondary = window.getCssVar('--text-secondary', '#a4a4a4');
    const primaryColor = window.getCssVar('--primary-color', '#a09f6c');
    const currentTokenIdx = window.state.currentTokenIndex || 0;
    const highlightX = Math.max(0, currentTokenIdx - START_TOKEN_IDX);

    // Velocity traces
    const velocityTraces = [];
    loadedTraits.forEach((traitName, idx) => {
        const activations = traitActivations[traitName];
        const velocity = computeVelocity(activations);
        const smoothedVelocity = smoothData(velocity, 3);
        const color = window.getChartColors()[idx % 10];

        const vs = traitData[traitName]?.metadata?.vector_source || {};
        const method = vs.method || 'probe';
        const vectorInfo = vs.layer !== undefined ? `<br><span style="color:#888">L${vs.layer} ${method}</span>` : '';

        velocityTraces.push({
            x: Array.from({length: smoothedVelocity.length}, (_, i) => i + 0.5),
            y: smoothedVelocity,
            type: 'scatter',
            mode: 'lines',
            name: window.getDisplayName(traitName),
            line: { color: color, width: 1.5 },
            hovertemplate: `<b>${window.getDisplayName(traitName)}</b>${vectorInfo}<br>Token %{x:.0f}<br>Velocity: %{y:.4f}<extra></extra>`
        });
    });

    const shapes = [
        { type: 'line', x0: (nPromptTokens - START_TOKEN_IDX) - 0.5, x1: (nPromptTokens - START_TOKEN_IDX) - 0.5,
          y0: 0, y1: 1, yref: 'paper', line: { color: textSecondary, width: 1, dash: 'dash' } },
        { type: 'line', x0: highlightX, x1: highlightX,
          y0: 0, y1: 1, yref: 'paper', line: { color: primaryColor, width: 2 } }
    ];

    const velocityLayout = window.getPlotlyLayout({
        xaxis: { title: '', tickmode: 'array', tickvals: tickVals, ticktext: tickText, tickangle: -45, tickfont: { size: 8 }, showgrid: true },
        yaxis: { title: 'Velocity', zeroline: true, zerolinewidth: 1, zerolinecolor: textSecondary, showgrid: true },
        shapes: shapes,
        margin: { l: 50, r: 20, t: 10, b: 80 },
        height: 300,
        showlegend: false
    });

    Plotly.newPlot('token-velocity-plot', velocityTraces, velocityLayout, { responsive: true, displayModeBar: false });

    // Click handler to update token slider
    const velocityPlot = document.getElementById('token-velocity-plot');
    velocityPlot.on('plotly_click', (d) => {
        const tokenIdx = Math.round(d.points[0].x) + START_TOKEN_IDX;
        if (window.state.currentTokenIndex !== tokenIdx) {
            window.state.currentTokenIndex = tokenIdx;
            window.renderPromptPicker?.();
            window.renderCurrentView?.();
        }
    });
}

/**
 * Render the Activation Magnitude plot showing ||h|| by layer (layer on y-axis).
 */
function renderActivationMagnitudePlot(traitData, loadedTraits) {
    const firstTraitData = traitData[loadedTraits[0]];

    if (!firstTraitData.activation_norms) {
        const plotDiv = document.getElementById('activation-magnitude-plot');
        plotDiv.innerHTML = `
            <div class="info">
                Activation norms not available. Re-run projection script to generate.
            </div>
        `;
        return;
    }

    const promptNorms = firstTraitData.activation_norms.prompt;
    const responseNorms = firstTraitData.activation_norms.response;
    const nLayers = promptNorms.length;
    const layerIndices = Array.from({length: nLayers}, (_, i) => i);
    const combinedNorms = promptNorms.map((p, i) => (p + responseNorms[i]) / 2);

    const textSecondary = window.getCssVar('--text-secondary', '#a4a4a4');

    // Layer on x-axis, L2 norm on y-axis
    const traces = [
        { x: layerIndices, y: promptNorms, type: 'scatter', mode: 'lines+markers', name: 'Prompt',
          line: { color: '#4a9eff', width: 2 }, marker: { size: 4 },
          hovertemplate: '<b>Prompt</b><br>Layer %{x}: %{y:.1f}<extra></extra>' },
        { x: layerIndices, y: responseNorms, type: 'scatter', mode: 'lines+markers', name: 'Response',
          line: { color: '#ff6b6b', width: 2 }, marker: { size: 4 },
          hovertemplate: '<b>Response</b><br>Layer %{x}: %{y:.1f}<extra></extra>' },
        { x: layerIndices, y: combinedNorms, type: 'scatter', mode: 'lines+markers', name: 'Combined',
          line: { color: textSecondary, width: 2, dash: 'dash' }, marker: { size: 4 },
          hovertemplate: '<b>Combined</b><br>Layer %{x}: %{y:.1f}<extra></extra>' }
    ];

    const layout = window.getPlotlyLayout({
        xaxis: { title: 'Layer', tickmode: 'linear', tick0: 0, dtick: 5, showgrid: true },
        yaxis: { title: '||h|| (L2 norm)', showgrid: true },
        margin: { l: 50, r: 20, t: 10, b: 40 },
        height: 300,
        legend: { orientation: 'h', yanchor: 'top', y: -0.15, xanchor: 'center', x: 0.5, font: { size: 10 } },
        showlegend: true
    });

    Plotly.newPlot('activation-magnitude-plot', traces, layout, { responsive: true, displayModeBar: false });
}


// Export to global scope
window.renderTraitDynamics = renderTraitDynamics;
