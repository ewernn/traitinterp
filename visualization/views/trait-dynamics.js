// Trait Dynamics View - Watch the model's internal state evolve token-by-token
// Core insight: "See how the model is thinking" by projecting onto trait vectors
//
// Sections:
// 1. Token Trajectory: X=tokens, Y=projection (best layer) + velocity/acceleration
// 2. Activation Magnitude

// Color palette for traits (distinct, colorblind-friendly)
const TRAIT_COLORS = [
    '#4a9eff',  // blue
    '#ff6b6b',  // red
    '#51cf66',  // green
    '#ffd43b',  // yellow
    '#cc5de8',  // purple
    '#ff922b',  // orange
    '#20c997',  // teal
    '#f06595',  // pink
    '#748ffc',  // indigo
    '#a9e34b',  // lime
];

// Tokens to skip: BOS (0) + undifferentiated warmup tokens (1)
// Early tokens have uniform activations across prompts - not informative
const START_TOKEN_IDX = 2;

/**
 * Setup click handlers for subsection info toggles (► triangles)
 */
function setupSubsectionInfoToggles() {
    const container = document.querySelector('.tool-view');
    if (!container || container.dataset.togglesSetup) return;
    container.dataset.togglesSetup = 'true';

    container.addEventListener('click', (e) => {
        const toggle = e.target.closest('.subsection-info-toggle');
        if (!toggle) return;

        const targetId = toggle.dataset.target;
        const infoDiv = document.getElementById(targetId);
        if (infoDiv) {
            const isShown = infoDiv.classList.toggle('show');
            toggle.textContent = isShown ? '▼' : '►';
        }
    });
}

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

            // Merge response data with projection data (projection is slim, needs tokens)
            if (responseData) {
                projData.prompt = responseData.prompt;
                projData.response = responseData.response;
                // Preserve projection metadata but add inference model from response
                if (responseData.metadata?.inference_model && !projData.metadata?.inference_model) {
                    projData.metadata = projData.metadata || {};
                    projData.metadata.inference_model = responseData.metadata.inference_model;
                }
            }

            traitData[trait.name] = projData;
        } catch (error) {
            failedTraits.push(trait.name);
        }
    }

    clearTimeout(loadingTimeout);

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

    // Build model info HTML
    let modelInfoHtml = `Inference model: <code>${inferenceModel}</code>`;
    if (vectorSource.model) {
        modelInfoHtml += ` · Vector from: <code>${vectorSource.model}</code>`;
    }
    if (vectorSource.method && vectorSource.layer !== undefined) {
        modelInfoHtml += ` (${vectorSource.method} L${vectorSource.layer})`;
    }

    // Build HTML
    let failedHtml = '';
    if (failedTraits.length > 0) {
        failedHtml = `
            <div class="tool-description">
                No data for: ${failedTraits.map(t => window.getDisplayName(t)).join(', ')}
            </div>
        `;
    }

    // Determine projection mode and centering
    const projectionMode = window.state.projectionMode || 'cosine';
    const isCosine = projectionMode === 'cosine';
    const isCentered = window.state.projectionCentered !== false;  // default true

    container.innerHTML = `
        <div class="tool-view">
            <div class="page-intro">
                <div class="page-intro-text">Watch traits evolve token-by-token during generation.</div>
                <div class="page-intro-model">${modelInfoHtml}</div>
            </div>
            ${failedHtml}

            <section>
                <h2>Token Trajectory <span class="subsection-info-toggle" data-target="info-token-trajectory">►</span></h2>
                <div class="subsection-info" id="info-token-trajectory">
                    ${isCosine
                        ? 'Cosine similarity: proj / ||h||. Shows directional alignment with trait vector, independent of activation magnitude.'
                        : 'Raw projection: a·v / ||v||. Shows absolute trait signal strength (affected by activation magnitude).'}
                    ${isCentered ? ' Centered by subtracting training baseline.' : ''}
                </div>
                <div class="projection-toggle">
                    <span class="projection-toggle-label">Mode:</span>
                    <div class="projection-toggle-btns">
                        <button class="projection-toggle-btn ${isCosine ? 'active' : ''}" data-mode="cosine">Cosine</button>
                        <button class="projection-toggle-btn ${!isCosine ? 'active' : ''}" data-mode="vnorm">Magnitude</button>
                    </div>
                    <label class="projection-toggle-checkbox">
                        <input type="checkbox" id="projection-centered-toggle" ${isCentered ? 'checked' : ''}>
                        <span>Centered</span>
                    </label>
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
                <h3>Token Acceleration <span class="subsection-info-toggle" data-target="info-token-accel">►</span></h3>
                <div class="subsection-info" id="info-token-accel">Second derivative - where trajectory speeds up or slows down.</div>
                <div id="token-acceleration-plot"></div>
            </section>

            <section>
                <h3>Activation Magnitude <span class="subsection-info-toggle" data-target="info-act-magnitude">►</span></h3>
                <div class="subsection-info" id="info-act-magnitude">How the residual stream grows in magnitude as each layer adds information to the hidden state.</div>
                <div id="activation-magnitude-plot"></div>
            </section>
        </div>
    `;

    // Setup info toggles
    setupSubsectionInfoToggles();

    // Prepare data for plotting
    const traitActivations = {};  // Store smoothed activations for velocity/accel

    // Prepare traces for Token Trajectory (cosine or vnorm based on mode)
    const traces = [];

    // Get token norms from first trait (same for all traits since it's trait-independent at best layer)
    const firstTraitData = traitData[loadedTraits[0]];
    const hasTokenNorms = firstTraitData.token_norms != null;
    let allTokenNorms = null;
    if (hasTokenNorms) {
        const promptNorms = firstTraitData.token_norms.prompt;
        const responseNorms = firstTraitData.token_norms.response;
        allTokenNorms = [...promptNorms, ...responseNorms].slice(START_TOKEN_IDX);
    }

    // Warn if cosine mode but no token norms
    const canUseCosine = hasTokenNorms && allTokenNorms && allTokenNorms.length > 0;
    const effectiveMode = isCosine && canUseCosine ? 'cosine' : 'vnorm';

    loadedTraits.forEach((traitName, idx) => {
        const data = traitData[traitName];
        const promptProj = data.projections.prompt;
        const responseProj = data.projections.response;
        const allProj = [...promptProj, ...responseProj];

        // Get baseline from metadata (0 if not available)
        const baseline = data.metadata?.vector_source?.baseline || 0;

        // projections are now 1D arrays (one value per token at best layer)
        let rawVnorm = allProj.slice(START_TOKEN_IDX);

        // Subtract baseline if centering is enabled (for vnorm mode)
        if (isCentered && baseline !== 0) {
            rawVnorm = rawVnorm.map(v => v - baseline);
        }

        // Compute values based on mode
        let rawValues;
        if (effectiveMode === 'cosine' && canUseCosine) {
            // For cosine, divide by token norm (baseline already subtracted from vnorm)
            rawValues = rawVnorm.map((proj, i) => {
                const norm = allTokenNorms[i];
                return norm > 0 ? proj / norm : 0;
            });
        } else {
            rawValues = rawVnorm;
        }

        // Apply 3-token moving average
        const smoothedValues = smoothData(rawValues, 3);

        // Store vnorm activations for velocity/accel (always use vnorm for derivatives)
        traitActivations[traitName] = smoothData(rawVnorm, 3);

        const color = TRAIT_COLORS[idx % TRAIT_COLORS.length];
        const valueLabel = effectiveMode === 'cosine' ? 'Cosine' : 'Projection';
        const valueFormat = effectiveMode === 'cosine' ? '.4f' : '.3f';

        traces.push({
            x: Array.from({length: smoothedValues.length}, (_, i) => i),
            y: smoothedValues,
            type: 'scatter',
            mode: 'lines+markers',
            name: window.getDisplayName(traitName),
            line: { color: color, width: 1 },
            marker: { size: 1, color: color },
            hovertemplate: `<b>${window.getDisplayName(traitName)}</b><br>Token %{x}<br>${valueLabel}: %{y:${valueFormat}}<extra></extra>`
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
    const legendHtml = loadedTraits.map((traitName, idx) => {
        const data = traitData[traitName];
        const vs = data.metadata?.vector_source || {};
        const tooltipText = vs.layer !== undefined
            ? `L${vs.layer} ${vs.method || '?'} (${vs.selection_source || 'unknown'})`
            : 'no metadata';
        const color = TRAIT_COLORS[idx % TRAIT_COLORS.length];
        return `
            <span class="legend-item has-tooltip" data-tooltip="${tooltipText}">
                <span class="legend-color" style="background: ${color}"></span>
                ${window.getDisplayName(traitName)}
            </span>
        `;
    }).join('');

    // Token Trajectory plot
    const yAxisTitle = effectiveMode === 'cosine' ? 'Cosine (proj / ||h||)' : 'Projection (a·v / ||v||)';

    // Compute y-axis range: minimum ±0.15 for cosine mode, auto-expand if needed
    let yAxisConfig = { title: yAxisTitle, zeroline: true, zerolinewidth: 1, showgrid: true };
    if (effectiveMode === 'cosine') {
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
    }

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

    // Setup projection mode toggle buttons
    document.querySelectorAll('.projection-toggle-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const mode = btn.dataset.mode;
            if (mode && mode !== window.state.projectionMode) {
                window.setProjectionMode(mode);
            }
        });
    });

    // Setup centered checkbox
    const centeredCheckbox = document.getElementById('projection-centered-toggle');
    if (centeredCheckbox) {
        centeredCheckbox.addEventListener('change', () => {
            window.setProjectionCentered(centeredCheckbox.checked);
        });
    }

    // Render Token Magnitude plot (per-token norms)
    renderTokenMagnitudePlot(traitData, loadedTraits, tickVals, tickText, nPromptTokens);

    // Render Token Velocity and Acceleration plots
    renderTokenDerivativePlots(traitActivations, loadedTraits, tickVals, tickText, nPromptTokens);

    // Render Activation Magnitude plot (per-layer)
    renderActivationMagnitudePlot(traitData, loadedTraits);
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
 * Render Token Velocity and Token Acceleration plots (derivatives of smoothed trajectory)
 */
function renderTokenDerivativePlots(traitActivations, loadedTraits, tickVals, tickText, nPromptTokens) {
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
        const color = TRAIT_COLORS[idx % TRAIT_COLORS.length];

        velocityTraces.push({
            x: Array.from({length: smoothedVelocity.length}, (_, i) => i + 0.5),
            y: smoothedVelocity,
            type: 'scatter',
            mode: 'lines',
            name: window.getDisplayName(traitName),
            line: { color: color, width: 1.5 },
            hovertemplate: `<b>${window.getDisplayName(traitName)}</b><br>Token %{x:.0f}<br>Velocity: %{y:.4f}<extra></extra>`
        });
    });

    // Acceleration traces
    const accelTraces = [];
    loadedTraits.forEach((traitName, idx) => {
        const activations = traitActivations[traitName];
        const velocity = computeVelocity(activations);
        const acceleration = computeVelocity(velocity);
        const smoothedAccel = smoothData(acceleration, 3);
        const color = TRAIT_COLORS[idx % TRAIT_COLORS.length];

        accelTraces.push({
            x: Array.from({length: smoothedAccel.length}, (_, i) => i + 1),
            y: smoothedAccel,
            type: 'scatter',
            mode: 'lines',
            name: window.getDisplayName(traitName),
            line: { color: color, width: 1.5 },
            hovertemplate: `<b>${window.getDisplayName(traitName)}</b><br>Token %{x:.0f}<br>Acceleration: %{y:.4f}<extra></extra>`
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

    const accelLayout = window.getPlotlyLayout({
        xaxis: { title: '', tickmode: 'array', tickvals: tickVals, ticktext: tickText, tickangle: -45, tickfont: { size: 8 }, showgrid: true },
        yaxis: { title: 'Acceleration', zeroline: true, zerolinewidth: 1, zerolinecolor: textSecondary, showgrid: true },
        shapes: shapes,
        margin: { l: 50, r: 20, t: 10, b: 80 },
        height: 300,
        showlegend: false
    });

    Plotly.newPlot('token-velocity-plot', velocityTraces, velocityLayout, { responsive: true, displayModeBar: false });
    Plotly.newPlot('token-acceleration-plot', accelTraces, accelLayout, { responsive: true, displayModeBar: false });

    // Click handlers to update token slider
    const velocityPlot = document.getElementById('token-velocity-plot');
    const accelPlot = document.getElementById('token-acceleration-plot');

    velocityPlot.on('plotly_click', (d) => {
        const tokenIdx = Math.round(d.points[0].x) + START_TOKEN_IDX;
        if (window.state.currentTokenIndex !== tokenIdx) {
            window.state.currentTokenIndex = tokenIdx;
            window.renderPromptPicker?.();
            window.renderCurrentView?.();
        }
    });

    accelPlot.on('plotly_click', (d) => {
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
