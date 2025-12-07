// Trait Dynamics View - Watch the model's internal state evolve token-by-token
// Core insight: "See how the model is thinking" by projecting onto trait vectors
//
// Sections:
// 1. Token Trajectory: X=tokens, Y=activation (layer-averaged) + velocity/acceleration
// 2. Normalized layer derivatives (position, velocity, acceleration) - compact row
// 3. Activation Magnitude
// 4. Layer×Token Heatmaps

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

/**
 * Compute layer-averaged position, velocity, and acceleration for a trait.
 * Position = average projection across all tokens per layer
 * Velocity = first derivative (diff between layers)
 * Acceleration = second derivative (diff of velocity)
 *
 * If activationNorms provided, normalizes position by ||h|| at each layer first,
 * then computes derivatives. This separates "direction" from "magnitude".
 *
 * @param {Object} data - Trait projection data with projections.prompt and projections.response
 * @param {number[]} activationNorms - Optional per-layer activation norms for normalization
 * @returns {Object} { position: [nLayers], velocity: [nLayers-1], acceleration: [nLayers-2] }
 */
function computeLayerDerivatives(data, activationNorms = null) {
    const promptProj = data.projections.prompt;
    const responseProj = data.projections.response;
    const allProj = [...promptProj, ...responseProj];
    const nLayers = promptProj[0].length;
    const nSublayers = promptProj[0][0].length;

    // Compute layer-averaged position (average across all tokens and sublayers)
    const rawPosition = [];
    for (let layer = 0; layer < nLayers; layer++) {
        let sum = 0;
        let count = 0;
        for (let token = 0; token < allProj.length; token++) {
            for (let sublayer = 0; sublayer < nSublayers; sublayer++) {
                sum += allProj[token][layer][sublayer];
                count++;
            }
        }
        rawPosition.push(sum / count);
    }

    // Normalize by activation magnitude if provided
    const position = activationNorms
        ? rawPosition.map((p, i) => p / (activationNorms[i] || 1))
        : rawPosition;

    // Compute velocity (first derivative)
    const velocity = [];
    for (let i = 0; i < position.length - 1; i++) {
        velocity.push(position[i + 1] - position[i]);
    }

    // Compute acceleration (second derivative)
    const acceleration = [];
    for (let i = 0; i < velocity.length - 1; i++) {
        acceleration.push(velocity[i + 1] - velocity[i]);
    }

    return { position, velocity, acceleration };
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

    // Load data for ALL selected traits
    for (const trait of filteredTraits) {
        try {
            const fetchPath = window.paths.residualStreamData(trait, promptSet, promptId);
            const response = await fetch(fetchPath);

            if (!response.ok) {
                failedTraits.push(trait.name);
                continue;
            }

            const data = await response.json();
            traitData[trait.name] = data;
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

    // Build HTML
    let failedHtml = '';
    if (failedTraits.length > 0) {
        failedHtml = `
            <div class="tool-description">
                No data for: ${failedTraits.map(t => window.getDisplayName(t)).join(', ')}
            </div>
        `;
    }

    container.innerHTML = `
        <div class="tool-view">
            <div class="page-intro">
                <div class="page-intro-text">Watch traits evolve token-by-token during generation.</div>
            </div>
            ${failedHtml}

            <section>
                <h2>Token Trajectory <span class="subsection-info-toggle" data-target="info-token-trajectory">►</span></h2>
                <div class="subsection-info" id="info-token-trajectory">Layer-averaged projection per token. 3-token smoothing applied.</div>
                <div id="combined-activation-plot"></div>
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
                <div class="layer-plots-row">
                    <div class="layer-plot-item">
                        <h3>Layer Profile <span class="subsection-info-toggle" data-target="info-layer-derivs">►</span></h3>
                        <div class="subsection-info" id="info-layer-derivs">Shows which layers each trait is most active in. Normalized projection (proj/‖h‖) averaged across all tokens.</div>
                        <div id="layer-position-plot"></div>
                    </div>
                    <div class="layer-plot-item">
                        <h3>Activation Magnitude <span class="subsection-info-toggle" data-target="info-act-magnitude">►</span></h3>
                        <div class="subsection-info" id="info-act-magnitude">How the residual stream grows in magnitude as each layer adds information to the hidden state.</div>
                        <div id="activation-magnitude-plot"></div>
                    </div>
                </div>
            </section>

            <section>
                <h2>Layer × Token Heatmaps <span class="subsection-info-toggle" data-target="info-heatmaps">►</span></h2>
                <div class="subsection-info" id="info-heatmaps">y=layer, x=token, color=projection. How trait signal builds through layers for each token.</div>
                <div id="trait-heatmaps-container"></div>
            </section>
        </div>
    `;

    // Setup info toggles
    setupSubsectionInfoToggles();

    // Prepare data for plotting
    const startIdx = 1;  // Skip BOS token
    const traitActivations = {};  // Store smoothed activations for velocity/accel

    // Prepare traces for Token Trajectory
    const traces = [];

    loadedTraits.forEach((traitName, idx) => {
        const data = traitData[traitName];
        const promptProj = data.projections.prompt;
        const responseProj = data.projections.response;
        const allProj = [...promptProj, ...responseProj];
        const nLayers = promptProj[0].length;

        // Calculate activation strength for each token (average across all layers and sublayers)
        const rawActivations = [];
        for (let t = startIdx; t < allProj.length; t++) {
            let sum = 0;
            let count = 0;
            for (let l = 0; l < nLayers; l++) {
                for (let s = 0; s < 3; s++) {
                    sum += allProj[t][l][s];
                    count++;
                }
            }
            rawActivations.push(sum / count);
        }

        // Apply 3-token moving average
        const activations = smoothData(rawActivations, 3);
        traitActivations[traitName] = activations;

        const color = TRAIT_COLORS[idx % TRAIT_COLORS.length];

        traces.push({
            x: Array.from({length: activations.length}, (_, i) => i),
            y: activations,
            type: 'scatter',
            mode: 'lines+markers',
            name: window.getDisplayName(traitName),
            line: { color: color, width: 1 },
            marker: { size: 1, color: color },
            hovertemplate: `<b>${window.getDisplayName(traitName)}</b><br>Token %{x}<br>Activation: %{y:.3f}<extra></extra>`
        });
    });

    // Get display tokens (every 10th for x-axis labels)
    const displayTokens = allTokens.slice(startIdx);
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
    const highlightX = currentTokenIdx - startIdx;

    // Shapes for prompt/response separator and token highlight
    const shapes = [
        {
            type: 'line',
            x0: (nPromptTokens - startIdx) - 0.5,
            x1: (nPromptTokens - startIdx) - 0.5,
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
            x: (nPromptTokens - startIdx) / 2 - 0.5,
            y: 1.08, yref: 'paper',
            text: 'PROMPT', showarrow: false,
            font: { size: 11, color: textSecondary }
        },
        {
            x: (nPromptTokens - startIdx) + (displayTokens.length - (nPromptTokens - startIdx)) / 2 - 0.5,
            y: 1.08, yref: 'paper',
            text: 'RESPONSE', showarrow: false,
            font: { size: 11, color: textSecondary }
        }
    ];

    // Token Trajectory plot
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
        yaxis: {
            title: 'Activation (3-token avg)',
            zeroline: true, zerolinewidth: 1, showgrid: true
        },
        shapes: shapes,
        annotations: annotations,
        margin: { l: 60, r: 20, t: 40, b: 100 },
        height: 400,
        hovermode: 'closest',
        legend: { orientation: 'h', yanchor: 'top', y: -0.2, xanchor: 'center', x: 0.5, font: { size: 10 } },
        showlegend: true
    });

    Plotly.newPlot('combined-activation-plot', traces, mainLayout, { responsive: true, displayModeBar: false });

    // Hover-to-highlight and click-to-select
    const plotDiv = document.getElementById('combined-activation-plot');
    plotDiv.on('plotly_hover', (d) =>
        Plotly.restyle(plotDiv, {'opacity': traces.map((_, i) => i === d.points[0].curveNumber ? 1.0 : 0.2)})
    );
    plotDiv.on('plotly_unhover', () => Plotly.restyle(plotDiv, {'opacity': 1.0}));
    plotDiv.on('plotly_click', (d) => {
        const tokenIdx = Math.round(d.points[0].x) + startIdx;
        if (window.state.currentTokenIndex !== tokenIdx) {
            window.state.currentTokenIndex = tokenIdx;
            window.renderPromptPicker?.();
            window.renderCurrentView?.();
        }
    });

    // Render Token Velocity and Acceleration plots
    renderTokenDerivativePlots(traitActivations, loadedTraits, tickVals, tickText, nPromptTokens, startIdx);

    // Render Layer derivative plots (position, velocity, acceleration) - compact
    renderLayerDerivativePlots(traitData, loadedTraits);

    // Render Activation Magnitude plot
    renderActivationMagnitudePlot(traitData, loadedTraits);

    // Render Layer×Token Heatmaps (migrated from trait-trajectory)
    renderLayerTokenHeatmaps(traitData, loadedTraits, allTokens, nPromptTokens);
}


/**
 * Render Token Velocity and Token Acceleration plots (derivatives of smoothed trajectory)
 */
function renderTokenDerivativePlots(traitActivations, loadedTraits, tickVals, tickText, nPromptTokens, startIdx) {
    const textSecondary = window.getCssVar('--text-secondary', '#a4a4a4');
    const primaryColor = window.getCssVar('--primary-color', '#a09f6c');
    const currentTokenIdx = window.state.currentTokenIndex || 0;
    const highlightX = currentTokenIdx - startIdx;

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
        { type: 'line', x0: (nPromptTokens - startIdx) - 0.5, x1: (nPromptTokens - startIdx) - 0.5,
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
        const tokenIdx = Math.round(d.points[0].x) + startIdx;
        if (window.state.currentTokenIndex !== tokenIdx) {
            window.state.currentTokenIndex = tokenIdx;
            window.renderPromptPicker?.();
            window.renderCurrentView?.();
        }
    });

    accelPlot.on('plotly_click', (d) => {
        const tokenIdx = Math.round(d.points[0].x) + startIdx;
        if (window.state.currentTokenIndex !== tokenIdx) {
            window.state.currentTokenIndex = tokenIdx;
            window.renderPromptPicker?.();
            window.renderCurrentView?.();
        }
    });
}


/**
 * Render layer profile plot showing normalized projection per layer.
 */
function renderLayerDerivativePlots(traitData, loadedTraits) {
    const firstData = traitData[loadedTraits[0]];
    const nLayers = firstData.projections.prompt[0]?.length || 26;
    const layerIndices = Array.from({length: nLayers}, (_, i) => i);

    // Get activation norms for normalization
    let combinedNorms = null;
    if (firstData.activation_norms) {
        const promptNorms = firstData.activation_norms.prompt;
        const responseNorms = firstData.activation_norms.response;
        combinedNorms = promptNorms.map((p, i) => (p + responseNorms[i]) / 2);
    }

    // Compute derivatives for all traits
    const allDerivatives = {};
    loadedTraits.forEach((traitName) => {
        allDerivatives[traitName] = computeLayerDerivatives(traitData[traitName], combinedNorms);
    });

    // Create traces (layers on y-axis)
    const traces = [];
    loadedTraits.forEach((traitName, idx) => {
        const derivatives = allDerivatives[traitName];
        const traitColor = TRAIT_COLORS[idx % TRAIT_COLORS.length];
        traces.push({
            x: derivatives.position,
            y: layerIndices,
            type: 'scatter',
            mode: 'lines+markers',
            name: window.getDisplayName(traitName),
            line: { color: traitColor, width: 1.5 },
            marker: { size: 3 },
            hovertemplate: `<b>${window.getDisplayName(traitName)}</b><br>Layer %{y}: %{x:.3f}<extra></extra>`
        });
    });

    const layout = window.getPlotlyLayout({
        xaxis: { title: 'Normalized Projection', tickfont: { size: 8 }, zeroline: true, showgrid: true },
        yaxis: { title: 'Layer', tickmode: 'linear', tick0: 0, dtick: 5, tickfont: { size: 8 }, showgrid: true },
        margin: { l: 40, r: 20, t: 10, b: 60 },
        height: 300,
        showlegend: true,
        legend: { orientation: 'h', yanchor: 'top', y: -0.22, xanchor: 'center', x: 0.5, font: { size: 10 } }
    });

    Plotly.newPlot('layer-position-plot', traces, layout, { responsive: true, displayModeBar: false });
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

    // Layer on y-axis, L2 norm on x-axis
    const traces = [
        { x: promptNorms, y: layerIndices, type: 'scatter', mode: 'lines+markers', name: 'Prompt',
          line: { color: '#4a9eff', width: 2 }, marker: { size: 4 },
          hovertemplate: '<b>Prompt</b><br>Layer %{y}: %{x:.1f}<extra></extra>' },
        { x: responseNorms, y: layerIndices, type: 'scatter', mode: 'lines+markers', name: 'Response',
          line: { color: '#ff6b6b', width: 2 }, marker: { size: 4 },
          hovertemplate: '<b>Response</b><br>Layer %{y}: %{x:.1f}<extra></extra>' },
        { x: combinedNorms, y: layerIndices, type: 'scatter', mode: 'lines+markers', name: 'Combined',
          line: { color: textSecondary, width: 2, dash: 'dash' }, marker: { size: 4 },
          hovertemplate: '<b>Combined</b><br>Layer %{y}: %{x:.1f}<extra></extra>' }
    ];

    const layout = window.getPlotlyLayout({
        xaxis: { title: '||h|| (L2 norm)', showgrid: true },
        yaxis: { title: 'Layer', tickmode: 'linear', tick0: 0, dtick: 5, showgrid: true },
        margin: { l: 40, r: 20, t: 10, b: 60 },
        height: 300,
        legend: { orientation: 'h', yanchor: 'top', y: -0.22, xanchor: 'center', x: 0.5, font: { size: 10 } },
        showlegend: true
    });

    Plotly.newPlot('activation-magnitude-plot', traces, layout, { responsive: true, displayModeBar: false });
}


/**
 * Render Layer×Token Heatmaps for each trait (migrated from trait-trajectory)
 */
function renderLayerTokenHeatmaps(traitData, loadedTraits, allTokens, nPromptTokens) {
    const container = document.getElementById('trait-heatmaps-container');
    if (!container) return;

    // Create container for all heatmaps
    let html = '<div class="trait-heatmaps-grid">';
    loadedTraits.forEach((traitName) => {
        const traitId = traitName.replace(/\//g, '-');
        html += `
            <div class="trait-heatmap-item">
                <h4 title="${window.getDisplayName(traitName)}">${window.getDisplayName(traitName)}</h4>
                <div id="heatmap-${traitId}" class="trait-heatmap-plot"></div>
            </div>
        `;
    });
    html += '</div>';
    container.innerHTML = html;

    // Render each heatmap
    loadedTraits.forEach((traitName) => {
        const data = traitData[traitName];
        const promptProj = data.projections.prompt;
        const responseProj = data.projections.response;
        const allProj = [...promptProj, ...responseProj];

        const nLayers = promptProj[0].length;
        const startIdx = 1;  // Skip BOS

        // Average over sublayers to get [n_tokens, n_layers]
        const layerAvg = [];
        for (let t = startIdx; t < allProj.length; t++) {
            layerAvg[t - startIdx] = [];
            for (let l = 0; l < nLayers; l++) {
                const avg = (allProj[t][l][0] + allProj[t][l][1] + allProj[t][l][2]) / 3;
                layerAvg[t - startIdx][l] = avg;
            }
        }

        // Transpose for heatmap: [n_layers, n_tokens]
        const heatmapData = [];
        const nDisplayTokens = allProj.length - startIdx;
        for (let l = 0; l < nLayers; l++) {
            heatmapData[l] = [];
            for (let t = 0; t < nDisplayTokens; t++) {
                heatmapData[l][t] = layerAvg[t][l];
            }
        }

        const traitId = traitName.replace(/\//g, '-');
        const trace = {
            z: heatmapData,
            y: Array.from({length: nLayers}, (_, i) => `L${i}`),
            type: 'heatmap',
            colorscale: window.ASYMB_COLORSCALE,
            zmid: 0,
            showscale: false,
            hovertemplate: 'Layer: %{y}<br>Token: %{x}<br>Score: %{z:.2f}<extra></extra>'
        };

        const layout = window.getPlotlyLayout({
            xaxis: { showticklabels: false, title: '' },  // No x-axis labels
            yaxis: { tickfont: { size: 8 }, title: '' },
            margin: { l: 25, r: 5, t: 5, b: 5 },
            height: 180
        });

        Plotly.newPlot(`heatmap-${traitId}`, [trace], layout, { displayModeBar: false, responsive: true });
    });
}


// Export to global scope
window.renderTraitDynamics = renderTraitDynamics;
