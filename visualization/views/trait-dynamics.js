// Trait Dynamics View - Watch the model's internal state evolve token-by-token
// Core insight: "See how the model is thinking" by projecting onto trait vectors
//
// Consolidated view including:
// 1. Token Trajectory: X=tokens, Y=activation (layer-averaged) + velocity/acceleration
// 2. Normalized layer derivatives (position, velocity, acceleration) - compact row
// 3. Activation Magnitude
// 4. Layer×Token Heatmaps (migrated from trait-trajectory)
// 5. Activation Velocity + Trait Coupling (migrated from analysis-gallery)

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

    // Show loading state
    contentArea.innerHTML = `
        <div class="tool-view">
            <div class="page-intro">
                <div class="page-intro-text">Watch traits evolve token-by-token during generation.</div>
            </div>
            <div class="info">Loading data for ${filteredTraits.length} trait(s)...</div>
        </div>
    `;

    const traitData = {};
    const failedTraits = [];
    const promptSet = window.state.currentPromptSet;
    const promptId = window.state.currentPromptId;

    if (!promptSet || !promptId) {
        renderNoDataMessage(contentArea, filteredTraits, promptSet, promptId);
        return;
    }

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

    // Check if we have any data
    const loadedTraits = Object.keys(traitData);
    if (loadedTraits.length === 0) {
        renderNoDataMessage(contentArea, filteredTraits, promptSet, promptId);
        return;
    }

    // Render the full view
    renderCombinedGraph(contentArea, traitData, loadedTraits, failedTraits, promptSet, promptId);
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
                <h3>Layer Derivatives (Normalized) <span class="subsection-info-toggle" data-target="info-layer-derivs">►</span></h3>
                <div class="subsection-info" id="info-layer-derivs">proj/‖h‖ averaged across all tokens, showing how trait signal builds through layers.</div>
                <div class="layer-derivatives-row">
                    <div class="layer-deriv-item">
                        <div class="layer-deriv-label">Position</div>
                        <div id="layer-position-plot"></div>
                    </div>
                    <div class="layer-deriv-item">
                        <div class="layer-deriv-label">Velocity</div>
                        <div id="layer-velocity-plot"></div>
                    </div>
                    <div class="layer-deriv-item">
                        <div class="layer-deriv-label">Acceleration</div>
                        <div id="layer-acceleration-plot"></div>
                    </div>
                </div>
            </section>

            <section>
                <h3>Activation Magnitude <span class="subsection-info-toggle" data-target="info-act-magnitude">►</span></h3>
                <div class="subsection-info" id="info-act-magnitude">‖h‖ per layer, averaged across tokens. Raw activation norm through the network.</div>
                <div id="activation-magnitude-plot"></div>
            </section>

            <section>
                <h2>Layer × Token Heatmaps <span class="subsection-info-toggle" data-target="info-heatmaps">►</span></h2>
                <div class="subsection-info" id="info-heatmaps">y=layer, x=token, color=projection. How trait signal builds through layers for each token.</div>
                <div id="trait-heatmaps-container"></div>
            </section>

            <section>
                <h2>Activation Dynamics <span class="subsection-info-toggle" data-target="info-act-dynamics">►</span></h2>
                <div class="subsection-info" id="info-act-dynamics">Trait-independent activation analysis from per-token data.</div>

                <h3>Activation Velocity <span class="subsection-info-toggle" data-target="info-velocity">►</span></h3>
                <div class="subsection-info" id="info-velocity">‖h[L+1] − h[L]‖ per layer per token. How fast hidden state changes. Yellow = selected token.</div>
                <div id="velocity-heatmap-container"></div>

                <h3>Activation-Trait Coupling <span class="subsection-info-toggle" data-target="info-coupling">►</span></h3>
                <div class="subsection-info" id="info-coupling">Correlation: activation velocity vs trait projection change. High = trait tracks activation dynamics.</div>
                <div id="dynamics-correlation-container"></div>
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

    // Hover-to-highlight
    const plotDiv = document.getElementById('combined-activation-plot');
    plotDiv.on('plotly_hover', (d) =>
        Plotly.restyle(plotDiv, {'opacity': traces.map((_, i) => i === d.points[0].curveNumber ? 1.0 : 0.2)})
    );
    plotDiv.on('plotly_unhover', () => Plotly.restyle(plotDiv, {'opacity': 1.0}));

    // Render Token Velocity and Acceleration plots
    renderTokenDerivativePlots(traitActivations, loadedTraits, tickVals, tickText, nPromptTokens, startIdx);

    // Render Layer derivative plots (position, velocity, acceleration) - compact
    renderLayerDerivativePlots(traitData, loadedTraits);

    // Render Activation Magnitude plot
    renderActivationMagnitudePlot(traitData, loadedTraits);

    // Render Layer×Token Heatmaps (migrated from trait-trajectory)
    renderTraitHeatmaps(traitData, loadedTraits, allTokens, nPromptTokens);

    // Load and render per-token analysis data (migrated from analysis-gallery)
    loadAndRenderPerTokenAnalysis(promptSet, promptId);
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
        xaxis: { title: '', tickmode: 'array', tickvals: tickVals, ticktext: tickText, tickangle: -45, tickfont: { size: 8 } },
        yaxis: { title: 'Velocity', zeroline: true, zerolinewidth: 1, zerolinecolor: textSecondary },
        shapes: shapes,
        margin: { l: 50, r: 20, t: 10, b: 80 },
        height: 200,
        showlegend: false
    });

    const accelLayout = window.getPlotlyLayout({
        xaxis: { title: '', tickmode: 'array', tickvals: tickVals, ticktext: tickText, tickangle: -45, tickfont: { size: 8 } },
        yaxis: { title: 'Acceleration', zeroline: true, zerolinewidth: 1, zerolinecolor: textSecondary },
        shapes: shapes,
        margin: { l: 50, r: 20, t: 10, b: 80 },
        height: 200,
        showlegend: false
    });

    Plotly.newPlot('token-velocity-plot', velocityTraces, velocityLayout, { responsive: true, displayModeBar: false });
    Plotly.newPlot('token-acceleration-plot', accelTraces, accelLayout, { responsive: true, displayModeBar: false });
}


/**
 * Render three compact plots for position, velocity, and acceleration across layers.
 */
function renderLayerDerivativePlots(traitData, loadedTraits) {
    const firstData = traitData[loadedTraits[0]];
    const nLayers = firstData.projections.prompt[0]?.length || 26;
    const layerIndices = Array.from({length: nLayers}, (_, i) => i);
    const velocityIndices = Array.from({length: nLayers - 1}, (_, i) => i + 0.5);
    const accelIndices = Array.from({length: nLayers - 2}, (_, i) => i + 1);

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

    // Helper to create compact plot
    function renderCompactPlot(plotId, dataKey, xIndices) {
        const traces = [];
        loadedTraits.forEach((traitName, idx) => {
            const derivatives = allDerivatives[traitName];
            const traitColor = TRAIT_COLORS[idx % TRAIT_COLORS.length];
            traces.push({
                x: xIndices,
                y: derivatives[dataKey],
                type: 'scatter',
                mode: 'lines+markers',
                name: window.getDisplayName(traitName),
                line: { color: traitColor, width: 1.5 },
                marker: { size: 3 },
                hovertemplate: `Layer %{x:.1f}: %{y:.3f}<extra></extra>`
            });
        });

        const layout = window.getPlotlyLayout({
            xaxis: { title: '', tickmode: 'linear', tick0: 0, dtick: 5, tickfont: { size: 8 } },
            yaxis: { tickfont: { size: 8 }, zeroline: true },
            margin: { l: 40, r: 10, t: 5, b: 25 },
            height: 150,
            showlegend: false
        });

        Plotly.newPlot(plotId, traces, layout, { responsive: true, displayModeBar: false });
    }

    renderCompactPlot('layer-position-plot', 'position', layerIndices);
    renderCompactPlot('layer-velocity-plot', 'velocity', velocityIndices);
    renderCompactPlot('layer-acceleration-plot', 'acceleration', accelIndices);
}


/**
 * Render the Activation Magnitude plot showing ||h|| by layer.
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

    const traces = [
        { x: layerIndices, y: promptNorms, type: 'scatter', mode: 'lines+markers', name: 'Prompt',
          line: { color: '#4a9eff', width: 2 }, marker: { size: 4 } },
        { x: layerIndices, y: responseNorms, type: 'scatter', mode: 'lines+markers', name: 'Response',
          line: { color: '#ff6b6b', width: 2 }, marker: { size: 4 } },
        { x: layerIndices, y: combinedNorms, type: 'scatter', mode: 'lines+markers', name: 'Combined',
          line: { color: textSecondary, width: 2, dash: 'dash' }, marker: { size: 4 } }
    ];

    const layout = window.getPlotlyLayout({
        xaxis: { title: 'Layer', tickmode: 'linear', tick0: 0, dtick: 2 },
        yaxis: { title: '||h|| (L2 norm)' },
        margin: { l: 60, r: 20, t: 20, b: 50 },
        height: 250,
        legend: { orientation: 'h', y: 1.1, x: 0 },
        showlegend: true
    });

    Plotly.newPlot('activation-magnitude-plot', traces, layout, { responsive: true, displayModeBar: false });
}


/**
 * Render Layer×Token Heatmaps for each trait (migrated from trait-trajectory)
 */
function renderTraitHeatmaps(traitData, loadedTraits, allTokens, nPromptTokens) {
    const container = document.getElementById('trait-heatmaps-container');
    if (!container) return;

    // Create container for all heatmaps
    let html = '<div class="trait-heatmaps-grid">';
    loadedTraits.forEach((traitName) => {
        const traitId = traitName.replace(/\//g, '-');
        html += `
            <div class="trait-heatmap-item">
                <h4 title="${window.getDisplayName(traitName)}">${window.getDisplayName(traitName)}</h4>
                <div id="heatmap-${traitId}" style="width: 100%; height: 180px;"></div>
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


/**
 * Load and render per-token analysis data (migrated from analysis-gallery)
 */
async function loadAndRenderPerTokenAnalysis(promptSet, promptId) {
    const experiment = window.state.experimentData?.name;
    if (!experiment || !promptSet || !promptId) return;

    const url = window.paths.analysisPerToken(promptSet, promptId);

    try {
        const response = await fetch(url);
        if (!response.ok) {
            document.getElementById('velocity-heatmap-container').innerHTML =
                '<div class="info">Per-token analysis data not available.</div>';
            document.getElementById('dynamics-correlation-container').innerHTML = '';
            return;
        }

        const data = await response.json();
        renderVelocityHeatmap(data);
        renderDynamicsCorrelation(data);
    } catch (error) {
        document.getElementById('velocity-heatmap-container').innerHTML =
            '<div class="info">Failed to load per-token analysis data.</div>';
    }
}


/**
 * Render Activation Velocity heatmap (transposed: layers on y-axis)
 */
function renderVelocityHeatmap(data) {
    const container = document.getElementById('velocity-heatmap-container');
    if (!container) return;

    const currentTokenIdx = window.state?.currentTokenIndex || 0;

    // Build matrix [layers × tokens] (transposed from original)
    const nLayers = 25;  // layer transitions
    const nTokens = data.per_token.length;

    const zData = [];
    for (let l = 0; l < nLayers; l++) {
        zData[l] = data.per_token.map(t => (t.normalized_velocity_per_layer || [])[l] || 0);
    }

    const trace = {
        z: zData,
        x: Array.from({ length: nTokens }, (_, i) => i),
        y: Array.from({ length: nLayers }, (_, i) => `L${i}→${i+1}`),
        type: 'heatmap',
        colorscale: 'Viridis',
        hovertemplate: 'Token %{x}, %{y}<br>Velocity: %{z:.3f}<extra></extra>',
        showscale: true,
        colorbar: { thickness: 15, len: 0.8 }
    };

    // Highlight current token column
    const shapes = [{
        type: 'rect',
        x0: currentTokenIdx - 0.5,
        x1: currentTokenIdx + 0.5,
        y0: -0.5,
        y1: nLayers - 0.5,
        line: { color: '#ffff00', width: 2 },
        fillcolor: 'rgba(0,0,0,0)'
    }];

    const layout = window.getPlotlyLayout({
        margin: { l: 60, r: 50, t: 10, b: 40 },
        height: 300,
        xaxis: { title: 'Token', dtick: 10 },
        yaxis: { title: 'Layer Transition', tickfont: { size: 8 } },
        shapes
    });

    Plotly.newPlot(container, [trace], layout, { responsive: true, displayModeBar: false });
}


/**
 * Render Activation-Trait Coupling correlation chart
 */
function renderDynamicsCorrelation(data) {
    const container = document.getElementById('dynamics-correlation-container');
    if (!container) return;

    const firstToken = data.per_token.find(t => t.trait_scores_per_layer);
    if (!firstToken) {
        container.innerHTML = '<div class="info">No trait data available for correlation analysis.</div>';
        return;
    }

    const traits = Object.keys(firstToken.trait_scores_per_layer);

    // Compute correlation for each trait
    const correlations = traits.map(trait => {
        const velocities = [];
        const traitVelocities = [];

        data.per_token.forEach(t => {
            if (!t.normalized_velocity_per_layer || !t.trait_scores_per_layer?.[trait]) return;

            const traitScores = t.trait_scores_per_layer[trait];
            for (let i = 0; i < traitScores.length - 1; i++) {
                velocities.push(t.normalized_velocity_per_layer[i] || 0);
                traitVelocities.push(Math.abs(traitScores[i + 1] - traitScores[i]));
            }
        });

        if (velocities.length < 2) return { trait, corr: 0 };

        // Pearson correlation
        const n = velocities.length;
        const sumX = velocities.reduce((a, b) => a + b, 0);
        const sumY = traitVelocities.reduce((a, b) => a + b, 0);
        const sumXY = velocities.reduce((sum, x, i) => sum + x * traitVelocities[i], 0);
        const sumX2 = velocities.reduce((sum, x) => sum + x * x, 0);
        const sumY2 = traitVelocities.reduce((sum, y) => sum + y * y, 0);

        const num = n * sumXY - sumX * sumY;
        const den = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
        const corr = den === 0 ? 0 : num / den;

        return { trait, corr };
    });

    correlations.sort((a, b) => b.corr - a.corr);

    const trace = {
        x: correlations.map(c => c.corr),
        y: correlations.map(c => window.getDisplayName ? window.getDisplayName(c.trait) : c.trait),
        type: 'bar',
        orientation: 'h',
        marker: {
            color: correlations.map(c => c.corr > 0.3 ? '#27ae60' : c.corr > 0.1 ? '#f39c12' : '#95a5a6')
        },
        hovertemplate: '%{y}: r = %{x:.3f}<extra></extra>'
    };

    const layout = window.getPlotlyLayout({
        margin: { l: 120, r: 20, t: 10, b: 40 },
        height: Math.max(200, correlations.length * 25),
        xaxis: { title: 'Correlation (r)', range: [-0.2, 1] },
        yaxis: { tickfont: { size: 10 } }
    });

    Plotly.newPlot(container, [trace], layout, { responsive: true, displayModeBar: false });
}


// Export to global scope
window.renderTraitDynamics = renderTraitDynamics;
