// Trait Dynamics View - Watch the model's internal state evolve token-by-token
// Core insight: "See how the model is thinking" by projecting onto trait vectors
//
// Two complementary views:
// 1. Token Trajectory: X=tokens, Y=activation (layer-averaged) - how traits evolve during generation
// 2. Layer Evolution: X=layers, Y=projection (token-averaged) - how traits emerge through the network

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

// Derivative overlay colors (position, velocity, acceleration)
const DERIVATIVE_COLORS = {
    position: '#2E86AB',     // blue
    velocity: '#A23B72',     // magenta
    acceleration: '#F18F01'  // orange
};

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
 * @returns {Object} { position: [26], velocity: [25], acceleration: [24] }
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
        // Show education sections even without data
        contentArea.innerHTML = `
            <div class="tool-view">
                <div class="page-intro">
                    <div class="page-intro-text">Watch traits evolve token-by-token during generation.</div>
                </div>
                <div class="info">Select at least one trait from the sidebar to view activation trajectories.</div>
            </div>
        `;
        if (window.MathJax) MathJax.typesetPromise();
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
    if (window.MathJax) MathJax.typesetPromise();

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
            console.log(`[${trait.name}] Fetching prompt activation data for ${promptSet}/${promptId}`);
            const response = await fetch(fetchPath);

            if (!response.ok) {
                console.log(`[${trait.name}] No data available for ${promptSet}/${promptId} (${response.status})`);
                failedTraits.push(trait.name);
                continue;
            }

            const data = await response.json();
            console.log(`[${trait.name}] Data loaded successfully for ${promptSet}/${promptId}`);
            traitData[trait.name] = data;
        } catch (error) {
            console.log(`[${trait.name}] Load failed for ${promptSet}/${promptId}:`, error.message);
            failedTraits.push(trait.name);
        }
    }

    // Check if we have any data
    const loadedTraits = Object.keys(traitData);
    if (loadedTraits.length === 0) {
        renderNoDataMessage(contentArea, filteredTraits, promptSet, promptId);
        return;
    }

    // Render education + combined graph
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
    const nPromptTokens = promptTokens.length;  // Use actual array length
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
                <h2>Token Trajectory</h2>
                <div id="combined-activation-plot"></div>
            </section>
            <section>
                <h2>Normalized Position</h2>
                <div id="layer-position-plot"></div>
            </section>
            <section>
                <h2>Normalized Velocity</h2>
                <div id="layer-velocity-plot"></div>
            </section>
            <section>
                <h2>Normalized Acceleration</h2>
                <div id="layer-acceleration-plot"></div>
            </section>
            <section>
                <h2>Activation Magnitude</h2>
                <div id="activation-magnitude-plot"></div>
            </section>
        </div>
    `;

    // Prepare traces for each trait
    const traces = [];
    const startIdx = 1;  // Skip BOS token

    loadedTraits.forEach((traitName, idx) => {
        const data = traitData[traitName];
        const promptProj = data.projections.prompt;
        const responseProj = data.projections.response;
        const allProj = [...promptProj, ...responseProj];
        const nLayers = promptProj[0].length;

        // Calculate activation strength for each token (average across all layers and sublayers)
        const rawActivations = [];
        const displayTokens = [];

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
            displayTokens.push(allTokens[t]);
        }

        // Apply 3-token moving average to reduce noise
        const activations = smoothData(rawActivations, 3);

        const color = TRAIT_COLORS[idx % TRAIT_COLORS.length];

        traces.push({
            x: Array.from({length: activations.length}, (_, i) => i),
            y: activations,
            type: 'scatter',
            mode: 'lines+markers',
            name: window.getDisplayName(traitName),
            line: {
                color: color,
                width: 1
            },
            marker: {
                size: 1,
                color: color
            },
            text: displayTokens,
            hovertemplate: `<b>${window.getDisplayName(traitName)}</b><br>Token %{x}: %{text}<br>Activation: %{y:.3f}<extra></extra>`
        });
    });

    // Get display tokens from first trait for x-axis labels
    const displayTokens = [];
    for (let t = startIdx; t < allTokens.length; t++) {
        displayTokens.push(allTokens[t]);
    }

    // Get colors from CSS variables
    const textSecondary = window.getCssVar('--text-secondary', '#a4a4a4');
    const bgTertiary = window.getCssVar('--bg-tertiary', '#3a3a3a');
    const primaryColor = window.getCssVar('--primary-color', '#a09f6c');

    // Get current token index from global state (absolute index across prompt+response)
    // The graph skips BOS (startIdx=1), so token at absolute index N = x position (N - startIdx)
    const currentTokenIdx = window.state.currentTokenIndex || 0;
    const highlightX = currentTokenIdx - startIdx;

    // Add subtle vertical line separator between prompt and response
    const shapes = [
        {
            type: 'line',
            x0: (nPromptTokens - startIdx) - 0.5,
            x1: (nPromptTokens - startIdx) - 0.5,
            y0: 0,
            y1: 1,
            yref: 'paper',
            line: {
                color: textSecondary,
                width: 2,
                dash: 'dash'
            }
        },
        // Current token highlight line from global slider
        {
            type: 'line',
            x0: highlightX,
            x1: highlightX,
            y0: 0,
            y1: 1,
            yref: 'paper',
            line: {
                color: primaryColor,
                width: 2
            }
        }
    ];

    // Add annotations for prompt/response regions
    const annotations = [
        {
            x: (nPromptTokens - startIdx) / 2 - 0.5,
            y: 1.08,
            yref: 'paper',
            text: 'PROMPT',
            showarrow: false,
            font: {
                size: 11,
                color: textSecondary
            }
        },
        {
            x: (nPromptTokens - startIdx) + (displayTokens.length - (nPromptTokens - startIdx)) / 2 - 0.5,
            y: 1.08,
            yref: 'paper',
            text: 'RESPONSE',
            showarrow: false,
            font: {
                size: 11,
                color: textSecondary
            }
        }
    ];

    const layout = window.getPlotlyLayout({
        xaxis: {
            title: 'Token Position',
            tickmode: 'array',
            tickvals: Array.from({length: displayTokens.length}, (_, i) => i),
            ticktext: displayTokens,
            tickangle: -45,
            showgrid: false,
            tickfont: { size: 9 }
        },
        yaxis: {
            title: 'Activation (3-token avg)',
            zeroline: true,
            zerolinewidth: 1,
            showgrid: true
        },
        shapes: shapes,
        annotations: annotations,
        margin: { l: 60, r: 20, t: 40, b: 140 },
        height: 500,
        font: { size: 11 },
        hovermode: 'closest',
        legend: {
            orientation: 'h',
            yanchor: 'top',
            y: -0.25,
            xanchor: 'center',
            x: 0.5,
            font: { size: 10 },
            bgcolor: 'transparent'
        },
        showlegend: true
    });

    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['lasso2d', 'select2d']
    };

    Plotly.newPlot('combined-activation-plot', traces, layout, config);

    // Hover-to-highlight: dim other traces when hovering
    const plotDiv = document.getElementById('combined-activation-plot');
    plotDiv.on('plotly_hover', (d) =>
        Plotly.restyle(plotDiv, {'opacity': traces.map((_, i) => i === d.points[0].curveNumber ? 1.0 : 0.2)})
    );
    plotDiv.on('plotly_unhover', () => Plotly.restyle(plotDiv, {'opacity': 1.0}));

    // Render Layer derivative plots (position, velocity, acceleration)
    renderLayerDerivativePlots(traitData, loadedTraits);

    // Render Activation Magnitude plot (trait-independent)
    renderActivationMagnitudePlot(traitData, loadedTraits);
}


/**
 * Render three separate plots for position, velocity, and acceleration across layers.
 * Normalizes by activation magnitude when available.
 */
function renderLayerDerivativePlots(traitData, loadedTraits) {
    const nLayers = 26;
    const layerIndices = Array.from({length: nLayers}, (_, i) => i);
    const velocityIndices = Array.from({length: nLayers - 1}, (_, i) => i + 0.5);
    const accelIndices = Array.from({length: nLayers - 2}, (_, i) => i + 1);

    // Get activation norms for normalization (same for all traits, use first)
    const firstData = traitData[loadedTraits[0]];
    let combinedNorms = null;
    if (firstData.activation_norms) {
        const promptNorms = firstData.activation_norms.prompt;
        const responseNorms = firstData.activation_norms.response;
        combinedNorms = promptNorms.map((p, i) => (p + responseNorms[i]) / 2);
    }

    // Compute derivatives for all traits (normalized if norms available)
    const allDerivatives = {};
    loadedTraits.forEach((traitName) => {
        allDerivatives[traitName] = computeLayerDerivatives(traitData[traitName], combinedNorms);
    });

    // Helper to create a single plot
    function renderSinglePlot(plotId, yLabel, dataKey, xIndices) {
        const traces = [];

        loadedTraits.forEach((traitName, idx) => {
            const derivatives = allDerivatives[traitName];
            const traitColor = TRAIT_COLORS[idx % TRAIT_COLORS.length];
            const displayName = window.getDisplayName(traitName);

            traces.push({
                x: xIndices,
                y: derivatives[dataKey],
                type: 'scatter',
                mode: 'lines+markers',
                name: displayName,
                line: { color: traitColor, width: 2 },
                marker: { size: 4 },
                hovertemplate: `<b>${displayName}</b><br>Layer %{x:.1f}<br>${yLabel}: %{y:.2f}<extra></extra>`
            });
        });

        const textSecondary = window.getCssVar('--text-secondary', '#a4a4a4');

        const layout = window.getPlotlyLayout({
            xaxis: {
                title: 'Layer',
                tickmode: 'linear',
                tick0: 0,
                dtick: 2,
                showgrid: true,
                gridcolor: 'rgba(128,128,128,0.2)'
            },
            yaxis: {
                title: yLabel,
                zeroline: true,
                zerolinewidth: 1,
                zerolinecolor: textSecondary,
                showgrid: true
            },
            margin: { l: 60, r: 20, t: 20, b: 50 },
            height: 300,
            font: { size: 11 },
            hovermode: 'closest',
            legend: {
                orientation: 'h',
                yanchor: 'bottom',
                y: 1.02,
                xanchor: 'left',
                x: 0,
                font: { size: 10 },
                bgcolor: 'transparent'
            },
            showlegend: true
        });

        const config = {
            responsive: true,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['lasso2d', 'select2d']
        };

        Plotly.newPlot(plotId, traces, layout, config);

        // Hover-to-highlight
        const plotDiv = document.getElementById(plotId);
        plotDiv.on('plotly_hover', (d) => {
            Plotly.restyle(plotDiv, {'opacity': traces.map((_, i) => i === d.points[0].curveNumber ? 1.0 : 0.2)});
        });
        plotDiv.on('plotly_unhover', () => Plotly.restyle(plotDiv, {'opacity': 1.0}));
    }

    // Render all three plots
    const normLabel = combinedNorms ? ' (proj/||h||)' : '';
    renderSinglePlot('layer-position-plot', `Position${normLabel}`, 'position', layerIndices);
    renderSinglePlot('layer-velocity-plot', `Velocity${normLabel}`, 'velocity', velocityIndices);
    renderSinglePlot('layer-acceleration-plot', `Acceleration${normLabel}`, 'acceleration', accelIndices);
}


/**
 * Render the Activation Magnitude plot showing ||h|| by layer.
 * This is trait-independent - shows raw activation norm at each layer.
 */
function renderActivationMagnitudePlot(traitData, loadedTraits) {
    // Get activation norms from first trait's data (same for all traits)
    const firstTraitData = traitData[loadedTraits[0]];

    // Check if activation_norms exists (requires re-running projection script)
    if (!firstTraitData.activation_norms) {
        const plotDiv = document.getElementById('activation-magnitude-plot');
        plotDiv.innerHTML = `
            <div class="info">
                Activation norms not available. Re-run projection script to generate:
                <pre>python inference/project_raw_activations_onto_traits.py --experiment {exp} --prompt-set {set}</pre>
            </div>
        `;
        return;
    }

    const promptNorms = firstTraitData.activation_norms.prompt;
    const responseNorms = firstTraitData.activation_norms.response;
    const nLayers = promptNorms.length;
    const layerIndices = Array.from({length: nLayers}, (_, i) => i);

    // Compute combined average (prompt + response)
    const combinedNorms = promptNorms.map((p, i) => (p + responseNorms[i]) / 2);

    const textSecondary = window.getCssVar('--text-secondary', '#a4a4a4');

    const traces = [
        {
            x: layerIndices,
            y: promptNorms,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Prompt',
            line: { color: '#4a9eff', width: 2 },
            marker: { size: 4 },
            hovertemplate: '<b>Prompt</b><br>Layer %{x}<br>||h|| = %{y:.1f}<extra></extra>'
        },
        {
            x: layerIndices,
            y: responseNorms,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Response',
            line: { color: '#ff6b6b', width: 2 },
            marker: { size: 4 },
            hovertemplate: '<b>Response</b><br>Layer %{x}<br>||h|| = %{y:.1f}<extra></extra>'
        },
        {
            x: layerIndices,
            y: combinedNorms,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Combined',
            line: { color: textSecondary, width: 2, dash: 'dash' },
            marker: { size: 4 },
            hovertemplate: '<b>Combined</b><br>Layer %{x}<br>||h|| = %{y:.1f}<extra></extra>'
        }
    ];

    const layout = window.getPlotlyLayout({
        xaxis: {
            title: 'Layer',
            tickmode: 'linear',
            tick0: 0,
            dtick: 2,
            showgrid: true,
            gridcolor: 'rgba(128,128,128,0.2)'
        },
        yaxis: {
            title: '||h|| (L2 norm)',
            showgrid: true
        },
        margin: { l: 60, r: 20, t: 20, b: 50 },
        height: 300,
        font: { size: 11 },
        hovermode: 'closest',
        legend: {
            orientation: 'h',
            yanchor: 'bottom',
            y: 1.02,
            xanchor: 'left',
            x: 0,
            font: { size: 10 },
            bgcolor: 'transparent'
        },
        showlegend: true
    });

    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['lasso2d', 'select2d']
    };

    Plotly.newPlot('activation-magnitude-plot', traces, layout, config);
}


// Export to global scope
window.renderTraitDynamics = renderTraitDynamics;
