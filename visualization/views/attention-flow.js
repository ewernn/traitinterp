// Attention Flow Field Visualization
// Shows how attention patterns evolve as "flow" through the transformer

async function renderAttentionFlow() {
    const contentArea = document.getElementById('content-area');
    const state = window.getState();
    const filteredTraits = window.getFilteredTraits();

    contentArea.innerHTML = `
        <div class="card">
            <div class="card-title">Attention Flow Field</div>

            <div style="background: var(--bg-tertiary); padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                <p style="color: var(--text-secondary); margin: 0;">
                    Visualizing attention as <strong>information flow</strong> through the transformer.
                    Shows where information "flows from" at each layer and how flow patterns correlate with trait activation.
                </p>
            </div>

            <!-- Controls -->
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin-bottom: 20px;">
                <div>
                    <label style="color: var(--text-secondary); font-size: 12px;">Prompt Set</label>
                    <select id="flow-prompt-set" class="select-input" onchange="window.updateFlowVisualization()">
                        <option value="">Select prompt set...</option>
                        <option value="dynamic">Dynamic (trait transitions)</option>
                        <option value="single_trait">Single Trait</option>
                        <option value="multi_trait">Multi Trait</option>
                        <option value="adversarial">Adversarial</option>
                        <option value="baseline">Baseline</option>
                        <option value="real_world">Real World</option>
                    </select>
                </div>

                <div>
                    <label style="color: var(--text-secondary); font-size: 12px;">Prompt ID</label>
                    <input type="number" id="flow-prompt-id" class="input-field"
                           min="1" max="10" value="1" onchange="window.updateFlowVisualization()">
                </div>

                <div>
                    <label style="color: var(--text-secondary); font-size: 12px;">Visualization Mode</label>
                    <select id="flow-mode" class="select-input" onchange="window.updateFlowVisualization()">
                        <option value="velocity">Velocity Field</option>
                        <option value="acceleration">Acceleration Map</option>
                        <option value="flow-lines">Flow Lines</option>
                        <option value="attractor">Attractor Landscape</option>
                    </select>
                </div>
            </div>

            <!-- Main Visualization Area -->
            <div id="flow-viz-container" style="min-height: 400px;">
                <div style="text-align: center; color: var(--text-tertiary); padding: 50px;">
                    Select a prompt set and ID to visualize attention flow
                </div>
            </div>

            <!-- Dynamics Metrics -->
            <div id="dynamics-metrics" style="display: none; margin-top: 20px;">
                <h3 style="color: var(--text-primary); font-size: 14px; margin-bottom: 10px;">Flow Dynamics</h3>
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
                    <div class="metric-card">
                        <div class="metric-label">Critical Points</div>
                        <div class="metric-value" id="critical-count">-</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Max Velocity</div>
                        <div class="metric-value" id="max-velocity">-</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Bifurcations</div>
                        <div class="metric-value" id="bifurcation-count">-</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Flow Coherence</div>
                        <div class="metric-value" id="flow-coherence">-</div>
                    </div>
                </div>
            </div>
        </div>

        <style>
            .metric-card {
                background: var(--bg-secondary);
                padding: 12px;
                border-radius: 8px;
                text-align: center;
            }
            .metric-label {
                color: var(--text-secondary);
                font-size: 11px;
                margin-bottom: 4px;
            }
            .metric-value {
                color: var(--primary-color);
                font-size: 18px;
                font-weight: 600;
            }
        </style>
    `;
}

window.updateFlowVisualization = async function() {
    const promptSet = document.getElementById('flow-prompt-set').value;
    const promptId = document.getElementById('flow-prompt-id').value;
    const mode = document.getElementById('flow-mode').value;

    if (!promptSet || !promptId) return;

    const container = document.getElementById('flow-viz-container');
    const metricsDiv = document.getElementById('dynamics-metrics');

    // Show loading state
    container.innerHTML = '<div style="text-align: center; padding: 50px;">Loading dynamics data...</div>';

    try {
        // Load residual stream data
        const dataPath = `/api/experiments/${window.getState().experiment}/inference/raw/residual/${promptSet}/${promptId}.pt`;

        // For now, simulate with computed dynamics
        const dynamics = await computeDynamics(promptSet, promptId);

        // Render based on mode
        switch(mode) {
            case 'velocity':
                renderVelocityField(container, dynamics);
                break;
            case 'acceleration':
                renderAccelerationMap(container, dynamics);
                break;
            case 'flow-lines':
                renderFlowLines(container, dynamics);
                break;
            case 'attractor':
                renderAttractorLandscape(container, dynamics);
                break;
        }

        // Update metrics
        updateDynamicsMetrics(dynamics, metricsDiv);
        metricsDiv.style.display = 'block';

    } catch (error) {
        container.innerHTML = `<div style="color: var(--danger); text-align: center; padding: 50px;">
            Error loading dynamics: ${error.message}
        </div>`;
    }
}

async function computeDynamics(promptSet, promptId) {
    // This would normally load and process the .pt file
    // For now, return mock dynamics data

    const nLayers = 26;
    const nTokens = 15;

    // Create mock velocity field
    const velocityField = [];
    for (let l = 0; l < nLayers - 1; l++) {
        const row = [];
        for (let t = 0; t < nTokens; t++) {
            // Simulate higher velocity in middle layers and at trait transition points
            const layerFactor = Math.exp(-Math.pow(l - 13, 2) / 50);
            const tokenFactor = (promptSet === 'dynamic' && t === 7) ? 2.0 : 1.0;
            row.push(Math.random() * layerFactor * tokenFactor);
        }
        velocityField.push(row);
    }

    return {
        velocityField,
        criticalPoints: [
            {layer: 14, token: 7, type: 'bifurcation'},
            {layer: 16, token: 9, type: 'stationary'},
            {layer: 18, token: 5, type: 'inflection'}
        ],
        maxVelocity: 2.4,
        coherence: 0.72
    };
}

function renderVelocityField(container, dynamics) {
    const trace = {
        z: dynamics.velocityField,
        type: 'heatmap',
        colorscale: 'Viridis',
        hovertemplate: 'Layer: %{y}<br>Token: %{x}<br>Velocity: %{z:.3f}<extra></extra>',
        colorbar: {
            title: 'Velocity',
            titleside: 'right'
        }
    };

    // Overlay critical points
    const annotations = dynamics.criticalPoints.map(point => ({
        x: point.token,
        y: point.layer,
        text: point.type === 'bifurcation' ? '⦻' : (point.type === 'stationary' ? '○' : '△'),
        showarrow: false,
        font: {
            size: 16,
            color: point.type === 'bifurcation' ? '#ff6b6b' : '#4dabf7'
        }
    }));

    const layout = window.getPlotlyLayout({
        title: 'Representation Velocity Field',
        xaxis: { title: 'Token Position' },
        yaxis: { title: 'Layer' },
        annotations: annotations,
        height: 400
    });

    container.innerHTML = '<div id="velocity-plot"></div>';
    Plotly.newPlot('velocity-plot', [trace], layout, {displayModeBar: false});
}

function renderAccelerationMap(container, dynamics) {
    // Compute acceleration from velocity
    const acceleration = [];
    for (let l = 0; l < dynamics.velocityField.length - 1; l++) {
        const row = [];
        for (let t = 0; t < dynamics.velocityField[0].length; t++) {
            const accel = dynamics.velocityField[l+1][t] - dynamics.velocityField[l][t];
            row.push(accel);
        }
        acceleration.push(row);
    }

    const trace = {
        z: acceleration,
        type: 'heatmap',
        colorscale: 'RdBu',
        zmid: 0,
        hovertemplate: 'Layer: %{y}<br>Token: %{x}<br>Acceleration: %{z:.3f}<extra></extra>',
        colorbar: {
            title: 'Acceleration',
            titleside: 'right'
        }
    };

    const layout = window.getPlotlyLayout({
        title: 'Attention Acceleration (2nd Derivative)',
        xaxis: { title: 'Token Position' },
        yaxis: { title: 'Layer' },
        height: 400
    });

    container.innerHTML = '<div id="accel-plot"></div>';
    Plotly.newPlot('accel-plot', [trace], layout, {displayModeBar: false});
}

function renderFlowLines(container, dynamics) {
    // Create flow lines showing information paths
    container.innerHTML = `
        <div id="flow-lines-plot"></div>
        <div style="margin-top: 15px; padding: 15px; background: var(--bg-secondary); border-radius: 8px;">
            <h4 style="color: var(--text-primary); margin: 0 0 10px 0; font-size: 13px;">Flow Line Interpretation</h4>
            <p style="color: var(--text-secondary); font-size: 12px; margin: 5px 0;">
                <span style="color: var(--primary-color);">→</span> Information flow direction<br>
                <span style="color: var(--danger);">●</span> Bifurcation point (trait decision)<br>
                <span style="color: var(--info-text);">○</span> Stationary point (stable representation)<br>
                Line thickness indicates flow strength
            </p>
        </div>
    `;

    // Create streamline plot
    const traces = [];

    // Add flow lines for each token
    for (let t = 0; t < dynamics.velocityField[0].length; t++) {
        const x = [];
        const y = [];
        const text = [];

        for (let l = 0; l < dynamics.velocityField.length; l++) {
            x.push(t + (Math.random() - 0.5) * 0.2);  // Add small jitter
            y.push(l);
            text.push(`Velocity: ${dynamics.velocityField[l][t].toFixed(3)}`);
        }

        traces.push({
            x: x,
            y: y,
            mode: 'lines',
            line: {
                color: dynamics.velocityField.map(row => row[t]),
                colorscale: 'Viridis',
                width: 2
            },
            text: text,
            hoverinfo: 'text',
            showlegend: false
        });
    }

    const layout = window.getPlotlyLayout({
        title: 'Information Flow Lines',
        xaxis: { title: 'Token Position', range: [-0.5, dynamics.velocityField[0].length - 0.5] },
        yaxis: { title: 'Layer' },
        height: 400
    });

    Plotly.newPlot('flow-lines-plot', traces, layout, {displayModeBar: false});
}

function renderAttractorLandscape(container, dynamics) {
    // Create a 3D surface representing the attractor landscape
    container.innerHTML = `
        <div id="attractor-plot"></div>
        <div style="margin-top: 15px; padding: 15px; background: var(--bg-secondary); border-radius: 8px;">
            <h4 style="color: var(--text-primary); margin: 0 0 10px 0; font-size: 13px;">Attractor Landscape</h4>
            <p style="color: var(--text-secondary); font-size: 12px; margin: 0;">
                Visualizing the "energy landscape" of representation space.<br>
                <strong>Valleys</strong> = stable attractors (traits)<br>
                <strong>Ridges</strong> = decision boundaries between traits<br>
                <strong>Depth</strong> = strength of attractor pull
            </p>
        </div>
    `;

    // Create a 3D surface
    const surface = {
        z: dynamics.velocityField,
        type: 'surface',
        colorscale: 'Viridis',
        contours: {
            z: {
                show: true,
                usecolormap: true,
                project: {z: true}
            }
        }
    };

    const layout = {
        ...window.getPlotlyLayout({
            title: 'Attractor Landscape',
            height: 500
        }),
        scene: {
            xaxis: {title: 'Token Position'},
            yaxis: {title: 'Layer'},
            zaxis: {title: 'Potential Energy'},
            camera: {
                eye: {x: 1.5, y: 1.5, z: 1.5}
            }
        }
    };

    Plotly.newPlot('attractor-plot', [surface], layout, {displayModeBar: false});
}

function updateDynamicsMetrics(dynamics, metricsDiv) {
    document.getElementById('critical-count').textContent = dynamics.criticalPoints.length;
    document.getElementById('max-velocity').textContent = dynamics.maxVelocity.toFixed(2);
    document.getElementById('bifurcation-count').textContent =
        dynamics.criticalPoints.filter(p => p.type === 'bifurcation').length;
    document.getElementById('flow-coherence').textContent = `${(dynamics.coherence * 100).toFixed(0)}%`;
}

// Export to global scope
window.renderAttentionFlow = renderAttentionFlow;