/**
 * Multi-Layer Steering Heatmap
 *
 * Visualizes center × width grid showing where traits "live" in the model.
 * - Y-axis: Center layer
 * - X-axis: Width (number of layers steered)
 * - Color: Delta (trait improvement over baseline)
 */

async function renderMultiLayerHeatmap() {
    const contentArea = document.getElementById('content-area');
    contentArea.innerHTML = '<div class="loading">Loading multi-layer heatmap data...</div>';

    const experiment = window.state.experimentData?.name;
    if (!experiment) {
        contentArea.innerHTML = '<div class="error">No experiment selected</div>';
        return;
    }

    // For now, hardcode the trait - TODO: discover traits with heatmap data
    const trait = 'epistemic/optimism';
    const heatmapPath = `/experiments/${experiment}/steering/${trait}/center_width_heatmap.json`;
    const resultsPath = `/experiments/${experiment}/steering/${trait}/results.json`;

    let heatmapData, resultsData;
    try {
        const heatmapResp = await fetch(heatmapPath);
        if (!heatmapResp.ok) {
            throw new Error('Heatmap data not found');
        }
        heatmapData = await heatmapResp.json();

        const resultsResp = await fetch(resultsPath);
        resultsData = resultsResp.ok ? await resultsResp.json() : null;
    } catch (error) {
        contentArea.innerHTML = `
            <div class="tool-view">
                <div class="no-data">
                    <p>No multi-layer heatmap data found</p>
                    <small>Generate with the center×width sweep analysis script first.</small>
                </div>
            </div>
        `;
        return;
    }

    // Build the view
    contentArea.innerHTML = `
        <div class="tool-view">
            <div class="page-intro">
                <div class="page-intro-text">Multi-layer steering analysis: center × width grid showing optimal layer combinations.</div>
                <div class="intro-example">
                    <div><span class="example-label">Center:</span> Middle layer of the steering range</div>
                    <div><span class="example-label">Width:</span> Number of consecutive layers steered</div>
                </div>
            </div>

            <div class="sweep-controls">
                <div class="control-group">
                    <label>Color by:</label>
                    <select id="heatmap-metric">
                        <option value="delta" selected>Delta (trait improvement)</option>
                        <option value="coherence">Coherence</option>
                        <option value="combined">Combined (delta × coh/100)</option>
                    </select>
                </div>
            </div>

            <div id="heatmap-container" style="width: 100%; height: 500px;"></div>

            <div class="summary-grid" style="margin-top: 20px;">
                <div class="summary-card">
                    <div class="card-title">Top Configurations</div>
                    <div class="card-content">
                        <table class="data-table" id="top-configs-table">
                            <thead>
                                <tr>
                                    <th>#</th>
                                    <th>Center</th>
                                    <th>Width</th>
                                    <th>Layers</th>
                                    <th>Delta</th>
                                    <th>Coh</th>
                                </tr>
                            </thead>
                            <tbody></tbody>
                        </table>
                    </div>
                </div>
                <div class="summary-card">
                    <div class="card-title">vs Single-Layer Best</div>
                    <div class="card-content" id="comparison-content"></div>
                </div>
            </div>
        </div>
    `;

    // Store data for re-rendering
    window._multiLayerHeatmapData = { heatmapData, resultsData };

    // Initial render
    renderMultiLayerHeatmapPlot('delta');
    renderTopConfigs(heatmapData);
    renderComparison(heatmapData, resultsData);

    // Event listener for metric change
    document.getElementById('heatmap-metric').addEventListener('change', (e) => {
        renderMultiLayerHeatmapPlot(e.target.value);
    });
}

function renderMultiLayerHeatmapPlot(metric) {
    const { heatmapData } = window._multiLayerHeatmapData;
    const { centers, widths, delta_grid, coherence_grid } = heatmapData;

    // Choose data based on metric
    let z, zLabel, zMin, zMax;
    if (metric === 'delta') {
        z = delta_grid;
        zLabel = 'Delta';
        zMin = 0;
        zMax = 35;
    } else if (metric === 'coherence') {
        z = coherence_grid;
        zLabel = 'Coherence';
        zMin = 50;
        zMax = 90;
    } else {
        // Combined: delta × (coherence/100)
        z = delta_grid.map((row, i) =>
            row.map((d, j) => {
                const c = coherence_grid[i][j];
                return (d !== null && c !== null) ? d * (c / 100) : null;
            })
        );
        zLabel = 'Combined';
        zMin = 0;
        zMax = 25;
    }

    // Create hover text
    const hovertext = centers.map((center, i) =>
        widths.map((width, j) => {
            const delta = delta_grid[i][j];
            const coherence = coherence_grid[i][j];
            if (delta === null) return '';

            const half = Math.floor(width / 2);
            const layerRange = `L${center - half}-L${center + half}`;
            return `Center: L${center}<br>Width: ${width}<br>Layers: ${layerRange}<br>Delta: ${delta?.toFixed(1) || '--'}<br>Coherence: ${coherence?.toFixed(1) || '--'}`;
        })
    );

    const colorscale = [
        [0, '#ef4444'],
        [0.25, '#fef3c7'],
        [0.45, '#fefce8'],
        [0.55, '#d9f99d'],
        [1, '#22c55e']
    ];

    const trace = {
        type: 'heatmap',
        x: widths.map(w => `W=${w}`),
        y: centers.map(c => `L${c}`),
        z: z,
        hovertext: hovertext,
        hoverinfo: 'text',
        colorscale: colorscale,
        colorbar: {
            title: zLabel,
            titleside: 'right'
        },
        zmin: zMin,
        zmax: zMax
    };

    const layout = {
        title: '',
        xaxis: {
            title: 'Width (# layers steered)',
            side: 'bottom'
        },
        yaxis: {
            title: 'Center Layer',
            autorange: 'reversed'
        },
        margin: { t: 20, b: 60, l: 60, r: 100 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#e0e0e0' }
    };

    Plotly.newPlot('heatmap-container', [trace], layout, { responsive: true });
}

function renderTopConfigs(heatmapData) {
    const { centers, widths, delta_grid, coherence_grid } = heatmapData;

    // Collect all valid configs
    const configs = [];
    centers.forEach((center, ci) => {
        widths.forEach((width, wi) => {
            const delta = delta_grid[ci][wi];
            const coherence = coherence_grid[ci][wi];
            if (delta !== null) {
                const half = Math.floor(width / 2);
                configs.push({
                    center,
                    width,
                    layers: `L${center - half}-${center + half}`,
                    delta,
                    coherence
                });
            }
        });
    });

    // Sort by delta
    configs.sort((a, b) => b.delta - a.delta);

    // Render table
    const tbody = document.querySelector('#top-configs-table tbody');
    tbody.innerHTML = configs.slice(0, 8).map((c, i) => `
        <tr>
            <td>${i + 1}</td>
            <td>L${c.center}</td>
            <td>${c.width}</td>
            <td>${c.layers}</td>
            <td style="color: ${c.delta > 25 ? '#22c55e' : '#e0e0e0'}">${c.delta.toFixed(1)}</td>
            <td style="color: ${c.coherence > 80 ? '#22c55e' : c.coherence < 70 ? '#ef4444' : '#e0e0e0'}">${c.coherence.toFixed(1)}</td>
        </tr>
    `).join('');
}

function renderComparison(heatmapData, resultsData) {
    const content = document.getElementById('comparison-content');

    if (!resultsData) {
        content.innerHTML = '<p style="color: #888;">No results.json found</p>';
        return;
    }

    const baseline = resultsData.baseline?.trait_mean || 0;
    const { centers, widths, delta_grid, coherence_grid } = heatmapData;

    // Find best single-layer
    let bestSingle = null;
    for (const run of resultsData.runs || []) {
        if (run.config.layers.length === 1) {
            const trait = run.result?.trait_mean;
            const coherence = run.result?.coherence_mean;
            if (trait && (!bestSingle || trait > bestSingle.trait)) {
                bestSingle = {
                    layer: run.config.layers[0],
                    coef: run.config.coefficients[0],
                    trait,
                    coherence,
                    delta: trait - baseline
                };
            }
        }
    }

    // Find best multi-layer
    let bestMulti = null;
    centers.forEach((center, ci) => {
        widths.forEach((width, wi) => {
            if (width === 1) return;
            const delta = delta_grid[ci][wi];
            const coherence = coherence_grid[ci][wi];
            if (delta !== null && (!bestMulti || delta > bestMulti.delta)) {
                const half = Math.floor(width / 2);
                bestMulti = {
                    center,
                    width,
                    layers: `L${center - half}-${center + half}`,
                    delta,
                    coherence
                };
            }
        });
    });

    if (bestSingle && bestMulti) {
        const deltaDiff = bestMulti.delta - bestSingle.delta;
        const cohDiff = bestMulti.coherence - bestSingle.coherence;

        content.innerHTML = `
            <table class="data-table" style="font-size: 12px;">
                <tr>
                    <td></td>
                    <td><strong>Single</strong></td>
                    <td><strong>Multi</strong></td>
                    <td><strong>Δ</strong></td>
                </tr>
                <tr>
                    <td>Config</td>
                    <td>L${bestSingle.layer}</td>
                    <td>${bestMulti.layers}</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td>Delta</td>
                    <td>${bestSingle.delta.toFixed(1)}</td>
                    <td>${bestMulti.delta.toFixed(1)}</td>
                    <td style="color: ${deltaDiff > 0 ? '#22c55e' : '#ef4444'}">${deltaDiff > 0 ? '+' : ''}${deltaDiff.toFixed(1)}</td>
                </tr>
                <tr>
                    <td>Coh</td>
                    <td>${bestSingle.coherence.toFixed(1)}</td>
                    <td>${bestMulti.coherence.toFixed(1)}</td>
                    <td style="color: ${cohDiff > 0 ? '#22c55e' : '#ef4444'}">${cohDiff > 0 ? '+' : ''}${cohDiff.toFixed(1)}</td>
                </tr>
            </table>
            <p style="color: #888; font-size: 11px; margin-top: 8px;">First-pass results. Further iteration may improve.</p>
        `;
    }
}

// Make globally available
window.renderMultiLayerHeatmap = renderMultiLayerHeatmap;
