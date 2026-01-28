// Trait Correlation View - Analyze relationships between trait projections
//
// Loads pre-computed correlation data from:
//   experiments/{exp}/analysis/trait_correlation/{prompt_set}.json
//
// To generate: python analysis/trait_correlation.py --experiment {exp} --prompt-set {prompt_set}

async function renderTraitCorrelation() {
    const contentArea = document.getElementById('content-area');

    if (!window.state.currentExperiment) {
        contentArea.innerHTML = `
            <div class="tool-view">
                <div class="no-data">
                    <p>Please select an experiment from the sidebar</p>
                </div>
            </div>
        `;
        return;
    }

    const promptSet = window.state.currentPromptSet;
    if (!promptSet) {
        contentArea.innerHTML = `
            <div class="tool-view">
                <div class="page-intro">
                    <div class="page-intro-text">Analyze correlations between trait projections across prompts.</div>
                </div>
                <div class="info">Select a prompt set from the prompt picker to analyze trait correlations.</div>
            </div>
        `;
        return;
    }

    // Load pre-computed correlation data
    const dataFile = `/experiments/${window.state.currentExperiment}/analysis/trait_correlation/${promptSet.replace('/', '_')}.json`;

    let data;
    try {
        const response = await fetch(dataFile);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        data = await response.json();
    } catch (e) {
        contentArea.innerHTML = `
            <div class="tool-view">
                <div class="page-intro">
                    <div class="page-intro-text">Analyze correlations between trait projections.</div>
                </div>
                <div class="info">
                    No correlation data found for prompt set "${promptSet}".
                    <br><br>
                    Generate it with:
                    <pre>python analysis/trait_correlation.py --experiment ${window.state.currentExperiment} --prompt-set ${promptSet}</pre>
                </div>
            </div>
        `;
        return;
    }

    // Store for slider updates
    window.state.traitCorrelationData = data;

    const currentOffset = window.state.correlationOffset || 0;
    const maxOffset = data.max_offset || 10;

    contentArea.innerHTML = `
        <div class="tool-view">
            <div class="page-intro">
                <div class="page-intro-text">Analyze correlations between trait projections.</div>
                <div class="page-intro-model">Prompt set: <code>${promptSet}</code> (${data.n_prompts} prompts, ${data.traits.length} traits)</div>
            </div>

            <section>
                ${ui.renderSubsection({
                    title: 'Trait Correlation Matrix',
                    infoId: 'info-trait-correlation',
                    infoText: 'Token-level correlation between trait projections. Upper triangle: row trait leads column trait by k tokens. Lower triangle: column trait leads row trait. Diagonal: autocorrelation at offset k.',
                    level: 'h2'
                })}
                <div class="projection-toggle">
                    <label class="projection-toggle-label">Offset: <span id="offset-value">${currentOffset}</span> tokens</label>
                    <input type="range" id="correlation-offset-slider" min="0" max="${maxOffset}" value="${currentOffset}" style="width: 200px; margin-left: 8px;">
                </div>
                <div id="correlation-heatmap"></div>
                <div id="correlation-legend" class="chart-legend" style="margin-top: 8px; font-size: 12px; color: var(--text-secondary);">
                    <span>Upper △: row leads col by +k</span>
                    <span style="margin-left: 16px;">Lower △: col leads row by +k</span>
                    <span style="margin-left: 16px;">Diagonal: autocorrelation at k</span>
                </div>
            </section>

            <section>
                ${ui.renderSubsection({
                    title: 'Correlation Decay',
                    infoId: 'info-correlation-decay',
                    infoText: 'How trait correlations change with token offset. Fast decay = local relationship. Slow decay = persistent relationship.',
                    level: 'h2'
                })}
                <div id="correlation-decay-plot"></div>
            </section>

            <section>
                ${ui.renderSubsection({
                    title: 'Response-Level Correlation',
                    infoId: 'info-response-correlation',
                    infoText: 'Correlation of mean projection per response (not token-level). Shows which traits co-occur across prompts.',
                    level: 'h2'
                })}
                <div id="response-correlation-heatmap"></div>
            </section>
        </div>
    `;

    window.setupSubsectionInfoToggles();

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
        window.state.correlationOffset = offset;
        renderCorrelationHeatmap(offset);
    });
}


function renderCorrelationHeatmap(offset) {
    const data = window.state.traitCorrelationData;
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
        colorscale: window.CORRELATION_COLORSCALE,
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
    const annotations = [];
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            const val = displayMatrix[i][j];
            annotations.push({
                x: trait_labels[j],
                y: trait_labels[i],
                text: val.toFixed(2),
                showarrow: false,
                font: {
                    size: 11,
                    color: Math.abs(val) > 0.5 ? 'white' : '#333'
                }
            });
        }
    }

    const layout = window.buildChartLayout({
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

    window.renderChart('correlation-heatmap', [trace], layout);
}


function renderCorrelationDecay() {
    const data = window.state.traitCorrelationData;
    if (!data) return;

    const { trait_labels, correlations_by_offset, max_offset } = data;
    const colors = window.getChartColors();
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

    const layout = window.buildChartLayout({
        preset: 'timeSeries',
        traces,
        height: 350,
        legendPosition: 'right',
        xaxis: { title: 'Token Offset', dtick: 1 },
        yaxis: { title: 'Correlation', range: [-1, 1], zeroline: true, zerolinewidth: 1 },
        margin: { r: 150 }
    });

    window.renderChart('correlation-decay-plot', traces, layout);
}


function renderResponseCorrelation() {
    const data = window.state.traitCorrelationData;
    if (!data || !data.response_correlation) return;

    const { trait_labels, response_correlation } = data;
    const n = trait_labels.length;

    const trace = {
        z: response_correlation,
        x: trait_labels,
        y: trait_labels,
        type: 'heatmap',
        colorscale: window.CORRELATION_COLORSCALE,
        zmin: -1,
        zmax: 1,
        hovertemplate: '%{y} ↔ %{x}<br>r = %{z:.3f}<extra></extra>',
        showscale: true,
        colorbar: {
            title: 'Correlation',
            titleside: 'right'
        }
    };

    const annotations = [];
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            const val = response_correlation[i][j];
            annotations.push({
                x: trait_labels[j],
                y: trait_labels[i],
                text: val.toFixed(2),
                showarrow: false,
                font: {
                    size: 11,
                    color: Math.abs(val) > 0.5 ? 'white' : '#333'
                }
            });
        }
    }

    const layout = window.buildChartLayout({
        preset: 'heatmap',
        traces: [trace],
        title: 'Response-Level Correlation (mean projection per response)',
        xaxis: { title: '', tickangle: -45 },
        yaxis: { title: '', autorange: 'reversed' },
        annotations: annotations,
        margin: { l: 100, r: 80, t: 60, b: 100 }
    });

    window.renderChart('response-correlation-heatmap', [trace], layout);
}


// Export
window.renderTraitCorrelation = renderTraitCorrelation;
