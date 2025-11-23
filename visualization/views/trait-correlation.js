// Trait Correlation Matrix View
// Compute pairwise correlations between trait projections to identify independence vs redundancy

async function renderTraitCorrelation() {
    const contentArea = document.getElementById('content-area');
    const filteredTraits = window.getFilteredTraits();

    if (filteredTraits.length < 2) {
        contentArea.innerHTML = `
            <div class="card">
                <div class="card-title">Trait Correlation Matrix</div>
                <div class="info">Select at least 2 traits to compute correlation matrix.</div>
            </div>
        `;
        return;
    }

    contentArea.innerHTML = `
        <div class="explanation">
            <div class="explanation-summary">Measure how similar different trait vectors are by computing pairwise correlations across token projections.</div>
            <div class="explanation-details">
                <h4>What This Measures</h4>
                <p><strong>Correlation Matrix:</strong> For each pair of traits, compute Pearson correlation coefficient across all token projections for a given prompt. High correlation (near +1 or -1) suggests traits may be measuring similar computations.</p>

                <h4>Interpretation</h4>
                <ul>
                    <li><strong>r ≈ +1.0:</strong> Traits move together (may be redundant)</li>
                    <li><strong>r ≈ 0.0:</strong> Traits are independent (measuring different computations)</li>
                    <li><strong>r ≈ -1.0:</strong> Traits move in opposite directions (inversely related)</li>
                </ul>

                <h4>Use Cases</h4>
                <ul>
                    <li>Identify redundant traits that can be merged</li>
                    <li>Validate framework coherence (expect low correlations for well-designed traits)</li>
                    <li>Discover trait relationships and confounds</li>
                </ul>

                <p><strong>Method:</strong> Pearson correlation coefficient \\(r = \\frac{\\sum (x_i - \\bar{x})(y_i - \\bar{y})}{\\sqrt{\\sum (x_i - \\bar{x})^2 \\sum (y_i - \\bar{y})^2}}\\)</p>
            </div>
        </div>

        <div id="correlation-loading" style="padding: 16px; color: var(--text-secondary);">
            Loading trait projections for prompt ${window.state.currentPrompt}...
        </div>

        <div id="correlation-content" style="display: none;"></div>
    `;

    try {
        // Load projection data for all selected traits
        // Use global paths singleton (already loaded and has experiment set)
        const promptNum = window.state.currentPrompt;

        const projectionData = {};
        const loadingPromises = filteredTraits.map(async trait => {
            try {
                // Try all-layers data first (has projections for all layers)
                const allLayersUrl = window.paths.allLayersData(trait, promptNum);
                const allLayersResponse = await fetch(allLayersUrl);

                if (allLayersResponse.ok) {
                    const data = await allLayersResponse.json();
                    // Extract layer 16 projections (middle layer)
                    if (data.projections && data.projections.response && data.projections.response.length > 0) {
                        // Get layer 16 (or middle layer)
                        const layerIdx = Math.min(16, data.projections.response[0].length - 1);
                        const scores = data.projections.response.map(tokenProjs => 
                            (tokenProjs[layerIdx][0] + tokenProjs[layerIdx][1] + tokenProjs[layerIdx][2]) / 3
                        );
                        projectionData[trait.name] = {
                            scores: scores,
                            tokens: data.response.tokens  // Fixed: was data.tokens.response
                        };
                        console.log(`Loaded ${scores.length} projections for ${trait.name}`);
                    }
                } else {
                    console.warn(`No all-layers data for ${trait.name} prompt ${promptNum}`);
                }
            } catch (e) {
                console.error(`Failed to load projections for ${trait.name}:`, e);
            }
        });

        await Promise.all(loadingPromises);

        const traitsWithData = Object.keys(projectionData);

        if (traitsWithData.length < 2) {
            document.getElementById('correlation-loading').innerHTML = `
                <div style="color: var(--danger); font-size: 12px;">
                    ⚠️ Not enough projection data available for selected traits.
                    <br><br>
                    <span style="color: var(--text-secondary); font-size: 11px;">
                        Only ${traitsWithData.length} trait(s) have inference data for prompt ${promptNum}.
                        Need at least 2 traits with all-layers projection data.
                    </span>
                </div>
            `;
            return;
        }

        // Compute correlation matrix
        const correlationMatrix = computeCorrelationMatrix(traitsWithData, projectionData);

        // Hide loading, show content
        document.getElementById('correlation-loading').style.display = 'none';
        document.getElementById('correlation-content').style.display = 'block';

        // Render correlation heatmap
        renderCorrelationHeatmap(traitsWithData, correlationMatrix, projectionData);

    } catch (error) {
        console.error('Error computing trait correlations:', error);
        document.getElementById('correlation-loading').innerHTML = `
            <div style="color: var(--danger); font-size: 12px;">
                Error: ${error.message}
            </div>
        `;
    }
}

/**
 * Compute pairwise Pearson correlation matrix
 * @param {Array<string>} traitNames - List of trait names
 * @param {Object} projectionData - Map of trait name to {scores, tokens}
 * @returns {Array<Array<number>>} - 2D correlation matrix
 */
function computeCorrelationMatrix(traitNames, projectionData) {
    const n = traitNames.length;
    const matrix = Array(n).fill(0).map(() => Array(n).fill(0));

    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            if (i === j) {
                matrix[i][j] = 1.0; // Perfect correlation with self
            } else {
                const x = projectionData[traitNames[i]].scores;
                const y = projectionData[traitNames[j]].scores;
                matrix[i][j] = pearsonCorrelation(x, y);
            }
        }
    }

    return matrix;
}

/**
 * Compute Pearson correlation coefficient between two arrays
 * @param {Array<number>} x - First array
 * @param {Array<number>} y - Second array
 * @returns {number} - Correlation coefficient (-1 to +1)
 */
function pearsonCorrelation(x, y) {
    const n = Math.min(x.length, y.length);
    if (n === 0) return 0;

    // Compute means
    const meanX = x.slice(0, n).reduce((a, b) => a + b, 0) / n;
    const meanY = y.slice(0, n).reduce((a, b) => a + b, 0) / n;

    // Compute covariance and standard deviations
    let cov = 0;
    let varX = 0;
    let varY = 0;

    for (let i = 0; i < n; i++) {
        const dx = x[i] - meanX;
        const dy = y[i] - meanY;
        cov += dx * dy;
        varX += dx * dx;
        varY += dy * dy;
    }

    // Handle edge cases
    if (varX === 0 || varY === 0) return 0;

    return cov / Math.sqrt(varX * varY);
}

/**
 * Render correlation heatmap
 * @param {Array<string>} traitNames - List of trait names
 * @param {Array<Array<number>>} matrix - Correlation matrix
 * @param {Object} projectionData - Raw projection data
 */
function renderCorrelationHeatmap(traitNames, matrix, projectionData) {
    const contentDiv = document.getElementById('correlation-content');

    // Get display names
    const displayNames = traitNames.map(name => window.getDisplayName(name));

    // Find strongest correlations (excluding diagonal)
    let strongestPairs = [];
    for (let i = 0; i < traitNames.length; i++) {
        for (let j = i + 1; j < traitNames.length; j++) {
            strongestPairs.push({
                trait1: traitNames[i],
                trait2: traitNames[j],
                correlation: matrix[i][j]
            });
        }
    }
    strongestPairs.sort((a, b) => Math.abs(b.correlation) - Math.abs(a.correlation));

    // Compute average absolute correlation (excluding diagonal)
    let sumAbsCorr = 0;
    let count = 0;
    for (let i = 0; i < traitNames.length; i++) {
        for (let j = 0; j < traitNames.length; j++) {
            if (i !== j) {
                sumAbsCorr += Math.abs(matrix[i][j]);
                count++;
            }
        }
    }
    const avgAbsCorr = count > 0 ? sumAbsCorr / count : 0;

    // Summary stats
    const topPairs = strongestPairs.slice(0, 5);
    const summaryHTML = `
        <div style="margin: 16px 0; font-size: 12px;">
            <div style="display: flex; gap: 24px; margin-bottom: 12px;">
                <div><span style="color: var(--text-secondary);">Traits:</span> <span style="font-size: 14px; color: var(--text-primary);">${traitNames.length}</span></div>
                <div><span style="color: var(--text-secondary);">Avg |r|:</span> <span style="font-size: 14px; color: var(--text-primary);">${avgAbsCorr.toFixed(3)}</span></div>
                <div><span style="color: var(--text-secondary);">Prompt:</span> <span style="font-size: 14px; color: var(--text-primary);">${window.state.currentPrompt}</span></div>
            </div>

            ${topPairs.length > 0 ? `
                <div style="margin-top: 12px;">
                    <div style="font-weight: 600; color: var(--text-primary); margin-bottom: 6px;">Strongest Correlations:</div>
                    <table style="width: 100%; border-collapse: collapse; font-size: 11px;">
                        <thead>
                            <tr>
                                <th style="text-align: left; padding: 2px 8px 6px 0; color: var(--text-secondary); font-weight: 400;">Trait 1</th>
                                <th style="text-align: left; padding: 2px 8px 6px 8px; color: var(--text-secondary); font-weight: 400;">Trait 2</th>
                                <th style="text-align: right; padding: 2px 8px 6px 8px; color: var(--text-secondary); font-weight: 400;">r</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${topPairs.map(pair => {
                                const colorClass = Math.abs(pair.correlation) > 0.7 ? 'var(--danger)' :
                                                   Math.abs(pair.correlation) > 0.4 ? 'var(--primary-color)' :
                                                   'var(--text-primary)';
                                return `
                                    <tr>
                                        <td style="padding: 2px 8px 2px 0; color: var(--text-primary);">${window.getDisplayName(pair.trait1)}</td>
                                        <td style="padding: 2px 8px; color: var(--text-primary);">${window.getDisplayName(pair.trait2)}</td>
                                        <td style="padding: 2px 8px; text-align: right; color: ${colorClass}; font-weight: 600;">${pair.correlation.toFixed(3)}</td>
                                    </tr>
                                `;
                            }).join('')}
                        </tbody>
                    </table>
                </div>
            ` : ''}
        </div>
    `;

    contentDiv.innerHTML = summaryHTML + `
        <div style="margin-top: 16px;">
            <div class="card-title" style="margin-bottom: 8px;">Correlation Matrix</div>
            <div id="correlation-heatmap" style="width: 100%; height: 600px;"></div>
        </div>
    `;

    // Render heatmap with Plotly
    const reversedDisplayNames = displayNames.slice().reverse();
    const reversedMatrix = matrix.slice().reverse();

    const trace = {
        z: reversedMatrix,
        x: displayNames,
        y: reversedDisplayNames,
        type: 'heatmap',
        colorscale: [
            [0, '#d62728'],    // Strong negative (red)
            [0.5, '#ffffff'],  // Zero (white)
            [1, '#2ca02c']     // Strong positive (green)
        ],
        zmid: 0,
        zmin: -1,
        zmax: 1,
        colorbar: {
            title: { text: 'r', font: { size: 12 } },
            titleside: 'right',
            tickvals: [-1, -0.5, 0, 0.5, 1],
            ticktext: ['-1.0', '-0.5', '0.0', '0.5', '1.0']
        },
        hovertemplate: '%{y} ↔ %{x}<br>r = %{z:.3f}<extra></extra>',
        texttemplate: '%{z:.2f}',
        textfont: { size: 10 },
        showscale: true
    };

    Plotly.newPlot('correlation-heatmap', [trace], window.getPlotlyLayout({
        margin: { l: 150, r: 80, t: 150, b: 80 },
        xaxis: {
            title: '',
            side: 'top',
            tickangle: -45,
            tickfont: { size: 10 }
        },
        yaxis: {
            title: '',
            tickfont: { size: 10 }
        },
        height: 600
    }), { displayModeBar: false });

    // Render math
    if (window.MathJax) {
        MathJax.typesetPromise();
    }
}

// Export to global scope
window.renderTraitCorrelation = renderTraitCorrelation;
