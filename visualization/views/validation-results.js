// Validation Evaluation View - Comprehensive Heatmaps
// Shows all metrics (accuracy, effect_size, polarity) per method × trait × layer

let validationData = null;
let selectedMethod = 'probe';

async function loadValidationData() {
    if (validationData) return validationData;

    try {
        // Use global paths singleton with current experiment
        paths.setExperiment(window.state.currentExperiment);
        const url = paths.validationResults();
        const response = await fetch(url);
        if (!response.ok) throw new Error('Validation results not found');
        validationData = await response.json();
        return validationData;
    } catch (error) {
        console.error('Error loading validation data:', error);
        return null;
    }
}

function renderValidationResults() {
    const contentArea = document.getElementById('content-area');
    contentArea.innerHTML = '<div class="loading">Loading validation results...</div>';

    loadValidationData().then(data => {
        if (!data) {
            contentArea.innerHTML = `
                <div class="error">
                    <h3>Validation Results Not Found</h3>
                    <p>Run the validation evaluation script first:</p>
                    <pre>python analysis/evaluate_on_validation.py --experiment ${window.state.experiment}</pre>
                </div>
            `;
            return;
        }

        renderValidationContent(data);
    });
}

function renderValidationContent(data) {
    const contentArea = document.getElementById('content-area');
    const methods = ['probe', 'gradient', 'mean_diff', 'ica'];
    const traits = [...new Set(data.all_results.map(r => r.trait))].sort();
    const layers = [...new Set(data.all_results.map(r => r.layer))].sort((a, b) => a - b);

    // Compute overall stats
    const methodStats = computeMethodStats(data.all_results);
    const polarityIssues = data.all_results.filter(r => !r.polarity_correct);

    contentArea.innerHTML = `
        <div class="validation-container">
            <!-- Explanation -->
            <div class="explanation" style="margin-bottom: 24px;">
                <p style="font-size: 13px; color: var(--text-primary); max-width: 600px;">
                    Evaluate extracted vectors on held-out validation data to measure how well they generalize to new, unseen examples.
                </p>
                <ul style="font-size: 12px; color: var(--text-secondary); padding-left: 16px; margin-top: 8px;">
                    <li style="font-weight: 600; color: var(--text-primary); list-style: none; margin-left: -16px; margin-bottom: 4px;">Key Metrics</li>
                    <li><strong>Accuracy:</strong> Classification accuracy using a midpoint threshold.</li>
                    <li><strong>Effect Size (d):</strong> Standardized separation between positive and negative groups.</li>
                    <li><strong>P-value:</strong> Probability the observed separation is due to random chance (lower is better).</li>
                    <li><strong>Polarity ✓:</strong> Confirms the vector points from negative to positive examples.</li>
                </ul>
            </div>

            <!-- Summary Stats -->
            <div class="summary-stats" style="display: flex; flex-wrap: wrap; gap: 24px; margin-bottom: 20px;">
                ${methods.map(m => {
                    const stats = methodStats.byMethod[m];
                    if (!stats) return '';
                    const isBest = m === methodStats.best.method;
                    return `
                        <div class="summary-stat-group">
                            <div style="font-size: 11px; color: var(--text-secondary); text-transform: uppercase; ${isBest ? 'font-weight: 600;' : ''}">${m}</div>
                            <div style="font-size: 14px; font-weight: 600; color: var(--text-primary);">${(stats.meanAcc * 100).toFixed(1)}%</div>
                            <div style="font-size: 10px; color: var(--text-tertiary);">
                                d=${stats.meanEffect.toFixed(1)} | pol: ${(stats.polarityRate * 100).toFixed(0)}%
                            </div>
                        </div>
                    `;
                }).join('')}
            </div>

            <!-- Method Tabs -->
            <div class="method-tabs" style="display: flex; gap: 8px; margin-bottom: 16px;">
                ${methods.map(m => `
                    <button class="method-tab ${m === selectedMethod ? 'active' : ''}"
                            onclick="selectValidationMethod('${m}')"
                            style="padding: 4px 8px; border: none; border-radius: 4px; cursor: pointer;
                                   background: transparent;
                                   font-weight: ${m === selectedMethod ? '600' : '400'};
                                   color: ${m === selectedMethod ? 'var(--accent)' : 'var(--text-secondary)'};">
                        ${m}
                    </button>
                `).join('')}
            </div>

            <!-- Mega Heatmaps Container -->
            <div id="validation-heatmaps" style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px;">
            </div>

            <!-- Top Vectors Per Trait -->
            <div id="top-vectors-container" style="margin-top: 32px;"></div>

            <!-- Cross-Trait Independence Matrix -->
            <div style="margin-top: 32px;">
                <h3 style="font-size: 14px; font-weight: 600; margin-bottom: 4px; color: var(--text-primary);">Cross-Trait Independence (${selectedMethod}, layer 16)</h3>
                <p style="font-size: 12px; color: var(--text-secondary); margin-bottom: 12px; max-width: 600px;">
                    Measures vector interference. A vector for one trait (e.g., 'refusal') should have random (≈50%) accuracy on data for an unrelated trait (e.g., 'positivity'). The diagonal shows a vector's accuracy on its own trait data.
                </p>
                <div id="cross-trait-matrix-container"></div>
            </div>
        </div>
    `;

    renderMethodHeatmaps(data, selectedMethod, traits, layers);
    renderTopVectorsAcrossMethods(data, traits);
    renderCrossTraitMatrix(data.cross_trait_matrix);
}

function selectValidationMethod(method) {
    selectedMethod = method;
    if (validationData) {
        const traits = [...new Set(validationData.all_results.map(r => r.trait))].sort();
        const layers = [...new Set(validationData.all_results.map(r => r.layer))].sort((a, b) => a - b);

        // Update tab styling
        document.querySelectorAll('.method-tab').forEach(tab => {
            const isActive = tab.textContent.trim() === method;
            tab.style.background = 'transparent';
            tab.style.color = isActive ? 'var(--accent)' : 'var(--text-secondary)';
            tab.style.fontWeight = isActive ? '600' : '400';
            tab.className = `method-tab ${isActive ? 'active' : ''}`;
        });

        renderMethodHeatmaps(validationData, method, traits, layers);
    }
}

function renderMethodHeatmaps(data, method, traits, layers) {
    const container = document.getElementById('validation-heatmaps');
    if (!container) return;

    const methodResults = data.all_results.filter(r => r.method === method);

    // Build data matrices for each metric
    const metrics = [
        { key: 'val_accuracy', name: 'Accuracy (%)', scale: 100, colorscale: [[0, 'rgba(76, 175, 80, 0)'], [1, 'rgba(76, 175, 80, 1)']], zmin: 0, zmax: 100 },
        { key: 'val_effect_size', name: 'Effect Size (Cohen\'s d)', scale: 1, colorscale: [[0, 'rgba(76, 175, 80, 0)'], [1, 'rgba(76, 175, 80, 1)']], zmin: 0, zmax: 5 },
        { key: 'polarity_correct', name: 'Polarity Correct', scale: 1, colorscale: [[0, 'rgba(76, 175, 80, 0)'], [1, 'rgba(76, 175, 80, 1)']], zmin: 0, zmax: 1, binary: true },
        { key: 'val_p_value', name: 'P-value', scale: 1, colorscale: [[0, 'rgba(76, 175, 80, 1)'], [1, 'rgba(76, 175, 80, 0)']], zmin: 0, zmax: 0.1 }
    ];

    // Get short trait names for y-axis
    const shortTraits = traits.map(t => t.split('/').pop().substring(0, 15));

    container.innerHTML = '';

    metrics.forEach((metric, idx) => {
        // Build z matrix: traits × layers
        const z = traits.map(trait => {
            return layers.map(layer => {
                const result = methodResults.find(r => r.trait === trait && r.layer === layer);
                if (!result) return null;
                const val = result[metric.key];
                if (val === null || val === undefined) return null;
                return metric.binary ? (val ? 1 : 0) : val * metric.scale;
            });
        });

        // Create heatmap div
        const heatmapDiv = document.createElement('div');
        heatmapDiv.id = `heatmap-${metric.key}`;
        heatmapDiv.style.height = `${Math.max(200, traits.length * 25 + 80)}px`;
        container.appendChild(heatmapDiv);

        // Create heatmap with Plotly
        const heatmapData = [{
            z: z,
            x: layers,
            y: shortTraits,
            type: 'heatmap',
            colorscale: metric.colorscale,
            zmin: metric.zmin,
            zmax: metric.zmax,
            hoverongaps: false,
            hovertemplate: 'Trait: %{y}<br>Layer: %{x}<br>Value: %{z:.2f}<extra></extra>',
            colorbar: {
                title: metric.name,
                titleside: 'right',
                thickness: 15,
                len: 0.9
            }
        }];

        // Add annotations for best values per trait
        const annotations = [];
        traits.forEach((trait, traitIdx) => {
            let bestLayer = 0, bestVal = metric.key === 'val_accuracy' ? 0 : -Infinity;
            layers.forEach((layer, layerIdx) => {
                const val = z[traitIdx][layerIdx];
                if (val !== null) {
                    if (metric.key === 'val_accuracy' && val > bestVal) {
                        bestVal = val;
                        bestLayer = layer;
                    } else if (metric.key === 'val_effect_size' && val > bestVal) {
                        bestVal = val;
                        bestLayer = layer;
                    }
                }
            });

            // Only annotate accuracy heatmap with best layer
            if (metric.key === 'val_accuracy' && bestVal > 75) {
                annotations.push({
                    x: bestLayer,
                    y: shortTraits[traitIdx],
                    text: '★',
                    showarrow: false,
                    font: { size: 10 }
                });
            }
        });

        const styles = getComputedStyle(document.documentElement);
        const layout = {
            title: {
                text: `${metric.name} by Trait × Layer (${method})`,
                font: { size: 14, color: styles.getPropertyValue('--text-primary').trim() }
            },
            font: {
                color: styles.getPropertyValue('--text-primary').trim()
            },
            xaxis: {
                title: 'Layer',
                tickmode: 'linear',
                dtick: 2,
                color: styles.getPropertyValue('--text-secondary').trim()
            },
            yaxis: {
                title: '',
                automargin: true,
                color: styles.getPropertyValue('--text-secondary').trim()
            },
            paper_bgcolor: styles.getPropertyValue('--bg-primary').trim(),
            plot_bgcolor: styles.getPropertyValue('--bg-primary').trim(),
            margin: { l: 120, r: 80, t: 40, b: 40 },
            annotations: annotations
        };

        // Update colorbar fonts separately
        if (heatmapData[0].colorbar) {
            heatmapData[0].colorbar.title.font = { color: styles.getPropertyValue('--text-secondary').trim() };
            heatmapData[0].colorbar.tickfont = { color: styles.getPropertyValue('--text-secondary').trim() };
        }

        Plotly.newPlot(heatmapDiv.id, heatmapData, layout, { responsive: true });
    });
}

function renderTopVectorsAcrossMethods(data, traits) {
    const container = document.getElementById('top-vectors-container');
    if (!container) return;

    // --- Pre-compute the max effect size for each trait for normalization ---
    const maxEffectSizes = {};
    traits.forEach(trait => {
        const traitResults = data.all_results.filter(r => r.trait === trait && r.val_effect_size !== null);
        if (traitResults.length > 0) {
            const maxEffect = Math.max(...traitResults.map(r => r.val_effect_size));
            maxEffectSizes[trait] = maxEffect > 0 ? maxEffect : 1; // Avoid division by zero
        } else {
            maxEffectSizes[trait] = 1;
        }
    });

    // --- Helper function to calculate the new quality score ---
    const calculateQualityScore = (result, max_d) => {
        if (result.val_accuracy === null || result.val_effect_size === null) {
            return 0;
        }
        const normalizedEffectSize = (result.val_effect_size || 0) / max_d;
        const score = (0.5 * result.val_accuracy) + (0.5 * normalizedEffectSize);
        return score;
    };


    const tableRows = traits.map(trait => {
        // Get all results for this trait across all methods and layers
        const allTraitResults = data.all_results.filter(r => r.trait === trait);

        // Sort by val_accuracy, then val_effect_size
        const sortedResults = allTraitResults.sort((a, b) => {
            if (a.val_accuracy !== b.val_accuracy) {
                return b.val_accuracy - a.val_accuracy;
            }
            return b.val_effect_size - a.val_effect_size;
        });

        // Get top 3
        const top3 = sortedResults.slice(0, 3);

        const shortTrait = trait.split('/').pop();
        const max_d_for_trait = maxEffectSizes[trait];

        const rowsHtml = top3.map((best, index) => {
            if (!best) return '';
            const qualityScore = calculateQualityScore(best, max_d_for_trait);
            const accColor = best.val_accuracy >= 0.9 ? 'var(--success)' :
                             best.val_accuracy >= 0.75 ? 'var(--warning)' : 'var(--danger)';
            return `
                <tr>
                    ${index === 0 ? `<td rowspan="3" style="padding: 8px 12px 8px 8px; vertical-align: top;"><strong>${shortTrait}</strong></td>` : ''}
                    <td style="padding: 4px 8px; text-align: center; color: var(--text-secondary);">${best.method}</td>
                    <td style="padding: 4px 8px; text-align: center; color: var(--text-primary);">${best.layer}</td>
                    <td style="padding: 4px 8px; text-align: center; color: ${accColor}; font-weight: bold;">${(best.val_accuracy * 100).toFixed(1)}%</td>
                    <td style="padding: 4px 8px; text-align: center; color: var(--text-primary);">${best.val_effect_size?.toFixed(2) || 'N/A'}</td>
                    <td style="padding: 4px 8px; text-align: center; color: var(--text-primary);">${best.val_p_value < 0.001 ? '<0.001' : best.val_p_value?.toFixed(3) || 'N/A'}</td>
                    <td style="padding: 4px 8px; text-align: center; color: var(--text-primary);">${best.polarity_correct ? '✓' : '✗'}</td>
                    <td style="padding: 4px 8px; text-align: center; font-weight: 600; color: var(--text-primary);">${qualityScore.toFixed(2)}</td>
                </tr>
            `;
        }).join('');

        return rowsHtml;
    }).join('');


    // Summary table
    container.innerHTML = `
        <h3 style="font-size: 14px; font-weight: 600; margin-bottom: 4px; color: var(--text-primary);">Top 3 Vectors Per Trait</h3>
        <p style="font-size: 12px; color: var(--text-secondary); margin-bottom: 12px; max-width: 600px;">
            The best vectors for each trait across all methods, ranked by Accuracy then Effect Size.
        </p>
        <table style="width: 100%; border-collapse: collapse; font-size: 12px;">
            <thead>
                <tr>
                    <th style="padding: 0 8px 8px 8px; text-align: left; font-weight: 600; color: var(--text-secondary);">Trait</th>
                    <th style="padding: 0 8px 8px 8px; text-align: center; font-weight: 600; color: var(--text-secondary);">Method</th>
                    <th style="padding: 0 8px 8px 8px; text-align: center; font-weight: 600; color: var(--text-secondary);">Layer</th>
                    <th style="padding: 0 8px 8px 8px; text-align: center; font-weight: 600; color: var(--text-secondary);">Accuracy</th>
                    <th style="padding: 0 8px 8px 8px; text-align: center; font-weight: 600; color: var(--text-secondary);">Effect Size</th>
                    <th style="padding: 0 8px 8px 8px; text-align: center; font-weight: 600; color: var(--text-secondary);">P-value</th>
                    <th style="padding: 0 8px 8px 8px; text-align: center; font-weight: 600; color: var(--text-secondary);">Polarity</th>
                    <th style="padding: 0 8px 8px 8px; text-align: center; font-weight: 600; color: var(--text-secondary);">Quality Score</th>
                </tr>
            </thead>
            <tbody>
                ${tableRows}
            </tbody>
        </table>
    `;
}

function computeMethodStats(results) {
    const methods = ['probe', 'gradient', 'mean_diff', 'ica'];
    const byMethod = {};

    methods.forEach(method => {
        const methodResults = results.filter(r => r.method === method);
        if (methodResults.length === 0) return;

        const accuracies = methodResults.map(r => r.val_accuracy).filter(v => v !== null);
        const effects = methodResults.map(r => r.val_effect_size).filter(v => v !== null && !isNaN(v));
        const polarityOk = methodResults.filter(r => r.polarity_correct).length;

        byMethod[method] = {
            meanAcc: accuracies.length > 0 ? accuracies.reduce((a, b) => a + b, 0) / accuracies.length : 0,
            maxAcc: accuracies.length > 0 ? Math.max(...accuracies) : 0,
            meanEffect: effects.length > 0 ? effects.reduce((a, b) => a + b, 0) / effects.length : 0,
            polarityRate: methodResults.length > 0 ? polarityOk / methodResults.length : 0
        };
    });

    // Find best method
    let bestMethod = 'probe';
    let bestAcc = 0;
    for (const [method, stats] of Object.entries(byMethod)) {
        if (stats.meanAcc > bestAcc) {
            bestAcc = stats.meanAcc;
            bestMethod = method;
        }
    }

    return { byMethod, best: { method: bestMethod, accuracy: bestAcc } };
}

function renderCrossTraitMatrix(matrix) {
    const container = document.getElementById('cross-trait-matrix-container');
    if (!container || !matrix) {
        if (container) container.innerHTML = '<p style="color: var(--text-secondary);">No cross-trait matrix available.</p>';
        return;
    }

    const keys = Object.keys(matrix);
    const z = keys.map(i => keys.map(j => matrix[i][j] * 100));

    const styles = getComputedStyle(document.documentElement);

    const heatmapData = [{
        z: z,
        x: keys,
        y: keys,
        type: 'heatmap',
        colorscale: [[0, 'rgba(76, 175, 80, 0)'], [1, 'rgba(76, 175, 80, 1)']],
        zmin: 0,
        zmax: 100,
        hovertemplate: 'Test: %{y}<br>Vector: %{x}<br>Accuracy: %{z:.1f}%<extra></extra>',
        colorbar: {
            title: 'Accuracy %',
            titleside: 'right'
        }
    }];

    const layout = {
        font: {
            color: styles.getPropertyValue('--text-primary').trim()
        },
        xaxis: { title: 'Vector from trait', tickangle: 45, color: styles.getPropertyValue('--text-secondary').trim() },
        yaxis: { title: 'Test data from trait', color: styles.getPropertyValue('--text-secondary').trim() },
        paper_bgcolor: styles.getPropertyValue('--bg-primary').trim(),
        plot_bgcolor: styles.getPropertyValue('--bg-primary').trim(),
        margin: { l: 100, r: 50, t: 20, b: 100 },
        width: Math.min(600, keys.length * 50 + 150),
        height: Math.min(500, keys.length * 40 + 120)
    };

    // Update colorbar fonts separately
    if (heatmapData[0].colorbar) {
        heatmapData[0].colorbar.title.font = { color: styles.getPropertyValue('--text-secondary').trim() };
        heatmapData[0].colorbar.tickfont = { color: styles.getPropertyValue('--text-secondary').trim() };
    }

    Plotly.newPlot(container, heatmapData, layout, { responsive: true });
}

// Export
window.renderValidationResults = renderValidationResults;
window.selectValidationMethod = selectValidationMethod;
