// Trait Extraction - Comprehensive view of extraction quality, methods, and vector properties

async function renderTraitExtraction() {
    const contentArea = document.getElementById('content-area');

    // Show loading state only if fetch takes > 150ms
    const loadingTimeout = setTimeout(() => {
        contentArea.innerHTML = '<div class="loading">Loading extraction evaluation data...</div>';
    }, 150);

    // Load extraction evaluation data
    const evalData = await loadExtractionEvaluation();

    clearTimeout(loadingTimeout);

    if (!evalData || !evalData.all_results || evalData.all_results.length === 0) {
        contentArea.innerHTML = `
            <div class="tool-view">
                <div class="no-data">
                    <p>No extraction evaluation data</p>
                    <small>Run: <code>python analysis/vectors/extraction_evaluation.py --experiment ${window.state.experimentData?.name || 'your_experiment'}</code></small>
                </div>
            </div>
        `;
        return;
    }

    // Get extraction model from experiment config
    const config = window.state.experimentData?.experimentConfig;
    const extractionVariant = evalData.model_variant || config?.defaults?.extraction || 'base';
    const extractionModel = config?.model_variants?.[extractionVariant]?.model || 'unknown';

    // Build the comprehensive view
    contentArea.innerHTML = `
        <div class="tool-view">
            <!-- Page intro -->
            <div class="page-intro">
                <div class="page-intro-text">Measure quality of extracted trait vectors.</div>
                <div class="page-intro-model">Extraction model: <code>${extractionModel}</code></div>
                <div class="intro-example">
                    <div><span class="example-label">Mean diff example (refusal):</span></div>
                    <div><span class="pos">v_pos</span>  "How do I make a bomb?" → model refuses</div>
                    <div><span class="neg">v_neg</span>  "How do I make a cake?" → model answers</div>
                    <div class="intro-example-result">v_refusal = mean(<span class="pos">v_pos</span>) - mean(<span class="neg">v_neg</span>)</div>
                </div>
            </div>

            <!-- Best Vectors Summary -->
            <section>
                <h3 class="subsection-header" id="best-vectors">
                    Best Vectors Summary
                    <span class="subsection-info-toggle" data-target="info-best-vectors">►</span>
                </h3>
                <div class="subsection-info" id="info-best-vectors">Best vector per trait by effect size. \\(d = \\frac{\\mu_{pos} - \\mu_{neg}}{\\sigma_{pooled}}\\)</div>
                <div id="best-vectors-summary-container"></div>
            </section>

            <!-- Per-Trait Heatmaps -->
            <section>
                <h3 class="subsection-header" id="heatmaps">
                    Per-Trait Heatmaps (Layer × Method)
                    <span class="subsection-info-toggle" data-target="info-heatmaps">►</span>
                </h3>
                <div class="subsection-info" id="info-heatmaps">Validation accuracy across layers (rows) and methods (columns). Bright = high accuracy. ★ = best.</div>
                <div id="trait-heatmaps-container"></div>
            </section>

            <!-- Logit Lens -->
            <section>
                <h3 class="subsection-header" id="logit-lens">
                    Token Decode (Logit Lens)
                    <span class="subsection-info-toggle" data-target="info-logit-lens">►</span>
                </h3>
                <div class="subsection-info" id="info-logit-lens">Project vectors through unembedding to see which tokens they represent. Late layer (90% depth) shown.</div>
                <div id="logit-lens-container"></div>
            </section>

            <!-- Reference (collapsible) -->
            <section>
                <details class="reference-section">
                    <summary><h3 style="display: inline;">Reference</h3></summary>
                    <div class="reference-content">
                        <h4>Notation</h4>
                        ${renderNotation()}
                        <h4>Extraction Methods</h4>
                        ${renderExtractionTechniques()}
                        <h4>Quality Metrics</h4>
                        ${renderMetricsDefinitions()}
                    </div>
                </details>
            </section>

        </div>
    `;

    // Render each visualization
    renderBestVectorsSummary(evalData);
    renderTraitHeatmaps(evalData);
    renderLogitLensSection(evalData);

    // Render math after all content is in DOM
    if (window.MathJax) {
        MathJax.typesetPromise();
    }

    // Setup info toggles
    window.setupSubsectionInfoToggles();
}


async function loadExtractionEvaluation() {
    try {
        const url = window.paths.extractionEvaluation();
        const response = await fetch(url);
        if (!response.ok) return null;
        return await response.json();
    } catch (error) {
        console.error('Failed to load extraction evaluation:', error);
        return null;
    }
}


/**
 * Compute best vector per trait from all_results using effect_size.
 * Returns: {trait: {layer, method, score}}
 */
function computeBestVectors(allResults) {
    const bestByTrait = {};
    for (const r of allResults) {
        const trait = r.trait;
        const effectSize = r.val_effect_size;
        if (effectSize == null) continue;

        if (!bestByTrait[trait] || effectSize > bestByTrait[trait].score) {
            bestByTrait[trait] = {
                layer: r.layer,
                method: r.method,
                score: effectSize,
                source: 'effect_size'
            };
        }
    }
    return bestByTrait;
}


/**
 * Render best vectors summary table - one row per trait with key metrics
 */
function renderBestVectorsSummary(evalData) {
    const container = document.getElementById('best-vectors-summary-container');
    if (!container) return;

    const allResults = evalData.all_results || [];
    const bestVectors = computeBestVectors(allResults);

    if (Object.keys(bestVectors).length === 0) {
        container.innerHTML = '<p>No extraction results available.</p>';
        return;
    }

    // Filter by selected traits from sidebar
    const filteredTraits = window.getFilteredTraits();
    const selectedTraitNames = new Set(filteredTraits.map(t => t.name));
    const traits = selectedTraitNames.size > 0
        ? Object.keys(bestVectors).filter(t => selectedTraitNames.has(t))
        : Object.keys(bestVectors);

    // Build rows with metrics from best vector
    const rows = traits.map(trait => {
        const best = bestVectors[trait];
        // Find the full result for this best vector
        const result = allResults.find(r =>
            r.trait === trait && r.method === best.method && r.layer === best.layer
        );

        return {
            trait: window.getDisplayName(trait),
            method: best.method,
            layer: best.layer,
            accuracy: result?.val_accuracy ?? null,
            effectSize: result?.val_effect_size ?? null,
            auc: result?.val_auc_roc ?? null,
            drop: result?.accuracy_drop ?? null,
            source: best.source || 'effect_size'
        };
    }).sort((a, b) => a.trait.localeCompare(b.trait));

    let html = `
        <table class="data-table best-vectors-table">
            <thead>
                <tr>
                    <th>Trait</th>
                    <th>Best Method</th>
                    <th>Layer</th>
                    <th>Val Accuracy</th>
                    <th>Effect Size (d)</th>
                    <th>AUC-ROC</th>
                    <th>Acc Drop</th>
                </tr>
            </thead>
            <tbody>
    `;

    rows.forEach(row => {
        html += `
            <tr>
                <td><strong>${row.trait}</strong></td>
                <td>${row.method}</td>
                <td>L${row.layer}</td>
                <td>${row.accuracy !== null ? (row.accuracy * 100).toFixed(1) + '%' : 'N/A'}</td>
                <td>${row.effectSize !== null ? row.effectSize.toFixed(2) : 'N/A'}</td>
                <td>${row.auc !== null ? (row.auc * 100).toFixed(1) + '%' : 'N/A'}</td>
                <td>${row.drop !== null ? (row.drop * 100).toFixed(1) + '%' : 'N/A'}</td>
            </tr>
        `;
    });

    html += `
            </tbody>
        </table>
    `;

    container.innerHTML = html;
}


function renderNotation() {
    return `
        <div class="category-reference">
            <details>
                <summary>Input Shapes & Variables</summary>
                <table class="def-table">
                    <tr><td>$$n$$</td><td>Number of examples (train or validation split)</td></tr>
                    <tr><td>$$d$$</td><td>Hidden dimension (model-specific)</td></tr>
                    <tr><td>$$L$$</td><td>Number of layers (model-specific)</td></tr>
                    <tr><td>$$\\mathbf{A} \\in \\mathbb{R}^{n \\times d}$$</td><td>Activation matrix (token-averaged per example)</td></tr>
                    <tr><td>$$\\vec{v} \\in \\mathbb{R}^d$$</td><td>Trait vector (direction in activation space)</td></tr>
                    <tr><td>$$\\vec{a}_i \\in \\mathbb{R}^d$$</td><td>Single example's activation (row of A)</td></tr>
                    <tr><td>$$y_i \\in \\{+1, -1\\}$$</td><td>Binary label (positive/negative trait)</td></tr>
                </table>
            </details>
            <details>
                <summary>Key Quantities</summary>
                <table class="def-table">
                    <tr><td>$$\\vec{a} \\cdot \\vec{v}$$</td><td>Projection score (dot product)</td></tr>
                    <tr><td>$$\\mu_{\\text{pos}}, \\mu_{\\text{neg}}$$</td><td>Mean projection for pos/neg examples</td></tr>
                    <tr><td>$$\\sigma_{\\text{pooled}}$$</td><td>Pooled standard deviation</td></tr>
                    <tr><td>$$||\\vec{v}||_2$$</td><td>L2 norm (vector magnitude)</td></tr>
                </table>
            </details>
            <details>
                <summary>Pipeline Context</summary>
                <ul>
                    <li><strong>Train split:</strong> 80% of examples → used to extract vectors</li>
                    <li><strong>Val split:</strong> 20% of examples → used to evaluate vectors</li>
                    <li><strong>Per-layer:</strong> Vectors extracted independently for each layer</li>
                    <li><strong>Per-method:</strong> 4 extraction methods × L layers = 4L vectors/trait</li>
                </ul>
            </details>
        </div>
    `;
}


function renderExtractionTechniques() {
    return `
        <div class="category-reference">
            <details>
                <summary>Mean Difference</summary>
                <p>$$\\vec{v} = \\text{mean}(\\mathbf{A}_{\\text{pos}}) - \\text{mean}(\\mathbf{A}_{\\text{neg}})$$</p>
                <p>Direction between cluster centroids. Fast baseline, but ignores class shape/spread.</p>
            </details>
            <details>
                <summary>Linear Probe</summary>
                <p>$$\\min_\\vec{w} \\sum_i \\log(1 + e^{-y_i (\\vec{w} \\cdot \\vec{a}_i)})$$</p>
                <p>Logistic regression weights. Optimizes for <em>separability</em>, not just distance—handles overlap better.</p>
            </details>
            <details>
                <summary>Gradient</summary>
                <p>$$\\max_\\vec{v} \\left( \\text{mean}(\\mathbf{A}_{\\text{pos}} \\cdot \\vec{v}) - \\text{mean}(\\mathbf{A}_{\\text{neg}} \\cdot \\vec{v}) \\right)$$</p>
                <p>Direct optimization of separation. Best for low-separability traits where other methods fail.</p>
            </details>
            <details>
                <summary>Random Baseline</summary>
                <p>$$\\vec{v} \\sim \\mathcal{N}(0, I), \\quad \\|\\vec{v}\\| = 1$$</p>
                <p>Random unit vector. Sanity check—should get ~50% accuracy. If not, something's wrong.</p>
            </details>
        </div>
    `;
}


function renderMetricsDefinitions() {
    return `
        <div class="category-reference">
            <details>
                <summary>Accuracy</summary>
                <p>$$\\text{acc} = \\frac{\\text{correct classifications}}{\\text{total examples}}$$</p>
                <p>Percentage of validation examples correctly classified. Range: 0-1. <strong class="quality-good">Good: &gt; 0.90</strong></p>
            </details>
            <details>
                <summary>AUC-ROC</summary>
                <p>$$\\text{AUC} = \\int_0^1 \\text{TPR}(\\text{FPR}^{-1}(t)) \\, dt$$</p>
                <p>Area Under ROC Curve. Threshold-independent. Range: 0.5-1. <strong class="quality-good">Good: &gt; 0.90</strong></p>
            </details>
            <details>
                <summary>Effect Size (Cohen's d)</summary>
                <p>$$d = \\frac{\\mu_{\\text{pos}} - \\mu_{\\text{neg}}}{\\sigma_{\\text{pooled}}}$$</p>
                <p>Separation in standard deviation units. Range: 0-∞. <strong class="quality-good">Good: &gt; 1.5</strong></p>
            </details>
            <details>
                <summary>Vector Norm</summary>
                <p>$$||\\vec{v}||_2 = \\sqrt{\\sum_i v_i^2}$$</p>
                <p>L2 norm of vector. Range: 0-∞. Typical: 15-40</p>
            </details>
            <details>
                <summary>Separation Margin</summary>
                <p>$$(\\mu_{\\text{pos}} - \\sigma_{\\text{pos}}) - (\\mu_{\\text{neg}} + \\sigma_{\\text{neg}})$$</p>
                <p>Gap between distributions. Positive = good separation. <strong class="quality-good">Good: &gt; 0</strong></p>
            </details>
            <details>
                <summary>Sparsity & Overlap</summary>
                <p><strong>Sparsity:</strong> % of near-zero components (0 = dense, 1 = sparse)</p>
                <p><strong>Overlap:</strong> Distribution overlap estimate. <strong class="quality-good">Good: &lt; 0.2</strong></p>
            </details>
        </div>
    `;
}


function renderTraitHeatmaps(evalData) {
    const container = document.getElementById('trait-heatmaps-container');
    if (!container) return;

    const allResults = evalData.all_results || [];
    if (allResults.length === 0) {
        container.innerHTML = '<p>No results to display.</p>';
        return;
    }

    // Filter by selected traits from sidebar
    const filteredTraits = window.getFilteredTraits();
    const selectedTraitNames = new Set(filteredTraits.map(t => t.name));
    const results = selectedTraitNames.size > 0
        ? allResults.filter(r => selectedTraitNames.has(r.trait))
        : allResults;

    if (results.length === 0) {
        container.innerHTML = '<p>No results for selected traits.</p>';
        return;
    }

    // Compute max effect per trait for normalization
    const maxEffectPerTrait = {};
    results.forEach(r => {
        if (!maxEffectPerTrait[r.trait] || r.val_effect_size > maxEffectPerTrait[r.trait]) {
            maxEffectPerTrait[r.trait] = r.val_effect_size || 0;
        }
    });

    // Score computation function
    const computeScore = (r) => {
        const maxEffect = maxEffectPerTrait[r.trait] || 1;
        const normEffect = (r.val_effect_size || 0) / maxEffect;
        const accDrop = r.accuracy_drop || 0;
        const polarity = r.polarity_correct ? 1 : 0;
        return ((r.val_accuracy || 0) + normEffect + (1 - accDrop)) / 3 * polarity;
    };

    // Group by trait
    const traitGroups = {};
    results.forEach(r => {
        if (!traitGroups[r.trait]) traitGroups[r.trait] = [];
        traitGroups[r.trait].push(r);
    });

    const traits = Object.keys(traitGroups).sort();

    // Compute best vectors for star indicators
    const bestVectors = computeBestVectors(results);
    const hasBestVectors = Object.keys(bestVectors).length > 0;

    // Create grid with legend below
    container.innerHTML = `
        <div class="trait-heatmaps-grid" id="heatmaps-grid"></div>
        <div class="heatmap-legend-footer">
            <span class="file-hint">${traits.length} traits</span>
            ${hasBestVectors ? '<span class="file-hint" title="Best layer by effect size">★ = best</span>' : ''}
            <div class="heatmap-legend">
                <span>Score:</span>
                <div>
                    <div class="heatmap-legend-bar"></div>
                    <div class="heatmap-legend-labels">
                        <span>0%</span>
                        <span>50%</span>
                        <span>100%</span>
                    </div>
                </div>
            </div>
        </div>
    `;

    const grid = document.getElementById('heatmaps-grid');

    // Create compact heatmap for each trait
    traits.forEach(trait => {
        const traitResults = traitGroups[trait];
        const traitId = trait.replace(/\//g, '-');
        const displayName = window.getDisplayName(trait);
        const bestInfo = bestVectors[trait];

        const traitDiv = document.createElement('div');
        traitDiv.className = 'trait-heatmap-item';
        traitDiv.innerHTML = `
            <h4 title="${displayName}${bestInfo ? ` (best: L${bestInfo.layer} ${bestInfo.method} from ${bestInfo.source})` : ''}">${displayName}</h4>
            <div id="heatmap-${traitId}" class="chart-container-sm"></div>
        `;

        grid.appendChild(traitDiv);

        renderSingleTraitHeatmap(traitResults, `heatmap-${traitId}`, computeScore, true, bestInfo);
    });
}


function renderSingleTraitHeatmap(traitResults, containerId, computeScore, compact = false, bestInfo = null) {
    const methods = ['mean_diff', 'probe', 'gradient'];
    const layers = Array.from(new Set(traitResults.map(r => r.layer))).sort((a, b) => a - b);

    // Build matrix: layers × methods, value = score
    const matrix = [];
    layers.forEach(layer => {
        const row = methods.map(method => {
            const result = traitResults.find(r => r.layer === layer && r.method === method);
            return result ? computeScore(result) * 100 : null;
        });
        matrix.push(row);
    });

    // Compute dynamic zmin from actual data (round down to nearest 10)
    const allValues = matrix.flat().filter(v => v !== null);
    const minValue = allValues.length > 0 ? Math.min(...allValues) : 0;
    const zmin = Math.floor(minValue / 10) * 10;

    const xLabels = compact ? ['MD', 'Pr', 'Gr'] : methods;

    const trace = {
        z: matrix,
        x: xLabels,
        y: layers,
        type: 'heatmap',
        colorscale: window.ASYMB_COLORSCALE,
        hovertemplate: '%{x} L%{y}: %{z:.1f}%<extra></extra>',
        zmin: zmin,
        zmax: 100,
        showscale: !compact
    };

    if (!compact) {
        trace.colorbar = {
            title: { text: 'Score %', font: { size: 11 } },
            tickvals: [0, 50, 100],
            ticktext: ['0%', '50%', '100%']
        };
    }

    const layout = compact ? {
        margin: { l: 5, r: 5, t: 5, b: 25 },
        xaxis: { tickfont: { size: 8 }, tickangle: 0 },
        yaxis: { showticklabels: false, title: '' },
        height: 180,
        annotations: []
    } : {
        margin: { l: 40, r: 80, t: 20, b: 60 },
        xaxis: { title: 'Method', tickfont: { size: 11 } },
        yaxis: { title: 'Layer', tickfont: { size: 10 } },
        height: 400,
        annotations: []
    };

    // Add star annotation on best cell if available
    if (bestInfo && bestInfo.layer !== undefined && bestInfo.method) {
        const methodIdx = methods.indexOf(bestInfo.method);
        const layerIdx = layers.indexOf(bestInfo.layer);
        if (methodIdx >= 0 && layerIdx >= 0) {
            layout.annotations.push({
                x: xLabels[methodIdx],
                y: bestInfo.layer,
                text: '★',
                showarrow: false,
                font: { size: compact ? 10 : 14, color: '#000' },
                xanchor: 'center',
                yanchor: 'middle'
            });
        }
    }

    Plotly.newPlot(containerId, [trace], window.getPlotlyLayout(layout), { displayModeBar: false, responsive: true });
}


// =========================================================================
// Logit Lens - Token decode for trait vectors
// =========================================================================

/**
 * Render the logit lens section with all traits
 */
async function renderLogitLensSection(evalData) {
    const container = document.getElementById('logit-lens-container');
    if (!container) return;

    const allResults = evalData.all_results || [];
    const traits = [...new Set(allResults.map(r => r.trait))].sort();

    // Get model variant from eval data (extraction model variant)
    const modelVariant = evalData.model_variant || 'base';

    if (traits.length === 0) {
        container.innerHTML = '<p class="na">No traits available.</p>';
        return;
    }

    // Show loading
    container.innerHTML = '<p class="hint">Loading token decodes...</p>';

    // Load all logit lens data in parallel
    const results = await Promise.all(traits.map(async trait => {
        try {
            const url = window.paths.logitLens(trait, modelVariant);
            const response = await fetch(url);
            if (!response.ok) return { trait, data: null };
            return { trait, data: await response.json() };
        } catch {
            return { trait, data: null };
        }
    }));

    // Filter to traits that have data
    const withData = results.filter(r => r.data);

    if (withData.length === 0) {
        container.innerHTML = '<p class="hint">No logit lens data. Run: <code>python extraction/run_pipeline.py --experiment {exp} --traits {trait} --only-stage 5</code></p>';
        return;
    }

    // Build table
    const escapeHtml = (str) => {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    };

    const renderTokens = (tokens, limit = 5) => {
        if (!tokens || !Array.isArray(tokens)) return '<span class="na">—</span>';
        return tokens.slice(0, limit)
            .map(t => `<span class="ll-token">${escapeHtml(t.token)}</span>`)
            .join(' ');
    };

    let html = `
        <table class="data-table ll-table">
            <thead>
                <tr>
                    <th>Trait</th>
                    <th>Layer</th>
                    <th>→ Toward</th>
                    <th>← Away</th>
                </tr>
            </thead>
            <tbody>
    `;

    for (const { trait, data } of withData) {
        // Pick best method
        const methodPriority = ['probe', 'mean_diff', 'gradient'];
        const method = methodPriority.find(m => data.methods[m]) || Object.keys(data.methods)[0];
        const methodData = data.methods[method];
        if (!methodData || !methodData.late) continue;

        const displayName = window.getDisplayName(trait);
        const late = methodData.late;

        html += `
            <tr>
                <td><strong>${displayName}</strong><br><span class="hint">${method}</span></td>
                <td class="hint">L${late.layer}</td>
                <td class="ll-toward">${renderTokens(late.toward)}</td>
                <td class="ll-away">${renderTokens(late.away)}</td>
            </tr>
        `;
    }

    html += '</tbody></table>';
    container.innerHTML = html;
}

// Export
window.renderTraitExtraction = renderTraitExtraction;
