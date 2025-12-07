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

    // Build the comprehensive view
    contentArea.innerHTML = `
        <div class="tool-view">
            <!-- Page intro -->
            <div class="page-intro">
                <div class="page-intro-text">Measure quality of extracted trait vectors.</div>
                <div class="intro-example">
                    <div><span class="example-label">Example (refusal):</span></div>
                    <div><span class="pos">v_pos</span>  "How do I make a bomb?" → model refuses</div>
                    <div><span class="neg">v_neg</span>  "How do I make a cake?" → model answers</div>
                    <div class="intro-example-result">v_refusal = <span class="pos">v_pos</span> - <span class="neg">v_neg</span></div>
                </div>
            </div>

            <!-- Table of Contents -->
            <nav class="toc">
                <div class="toc-title">Contents</div>
                <ol class="toc-list">
                    <li><a href="#heatmaps">Heatmaps</a></li>
                    <li><a href="#similarity">Similarity Matrix</a></li>
                    <li><a href="#method-breakdown">Per-Method Distributions</a></li>
                </ol>
            </nav>

            <!-- Section 1: Visualizations -->
            <section>
                <h3 class="subsection-header" id="heatmaps">
                    <span class="subsection-num">1.</span>
                    <span class="subsection-title">Per-Trait Quality Heatmaps (Layer × Method)</span>
                    <span class="subsection-info-toggle" data-target="info-heatmaps">►</span>
                </h3>
                <div class="subsection-info" id="info-heatmaps">Each heatmap shows validation accuracy across layers (rows) and extraction methods (columns) for one trait. Bright = high accuracy.</div>
                <div id="trait-heatmaps-container"></div>

                <h3 class="subsection-header" id="similarity">
                    <span class="subsection-num">2.</span>
                    <span class="subsection-title">Best-Vector Similarity Matrix (Trait Independence)</span>
                    <span class="subsection-info-toggle" data-target="info-similarity">►</span>
                </h3>
                <div class="subsection-info" id="info-similarity">Cosine similarity between best vectors for each trait pair. Low similarity means traits capture independent directions.</div>
                <div id="best-vector-similarity-container"></div>

                <h3 class="subsection-header" id="method-breakdown">
                    <span class="subsection-num">3.</span>
                    <span class="subsection-title">Per-Method Distributions</span>
                    <span class="subsection-info-toggle" data-target="info-method-breakdown">►</span>
                </h3>
                <div class="subsection-info" id="info-method-breakdown">Each row shows one extraction method. Histograms show distribution of Score, Accuracy, Effect (normalized), and 1−Drop across all layer×trait combinations. Compare methods to see which produces consistently good vectors.</div>
                <div id="method-breakdown-container"></div>
            </section>

            <!-- Section 2: Notation -->
            <section>
                <h2>Notation & Definitions <span class="subsection-info-toggle" data-target="info-notation">►</span></h2>
                <div class="subsection-info" id="info-notation">Symbols used throughout extraction pipeline. Each example's activation is the average across all response tokens, giving a single d-dimensional vector per example.</div>
                ${renderNotation()}
            </section>

            <!-- Section 3: Extraction Techniques -->
            <section>
                <h3 class="subsection-header" id="extraction-techniques">
                    <span class="subsection-num">4.</span>
                    <span class="subsection-title">Extraction Techniques</span>
                    <span class="subsection-info-toggle" data-target="info-techniques">►</span>
                </h3>
                <div class="subsection-info" id="info-techniques">
                    <strong>Key insight:</strong> Mean diff finds the direction between cluster centers; probe finds the direction that <em>best separates</em> the clusters (which may differ if clusters overlap or have different shapes). Probe typically outperforms when there's noise or class overlap.
                </div>
                ${renderExtractionTechniques()}
            </section>

            <!-- Section 4: Metrics Definitions -->
            <section>
                <h2>Quality Metrics <span class="subsection-info-toggle" data-target="info-metrics">►</span></h2>
                <div class="subsection-info" id="info-metrics">Computed on held-out validation data (20%). Accuracy >90% is good, effect size >1.5 is large.</div>
                ${renderMetricsDefinitions()}
            </section>

            <!-- Section 5: Scoring Method -->
            <section>
                <h2>Scoring & Ranking <span class="subsection-info-toggle" data-target="info-scoring">►</span></h2>
                <div class="subsection-info" id="info-scoring">Combined score balances accuracy (50%) and effect size (50%). High accuracy with tiny effect may be overfitting.</div>
                ${renderScoringExplanation()}
            </section>

            <!-- Section 6: All Metrics Overview -->
            <section>
                <h2>All Metrics Overview</h2>
                <div id="all-metrics-container"></div>
            </section>

        </div>
    `;

    // Render each visualization
    renderTraitHeatmaps(evalData);
    renderMethodBreakdown(evalData);
    renderBestVectorSimilarity(evalData);
    renderAllMetricsOverview(evalData);

    // Render math after all content is in DOM
    if (window.MathJax) {
        MathJax.typesetPromise();
    }

    // Setup info toggles
    setupSubsectionInfoToggles();
}


/**
 * Setup click handlers for subsection info toggles (▼ triangles)
 * Uses event delegation to handle dynamically added content
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


function renderScoringExplanation() {
    return `
        <div class="category-reference">
            <details>
                <summary>Combined Score Formula</summary>
                <p>$$\\text{score} = \\frac{\\text{accuracy} + \\text{norm\\_effect} + (1 - \\text{accuracy\\_drop})}{3} \\times \\text{polarity}$$</p>
                <ul>
                    <li><strong>Accuracy:</strong> Classification on held-out validation</li>
                    <li><strong>Normalized Effect:</strong> Cohen's d / max d for trait (separation for steering)</li>
                    <li><strong>1 − Drop:</strong> Generalization quality (high = validates well)</li>
                    <li><strong>Polarity:</strong> Wrong polarity (neg > pos) → score = 0</li>
                </ul>
                <p>Equal weights: accuracy (does it work?), effect (can it steer?), generalization (will it transfer?)</p>
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

    // Get best_vectors for star indicators
    const bestVectors = evalData.best_vectors || {};
    const hasBestVectors = Object.keys(bestVectors).length > 0;

    // Create grid with legend below
    container.innerHTML = `
        <div class="trait-heatmaps-grid" id="heatmaps-grid"></div>
        <div class="heatmap-legend-footer">
            <span class="file-hint">${traits.length} traits</span>
            ${hasBestVectors ? '<span class="file-hint" title="Best layer from steering results or effect size">★ = validated best</span>' : ''}
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


/**
 * Render per-method breakdown: 6 methods × 4 metrics = 24 histograms
 * Each histogram shows distribution of a metric for one method
 */
function renderMethodBreakdown(evalData) {
    const container = document.getElementById('method-breakdown-container');
    if (!container) return;

    const rawResults = evalData.all_results || [];
    if (rawResults.length === 0) {
        container.innerHTML = '<p>No results available.</p>';
        return;
    }

    // Filter by selected traits from sidebar
    const filteredTraits = window.getFilteredTraits();
    const selectedTraitNames = new Set(filteredTraits.map(t => t.name));
    const allResults = selectedTraitNames.size > 0
        ? rawResults.filter(r => selectedTraitNames.has(r.trait))
        : rawResults;

    if (allResults.length === 0) {
        container.innerHTML = '<p>No results for selected traits.</p>';
        return;
    }

    const methods = ['mean_diff', 'probe', 'gradient', 'random_baseline'];
    const methodLabels = {
        'mean_diff': 'Mean Diff',
        'probe': 'Probe',
        'gradient': 'Gradient',
        'random_baseline': 'Random'
    };

    // Compute max effect per trait for normalization
    const maxEffectPerTrait = {};
    allResults.forEach(r => {
        if (!maxEffectPerTrait[r.trait] || r.val_effect_size > maxEffectPerTrait[r.trait]) {
            maxEffectPerTrait[r.trait] = r.val_effect_size || 0;
        }
    });

    // Score computation
    const computeScore = (r) => {
        const maxEffect = maxEffectPerTrait[r.trait] || 1;
        const normEffect = (r.val_effect_size || 0) / maxEffect;
        const accDrop = r.accuracy_drop || 0;
        const polarity = r.polarity_correct ? 1 : 0;
        return ((r.val_accuracy || 0) + normEffect + (1 - accDrop)) / 3 * polarity;
    };

    // Metric definitions (no preset ranges - will use data min/max)
    const metrics = [
        { key: 'score', label: 'Score', getValue: r => computeScore(r) * 100 },
        { key: 'accuracy', label: 'Accuracy', getValue: r => (r.val_accuracy || 0) * 100 },
        { key: 'effect', label: 'Effect (norm)', getValue: r => ((r.val_effect_size || 0) / (maxEffectPerTrait[r.trait] || 1)) * 100 },
        { key: 'drop', label: '1−Drop', getValue: r => (1 - (r.accuracy_drop || 0)) * 100 }
    ];

    // Build HTML grid: rows = methods, cols = metrics
    let html = `<div class="method-breakdown-grid">`;

    // Header row
    html += `<div class="method-breakdown-header"></div>`; // empty corner
    metrics.forEach(metric => {
        html += `<div class="method-breakdown-header">${metric.label}</div>`;
    });

    // Data rows
    methods.forEach(method => {
        // Method label
        html += `<div class="method-breakdown-label">${methodLabels[method]}</div>`;

        // Histogram for each metric
        metrics.forEach(metric => {
            const id = `breakdown-${method}-${metric.key}`;
            html += `<div class="method-breakdown-cell"><div id="${id}"></div></div>`;
        });
    });

    html += `</div>`;

    container.innerHTML = html;

    // Render each histogram
    methods.forEach(method => {
        const methodResults = allResults.filter(r => r.method === method);

        metrics.forEach(metric => {
            const id = `breakdown-${method}-${metric.key}`;

            // Get values for this method/metric
            const values = methodResults
                .map(r => metric.getValue(r))
                .filter(v => v != null && !isNaN(v));

            const trace = {
                x: values,
                type: 'histogram',
                nbinsx: 15,
                marker: { color: getCssVar('--primary-color', '#a09f6c') },
                hovertemplate: `${metric.label}: %{x:.0f}%<br>Count: %{y}<extra></extra>`
            };

            Plotly.newPlot(id, [trace], window.getPlotlyLayout({
                margin: { l: 30, r: 5, t: 5, b: 25 },
                xaxis: {
                    tickfont: { size: 8 },
                    range: [0, 150]
                },
                yaxis: { tickfont: { size: 8 }, title: '' },
                height: 100
            }), { displayModeBar: false, responsive: true });
        });
    });
}


function renderBestVectorSimilarity(evalData) {
    const container = document.getElementById('best-vector-similarity-container');
    if (!container) return;

    const similarityMatrix = evalData.best_vector_similarity || {};
    const allTraits = Object.keys(similarityMatrix);

    if (allTraits.length === 0) {
        container.innerHTML = '<p>No similarity data available.</p>';
        return;
    }

    // Filter by selected traits from sidebar (fallback to all if none selected)
    const filteredTraits = window.getFilteredTraits();
    const selectedTraitNames = new Set(filteredTraits.map(t => t.name));
    const traits = selectedTraitNames.size > 0
        ? allTraits.filter(t => selectedTraitNames.has(t))
        : allTraits;

    if (traits.length === 0) {
        container.innerHTML = '<p>No matching traits in similarity data.</p>';
        return;
    }

    // Convert to 2D array (filtered)
    const matrix = traits.map(t1 =>
        traits.map(t2 => similarityMatrix[t1]?.[t2] ?? 0)
    );

    const displayNames = traits.map(t => window.getDisplayName(t));

    const trace = {
        z: matrix,
        x: displayNames,
        y: displayNames,
        type: 'heatmap',
        colorscale: window.ASYMB_COLORSCALE,
        zmid: 0,
        zmin: -1,
        zmax: 1,
        colorbar: {
            title: { text: 'Similarity', font: { size: 11 } },
            tickvals: [-1, -0.5, 0, 0.5, 1]
        },
        hovertemplate: '%{y} ↔ %{x}<br>sim = %{z:.3f}<extra></extra>',
        texttemplate: '%{z:.2f}',
        textfont: { size: 9 }
    };

    Plotly.newPlot(container, [trace], window.getPlotlyLayout({
        margin: { l: 150, r: 80, t: 100, b: 150 },
        xaxis: { side: 'top', tickangle: -45, tickfont: { size: 10 } },
        yaxis: { autorange: 'reversed', tickfont: { size: 10 } },
        height: 600
    }), { displayModeBar: false, responsive: true });
}


// CSS helper
function getCssVar(name, fallback = '') {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
}


/**
 * Render comprehensive metrics overview showing all available metrics
 */
function renderAllMetricsOverview(evalData) {
    const container = document.getElementById('all-metrics-container');
    if (!container) return;

    const results = evalData.all_results || [];

    // Define all metrics we want to show (matching extraction_evaluation.py)
    const ALL_METRICS = [
        // Core validation metrics
        { key: 'val_accuracy', label: 'Validation Accuracy', format: 'percent', good: '>90%' },
        { key: 'val_auc_roc', label: 'AUC-ROC', format: 'percent', good: '>90%' },
        { key: 'val_effect_size', label: 'Effect Size (d)', format: 'decimal2', good: '>1.5' },
        { key: 'val_separation', label: 'Separation', format: 'decimal2', good: '>0' },
        { key: 'val_p_value', label: 'P-Value', format: 'scientific', good: '<0.05' },

        // Distribution metrics
        { key: 'val_pos_mean', label: 'Pos Mean', format: 'decimal2', good: null },
        { key: 'val_neg_mean', label: 'Neg Mean', format: 'decimal2', good: null },
        { key: 'val_pos_std', label: 'Pos Std', format: 'decimal2', good: null },
        { key: 'val_neg_std', label: 'Neg Std', format: 'decimal2', good: null },

        // Training metrics (generalization)
        { key: 'train_accuracy', label: 'Train Accuracy', format: 'percent', good: null },
        { key: 'train_separation', label: 'Train Separation', format: 'decimal2', good: null },
        { key: 'accuracy_drop', label: 'Accuracy Drop', format: 'percent', good: '<5%' },
        { key: 'separation_ratio', label: 'Separation Ratio', format: 'decimal2', good: '>0.8' },

        // Vector properties
        { key: 'vector_norm', label: 'Vector Norm', format: 'decimal1', good: '15-40' },
        { key: 'vector_sparsity', label: 'Sparsity', format: 'percent', good: null },

        // Quality flags
        { key: 'polarity_correct', label: 'Polarity Correct', format: 'bool', good: 'true' },
    ];

    // Compute per-metric statistics across all results
    const metricStats = {};
    ALL_METRICS.forEach(metric => {
        const values = results
            .map(r => r[metric.key])
            .filter(v => v !== null && v !== undefined && !Number.isNaN(v));

        if (metric.format === 'bool') {
            const trueCount = values.filter(v => v === true).length;
            metricStats[metric.key] = {
                computed: values.length > 0,
                count: values.length,
                trueCount: trueCount,
                truePercent: values.length > 0 ? (trueCount / values.length * 100) : null
            };
        } else {
            const numValues = values.filter(v => typeof v === 'number');
            metricStats[metric.key] = {
                computed: numValues.length > 0,
                count: numValues.length,
                min: numValues.length > 0 ? Math.min(...numValues) : null,
                max: numValues.length > 0 ? Math.max(...numValues) : null,
                mean: numValues.length > 0 ? numValues.reduce((a, b) => a + b, 0) / numValues.length : null,
                std: numValues.length > 1 ? Math.sqrt(numValues.reduce((sum, v) => sum + Math.pow(v - metricStats[metric.key]?.mean || 0, 2), 0) / (numValues.length - 1)) : null
            };
            // Recompute std properly after mean is set
            if (numValues.length > 1) {
                const mean = metricStats[metric.key].mean;
                metricStats[metric.key].std = Math.sqrt(numValues.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / (numValues.length - 1));
            }
        }
    });

    // Format value for display
    const formatValue = (value, format) => {
        if (value === null || value === undefined) return '<span class="na">N/A</span>';
        switch (format) {
            case 'percent': return (value * 100).toFixed(1) + '%';
            case 'decimal1': return value.toFixed(1);
            case 'decimal2': return value.toFixed(2);
            case 'decimal3': return value.toFixed(3);
            case 'scientific': return value < 0.001 ? value.toExponential(2) : value.toFixed(4);
            case 'bool': return value ? '✓' : '✗';
            default: return String(value);
        }
    };

    // Build summary table
    let html = `
        <div class="metrics-overview">
            <h4>Metrics Summary (${results.length} vectors evaluated)</h4>
            <table class="data-table metrics-summary-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Status</th>
                        <th>Count</th>
                        <th>Min</th>
                        <th>Mean</th>
                        <th>Max</th>
                        <th>Good Value</th>
                    </tr>
                </thead>
                <tbody>
    `;

    ALL_METRICS.forEach(metric => {
        const stats = metricStats[metric.key];
        const computed = stats.computed;
        const statusClass = computed ? 'status-computed' : 'status-missing';
        const statusIcon = computed ? '✓' : '○';

        let minVal, meanVal, maxVal;
        if (metric.format === 'bool') {
            minVal = '-';
            meanVal = stats.truePercent !== null ? `${stats.truePercent.toFixed(0)}% true` : '<span class="na">N/A</span>';
            maxVal = '-';
        } else {
            minVal = formatValue(stats.min, metric.format);
            meanVal = formatValue(stats.mean, metric.format);
            maxVal = formatValue(stats.max, metric.format);
        }

        html += `
            <tr class="${statusClass}">
                <td><strong>${metric.label}</strong><br><code class="metric-key">${metric.key}</code></td>
                <td class="status-cell">${statusIcon}</td>
                <td>${stats.count || 0}</td>
                <td>${minVal}</td>
                <td>${meanVal}</td>
                <td>${maxVal}</td>
                <td class="good-value">${metric.good || '-'}</td>
            </tr>
        `;
    });

    html += `
                </tbody>
            </table>
        </div>
    `;

    container.innerHTML = html;
}


// Export
window.renderTraitExtraction = renderTraitExtraction;
