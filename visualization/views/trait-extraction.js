// Trait Extraction - Comprehensive view of extraction quality, methods, and vector properties

async function renderTraitExtraction() {
    const contentArea = document.getElementById('content-area');

    contentArea.innerHTML = '<div class="loading">Loading extraction evaluation data...</div>';

    // Load extraction evaluation data
    const evalData = await loadExtractionEvaluation();

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
                    <li><a href="#distributions">Metric Distributions</a></li>
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

                <h3 class="subsection-header" id="distributions">
                    <span class="subsection-num">3.</span>
                    <span class="subsection-title">Metric Distributions</span>
                    <span class="subsection-info-toggle" data-target="info-distributions">►</span>
                </h3>
                <div class="subsection-info" id="info-distributions">Distributions across all vectors. The ~50% accuracy peak is random_baseline (sanity check). Scatter plot: upper-right = good vectors, lower-right = suspicious (high acc, low effect).</div>
                <div id="metric-distributions-container"></div>

                <h3 class="subsection-header" id="method-breakdown">
                    <span class="subsection-num">4.</span>
                    <span class="subsection-title">Per-Method Distributions</span>
                    <span class="subsection-info-toggle" data-target="info-method-breakdown">►</span>
                </h3>
                <div class="subsection-info" id="info-method-breakdown">Each row shows one extraction method. Histograms show distribution of Score, Accuracy, Effect (normalized), and 1−Drop across all layer×trait combinations. Compare methods to see which produces consistently good vectors.</div>
                <div id="method-breakdown-container"></div>

                <h3 class="subsection-header" id="trait-breakdown">
                    <span class="subsection-num">5.</span>
                    <span class="subsection-title">Per-Trait Distributions</span>
                    <span class="subsection-info-toggle" data-target="info-trait-breakdown">►</span>
                </h3>
                <div class="subsection-info" id="info-trait-breakdown">Each row shows one trait. Histograms show distribution across all method×layer combinations. See which traits are easy vs hard to extract.</div>
                <div id="trait-breakdown-container"></div>

                <h3 class="subsection-header" id="layer-breakdown">
                    <span class="subsection-num">6.</span>
                    <span class="subsection-title">Per-Layer Distributions</span>
                    <span class="subsection-info-toggle" data-target="info-layer-breakdown">►</span>
                </h3>
                <div class="subsection-info" id="info-layer-breakdown">Each row shows one layer. Histograms show distribution across all method×trait combinations. See which layers produce better vectors.</div>
                <div id="layer-breakdown-container"></div>

                <h3 class="subsection-header" id="layer-trait-heatmaps">
                    <span class="subsection-num">7.</span>
                    <span class="subsection-title">Layer × Trait Heatmaps (per method)</span>
                    <span class="subsection-info-toggle" data-target="info-layer-trait">►</span>
                </h3>
                <div class="subsection-info" id="info-layer-trait">One heatmap per method showing score across all layers (y-axis) and traits (x-axis). Find the best layer for each trait.</div>
                <div id="layer-trait-heatmaps-container"></div>
            </section>

            <!-- Section 2: Notation -->
            <section>
                <h2>Notation & Definitions <span class="info-icon" data-info="notation">ⓘ</span></h2>
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
                <h2>Quality Metrics <span class="info-icon" data-info="metrics">ⓘ</span></h2>
                ${renderMetricsDefinitions()}
            </section>

            <!-- Section 5: Scoring Method -->
            <section>
                <h2>Scoring & Ranking <span class="info-icon" data-info="scoring">ⓘ</span></h2>
                ${renderScoringExplanation()}
            </section>

            <!-- Section 6: All Metrics Overview -->
            <section>
                <h2>All Metrics Overview</h2>
                <div id="all-metrics-container"></div>
            </section>

            <!-- Tooltip container -->
            <div id="section-info-tooltip" class="tooltip"></div>
        </div>
    `;

    // Render each visualization
    renderQualityTable(evalData);
    renderTraitHeatmaps(evalData);
    renderBestVectors(evalData);
    renderMetricDistributions(evalData);
    renderMethodBreakdown(evalData);
    renderTraitBreakdown(evalData);
    renderLayerBreakdown(evalData);
    renderLayerTraitHeatmaps(evalData);
    renderBestVectorSimilarity(evalData);
    renderAllMetricsOverview(evalData);

    // Render math after all content is in DOM
    if (window.MathJax) {
        MathJax.typesetPromise();
    }

    // Setup info tooltips
    setupSectionInfoTooltips();
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


// Info tooltip content for each section
const SECTION_INFO_CONTENT = {
    notation: `
        <h4>About This Notation</h4>
        <p>These symbols are used consistently throughout the extraction pipeline and evaluation metrics.</p>
        <p><strong>Key insight:</strong> Each example's activation is the <em>average</em> across all response tokens, giving a single d-dimensional vector per example.</p>
    `,
    metrics: `
        <h4>Interpreting Quality Metrics</h4>
        <p>All metrics are computed on <strong>held-out validation data</strong> (20% of examples) to measure generalization.</p>
        <ul>
            <li><strong>Accuracy:</strong> Can the vector classify unseen examples? >90% is good.</li>
            <li><strong>Effect Size (d):</strong> How separated are the distributions? >1.5 is large effect.</li>
            <li><strong>Norm:</strong> Vector magnitude. Typical range: 15-40.</li>
            <li><strong>Margin:</strong> Gap between distributions. Positive = no overlap.</li>
        </ul>
    `,
    scoring: `
        <h4>Why This Scoring Formula?</h4>
        <p>The combined score balances two goals:</p>
        <ul>
            <li><strong>Accuracy (50%):</strong> Practical utility—can we use this vector?</li>
            <li><strong>Effect Size (50%):</strong> Robustness—how confident is the separation?</li>
        </ul>
        <p>Effect size is normalized per-trait because scales vary (0.5–5.0). This makes cross-trait comparison fair.</p>
        <p><em>A vector with 95% accuracy but tiny effect size may be overfitting. This score catches that.</em></p>
    `
};


function setupSectionInfoTooltips() {
    const tooltip = document.getElementById('section-info-tooltip');
    if (!tooltip) return;

    const icons = document.querySelectorAll('.info-icon');

    icons.forEach(icon => {
        icon.addEventListener('click', (e) => {
            e.stopPropagation();
            const key = icon.dataset.info;
            const content = SECTION_INFO_CONTENT[key];

            if (tooltip.classList.contains('show') && tooltip.dataset.activeKey === key) {
                // Toggle off if clicking same icon
                tooltip.classList.remove('show');
                return;
            }

            tooltip.innerHTML = content;
            tooltip.dataset.activeKey = key;

            // Position near the icon
            const rect = icon.getBoundingClientRect();
            const containerRect = document.querySelector('.tool-view').getBoundingClientRect();

            tooltip.style.top = `${rect.bottom - containerRect.top + 8}px`;
            tooltip.style.left = `${rect.left - containerRect.left}px`;

            tooltip.classList.add('show');
        });
    });

    // Close tooltip when clicking elsewhere
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.info-icon') && !e.target.closest('.tooltip')) {
            tooltip.classList.remove('show');
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
        <div class="grid">
            <div class="card">
                <h4>Input Shapes</h4>
                <table class="def-table">
                    <tr><td>$$n$$</td><td>Number of examples (train or validation split)</td></tr>
                    <tr><td>$$d$$</td><td>Hidden dimension (2304 for Gemma 2B)</td></tr>
                    <tr><td>$$L$$</td><td>Number of layers (26 for Gemma 2B)</td></tr>
                    <tr><td>$$\\mathbf{A} \\in \\mathbb{R}^{n \\times d}$$</td><td>Activation matrix (token-averaged per example)</td></tr>
                </table>
            </div>

            <div class="card">
                <h4>Variables</h4>
                <table class="def-table">
                    <tr><td>$$\\vec{v} \\in \\mathbb{R}^d$$</td><td>Trait vector (direction in activation space)</td></tr>
                    <tr><td>$$\\vec{a}_i \\in \\mathbb{R}^d$$</td><td>Single example's activation (row of A)</td></tr>
                    <tr><td>$$y_i \\in \\{+1, -1\\}$$</td><td>Binary label (positive/negative trait)</td></tr>
                    <tr><td>$$\\text{pos}, \\text{neg}$$</td><td>Subscripts for positive/negative example subsets</td></tr>
                </table>
            </div>

            <div class="card">
                <h4>Key Quantities</h4>
                <table class="def-table">
                    <tr><td>$$\\vec{a} \\cdot \\vec{v}$$</td><td>Projection score (dot product)</td></tr>
                    <tr><td>$$\\mu_{\\text{pos}}, \\mu_{\\text{neg}}$$</td><td>Mean projection for pos/neg examples</td></tr>
                    <tr><td>$$\\sigma_{\\text{pooled}}$$</td><td>Pooled standard deviation</td></tr>
                    <tr><td>$$||\\vec{v}||_2$$</td><td>L2 norm (vector magnitude)</td></tr>
                </table>
            </div>

            <div class="card">
                <h4>Pipeline Context</h4>
                <table class="def-table">
                    <tr><td><strong>Train split</strong></td><td>80% of examples → used to extract vectors</td></tr>
                    <tr><td><strong>Val split</strong></td><td>20% of examples → used to evaluate vectors</td></tr>
                    <tr><td><strong>Per-layer</strong></td><td>Vectors extracted independently for each layer</td></tr>
                    <tr><td><strong>Per-method</strong></td><td>4 extraction methods × 26 layers = 104 vectors/trait</td></tr>
                </table>
            </div>
        </div>
    `;
}


function renderExtractionTechniques() {
    return `
        <div class="grid">
            <div class="card">
                <h4>Mean Difference</h4>
                <p>$$\\vec{v} = \\text{mean}(\\mathbf{A}_{\\text{pos}}) - \\text{mean}(\\mathbf{A}_{\\text{neg}})$$</p>
                <p>Direction between cluster centroids. Fast baseline, but ignores class shape/spread.</p>
            </div>

            <div class="card">
                <h4>Linear Probe</h4>
                <p>$$\\min_\\vec{w} \\sum_i \\log(1 + e^{-y_i (\\vec{w} \\cdot \\vec{a}_i)})$$</p>
                <p>Logistic regression weights. Optimizes for <em>separability</em>, not just distance—handles overlap better.</p>
            </div>

            <div class="card">
                <h4>Gradient</h4>
                <p>$$\\max_\\vec{v} \\left( \\text{mean}(\\mathbf{A}_{\\text{pos}} \\cdot \\vec{v}) - \\text{mean}(\\mathbf{A}_{\\text{neg}} \\cdot \\vec{v}) \\right)$$</p>
                <p>Direct optimization of separation. Best for low-separability traits where other methods fail.</p>
            </div>

            <div class="card">
                <h4>Random Baseline</h4>
                <p>$$\\vec{v} \\sim \\mathcal{N}(0, I), \\quad \\|\\vec{v}\\| = 1$$</p>
                <p>Random unit vector. Sanity check—should get ~50% accuracy. If not, something's wrong.</p>
            </div>
        </div>
    `;
}


function renderMetricsDefinitions() {
    return `
        <div class="grid">
            <div class="card">
                <h4>Accuracy</h4>
                <p>$$\\text{acc} = \\frac{\\text{correct classifications}}{\\text{total examples}}$$</p>
                <p>Percentage of validation examples correctly classified as positive/negative.</p>
                <p><strong>Range:</strong> 0-1 (50% = random, 100% = perfect). <strong class="quality-good">Good: &gt; 0.90</strong></p>
            </div>

            <div class="card">
                <h4>AUC-ROC</h4>
                <p>$$\\text{AUC} = \\int_0^1 \\text{TPR}(\\text{FPR}^{-1}(t)) \\, dt$$</p>
                <p>Area Under ROC Curve. Threshold-independent measure of classification quality.</p>
                <p><strong>Range:</strong> 0.5-1 (0.5 = random, 1 = perfect). <strong class="quality-good">Good: &gt; 0.90</strong></p>
            </div>

            <div class="card">
                <h4>Effect Size (Cohen's d)</h4>
                <p>$$d = \\frac{\\mu_{\\text{pos}} - \\mu_{\\text{neg}}}{\\sigma_{\\text{pooled}}}$$</p>
                <p>Magnitude of separation between positive/negative distributions in standard deviation units.</p>
                <p><strong>Range:</strong> 0-∞ (0 = no separation, &gt;2 = large effect). <strong class="quality-good">Good: &gt; 1.5</strong></p>
            </div>

            <div class="card">
                <h4>Vector Norm</h4>
                <p>$$||\\vec{v}||_2 = \\sqrt{\\sum_i v_i^2}$$</p>
                <p>L2 norm of the vector. Indicates magnitude/strength.</p>
                <p><strong>Range:</strong> 0-∞. <strong>Typical:</strong> 15-40 for normalized vectors</p>
            </div>

            <div class="card">
                <h4>Separation Margin</h4>
                <p>$$(\\mu_{\\text{pos}} - \\sigma_{\\text{pos}}) - (\\mu_{\\text{neg}} + \\sigma_{\\text{neg}})$$</p>
                <p>Gap between distributions. Positive = good separation, negative = overlap.</p>
                <p><strong>Range:</strong> -∞ to +∞. <strong class="quality-good">Good: &gt; 0</strong></p>
            </div>

            <div class="card">
                <h4>Sparsity</h4>
                <p>$$\\text{sparsity} = \\frac{|\\{i : |v_i| < 0.01\\}|}{d}$$</p>
                <p>Percentage of near-zero components. High sparsity = interpretable, focused vector.</p>
                <p><strong>Range:</strong> 0-1 (0 = dense, 1 = sparse)</p>
            </div>

            <div class="card">
                <h4>Overlap Coefficient</h4>
                <p>$$\\text{overlap} \\approx 1 - \\frac{|\\mu_{\\text{pos}} - \\mu_{\\text{neg}}|}{4\\sigma_{\\text{pooled}}}$$</p>
                <p>Estimate of distribution overlap (0 = no overlap, 1 = complete overlap).</p>
                <p><strong>Range:</strong> 0-1. <strong class="quality-good">Good: &lt; 0.2</strong></p>
            </div>
        </div>
    `;
}


function renderScoringExplanation() {
    return `
        <div class="card">
            <h4>Combined Score Formula</h4>
            <p>$$\\text{score} = \\frac{\\text{accuracy} + \\text{norm\\_effect} + (1 - \\text{accuracy\\_drop})}{3} \\times \\text{polarity}$$</p>

            <h4>Components (equal weights)</h4>
            <ul>
                <li><strong>Accuracy:</strong> Classification performance on held-out validation examples.</li>
                <li><strong>Normalized Effect Size:</strong> Cohen's d / max Cohen's d for trait. Measures separation quality for steering.</li>
                <li><strong>1 − Accuracy Drop:</strong> Generalization quality. High = validates well, low = overfitting.</li>
                <li><strong>Polarity:</strong> Hard multiplier. Wrong polarity (neg > pos) → score = 0.</li>
            </ul>

            <h4>Why Equal Weights?</h4>
            <p>No strong prior for weighting. Equal weights until empirical evidence suggests otherwise. All three components matter: accuracy (does it work?), effect size (can it steer?), generalization (will it transfer?).</p>
        </div>
    `;
}


function renderQualityTable(evalData) {
    const container = document.getElementById('quality-table-container');
    if (!container) return;

    const results = evalData.all_results || [];
    if (results.length === 0) {
        container.innerHTML = '<p>No results to display.</p>';
        return;
    }

    // Use combined_score from backend if available, otherwise compute it
    const augmentedResults = results.map(r => {
        if (r.combined_score != null) {
            return r;
        }
        // Fallback: compute with new formula
        const maxEffectPerTrait = {};
        results.forEach(r2 => {
            if (!maxEffectPerTrait[r2.trait] || r2.val_effect_size > maxEffectPerTrait[r2.trait]) {
                maxEffectPerTrait[r2.trait] = r2.val_effect_size || 0;
            }
        });
        const max_d = maxEffectPerTrait[r.trait] || 1;
        const normEffect = (r.val_effect_size || 0) / max_d;
        const accDrop = r.accuracy_drop || 0;
        const polarity = r.polarity_correct ? 1 : 0;
        const score = ((r.val_accuracy || 0) + normEffect + (1 - accDrop)) / 3 * polarity;
        return { ...r, combined_score: score };
    });

    // Build table HTML
    const tableHTML = `
        <div class="scrollable-container-lg">
            <table class="data-table" id="extraction-quality-table">
                <thead>
                    <tr>
                        <th class="sortable" data-column="trait">Trait<span class="sort-indicator">↕</span></th>
                        <th class="sortable" data-column="method">Method<span class="sort-indicator">↕</span></th>
                        <th class="sortable" data-column="layer">Layer<span class="sort-indicator">↕</span></th>
                        <th class="sortable sort-active" data-column="combined_score">Score<span class="sort-indicator">↓</span></th>
                        <th class="sortable" data-column="val_accuracy">Accuracy<span class="sort-indicator">↕</span></th>
                        <th class="sortable" data-column="val_auc_roc">AUC<span class="sort-indicator">↕</span></th>
                        <th class="sortable" data-column="val_effect_size">Effect Size (d)<span class="sort-indicator">↕</span></th>
                        <th class="sortable" data-column="vector_norm">Norm<span class="sort-indicator">↕</span></th>
                    </tr>
                </thead>
                <tbody>
                    ${augmentedResults
                        .sort((a, b) => b.combined_score - a.combined_score)
                        .map(r => {
                            const accClass = (r.val_accuracy >= 0.9) ? 'quality-good' : (r.val_accuracy >= 0.75) ? 'quality-ok' : 'quality-bad';
                            const aucClass = (r.val_auc_roc >= 0.9) ? 'quality-good' : (r.val_auc_roc >= 0.75) ? 'quality-ok' : 'quality-bad';
                            return `
                                <tr>
                                    <td>${window.getDisplayName(r.trait)}</td>
                                    <td>${r.method}</td>
                                    <td>${r.layer}</td>
                                    <td><strong>${r.combined_score.toFixed(3)}</strong></td>
                                    <td class="${accClass}">${(r.val_accuracy * 100).toFixed(1)}%</td>
                                    <td class="${aucClass}">${(r.val_auc_roc * 100).toFixed(1)}%</td>
                                    <td>${r.val_effect_size?.toFixed(2) ?? 'N/A'}</td>
                                    <td>${r.vector_norm?.toFixed(1) ?? 'N/A'}</td>
                                </tr>
                            `;
                        }).join('')}
                </tbody>
            </table>
        </div>
    `;

    container.innerHTML = tableHTML;

    // Add sort functionality
    container.querySelectorAll('.sortable').forEach(th => {
        th.addEventListener('click', () => {
            const column = th.dataset.column;
            const direction = sortDirection[column] === 'asc' ? 'desc' : 'asc';
            sortDirection[column] = direction;

            // Update visual indicators
            container.querySelectorAll('.sortable').forEach(header => {
                header.classList.remove('sort-active');
                header.querySelector('.sort-indicator').textContent = '↕';
            });
            th.classList.add('sort-active');
            th.querySelector('.sort-indicator').textContent = direction === 'asc' ? '↑' : '↓';

            sortQualityTable(augmentedResults, column, container);
        });
    });
}


let sortDirection = { combined_score: 'desc' };
function sortQualityTable(results, column, container) {
    const direction = sortDirection[column];

    const sorted = [...results].sort((a, b) => {
        let valA = a[column];
        let valB = b[column];

        if (column === 'trait' || column === 'method') {
            valA = valA || '';
            valB = valB || '';
            return direction === 'asc' ? valA.localeCompare(valB) : valB.localeCompare(valA);
        } else {
            valA = valA || 0;
            valB = valB || 0;
            return direction === 'asc' ? valA - valB : valB - valA;
        }
    });

    // Re-render tbody
    const tbody = container.querySelector('tbody');
    tbody.innerHTML = sorted.map(r => {
        const accClass = (r.val_accuracy >= 0.9) ? 'quality-good' : (r.val_accuracy >= 0.75) ? 'quality-ok' : 'quality-bad';
        const aucClass = (r.val_auc_roc >= 0.9) ? 'quality-good' : (r.val_auc_roc >= 0.75) ? 'quality-ok' : 'quality-bad';
        return `
            <tr>
                <td>${window.getDisplayName(r.trait)}</td>
                <td>${r.method}</td>
                <td>${r.layer}</td>
                <td><strong>${r.combined_score.toFixed(3)}</strong></td>
                <td class="${accClass}">${(r.val_accuracy * 100).toFixed(1)}%</td>
                <td class="${aucClass}">${(r.val_auc_roc * 100).toFixed(1)}%</td>
                <td>${r.val_effect_size?.toFixed(2) ?? 'N/A'}</td>
                <td>${r.vector_norm?.toFixed(1) ?? 'N/A'}</td>
            </tr>
        `;
    }).join('');
}


function renderTraitHeatmaps(evalData) {
    const container = document.getElementById('trait-heatmaps-container');
    if (!container) return;

    const results = evalData.all_results || [];
    if (results.length === 0) {
        container.innerHTML = '<p>No results to display.</p>';
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

    // Create grid with legend below
    container.innerHTML = `
        <div class="trait-heatmaps-grid" id="heatmaps-grid"></div>
        <div class="heatmap-legend-footer">
            <span class="file-hint">${traits.length} traits</span>
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

        const traitDiv = document.createElement('div');
        traitDiv.className = 'trait-heatmap-item';
        traitDiv.innerHTML = `
            <h4 title="${displayName}">${displayName}</h4>
            <div id="heatmap-${traitId}" class="chart-container-sm"></div>
        `;

        grid.appendChild(traitDiv);

        renderSingleTraitHeatmap(traitResults, `heatmap-${traitId}`, computeScore, true);
    });
}


function renderSingleTraitHeatmap(traitResults, containerId, computeScore, compact = false) {
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

    const trace = {
        z: matrix,
        x: compact ? ['MD', 'Pr', 'Gr'] : methods,
        y: layers,
        type: 'heatmap',
        colorscale: window.ASYMB_COLORSCALE,
        hovertemplate: '%{x} L%{y}: %{z:.1f}%<extra></extra>',
        zmin: 0,
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
        margin: { l: 25, r: 5, t: 5, b: 25 },
        xaxis: { tickfont: { size: 8 }, tickangle: 0 },
        yaxis: { tickfont: { size: 8 }, title: '' },
        height: 120
    } : {
        margin: { l: 40, r: 80, t: 20, b: 60 },
        xaxis: { title: 'Method', tickfont: { size: 11 } },
        yaxis: { title: 'Layer', tickfont: { size: 10 } },
        height: 400
    };

    Plotly.newPlot(containerId, [trace], window.getPlotlyLayout(layout), { displayModeBar: false, responsive: true });
}


function renderBestVectors(evalData) {
    const container = document.getElementById('best-vectors-container');
    if (!container) return;

    const bestPerTrait = evalData.best_per_trait || [];
    if (bestPerTrait.length === 0) {
        container.innerHTML = '<p>No best vectors found.</p>';
        return;
    }

    const tableHTML = `
        <table class="data-table">
            <thead>
                <tr>
                    <th>Trait</th>
                    <th>Best Method</th>
                    <th>Best Layer</th>
                    <th>Accuracy</th>
                    <th>Effect Size (d)</th>
                    <th>Norm</th>
                </tr>
            </thead>
            <tbody>
                ${bestPerTrait.map(r => `
                    <tr>
                        <td><strong>${window.getDisplayName(r.trait)}</strong></td>
                        <td>${r.method}</td>
                        <td>${r.layer}</td>
                        <td>${(r.val_accuracy * 100).toFixed(1)}%</td>
                        <td>${r.val_effect_size?.toFixed(2) ?? 'N/A'}</td>
                        <td>${r.vector_norm?.toFixed(1) ?? 'N/A'}</td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;

    container.innerHTML = tableHTML;
}


function renderMetricDistributions(evalData) {
    const container = document.getElementById('metric-distributions-container');
    if (!container) return;

    const allResults = evalData.all_results || [];
    if (allResults.length === 0) {
        container.innerHTML = '<p>No results available.</p>';
        return;
    }

    // Method colors for scatter plot
    const methodColors = {
        'mean_diff': '#4e79a7',
        'probe': '#f28e2b',
        'gradient': '#76b7b2',
        'random_baseline': '#b07aa1'
    };

    // Create grid layout
    container.innerHTML = `
        <div class="metric-distributions-grid">
            <div class="distribution-item">
                <h4>Validation Accuracy</h4>
                <div id="hist-val-accuracy" class="chart-container-sm"></div>
            </div>
            <div class="distribution-item">
                <h4>Effect Size (Cohen's d)</h4>
                <div id="hist-effect-size" class="chart-container-sm"></div>
            </div>
            <div class="distribution-item">
                <h4>Accuracy Drop (Overfitting)</h4>
                <div id="hist-accuracy-drop" class="chart-container-sm"></div>
            </div>
            <div class="distribution-item distribution-item-wide distribution-item-spaced">
                <h4>Accuracy vs Effect Size <span class="chart-subtitle">(upper-right = good, lower-right = suspicious)</span></h4>
                <div id="scatter-acc-effect" class="chart-container-md"></div>
            </div>
            <div class="distribution-item distribution-item-wide distribution-item-spaced">
                <h4>
                    Component Heatmaps <span class="chart-subtitle">(layer × method for selected trait)</span>
                    <span class="subsection-info-toggle" data-target="info-component-heatmaps">►</span>
                </h4>
                <div class="subsection-info" id="info-component-heatmaps">
                    <p><strong>Score</strong> = (Accuracy + NormEffect + (1−Drop)) / 3 × Polarity</p>
                    <p><strong>Accuracy</strong>: Classification accuracy on held-out examples (threshold = midpoint between class means)</p>
                    <p><strong>Effect (normalized)</strong>: Cohen's d / max Cohen's d for this trait. High = clean separation, good for steering.</p>
                    <p><strong>1−Drop</strong>: 1 − (train_acc − val_acc). High = generalizes well, low = overfitting.</p>
                </div>
                <div class="trait-selector-row">
                    <label>Trait: </label>
                    <select id="score-trait-selector"></select>
                </div>
                <div class="component-heatmaps-grid">
                    <div class="component-heatmap-item">
                        <div class="component-heatmap-label">Combined Score</div>
                        <div id="heatmap-score"></div>
                    </div>
                    <div class="component-heatmap-item">
                        <div class="component-heatmap-label">Validation Accuracy</div>
                        <div id="heatmap-accuracy"></div>
                    </div>
                    <div class="component-heatmap-item">
                        <div class="component-heatmap-label">Effect Size (normalized)</div>
                        <div id="heatmap-effect"></div>
                    </div>
                    <div class="component-heatmap-item">
                        <div class="component-heatmap-label">1 − Accuracy Drop</div>
                        <div id="heatmap-drop"></div>
                    </div>
                </div>
            </div>
        </div>
    `;

    // 1. Validation Accuracy Histogram
    const accValues = allResults.map(r => r.val_accuracy * 100).filter(v => v != null && !isNaN(v));
    Plotly.newPlot('hist-val-accuracy', [{
        x: accValues,
        type: 'histogram',
        nbinsx: 20,
        marker: { color: getCssVar('--primary-color', '#a09f6c') },
        hovertemplate: '%{x:.0f}%: %{y} vectors<extra></extra>'
    }], window.getPlotlyLayout({
        margin: { l: 45, r: 10, t: 10, b: 40 },
        xaxis: { title: 'Accuracy (%)', range: [40, 105] },
        yaxis: { title: 'Count' },
        height: 180
    }), { displayModeBar: false, responsive: true });

    // 2. Effect Size Histogram
    const effectValues = allResults.map(r => r.val_effect_size).filter(v => v != null && !isNaN(v));
    Plotly.newPlot('hist-effect-size', [{
        x: effectValues,
        type: 'histogram',
        nbinsx: 20,
        marker: { color: getCssVar('--primary-color', '#a09f6c') },
        hovertemplate: 'd=%{x:.1f}: %{y} vectors<extra></extra>'
    }], window.getPlotlyLayout({
        margin: { l: 45, r: 10, t: 10, b: 40 },
        xaxis: { title: "Cohen's d" },
        yaxis: { title: 'Count' },
        height: 180
    }), { displayModeBar: false, responsive: true });

    // 3. Accuracy Drop Histogram
    const dropValues = allResults.map(r => r.accuracy_drop * 100).filter(v => v != null && !isNaN(v));
    Plotly.newPlot('hist-accuracy-drop', [{
        x: dropValues,
        type: 'histogram',
        nbinsx: 20,
        marker: { color: getCssVar('--primary-color', '#a09f6c') },
        hovertemplate: '%{x:.1f}% drop: %{y} vectors<extra></extra>'
    }], window.getPlotlyLayout({
        margin: { l: 45, r: 10, t: 10, b: 40 },
        xaxis: { title: 'Accuracy Drop (%)' },
        yaxis: { title: 'Count' },
        height: 180
    }), { displayModeBar: false, responsive: true });

    // 4. Scatter: Accuracy vs Effect Size (colored by method)
    const methods = [...new Set(allResults.map(r => r.method))];
    const scatterTraces = methods.map(method => {
        const methodResults = allResults.filter(r => r.method === method);
        return {
            x: methodResults.map(r => r.val_accuracy * 100),
            y: methodResults.map(r => r.val_effect_size),
            mode: 'markers',
            type: 'scatter',
            name: method,
            marker: {
                color: methodColors[method] || '#888',
                size: 6,
                opacity: 0.7
            },
            text: methodResults.map(r => `${window.getDisplayName(r.trait)}<br>Layer ${r.layer}`),
            hovertemplate: '%{text}<br>Acc: %{x:.1f}%<br>d: %{y:.2f}<extra>%{fullData.name}</extra>'
        };
    });

    Plotly.newPlot('scatter-acc-effect', scatterTraces, window.getPlotlyLayout({
        margin: { l: 50, r: 10, t: 10, b: 45 },
        xaxis: { title: 'Validation Accuracy (%)', range: [40, 105] },
        yaxis: { title: "Effect Size (Cohen's d)" },
        height: 280,
        legend: { orientation: 'h', y: -0.25, x: 0.5, xanchor: 'center' },
        showlegend: true
    }), { displayModeBar: false, responsive: true });

    // 5. Score by Method (per-trait, per-layer)
    const traits = [...new Set(allResults.map(r => r.trait))].sort();
    const traitSelector = document.getElementById('score-trait-selector');

    // Populate dropdown
    traits.forEach(trait => {
        const option = document.createElement('option');
        option.value = trait;
        option.textContent = window.getDisplayName(trait);
        traitSelector.appendChild(option);
    });

    // Compute score for all results
    const maxEffectPerTrait = {};
    allResults.forEach(r => {
        if (!maxEffectPerTrait[r.trait] || r.val_effect_size > maxEffectPerTrait[r.trait]) {
            maxEffectPerTrait[r.trait] = r.val_effect_size || 0;
        }
    });

    const computeScore = (r) => {
        const maxEffect = maxEffectPerTrait[r.trait] || 1;
        const normEffect = (r.val_effect_size || 0) / maxEffect;
        const accDrop = r.accuracy_drop || 0;
        const polarity = r.polarity_correct ? 1 : 0;
        // Equal weights: accuracy + normalized effect + (1 - overfitting)
        const baseScore = ((r.val_accuracy || 0) + normEffect + (1 - accDrop)) / 3;
        return baseScore * polarity;
    };

    const renderComponentHeatmaps = (selectedTrait) => {
        const traitResults = allResults.filter(r => r.trait === selectedTrait);
        const methodOrder = ['mean_diff', 'probe', 'gradient'];
        const methodLabels = ['MD', 'Pr', 'Gr'];
        const layers = [...new Set(traitResults.map(r => r.layer))].sort((a, b) => a - b);

        // Get max effect for normalization
        const maxEffect = maxEffectPerTrait[selectedTrait] || 1;

        // Build matrices for each component
        const buildMatrix = (getValue) => {
            return layers.map(layer => {
                return methodOrder.map(method => {
                    const result = traitResults.find(r => r.layer === layer && r.method === method);
                    return result ? getValue(result) : null;
                });
            });
        };

        const matrices = {
            score: buildMatrix(r => computeScore(r) * 100),
            accuracy: buildMatrix(r => (r.val_accuracy || 0) * 100),
            effect: buildMatrix(r => ((r.val_effect_size || 0) / maxEffect) * 100),
            drop: buildMatrix(r => (1 - (r.accuracy_drop || 0)) * 100)
        };

        const configs = [
            { id: 'heatmap-score', matrix: matrices.score, label: 'Score' },
            { id: 'heatmap-accuracy', matrix: matrices.accuracy, label: 'Acc %' },
            { id: 'heatmap-effect', matrix: matrices.effect, label: 'Effect' },
            { id: 'heatmap-drop', matrix: matrices.drop, label: '1−Drop' }
        ];

        configs.forEach(cfg => {
            const trace = {
                z: cfg.matrix,
                x: methodLabels,
                y: layers,
                type: 'heatmap',
                colorscale: window.ASYMB_COLORSCALE,
                zmin: cfg.id === 'heatmap-drop' ? 50 : 0,
                zmax: 100,
                showscale: false,
                hovertemplate: `%{x} L%{y}<br>${cfg.label}: %{z:.1f}<extra></extra>`
            };

            Plotly.newPlot(cfg.id, [trace], window.getPlotlyLayout({
                margin: { l: 25, r: 5, t: 5, b: 25 },
                xaxis: { tickfont: { size: 8 }, tickangle: 0 },
                yaxis: { tickfont: { size: 8 }, dtick: 5 },
                height: 200
            }), { displayModeBar: false, responsive: true });
        });
    };

    // Initial render
    renderComponentHeatmaps(traits[0]);

    // Update on change
    traitSelector.addEventListener('change', (e) => {
        renderComponentHeatmaps(e.target.value);
    });

    // Re-setup toggles for dynamically added content
    setupSubsectionInfoToggles();
}


/**
 * Render per-method breakdown: 6 methods × 4 metrics = 24 histograms
 * Each histogram shows distribution of a metric for one method
 */
function renderMethodBreakdown(evalData) {
    const container = document.getElementById('method-breakdown-container');
    if (!container) return;

    const allResults = evalData.all_results || [];
    if (allResults.length === 0) {
        container.innerHTML = '<p>No results available.</p>';
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


/**
 * Render per-trait breakdown: histograms for each trait across all methods×layers
 */
function renderTraitBreakdown(evalData) {
    const container = document.getElementById('trait-breakdown-container');
    if (!container) return;

    const allResults = evalData.all_results || [];
    if (allResults.length === 0) {
        container.innerHTML = '<p>No results available.</p>';
        return;
    }

    const traits = [...new Set(allResults.map(r => r.trait))].sort();

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

    // Metric definitions
    const metrics = [
        { key: 'score', label: 'Score', getValue: r => computeScore(r) * 100 },
        { key: 'accuracy', label: 'Accuracy', getValue: r => (r.val_accuracy || 0) * 100 },
        { key: 'effect', label: 'Effect (norm)', getValue: r => ((r.val_effect_size || 0) / (maxEffectPerTrait[r.trait] || 1)) * 100 },
        { key: 'drop', label: '1−Drop', getValue: r => (1 - (r.accuracy_drop || 0)) * 100 }
    ];

    // Build HTML grid: rows = traits, cols = metrics
    let html = `<div class="method-breakdown-grid">`;

    // Header row
    html += `<div class="method-breakdown-header"></div>`;
    metrics.forEach(metric => {
        html += `<div class="method-breakdown-header">${metric.label}</div>`;
    });

    // Data rows
    traits.forEach(trait => {
        const displayName = window.getDisplayName(trait);
        html += `<div class="method-breakdown-label" title="${displayName}">${displayName}</div>`;

        metrics.forEach(metric => {
            const id = `trait-breakdown-${trait.replace(/\//g, '-')}-${metric.key}`;
            html += `<div class="method-breakdown-cell"><div id="${id}"></div></div>`;
        });
    });

    html += `</div>`;
    container.innerHTML = html;

    // Render each histogram
    traits.forEach(trait => {
        const traitResults = allResults.filter(r => r.trait === trait);

        metrics.forEach(metric => {
            const id = `trait-breakdown-${trait.replace(/\//g, '-')}-${metric.key}`;

            const values = traitResults
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
                xaxis: { tickfont: { size: 8 }, range: [0, 150] },
                yaxis: { tickfont: { size: 8 }, title: '' },
                height: 80
            }), { displayModeBar: false, responsive: true });
        });
    });
}


/**
 * Render per-layer breakdown: histograms for each layer across all methods×traits
 */
function renderLayerBreakdown(evalData) {
    const container = document.getElementById('layer-breakdown-container');
    if (!container) return;

    const allResults = evalData.all_results || [];
    if (allResults.length === 0) {
        container.innerHTML = '<p>No results available.</p>';
        return;
    }

    const layers = [...new Set(allResults.map(r => r.layer))].sort((a, b) => a - b);

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

    // Metric definitions
    const metrics = [
        { key: 'score', label: 'Score', getValue: r => computeScore(r) * 100 },
        { key: 'accuracy', label: 'Accuracy', getValue: r => (r.val_accuracy || 0) * 100 },
        { key: 'effect', label: 'Effect (norm)', getValue: r => ((r.val_effect_size || 0) / (maxEffectPerTrait[r.trait] || 1)) * 100 },
        { key: 'drop', label: '1−Drop', getValue: r => (1 - (r.accuracy_drop || 0)) * 100 }
    ];

    // Build HTML grid: rows = layers, cols = metrics
    let html = `<div class="method-breakdown-grid">`;

    // Header row
    html += `<div class="method-breakdown-header"></div>`;
    metrics.forEach(metric => {
        html += `<div class="method-breakdown-header">${metric.label}</div>`;
    });

    // Data rows
    layers.forEach(layer => {
        html += `<div class="method-breakdown-label">Layer ${layer}</div>`;

        metrics.forEach(metric => {
            const id = `layer-breakdown-${layer}-${metric.key}`;
            html += `<div class="method-breakdown-cell"><div id="${id}"></div></div>`;
        });
    });

    html += `</div>`;
    container.innerHTML = html;

    // Render each histogram
    layers.forEach(layer => {
        const layerResults = allResults.filter(r => r.layer === layer);

        metrics.forEach(metric => {
            const id = `layer-breakdown-${layer}-${metric.key}`;

            const values = layerResults
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
                xaxis: { tickfont: { size: 8 }, range: [0, 150] },
                yaxis: { tickfont: { size: 8 }, title: '' },
                height: 80
            }), { displayModeBar: false, responsive: true });
        });
    });
}


/**
 * Render layer × trait heatmaps - one per method
 * Shows score across all layers (y) and traits (x) for each method
 */
function renderLayerTraitHeatmaps(evalData) {
    const container = document.getElementById('layer-trait-heatmaps-container');
    if (!container) return;

    const allResults = evalData.all_results || [];
    if (allResults.length === 0) {
        container.innerHTML = '<p>No results available.</p>';
        return;
    }

    const methods = ['probe', 'gradient', 'mean_diff'];
    const methodLabels = { 'probe': 'Probe', 'gradient': 'Gradient', 'mean_diff': 'Mean Diff' };

    const traits = [...new Set(allResults.map(r => r.trait))].sort();
    const layers = [...new Set(allResults.map(r => r.layer))].sort((a, b) => a - b);
    const traitLabels = traits.map(t => window.getDisplayName(t));

    // Compute max effect per trait for normalization
    const maxEffectPerTrait = {};
    allResults.forEach(r => {
        if (!maxEffectPerTrait[r.trait] || r.val_effect_size > maxEffectPerTrait[r.trait]) {
            maxEffectPerTrait[r.trait] = r.val_effect_size || 0;
        }
    });

    // v2 score (no 1-Drop)
    const computeScore = (r) => {
        const maxEffect = maxEffectPerTrait[r.trait] || 1;
        const normEffect = (r.val_effect_size || 0) / maxEffect;
        const polarity = r.polarity_correct ? 1 : 0;
        return (r.val_accuracy * 0.43 + normEffect * 0.57) * polarity;
    };

    // Build grid: one heatmap per method
    let html = `<div class="layer-trait-grid">`;
    methods.forEach(method => {
        html += `
            <div class="layer-trait-item">
                <h4>${methodLabels[method]}</h4>
                <div id="layer-trait-${method}"></div>
            </div>
        `;
    });
    html += `</div>`;
    container.innerHTML = html;

    // Render each heatmap
    methods.forEach(method => {
        const methodResults = allResults.filter(r => r.method === method);

        // Build matrix: layers (rows) × traits (cols)
        const matrix = layers.map(layer => {
            return traits.map(trait => {
                const result = methodResults.find(r => r.layer === layer && r.trait === trait);
                return result ? computeScore(result) * 100 : null;
            });
        });

        const trace = {
            z: matrix,
            x: traitLabels,
            y: layers,
            type: 'heatmap',
            colorscale: window.ASYMB_COLORSCALE,
            zmin: 0,
            zmax: 100,
            showscale: method === 'mean_diff',  // Only show scale on last one
            hovertemplate: `%{x}<br>Layer %{y}<br>Score: %{z:.1f}%<extra></extra>`
        };

        if (method === 'mean_diff') {
            trace.colorbar = {
                title: { text: 'Score %', font: { size: 11 } },
                tickvals: [0, 50, 100],
                ticktext: ['0%', '50%', '100%']
            };
        }

        Plotly.newPlot(`layer-trait-${method}`, [trace], window.getPlotlyLayout({
            margin: { l: 30, r: method === 'mean_diff' ? 80 : 10, t: 10, b: 80 },
            xaxis: { tickfont: { size: 9 }, tickangle: -45 },
            yaxis: { tickfont: { size: 9 }, title: 'Layer', dtick: 5 },
            height: 350
        }), { displayModeBar: false, responsive: true });
    });
}


function renderBestVectorSimilarity(evalData) {
    const container = document.getElementById('best-vector-similarity-container');
    if (!container) return;

    const similarityMatrix = evalData.best_vector_similarity || {};
    const traits = Object.keys(similarityMatrix);

    if (traits.length === 0) {
        container.innerHTML = '<p>No similarity matrix available.</p>';
        return;
    }

    // Convert to 2D array
    const matrix = traits.map(t1 =>
        traits.map(t2 => similarityMatrix[t1][t2])
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
        yaxis: { tickfont: { size: 10 } },
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
