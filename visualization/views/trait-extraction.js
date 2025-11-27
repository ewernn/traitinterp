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
                    <!-- <li><a href="#table">Quality Table</a></li> -->
                    <!-- <li><a href="#best">Best Vectors</a></li> -->
                    <li><a href="#methods">Method Comparison</a></li>
                    <li><a href="#consistency">Cross-Layer Consistency</a></li>
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

                <!--
                <h3 class="subsection-header" id="table">
                    <span class="subsection-num">3.</span>
                    <span class="subsection-title">All Vectors - Sortable Quality Table</span>
                    <span class="subsection-info-toggle" data-target="info-table">►</span>
                </h3>
                <div class="subsection-info" id="info-table">Every extracted vector ranked by combined score (50% accuracy + 50% normalized effect size). Click headers to sort.</div>
                <div id="quality-table-container"></div>

                <h3 class="subsection-header" id="best">
                    <span class="subsection-num">4.</span>
                    <span class="subsection-title">Best Vector Per Trait</span>
                    <span class="subsection-info-toggle" data-target="info-best">►</span>
                </h3>
                <div class="subsection-info" id="info-best">The single best-performing vector for each trait, showing which method and layer worked best.</div>
                <div id="best-vectors-container"></div>
                -->

                <h3 class="subsection-header" id="methods">
                    <span class="subsection-num">5.</span>
                    <span class="subsection-title">Method Comparison</span>
                    <span class="subsection-info-toggle" data-target="info-methods">►</span>
                </h3>
                <div class="subsection-info" id="info-methods">Average validation accuracy across all traits for each extraction method.</div>
                <div id="method-comparison-container"></div>

                <h3 class="subsection-header" id="consistency">
                    <span class="subsection-num">6.</span>
                    <span class="subsection-title">Cross-Layer Consistency</span>
                    <span class="subsection-info-toggle" data-target="info-consistency">►</span>
                </h3>
                <div class="subsection-info" id="info-consistency">How similar the best vector is across different layers. High consistency means the trait direction is stable throughout the model.</div>
                <div id="cross-layer-consistency-container"></div>
            </section>

            <!-- Section 2: Notation -->
            <section>
                <h2>Notation & Definitions <span class="info-icon" data-info="notation">ⓘ</span></h2>
                ${renderNotation()}
            </section>

            <!-- Section 3: Extraction Techniques -->
            <section>
                <h2>Extraction Techniques <span class="info-icon" data-info="techniques">ⓘ</span></h2>
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
                ${renderScoringExplanation(evalData)}
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
    renderMethodComparison(evalData);
    renderBestVectorSimilarity(evalData);
    renderCrossLayerConsistency(evalData);
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
 */
function setupSubsectionInfoToggles() {
    document.querySelectorAll('.subsection-info-toggle').forEach(toggle => {
        toggle.addEventListener('click', () => {
            const targetId = toggle.dataset.target;
            const infoDiv = document.getElementById(targetId);
            if (infoDiv) {
                const isShown = infoDiv.classList.toggle('show');
                toggle.textContent = isShown ? '▼' : '►';
            }
        });
    });
}


// Info tooltip content for each section
const SECTION_INFO_CONTENT = {
    notation: `
        <h4>About This Notation</h4>
        <p>These symbols are used consistently throughout the extraction pipeline and evaluation metrics.</p>
        <p><strong>Key insight:</strong> Each example's activation is the <em>average</em> across all response tokens, giving a single d-dimensional vector per example.</p>
    `,
    techniques: `
        <h4>Choosing an Extraction Method</h4>
        <ul>
            <li><strong>Mean Diff:</strong> Fast baseline. Use for initial exploration.</li>
            <li><strong>Probe:</strong> Best for high-separability traits (>80% accuracy). Optimized for classification.</li>
            <li><strong>ICA:</strong> Use when traits overlap or interfere. Finds independent directions.</li>
            <li><strong>Gradient:</strong> Best for low-separability traits. Can find subtle directions that other methods miss.</li>
        </ul>
        <p><em>Tip: Compare methods in the heatmaps above to see which works best for each trait.</em></p>
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
                <p>Simple baseline: average positive activations minus average negative activations.</p>
                <p><strong>Use:</strong> Quick baseline, interpretable direction.</p>
            </div>

            <div class="card">
                <h4>Linear Probe</h4>
                <p>$$\\min_\\vec{w} \\sum_i \\log(1 + e^{-y_i (\\vec{w} \\cdot \\vec{a}_i)})$$</p>
                <p>Train logistic regression classifier, use weights as vector.</p>
                <p><strong>Use:</strong> Best for high-separability traits. Optimized for classification.</p>
            </div>

            <div class="card">
                <h4>ICA (Independent Component Analysis)</h4>
                <p>$$\\mathbf{A} = \\mathbf{S} \\mathbf{M}, \\quad \\text{maximize independence of } \\mathbf{S}$$</p>
                <p>Separate mixed signals into independent components, select component with best separation.</p>
                <p><strong>Use:</strong> When traits overlap or interfere. Finds independent directions.</p>
            </div>

            <div class="card">
                <h4>Gradient Optimization</h4>
                <p>$$\\max_\\vec{v} \\left( \\text{mean}(\\mathbf{A}_{\\text{pos}} \\cdot \\vec{v}) - \\text{mean}(\\mathbf{A}_{\\text{neg}} \\cdot \\vec{v}) \\right)$$</p>
                <p>Directly optimize vector to maximize separation between positive/negative projections.</p>
                <p><strong>Use:</strong> Best for low-separability traits. Adaptive optimization.</p>
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


function renderScoringExplanation(evalData) {
    return `
        <div class="card">
            <h4>Combined Score Formula</h4>
            <p>$$\\text{score} = 0.5 \\times \\text{accuracy} + 0.5 \\times \\frac{\\text{effect\\_size}}{\\text{max\\_effect\\_size}}$$</p>

            <h4>Rationale</h4>
            <ul>
                <li><strong>Accuracy (50%):</strong> Measures classification performance. Essential for practical use.</li>
                <li><strong>Normalized Effect Size (50%):</strong> Measures separation magnitude. Prevents overfitting to binary classification.</li>
                <li><strong>Why normalize effect size?</strong> Scale varies across traits (0.5-5.0). Normalization makes comparison fair.</li>
                <li><strong>Why 50/50?</strong> Balances classification accuracy with separation strength. Both matter for vector quality.</li>
            </ul>

            <h4>Alternative Scoring Methods</h4>
            <ul>
                <li><strong>Accuracy-only:</strong> rank by <code>val_accuracy</code></li>
                <li><strong>Effect-size-only:</strong> rank by <code>val_effect_size</code></li>
                <li><strong>Weighted custom:</strong> adjust weights interactively (future feature)</li>
            </ul>
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

    // Compute combined score for each result
    const maxEffectPerTrait = {};
    results.forEach(r => {
        if (!maxEffectPerTrait[r.trait] || r.val_effect_size > maxEffectPerTrait[r.trait]) {
            maxEffectPerTrait[r.trait] = r.val_effect_size || 0;
        }
    });

    const augmentedResults = results.map(r => {
        const max_d = maxEffectPerTrait[r.trait] || 1;
        const score = 0.5 * (r.val_accuracy || 0) + 0.5 * ((r.val_effect_size || 0) / max_d);
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
                <span>Accuracy:</span>
                <div>
                    <div class="heatmap-legend-bar"></div>
                    <div class="heatmap-legend-labels">
                        <span>50%</span>
                        <span>75%</span>
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

        renderSingleTraitHeatmap(traitResults, `heatmap-${traitId}`, true);
    });
}


function renderSingleTraitHeatmap(traitResults, containerId, compact = false) {
    const methods = ['mean_diff', 'probe', 'ica', 'gradient', 'pca_diff', 'random_baseline'];
    const layers = Array.from(new Set(traitResults.map(r => r.layer))).sort((a, b) => a - b);

    // Build matrix: layers × methods, value = accuracy
    const matrix = [];
    layers.forEach(layer => {
        const row = methods.map(method => {
            const result = traitResults.find(r => r.layer === layer && r.method === method);
            return result ? result.val_accuracy * 100 : null;
        });
        matrix.push(row);
    });

    const trace = {
        z: matrix,
        x: compact ? ['MD', 'Pr', 'ICA', 'Gr', 'PCA', 'Rnd'] : methods,
        y: layers,
        type: 'heatmap',
        colorscale: window.ASYMB_COLORSCALE,
        hovertemplate: '%{x} L%{y}: %{z:.1f}%<extra></extra>',
        zmin: 50,
        zmax: 100,
        showscale: !compact
    };

    if (!compact) {
        trace.colorbar = {
            title: { text: 'Accuracy %', font: { size: 11 } },
            tickvals: [50, 75, 90, 100],
            ticktext: ['50%', '75%', '90%', '100%']
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


function renderMethodComparison(evalData) {
    const container = document.getElementById('method-comparison-container');
    if (!container) return;

    const methodSummary = evalData.method_summary || {};
    if (Object.keys(methodSummary).length === 0) {
        container.innerHTML = '<p>No method summary available.</p>';
        return;
    }

    // Extract mean accuracy per method
    const accMean = methodSummary['val_accuracy_mean'] || {};
    const methods = Object.keys(accMean);
    const meanAccuracies = methods.map(m => accMean[m] * 100);

    const trace = {
        x: methods,
        y: meanAccuracies,
        type: 'bar',
        marker: { color: getCssVar('--primary-color', '#a09f6c') },
        text: meanAccuracies.map(v => v.toFixed(1) + '%'),
        textposition: 'outside'
    };

    Plotly.newPlot(container, [trace], window.getPlotlyLayout({
        margin: { l: 60, r: 20, t: 20, b: 60 },
        xaxis: { title: 'Method' },
        yaxis: { title: 'Mean Validation Accuracy (%)', range: [0, 100] },
        height: 300
    }), { displayModeBar: false, responsive: true });
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


function renderCrossLayerConsistency(evalData) {
    const container = document.getElementById('cross-layer-consistency-container');
    if (!container) return;

    const consistencyData = evalData.cross_layer_consistency || {};
    const traits = Object.keys(consistencyData);

    if (traits.length === 0) {
        container.innerHTML = '<p>No cross-layer consistency data available.</p>';
        return;
    }

    // Extract mean consistency per trait
    const traitNames = traits.map(t => window.getDisplayName(t));
    const meanValues = traits.map(t => consistencyData[t]?.mean || 0);

    // Sort by consistency (descending)
    const sortedIndices = meanValues.map((v, i) => i).sort((a, b) => meanValues[b] - meanValues[a]);
    const sortedTraits = sortedIndices.map(i => traitNames[i]);
    const sortedValues = sortedIndices.map(i => meanValues[i]);

    const trace = {
        x: sortedTraits,
        y: sortedValues,
        type: 'bar',
        marker: {
            color: sortedValues.map(v => v >= 0.8 ? getCssVar('--success', '#4a9') : v >= 0.5 ? getCssVar('--warning', '#a94') : getCssVar('--danger', '#a44'))
        },
        text: sortedValues.map(v => v.toFixed(2)),
        textposition: 'outside'
    };

    Plotly.newPlot(container, [trace], window.getPlotlyLayout({
        margin: { l: 60, r: 20, t: 20, b: 120 },
        xaxis: { tickangle: -45, tickfont: { size: 10 } },
        yaxis: { title: 'Mean Cosine Similarity', range: [0, 1] },
        height: 350
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
        { key: 'vector_mean', label: 'Vector Mean', format: 'decimal3', good: null },
        { key: 'vector_std', label: 'Vector Std', format: 'decimal3', good: null },

        // Quality flags
        { key: 'polarity_correct', label: 'Polarity Correct', format: 'bool', good: 'true' },

        // Advanced metrics
        { key: 'top_dims', label: 'Top Dims', format: 'array', good: null },
        { key: 'interference', label: 'Interference', format: 'decimal3', good: '<0.3' },
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
        } else if (metric.format === 'array') {
            metricStats[metric.key] = {
                computed: values.length > 0,
                count: values.length
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
            case 'array': return Array.isArray(value) ? value.join(', ') : String(value);
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
        } else if (metric.format === 'array') {
            minVal = '-';
            meanVal = computed ? `${stats.count} vectors` : '<span class="na">N/A</span>';
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

    // Add cross-layer consistency if available
    const crossLayerData = evalData.cross_layer_consistency || {};
    if (Object.keys(crossLayerData).length > 0) {
        html += `
            <div class="metrics-section">
                <h4>Cross-Layer Consistency (per trait)</h4>
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Trait</th>
                            <th>Mean</th>
                            <th>Std</th>
                            <th>Min</th>
                            <th>Max</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${Object.entries(crossLayerData).map(([trait, data]) => `
                            <tr>
                                <td>${window.getDisplayName(trait)}</td>
                                <td>${data.mean?.toFixed(3) ?? '<span class="na">N/A</span>'}</td>
                                <td>${data.std?.toFixed(3) ?? '<span class="na">N/A</span>'}</td>
                                <td>${data.min?.toFixed(3) ?? '<span class="na">N/A</span>'}</td>
                                <td>${data.max?.toFixed(3) ?? '<span class="na">N/A</span>'}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        `;
    } else {
        html += `<div class="metrics-section"><h4>Cross-Layer Consistency</h4><p class="na">Not computed yet</p></div>`;
    }

    // Add interference if available
    const interferenceData = evalData.interference || {};
    if (Object.keys(interferenceData).length > 0) {
        html += `
            <div class="metrics-section">
                <h4>Interference (max cosine similarity with other traits)</h4>
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Trait</th>
                            <th>Interference</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${Object.entries(interferenceData)
                            .sort((a, b) => b[1] - a[1])
                            .map(([trait, value]) => {
                                const status = value < 0.3 ? 'quality-good' : value < 0.5 ? 'quality-ok' : 'quality-bad';
                                const label = value < 0.3 ? 'Independent' : value < 0.5 ? 'Some overlap' : 'High overlap';
                                return `
                                    <tr>
                                        <td>${window.getDisplayName(trait)}</td>
                                        <td>${value.toFixed(3)}</td>
                                        <td class="${status}">${label}</td>
                                    </tr>
                                `;
                            }).join('')}
                    </tbody>
                </table>
            </div>
        `;
    } else {
        html += `<div class="metrics-section"><h4>Interference</h4><p class="na">Not computed yet</p></div>`;
    }

    // Add summary stats if available
    const summaryStats = evalData.summary_stats || {};
    if (Object.keys(summaryStats).length > 0) {
        html += `
            <div class="metrics-section">
                <h4>Summary Statistics</h4>
                <div class="stats-row">
                    ${summaryStats.best_method ? `<span><strong>Best Method:</strong> ${summaryStats.best_method}</span>` : ''}
                    ${summaryStats.best_layer ? `<span><strong>Best Layer:</strong> ${summaryStats.best_layer}</span>` : ''}
                    ${summaryStats.mean_val_accuracy ? `<span><strong>Mean Val Acc:</strong> ${(summaryStats.mean_val_accuracy * 100).toFixed(1)}%</span>` : ''}
                    ${summaryStats.mean_cross_layer_consistency ? `<span><strong>Mean Consistency:</strong> ${summaryStats.mean_cross_layer_consistency.toFixed(3)}</span>` : ''}
                    ${summaryStats.mean_interference ? `<span><strong>Mean Interference:</strong> ${summaryStats.mean_interference.toFixed(3)}</span>` : ''}
                </div>
            </div>
        `;
    }

    container.innerHTML = html;
}


// Export
window.renderTraitExtraction = renderTraitExtraction;
