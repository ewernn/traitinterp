/**
 * Extraction view — orchestrator + reference section.
 *
 * Composes the page from the four section files:
 *   1. Best Vectors Summary
 *   2. Per-Trait Heatmaps (Layer × Method)
 *   3. Vector Geometry
 *   4. Logit Lens
 * Plus a collapsible Reference section (notation + methods + metrics).
 */

import { requireExperiment, deferredLoading, renderRunHint, renderSubsection, renderSegmentedControl } from '../../core/ui.js';
import { fetchJSON, renderMath } from '../../core/utils.js';
import { extractionState, resetExtractionState } from './extraction-data.js';
import { renderBestVectorsSummary } from './section-best-vectors.js';
import { renderTraitHeatmaps } from './section-heatmaps.js';
import { renderVectorGeometrySection } from './section-vector-geometry.js';
import { renderLogitLensSection } from './section-logit-lens.js';

/** Re-render data sections after a filter change. */
function rerenderSections(evalData) {
    renderBestVectorsSummary(evalData);
    renderTraitHeatmaps(evalData);
    renderLogitLensSection(evalData).catch(err => {
        const c = document.getElementById('logit-lens-container');
        if (c) c.innerHTML = `<div class="info">Logit lens load failed: ${err.message}</div>`;
    });
}

/** Build chip-style selector for component or position. Hidden if only one option. */
function renderFilterChips(evalData) {
    const recs = evalData.all_results || [];
    const components = evalData.components || [...new Set(recs.map(r => r.component).filter(Boolean))].sort();
    const positions = evalData.positions || [...new Set(recs.map(r => r.position).filter(Boolean))].sort();

    if (components.length <= 1 && positions.length <= 1) return '';

    const chipGroup = (label, options, current, dataAttr) => `
        <div class="chip-group" data-attr="${dataAttr}">
            <span class="chip-group-label">${label}:</span>
            ${options.map(opt => `
                <button class="chip ${opt === current ? 'active' : ''}" data-value="${opt}">${opt}</button>
            `).join('')}
        </div>
    `;

    let html = '<div class="extraction-filters" style="display:flex;gap:var(--space-md);margin-bottom:var(--space-md);flex-wrap:wrap;">';
    if (components.length > 1) html += chipGroup('Component', components, extractionState.componentFilter, 'component');
    if (positions.length > 1) html += chipGroup('Position', positions, extractionState.positionFilter, 'position');
    html += '</div>';
    return html;
}

/** Wire up chip clicks to update state and re-render. */
function attachChipHandlers(evalData) {
    document.querySelectorAll('.extraction-filters .chip-group').forEach(group => {
        const attr = group.dataset.attr;
        group.querySelectorAll('.chip').forEach(btn => {
            btn.addEventListener('click', () => {
                const value = btn.dataset.value;
                if (attr === 'component') extractionState.componentFilter = value;
                if (attr === 'position') extractionState.positionFilter = value;
                group.querySelectorAll('.chip').forEach(b => b.classList.toggle('active', b.dataset.value === value));
                rerenderSections(evalData);
            });
        });
    });
}

/**
 * Trait Extraction — comprehensive view of extraction quality, methods, and vector properties.
 */
async function renderExtraction() {
    const contentArea = document.getElementById('content-area');

    if (requireExperiment(contentArea)) return;

    const { cancel } = deferredLoading(contentArea, 'Loading extraction evaluation data...');
    const evalData = await fetchJSON(window.paths.extractionEvaluation());
    cancel();

    if (!evalData || !evalData.all_results || evalData.all_results.length === 0) {
        contentArea.innerHTML = `<div class="tool-view">${renderRunHint(
            'No extraction evaluation data',
            `python analysis/vectors/extraction_evaluation.py --experiment ${window.state.experimentData?.name || 'your_experiment'}`
        )}</div>`;
        return;
    }

    // Get extraction model from experiment config
    const config = window.state.experimentData?.experimentConfig;
    const extractionVariant = evalData.model_variant || config?.defaults?.extraction || 'base';
    const extractionModel = config?.model_variants?.[extractionVariant]?.model || 'unknown';

    // Default filters: pick first available so multi-component files have a sensible
    // initial view (residual if present, else first listed). Single-component files
    // stay unfiltered.
    const recs = evalData.all_results || [];
    const components = evalData.components || [...new Set(recs.map(r => r.component).filter(Boolean))].sort();
    const positions = evalData.positions || [...new Set(recs.map(r => r.position).filter(Boolean))].sort();
    if (components.length > 1 && extractionState.componentFilter == null) {
        extractionState.componentFilter = components.includes('residual') ? 'residual' : components[0];
    }
    if (positions.length > 1 && extractionState.positionFilter == null) {
        extractionState.positionFilter = positions[0];
    }

    contentArea.innerHTML = `
        <div class="tool-view">
            <div class="page-intro">
                <div class="page-intro-text">Measure quality of extracted trait vectors.</div>
                <div class="page-intro-model">Extraction model: <code>${extractionModel}</code></div>
            </div>

            ${renderFilterChips(evalData)}

            <section>
                ${renderSubsection({
                    title: 'Best Vectors Summary',
                    infoId: 'info-best-vectors',
                    infoText: 'Best (layer, method) per trait, ranked by val effect size d. Higher d and higher val accuracy mean cleaner separation between positive and negative examples.'
                })}
                <div id="best-vectors-summary-container"></div>
            </section>

            <section>
                ${renderSubsection({
                    title: 'Per-Trait Heatmaps (Layer × Method)',
                    infoId: 'info-heatmaps',
                    infoText: 'Rows are layers, columns are methods (MD = mean_diff, Pr = linear probe; gradient is tracked elsewhere). Metric toggle picks what each cell shows: signed Cohen&#39;s d (diverging, red = polarity flipped), val accuracy (0–100%, diverging around 50% chance), or the combined score — mean of val_accuracy and normalized effect size, zeroed when polarity flips (0–1, sequential). ★ marks best layer by absolute effect size.'
                })}
                <div id="trait-heatmaps-container"></div>
            </section>

            <section>
                ${renderSubsection({
                    title: 'Vector Geometry',
                    infoId: 'info-vector-geometry',
                    infoText: 'Cosine similarity between extracted trait vectors, per (method, layer). Scatter: PCA-2D projection of the vectors — close points = similar directions. Click a point to see its ranked neighbors (most similar and most dissimilar traits with cos-sim values).'
                })}
                <div id="vector-geometry-container"></div>
            </section>

            <section>
                ${renderSubsection({
                    title: 'Logit Lens',
                    infoId: 'info-logit-lens',
                    infoText: 'Top vocabulary tokens each vector points toward and away from, via the unembedding at layer n_layers÷2 + 10 — a heuristic favoring ~90% depth (e.g. L26 on 32-layer Qwen3.5-9B, L50 on 80-layer Llama 70B). Coherent lists confirm the vector captured the intended concept.'
                })}
                <div id="logit-lens-container"></div>
            </section>

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
    renderVectorGeometrySection().catch(err => {
        const container = document.getElementById('vector-geometry-container');
        if (container) container.innerHTML = `<div class="info">Vector geometry load failed: ${err.message}</div>`;
    });
    renderLogitLensSection(evalData).catch(err => {
        const container = document.getElementById('logit-lens-container');
        if (container) container.innerHTML = `<div class="info">Logit lens load failed: ${err.message}</div>`;
    });

    attachChipHandlers(evalData);

    renderMath(document.getElementById('content-area'));

    window.setupSubsectionInfoToggles();
}

// === Reference content (static HTML) ===

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
                    <li><strong>Per-method:</strong> 3 signal methods (probe, mean_diff, gradient) × L layers = 3L vectors/trait. An optional <code>random_baseline</code> sanity check is available but excluded from the main charts.</li>
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
                <p>L2 norm of vector. Range: 0-∞. Model-dependent — compare within the same model family rather than to an absolute target.</p>
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

export { renderExtraction, resetExtractionState };

// Keep window.* for router + state.js reference
window.renderExtraction = renderExtraction;
window.resetExtractionState = resetExtractionState;
