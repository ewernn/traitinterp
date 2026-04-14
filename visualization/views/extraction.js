import { fetchJSON, escapeHtml } from '../core/utils.js';
import { getDisplayName, DELTA_COLORSCALE } from '../core/display.js';
import { buildChartLayout, renderChart } from '../core/charts.js';
import { requireExperiment, deferredLoading, renderRunHint, renderSubsection } from '../core/ui.js';

// Trait Extraction - Comprehensive view of extraction quality, methods, and vector properties

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

    // Build the comprehensive view
    contentArea.innerHTML = `
        <div class="tool-view">
            <!-- Page intro -->
            <div class="page-intro">
                <div class="page-intro-text">Measure quality of extracted trait vectors.</div>
                <div class="page-intro-model">Extraction model: <code>${extractionModel}</code></div>
            </div>

            <!-- Best Vectors Summary -->
            <section>
                ${renderSubsection({
                    title: 'Best Vectors Summary',
                    infoId: 'info-best-vectors',
                    infoText: 'Best (layer, method) per trait, ranked by val effect size d. Higher d and higher val accuracy mean cleaner separation between positive and negative examples.'
                })}
                <div id="best-vectors-summary-container"></div>
            </section>

            <!-- Per-Trait Heatmaps -->
            <section>
                ${renderSubsection({
                    title: 'Per-Trait Heatmaps (Layer × Method effect_size)',
                    infoId: 'info-heatmaps',
                    infoText: 'Signed Cohen&#39;s d for each layer (rows) and method (MD, Pr). Green = correct direction with strong separation, red = polarity flipped, white = weak. ★ marks best layer.'
                })}
                <div id="trait-heatmaps-container"></div>
            </section>

            <!-- Logit Lens -->
            <section>
                ${renderSubsection({
                    title: 'Token Decode (Logit Lens)',
                    infoId: 'info-logit-lens',
                    infoText: 'Top vocabulary tokens each vector points toward and away from, via the unembedding at ~90% depth. Coherent lists confirm the vector captured the intended concept.'
                })}
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
    renderLogitLensSection(evalData).catch(err => {
        const container = document.getElementById('logit-lens-container');
        if (container) container.innerHTML = `<div class="info">Logit lens load failed: ${err.message}</div>`;
    });

    // Render math after all content is in DOM
    if (window.MathJax) {
        MathJax.typesetPromise();
    }

    // Setup info toggles
    window.setupSubsectionInfoToggles();
}


/** Return trait names selected in sidebar filter, or empty set for "show all" */
function getSelectedTraitNames() {
    const filteredTraits = window.getFilteredTraits();
    return new Set(filteredTraits.map(t => t.name));
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
                score: effectSize
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
    const selectedTraitNames = getSelectedTraitNames();
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
            trait: getDisplayName(trait),
            method: best.method,
            layer: best.layer,
            accuracy: result?.val_accuracy ?? null,
            effectSize: result?.val_effect_size ?? null
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


function renderTraitHeatmaps(evalData) {
    const container = document.getElementById('trait-heatmaps-container');
    if (!container) return;

    const allResults = evalData.all_results || [];
    if (allResults.length === 0) {
        container.innerHTML = '<p>No results to display.</p>';
        return;
    }

    // Filter by selected traits from sidebar
    const selectedTraitNames = getSelectedTraitNames();
    const results = selectedTraitNames.size > 0
        ? allResults.filter(r => selectedTraitNames.has(r.trait))
        : allResults;

    if (results.length === 0) {
        container.innerHTML = '<p>No results for selected traits.</p>';
        return;
    }

    // Group by trait
    const traitGroups = {};
    results.forEach(r => {
        if (!traitGroups[r.trait]) traitGroups[r.trait] = [];
        traitGroups[r.trait].push(r);
    });

    const traits = Object.keys(traitGroups).sort();

    // Compute best vectors for star indicators
    const bestVectors = computeBestVectors(results);

    // Create grid with legend below
    container.innerHTML = `
        <div class="trait-heatmaps-grid" id="heatmaps-grid"></div>
        <div class="heatmap-legend-footer">
            <span class="file-hint">${traits.length} traits</span>
            <span class="file-hint" title="Best layer by effect size">★ = best</span>
            <div class="heatmap-legend">
                <span title="Cohen's d; negative = wrong polarity">Effect Size (d):</span>
                <div>
                    <div class="heatmap-legend-bar heatmap-legend-bar-diverging"></div>
                    <div class="heatmap-legend-labels">
                        <span>−max</span>
                        <span>0</span>
                        <span>+max</span>
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
        const displayName = getDisplayName(trait);
        const bestInfo = bestVectors[trait];

        const traitDiv = document.createElement('div');
        traitDiv.className = 'trait-heatmap-item';
        traitDiv.innerHTML = `
            <h4 title="${displayName}${bestInfo ? ` (best: L${bestInfo.layer} ${bestInfo.method})` : ''}">${displayName}</h4>
            <div id="heatmap-${traitId}" class="chart-container-sm"></div>
        `;

        grid.appendChild(traitDiv);

        renderSingleTraitHeatmap(traitResults, `heatmap-${traitId}`, bestInfo);
    });
}


function renderSingleTraitHeatmap(traitResults, containerId, bestInfo = null) {
    const methods = ['mean_diff', 'probe'];
    const layers = Array.from(new Set(traitResults.map(r => r.layer))).sort((a, b) => a - b);

    // Build matrix: layers × methods, value = signed effect size (flipped if polarity wrong)
    const matrix = [];
    layers.forEach(layer => {
        const row = methods.map(method => {
            const result = traitResults.find(r => r.layer === layer && r.method === method);
            if (!result || result.val_effect_size == null) return null;
            return result.polarity_correct ? result.val_effect_size : -result.val_effect_size;
        });
        matrix.push(row);
    });

    // Symmetric zrange around 0 based on per-trait max |effect|
    const allValues = matrix.flat().filter(v => v !== null);
    const absMax = allValues.length > 0 ? Math.max(...allValues.map(Math.abs)) : 1;
    const zbound = Math.ceil(absMax);

    const xLabels = ['MD', 'Pr'];

    const trace = {
        z: matrix,
        x: xLabels,
        y: layers,
        type: 'heatmap',
        colorscale: DELTA_COLORSCALE,
        hovertemplate: '%{x} L%{y}: d=%{z:.2f}<extra></extra>',
        zmin: -zbound,
        zmax: zbound,
        zmid: 0,
        showscale: false
    };

    // Build annotations array
    const annotations = [];
    if (bestInfo && bestInfo.layer !== undefined && bestInfo.method) {
        const methodIdx = methods.indexOf(bestInfo.method);
        const layerIdx = layers.indexOf(bestInfo.layer);
        if (methodIdx >= 0 && layerIdx >= 0) {
            annotations.push({
                x: xLabels[methodIdx],
                y: bestInfo.layer,
                text: '★',
                showarrow: false,
                font: { size: 10, color: '#000' },
                xanchor: 'center',
                yanchor: 'middle'
            });
        }
    }

    const layout = buildChartLayout({
        preset: 'heatmap',
        traces: [trace],
        height: 180,
        legendPosition: 'none',
        xaxis: { tickfont: { size: 8 }, tickangle: 0 },
        yaxis: { showticklabels: false, title: '' },
        margin: { l: 5, r: 5, t: 5, b: 25 },
        annotations
    });
    renderChart(containerId, [trace], layout);
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
        const data = await fetchJSON(window.paths.logitLens(trait, modelVariant));
        return { trait, data };
    }));

    // Filter to traits that have data
    const withData = results.filter(r => r.data);

    if (withData.length === 0) {
        const expName = window.state.experimentData?.name || '<exp>';
        container.innerHTML = renderRunHint(
            'No logit lens data.',
            `python analysis/vectors/logit_lens.py --experiment ${expName} --all-traits --save`
        );
        return;
    }

    // Build table
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

        const displayName = getDisplayName(trait);
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

/** Reset extraction-local state (called on experiment change). */
function resetExtractionState() {
    // No module-local caches currently — stub is here for symmetry
    // with the other views, so state.js can call it unconditionally.
}

// ES module exports
export { renderExtraction, resetExtractionState };

// Keep window.* for router + state.js reference
window.renderExtraction = renderExtraction;
window.resetExtractionState = resetExtractionState;
