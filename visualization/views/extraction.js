import { fetchJSON, escapeHtml } from '../core/utils.js';
import { getDisplayName, DELTA_COLORSCALE, ASYMB_COLORSCALE, getChartColors } from '../core/display.js';
import { buildChartLayout, renderChart } from '../core/charts.js';
import { requireExperiment, deferredLoading, renderRunHint, renderSubsection, renderSegmentedControl } from '../core/ui.js';
import { renderStyledSelect, wireStyledSelect } from '../components/styled-select.js';

// Heatmap metric toggle — module-local state, default to signed effect size
let heatmapMetric = 'effect_size';

// Vector Geometry subsection — module-local state
let vgMethod = null;      // currently-selected method
let vgLayer = null;       // currently-selected layer
let vgSelectedTrait = null;   // click-to-inspect

// Metric config: how to compute cell value, colorscale, z-range, legend label
const METRIC_CONFIG = {
    effect_size: {
        label: 'Effect Size (d)',
        legendLabels: ['−max', '0', '+max'],
        legendBarClass: 'heatmap-legend-bar-diverging',
        colorscale: DELTA_COLORSCALE,
        hoverSuffix: 'd=%{z:.2f}',
        // Signed by polarity: green = correct direction, red = flipped
        computeCell: (r) => r.val_effect_size == null ? null
            : (r.polarity_correct ? r.val_effect_size : -r.val_effect_size),
        zRange: (values) => {
            const absMax = values.length ? Math.max(...values.map(Math.abs)) : 1;
            const b = Math.ceil(absMax);
            return { zmin: -b, zmax: b, zmid: 0 };
        },
    },
    val_accuracy: {
        label: 'Val Accuracy (%)',
        legendLabels: ['0%', '50%', '100%'],
        legendBarClass: 'heatmap-legend-bar-diverging',
        colorscale: DELTA_COLORSCALE,
        hoverSuffix: 'acc=%{z:.1f}%',
        // Accuracy relative to chance: 50% = neutral
        computeCell: (r) => r.val_accuracy == null ? null : r.val_accuracy * 100,
        zRange: () => ({ zmin: 0, zmax: 100, zmid: 50 }),
    },
    combined: {
        label: 'Combined Score',
        legendLabels: ['0', '', '1'],
        legendBarClass: '',  // sequential green
        colorscale: ASYMB_COLORSCALE,
        hoverSuffix: 'score=%{z:.2f}',
        computeCell: (r) => r.combined_score == null ? null : r.combined_score,
        zRange: () => ({ zmin: 0, zmax: 1 }),
    },
};

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
                    title: 'Per-Trait Heatmaps (Layer × Method)',
                    infoId: 'info-heatmaps',
                    infoText: 'Rows are layers, columns are methods (MD, Pr). Metric toggle picks what each cell shows: signed Cohen&#39;s d (diverging, red = polarity flipped), val accuracy (0–100%, diverging around 50% chance), or the pipeline&#39;s combined score (0–1, sequential). ★ marks best layer by absolute effect size.'
                })}
                <div id="trait-heatmaps-container"></div>
            </section>

            <!-- Vector Geometry -->
            <section>
                ${renderSubsection({
                    title: 'Vector Geometry',
                    infoId: 'info-vector-geometry',
                    infoText: 'Cosine similarity between extracted trait vectors, per (method, layer). Scatter: PCA-2D projection of the vectors — close points = similar directions. Click a point to see its ranked neighbors (most similar and most dissimilar traits with cos-sim values).'
                })}
                <div id="vector-geometry-container"></div>
            </section>

            <!-- Logit Lens -->
            <section>
                ${renderSubsection({
                    title: 'Token Decode (Logit Lens)',
                    infoId: 'info-logit-lens',
                    infoText: 'Top vocabulary tokens each vector points toward and away from, via the unembedding at layer n_layers/2 + 10 (L26 on 32-layer Qwen, L50 on 80-layer Llama). Coherent lists confirm the vector captured the intended concept.'
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
    renderVectorGeometrySection().catch(err => {
        const container = document.getElementById('vector-geometry-container');
        if (container) container.innerHTML = `<div class="info">Vector geometry load failed: ${err.message}</div>`;
    });
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

    const metricToggle = renderSegmentedControl({
        id: 'heatmap-metric-control',
        options: [
            { value: 'effect_size', label: 'Effect Size' },
            { value: 'val_accuracy', label: 'Val Accuracy' },
            { value: 'combined', label: 'Combined' },
        ],
        selected: heatmapMetric,
        dataAttr: 'metric',
    });

    const cfg = METRIC_CONFIG[heatmapMetric];
    container.innerHTML = `
        <div class="heatmap-metric-toggle">
            <span class="cb-label">Metric:</span>
            ${metricToggle}
        </div>
        <div class="trait-heatmaps-grid" id="heatmaps-grid"></div>
        <div class="heatmap-legend-footer">
            <span class="file-hint">${traits.length} traits</span>
            <span class="file-hint" title="Best layer by effect size">★ = best</span>
            <div class="heatmap-legend">
                <span>${cfg.label}:</span>
                <div>
                    <div class="heatmap-legend-bar ${cfg.legendBarClass}"></div>
                    <div class="heatmap-legend-labels">
                        <span>${cfg.legendLabels[0]}</span>
                        <span>${cfg.legendLabels[1]}</span>
                        <span>${cfg.legendLabels[2]}</span>
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

    // Wire metric toggle — re-render heatmaps on change
    const metricControl = document.getElementById('heatmap-metric-control');
    if (metricControl) {
        metricControl.addEventListener('click', (e) => {
            const btn = e.target.closest('button[data-metric]');
            if (!btn || btn.dataset.metric === heatmapMetric) return;
            heatmapMetric = btn.dataset.metric;
            renderTraitHeatmaps(evalData);
        });
    }
}


function renderSingleTraitHeatmap(traitResults, containerId, bestInfo = null) {
    const methods = ['mean_diff', 'probe'];
    const layers = Array.from(new Set(traitResults.map(r => r.layer))).sort((a, b) => a - b);
    const cfg = METRIC_CONFIG[heatmapMetric];

    // Build matrix: layers × methods, value per current metric
    const matrix = [];
    layers.forEach(layer => {
        const row = methods.map(method => {
            const result = traitResults.find(r => r.layer === layer && r.method === method);
            return result ? cfg.computeCell(result) : null;
        });
        matrix.push(row);
    });

    const allValues = matrix.flat().filter(v => v !== null);
    const { zmin, zmax, zmid } = cfg.zRange(allValues);

    const xLabels = ['MD', 'Pr'];

    const trace = {
        z: matrix,
        x: xLabels,
        y: layers,
        type: 'heatmap',
        colorscale: cfg.colorscale,
        hovertemplate: `%{x} L%{y}: ${cfg.hoverSuffix}<extra></extra>`,
        zmin,
        zmax,
        ...(zmid !== undefined ? { zmid } : {}),
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

    // Pick the display layer: closest to (n_layers/2 + 10), middle-late residual where the
    // readout tends to show whole-word tokens rather than subword fragments.
    const pickDisplayLayer = (layerNums, nLayers) => {
        if (!layerNums.length) return null;
        const target = Math.floor(nLayers / 2) + 10;
        return layerNums.reduce((best, L) => Math.abs(L - target) < Math.abs(best - target) ? L : best, layerNums[0]);
    };

    for (const { trait, data } of withData) {
        // Pick best method
        const methodPriority = ['probe', 'mean_diff', 'gradient'];
        const method = methodPriority.find(m => data.methods[m]) || Object.keys(data.methods)[0];
        const methodData = data.methods[method];
        if (!methodData) continue;

        // Handle both schemas: new `per_layer: {L: {...}}` and legacy `late: {...}`
        let chosen;
        if (methodData.per_layer) {
            const layerNums = Object.keys(methodData.per_layer).map(Number);
            const pick = pickDisplayLayer(layerNums, data.n_layers || layerNums.length);
            chosen = methodData.per_layer[pick];
        } else if (methodData.late) {
            chosen = methodData.late;
        } else {
            continue;
        }
        if (!chosen) continue;

        const displayName = getDisplayName(trait);

        html += `
            <tr>
                <td><strong>${displayName}</strong><br><span class="hint">${method}</span></td>
                <td class="hint">L${chosen.layer}<br><span class="hint">${chosen.pct}%</span></td>
                <td class="ll-toward">${renderTokens(chosen.toward)}</td>
                <td class="ll-away">${renderTokens(chosen.away)}</td>
            </tr>
        `;
    }

    html += '</tbody></table>';
    container.innerHTML = html;
}

/** Reset extraction-local state (called on experiment change). */
function resetExtractionState() {
    vgMethod = null;
    vgLayer = null;
    vgSelectedTrait = null;
}


// =============================================================================
// Vector Geometry subsection
// =============================================================================

/**
 * Render the Vector Geometry subsection: method/layer controls + scatter + neighbors.
 * Loads precomputed vector_geometry.json; shows a run-hint if missing.
 */
async function renderVectorGeometrySection() {
    const container = document.getElementById('vector-geometry-container');
    if (!container) return;

    const data = await fetchJSON(window.paths.vectorGeometry());
    if (!data || !data.data || !data.methods || data.methods.length === 0) {
        const expName = window.state.experimentData?.name || 'your_experiment';
        container.innerHTML = renderRunHint(
            'No vector geometry data',
            `python analysis/vectors/trait_vector_geometry.py --experiment ${expName}`
        );
        return;
    }

    // Initialize defaults if not already set
    if (!vgMethod || !data.methods.includes(vgMethod)) vgMethod = data.methods[0];
    const layersForMethod = Object.keys(data.data[vgMethod] || {}).map(Number).sort((a, b) => a - b);
    if (layersForMethod.length === 0) {
        container.innerHTML = `<div class="info">No layers found for method <code>${vgMethod}</code>.</div>`;
        return;
    }
    if (vgLayer == null || !layersForMethod.includes(vgLayer)) {
        vgLayer = layersForMethod[Math.floor(layersForMethod.length / 2)];
    }

    container.innerHTML = `
        <div class="vg-controls">
            <div class="vg-control-group">
                <span class="cb-label">Method:</span>
                <div id="vg-method-select-wrap"></div>
            </div>
            <div class="vg-control-group">
                <span class="cb-label">Layer:</span>
                <input type="range" id="vg-layer-slider"
                       min="${layersForMethod[0]}" max="${layersForMethod[layersForMethod.length - 1]}"
                       step="1" value="${vgLayer}"
                       style="width: 200px; accent-color: var(--form-accent);">
                <span class="cb-label" id="vg-layer-label">L${vgLayer}</span>
            </div>
        </div>
        <div class="vg-panels">
            <div id="vg-scatter" class="vg-panel-scatter"></div>
            <div id="vg-neighbors" class="vg-panel-neighbors"></div>
        </div>
    `;

    // Render the styled-select for methods
    const methodWrap = document.getElementById('vg-method-select-wrap');
    methodWrap.innerHTML = renderStyledSelect({
        id: 'vg-method-select',
        options: data.methods.map(m => ({ value: m, label: m })),
        selected: vgMethod,
        onChange: (val) => {
            vgMethod = val;
            const layers = Object.keys(data.data[vgMethod] || {}).map(Number).sort((a, b) => a - b);
            if (!layers.includes(vgLayer)) vgLayer = layers[Math.floor(layers.length / 2)];
            const slider = document.getElementById('vg-layer-slider');
            if (slider) {
                slider.min = layers[0];
                slider.max = layers[layers.length - 1];
                slider.value = vgLayer;
                document.getElementById('vg-layer-label').textContent = `L${vgLayer}`;
            }
            vgSelectedTrait = null;
            renderVectorGeometryPanels(data);
        },
    });
    wireStyledSelect(methodWrap);

    // Wire the layer slider — snap to nearest available layer
    const slider = document.getElementById('vg-layer-slider');
    const label = document.getElementById('vg-layer-label');
    slider.addEventListener('input', () => {
        const layers = Object.keys(data.data[vgMethod] || {}).map(Number).sort((a, b) => a - b);
        const requested = parseInt(slider.value);
        // Snap to nearest existing layer (handles sparse coverage)
        const nearest = layers.reduce((best, l) =>
            Math.abs(l - requested) < Math.abs(best - requested) ? l : best, layers[0]);
        if (nearest !== vgLayer) {
            vgLayer = nearest;
            slider.value = vgLayer;
            label.textContent = `L${vgLayer}`;
            renderVectorGeometryPanels(data);
        } else {
            label.textContent = `L${vgLayer}`;
        }
    });

    renderVectorGeometryPanels(data);
}

/** Render both scatter + neighbors for the currently selected (method, layer). */
function renderVectorGeometryPanels(data) {
    const slice = data.data[vgMethod]?.[String(vgLayer)];
    if (!slice) return;
    // Default-select first trait so neighbors panel has something to show.
    if (!vgSelectedTrait || !slice.traits.includes(vgSelectedTrait)) {
        vgSelectedTrait = slice.traits[0];
    }
    renderVectorGeometryScatter(slice, data);
    renderVectorGeometryNeighbors(slice, data);
}

/** Scatter of trait vectors in PCA-2D space. */
function renderVectorGeometryScatter(slice, data) {
    const palette = getChartColors();
    const colors = slice.traits.map((_, i) => palette[i % palette.length]);
    const sizes = slice.traits.map(t => t === vgSelectedTrait ? 14 : 8);
    const borders = slice.traits.map(t => t === vgSelectedTrait ? 2 : 0);

    const trace = {
        type: 'scatter',
        mode: 'markers+text',
        x: slice.coords_2d.map(c => c[0]),
        y: slice.coords_2d.map(c => c[1]),
        text: slice.traits.map(getDisplayName),
        textposition: 'top center',
        textfont: { size: 9, color: '#aaa' },
        marker: {
            size: sizes,
            color: colors,
            line: { width: borders, color: '#eee' },
        },
        customdata: slice.traits,
        hovertemplate: '<b>%{text}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>',
    };

    const layout = buildChartLayout({
        preset: null,
        traces: [trace],
        height: 360,
        legendPosition: 'none',
        xaxis: { title: 'PC1', zeroline: true, showgrid: false },
        yaxis: { title: 'PC2', zeroline: true, showgrid: false },
        margin: { l: 50, r: 20, t: 20, b: 40 },
    });

    renderChart('vg-scatter', [trace], layout);

    // Click a point → update selection + re-render both panels
    const plotDiv = document.getElementById('vg-scatter');
    plotDiv.on('plotly_click', (ev) => {
        const pt = ev.points?.[0];
        if (!pt || !pt.customdata) return;
        vgSelectedTrait = pt.customdata;
        renderVectorGeometryPanels(data);
    });
}

/** Ranked neighbor list for the selected trait at the current slice. */
function renderVectorGeometryNeighbors(slice, data) {
    const panel = document.getElementById('vg-neighbors');
    if (!panel) return;

    const { traits, cos_sim: cos } = slice;
    const idx = traits.indexOf(vgSelectedTrait);
    if (idx < 0) { panel.innerHTML = ''; return; }

    const pairs = traits
        .map((t, i) => ({ trait: t, sim: cos[idx][i] }))
        .filter(p => p.trait !== vgSelectedTrait)
        .sort((a, b) => b.sim - a.sim);

    const N = Math.min(10, pairs.length);
    const top = pairs.slice(0, N);
    // When there are few traits, top+bottom may overlap — take the tail of what's left.
    const bottom = pairs.slice(-Math.min(N, Math.max(0, pairs.length - N))).reverse();

    const row = (p) => {
        const v = p.sim;
        const cls = v > 0.3 ? 'pos' : v < -0.1 ? 'neg' : '';
        return `
            <div class="vg-neighbor-row" data-trait="${p.trait}">
                <span class="vg-neighbor-sim ${cls}">${v >= 0 ? '+' : ''}${v.toFixed(3)}</span>
                <span class="vg-neighbor-name">${escapeHtml(getDisplayName(p.trait))}</span>
            </div>
        `;
    };

    panel.innerHTML = `
        <div class="vg-neighbors-header">
            <span class="vg-neighbors-label">Neighbors of</span>
            <span class="vg-neighbors-selected">${escapeHtml(getDisplayName(vgSelectedTrait))}</span>
        </div>
        <div class="vg-neighbors-subhead">Most similar</div>
        <div class="vg-neighbors-list">${top.map(row).join('')}</div>
        ${bottom.length > 0 ? `
            <div class="vg-neighbors-subhead">Most dissimilar</div>
            <div class="vg-neighbors-list">${bottom.map(row).join('')}</div>
        ` : ''}
    `;

    panel.querySelectorAll('.vg-neighbor-row').forEach(el => {
        el.addEventListener('click', () => {
            vgSelectedTrait = el.dataset.trait;
            renderVectorGeometryPanels(data);
        });
    });
}

// ES module exports
export { renderExtraction, resetExtractionState };

// Keep window.* for router + state.js reference
window.renderExtraction = renderExtraction;
window.resetExtractionState = resetExtractionState;
