// Layer Deep Dive View - SAE Feature Decomposition + Attention Patterns
// Shows which interpretable SAE features are active at each token
// Plus attention patterns from per_token analysis data
// Uses global token slider from prompt-picker.js (currentTokenIndex)

// Cache for SAE data and labels
let saeDataCache = null;
let saeLabelsCache = null;
let saeCacheKey = null;  // Track which prompt the cache is for

// Cache for per_token data (contains attention patterns)
let perTokenCache = null;
let perTokenCacheKey = null;

// Cache for full attention data (26 layers × 8 heads)
let attentionCache = null;
let attentionCacheKey = null;

// Cache for logit lens data
let logitLensCache = null;
let logitLensCacheKey = null;

// Currently displayed layer for logit lens (default to last layer)
let currentLogitLensLayer = 25;

/**
 * Load SAE feature labels (cached after first load)
 */
async function loadSaeLabels() {
    if (saeLabelsCache) return saeLabelsCache;

    try {
        const response = await fetch('/sae/gemma-scope-2b-pt-res-canonical/layer_16_width_16k_canonical/feature_labels.json');
        if (!response.ok) return null;
        saeLabelsCache = await response.json();
        return saeLabelsCache;
    } catch (e) {
        console.log('SAE labels not available:', e.message);
        return null;
    }
}

/**
 * Load encoded SAE features for a prompt
 */
async function loadSaeData(promptSet, promptId) {
    const experiment = window.paths.getExperiment();
    const path = `/experiments/${experiment}/inference/sae/${promptSet}/${promptId}_sae.pt.json`;

    try {
        const response = await fetch(path);
        if (!response.ok) return null;
        return await response.json();
    } catch (e) {
        console.log('SAE data not available:', e.message);
        return null;
    }
}

/**
 * Load per_token analysis data (contains attention patterns, trait scores, etc.)
 */
async function loadPerTokenData(promptSet, promptId) {
    const experiment = window.paths.getExperiment();
    const path = `/experiments/${experiment}/analysis/per_token/${promptSet}/${promptId}.json`;

    try {
        const response = await fetch(path);
        if (!response.ok) return null;
        return await response.json();
    } catch (e) {
        console.log('Per-token data not available:', e.message);
        return null;
    }
}

/**
 * Load full attention data (26 layers × 8 heads) for a prompt
 */
async function loadAttentionData(promptSet, promptId) {
    const experiment = window.paths.getExperiment();
    const path = `/experiments/${experiment}/analysis/per_token/${promptSet}/${promptId}_attention.json`;

    try {
        const response = await fetch(path);
        if (!response.ok) return null;
        return await response.json();
    } catch (e) {
        console.log('Attention data not available:', e.message);
        return null;
    }
}

/**
 * Load logit lens data (per-layer predictions) for a prompt
 */
async function loadLogitLensData(promptSet, promptId) {
    const experiment = window.paths.getExperiment();
    const path = `/experiments/${experiment}/analysis/per_token/${promptSet}/${promptId}_logit_lens.json`;

    try {
        const response = await fetch(path);
        if (!response.ok) return null;
        return await response.json();
    } catch (e) {
        console.log('Logit lens data not available:', e.message);
        return null;
    }
}

/**
 * Render the placeholder/setup instructions when SAE data isn't available
 */
function renderSetupInstructions(contentArea) {
    const experiment = window.paths.getExperiment();

    contentArea.innerHTML = `
        <div class="tool-view">
            <div class="page-intro">
                <div class="page-intro-text">Decompose activations into interpretable SAE features.</div>
            </div>
            <section>
                <h2>Setup Required</h2>
                <div class="card">
                    <p>SAE feature decomposition requires encoding raw activations through the Sparse Autoencoder.
                    This converts 2,304 raw neuron activations into 16,384 interpretable features.</p>

                    <table class="def-table" style="margin-top: 12px;">
                        <tr>
                            <td>1. Download labels</td>
                            <td><code>python sae/download_subset.py -n 1000</code></td>
                        </tr>
                        <tr>
                            <td>2. Encode activations</td>
                            <td><code>python sae/encode_sae_features.py --experiment ${experiment} --device mps</code></td>
                        </tr>
                        <tr>
                            <td>3. Refresh page</td>
                            <td>Data will appear automatically</td>
                        </tr>
                    </table>
                    <p style="margin-top: 8px; color: var(--text-tertiary);">Step 2 downloads ~300MB SAE weights on first run</p>
                </div>
            </section>

            ${renderEducationSection()}
        </div>
    `;
}

/**
 * Render education section about SAE features
 */
function renderEducationSection() {
    return `
        <section>
            <h2>Understanding SAE Features</h2>
            <div class="grid">
                <div class="card">
                    <h4>What Are SAE Features?</h4>
                    <p>Sparse Autoencoders decompose polysemantic neurons into monosemantic features.
                    Each feature ideally represents a single interpretable concept.</p>
                    <p><strong>Input:</strong> 2,304 neurons → <strong>Output:</strong> 16,384 sparse features</p>
                </div>

                <div class="card">
                    <h4>Why Sparsity Matters</h4>
                    <p>Only ~50-200 features activate per token (out of 16k).
                    This sparsity makes it easy to see "what's firing" at each position.</p>
                </div>

                <div class="card">
                    <h4>GemmaScope</h4>
                    <p>Google's SAE trained on Gemma 2B base model activations.
                    Feature descriptions from Neuronpedia's automated interpretability.</p>
                    <p><strong>Note:</strong> Trained on base model, not instruction-tuned</p>
                </div>

                <div class="card">
                    <h4>Limitations</h4>
                    <p>Feature descriptions are auto-generated and may be approximate.
                    Some features lack clear interpretations.</p>
                </div>
            </div>
        </section>
    `;
}

/**
 * Render the main visualization with SAE features and attention patterns
 */
function renderVisualization(contentArea, saeData, saeLabels, attentionData, logitLensData) {
    // Get token info from whichever source is available
    const tokenIdx = window.state?.currentTokenIndex || 0;
    const tokens = saeData?.tokens || attentionData?.tokens || [];
    const nPromptTokens = saeData?.n_prompt_tokens || attentionData?.n_prompt_tokens || 0;

    // Clamp to valid range
    const clampedIdx = Math.max(0, Math.min(tokenIdx, tokens.length - 1));
    const isPromptToken = clampedIdx < nPromptTokens;

    const currentToken = tokens[clampedIdx] || '';
    const displayToken = currentToken.replace(/\n/g, '↵').replace(/ /g, '·');
    const tokenPhase = isPromptToken ? 'prompt' : 'response';

    // Check if full attention data exists for this token
    const tokenAttn = attentionData?.attention?.[clampedIdx];
    const hasFullAttention = tokenAttn?.by_layer?.length > 0;

    let html = `
        <div class="tool-view">
            <div class="page-intro">
                <div class="page-intro-text">Mechanistic analysis: SAE features and attention patterns.</div>
            </div>
            <section>
                <h2>Token ${clampedIdx}: <code>${escapeHtml(displayToken)}</code></h2>
                <div class="stats-row">
                    <span><strong>Phase:</strong> ${tokenPhase}</span>
                    <span><strong>Context size:</strong> ${tokenAttn?.context_size || clampedIdx + 1}</span>
                </div>
    `;

    // Check if we have logit lens data for this token
    const hasLogitLens = logitLensData?.predictions?.[clampedIdx];

    // Attention heatmaps section (if full attention data available)
    if (hasFullAttention) {
        const hideSink = window.state?.hideAttentionSink ?? true;
        html += `
                <div class="card">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
                        <div style="flex: 1;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <h4 style="margin: 0;">Attention by Layer (head-averaged)</h4>
                                <label class="toggle-label" style="display: flex; align-items: center; gap: 6px; font-size: 12px; color: var(--text-secondary);">
                                    <input type="checkbox" id="hide-sink-toggle" ${hideSink ? 'checked' : ''}>
                                    Hide attention sink (pos 0)
                                </label>
                            </div>
                            <p class="section-desc">How attention to context positions evolves across layers. ${hasLogitLens ? '<em>Hover over a layer row to see its predictions.</em>' : ''}</p>
                        </div>
                        ${hasLogitLens ? `
                        <div id="logit-lens-panel" style="width: 200px; margin-left: 16px; padding: 8px; background: var(--bg-secondary); border-radius: 4px;">
                            <div style="font-size: 11px; color: var(--text-secondary); margin-bottom: 4px;">
                                <strong>Predictions @ <span id="logit-lens-layer">L${currentLogitLensLayer}</span></strong>
                            </div>
                            <div id="logit-lens-chart" style="width: 100%; height: 120px;"></div>
                            <div id="logit-lens-actual" style="font-size: 10px; color: var(--text-tertiary); margin-top: 4px;"></div>
                        </div>
                        ` : ''}
                    </div>
                    <div id="attention-layers-heatmap" style="width: 100%; height: 400px;"></div>
                </div>

                <div class="card">
                    <h4>Attention by Head (Layer 16)</h4>
                    <p class="section-desc">What each attention head focuses on</p>
                    <div id="attention-heads-heatmap" style="width: 100%; height: 300px;"></div>
                </div>
        `;
    }

    // SAE Features section (if available)
    if (saeData && saeLabels) {
        html += `
                <div class="card">
                    <h4>Top 20 SAE Features</h4>
                    <div id="sae-feature-chart" style="width: 100%; min-height: 500px;"></div>
                </div>

                <div class="card">
                    <h4>Sparsity Stats</h4>
                    <div class="stats-row">
                        <span><strong>Avg Active:</strong> ${(saeData.avg_active_features || 0).toFixed(0)}</span>
                        <span><strong>Min:</strong> ${saeData.min_active_features || 0}</span>
                        <span><strong>Max:</strong> ${saeData.max_active_features || 0}</span>
                        <span><strong>Tokens:</strong> ${saeData.num_tokens || 0}</span>
                    </div>
                </div>
        `;
    }

    // Show message if missing data
    if (!hasFullAttention && !saeData) {
        html += `
                <div class="card">
                    <p>No mechanistic data available for this prompt set.</p>
                    <p class="section-desc">Run SAE encoding or extract attention data.</p>
                </div>
        `;
    }

    html += `
            </section>
            ${renderEducationSection()}
        </div>
    `;

    contentArea.innerHTML = html;

    // Render charts
    if (hasFullAttention) {
        renderLayersHeatmap(tokenAttn, tokens);
        renderHeadsHeatmap(tokenAttn, tokens, 16);

        // Wire up sink toggle
        const sinkToggle = document.getElementById('hide-sink-toggle');
        if (sinkToggle) {
            sinkToggle.addEventListener('change', (e) => {
                window.state.hideAttentionSink = e.target.checked;
                renderLayersHeatmap(tokenAttn, tokens);
                renderHeadsHeatmap(tokenAttn, tokens, 16);
            });
        }

        // Render logit lens and wire up hover
        if (hasLogitLens) {
            renderLogitLensChart(logitLensData, clampedIdx, currentLogitLensLayer);
            setupLogitLensHover(logitLensData, clampedIdx);
        }
    }
    if (saeData && saeLabels) {
        renderFeatureChart(saeData, saeLabels, clampedIdx);
    }
}

/**
 * Render layers × context heatmap (head-averaged attention across all layers)
 */
function renderLayersHeatmap(tokenAttn, allTokens) {
    const chartDiv = document.getElementById('attention-layers-heatmap');
    if (!chartDiv) return;

    const hideSink = window.state?.hideAttentionSink ?? true;
    const startPos = hideSink ? 1 : 0;  // Skip position 0 if hiding sink
    const contextSize = tokenAttn.context_size || 1;
    const nLayers = tokenAttn.by_layer?.length || 26;

    // Build z matrix: [layers, context] - head-averaged
    const z = [];
    for (let layer = 0; layer < nLayers; layer++) {
        const layerData = tokenAttn.by_layer[layer] || [];
        // Average across 8 heads
        const avgAttn = [];
        for (let pos = startPos; pos < contextSize; pos++) {
            let sum = 0;
            let count = 0;
            for (let head = 0; head < layerData.length; head++) {
                if (layerData[head] && pos < layerData[head].length) {
                    sum += layerData[head][pos];
                    count++;
                }
            }
            avgAttn.push(count > 0 ? sum / count : 0);
        }
        z.push(avgAttn);
    }

    // X-axis: context positions
    const xLabels = [];
    for (let i = startPos; i < contextSize; i++) {
        const token = allTokens[i] || `[${i}]`;
        const shortToken = token.slice(0, 6).replace(/\n/g, '↵').replace(/ /g, '·');
        xLabels.push(`${i}:${shortToken}`);
    }

    // Y-axis: layers
    const yLabels = [];
    for (let i = 0; i < nLayers; i++) {
        yLabels.push(`L${i}`);
    }

    // Build colorscale from CSS variables
    const colorscale = getAttentionColorscale();

    const trace = {
        type: 'heatmap',
        z: z,
        x: xLabels,
        y: yLabels,
        colorscale: colorscale,
        hovertemplate: 'Layer %{y}<br>Position %{x}<br>Attention: %{z:.4f}<extra></extra>'
    };

    const layout = window.getPlotlyLayout({
        margin: { l: 40, r: 10, t: 5, b: 80 },
        height: 400,
        xaxis: {
            title: 'Context Position',
            tickangle: -45,
            tickfont: { size: 8 }
        },
        yaxis: {
            title: 'Layer',
            tickfont: { size: 9 }
        }
    });

    Plotly.newPlot(chartDiv, [trace], layout, {
        responsive: true,
        displayModeBar: false
    });
}

/**
 * Render heads × context heatmap for a specific layer
 */
function renderHeadsHeatmap(tokenAttn, allTokens, layer) {
    const chartDiv = document.getElementById('attention-heads-heatmap');
    if (!chartDiv) return;

    const hideSink = window.state?.hideAttentionSink ?? true;
    const startPos = hideSink ? 1 : 0;  // Skip position 0 if hiding sink
    const contextSize = tokenAttn.context_size || 1;
    const layerData = tokenAttn.by_layer?.[layer] || [];
    const nHeads = layerData.length || 8;

    // Build z matrix: [heads, context]
    const z = [];
    for (let head = 0; head < nHeads; head++) {
        const headAttn = layerData[head] || [];
        // Pad to context size if needed
        const row = [];
        for (let pos = startPos; pos < contextSize; pos++) {
            row.push(pos < headAttn.length ? headAttn[pos] : 0);
        }
        z.push(row);
    }

    // X-axis: context positions
    const xLabels = [];
    for (let i = startPos; i < contextSize; i++) {
        const token = allTokens[i] || `[${i}]`;
        const shortToken = token.slice(0, 6).replace(/\n/g, '↵').replace(/ /g, '·');
        xLabels.push(`${i}:${shortToken}`);
    }

    // Y-axis: heads
    const yLabels = [];
    for (let i = 0; i < nHeads; i++) {
        yLabels.push(`Head ${i}`);
    }

    // Build colorscale from CSS variables
    const colorscale = getAttentionColorscale();

    const trace = {
        type: 'heatmap',
        z: z,
        x: xLabels,
        y: yLabels,
        colorscale: colorscale,
        hovertemplate: '%{y}<br>Position %{x}<br>Attention: %{z:.4f}<extra></extra>'
    };

    const layout = window.getPlotlyLayout({
        margin: { l: 60, r: 10, t: 5, b: 80 },
        height: 300,
        xaxis: {
            title: 'Context Position',
            tickangle: -45,
            tickfont: { size: 8 }
        },
        yaxis: {
            title: `Layer ${layer} Heads`,
            tickfont: { size: 10 }
        }
    });

    Plotly.newPlot(chartDiv, [trace], layout, {
        responsive: true,
        displayModeBar: false
    });
}

/**
 * Build attention colorscale from CSS theme variables
 * Goes from background (low attention) to primary color (high attention)
 */
function getAttentionColorscale() {
    const bgColor = window.getCssVar?.('--bg-secondary', '#eaeae8') || '#eaeae8';
    const midColor = window.getCssVar?.('--form-accent', '#9a9970') || '#9a9970';
    const primaryColor = window.getCssVar?.('--primary-color', '#7a7950') || '#7a7950';

    return [
        [0, bgColor],
        [0.5, midColor],
        [1, primaryColor]
    ];
}

/**
 * Render feature bar chart for a specific token
 */
function renderFeatureChart(saeData, saeLabels, tokenIdx) {
    const chartDiv = document.getElementById('sae-feature-chart');
    if (!chartDiv) return;

    // Get top-k feature data for this token
    const indices = saeData.top_k_indices?.[tokenIdx] || [];
    const values = saeData.top_k_values?.[tokenIdx] || [];
    const features = saeLabels?.features || {};

    // Build chart data (top 20 for readability)
    const numToShow = Math.min(20, indices.length);
    const chartData = [];

    for (let i = 0; i < numToShow; i++) {
        const featureIdx = indices[i];
        const activation = values[i];
        const featureInfo = features[String(featureIdx)] || {};
        const description = featureInfo.description || `Feature ${featureIdx}`;

        chartData.push({
            featureIdx,
            activation,
            description: truncate(description, 60),
            fullDescription: description
        });
    }

    // Sort by activation descending, then reverse so highest is at TOP
    chartData.sort((a, b) => b.activation - a.activation);
    chartData.reverse();

    const primaryColor = window.getCssVar?.('--primary-color', '#a09f6c') || '#a09f6c';

    const trace = {
        type: 'bar',
        orientation: 'h',
        y: chartData.map(d => d.description),
        x: chartData.map(d => d.activation),
        text: chartData.map(d => `#${d.featureIdx}`),
        textposition: 'inside',
        insidetextanchor: 'start',
        marker: { color: primaryColor },
        hovertemplate: chartData.map(d =>
            `<b>Feature ${d.featureIdx}</b><br>${d.fullDescription}<br>Activation: %{x:.3f}<extra></extra>`
        )
    };

    const layout = window.getPlotlyLayout({
        margin: { l: 350, r: 10, t: 5, b: 30 },
        height: Math.max(500, numToShow * 28),
        xaxis: { title: 'Activation Strength', zeroline: true },
        yaxis: { automargin: true, tickfont: { size: 11 } }
    });

    Plotly.newPlot(chartDiv, [trace], layout, {
        responsive: true,
        displayModeBar: false
    });
}

/**
 * Render logit lens top-5 predictions chart
 */
function renderLogitLensChart(logitLensData, tokenIdx, layer) {
    const chartDiv = document.getElementById('logit-lens-chart');
    const actualDiv = document.getElementById('logit-lens-actual');
    const layerSpan = document.getElementById('logit-lens-layer');
    if (!chartDiv) return;

    const prediction = logitLensData?.predictions?.[tokenIdx];
    if (!prediction) return;

    const layerData = prediction.by_layer?.[layer];
    if (!layerData) return;

    // Update layer label
    if (layerSpan) {
        layerSpan.textContent = `L${layer}`;
    }

    // Get top 5 predictions
    const top5 = (layerData.top_k || []).slice(0, 5);

    // Show actual next token info
    if (actualDiv) {
        const actualToken = prediction.actual_next_token || '?';
        const actualRank = layerData.actual_rank;
        const actualProb = layerData.actual_prob;
        const displayToken = actualToken.replace(/\n/g, '↵').replace(/ /g, '·');

        if (actualRank && actualRank <= 5) {
            actualDiv.innerHTML = `Actual: <code>${escapeHtml(displayToken)}</code> <span style="color: var(--success-color);">✓ rank ${actualRank}</span>`;
        } else if (actualRank) {
            actualDiv.innerHTML = `Actual: <code>${escapeHtml(displayToken)}</code> rank ${actualRank} (${(actualProb * 100).toFixed(1)}%)`;
        } else {
            actualDiv.innerHTML = `Actual: <code>${escapeHtml(displayToken)}</code>`;
        }
    }

    // Build horizontal bar chart
    const tokens = top5.map(p => {
        const t = (p.token || '').replace(/\n/g, '↵').replace(/ /g, '·');
        return t.slice(0, 8);
    });
    const probs = top5.map(p => p.prob || 0);

    const primaryColor = window.getCssVar?.('--primary-color', '#a09f6c') || '#a09f6c';

    const trace = {
        type: 'bar',
        orientation: 'h',
        y: tokens.reverse(),  // Reverse so #1 is at top
        x: probs.reverse(),
        text: probs.map(p => `${(p * 100).toFixed(0)}%`).reverse(),
        textposition: 'inside',
        insidetextanchor: 'end',
        marker: { color: primaryColor },
        hoverinfo: 'none'
    };

    const layout = window.getPlotlyLayout({
        margin: { l: 50, r: 5, t: 0, b: 0 },
        height: 120,
        xaxis: {
            showticklabels: false,
            showgrid: false,
            zeroline: false,
            range: [0, 1]
        },
        yaxis: {
            tickfont: { size: 9 }
        }
    });

    Plotly.newPlot(chartDiv, [trace], layout, {
        responsive: true,
        displayModeBar: false,
        staticPlot: true
    });
}

/**
 * Setup hover listener on attention heatmap to update logit lens
 */
function setupLogitLensHover(logitLensData, tokenIdx) {
    const heatmapDiv = document.getElementById('attention-layers-heatmap');
    if (!heatmapDiv) return;

    heatmapDiv.on('plotly_hover', (data) => {
        if (!data.points || data.points.length === 0) return;

        // Get layer from y value (e.g., "L16" -> 16)
        const yLabel = data.points[0].y;
        const layer = parseInt(yLabel.replace('L', ''));

        if (!isNaN(layer) && layer !== currentLogitLensLayer) {
            currentLogitLensLayer = layer;
            renderLogitLensChart(logitLensData, tokenIdx, layer);
        }
    });

    heatmapDiv.on('plotly_unhover', () => {
        // Snap back to layer 25 on unhover
        if (currentLogitLensLayer !== 25) {
            currentLogitLensLayer = 25;
            renderLogitLensChart(logitLensData, tokenIdx, 25);
        }
    });
}

/**
 * Utility: Truncate string with ellipsis
 */
function truncate(str, maxLen) {
    if (!str || str.length <= maxLen) return str;
    return str.slice(0, maxLen - 3) + '...';
}

/**
 * Utility: Escape HTML
 */
function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/&/g, '&amp;')
              .replace(/</g, '&lt;')
              .replace(/>/g, '&gt;')
              .replace(/"/g, '&quot;');
}

/**
 * Main render function - called by app.js when view changes or token changes
 */
async function renderLayerDeepDive() {
    const contentArea = document.getElementById('content-area');
    const experiment = window.paths?.getExperiment();
    const promptSet = window.state?.currentPromptSet;
    const promptId = window.state?.currentPromptId;
    const cacheKey = `${experiment}:${promptSet}:${promptId}`;

    // If all caches valid for this prompt, just re-render with new token (no loading flash)
    const hasSaeCache = saeDataCache && saeLabelsCache && saeCacheKey === cacheKey;
    const hasAttentionCache = attentionCache && attentionCacheKey === cacheKey;
    const hasLogitLensCache = logitLensCache && logitLensCacheKey === cacheKey;

    if ((hasSaeCache || hasAttentionCache) && (hasLogitLensCache || !hasAttentionCache)) {
        renderVisualization(contentArea, saeDataCache, saeLabelsCache, attentionCache, logitLensCache);
        return;
    }

    // Show loading only on initial load or prompt change
    contentArea.innerHTML = '<div class="tool-view"><p>Loading data...</p></div>';

    // Try to load SAE labels, SAE data, attention data, and logit lens in parallel
    const [saeLabels, saeData, attentionData, logitLensData] = await Promise.all([
        loadSaeLabels(),
        promptSet && promptId ? loadSaeData(promptSet, promptId) : null,
        promptSet && promptId ? loadAttentionData(promptSet, promptId) : null,
        promptSet && promptId ? loadLogitLensData(promptSet, promptId) : null
    ]);

    // If no data available at all, show setup instructions
    if (!saeData && !attentionData) {
        renderSetupInstructions(contentArea);
        return;
    }

    // Cache and render
    if (saeData) {
        saeDataCache = saeData;
        saeLabelsCache = saeLabels;
        saeCacheKey = cacheKey;
    }
    if (attentionData) {
        attentionCache = attentionData;
        attentionCacheKey = cacheKey;
    }
    if (logitLensData) {
        logitLensCache = logitLensData;
        logitLensCacheKey = cacheKey;
    }

    renderVisualization(contentArea, saeDataCache, saeLabelsCache, attentionCache, logitLensCache);
}

// Export to global scope
window.renderLayerDeepDive = renderLayerDeepDive;
