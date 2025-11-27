// Layer Deep Dive View - SAE Feature Decomposition
// Shows which interpretable SAE features are active at each token
// Uses global token slider from prompt-picker.js (currentTokenIndex)

// Cache for SAE data and labels
let saeDataCache = null;
let saeLabelsCache = null;
let saeCacheKey = null;  // Track which prompt the cache is for

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
 * Render the main SAE visualization when data is available
 */
function renderSaeVisualization(contentArea, saeData, saeLabels) {
    // Use global token index directly - SAE data now includes prompt+response
    const tokenIdx = window.state?.currentTokenIndex || 0;
    const tokens = saeData.tokens || [];
    const nPromptTokens = saeData.n_prompt_tokens || 0;

    // Clamp to valid range
    const clampedIdx = Math.max(0, Math.min(tokenIdx, tokens.length - 1));
    const isPromptToken = clampedIdx < nPromptTokens;

    const currentToken = tokens[clampedIdx] || '';
    const displayToken = currentToken.replace(/\n/g, '↵').replace(/ /g, '·');
    const tokenPhase = isPromptToken ? 'prompt' : 'response';

    contentArea.innerHTML = `
        <div class="tool-view">
            <div class="page-intro">
                <div class="page-intro-text">Decompose activations into interpretable SAE features.</div>
            </div>
            <section>
                <h2>Token ${clampedIdx}: <code>${escapeHtml(displayToken)}</code></h2>
                <div class="stats-row">
                    <span><strong>Phase:</strong> ${tokenPhase}</span>
                    <span><strong>Layer:</strong> ${saeData.layer || 16}</span>
                </div>

                <div class="card">
                    <h4>Top 20 Features</h4>
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
            </section>

            ${renderEducationSection()}
        </div>
    `;

    // Render the chart
    renderFeatureChart(saeData, saeLabels, clampedIdx);
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

    // If cache is valid for this prompt, just re-render with new token (no loading flash)
    if (saeDataCache && saeLabelsCache && saeCacheKey === cacheKey) {
        renderSaeVisualization(contentArea, saeDataCache, saeLabelsCache);
        return;
    }

    // Show loading only on initial load or prompt change
    contentArea.innerHTML = '<div class="tool-view"><p>Loading SAE data...</p></div>';

    // Try to load SAE labels and data
    const [saeLabels, saeData] = await Promise.all([
        loadSaeLabels(),
        promptSet && promptId ? loadSaeData(promptSet, promptId) : null
    ]);

    // If no SAE data available, show setup instructions
    if (!saeData || !saeLabels) {
        renderSetupInstructions(contentArea);
        return;
    }

    // Cache and render
    saeDataCache = saeData;
    saeLabelsCache = saeLabels;
    saeCacheKey = cacheKey;
    renderSaeVisualization(contentArea, saeData, saeLabels);
}

// Export to global scope
window.renderLayerDeepDive = renderLayerDeepDive;
