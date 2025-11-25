// Token Explorer View - Interactive per-token visualization
// Uses the token slider to show per-token metrics that update dynamically
//
// ============================================================================
// GRAPH REFERENCE
// ============================================================================
//
// 1. PCA TRAJECTORY
//    ---------------
//    What: Each token's path through 2D PCA space across all 26 layers
//    Math: PCA fitted on all hidden states [n_tokens × 26 layers × 2304] → 2D
//          For each token: trajectory = PCA.transform(hidden[token, :, :])  → [26, 2]
//    Read: Green dot = layer 0, Red square = layer 25
//          Current token in red, others faint gray
//          Curved path = complex transformation, straight = simple change
//    Note: BOS (token 0) excluded - it's an outlier that distorts the scale
//
// 2. VELOCITY THROUGH LAYERS
//    ------------------------
//    What: How fast this token's representation changes at each layer transition
//    Math: velocity[L] = hidden[L+1] - hidden[L]                    → [25, 2304]
//          normalized_velocity[L] = ||velocity[L]|| / ||hidden[L]|| → [25]
//    Read: High peaks = major transformation happening
//          Typical pattern: high (L0-6) → low (L7-22) → high (L23-24)
//          Values > 1.0 mean token moved more than its own magnitude
//
// 3. TRAIT SCORES (Layer 16)
//    ------------------------
//    What: How strongly this token activates each of 10 trait directions
//    Math: score = hidden[token, L16, :] · (trait_vector / ||trait_vector||)
//          Result is a scalar for each trait
//    Read: Teal bars = positive activation (expresses trait)
//          Red bars = negative (anti-expresses trait)
//          Sorted by |score| so strongest traits at top
//          Layer 16 chosen as middle layer (semantic, not output-specific)
//
// 4. ATTENTION PATTERN (Layer 16)
//    -----------------------------
//    What: Where this token "looks" - attention distribution over context
//    Math: For 8-head attention: avg_attn = mean(attn_weights, dim=heads)
//          Shows the row for this token: attn[token_idx, :]  → [context_size]
//    Read: Tall bars = tokens this position attends to strongly
//          Look for: attention to BOS, content words, recent tokens
//    Note: Only available when attention was captured (dynamic prompts)
//
// 5. ALL TOKENS: TRAIT SCORES AT LAYER 16
//    ---------------------------------------
//    What: Heatmap showing which tokens activate which traits (layer 16 snapshot)
//    Math: For each token and each trait:
//          score = hidden[token, L16, :] · normalized_trait_vector
//          Result: [n_tokens, 10_traits] matrix
//    Read: Red = positive activation, Blue = negative activation
//          Yellow box highlights current token
//          Shows which parts of prompt/response carry which traits
//
// 6. DISTANCE TO OTHER TOKENS (Layer 16)
//    -------------------------------------
//    What: How similar/different this token is to all others at layer 16
//    Math: cos_sim = normalized(hidden[this]) · normalized(hidden[other])
//          distance = 1 - cos_sim   (0 = identical, 2 = opposite)
//    Read: Short bars = similar representation (related meaning)
//          Tall bars = different representation
//          Blue = prompt tokens, Red = response tokens
//          Vertical line marks where response starts
//
// ============================================================================

let tokenExplorerData = null;
let tokenExplorerCache = { promptSet: null, promptId: null };

async function loadTokenExplorerData() {
    const promptSet = window.state.currentPromptSet;
    const promptId = window.state.currentPromptId;

    // Check cache
    if (tokenExplorerCache.promptSet === promptSet &&
        tokenExplorerCache.promptId === promptId &&
        tokenExplorerData) {
        return tokenExplorerData;
    }

    // Load from analysis folder using PathBuilder
    const url = window.paths.analysisPerToken(promptSet, promptId);

    try {
        const response = await fetch(url);
        if (!response.ok) {
            console.error('Failed to load per-token data:', response.status);
            return null;
        }
        tokenExplorerData = await response.json();
        tokenExplorerCache = { promptSet, promptId };
        return tokenExplorerData;
    } catch (error) {
        console.error('Error loading per-token data:', error);
        return null;
    }
}


async function renderTokenExplorer() {
    const contentArea = document.getElementById('content-area');

    // Check if we already have the DOM structure (avoid scroll reset on slider move)
    const existingExplorer = contentArea.querySelector('.token-explorer');
    const dataAlreadyCached = tokenExplorerCache.promptSet === window.state.currentPromptSet &&
                              tokenExplorerCache.promptId === window.state.currentPromptId &&
                              tokenExplorerData;

    // Only show loading if we need to fetch new data
    if (!dataAlreadyCached) {
        contentArea.innerHTML = '<div class="loading">Loading token explorer...</div>';
    }

    const data = await loadTokenExplorerData();

    if (!data) {
        contentArea.innerHTML = `
            <div class="error" style="margin: 16px; padding: 16px;">
                <h3>No per-token data available</h3>
                <p>Run the per-token analysis script first:</p>
                <code>python experiments/{experiment}/analysis/compute_per_token_metrics.py</code>
            </div>
        `;
        return;
    }

    // Get current token index from state
    const tokenIdx = Math.min(window.state.currentTokenIndex || 0, data.n_total_tokens - 1);
    const tokenData = data.per_token[tokenIdx];

    if (!tokenData) {
        contentArea.innerHTML = '<div class="error">Invalid token index</div>';
        return;
    }

    // If DOM exists and data was cached, just update plots (no innerHTML replacement)
    if (existingExplorer && dataAlreadyCached) {
        // Update header text only
        const tokenLabel = existingExplorer.querySelector('.token-label');
        const phaseBadge = existingExplorer.querySelector('.token-phase-badge');
        if (tokenLabel) tokenLabel.textContent = `Token ${tokenIdx}: "${tokenData.token}"`;
        if (phaseBadge) {
            phaseBadge.textContent = tokenData.phase;
            phaseBadge.className = `token-phase-badge ${tokenData.phase}`;
        }

        // Re-render plots (containers already exist)
        renderPCATrajectory(data, tokenIdx);
        renderVelocityPlot(tokenData);
        renderTraitScores(tokenData);
        renderAttentionPattern(tokenData, data);
        renderAllTokensTraitHeatmap(data, tokenIdx);
        renderDistancePlot(tokenData, data);
        return;
    }

    // Build the layout (only when DOM doesn't exist or data changed)
    contentArea.innerHTML = `
        <div class="token-explorer">
            <div class="token-explorer-header">
                <span class="token-label">Token ${tokenIdx}: "${escapeHtml(tokenData.token)}"</span>
                <span class="token-phase-badge ${tokenData.phase}">${tokenData.phase}</span>
            </div>

            <div class="token-explorer-grid">
                <!-- Row 1: PCA Trajectory + Velocity -->
                <div class="explorer-panel" id="pca-trajectory-panel">
                    <h4>Token Trajectory (PCA)</h4>
                    <div id="pca-trajectory-plot"></div>
                </div>

                <div class="explorer-panel" id="velocity-panel">
                    <h4>Velocity Through Layers</h4>
                    <div id="velocity-plot"></div>
                </div>

                <!-- Row 2: Trait Scores + Attention -->
                <div class="explorer-panel" id="trait-scores-panel">
                    <h4>Trait Scores (Layer 16)</h4>
                    <div id="trait-scores-plot"></div>
                </div>

                <div class="explorer-panel" id="attention-panel">
                    <h4>Attention Pattern (Layer 16)</h4>
                    <div id="attention-plot"></div>
                </div>

                <!-- Row 3: All Tokens Trait Heatmap + Distance -->
                <div class="explorer-panel wide" id="all-tokens-trait-panel">
                    <h4>All Tokens: Trait Scores at Layer 16</h4>
                    <div id="all-tokens-trait-plot"></div>
                </div>

                <div class="explorer-panel" id="distance-panel">
                    <h4>Distance to Other Tokens (Layer 16)</h4>
                    <div id="distance-plot"></div>
                </div>
            </div>
        </div>
    `;

    // Render all plots
    renderPCATrajectory(data, tokenIdx);
    renderVelocityPlot(tokenData);
    renderTraitScores(tokenData);
    renderAttentionPattern(tokenData, data);
    renderAllTokensTraitHeatmap(data, tokenIdx);
    renderDistancePlot(tokenData, data);
}


function renderPCATrajectory(data, currentTokenIdx) {
    const container = document.getElementById('pca-trajectory-plot');
    if (!container) return;

    const traces = [];

    // Plot all tokens faintly (skip token 0 = BOS, it has wildly different dynamics)
    data.per_token.forEach((token, idx) => {
        if (!token.pca_trajectory) return;
        if (idx === 0) return;  // Skip BOS - it's an outlier

        const x = token.pca_trajectory.map(p => p[0]);
        const y = token.pca_trajectory.map(p => p[1]);

        const isCurrentToken = idx === currentTokenIdx;

        traces.push({
            x: x,
            y: y,
            mode: 'lines',
            line: {
                color: isCurrentToken ? '#ff6b6b' : 'rgba(100, 100, 100, 0.2)',
                width: isCurrentToken ? 3 : 1
            },
            name: isCurrentToken ? `Token ${idx}: ${token.token}` : '',
            showlegend: isCurrentToken,
            hoverinfo: isCurrentToken ? 'text' : 'skip',
            text: isCurrentToken ? token.pca_trajectory.map((_, i) => `Layer ${i}`) : null
        });

        // Add markers for start/end of current token
        if (isCurrentToken) {
            traces.push({
                x: [x[0]],
                y: [y[0]],
                mode: 'markers',
                marker: { color: 'green', size: 12, symbol: 'circle' },
                name: 'Start (L0)',
                showlegend: true
            });
            traces.push({
                x: [x[x.length - 1]],
                y: [y[y.length - 1]],
                mode: 'markers',
                marker: { color: 'red', size: 12, symbol: 'square' },
                name: 'End (L25)',
                showlegend: true
            });
        }
    });

    const layout = {
        xaxis: { title: 'PC1', zeroline: false },
        yaxis: { title: 'PC2', zeroline: false },
        margin: { l: 50, r: 20, t: 20, b: 40 },
        height: 220,
        showlegend: true,
        legend: { x: 0, y: 1, bgcolor: 'rgba(255,255,255,0.8)' }
    };

    Plotly.newPlot(container, traces, layout, { responsive: true });
}


function renderVelocityPlot(tokenData) {
    const container = document.getElementById('velocity-plot');
    if (!container || !tokenData.normalized_velocity_per_layer) return;

    const layers = tokenData.normalized_velocity_per_layer.map((_, i) => i);

    const traces = [{
        x: layers,
        y: tokenData.normalized_velocity_per_layer,
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Normalized Velocity',
        line: { color: '#4ecdc4', width: 2 },
        marker: { size: 6 }
    }];

    const layout = {
        xaxis: { title: 'Layer Transition', dtick: 5 },
        yaxis: { title: 'Normalized Velocity' },
        margin: { l: 50, r: 20, t: 20, b: 40 },
        height: 220
    };

    Plotly.newPlot(container, traces, layout, { responsive: true });
}


function renderTraitScores(tokenData) {
    const container = document.getElementById('trait-scores-plot');
    if (!container || !tokenData.trait_scores_per_layer) return;

    // Get scores at layer 16
    const traits = Object.keys(tokenData.trait_scores_per_layer);
    const scores = traits.map(t => tokenData.trait_scores_per_layer[t][16]);

    // Sort by absolute value
    const sorted = traits.map((t, i) => ({ trait: t, score: scores[i] }))
        .sort((a, b) => Math.abs(b.score) - Math.abs(a.score));

    const traces = [{
        x: sorted.map(s => s.score),
        y: sorted.map(s => s.trait),
        type: 'bar',
        orientation: 'h',
        marker: {
            color: sorted.map(s => s.score > 0 ? '#4ecdc4' : '#ff6b6b')
        }
    }];

    const layout = {
        xaxis: { title: 'Projection Score', zeroline: true },
        yaxis: { automargin: true },
        margin: { l: 120, r: 20, t: 20, b: 40 },
        height: 220
    };

    Plotly.newPlot(container, traces, layout, { responsive: true });
}


function renderAttentionPattern(tokenData, data) {
    const container = document.getElementById('attention-plot');
    if (!container) return;

    if (!tokenData.attention_pattern_L16) {
        container.innerHTML = `<div style="padding: 20px; color: var(--text-secondary); text-align: center;">
            No attention data for this token<br>
            <small>(Only available for dynamic prompts)</small>
        </div>`;
        return;
    }

    const attnPattern = tokenData.attention_pattern_L16;  // [n_heads, context_size] or [1, context_size]
    const contextSize = tokenData.attention_context_size;
    const nHeads = attnPattern.length;

    // Average across heads (handles both 8-head and 1-head formats)
    const avgAttn = [];
    for (let i = 0; i < contextSize; i++) {
        let sum = 0;
        for (let h = 0; h < nHeads; h++) {
            sum += attnPattern[h][i] || 0;
        }
        avgAttn.push(sum / nHeads);
    }

    // Get token labels for context
    const contextTokens = data.tokens.slice(0, contextSize);

    const traces = [{
        x: contextTokens.map((t, i) => i),
        y: avgAttn,
        type: 'bar',
        marker: { color: '#9b59b6' },
        hovertext: contextTokens.map((t, i) => `${i}: "${t}" (${(avgAttn[i] * 100).toFixed(1)}%)`),
        hoverinfo: 'text'
    }];

    const layout = {
        xaxis: { title: 'Context Position', dtick: 10 },
        yaxis: { title: 'Attention Weight' },
        margin: { l: 50, r: 20, t: 20, b: 40 },
        height: 220
    };

    Plotly.newPlot(container, traces, layout, { responsive: true });
}


function renderAllTokensTraitHeatmap(data, currentTokenIdx) {
    const container = document.getElementById('all-tokens-trait-plot');
    if (!container || !data.per_token || data.per_token.length === 0) return;

    // Get trait names from first token
    const firstToken = data.per_token.find(t => t.trait_scores_per_layer);
    if (!firstToken || !firstToken.trait_scores_per_layer) return;

    const traits = Object.keys(firstToken.trait_scores_per_layer);

    // Build matrix: [n_tokens, n_traits] at layer 16
    const heatmapData = [];
    const tokenLabels = [];

    data.per_token.forEach((token, idx) => {
        if (!token.trait_scores_per_layer) return;

        const scores = traits.map(trait => token.trait_scores_per_layer[trait][16]);
        heatmapData.push(scores);

        // Truncate long tokens for display
        const tokenText = token.token.length > 10 ? token.token.slice(0, 10) + '…' : token.token;
        tokenLabels.push(`${idx}: ${tokenText}`);
    });

    // Transpose for Plotly (wants [traits, tokens])
    const transposed = traits.map((_, traitIdx) =>
        heatmapData.map(row => row[traitIdx])
    );

    const trace = {
        z: transposed,
        x: tokenLabels,
        y: traits,
        type: 'heatmap',
        colorscale: 'RdBu',
        zmid: 0,
        hovertemplate: '%{y}<br>Token %{x}<br>Score: %{z:.2f}<extra></extra>',
        showscale: true
    };

    // Add marker for current token
    const shapes = [{
        type: 'rect',
        x0: currentTokenIdx - 0.5,
        x1: currentTokenIdx + 0.5,
        y0: -0.5,
        y1: traits.length - 0.5,
        line: { color: '#ffff00', width: 3 },
        fillcolor: 'rgba(0,0,0,0)'
    }];

    const layout = {
        xaxis: {
            title: 'Token Position',
            tickangle: -45,
            tickfont: { size: 8 },
            // Show every 10th token for readability
            tickmode: 'linear',
            tick0: 0,
            dtick: 10
        },
        yaxis: {
            title: '',
            tickfont: { size: 10 },
            automargin: true
        },
        margin: { l: 100, r: 60, t: 20, b: 80 },
        height: 220,
        shapes: shapes
    };

    Plotly.newPlot(container, [trace], layout, { responsive: true });
}


function renderDistancePlot(tokenData, data) {
    const container = document.getElementById('distance-plot');
    if (!container || !tokenData.distance_to_others_L16) return;

    const distances = tokenData.distance_to_others_L16;

    // Color by phase
    const colors = distances.map((_, i) =>
        i < data.n_prompt_tokens ? '#3498db' : '#e74c3c'
    );

    const traces = [{
        x: data.tokens.map((t, i) => i),
        y: distances,
        type: 'bar',
        marker: { color: colors },
        hovertext: data.tokens.map((t, i) => `${i}: "${t}" (dist: ${distances[i].toFixed(3)})`),
        hoverinfo: 'text'
    }];

    const layout = {
        xaxis: { title: 'Token Position', dtick: 10 },
        yaxis: { title: 'Cosine Distance' },
        margin: { l: 50, r: 20, t: 20, b: 40 },
        height: 220,
        annotations: [{
            x: data.n_prompt_tokens,
            y: 1,
            xref: 'x',
            yref: 'paper',
            text: 'Response starts',
            showarrow: true,
            arrowhead: 2,
            ax: 0,
            ay: -30
        }]
    };

    Plotly.newPlot(container, traces, layout, { responsive: true });
}


function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}


// Add CSS
const tokenExplorerStyles = `
.token-explorer {
    padding: 16px;
}

.token-explorer-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border-color);
}

.token-label {
    font-size: 18px;
    font-weight: 600;
    color: var(--text-primary);
}

.token-phase-badge {
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 500;
    text-transform: uppercase;
}

.token-phase-badge.prompt {
    background: #3498db;
    color: white;
}

.token-phase-badge.response {
    background: #e74c3c;
    color: white;
}

.token-explorer-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 16px;
}

.explorer-panel {
    background: var(--surface-color);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 12px;
}

.explorer-panel.wide {
    grid-column: span 2;
}

.explorer-panel h4 {
    margin: 0 0 8px 0;
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
}
`;

if (!document.getElementById('token-explorer-styles')) {
    const styleSheet = document.createElement('style');
    styleSheet.id = 'token-explorer-styles';
    styleSheet.textContent = tokenExplorerStyles;
    document.head.appendChild(styleSheet);
}
