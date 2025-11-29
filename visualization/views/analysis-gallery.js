// Analysis Gallery View - Unified live-rendered analysis with token slider support
// Replaces static PNGs with interactive Plotly visualizations

let galleryData = null;
let galleryCache = { experiment: null, promptSet: null, promptId: null };

// =============================================================================
// DATA LOADING
// =============================================================================

async function loadGalleryData() {
    const experiment = window.state.experimentData?.name;
    const promptSet = window.state.currentPromptSet;
    const promptId = window.state.currentPromptId;

    if (!experiment || !promptSet || !promptId) return null;

    // Check cache
    if (galleryCache.experiment === experiment &&
        galleryCache.promptSet === promptSet &&
        galleryCache.promptId === promptId &&
        galleryData) {
        return galleryData;
    }

    const url = window.paths.analysisPerToken(promptSet, promptId);

    try {
        const response = await fetch(url);
        if (!response.ok) {
            console.error('Failed to load per-token data:', response.status);
            return null;
        }
        galleryData = await response.json();
        galleryCache = { experiment, promptSet, promptId };
        return galleryData;
    } catch (error) {
        console.error('Error loading per-token data:', error);
        return null;
    }
}

// =============================================================================
// MAIN RENDER
// =============================================================================

async function renderAnalysisGallery() {
    const contentArea = document.getElementById('content-area');
    const experiment = window.state.experimentData?.name;

    if (!experiment) {
        contentArea.innerHTML = '<div class="error">No experiment selected</div>';
        return;
    }

    // Check if DOM exists and data is cached (avoid scroll reset on slider move)
    const existingGallery = contentArea.querySelector('.analysis-gallery');
    const dataIsCached = galleryCache.experiment === experiment &&
                         galleryCache.promptSet === window.state.currentPromptSet &&
                         galleryCache.promptId === window.state.currentPromptId &&
                         galleryData;

    if (!dataIsCached) {
        contentArea.innerHTML = '<div class="loading">Loading analysis data...</div>';
    }

    const data = await loadGalleryData();

    if (!data) {
        contentArea.innerHTML = `
            <div class="info" style="margin: 16px; padding: 16px;">
                <h3>No per-token data available</h3>
                <p>Run the per-token analysis script first:</p>
                <code>python experiments/${experiment}/analysis/compute_per_token_all_sets.py</code>
            </div>
        `;
        return;
    }

    const tokenIdx = Math.min(window.state.currentTokenIndex || 0, data.n_total_tokens - 1);
    const tokenData = data.per_token[tokenIdx];

    // If DOM exists and data cached, just update visualizations
    if (existingGallery && dataIsCached) {
        updateGalleryVisualizations(data, tokenIdx, tokenData);
        return;
    }

    // Full render
    contentArea.innerHTML = `
        <div class="tool-view analysis-gallery">
            <div class="page-intro">
                <div class="page-intro-text">Per-token activation dynamics.</div>
            </div>

            <section>
                <h3>Activation Velocity</h3>
                <p class="section-desc">How fast each token's hidden state changes between layers. Yellow = selected token.</p>
                <div class="card">
                    <div id="velocity-heatmap-container"></div>
                </div>
            </section>

            <section>
                <h3>Activation-Trait Coupling</h3>
                <p class="section-desc">Correlation between activation velocity and trait change magnitude (across all tokens and layers).</p>
                <div class="card">
                    <div id="dynamics-correlation-container"></div>
                </div>
            </section>

            ${getCategoryReference()}
        </div>
    `;

    // Render all visualizations
    renderAllVisualizations(data, tokenIdx, tokenData);
}

function updateGalleryVisualizations(data, tokenIdx, tokenData) {
    // Re-render all (Plotly handles updates efficiently)
    renderAllVisualizations(data, tokenIdx, tokenData);
}

function renderAllVisualizations(data, tokenIdx, tokenData) {
    renderVelocityHeatmap(data, tokenIdx);
    renderDynamicsCorrelation(data);
}

// =============================================================================
// ALL TOKENS VISUALIZATIONS (slider highlights)
// =============================================================================

function renderVelocityHeatmap(data, currentTokenIdx) {
    const container = document.getElementById('velocity-heatmap-container');
    if (!container) return;

    // Build matrix [tokens × layer_transitions]
    const nLayers = 25; // transitions
    const zData = data.per_token.map(t => t.normalized_velocity_per_layer || new Array(nLayers).fill(0));

    const trace = {
        z: zData,
        x: Array.from({ length: nLayers }, (_, i) => i),
        y: data.tokens.map((t, i) => i),
        type: 'heatmap',
        colorscale: 'Viridis',
        hovertemplate: 'Token %{y}, Layer %{x}→%{x+1}<br>Velocity: %{z:.3f}<extra></extra>',
        showscale: true,
        colorbar: { thickness: 15, len: 0.8 }
    };

    // Highlight current token row
    const shapes = [{
        type: 'rect',
        x0: -0.5,
        x1: nLayers - 0.5,
        y0: currentTokenIdx - 0.5,
        y1: currentTokenIdx + 0.5,
        line: { color: '#ffff00', width: 2 },
        fillcolor: 'rgba(0,0,0,0)'
    }];

    const layout = window.getPlotlyLayout({
        margin: { l: 50, r: 50, t: 10, b: 40 },
        height: 250,
        xaxis: { title: 'Layer Transition', dtick: 5 },
        yaxis: { title: 'Token', dtick: 10 },
        shapes
    });

    Plotly.newPlot(container, [trace], layout, { responsive: true });
}

// =============================================================================
// AGGREGATE VISUALIZATIONS
// =============================================================================

function renderDynamicsCorrelation(data) {
    const container = document.getElementById('dynamics-correlation-container');
    if (!container) return;

    const firstToken = data.per_token.find(t => t.trait_scores_per_layer);
    if (!firstToken) {
        container.innerHTML = '<div class="no-data">No trait data</div>';
        return;
    }

    const traits = Object.keys(firstToken.trait_scores_per_layer);

    // For each trait, compute correlation between normalized velocity and |trait velocity|
    const correlations = traits.map(trait => {
        const velocities = [];
        const traitVelocities = [];

        data.per_token.forEach(t => {
            if (!t.normalized_velocity_per_layer || !t.trait_scores_per_layer?.[trait]) return;

            const traitScores = t.trait_scores_per_layer[trait];
            // Trait velocity = diff of trait scores across layers
            for (let i = 0; i < traitScores.length - 1; i++) {
                velocities.push(t.normalized_velocity_per_layer[i] || 0);
                traitVelocities.push(Math.abs(traitScores[i + 1] - traitScores[i]));
            }
        });

        if (velocities.length < 2) return { trait, corr: 0 };

        // Pearson correlation
        const n = velocities.length;
        const sumX = velocities.reduce((a, b) => a + b, 0);
        const sumY = traitVelocities.reduce((a, b) => a + b, 0);
        const sumXY = velocities.reduce((sum, x, i) => sum + x * traitVelocities[i], 0);
        const sumX2 = velocities.reduce((sum, x) => sum + x * x, 0);
        const sumY2 = traitVelocities.reduce((sum, y) => sum + y * y, 0);

        const num = n * sumXY - sumX * sumY;
        const den = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
        const corr = den === 0 ? 0 : num / den;

        return { trait, corr };
    });

    // Sort by correlation (descending)
    correlations.sort((a, b) => b.corr - a.corr);

    const trace = {
        x: correlations.map(c => c.corr),
        y: correlations.map(c => c.trait),
        type: 'bar',
        orientation: 'h',
        marker: {
            color: correlations.map(c => c.corr > 0.3 ? '#27ae60' : c.corr > 0.1 ? '#f39c12' : '#95a5a6')
        },
        hovertemplate: '%{y}: r = %{x:.3f}<extra></extra>'
    };

    const layout = window.getPlotlyLayout({
        margin: { l: 100, r: 20, t: 10, b: 40 },
        height: 250,
        xaxis: { title: 'Correlation (r)', range: [-0.2, 1] },
        yaxis: { tickfont: { size: 10 } }
    });

    Plotly.newPlot(container, [trace], layout, { responsive: true });
}

// =============================================================================
// REFERENCE SECTION
// =============================================================================

function getCategoryReference() {
    return `
        <div class="category-reference">
            <h3>Reference</h3>

            <details>
                <summary>Activation Velocity</summary>
                <p>How fast each token's hidden state vector changes between layers (trait-independent).</p>
                <p><strong>Math:</strong> velocity[L] = ||h[L+1] - h[L]|| / ||h[L]||</p>
                <p><strong>Read:</strong> Bright = major transformation. Typical pattern: high early (L0-6), low middle (L7-22), high late (L23-24).</p>
            </details>

            <details>
                <summary>Activation-Trait Coupling</summary>
                <p>Are trait projection changes correlated with overall hidden state changes?</p>
                <p><strong>Math:</strong> For each trait, Pearson(activation_velocity, |Δtrait|) across all (token, layer) pairs.</p>
                <p><strong>Read:</strong> High r (green) = trait changes happen when hidden state changes most. Low r (gray) = trait changes independently.</p>
            </details>
        </div>
    `;
}

// =============================================================================
// UTILITIES
// =============================================================================

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
