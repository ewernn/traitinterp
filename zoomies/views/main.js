/**
 * Zoomies Main View
 * Hero section (100vh) with zoom navigation + data section below the fold.
 */

window.zoomies = window.zoomies || {};

// Gemma 2B dimensions
const MODEL_DIMS = {
    hidden_size: 2304,
    num_layers: 26,
    num_heads: 8,
    num_kv_heads: 4,
    head_dim: 256,
    intermediate_size: 9216,
    vocab_size: 256000,
};

/**
 * Render the main view
 */
window.zoomies.renderMainView = function(container) {
    const state = window.zoomies.state;

    // Two-section layout: hero (100vh) + data (below fold)
    container.innerHTML = `
        <div class="main-view">
            <!-- Hero: Full viewport, split layout -->
            <section class="hero-section">
                <div class="hero-selectors" id="hero-selectors"></div>
                <div class="hero-split">
                    <div class="hero-left">
                        <div class="diagram-container" id="diagram-container"></div>
                    </div>
                    <div class="hero-right">
                        <div class="info-panel" id="info-panel"></div>
                    </div>
                </div>
                <div class="scroll-hint" id="scroll-hint">
                    <span>↓ scroll for data</span>
                </div>
            </section>

            <!-- Data: Below the fold -->
            <section class="data-section" id="data-section">
                <div class="data-header">
                    <div class="data-explanation" id="data-explanation"></div>
                </div>
                <div class="data-content" id="data-content">
                    <div class="loading">Loading...</div>
                </div>
            </section>
        </div>
    `;

    // Render selectors
    const selectorsEl = document.getElementById('hero-selectors');
    if (selectorsEl && window.zoomies.renderSelectors) {
        window.zoomies.renderSelectors(selectorsEl);
    }

    // Render diagram
    renderDiagram(state);

    // Render info panel
    renderInfoPanel(state);

    // Render data section (async)
    renderDataSection(state);

    // Hide scroll hint when scrolled
    setupScrollHint();
};

/**
 * Render the zoom diagram
 */
function renderDiagram(state) {
    const diagramEl = document.getElementById('diagram-container');
    if (!diagramEl || !window.zoomies.renderDiagram) return;

    window.zoomies.renderDiagram(diagramEl, {
        layerScope: state.layerScope,
        componentScope: state.componentScope || 'all',
        onLayerClick: (layer) => {
            window.zoomies.setState({ layerScope: layer });
        },
        onComponentClick: (component) => {
            window.zoomies.setState({ componentScope: component });
        },
        onBackClick: () => {
            if (state.componentScope && state.componentScope !== 'all') {
                window.zoomies.setState({ componentScope: 'all' });
            } else {
                // Reset both layer and token scope when going back to all layers
                window.zoomies.setState({ layerScope: 'all', tokenScope: 'all' });
            }
        },
    });
}

/**
 * Render the info panel with tensor dimensions
 */
function renderInfoPanel(state) {
    const panelEl = document.getElementById('info-panel');
    if (!panelEl) return;

    const { layerScope, componentScope, mode } = state;
    const d = MODEL_DIMS;

    // Build explanation based on current granularity
    let explanation = '';
    let tensors = [];

    if (layerScope === 'all') {
        // All layers view
        explanation = mode === 'inference'
            ? 'Residual stream flows through all 26 layers, transforming token representations.'
            : 'Trait vectors are extracted at each layer to find where behaviors are encoded.';
        tensors = [
            { name: 'Residual Stream', shape: `[seq, ${d.hidden_size}]`, desc: 'Hidden states at each position' },
            { name: 'Embedding', shape: `[${d.vocab_size}, ${d.hidden_size}]`, desc: 'Token → vector lookup' },
            { name: 'Unembedding', shape: `[${d.hidden_size}, ${d.vocab_size}]`, desc: 'Vector → logits' },
        ];
    } else if (componentScope === 'all') {
        // Single layer view
        explanation = `Layer ${layerScope}: Attention reads from other positions, MLP transforms each position independently.`;
        tensors = [
            { name: 'Input', shape: `[seq, ${d.hidden_size}]`, desc: 'From previous layer' },
            { name: 'Attention Out', shape: `[seq, ${d.hidden_size}]`, desc: 'After attention + residual' },
            { name: 'MLP Out', shape: `[seq, ${d.hidden_size}]`, desc: 'After MLP + residual' },
        ];
    } else if (componentScope === 'attention') {
        // Attention detail
        explanation = `Attention uses ${d.num_heads} query heads and ${d.num_kv_heads} key/value heads (grouped query attention).`;
        tensors = [
            { name: 'Q projection', shape: `[${d.hidden_size}, ${d.num_heads * d.head_dim}]`, desc: `${d.num_heads} heads × ${d.head_dim} dim` },
            { name: 'K projection', shape: `[${d.hidden_size}, ${d.num_kv_heads * d.head_dim}]`, desc: `${d.num_kv_heads} KV heads` },
            { name: 'V projection', shape: `[${d.hidden_size}, ${d.num_kv_heads * d.head_dim}]`, desc: `${d.num_kv_heads} KV heads` },
            { name: 'Attn weights', shape: `[${d.num_heads}, seq, seq]`, desc: 'Per-head attention pattern' },
            { name: 'Output proj', shape: `[${d.num_heads * d.head_dim}, ${d.hidden_size}]`, desc: 'Combine heads' },
        ];
    } else if (componentScope === 'mlp') {
        // MLP detail
        explanation = `MLP expands to ${d.intermediate_size} dimensions with gated activation, then projects back.`;
        tensors = [
            { name: 'Gate proj', shape: `[${d.hidden_size}, ${d.intermediate_size}]`, desc: 'Gating pathway' },
            { name: 'Up proj', shape: `[${d.hidden_size}, ${d.intermediate_size}]`, desc: 'Value pathway' },
            { name: 'Down proj', shape: `[${d.intermediate_size}, ${d.hidden_size}]`, desc: 'Back to residual' },
        ];
    }

    // Render panel
    panelEl.innerHTML = `
        <div class="info-explanation">${explanation}</div>
        <div class="info-tensors">
            <h4>Tensor Shapes</h4>
            ${tensors.map(t => `
                <div class="tensor-row">
                    <span class="tensor-name">${t.name}</span>
                    <code class="tensor-shape">${t.shape}</code>
                    <span class="tensor-desc">${t.desc}</span>
                </div>
            `).join('')}
        </div>
        <div class="info-constants">
            <h4>Model Constants</h4>
            <div class="constants-grid">
                <span>d_model</span><code>${d.hidden_size}</code>
                <span>layers</span><code>${d.num_layers}</code>
                <span>heads</span><code>${d.num_heads}</code>
                <span>kv_heads</span><code>${d.num_kv_heads}</code>
                <span>head_dim</span><code>${d.head_dim}</code>
                <span>d_ff</span><code>${d.intermediate_size}</code>
            </div>
        </div>
    `;
}

/**
 * Render the data section based on current position
 */
async function renderDataSection(state) {
    const explanationEl = document.getElementById('data-explanation');
    const contentEl = document.getElementById('data-content');
    const scrollHint = document.getElementById('scroll-hint');

    if (!explanationEl || !contentEl) return;

    const registry = window.zoomies.registry;
    const positionKey = window.zoomies.getPositionKey();

    // Get explanation
    const explanation = registry.getExplanation(positionKey, state);
    if (explanation) {
        explanationEl.innerHTML = `
            <h2 class="explanation-title">${explanation.title}</h2>
            <div class="explanation-content">${explanation.content}</div>
        `;
    } else {
        explanationEl.innerHTML = `
            <h2 class="explanation-title">Explore</h2>
            <div class="explanation-content">
                Click on elements in the diagram above to navigate.
            </div>
        `;
    }

    // Get fetcher and renderer
    const fetcher = registry.getFetcher(positionKey);
    const renderer = registry.getRenderer(positionKey);

    if (!fetcher || !renderer) {
        contentEl.innerHTML = `
            <div class="no-data">
                <p>No data view for this position yet.</p>
                <p style="margin-top: 8px; font-size: var(--text-xs); color: var(--text-tertiary);">
                    Position: ${positionKey}
                </p>
            </div>
        `;
        // Hide scroll hint if no data
        if (scrollHint) scrollHint.style.opacity = '0.3';
        return;
    }

    // Show scroll hint
    if (scrollHint) scrollHint.style.opacity = '1';

    // Fetch data
    contentEl.innerHTML = '<div class="loading">Loading data...</div>';

    try {
        const data = await fetcher(state);

        if (!data) {
            // Show helpful debug info
            const { experiment, selectedTraits, promptSet, promptId, mode } = state;
            contentEl.innerHTML = `
                <div class="no-data">
                    <p>No data available for current selection.</p>
                    <div style="margin-top: 16px; font-size: var(--text-xs); color: var(--text-tertiary); text-align: left; max-width: 400px; margin-left: auto; margin-right: auto;">
                        <p><strong>Current state:</strong></p>
                        <ul style="margin-top: 8px; padding-left: 20px;">
                            <li>Mode: ${mode || 'not set'}</li>
                            <li>Experiment: ${experiment || 'not set'}</li>
                            <li>Traits: ${selectedTraits?.length ? selectedTraits.join(', ') : 'none selected'}</li>
                            <li>Prompt set: ${promptSet || 'not set'}</li>
                            <li>Prompt ID: ${promptId || 'not set'}</li>
                            <li>Position: ${positionKey}</li>
                        </ul>
                        <p style="margin-top: 12px;">Check the browser console for more details.</p>
                    </div>
                </div>
            `;
            return;
        }

        // Render the data
        renderer(data, contentEl, state);

    } catch (e) {
        console.error('Failed to load data:', e);
        contentEl.innerHTML = `
            <div class="error">
                Failed to load data: ${e.message}
            </div>
        `;
    }
}

/**
 * Setup scroll hint behavior
 */
function setupScrollHint() {
    const scrollHint = document.getElementById('scroll-hint');
    const heroSection = document.querySelector('.hero-section');

    if (!scrollHint || !heroSection) return;

    // Click to scroll
    scrollHint.addEventListener('click', () => {
        const dataSection = document.getElementById('data-section');
        if (dataSection) {
            dataSection.scrollIntoView({ behavior: 'smooth' });
        }
    });

    // Fade hint when scrolled
    const observer = new IntersectionObserver(
        (entries) => {
            entries.forEach(entry => {
                scrollHint.style.opacity = entry.isIntersecting ? '1' : '0';
            });
        },
        { threshold: 0.9 }
    );

    observer.observe(heroSection);
}
