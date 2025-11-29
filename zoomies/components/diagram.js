/**
 * Zoomies Diagram
 * Minimalist transformer visualization with zoom transitions.
 *
 * Zoom levels:
 *   0: All layers (show layer 25, ..., layer 0)
 *   1: Single layer (show attention + MLP + residual connections)
 *   2: Component (attention heads OR MLP neurons) - future
 */

window.zoomies = window.zoomies || {};

const TRANSITION_MS = 400;

/**
 * Render the diagram based on current zoom level
 */
window.zoomies.renderDiagram = function(container, props = {}) {
    const {
        layerScope = 'all',
        componentScope = 'all',  // 'all' | 'attention' | 'mlp'
        onLayerClick = () => {},
        onComponentClick = () => {},
        onBackClick = () => {},
    } = props;

    // Determine zoom level
    if (layerScope === 'all') {
        renderAllLayers(container, { onLayerClick });
    } else if (componentScope === 'all') {
        renderSingleLayer(container, {
            layer: layerScope,
            onComponentClick,
            onBackClick
        });
    } else {
        renderComponent(container, {
            layer: layerScope,
            component: componentScope,
            onBackClick
        });
    }
};

/**
 * Zoom level 0: All layers view
 * Shows: Layer 25, ..., Layer 0 (simplified)
 */
function renderAllLayers(container, { onLayerClick }) {
    const html = `
        <div class="diagram-zoom-container" data-zoom="out">
            <div class="diagram-stack">
                <!-- Output (unembedding) -->
                <div class="diagram-layer static">
                    <span class="layer-label">Output</span>
                    <div class="layer-block output"></div>
                </div>

                <!-- Layer 25 (not clickable) -->
                <div class="diagram-layer static">
                    <span class="layer-label">Layer 25</span>
                    <div class="layer-block"></div>
                </div>

                <div class="diagram-dots">
                    <span>⋮</span>
                    <span>⋮</span>
                    <span>⋮</span>
                </div>

                <!-- Layer 0 (clickable) -->
                <div class="diagram-layer clickable" data-layer="0">
                    <span class="layer-label">Layer 0</span>
                    <div class="layer-block"></div>
                </div>

                <!-- Input (embedding) -->
                <div class="diagram-layer static">
                    <span class="layer-label">Embed</span>
                    <div class="layer-block embed"></div>
                </div>
            </div>
            <p class="diagram-hint">Click Layer 0 to explore</p>
        </div>
    `;

    container.innerHTML = html;

    // Add click handlers
    container.querySelectorAll('.diagram-layer.clickable').forEach(el => {
        el.addEventListener('click', () => {
            const layer = parseInt(el.dataset.layer, 10);

            // Trigger zoom animation
            const zoomContainer = container.querySelector('.diagram-zoom-container');
            zoomContainer.classList.add('zooming-in');

            setTimeout(() => {
                onLayerClick(layer);
            }, TRANSITION_MS);
        });
    });
}

/**
 * Zoom level 1: Single layer view
 * Shows: Attention block + MLP block + residual connections
 */
function renderSingleLayer(container, { layer, onComponentClick, onBackClick }) {
    const html = `
        <div class="diagram-zoom-container" data-zoom="in">
            <button class="diagram-back" title="Back to all layers">
                ← All Layers
            </button>

            <div class="diagram-layer-detail">
                <h3 class="layer-title">Layer ${layer}</h3>

                <div class="transformer-block">
                    <!-- Output to next layer -->
                    <div class="residual-stream">
                        <span class="stream-label">to layer ${layer + 1}</span>
                        <div class="stream-arrow">↑</div>
                    </div>

                    <!-- Add node (after MLP) -->
                    <div class="add-node">+</div>

                    <!-- MLP block (top) -->
                    <div class="component-row">
                        <div class="skip-connection left"></div>
                        <div class="component clickable" data-component="mlp">
                            <span>MLP</span>
                        </div>
                        <div class="skip-connection right"></div>
                    </div>

                    <!-- Add node (after attention) -->
                    <div class="add-node">+</div>

                    <!-- Attention block (bottom) -->
                    <div class="component-row">
                        <div class="skip-connection left"></div>
                        <div class="component clickable" data-component="attention">
                            <span>Attention</span>
                        </div>
                        <div class="skip-connection right"></div>
                    </div>

                    <!-- Input from previous layer -->
                    <div class="residual-stream">
                        <div class="stream-arrow">↑</div>
                        <span class="stream-label">${layer === 0 ? 'from embed' : 'from layer ' + (layer - 1)}</span>
                    </div>
                </div>
            </div>
        </div>
    `;

    container.innerHTML = html;

    // Trigger entrance animation
    requestAnimationFrame(() => {
        container.querySelector('.diagram-zoom-container').classList.add('zoomed');
    });

    // Back button
    container.querySelector('.diagram-back').addEventListener('click', () => {
        const zoomContainer = container.querySelector('.diagram-zoom-container');
        zoomContainer.classList.add('zooming-out');

        setTimeout(() => {
            onBackClick();
        }, TRANSITION_MS);
    });

    // Component clicks
    container.querySelectorAll('.component.clickable').forEach(el => {
        el.addEventListener('click', () => {
            const component = el.dataset.component;
            onComponentClick(component);
        });
    });
}

/**
 * Zoom level 2: Component view (future)
 * Shows: Attention heads OR MLP neurons
 */
function renderComponent(container, { layer, component, onBackClick }) {
    const html = `
        <div class="diagram-zoom-container" data-zoom="component">
            <button class="diagram-back" title="Back to layer">
                ← Layer ${layer}
            </button>

            <div class="diagram-component-detail">
                <h3 class="component-title">${component === 'attention' ? 'Attention' : 'MLP'}</h3>
                <p class="coming-soon">Detailed view coming soon...</p>
            </div>
        </div>
    `;

    container.innerHTML = html;

    container.querySelector('.diagram-back').addEventListener('click', onBackClick);
}
