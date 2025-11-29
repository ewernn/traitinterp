/**
 * Zoomies Extraction Plugin
 * Explanations, fetchers, and renderers for extraction mode.
 */

window.zoomies = window.zoomies || {};

// Wait for registry to be ready
document.addEventListener('DOMContentLoaded', () => {
    const registry = window.zoomies.registry;
    if (!registry) {
        console.error('Registry not available for extraction plugin');
        return;
    }

    // =========================================================================
    // EXPLANATIONS
    // =========================================================================

    registry.explanation('extraction:all', {
        title: 'Extraction: All Layers',
        content: `
            Trait vectors are extracted at each layer using multiple methods.
            The heatmaps below show validation accuracy by layer and method.
            <br><br>
            <strong>Methods:</strong>
            <ul style="margin-top: 8px; padding-left: 20px;">
                <li><strong>Mean Diff</strong> - Simple mean difference between classes</li>
                <li><strong>Probe</strong> - Logistic regression weights</li>
                <li><strong>ICA</strong> - Independent component with best separation</li>
                <li><strong>Gradient</strong> - Optimized direction for separation</li>
            </ul>
            <br>
            Click a layer to focus on vectors extracted at that depth.
        `,
    });

    registry.explanation('extraction:layer', {
        title: (state) => `Extraction: Layer ${state.layerScope}`,
        content: (state) => {
            const layer = state.layerScope;
            const layerType = layer < 8 ? 'early' : layer < 18 ? 'middle' : 'late';
            const layerDesc = {
                early: 'processes token-level features and syntax',
                middle: 'encodes semantic meaning (best for trait generalization)',
                late: 'formats output and follows instructions',
            }[layerType];

            return `
                Layer ${layer} is a <strong>${layerType}</strong> layer that ${layerDesc}.
                <br><br>
                Below are the vectors extracted at this layer by each method,
                along with their validation accuracy and separation metrics.
            `;
        },
    });

    // =========================================================================
    // FETCHERS
    // =========================================================================

    registry.fetcher('extraction', async (state) => {
        const { experiment } = state;
        if (!experiment) return null;

        try {
            const resp = await fetch(`/experiments/${experiment}/extraction/extraction_evaluation.json`);
            if (!resp.ok) return null;
            return await resp.json();
        } catch (e) {
            console.error('Failed to fetch extraction data:', e);
            return null;
        }
    });

    // =========================================================================
    // RENDERERS
    // =========================================================================

    registry.renderer('extraction:all', (data, container, state) => {
        if (!data || !data.traits) {
            container.innerHTML = '<div class="no-data">No extraction data available</div>';
            return;
        }

        const selectedTraits = state.selectedTraits;
        const traits = Object.keys(data.traits).filter(t =>
            selectedTraits.length === 0 || selectedTraits.includes(t)
        );

        if (traits.length === 0) {
            container.innerHTML = '<div class="no-data">No traits selected</div>';
            return;
        }

        // Build layer×method heatmap for each trait
        let html = '<div class="extraction-grids">';

        traits.forEach(trait => {
            const traitData = data.traits[trait];
            if (!traitData || !traitData.methods) return;

            const displayName = window.zoomies.formatTraitName(trait);

            html += `
                <div class="extraction-grid">
                    <h3>${displayName}</h3>
                    <div class="heatmap-container" id="heatmap-${trait.replace(/\//g, '-')}"></div>
                </div>
            `;
        });

        html += '</div>';
        container.innerHTML = html;

        // Render heatmaps with Plotly
        traits.forEach(trait => {
            const traitData = data.traits[trait];
            if (!traitData || !traitData.methods) return;

            const containerId = `heatmap-${trait.replace(/\//g, '-')}`;
            const heatmapEl = document.getElementById(containerId);
            if (!heatmapEl) return;

            renderExtractionHeatmap(heatmapEl, traitData);
        });
    });

    registry.renderer('extraction:layer', (data, container, state) => {
        if (!data || !data.traits) {
            container.innerHTML = '<div class="no-data">No extraction data available</div>';
            return;
        }

        const layer = state.layerScope;
        const selectedTraits = state.selectedTraits;
        const traits = Object.keys(data.traits).filter(t =>
            selectedTraits.length === 0 || selectedTraits.includes(t)
        );

        // Show vectors for this specific layer
        let html = `<div class="layer-vectors">`;
        html += `<h3>Vectors at Layer ${layer}</h3>`;

        html += '<table class="vector-table"><thead><tr>';
        html += '<th>Trait</th><th>Method</th><th>Val Accuracy</th><th>Separation</th>';
        html += '</tr></thead><tbody>';

        traits.forEach(trait => {
            const traitData = data.traits[trait];
            if (!traitData || !traitData.methods) return;

            const displayName = window.zoomies.formatTraitName(trait);

            Object.entries(traitData.methods).forEach(([method, methodData]) => {
                const layerMetrics = methodData.layers?.[layer];
                if (!layerMetrics) return;

                const acc = (layerMetrics.val_accuracy * 100).toFixed(1);
                const sep = layerMetrics.separation?.toFixed(2) || '-';

                html += `<tr>
                    <td>${displayName}</td>
                    <td>${method}</td>
                    <td>${acc}%</td>
                    <td>${sep}</td>
                </tr>`;
            });
        });

        html += '</tbody></table></div>';
        container.innerHTML = html;
    });
});

/**
 * Render extraction heatmap (layer × method)
 */
function renderExtractionHeatmap(container, traitData) {
    const methods = Object.keys(traitData.methods).filter(m =>
        ['mean_diff', 'probe', 'ica', 'gradient'].includes(m)
    );
    const layers = window.zoomies.LAYERS;

    // Build z-matrix (methods × layers)
    const z = methods.map(method => {
        const methodData = traitData.methods[method];
        return Array.from({ length: layers }, (_, l) => {
            const layerData = methodData.layers?.[l];
            return layerData?.val_accuracy ?? 0.5;
        });
    });

    const trace = {
        z: z,
        x: Array.from({ length: layers }, (_, i) => i),
        y: methods,
        type: 'heatmap',
        colorscale: [
            [0, '#4285f4'],
            [0.5, '#ffffff'],
            [1, '#ea4335']
        ],
        zmin: 0.3,
        zmax: 1.0,
        hovertemplate: 'Layer %{x}<br>Method: %{y}<br>Accuracy: %{z:.1%}<extra></extra>',
    };

    const layout = {
        margin: { t: 20, b: 40, l: 80, r: 20 },
        height: 150,
        xaxis: { title: 'Layer', dtick: 5 },
        yaxis: { title: '' },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
    };

    Plotly.newPlot(container, [trace], layout, { responsive: true });

    // Add click handler for layer selection
    container.on('plotly_click', (eventData) => {
        const layer = eventData.points[0].x;
        window.zoomies.setState({ layerScope: layer });
    });
}

// Add extraction-specific styles
(function() {
const style = document.createElement('style');
style.textContent = `
    .extraction-grids {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
        gap: 24px;
    }
    .extraction-grid h3 {
        font-size: var(--text-base);
        margin-bottom: 8px;
        color: var(--text-primary);
    }
    .heatmap-container {
        width: 100%;
        min-height: 150px;
    }
    .layer-vectors {
        padding: 16px;
    }
    .layer-vectors h3 {
        margin-bottom: 16px;
    }
    .vector-table {
        width: 100%;
        border-collapse: collapse;
    }
    .vector-table th,
    .vector-table td {
        padding: 8px 12px;
        text-align: left;
        border-bottom: 1px solid var(--border-color);
    }
    .vector-table th {
        background: var(--bg-secondary);
        font-weight: 500;
    }
    .vector-table tr:hover {
        background: var(--bg-secondary);
    }
`;
document.head.appendChild(style);
})();
