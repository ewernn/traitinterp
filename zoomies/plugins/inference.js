/**
 * Zoomies Inference Plugin
 * Explanations, fetchers, and renderers for inference mode.
 */

window.zoomies = window.zoomies || {};

// Cache for inference data per trait
let inferenceCache = {};

// Wait for registry to be ready
document.addEventListener('DOMContentLoaded', () => {
    const registry = window.zoomies.registry;
    if (!registry) {
        console.error('Registry not available for inference plugin');
        return;
    }

    // =========================================================================
    // EXPLANATIONS
    // =========================================================================

    registry.explanation('inference:all:all', {
        title: 'Inference: Full Trajectory',
        content: `
            The heatmap shows trait activation across all tokens (x-axis) and
            all layers (y-axis). Each cell is the projection of the hidden state
            onto the trait vector.
            <br><br>
            <strong>Reading the heatmap:</strong>
            <ul style="margin-top: 8px; padding-left: 20px;">
                <li><span style="color: #ea4335;">Red</span> = high trait activation (expressing the trait)</li>
                <li><span style="color: #4285f4;">Blue</span> = low trait activation (opposite of trait)</li>
                <li>White = neutral</li>
            </ul>
            <br>
            Click a layer to see the line chart over tokens.
            Click a token to see the vertical trajectory across layers.
        `,
    });

    registry.explanation('inference:all:layer', {
        title: (state) => `Inference: Layer ${state.layerScope} Dynamics`,
        content: (state) => `
            Line chart showing trait activation over tokens at layer ${state.layerScope}.
            <br><br>
            Each line represents a selected trait. Watch for:
            <ul style="margin-top: 8px; padding-left: 20px;">
                <li><strong>Commitment points</strong> - where traits lock in</li>
                <li><strong>Velocity</strong> - rapid changes indicate decisions</li>
                <li><strong>Prompt/response boundary</strong> - marked by vertical line</li>
            </ul>
        `,
    });

    registry.explanation('inference:token:all', {
        title: (state) => `Inference: Token ${state.tokenScope} Trajectory`,
        content: `
            Vertical bar chart showing trait activation across all layers for this token.
            <br><br>
            This reveals <strong>where</strong> in the model the trait is computed.
            Early layers often show noise; middle/late layers show the decision.
        `,
    });

    registry.explanation('inference:token:layer', {
        title: (state) => `Microscope: Token ${state.tokenScope}, Layer ${state.layerScope}`,
        content: `
            Deep dive into a single position in the residual stream.
            <br><br>
            <strong>Available analysis:</strong>
            <ul style="margin-top: 8px; padding-left: 20px;">
                <li>Raw activation values</li>
                <li>Top SAE features (layer 16 only)</li>
                <li>Attention patterns (coming soon)</li>
            </ul>
        `,
    });

    // =========================================================================
    // FETCHERS
    // =========================================================================

    registry.fetcher('inference', async (state) => {
        const { experiment, selectedTraits, promptSet, promptId } = state;

        // Validate required state
        if (!experiment || !selectedTraits || selectedTraits.length === 0) {
            console.log('Inference fetcher: missing experiment or traits');
            return null;
        }
        if (!promptSet || !promptId) {
            console.log('Inference fetcher: missing promptSet or promptId');
            return null;
        }

        console.log('Inference fetcher: loading data for', { experiment, selectedTraits, promptSet, promptId });

        // Make sure paths has the experiment set
        if (window.zoomies.paths) {
            window.zoomies.paths.setExperiment(experiment);
        }

        // Fetch data for each selected trait
        const results = {};

        for (const trait of selectedTraits) {
            const cacheKey = `${experiment}:${trait}:${promptSet}:${promptId}`;

            if (inferenceCache[cacheKey]) {
                results[trait] = inferenceCache[cacheKey];
                continue;
            }

            try {
                // Use paths helper if available, otherwise hardcoded path
                let path;
                if (window.zoomies.paths && window.zoomies.paths.residualStreamData) {
                    path = window.zoomies.paths.residualStreamData(trait, promptSet, promptId);
                } else {
                    path = `/experiments/${experiment}/inference/${trait}/residual_stream/${promptSet}/${promptId}.json`;
                }

                console.log(`Fetching inference data: ${path}`);
                const resp = await fetch(path);
                if (!resp.ok) {
                    console.warn(`Failed to fetch ${path}: ${resp.status}`);
                    continue;
                }
                const data = await resp.json();
                inferenceCache[cacheKey] = data;
                results[trait] = data;
            } catch (e) {
                console.warn(`Failed to fetch inference data for ${trait}:`, e);
            }
        }

        if (Object.keys(results).length === 0) {
            console.log('Inference fetcher: no data loaded');
            return null;
        }

        // Combine into unified structure
        // Get tokens and prompt info from first result
        const firstTrait = Object.keys(results)[0];
        const firstData = results[firstTrait];

        // Handle different data formats (response.tokens vs tokens)
        const tokens = firstData.tokens || firstData.response?.tokens || [];
        const promptTokenCount = firstData.prompt_token_count || firstData.prompt?.tokens?.length || 0;

        console.log('Inference fetcher: loaded', Object.keys(results).length, 'traits,', tokens.length, 'tokens');

        return {
            tokens,
            prompt_token_count: promptTokenCount,
            traits: results,
        };
    });

    // =========================================================================
    // RENDERERS
    // =========================================================================

    registry.renderer('inference:all:all', (data, container, state) => {
        if (!data || !data.traits) {
            container.innerHTML = '<div class="no-data">No inference data available</div>';
            return;
        }

        const traits = Object.keys(data.traits);

        let html = '<div class="inference-heatmaps">';

        traits.forEach(trait => {
            const displayName = window.zoomies.formatTraitName(trait);
            html += `
                <div class="inference-heatmap">
                    <h3>${displayName}</h3>
                    <div id="heatmap-inf-${trait.replace(/\//g, '-')}" class="plot-container"></div>
                </div>
            `;
        });

        html += '</div>';
        container.innerHTML = html;

        // Render heatmaps
        traits.forEach(trait => {
            const traitData = data.traits[trait];
            const containerId = `heatmap-inf-${trait.replace(/\//g, '-')}`;
            const el = document.getElementById(containerId);
            if (!el) return;

            renderInferenceHeatmap(el, traitData);
        });
    });

    registry.renderer('inference:all:layer', (data, container, state) => {
        if (!data || !data.traits) {
            container.innerHTML = '<div class="no-data">No inference data available</div>';
            return;
        }

        const layer = state.layerScope;
        const traits = Object.keys(data.traits);

        container.innerHTML = `
            <div id="line-chart" class="plot-container" style="height: 400px;"></div>
        `;

        // Get tokens from first trait
        const firstTraitData = data.traits[traits[0]];
        const promptTokens = firstTraitData.prompt?.tokens || [];
        const responseTokens = firstTraitData.response?.tokens || [];
        const allTokens = [...promptTokens, ...responseTokens];
        const nPromptTokens = promptTokens.length;

        // Build traces for each trait
        const traces = traits.map((trait, i) => {
            const traitData = data.traits[trait];
            const projections = traitData.projections || {};
            const promptProj = projections.prompt || [];
            const responseProj = projections.response || [];
            const allProj = [...promptProj, ...responseProj];

            // Get data for this layer (average over methods)
            const layerData = allProj.map(tokenProj => {
                const layerVals = tokenProj[layer] || [0, 0, 0];
                return layerVals.reduce((a, b) => a + b, 0) / layerVals.length;
            });

            return {
                x: Array.from({ length: layerData.length }, (_, i) => i),
                y: layerData,
                name: window.zoomies.formatTraitName(trait),
                type: 'scatter',
                mode: 'lines',
            };
        });

        // Add prompt/response boundary
        const shapes = [{
            type: 'line',
            x0: nPromptTokens - 0.5,
            x1: nPromptTokens - 0.5,
            y0: 0,
            y1: 1,
            yref: 'paper',
            line: { color: 'gray', dash: 'dot', width: 2 },
        }];

        const layout = {
            margin: { t: 30, b: 50, l: 60, r: 20 },
            xaxis: {
                title: 'Token',
                ticktext: allTokens.slice(0, 50),
                tickvals: Array.from({ length: Math.min(50, allTokens.length) }, (_, i) => i),
                tickangle: 45,
            },
            yaxis: { title: 'Projection' },
            shapes,
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            legend: { orientation: 'h', y: 1.1 },
        };

        Plotly.newPlot('line-chart', traces, layout, { responsive: true });

        // Click handler for token selection
        document.getElementById('line-chart').on('plotly_click', (eventData) => {
            const token = eventData.points[0].x;
            window.zoomies.setState({ tokenScope: token });
        });
    });

    registry.renderer('inference:token:all', (data, container, state) => {
        if (!data || !data.traits) {
            container.innerHTML = '<div class="no-data">No inference data available</div>';
            return;
        }

        const tokenIdx = state.tokenScope;
        const traits = Object.keys(data.traits);

        // Get tokens from first trait
        const firstTraitData = data.traits[traits[0]];
        const promptTokens = firstTraitData.prompt?.tokens || [];
        const responseTokens = firstTraitData.response?.tokens || [];
        const allTokens = [...promptTokens, ...responseTokens];
        const tokenStr = allTokens[tokenIdx] || '?';

        container.innerHTML = `
            <div class="token-info">
                <strong>Token ${tokenIdx}:</strong> "${tokenStr}"
            </div>
            <div id="vertical-chart" class="plot-container" style="height: 400px;"></div>
        `;

        // Build trace for each trait (bar chart across layers)
        const traces = traits.map(trait => {
            const traitData = data.traits[trait];
            const projections = traitData.projections || {};
            const promptProj = projections.prompt || [];
            const responseProj = projections.response || [];
            const allProj = [...promptProj, ...responseProj];

            // Get this token's value at each layer (average over methods)
            const values = Array.from({ length: window.zoomies.LAYERS }, (_, layerIdx) => {
                const tokenProj = allProj[tokenIdx];
                if (!tokenProj) return 0;
                const layerVals = tokenProj[layerIdx] || [0, 0, 0];
                return layerVals.reduce((a, b) => a + b, 0) / layerVals.length;
            });

            return {
                x: values,
                y: Array.from({ length: window.zoomies.LAYERS }, (_, i) => i),
                name: window.zoomies.formatTraitName(trait),
                type: 'bar',
                orientation: 'h',
            };
        });

        const layout = {
            margin: { t: 30, b: 50, l: 60, r: 20 },
            xaxis: { title: 'Projection' },
            yaxis: { title: 'Layer', dtick: 5 },
            barmode: 'group',
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            legend: { orientation: 'h', y: 1.1 },
        };

        Plotly.newPlot('vertical-chart', traces, layout, { responsive: true });

        // Click handler for layer selection
        document.getElementById('vertical-chart').on('plotly_click', (eventData) => {
            const layer = eventData.points[0].y;
            window.zoomies.setState({ layerScope: layer });
        });
    });

    registry.renderer('inference:token:layer', (data, container, state) => {
        if (!data || !data.traits) {
            container.innerHTML = '<div class="no-data">No inference data available</div>';
            return;
        }

        const tokenIdx = state.tokenScope;
        const layerIdx = state.layerScope;
        const traits = Object.keys(data.traits);

        // Get tokens from first trait
        const firstTraitData = data.traits[traits[0]];
        const promptTokens = firstTraitData.prompt?.tokens || [];
        const responseTokens = firstTraitData.response?.tokens || [];
        const allTokens = [...promptTokens, ...responseTokens];
        const tokenStr = allTokens[tokenIdx] || '?';

        let html = `
            <div class="microscope-view">
                <div class="token-info">
                    <strong>Token ${tokenIdx}:</strong> "${tokenStr}"
                    <span style="margin-left: 16px;">
                        <strong>Layer:</strong> ${layerIdx}
                    </span>
                </div>

                <h3>Trait Projections</h3>
                <div class="projection-list">
        `;

        traits.forEach(trait => {
            const traitData = data.traits[trait];
            const projections = traitData.projections || {};
            const promptProj = projections.prompt || [];
            const responseProj = projections.response || [];
            const allProj = [...promptProj, ...responseProj];

            // Get value for this token at this layer (average over methods)
            const tokenProj = allProj[tokenIdx];
            let value = 0;
            if (tokenProj) {
                const layerVals = tokenProj[layerIdx] || [0, 0, 0];
                value = layerVals.reduce((a, b) => a + b, 0) / layerVals.length;
            }

            const color = value > 0 ? 'var(--danger)' : value < 0 ? 'var(--primary-color)' : 'var(--text-tertiary)';

            html += `
                <div class="projection-item">
                    <span class="projection-name">${window.zoomies.formatTraitName(trait)}</span>
                    <span class="projection-value" style="color: ${color}">
                        ${value.toFixed(3)}
                    </span>
                </div>
            `;
        });

        html += `
                </div>

                ${layerIdx === 16 ? `
                    <h3 style="margin-top: 24px;">SAE Features</h3>
                    <div class="sae-placeholder">
                        SAE feature analysis coming soon...
                    </div>
                ` : `
                    <div class="no-sae">
                        SAE features only available for layer 16
                    </div>
                `}
            </div>
        `;

        container.innerHTML = html;
    });
});

/**
 * Render inference heatmap (tokens × layers)
 * Data format: traitData.projections = { prompt: [n_tokens][n_layers][3], response: [...] }
 */
function renderInferenceHeatmap(container, traitData) {
    const projections = traitData.projections || {};
    const promptProj = projections.prompt || [];
    const responseProj = projections.response || [];

    // Combine prompt and response projections
    const allProj = [...promptProj, ...responseProj];

    if (allProj.length === 0) {
        container.innerHTML = '<div class="no-data">No projection data</div>';
        return;
    }

    const nTokens = allProj.length;
    const nLayers = allProj[0]?.length || 26;
    const nPromptTokens = promptProj.length;

    // Build z-matrix (layers × tokens)
    // Average over the 3 methods per layer
    const z = Array.from({ length: nLayers }, (_, layerIdx) => {
        return allProj.map(tokenProj => {
            const layerVals = tokenProj[layerIdx] || [0, 0, 0];
            // Average over methods
            return layerVals.reduce((a, b) => a + b, 0) / layerVals.length;
        });
    });

    // Get tokens for hover labels
    const promptTokens = traitData.prompt?.tokens || [];
    const responseTokens = traitData.response?.tokens || [];
    const allTokens = [...promptTokens, ...responseTokens];

    const trace = {
        z: z,
        x: Array.from({ length: nTokens }, (_, i) => i),
        y: Array.from({ length: nLayers }, (_, i) => i),
        type: 'heatmap',
        colorscale: [
            [0, '#4285f4'],
            [0.5, '#ffffff'],
            [1, '#ea4335']
        ],
        zmid: 0,
        hovertemplate: 'Token %{x}: %{customdata}<br>Layer %{y}<br>Projection: %{z:.3f}<extra></extra>',
        customdata: Array.from({ length: nLayers }, () => allTokens),
    };

    // Prompt/response boundary
    const shapes = [{
        type: 'line',
        x0: nPromptTokens - 0.5,
        x1: nPromptTokens - 0.5,
        y0: -0.5,
        y1: nLayers - 0.5,
        line: { color: 'gray', dash: 'dot', width: 2 },
    }];

    const layout = {
        margin: { t: 20, b: 40, l: 50, r: 20 },
        height: 300,
        xaxis: { title: 'Token' },
        yaxis: { title: 'Layer', dtick: 5 },
        shapes,
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
    };

    Plotly.newPlot(container, [trace], layout, { responsive: true });

    // Click handlers
    container.on('plotly_click', (eventData) => {
        const token = eventData.points[0].x;
        const layer = eventData.points[0].y;
        window.zoomies.setState({ tokenScope: token, layerScope: layer });
    });
}
