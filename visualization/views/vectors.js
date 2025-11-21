// Vectors View

async function renderVectors() {
    const contentArea = document.getElementById('content-area');
    const filteredTraits = window.getFilteredTraits();

    if (filteredTraits.length === 0) {
        contentArea.innerHTML = `
            <div class="card">
                <div class="card-title">Vector Analysis</div>
                <div class="info">No traits selected. Select traits in the sidebar to view them.</div>
            </div>
        `;
        return;
    }

    // Show loading state
    contentArea.innerHTML = '<div class="loading">Loading vector analysis overview...</div>';

    // Filter to only traits with vector extraction
    const pathBuilder = new PathBuilder(window.state.experimentData.name);
    const traitsWithVectors = [];
    for (const trait of filteredTraits) {
        try {
            const hasVecs = await window.hasVectors(window.state.experimentData.name, trait);
            if (hasVecs) {
                traitsWithVectors.push(trait);
            } else {
                console.log(`Skipping ${trait.name} - no vectors directory`);
            }
        } catch (e) {
            console.log(`Skipping ${trait.name} - vector fetch failed`);
        }
    }

    if (traitsWithVectors.length === 0) {
        contentArea.innerHTML = `
            <div class="card">
                <div class="card-title">No Vector Extractions Found</div>
                <div class="info">
                    No traits in this experiment have completed vector extraction.
                    Only traits with <code>extraction/vectors/</code> directories are shown here.
                </div>
            </div>
        `;
        return;
    }

    console.log(`Found ${traitsWithVectors.length} traits with vectors (out of ${filteredTraits.length} total)`);

    // Detect number of layers from experiment metadata
    const firstTrait = traitsWithVectors[0];
    const nLayers = firstTrait.metadata?.n_layers || 26;  // Default to 26 for Gemma 2B
    console.log(`Detected ${nLayers} layers from metadata`);

    // Load all vector metadata (parallel fetching for speed)
    const vectorMetrics = {};
    const methods = ['mean_diff', 'probe', 'ica', 'gradient'];
    const layers = Array.from({ length: nLayers }, (_, i) => i);

    // Fetch all metadata AND prompt examples in parallel
    const fetchPromises = traitsWithVectors.flatMap(trait => [
        // Vector metadata
        ...methods.flatMap(method =>
            layers.map(layer => {
                const url = pathBuilder.vectorMetadata(trait, method, layer);
                return fetch(url)
                    .then(r => {
                        if (!r.ok) console.warn(`Failed to fetch: ${url}`);
                        return r.ok ? r.json() : null;
                    })
                    .then(data => ({ type: 'vector', trait: trait.name, method, layer, data }))
                    .catch(e => {
                        console.error(`Error fetching ${url}:`, e);
                        return { type: 'vector', trait: trait.name, method, layer, data: null };
                    });
            })
        ),
        // Prompt examples (first row of pos.csv and neg.csv)
        fetch(pathBuilder.responses(trait, 'pos', 'csv'))
            .then(r => r.ok ? r.text() : null)
            .then(text => {
                if (!text) return null;
                const lines = text.split('\n');
                if (lines.length < 2) return null;
                const headers = lines[0].split(',');
                const values = lines[1].split(',');
                const row = {};
                headers.forEach((h, i) => row[h] = values[i]);
                return { type: 'prompt', trait: trait.name, polarity: 'pos', data: row };
            })
            .catch(() => ({ type: 'prompt', trait: trait.name, polarity: 'pos', data: null })),
        fetch(pathBuilder.responses(trait, 'neg', 'csv'))
            .then(r => r.ok ? r.text() : null)
            .then(text => {
                if (!text) return null;
                const lines = text.split('\n');
                if (lines.length < 2) return null;
                const headers = lines[0].split(',');
                const values = lines[1].split(',');
                const row = {};
                headers.forEach((h, i) => row[h] = values[i]);
                return { type: 'prompt', trait: trait.name, polarity: 'neg', data: row };
            })
            .catch(() => ({ type: 'prompt', trait: trait.name, polarity: 'neg', data: null }))
    ]);

    const allResults = await Promise.all(fetchPromises);
    console.log(`Loaded ${allResults.length} results`);

    // Count successful loads (filter out null results)
    const successfulVectors = allResults.filter(r => r && r.type === 'vector' && r.data !== null).length;
    console.log(`Successfully loaded ${successfulVectors} vector metadata files`);

    // Organize results by trait
    for (const trait of traitsWithVectors) {
        try {
            const vectorData = {};
            for (const method of methods) {
                vectorData[method] = {};
            }

            // Fill in vector data from parallel fetch results (filter out nulls)
            allResults
                .filter(r => r && r.type === 'vector' && r.trait === trait.name && r.data !== null)
                .forEach(r => {
                    vectorData[r.method][r.layer] = r.data;
                });

            // Extract prompt examples (filter out nulls)
            const prompts = {};
            allResults
                .filter(r => r && r.type === 'prompt' && r.trait === trait.name && r.data !== null)
                .forEach(r => {
                    prompts[r.polarity] = r.data;
                });

            // Calculate best layer for each method
            const best = {};
            methods.forEach(method => {
                let bestLayer = -1;
                let bestValue = -Infinity;
                layers.forEach(layer => {
                    if (vectorData[method][layer]) {
                        const metadata = vectorData[method][layer];
                        let value;
                        if (method === 'probe') {
                            // Invert: smaller norm = stronger
                            value = metadata.vector_norm ? (1.0 / metadata.vector_norm) : 0;
                        } else if (method === 'gradient') {
                            // Use separation (unit normalized)
                            value = metadata.final_separation || metadata.vector_norm;
                        } else {
                            // Use magnitude
                            value = metadata.vector_norm;
                        }
                        if (!isNaN(value) && value > bestValue) {
                            bestValue = value;
                            bestLayer = layer;
                        }
                    }
                });
                best[method] = { layer: bestLayer, norm: bestValue };
            });

            vectorMetrics[trait.name] = { vectorData, best, prompts };
        } catch (e) {
            console.error(`Failed to load ${trait.name}:`, e);
        }
    }

    // Render overview with mini heatmaps
    let html = `
        <div class="explanation">
            <div class="explanation-summary">We can approximate what a model is "thinking" for a given trait by comparing how it activates when showing that trait versus when it doesn't.</div>
            <div class="explanation-details">
                <h4>Layer Architecture Intuition</h4>
                <ul>
                    <li><strong>Early layers (0-5):</strong> Syntax, local patterns, token-level features</li>
                    <li><strong>Middle layers (6-15):</strong> Semantic understanding, entity recognition, basic reasoning</li>
                    <li><strong>Late layers (16-24):</strong> Abstract concepts, complex reasoning, behavioral traits (refusal, uncertainty)</li>
                    <li><strong>Final layer (25):</strong> Projection towards output vocabulary (often degrades semantic quality)</li>
                </ul>

                <h4>Extraction Methods</h4>
                <p>Four approaches to extract trait direction vectors from activations \\(\\mathbf{A}_{\\text{pos}} \\in \\mathbb{R}^{n \\times d}\\) and \\(\\mathbf{A}_{\\text{neg}} \\in \\mathbb{R}^{m \\times d}\\):</p>

                <p><strong>1. Mean Difference</strong></p>
                <p>$$\\mathbf{v}_{\\text{mean}} = \\frac{1}{n}\\sum_{i=1}^n \\mathbf{a}_i^{\\text{pos}} - \\frac{1}{m}\\sum_{j=1}^m \\mathbf{a}_j^{\\text{neg}}$$</p>
                <ul>
                    <li>Simple cluster center separation</li>
                    <li>Unnormalized (typical norm ≈ 50-100)</li>
                    <li>Fast, no training required</li>
                </ul>

                <p><strong>2. Probe (Linear Classifier)</strong></p>
                <p>$$\\mathbf{v}_{\\text{probe}} = \\arg\\min_{\\mathbf{w}} \\sum_i \\log(1 + e^{-y_i \\mathbf{w}^T \\mathbf{a}_i}) + \\lambda \\|\\mathbf{w}\\|_2^2$$</p>
                <ul>
                    <li>Logistic regression weights (L2-regularized)</li>
                    <li>Normalized during training (norm ≈ 1-5)</li>
                    <li>Maximizes linear separability</li>
                </ul>

                <p><strong>3. ICA (Independent Components)</strong></p>
                <p>$$\\mathbf{v}_{\\text{ica}} = \\text{FastICA}([\\mathbf{A}_{\\text{pos}}; \\mathbf{A}_{\\text{neg}}])[k]$$</p>
                <ul>
                    <li>Extracts statistically independent components</li>
                    <li>Disentangles mixed trait signals via negentropy maximization</li>
                    <li>Variable norm depending on component strength</li>
                </ul>

                <p><strong>4. Gradient (Optimization)</strong></p>
                <p>$$\\mathbf{v}_{\\text{grad}} = \\arg\\min_{\\mathbf{v}} \\mathcal{L}(\\mathbf{v}) \\text{ where } \\mathcal{L} = \\|\\mathbf{v}^T \\mathbf{A}_{\\text{pos}} - \\alpha\\|^2 + \\|\\mathbf{v}^T \\mathbf{A}_{\\text{neg}}\\|^2$$</p>
                <ul>
                    <li>Custom objective via gradient descent</li>
                    <li>Unit normalized (norm = 1.0)</li>
                    <li>Can fail (NaN) if gradients vanish</li>
                </ul>

                <h4>Visualization Details</h4>
                <p>Each method is normalized independently (0-100% of that method's max across layers) to show within-method layer strength. Darker = stronger for that method at that layer.</p>
                <ul>
                    <li><strong>Probe:</strong> Inverted vector norm (smaller norm = stronger classifier due to L2 regularization)</li>
                    <li><strong>Gradient:</strong> Separation strength (vectors are unit normalized, norm = 1.0)</li>
                    <li><strong>Mean Diff & ICA:</strong> Vector magnitude (direction strength in activation space)</li>
                </ul>

                <p><strong>Note:</strong> All vectors extracted from token-averaged activations: \\(\\bar{\\mathbf{a}}_i = \\frac{1}{T} \\sum_{t=1}^T \\mathbf{h}_t^{(i)}\\)</p>
            </div>
        <div class="card">
            <div class="card-title">Vector Analysis Overview</div>
            <div class="info">
                Vector norms across ${nLayers} layers and 4 extraction methods. Click a row for detailed view.
            </div>
            <div style="margin-top: 15px;">
    `;

    traitsWithVectors.forEach(trait => {
        const displayName = window.getDisplayName(trait.name);
        const metrics = vectorMetrics[trait.name];
        if (!metrics) {
            console.warn(`No metrics for ${trait.name}`);
            return;
        }

        const best = metrics.best;

        // Check if we have any valid data
        const hasData = Object.values(best).some(b => b.layer !== -1 && b.norm !== -Infinity);
        if (!hasData) {
            console.warn(`No valid vector data for ${trait.name}`);
            return;
        }

        const bestMethod = Object.keys(best).reduce((a, b) =>
            best[a].norm > best[b].norm ? a : b
        );

        // Build compact prompt examples
        let examplesHtml = '<div style="font-size: 11px; color: var(--text-secondary);">No examples</div>';
        if (metrics.prompts && metrics.prompts.pos && metrics.prompts.neg) {
            const posInstruction = (metrics.prompts.pos.instruction || '').substring(0, 60);
            const negInstruction = (metrics.prompts.neg.instruction || '').substring(0, 60);
            examplesHtml = `
                <div style="font-size: 11px; color: var(--text-secondary); line-height: 1.4;">
                    <div style="margin-bottom: 3px;"><strong style="color: var(--primary-color);">+</strong> ${posInstruction}...</div>
                    <div><strong style="color: var(--accent-color);">−</strong> ${negInstruction}...</div>
                </div>
            `;
        }

        html += `
            <div style="display: grid; grid-template-columns: 180px 1fr 320px; gap: 15px; padding: 12px 12px 20px 12px; cursor: pointer; transition: background 0.2s;"
                 onclick="loadVectorAnalysis('${trait.name}')"
                 onmouseover="this.style.background='var(--bg-secondary)'"
                 onmouseout="this.style.background='transparent'">

                <div style="display: flex; flex-direction: column; justify-content: center;">
                    <div style="font-weight: 600; color: var(--text-primary); margin-bottom: 4px;">${displayName}</div>
                    <div style="font-size: 11px; color: var(--text-secondary);">
                        Best: ${bestMethod.replace(/_/g, ' ')} L${best[bestMethod].layer}
                    </div>
                </div>

                <div id="mini-heatmap-${trait.name}" style="height: 100px;"></div>

                <div style="display: flex; align-items: center;">
                    ${examplesHtml}
                </div>
            </div>
        `;
    });

    html += `
            </div>
            <div id="vector-details"></div>
        </div>
    `;

    contentArea.innerHTML = html;
    renderMath();
    renderMath();

    // Render mini heatmaps (normalized per method)
    traitsWithVectors.forEach(trait => {
        const metrics = vectorMetrics[trait.name];
        if (!metrics) {
            console.warn(`No metrics for trait: ${trait.name}`);
            return;
        }

        console.log(`Rendering heatmap for ${trait.name}:`, metrics);
        const methods = Object.keys(metrics.vectorData);
        if (methods.length === 0) {
            console.warn(`No methods found for trait: ${trait.name}`);
            return;
        }
        // Use detected number of layers
        const layers = Array.from({ length: nLayers }, (_, i) => i);

        // Normalize each method independently (0-100% of that method's max)
        // Probe: invert norm (smaller = stronger due to L2 regularization)
        // Gradient: use final_separation (unit normalized, so norm is always 1.0)
        // Mean_diff & ICA: use vector_norm
        const normalizedData = layers.map(layer => {
            return methods.map(method => {
                const metadata = metrics.vectorData[method][layer];
                if (!metadata) return null;

                if (method === 'probe') {
                    // Invert: smaller norm = stronger classifier
                    return metadata.vector_norm ? (1.0 / metadata.vector_norm) : null;
                } else if (method === 'gradient') {
                    // Use separation (vectors are unit normalized)
                    return metadata.final_separation || metadata.vector_norm;
                } else {
                    // Use vector magnitude
                    return metadata.vector_norm;
                }
            });
        });

        // Find max for each method
        const maxPerMethod = methods.map((method, methodIdx) => {
            const values = normalizedData.map(row => row[methodIdx]).filter(v => v !== null);
            return values.length > 0 ? Math.max(...values) : 1;
        });

        // Normalize each column by its max
        const heatmapData = normalizedData.map(row => {
            return row.map((value, methodIdx) => {
                if (value === null) return null;
                return (value / maxPerMethod[methodIdx]) * 100; // Percentage of method's max
            });
        });

        // Transpose for horizontal layout: methods become rows (y), layers become columns (x)
        const trace = {
            z: heatmapData[0].map((_, methodIdx) =>
                heatmapData.map(row => row[methodIdx])
            ),
            x: layers,
            y: methods.map(m => m.replace(/_/g, ' ')),
            type: 'heatmap',
            colorscale: 'Viridis',
            showscale: false,
            hovertemplate: '%{y}<br>Layer %{x}<br>%{z:.1f}%<extra></extra>',
            zmin: 0,
            zmax: 100
        };

        Plotly.newPlot(`mini-heatmap-${trait.name}`, [trace], window.getPlotlyLayout({
            margin: { l: 50, r: 5, t: 5, b: 20 },
            xaxis: {
                title: '',
                showticklabels: true,
                tickmode: 'linear',
                dtick: 5,  // Show every 5th layer
                tickfont: { size: 8 }
            },
            yaxis: { title: '', side: 'left', tickfont: { size: 8 } },
            height: 80
        }), { displayModeBar: false });
    });
}

// Load vector analysis
async function loadVectorAnalysis(traitName) {
    const detailsDiv = document.getElementById('vector-details');
    detailsDiv.innerHTML = '<div class="loading">Loading vector metadata...</div>';

    try {
        // Get trait metadata to determine number of layers
        const trait = window.state.experimentData.traits.find(t => t.name === traitName);
        const nLayers = trait?.metadata?.n_layers || 26;
        console.log(`Loading vector analysis for ${traitName} (${nLayers} layers)`);

        const pathBuilder = new PathBuilder(window.state.experimentData.name);
        const methods = ['mean_diff', 'probe', 'ica', 'gradient'];
        const layers = Array.from({ length: nLayers }, (_, i) => i);

        const vectorData = {};

        for (const method of methods) {
            vectorData[method] = {};
            for (const layer of layers) {
                try {
                    const response = await fetch(pathBuilder.vectorMetadata(trait, method, layer));
                    if (response.ok) {
                        vectorData[method][layer] = await response.json();
                    }
                } catch (e) {
                    // Vector doesn't exist
                }
            }
        }

        renderVectorHeatmap(traitName, vectorData);
    } catch (error) {
        console.error('Error loading vectors:', error);
        detailsDiv.innerHTML = '<div class="error">Failed to load vector data</div>';
    }
}

// Render vector heatmap (normalized per method)
function renderVectorHeatmap(traitName, vectorData) {
    const detailsDiv = document.getElementById('vector-details');
    const displayName = window.getDisplayName(traitName);

    const methods = Object.keys(vectorData);
    // Detect number of layers from actual data
    const nLayers = Object.keys(vectorData[methods[0]] || {}).length;
    const layers = Array.from({ length: nLayers }, (_, i) => i);

    // Collect raw metrics
    // Probe: invert norm (smaller = stronger due to L2 regularization)
    // Gradient: use final_separation (unit normalized, norm always 1.0)
    // Mean_diff & ICA: use vector_norm
    const rawData = layers.map(layer => {
        return methods.map(method => {
            const metadata = vectorData[method][layer];
            if (!metadata) return null;

            if (method === 'probe') {
                return metadata.vector_norm ? (1.0 / metadata.vector_norm) : null;
            } else if (method === 'gradient') {
                return metadata.final_separation || metadata.vector_norm;
            } else {
                return metadata.vector_norm;
            }
        });
    });

    // Find max for each method
    const maxPerMethod = methods.map((method, methodIdx) => {
        const values = rawData.map(row => row[methodIdx]).filter(v => v !== null);
        return values.length > 0 ? Math.max(...values) : 1;
    });

    // Normalize each column by its max (0-100%)
    const normalizedData = rawData.map(row => {
        return row.map((value, methodIdx) => {
            if (value === null) return null;
            return (value / maxPerMethod[methodIdx]) * 100;
        });
    });

    // Also store raw values for hover
    const customData = rawData.map((row, layerIdx) => {
        return row.map((value, methodIdx) => {
            return {
                raw: value,
                max: maxPerMethod[methodIdx],
                layer: layers[layerIdx],
                method: methods[methodIdx]
            };
        });
    });

    const trace = {
        z: normalizedData,
        x: methods.map(m => m.replace(/_/g, ' ').toUpperCase()),
        y: layers,
        type: 'heatmap',
        colorscale: 'Viridis',
        hovertemplate: 'Method: %{x}<br>Layer %{y}<br>Strength: %{z:.1f}% of max<br>Raw norm: %{customdata.raw:.2f}<br>Max for method: %{customdata.max:.2f}<extra></extra>',
        customdata: customData,
        zmin: 0,
        zmax: 100,
        colorbar: {
            title: '% of Max',
            titleside: 'right'
        }
    };

    let html = `
        <div style="margin-top: 30px;">
            <div class="card-title">${displayName} - Vector Strength by Method & Layer</div>
            <div class="info" style="margin-top: 10px; font-size: 13px;">
                Each method normalized independently (0-100% of that method's max across layers).
                This shows which layers are strongest for each method, not cross-method comparisons.
            </div>
            <div id="vector-heatmap" style="margin-top: 20px;"></div>
        </div>
    `;

    detailsDiv.innerHTML = html;

    Plotly.newPlot('vector-heatmap', [trace], window.getPlotlyLayout({
        title: 'Normalized Vector Strength (% of Method Max)',
        xaxis: { title: 'Extraction Method' },
        yaxis: { title: 'Layer' },  // Layer 0 at bottom, layer N at top
        height: 600
    }), { displayModeBar: false });
}

// Render Cross-Distribution Analysis
// Export to global scope
window.renderVectors = renderVectors;
