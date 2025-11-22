// Layer Dive View

async function renderLayerDeepDive() {
    const contentArea = document.getElementById('content-area');
    const filteredTraits = window.getFilteredTraits();

    if (filteredTraits.length === 0) {
        contentArea.innerHTML = `<div style="color: var(--text-secondary); font-size: 12px;">Select at least one trait to view layer internals</div>`;
        return;
    }

    // Load data for ALL selected traits
    contentArea.innerHTML = '<div id="all-traits-layer-dive"></div>';
    const container = document.getElementById('all-traits-layer-dive');

    for (const trait of filteredTraits) {
        // Create a unique div for this trait
        const traitDiv = document.createElement('div');
        traitDiv.id = `layer-dive-${trait.name}`;
        traitDiv.style.marginBottom = '32px';  // Airy spacing between traits
        container.appendChild(traitDiv);

        // Try to load the selected prompt for default layer 16 using PathBuilder
        try {
            const fetchPath = window.paths.tier3Data(trait, window.state.currentPrompt, 16);
            console.log(`[${trait.name}] Fetching layer deep dive data: ${fetchPath}`);
            const response = await fetch(fetchPath);
            if (!response.ok) throw new Error(`No data found for prompt ${window.state.currentPrompt}`);

            const data = await response.json();
            renderTier3DataInContainer(traitDiv.id, trait, data);
        } catch (error) {
            // No data yet - show instructions
            renderTier3InstructionsInContainer(traitDiv.id, trait, window.state.currentPrompt);
        }
    }
}

function renderTier3Instructions(trait, filteredTraits) {
    const contentArea = document.getElementById('content-area');

    // Find traits with tier 3 data
    const traitsWithTier3 = window.state.experimentData.traits.filter(t => t.hasTier3);

    let tier3StatusHtml = '';
    if (traitsWithTier3.length > 0) {
        const traitList = traitsWithTier3.map(t => `<strong>${window.getDisplayName(t.name)}</strong>`).join(', ');
        tier3StatusHtml = `
            <div style="background: var(--accent-color); color: white; padding: 12px 16px; border-radius: 6px; margin-bottom: 20px;">
                ✓ Layer internals available for: ${traitList}
                <br><span style="opacity: 0.9; font-size: 13px;">Select ${traitsWithTier3.length === 1 ? 'this trait' : 'one of these traits'} to view the data</span>
            </div>
        `;
    }

    contentArea.innerHTML = `
        <div class="explanation">
            <div class="explanation-summary">We can pinpoint exactly which neurons and attention mechanisms are responsible for computing a trait at a specific layer.</div>
            <div class="explanation-details">

            <h4>Captured Components</h4>

            <p><strong>Attention Projections (before multihead split):</strong></p>
            <p>$$\\mathbf{Q} = \\mathbf{h}\\mathbf{W}_Q, \\quad \\mathbf{K} = \\mathbf{h}\\mathbf{W}_K, \\quad \\mathbf{V} = \\mathbf{h}\\mathbf{W}_V$$</p>
            <p>Query, Key, Value projections \\(\\in \\mathbb{R}^{T \\times d}\\)</p>

            <p><strong>Attention Heads (post-split):</strong></p>
            <p>$$\\text{head}_i = \\text{Attention}(\\mathbf{Q}_i, \\mathbf{K}_i, \\mathbf{V}_i) = \\text{softmax}\\left(\\frac{\\mathbf{Q}_i \\mathbf{K}_i^T}{\\sqrt{d_k}}\\right)\\mathbf{V}_i$$</p>
            <p>Per-head outputs showing which heads activate for the trait</p>

            <p><strong>MLP Internals:</strong></p>
            <p>$$\\mathbf{h}_{\\text{mlp}} = \\mathbf{W}_2 \\cdot \\text{GELU}(\\mathbf{W}_1 \\mathbf{h})$$</p>
            <ul>
                <li>Pre-GELU: \\(\\mathbf{W}_1 \\mathbf{h} \\in \\mathbb{R}^{9216}\\) (2304 hidden × 4 expansion)</li>
                <li>Post-GELU: Shows which of 9216 neurons fire for trait</li>
            </ul>

            <h4>Use Case</h4>
            <p>Identifies specific neurons/heads responsible for trait computation, enabling:</p>
            <ul>
                <li>Neuron-level interpretation</li>
                <li>Surgical intervention (ablation studies)</li>
                <li>Feature visualization</li>
            </ul>
        </div>
        <div class="card">
            <div class="card-title">Layer Deep Dive: ${window.getDisplayName(trait.name)}</div>

            ${tier3StatusHtml}

            <div class="info" style="margin-bottom: 20px;">
                <strong>⚠️ No layer internals data available for ${trait.name}</strong>
            </div>

            <div style="background: var(--bg-tertiary); padding: 20px; border-radius: 8px;">
                <h3 style="color: var(--text-primary); margin-bottom: 15px;">Capture Layer Internals</h3>
                <p style="color: var(--text-secondary); margin-bottom: 15px;">
                    Capture complete internals (Q/K/V, attention heads, 9216 MLP neurons) for one layer:
                </p>
                <pre style="background: var(--bg-primary); color: var(--text-primary); padding: 15px; border-radius: 4px; margin: 15px 0; overflow-x: auto;">python inference/capture_layers.py \\
  --experiment ${window.state.experimentData.name} \\
  --mode single \\
  --layer 16 \\
  --prompt "What is the capital of France?" \\
  --save-json</pre>
                <p style="color: var(--text-secondary); margin-top: 15px;">
                    The <code style="background: var(--bg-primary); padding: 2px 6px; border-radius: 3px;">--save-json</code> flag creates visualization-friendly JSON files (~10-20 MB).
                </p>
                <p style="color: var(--text-tertiary); font-size: 13px; margin-top: 10px;">
                    This reveals which specific neurons and attention heads are responsible for ${window.getDisplayName(trait.name)}.
                </p>
            </div>
        </div>
    `;
}

function renderTier3InstructionsInContainer(containerId, trait, promptNum) {
    const container = document.getElementById(containerId);

    container.innerHTML = `
        <div style="margin-bottom: 4px;">
            <span style="color: var(--text-primary); font-size: 14px; font-weight: 600;">${window.getDisplayName(trait.name)}</span>
            <span style="color: var(--text-tertiary); font-size: 11px; margin-left: 8px;">⚠️ No data for prompt ${promptNum}</span>
        </div>
        <div style="color: var(--text-secondary); font-size: 11px; margin-bottom: 4px;">
            The file <code>prompt_${promptNum}_layer16.json</code> does not exist for this trait.
        </div>
        <div style="color: var(--text-secondary); font-size: 11px; margin-bottom: 4px;">
            To capture complete internals (Q/K/V, attention heads, 9216 MLP neurons):
        </div>
        <pre style="background: var(--bg-secondary); color: var(--text-primary); padding: 8px; border-radius: 4px; margin: 0; overflow-x: auto; font-size: 10px;">python inference/capture_layers.py --experiment ${window.state.experimentData.name} --mode single --layer 16 --prompt "..." --save-json</pre>
    `;
}

function renderTier3DataInContainer(containerId, trait, data) {
    const container = document.getElementById(containerId);

    // Get prompt GELU activations: [n_tokens, 9216]
    const promptGelu = data.internals.prompt.gelu;
    const promptTokens = data.prompt.tokens;

    // Combine prompt and response tokens and activations
    const responseGelu = data.internals.response.gelu;
    const responseTokens = data.response.tokens;

    const allTokens = [...promptTokens, ...responseTokens];
    const allGelu = [...promptGelu, ...responseGelu];

    container.innerHTML = `
        <div style="margin-bottom: 4px;">
            <span style="color: var(--text-primary); font-size: 14px; font-weight: 600;">${window.getDisplayName(trait.name)}</span>
            <span style="color: var(--text-secondary); font-size: 11px; margin-left: 8px;">Layer ${data.layer}</span>
        </div>
        <div style="color: var(--text-secondary); font-size: 10px; margin-bottom: 8px;">
            ${data.prompt.text} ${data.response.text}
        </div>
        <div id="top-neurons-${trait.name}" style="margin-top: 8px;"></div>
    `;

    // Render combined neuron activations (compact)
    renderTopNeuronsCompact(`top-neurons-${trait.name}`, allGelu, allTokens, promptTokens.length);
}

function renderTopNeuronsCompact(divId, geluActivations, tokens, promptLength) {
    // geluActivations: [n_tokens, 9216]
    const nTokens = geluActivations.length;
    const nNeurons = geluActivations[0].length;

    // Average absolute activation across all tokens for each neuron
    const neuronAvgAbs = new Array(nNeurons).fill(0);
    for (let n = 0; n < nNeurons; n++) {
        let sum = 0;
        for (let t = 0; t < nTokens; t++) {
            sum += Math.abs(geluActivations[t][n]);
        }
        neuronAvgAbs[n] = sum / nTokens;
    }

    // Get top 10 neurons
    const neuronIndices = Array.from({length: nNeurons}, (_, i) => i);
    neuronIndices.sort((a, b) => neuronAvgAbs[b] - neuronAvgAbs[a]);
    const topNeurons = neuronIndices.slice(0, 10);

    // Build heatmap data [10 neurons x n_tokens]
    const heatmapData = topNeurons.map(n =>
        geluActivations.map(tokenAct => tokenAct[n])
    );

    const trace = {
        z: heatmapData,
        x: tokens,
        y: topNeurons.map(n => `N${n}`),
        type: 'heatmap',
        colorscale: 'Viridis',
        hovertemplate: 'Token: %{x}<br>Neuron: %{y}<br>Activation: %{z:.3f}<extra></extra>'
    };

    const layout = {
        margin: { t: 5, b: 30, l: 40, r: 5 },
        height: 150,
        xaxis: {
            tickangle: -45,
            tickfont: { size: 9 }
        },
        yaxis: {
            tickfont: { size: 9 }
        },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#e0e0e0', family: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif' }
    };

    Plotly.newPlot(divId, [trace], layout, { displayModeBar: false });
}

function renderTier3Data(trait, data) {
    const contentArea = document.getElementById('content-area');

    // Get prompt GELU activations: [n_tokens, 9216]
    const promptGelu = data.internals.prompt.gelu;
    const promptTokens = data.prompt.tokens;

    contentArea.innerHTML = `
        <div class="explanation">
            <div class="explanation-summary">We can pinpoint exactly which neurons and attention mechanisms are responsible for computing a trait at layer ${data.layer}.</div>
            <div class="explanation-details">

            <h4>Components Analyzed</h4>

            <p><strong>Attention Projections:</strong></p>
            <p>$$\\mathbf{Q}, \\mathbf{K}, \\mathbf{V} = \\mathbf{h} \\mathbf{W}_Q, \\mathbf{h} \\mathbf{W}_K, \\mathbf{h} \\mathbf{W}_V \\in \\mathbb{R}^{T \\times d}$$</p>

            <p><strong>Attention Heads:</strong></p>
            <p>$$\\text{head}_i = \\text{softmax}\\left(\\frac{\\mathbf{Q}_i \\mathbf{K}_i^T}{\\sqrt{d_k}}\\right)\\mathbf{V}_i$$</p>

            <p><strong>MLP Neurons (9216 total):</strong></p>
            <p>$$\\mathbf{z} = \\text{GELU}(\\mathbf{W}_1 \\mathbf{h}) \\in \\mathbb{R}^{9216}$$</p>
            <p>Top 20 neurons ranked by \\(\\frac{1}{T}\\sum_t |z_t^{(i)}|\\) (average absolute activation)</p>

            <h4>Interpretation</h4>
            <p>Neurons with high activation magnitude are most responsible for this layer's contribution to the trait.</p>
            </div>
        <div class="card">
            <div class="card-title">Layer Deep Dive: ${window.getDisplayName(trait.name)} (Layer ${data.layer})</div>

            <div style="background: var(--bg-tertiary); padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                <div style="margin-bottom: 10px;">
                    <div style="color: var(--text-secondary); font-size: 13px; margin-bottom: 5px;">Full Conversation</div>
                    <div style="color: var(--text-primary); font-size: 16px; font-weight: 600;">${data.prompt.text} ${data.response.text}</div>
                </div>
                <div>
                    <div style="color: var(--text-secondary); font-size: 13px;">Layer</div>
                    <div style="color: var(--text-primary); font-size: 16px; font-weight: 600;">Layer ${data.layer} of 27</div>
                </div>
            </div>

            <div id="sublayer-trajectory" style="margin-bottom: 30px;"></div>
            <div id="contribution-breakdown" style="margin-bottom: 30px;"></div>
            <div id="head-contributions" style="margin-bottom: 30px;"></div>
            <div id="attention-heatmaps" style="margin-bottom: 30px;"></div>
            <div id="top-neurons-combined"></div>
        </div>
    `;

    // Combine prompt and response tokens and activations
    const responseGelu = data.internals.response.gelu;
    const responseTokens = data.response.tokens;

    const allTokens = [...promptTokens, ...responseTokens];
    const allGelu = [...promptGelu, ...responseGelu];

    // Render new visualizations if trait projections available
    if (data.trait_projections) {
        renderSublayerTrajectory(data, promptTokens.length);
        renderContributionBreakdown(data, promptTokens.length);
        renderHeadContributions(data, promptTokens.length);
    }

    // Store data for slider updates
    window.currentLayerData = { data, promptLength: promptTokens.length };

    // Render attention heatmaps
    if (data.internals.prompt.attn_weights) {
        renderAttentionHeatmaps(data, promptTokens.length, 0);  // Start with token 0
    }

    // Render combined neuron activations
    renderTopNeurons('top-neurons-combined', 'Top Neurons (Full Conversation)', allGelu, allTokens, promptTokens.length);

    // Setup math rendering and toggle listeners
    renderMath();
}

// ============================================================================
// Phase 1 Visualizations
// ============================================================================

function renderSublayerTrajectory(data, promptLength) {
    // Combine prompt and response projections
    const promptProj = data.trait_projections.prompt;
    const responseProj = data.trait_projections.response;

    const allTokens = [...data.prompt.tokens, ...data.response.tokens];
    const residualIn = [...promptProj.residual_in, ...responseProj.residual_in];
    const residualAfterAttn = [...promptProj.residual_after_attn, ...responseProj.residual_after_attn];
    const residualOut = [...promptProj.residual_out, ...responseProj.residual_out];

    const trace1 = {
        x: Array.from({length: allTokens.length}, (_, i) => i),
        y: residualIn,
        mode: 'lines+markers',
        name: 'Residual In',
        line: { color: '#888', width: 2 },
        marker: { size: 6 },
        hovertemplate: 'Token %{x}: %{text}<br>Trait Score: %{y:.3f}<extra></extra>',
        text: allTokens
    };

    const trace2 = {
        x: Array.from({length: allTokens.length}, (_, i) => i),
        y: residualAfterAttn,
        mode: 'lines+markers',
        name: 'After Attention',
        line: { color: '#4a9eff', width: 2 },
        marker: { size: 6 },
        hovertemplate: 'Token %{x}: %{text}<br>Trait Score: %{y:.3f}<extra></extra>',
        text: allTokens
    };

    const trace3 = {
        x: Array.from({length: allTokens.length}, (_, i) => i),
        y: residualOut,
        mode: 'lines+markers',
        name: 'After MLP (Output)',
        line: { color: '#4caf50', width: 2 },
        marker: { size: 6 },
        hovertemplate: 'Token %{x}: %{text}<br>Trait Score: %{y:.3f}<extra></extra>',
        text: allTokens
    };

    // Add separator line
    const shapes = [];
    if (promptLength) {
        shapes.push({
            type: 'line',
            xref: 'x',
            yref: 'paper',
            x0: promptLength - 0.5,
            x1: promptLength - 0.5,
            y0: 0,
            y1: 1,
            line: {
                color: 'rgba(255, 255, 255, 0.3)',
                width: 2,
                dash: 'dash'
            }
        });
    }

    const layout = window.getPlotlyLayout({
        title: '3-Checkpoint Trait Trajectory (Sublayer Evolution)',
        xaxis: { title: 'Token Position' },
        yaxis: { title: 'Trait Score' },
        height: 350,
        shapes: shapes
    });

    Plotly.newPlot('sublayer-trajectory', [trace1, trace2, trace3], layout, { displayModeBar: false });
}

function renderContributionBreakdown(data, promptLength) {
    // Combine prompt and response contributions
    const promptProj = data.trait_projections.prompt;
    const responseProj = data.trait_projections.response;

    const allTokens = [...data.prompt.tokens, ...data.response.tokens];
    const attnContrib = [...promptProj.attn_contribution, ...responseProj.attn_contribution];
    const mlpContrib = [...promptProj.mlp_contribution, ...responseProj.mlp_contribution];

    const trace1 = {
        x: Array.from({length: allTokens.length}, (_, i) => i),
        y: attnContrib,
        type: 'bar',
        name: 'Attention Contribution',
        marker: { color: '#4a9eff' },
        hovertemplate: 'Token %{x}: %{text}<br>Attention: %{y:.3f}<extra></extra>',
        text: allTokens
    };

    const trace2 = {
        x: Array.from({length: allTokens.length}, (_, i) => i),
        y: mlpContrib,
        type: 'bar',
        name: 'MLP Contribution',
        marker: { color: '#4caf50' },
        hovertemplate: 'Token %{x}: %{text}<br>MLP: %{y:.3f}<extra></extra>',
        text: allTokens
    };

    // Add separator line
    const shapes = [];
    if (promptLength) {
        shapes.push({
            type: 'line',
            xref: 'x',
            yref: 'paper',
            x0: promptLength - 0.5,
            x1: promptLength - 0.5,
            y0: 0,
            y1: 1,
            line: {
                color: 'rgba(255, 255, 255, 0.3)',
                width: 2,
                dash: 'dash'
            }
        });
    }

    const layout = window.getPlotlyLayout({
        title: 'Attention vs MLP Contribution (Per Token)',
        xaxis: { title: 'Token Position' },
        yaxis: { title: 'Trait Contribution' },
        barmode: 'group',
        height: 400,
        shapes: shapes
    });

    Plotly.newPlot('contribution-breakdown', [trace1, trace2], layout, { displayModeBar: false });
}

function renderHeadContributions(data, promptLength) {
    // Check if per-head contributions are available
    const promptProj = data.trait_projections.prompt;
    const responseProj = data.trait_projections.response;

    if (!promptProj.head_contributions && !responseProj.head_contributions) {
        document.getElementById('head-contributions').style.display = 'none';
        return;
    }

    document.getElementById('head-contributions').style.display = 'block';

    // Combine prompt and response head contributions
    // head_contributions is [n_heads, n_tokens]
    const headContribsPrompt = promptProj.head_contributions || [];
    const headContribsResponse = responseProj.head_contributions || [];

    const allTokens = [...data.prompt.tokens, ...data.response.tokens];
    const nHeads = headContribsPrompt.length || headContribsResponse.length;

    // Create traces for each head
    const traces = [];
    const colors = ['#4a9eff', '#ff6b6b', '#51cf66', '#ffc107', '#9c27b0', '#ff9800', '#00bcd4', '#e91e63'];

    for (let headIdx = 0; headIdx < nHeads; headIdx++) {
        // Combine prompt and response contributions for this head
        const promptContrib = headContribsPrompt[headIdx] || [];
        const responseContrib = headContribsResponse[headIdx] || [];
        const allContrib = [...promptContrib, ...responseContrib];

        traces.push({
            x: Array.from({length: allContrib.length}, (_, i) => i),
            y: allContrib,
            mode: 'lines+markers',
            name: `Head ${headIdx}`,
            line: { color: colors[headIdx % colors.length], width: 2 },
            marker: { size: 4 },
            hovertemplate: 'Token %{x}: %{text}<br>Head ' + headIdx + ': %{y:.3f}<extra></extra>',
            text: allTokens
        });
    }

    // Add separator line
    const shapes = [];
    if (promptLength) {
        shapes.push({
            type: 'line',
            xref: 'x',
            yref: 'paper',
            x0: promptLength - 0.5,
            x1: promptLength - 0.5,
            y0: 0,
            y1: 1,
            line: {
                color: 'rgba(255, 255, 255, 0.3)',
                width: 2,
                dash: 'dash'
            }
        });
    }

    const layout = window.getPlotlyLayout({
        title: 'Per-Head Trait Contributions (Which Heads Drive the Trait)',
        xaxis: { title: 'Token Position' },
        yaxis: { title: 'Head Contribution to Trait' },
        height: 400,
        shapes: shapes,
        showlegend: true,
        legend: {
            orientation: 'h',
            yanchor: 'bottom',
            y: -0.3,
            xanchor: 'center',
            x: 0.5
        }
    });

    Plotly.newPlot('head-contributions', traces, layout, { displayModeBar: false });
}

function renderAttentionHeatmaps(data, promptLength, tokenIdx = 0) {
    // Show attention patterns for a single query token
    // Dynamically handles any number of heads (Gemma 2B has 8 query heads with GQA)

    const allTokens = [...data.prompt.tokens, ...data.response.tokens];
    const isPromptToken = tokenIdx < promptLength;

    let attnWeights;  // [n_heads, seq_len] - just the row for this query token
    let contextTokens;

    if (isPromptToken) {
        // Query is in prompt - extract row from prompt attention
        const promptAttn = data.internals.prompt.attn_weights;  // [n_heads, prompt_len, prompt_len]
        attnWeights = promptAttn.map(head => head[tokenIdx]);  // [n_heads, prompt_len]
        contextTokens = data.prompt.tokens;
    } else {
        // Query is in response - extract row from response attention
        const responseIdx = tokenIdx - promptLength;
        const responseAttn = data.internals.response.attn_weights[responseIdx];  // [n_heads, context_len, context_len]
        const contextLen = promptLength + responseIdx + 1;
        attnWeights = responseAttn.map(head => head[head.length - 1]);  // Last row = this token's attention
        contextTokens = allTokens.slice(0, contextLen);  // All context up to this point
    }

    const nHeads = attnWeights.length;
    const nCols = Math.min(4, nHeads);  // Up to 4 columns
    const nRows = Math.ceil(nHeads / nCols);

    // Head colors (consistent with trait colors)
    const headColors = ['#4a9eff', '#ff6b6b', '#51cf66', '#ffd43b', '#cc5de8', '#ff922b', '#20c997', '#f06595'];

    const html = `
        <div style="margin-bottom: 20px; padding: 15px; background: var(--bg-secondary); border-radius: 8px;">
            <h3 style="margin: 0 0 15px 0; color: var(--text-primary);">Attention Patterns (${nHeads} Heads)</h3>

            <!-- Slider -->
            <div style="margin-bottom: 10px;">
                <label style="color: var(--text-secondary); font-size: 13px;">
                    Query Token: <span id="attn-slider-value" style="color: var(--text-primary); font-weight: 600;">${tokenIdx}</span> -
                    "<span id="attn-slider-token" style="color: var(--text-primary); font-weight: 600;">${allTokens[tokenIdx]}</span>"
                </label>
            </div>
            <input type="range" id="attn-slider" min="0" max="${allTokens.length - 1}" value="${tokenIdx}"
                   style="width: 100%; height: 4px; border-radius: 2px; background: var(--border-color); cursor: pointer;">

            <div style="margin-top: 10px; color: var(--text-secondary); font-size: 12px;">
                Attending to ${contextTokens.length} context tokens
            </div>
        </div>

        <!-- Head Patterns in Grid -->
        <div style="display: grid; grid-template-columns: repeat(${nCols}, 1fr); gap: 12px;">
            ${Array.from({length: nHeads}, (_, i) => `
                <div style="background: var(--bg-secondary); border-radius: 6px; padding: 8px;">
                    <div style="color: ${headColors[i % headColors.length]}; font-size: 11px; font-weight: 600; margin-bottom: 4px;">Head ${i}</div>
                    <div id="attn-head-${i}"></div>
                </div>
            `).join('')}
        </div>
    `;

    document.getElementById('attention-heatmaps').innerHTML = html;

    // Render each head as a 1-row heatmap
    for (let head = 0; head < nHeads; head++) {
        const headAttn = attnWeights[head];  // [context_len]

        // Create 1-row heatmap with query position highlighted
        const trace = {
            z: [headAttn],  // Single row
            x: contextTokens,
            y: [''],
            type: 'heatmap',
            colorscale: [
                [0, 'rgba(0,0,0,0)'],
                [0.5, headColors[head % headColors.length] + '80'],
                [1, headColors[head % headColors.length]]
            ],
            hovertemplate: 'To: %{x}<br>Weight: %{z:.3f}<extra></extra>',
            showscale: false
        };

        // Highlight the query position
        const shapes = [];
        const queryPosInContext = isPromptToken ? tokenIdx : headAttn.length - 1;
        shapes.push({
            type: 'rect',
            xref: 'x',
            yref: 'paper',
            x0: queryPosInContext - 0.5,
            x1: queryPosInContext + 0.5,
            y0: 0,
            y1: 1,
            line: { color: '#ffffff', width: 2 },
            fillcolor: 'rgba(255, 255, 255, 0.2)'
        });

        const layout = {
            xaxis: {
                tickangle: -45,
                tickfont: { size: 8, color: 'var(--text-secondary)' },
                showgrid: false
            },
            yaxis: { showticklabels: false, showgrid: false },
            height: 60,
            margin: { l: 5, r: 5, t: 0, b: 40 },
            shapes: shapes,
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)'
        };

        Plotly.newPlot(`attn-head-${head}`, [trace], layout, { displayModeBar: false });
    }

    // Add slider event listener
    document.getElementById('attn-slider').addEventListener('input', (e) => {
        const newTokenIdx = parseInt(e.target.value);
        document.getElementById('attn-slider-value').textContent = newTokenIdx;
        document.getElementById('attn-slider-token').textContent = allTokens[newTokenIdx];
        renderAttentionHeatmaps(data, promptLength, newTokenIdx);
    });
}

// ============================================================================
// Neuron Visualization
// ============================================================================

function renderTopNeurons(divId, title, geluActivations, tokens, promptLength = null) {
    // Store for access by slider callback
    window.currentLayerData = { data: window.currentLayerData?.data, promptLength };
    // geluActivations: [n_tokens, 9216]
    // Show per-token heatmap with slider
    // promptLength: if provided, adds visual separator between prompt and response

    if (!geluActivations || geluActivations.length === 0) {
        document.getElementById(divId).innerHTML = '<div style="color: var(--text-secondary);">No data</div>';
        return;
    }

    const nTokens = geluActivations.length;
    const nNeurons = geluActivations[0].length;

    // Find top 50 neurons by average activation magnitude
    const neuronAvg = new Array(nNeurons).fill(0);
    for (let t = 0; t < nTokens; t++) {
        for (let n = 0; n < nNeurons; n++) {
            neuronAvg[n] += Math.abs(geluActivations[t][n]);
        }
    }
    for (let n = 0; n < nNeurons; n++) {
        neuronAvg[n] /= nTokens;
    }

    const neuronIndices = Array.from({length: nNeurons}, (_, i) => i);
    neuronIndices.sort((a, b) => neuronAvg[b] - neuronAvg[a]);
    const topNeurons = neuronIndices.slice(0, 50);

    // Create heatmap data: [50_neurons, n_tokens]
    const heatmapData = topNeurons.map(neuronIdx => {
        return geluActivations.map(tokenActivations => tokenActivations[neuronIdx]);
    });

    // Create unique div IDs
    const heatmapId = `${divId}-heatmap`;
    const sliderId = `${divId}-slider`;
    const sliderValueId = `${divId}-slider-value`;
    const barChartId = `${divId}-bar`;

    const html = `
        <div style="margin-bottom: 30px;">
            <h3 style="margin-bottom: 15px; color: var(--text-primary);">${title}</h3>

            <div style="margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <label style="color: var(--text-secondary); font-weight: 600;">
                        Token Position: <span id="${sliderValueId}" style="color: var(--primary-color);">0</span> / ${nTokens - 1}
                    </label>
                    <span style="color: var(--text-secondary); font-size: 12px;">
                        Token: "<span id="${sliderValueId}-token" style="font-weight: 600;">${tokens[0]}</span>"
                    </span>
                </div>
                <input type="range" id="${sliderId}" min="0" max="${nTokens - 1}" value="0"
                       style="width: 100%; height: 3px; border-radius: 2px; background: var(--border-color);">
            </div>

            <div id="${heatmapId}" style="margin-bottom: 30px;"></div>
            <div id="${barChartId}"></div>
        </div>
    `;

    document.getElementById(divId).innerHTML = html;

    // Render heatmap (overview of all tokens × top neurons)
    const heatmapTrace = {
        z: heatmapData,
        x: tokens,
        y: topNeurons.map(n => `N${n}`),
        type: 'heatmap',
        colorscale: 'RdBu',
        zmid: 0,
        hovertemplate: 'Token: %{x}<br>Neuron: %{y}<br>Activation: %{z:.3f}<extra></extra>',
        colorbar: {
            title: 'Activation'
        }
    };

    // Build shapes array for heatmap
    const shapes = [{
        type: 'rect',
        xref: 'x',
        yref: 'paper',
        x0: -0.5,
        x1: 0.5,
        y0: 0,
        y1: 1,
        fillcolor: 'rgba(74, 158, 255, 0.15)',
        line: { width: 0 }
    }];

    // Add separator line between prompt and response if provided
    if (promptLength !== null) {
        shapes.push({
            type: 'line',
            xref: 'x',
            yref: 'paper',
            x0: promptLength - 0.5,
            x1: promptLength - 0.5,
            y0: 0,
            y1: 1,
            line: {
                color: 'rgba(255, 255, 255, 0.5)',
                width: 2,
                dash: 'dash'
            }
        });
    }

    // Add annotations for prompt/response regions if applicable
    const annotations = [];
    if (promptLength !== null) {
        annotations.push({
            x: (promptLength - 1) / 2,
            y: 1.05,
            xref: 'x',
            yref: 'paper',
            text: 'Prompt',
            showarrow: false,
            font: { size: 12, color: 'var(--text-secondary)' },
            xanchor: 'center'
        });
        annotations.push({
            x: promptLength + (nTokens - promptLength - 1) / 2,
            y: 1.05,
            xref: 'x',
            yref: 'paper',
            text: 'Response',
            showarrow: false,
            font: { size: 12, color: 'var(--text-secondary)' },
            xanchor: 'center'
        });
    }

    const heatmapLayout = window.getPlotlyLayout({
        title: 'Neuron Activations Across All Tokens (Top 50 Neurons)',
        xaxis: { title: 'Token', tickangle: -45 },
        yaxis: { title: 'Neuron', autorange: 'reversed' },
        height: 400,
        shapes: shapes,
        annotations: annotations
    });

    Plotly.newPlot(heatmapId, [heatmapTrace], heatmapLayout, { displayModeBar: false });

    // Function to update bar chart for selected token
    function updateBarChart(tokenIdx) {
        const tokenActivations = geluActivations[tokenIdx];
        const topActivations = topNeurons.map(n => tokenActivations[n]);

        const barTrace = {
            x: topNeurons.map(n => `N${n}`),
            y: topActivations,
            type: 'bar',
            marker: {
                color: topActivations.map(a => a > 0 ? '#4caf50' : '#f44336')
            },
            hovertemplate: 'Neuron: %{x}<br>Activation: %{y:.3f}<extra></extra>'
        };

        Plotly.newPlot(barChartId, [barTrace], window.getPlotlyLayout({
            title: `Neuron Activations for Token "${tokens[tokenIdx]}" (Position ${tokenIdx})`,
            xaxis: { title: 'Neuron Index', tickangle: -45 },
            yaxis: { title: 'Activation' },
            height: 350
        }), { displayModeBar: false });

        // Update heatmap highlight
        const newShapes = [{
            type: 'rect',
            xref: 'x',
            yref: 'paper',
            x0: tokenIdx - 0.5,
            x1: tokenIdx + 0.5,
            y0: 0,
            y1: 1,
            fillcolor: 'rgba(74, 158, 255, 0.15)',
            line: { width: 0 }
        }];

        // Add separator line if provided
        if (promptLength !== null) {
            newShapes.push({
                type: 'line',
                xref: 'x',
                yref: 'paper',
                x0: promptLength - 0.5,
                x1: promptLength - 0.5,
                y0: 0,
                y1: 1,
                line: {
                    color: 'rgba(255, 255, 255, 0.5)',
                    width: 2,
                    dash: 'dash'
                }
            });
        }

        // Preserve annotations when updating
        const layoutUpdate = { shapes: newShapes };
        if (annotations.length > 0) {
            layoutUpdate.annotations = annotations;
        }
        Plotly.relayout(heatmapId, layoutUpdate);

        // Update slider label
        document.getElementById(sliderValueId).textContent = tokenIdx;
        document.getElementById(`${sliderValueId}-token`).textContent = tokens[tokenIdx];
    }

    // Initialize with first token
    updateBarChart(0);

    // Add slider event listener
    document.getElementById(sliderId).addEventListener('input', (e) => {
        updateBarChart(parseInt(e.target.value));
    });
}

// Setup event listeners
function setupEventListeners() {
    // Experiment selection is now handled in loadExperiments()

    // Theme toggle
    document.getElementById('theme-toggle').addEventListener('click', toggleTheme);

    // Info tooltip
    document.getElementById('info-btn').addEventListener('click', (e) => {
        e.stopPropagation();
        toggleInfo();
    });

    // Select all traits button
    document.getElementById('select-all-btn').addEventListener('click', toggleAllTraits);

    // Prompt selector
    document.getElementById('prompt-selector').addEventListener('change', (e) => {
        window.state.currentPrompt = parseInt(e.target.value);
        console.log(`Prompt changed to: ${window.state.currentPrompt}`);
        renderView();
    });
}

// Utility functions
function showError(message) {
    document.getElementById('content-area').innerHTML = `
        <div class="error">
            <strong>Error:</strong> ${message}
            <br><br>
            Make sure you're running a local server from the trait-interp root directory:
            <pre style="background: var(--bg-secondary); color: var(--text-primary); padding: 10px; border-radius: 4px; margin-top: 10px;">cd trait-interp
python -m http.server 8000</pre>
            Then visit: <a href="http://localhost:8000/visualization/">http://localhost:8000/visualization/</a>
        </div>
    `;
}

// Initialize on page load
init();

// Export to global scope
window.renderLayerDeepDive = renderLayerDeepDive;
