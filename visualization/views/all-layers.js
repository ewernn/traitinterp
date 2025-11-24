// Monitoring View

async function renderAllLayers() {
    const contentArea = document.getElementById('content-area');
    const filteredTraits = window.getFilteredTraits();

    if (filteredTraits.length === 0) {
        contentArea.innerHTML = `
            <div class="card">
                <div class="card-title">All Layers</div>
                <div class="info">Select at least one trait to view trajectories</div>
            </div>
        `;
        return;
    }

    // Load data for ALL selected traits
    contentArea.innerHTML = '<div id="all-traits-container"></div>';
    const container = document.getElementById('all-traits-container');

    // Use global PathBuilder singleton with experiment set
    window.paths.setExperiment(window.state.experimentData.name);

    for (const trait of filteredTraits) {
        // Create a unique div for this trait
        // Sanitize trait name - replace slashes with dashes for valid HTML IDs
        const sanitizedName = trait.name.replace(/\//g, '-');
        const traitDiv = document.createElement('div');
        traitDiv.id = `trait-${sanitizedName}`;
        traitDiv.style.marginBottom = '20px';
        container.appendChild(traitDiv);

        // Try to load the selected prompt using global PathBuilder
        const promptSet = window.state.currentPromptSet;
        const promptId = window.state.currentPromptId;

        if (!promptSet || !promptId) {
            renderAllLayersInstructionsInContainer(traitDiv.id, trait, promptSet, promptId);
            continue;
        }

        try {
            const fetchPath = window.paths.residualStreamData(trait, promptSet, promptId);
            console.log(`[${trait.name}] Fetching trajectory data for ${promptSet}/${promptId}`);
            const response = await fetch(fetchPath);

            if (!response.ok) {
                console.log(`[${trait.name}] No data available for ${promptSet}/${promptId} (${response.status})`);
                renderAllLayersInstructionsInContainer(traitDiv.id, trait, promptSet, promptId);
                continue;
            }

            const data = await response.json();
            console.log(`[${trait.name}] Data loaded successfully for ${promptSet}/${promptId}`);
            renderAllLayersDataInContainer(traitDiv.id, trait, data);
        } catch (error) {
            console.log(`[${trait.name}] Load failed for ${promptSet}/${promptId}:`, error.message);
            renderAllLayersInstructionsInContainer(traitDiv.id, trait, promptSet, promptId);
        }
    }
}

function renderAllLayersInstructionsInContainer(containerId, trait, promptSet, promptId) {
    const container = document.getElementById(containerId);
    if (!container) {
        console.error(`Container ${containerId} not found`);
        return;
    }

    // Get layer count from metadata
    const nLayers = trait.metadata?.n_layers || 26;
    const nCheckpoints = nLayers * 3;
    const promptLabel = promptSet && promptId ? `${promptSet}/${promptId}` : 'none selected';

    container.innerHTML = `
        <div style="margin-bottom: 4px;">
            <span style="color: var(--text-primary); font-size: 14px; font-weight: 600;">${window.getDisplayName(trait.name)}</span>
            <span style="color: var(--text-tertiary); font-size: 11px; margin-left: 8px;">⚠️ No data for prompt ${promptLabel}</span>
        </div>
        <div style="color: var(--text-secondary); font-size: 11px; margin-bottom: 4px;">
            The file <code>${promptId}.json</code> does not exist for this trait in ${promptSet || 'the prompt set'}. You may need to run inference with this prompt.
        </div>
        <div style="color: var(--text-secondary); font-size: 11px; margin-bottom: 4px;">
            To capture per-token projections at all ${nCheckpoints} checkpoints (${nLayers} layers × 3 sublayers):
        </div>
        <pre style="background: var(--bg-secondary); color: var(--text-primary); padding: 8px; border-radius: 4px; margin: 0; overflow-x: auto; font-size: 10px;">python inference/capture.py --experiment ${window.state.experimentData.name} --prompt-set ${promptSet || 'PROMPT_SET'}</pre>
    `;
}

function renderAllLayersDataInContainer(containerId, trait, data) {
    const container = document.getElementById(containerId);
    if (!container) {
        console.error(`Container ${containerId} not found`);
        return;
    }

    try {
        const promptProj = data.projections.prompt;  // [n_tokens, n_layers, 3]
        const responseProj = data.projections.response;
        console.log('Prompt projections shape:', promptProj.length, 'tokens x', promptProj[0].length, 'layers x', promptProj[0][0].length, 'sublayers');
        console.log('Response projections shape:', responseProj.length, 'tokens x', responseProj[0].length, 'layers x', responseProj[0][0].length, 'sublayers');

    // Combine prompt and response projections and tokens
    const allTokens = [...data.prompt.tokens, ...data.response.tokens];
    const allProj = [...promptProj, ...responseProj];
    const nPromptTokens = data.prompt.n_tokens;
    const nTotalTokens = allTokens.length;
    const nLayers = promptProj[0].length;  // Detect from data
    const nCheckpoints = nLayers * 3;

    // Create unique IDs for this trait's elements
    // Sanitize trait name - replace slashes with dashes for valid HTML IDs
    const traitId = trait.name.replace(/\//g, '-');
    const trajectoryId = `trajectory-heatmap-${traitId}`;
    const sliderId = `unified-slider-${traitId}`;
    const sliderValueId = `unified-slider-value-${traitId}`;
    const sliderTokenId = `unified-slider-token-${traitId}`;
    const sliderPhaseId = `unified-slider-phase-${traitId}`;
    const logitContainerId = `logit-lens-container-${traitId}`;
    const logitTokenNumId = `logit-token-num-${traitId}`;
    const logitTokenTextId = `logit-token-text-${traitId}`;
    const logitPlotId = `logit-lens-plot-${traitId}`;
    const attentionViewerId = `attention-viewer-${traitId}`;

    container.innerHTML = `
        <div style="margin-bottom: 4px;">
            <span style="color: var(--text-primary); font-size: 14px; font-weight: 600;">${window.getDisplayName(trait.name)}</span>
        </div>
        <div style="color: var(--text-secondary); font-size: 10px; margin-bottom: 8px;">
            ${data.prompt.text} ${data.response.text}
        </div>
        <div id="${trajectoryId}"></div>
    `;

    // Render combined trajectory heatmap with separator line
    setTimeout(() => {
        try {
            renderCombinedTrajectoryHeatmap(trajectoryId, allProj, allTokens, nPromptTokens, 250);  // Reduced height
        } catch (plotError) {
            console.error(`[${trait.name}] Heatmap rendering failed:`, plotError);
            container.innerHTML += `<div class="info" style="color: var(--danger);">Failed to render heatmap: ${plotError.message}</div>`;
        }
    }, 0);

    // Math rendering removed (renderMath function no longer exists)
    } catch (error) {
        console.error(`[${trait.name}] Error rendering trajectory data:`, error);
        if (container) {
            container.innerHTML = `<div class="card"><div class="card-title">Error: ${window.getDisplayName(trait.name)}</div><div class="info">Failed to render trajectory data: ${error.message}</div></div>`;
        }
    }
}

function renderCombinedTrajectoryHeatmap(divId, projections, tokens, nPromptTokens, height = 400) {
    // projections: [n_tokens, n_layers, 3_sublayers]
    // We'll show layer-averaged (average over 3 sublayers)

    const nTokens = projections.length;
    const nLayers = projections[0].length;  // Dynamically get number of layers
    console.log(`Rendering combined trajectory heatmap for ${nTokens} tokens x ${nLayers} layers (${nPromptTokens} prompt + ${nTokens - nPromptTokens} response)`);

    // Average over sublayers to get [n_tokens, n_layers]
    // Skip BOS token (index 0) for better visualization dynamic range
    const startIdx = 1;  // Skip <bos>

    const layerAvg = [];
    for (let t = startIdx; t < nTokens; t++) {
        layerAvg[t - startIdx] = [];
        for (let l = 0; l < nLayers; l++) {
            const avg = (projections[t][l][0] + projections[t][l][1] + projections[t][l][2]) / 3;
            layerAvg[t - startIdx][l] = avg;
        }
    }

    // Transpose for heatmap: [n_layers, n_tokens-1] (excluding BOS)
    const heatmapData = [];
    const nDisplayTokens = nTokens - startIdx;
    for (let l = 0; l < nLayers; l++) {
        heatmapData[l] = [];
        for (let t = 0; t < nDisplayTokens; t++) {
            heatmapData[l][t] = layerAvg[t][l];
        }
    }

    console.log(`[${divId}] Heatmap data sample:`, heatmapData[0].slice(0, 5));

    // Create shapes array for separator line and current token highlight
    const shapes = [
        // Vertical line separating prompt and response (adjusted for skipped BOS)
        {
            type: 'line',
            xref: 'x',
            yref: 'paper',
            x0: (nPromptTokens - startIdx) - 0.5,
            x1: (nPromptTokens - startIdx) - 0.5,
            y0: 0,
            y1: 1,
            line: {
                color: 'rgba(255, 255, 255, 0.5)',
                width: 2,
                dash: 'dash'
            }
        },
        // Highlight for current token (will be updated by slider)
        {
            type: 'rect',
            xref: 'x',
            yref: 'paper',
            x0: -0.5,
            x1: 0.5,
            y0: 0,
            y1: 1,
            fillcolor: 'rgba(74, 158, 255, 0.2)',
            line: { width: 0 },
            name: 'token-highlight'  // ID for updating
        }
    ];

    const data = [{
        z: heatmapData,
        x: tokens.slice(startIdx),  // Skip BOS in token labels too
        y: Array.from({length: nLayers}, (_, i) => `L${i}`),
        type: 'heatmap',
        colorscale: 'RdBu',
        zmid: 0,
        hovertemplate: 'Token: %{x}<br>Layer: %{y}<br>Score: %{z:.2f}<extra></extra>'
    }];

    // Adjust margins based on height
    const isCompact = height < 300;
    const layout = {
        title: isCompact ? '' : 'Trait Trajectory',
        xaxis: {
            title: isCompact ? '' : 'Tokens',
            tickangle: -45,
            tickfont: { size: isCompact ? 9 : 12 }
        },
        yaxis: {
            title: isCompact ? '' : 'Layer',
            tickfont: { size: isCompact ? 9 : 12 }
        },
        shapes: shapes,
        height: height,
        margin: isCompact ? { t: 5, b: 30, l: 30, r: 5 } : { t: 40, b: 50, l: 40, r: 10 },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: {
            color: getComputedStyle(document.documentElement).getPropertyValue('--text-primary').trim()
        }
    };

    Plotly.newPlot(divId, data, layout, {displayModeBar: false});
}

function renderAttentionViewer(divId, data, allTokens, nPromptTokens) {
    // Get number of layers
    const nLayers = Object.keys(data.attention_weights.prompt).length;

    const html = `
        <div style="padding-top: 15px; margin-top: 32px;">
            <h3 style="color: var(--text-primary); margin-bottom: 5px; font-size: 14px;">Attention Patterns</h3>
            <p style="color: var(--text-secondary); margin-bottom: 10px; font-size: 11px;">
                Selected token's attention to context (controlled by slider above)
            </p>
            <div id="attention-heatmap"></div>
        </div>
    `;

    document.getElementById(divId).innerHTML = html;

    // Function to render attention heatmap for a specific token
    window.updateAttentionHeatmap = function(tokenIdx) {
        // Get attention data for this token
        let attnData;  // [n_layers, n_context]
        let contextTokens;

        if (tokenIdx < nPromptTokens) {
            // Prompt token - extract from full attention matrix
            // Data structure: prompt[layer_X] is [n_tokens, n_tokens] matrix
            attnData = [];
            for (let layer = 0; layer < nLayers; layer++) {
                const layerAttnMatrix = data.attention_weights.prompt[`layer_${layer}`];

                if (!layerAttnMatrix || !layerAttnMatrix[tokenIdx]) {
                    console.warn(`Missing attention data for prompt token ${tokenIdx}, layer ${layer}`);
                    attnData.push(new Array(tokenIdx + 1).fill(0));
                    continue;
                }

                // Extract this token's attention row, only up to current position (causal)
                const tokenAttn = layerAttnMatrix[tokenIdx].slice(0, tokenIdx + 1);
                attnData.push(tokenAttn);
            }
            contextTokens = data.prompt.tokens.slice(0, tokenIdx + 1);
        } else {
            // Response token - already stored as single attention vector
            // Data structure: response[token_idx][layer_X] is [context_length] vector
            const responseIdx = tokenIdx - nPromptTokens;
            attnData = [];
            for (let layer = 0; layer < nLayers; layer++) {
                const stepAttn = data.attention_weights.response[responseIdx];

                if (!stepAttn || !stepAttn[`layer_${layer}`]) {
                    console.warn(`Missing attention data for response token ${responseIdx}, layer ${layer}`);
                    attnData.push(new Array(tokenIdx + 1).fill(0));
                    continue;
                }

                const layerAttn = stepAttn[`layer_${layer}`];
                attnData.push(layerAttn);
            }
            contextTokens = allTokens.slice(0, tokenIdx + 1);
        }

        // Validate data dimensions
        const expectedContextLen = tokenIdx + 1;
        let hasValidData = true;
        for (let i = 0; i < attnData.length; i++) {
            if (!attnData[i] || attnData[i].length !== expectedContextLen) {
                console.warn(`Layer ${i} attention length mismatch: expected ${expectedContextLen}, got ${attnData[i]?.length || 0}`);
                hasValidData = false;
            }
        }

        if (!hasValidData) {
            console.error('Attention data validation failed, displaying with available data');
        }

        // Create heatmap
        const trace = {
            z: attnData,
            x: contextTokens,
            y: Array.from({length: nLayers}, (_, i) => `L${i}`),
            type: 'heatmap',
            colorscale: 'Viridis',
            hovertemplate: 'Context Token: %{x}<br>Layer: %{y}<br>Attention: %{z:.4f}<extra></extra>',
            colorbar: {
                title: 'Attention Weight'
            }
        };

        const layout = window.getPlotlyLayout({
            title: `"${allTokens[tokenIdx]}" (pos ${tokenIdx})`,
            xaxis: {
                title: 'Context',
                tickangle: -45,
                side: 'bottom'
            },
            yaxis: {
                title: 'Layer'
            },
            height: 300,
            margin: { t: 10, b: 50, l: 40, r: 10 }
        });

        Plotly.newPlot('attention-heatmap', [trace], layout, { displayModeBar: false });
    }

    // Initialize with first token
    window.updateAttentionHeatmap(0);
}

function renderLogitLens(data, tokenIdx, allTokens, nPromptTokens) {
    const container = document.getElementById('logit-lens-container');

    // Check if logit lens data is available
    if (!data.logit_lens || !data.logit_lens.response) {
        container.style.display = 'none';
        return;
    }

    container.style.display = 'block';

    // Determine if this is a prompt or response token
    const isPromptToken = tokenIdx < nPromptTokens;
    const logitData = isPromptToken ? data.logit_lens.prompt : data.logit_lens.response;
    const actualIdx = isPromptToken ? tokenIdx : (tokenIdx - nPromptTokens);

    // Update token info
    const tokenText = allTokens[tokenIdx];
    document.getElementById('logit-token-num').textContent = tokenIdx + 1;
    document.getElementById('logit-token-text').textContent = tokenText;

    // Layer indices from LOGIT_LENS_LAYERS config: [0,1,2,3,6,9,12,15,18,21,24,25]
    const layerKeys = Object.keys(logitData).sort((a, b) => {
        const aNum = parseInt(a.replace('layer_', ''));
        const bNum = parseInt(b.replace('layer_', ''));
        return aNum - bNum;
    });

    const layerIndices = layerKeys.map(k => parseInt(k.replace('layer_', '')));

    // Build traces for top-3 predictions
    const traces = [];
    const colors = ['#4a9eff', '#ff6b6b', '#51cf66'];  // Blue, Red, Green

    for (let k = 0; k < 3; k++) {
        const probs = [];
        const tokens = [];

        for (const layerKey of layerKeys) {
            const layerData = logitData[layerKey];

            if (layerData && layerData.tokens[actualIdx]) {
                probs.push(layerData.probs[actualIdx][k]);
                if (tokens.length === 0) {
                    tokens.push(layerData.tokens[actualIdx][k]);
                }
            } else {
                // Data not available for this layer/token
                probs.push(null);
            }
        }

        traces.push({
            x: layerIndices,
            y: probs,
            name: `"${tokens[0] || '?'}"`,
            mode: 'lines+markers',
            line: { width: 3, color: colors[k] },
            marker: { size: 8, color: colors[k] },
            connectgaps: false
        });
    }

    // Layout
    const layout = {
        xaxis: {
            title: 'Layer',
            tickvals: layerIndices,
            gridcolor: 'rgba(128, 128, 128, 0.2)',
            zeroline: false
        },
        yaxis: {
            title: 'Prob',
            range: [0, 1],
            gridcolor: 'rgba(128, 128, 128, 0.2)',
            zeroline: false
        },
        height: 200,
        margin: { t: 10, b: 40, l: 40, r: 10 },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: {
            color: getComputedStyle(document.documentElement).getPropertyValue('--text-primary').trim(),
            size: 10
        },
        showlegend: true,
        legend: {
            orientation: 'h',
            yanchor: 'bottom',
            y: -0.35,
            xanchor: 'center',
            x: 0.5,
            font: { size: 10 }
        }
    };

    Plotly.newPlot('logit-lens-plot', traces, layout, { responsive: true, displayModeBar: false });
}

function setupUnifiedSlider(data, allTokens, nPromptTokens, nTotalTokens) {
    const slider = document.getElementById('unified-slider');

    // Function to update all visualizations when slider moves
    function updateAllVisualizations(tokenIdx) {
        // Update slider labels
        document.getElementById('unified-slider-value').textContent = tokenIdx;
        document.getElementById('unified-slider-token').textContent = allTokens[tokenIdx];

        // Update phase indicator (Prompt vs Response)
        const phase = tokenIdx < nPromptTokens ? '[Prompt]' : '[Response]';
        document.getElementById('unified-slider-phase').textContent = phase;

        // Heatmap position (adjusted for skipped BOS token)
        const startIdx = 1;  // BOS is skipped in heatmap
        const heatmapIdx = tokenIdx - startIdx;

        // Update trajectory heatmap highlight
        const shapes = [
            // Separator line (adjusted for skipped BOS)
            {
                type: 'line',
                xref: 'x',
                yref: 'paper',
                x0: (nPromptTokens - startIdx) - 0.5,
                x1: (nPromptTokens - startIdx) - 0.5,
                y0: 0,
                y1: 1,
                line: {
                    color: 'rgba(255, 255, 255, 0.5)',
                    width: 2,
                    dash: 'dash'
                }
            },
            // Current token highlight (adjusted position for skipped BOS)
            {
                type: 'rect',
                xref: 'x',
                yref: 'paper',
                x0: heatmapIdx - 0.5,
                x1: heatmapIdx + 0.5,
                y0: 0,
                y1: 1,
                fillcolor: 'rgba(74, 158, 255, 0.2)',
                line: { width: 0 }
            }
        ];

        Plotly.relayout('trajectory-heatmap', { shapes: shapes });

        // Update logit lens if available
        renderLogitLens(data, tokenIdx, allTokens, nPromptTokens);

        // Update attention heatmap if available
        if (window.updateAttentionHeatmap) {
            window.updateAttentionHeatmap(tokenIdx);
        }
    }

    // Add slider event listener
    slider.addEventListener('input', (e) => {
        updateAllVisualizations(parseInt(e.target.value));
    });

    // Initialize at position 1 (skip BOS)
    updateAllVisualizations(1);
}

// Render Prompt Activation view
// Export to global scope
window.renderAllLayers = renderAllLayers;
