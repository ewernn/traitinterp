// Monitoring View

async function renderTraitTrajectory() {
    const contentArea = document.getElementById('content-area');
    const filteredTraits = window.getFilteredTraits();

    if (filteredTraits.length === 0) {
        contentArea.innerHTML = `
            <div class="tool-view">
                <div class="info">Select at least one trait to view trajectories</div>
            </div>
        `;
        return;
    }

    // Load data for ALL selected traits
    contentArea.innerHTML = `
        <div class="tool-view">
            <div class="page-intro">
                <div class="page-intro-text">View trait activations across all layers for a prompt.</div>
            </div>
            <div id="all-traits-container"></div>
        </div>
    `;
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
            const response = await fetch(fetchPath);

            if (!response.ok) {
                console.warn(`[${trait.name}] No projection data for ${promptSet}/${promptId}`);
                renderAllLayersInstructionsInContainer(traitDiv.id, trait, promptSet, promptId);
                continue;
            }

            const data = await response.json();
            renderAllLayersDataInContainer(traitDiv.id, trait, data);
        } catch (error) {
            console.warn(`[${trait.name}] Failed to load ${promptSet}/${promptId}:`, error.message);
            renderAllLayersInstructionsInContainer(traitDiv.id, trait, promptSet, promptId);
        }
    }

    // Add token highlights after all heatmaps are rendered
    // Prompt picker owns the highlight state; we just trigger the update
    const nPromptTokens = window.state.promptPickerCache?.nPromptTokens || 0;
    if (window.updatePlotTokenHighlights) {
        requestAnimationFrame(() => {
            window.updatePlotTokenHighlights(window.state.currentTokenIndex, nPromptTokens);
        });
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
        <div class="card">
            <h4>${window.getDisplayName(trait.name)} <small style="color: var(--text-tertiary); font-weight: normal;">⚠️ No data for ${promptLabel}</small></h4>
            <p style="color: var(--text-secondary); font-size: 11px; margin: 4px 0;">
                Missing projection data for <code>${trait.name}</code>. This usually means the trait doesn't have extraction vectors yet.
            </p>
            <p style="color: var(--text-secondary); font-size: 11px; margin: 4px 0;">
                1. Check vectors exist: <code>ls experiments/${window.state.experimentData.name}/extraction/${trait.name}/vectors/</code>
            </p>
            <p style="color: var(--text-secondary); font-size: 11px; margin: 4px 0;">
                2. If missing, run extraction first. If vectors exist, re-run projection:
            </p>
            <pre>python inference/project_raw_activations_onto_traits.py --experiment ${window.state.experimentData.name} --prompt-set ${promptSet || 'PROMPT_SET'}</pre>
        </div>
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

    // Combine prompt and response projections and tokens
    const allTokens = [...data.prompt.tokens, ...data.response.tokens];
    const allProj = [...promptProj, ...responseProj];
    const nPromptTokens = data.prompt.tokens.length;
    const nLayers = promptProj[0].length;

    // Create unique ID for this trait's heatmap
    // Sanitize trait name - replace slashes with dashes for valid HTML IDs
    const traitId = trait.name.replace(/\//g, '-');
    const trajectoryId = `trajectory-heatmap-${traitId}`;

    container.innerHTML = `
        <h4 style="margin-bottom: 4px;">${window.getDisplayName(trait.name)}</h4>
        <div id="${trajectoryId}" style="width: 100%;"></div>
    `;

    // Render combined trajectory heatmap with separator line
    setTimeout(() => {
        try {
            renderCombinedTrajectoryHeatmap(trajectoryId, allProj, allTokens, nPromptTokens, 250);
        } catch (plotError) {
            console.error(`[${trait.name}] Heatmap rendering failed:`, plotError);
            container.innerHTML += `<div class="info error">Failed to render heatmap: ${plotError.message}</div>`;
        }
    }, 0);

    // Math rendering removed (renderMath function no longer exists)
    } catch (error) {
        console.error(`[${trait.name}] Error rendering trajectory data:`, error);
        if (container) {
            container.innerHTML = `<div class="info error">Error rendering ${window.getDisplayName(trait.name)}: ${error.message}</div>`;
        }
    }
}

function renderCombinedTrajectoryHeatmap(divId, projections, tokens, nPromptTokens, height = 400) {
    // projections: [n_tokens, n_layers, 3_sublayers]
    // We'll show layer-averaged (average over 3 sublayers)

    const nTokens = projections.length;
    const nLayers = projections[0].length;

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

    // Shapes (separator + token highlight) are added by prompt-picker after render
    // This keeps prompt-picker as the single owner of token highlight state

    const data = [{
        z: heatmapData,
        x: tokens.slice(startIdx),  // Skip BOS in token labels too
        y: Array.from({length: nLayers}, (_, i) => `L${i}`),
        type: 'heatmap',
        colorscale: window.ASYMB_COLORSCALE,
        zmid: 0,
        hovertemplate: 'Token: %{x}<br>Layer: %{y}<br>Score: %{z:.2f}<extra></extra>'
    }];

    // Adjust margins based on height
    const isCompact = height < 300;
    const layout = window.getPlotlyLayout({
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
        height: height,
        margin: isCompact ? { t: 5, b: 30, l: 30, r: 5 } : { t: 40, b: 50, l: 40, r: 10 }
    });

    Plotly.newPlot(divId, data, layout, { displayModeBar: false, responsive: true });

    // Force Plotly to recalculate size after DOM has painted
    // Using double-RAF ensures layout is complete (especially for first trait on initial load)
    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            const el = document.getElementById(divId);
            if (el) Plotly.Plots.resize(divId);
        });
    });
}

// Export to global scope
window.renderTraitTrajectory = renderTraitTrajectory;
