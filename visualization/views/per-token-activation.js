// Prompt Activation View - All selected traits on single graph with legend

// Color palette for traits (distinct, colorblind-friendly)
const TRAIT_COLORS = [
    '#4a9eff',  // blue
    '#ff6b6b',  // red
    '#51cf66',  // green
    '#ffd43b',  // yellow
    '#cc5de8',  // purple
    '#ff922b',  // orange
    '#20c997',  // teal
    '#f06595',  // pink
    '#748ffc',  // indigo
    '#a9e34b',  // lime
];

async function renderPerTokenActivation() {
    const contentArea = document.getElementById('content-area');
    const filteredTraits = window.getFilteredTraits();

    if (filteredTraits.length === 0) {
        contentArea.innerHTML = `
            <div class="card">
                <div class="card-title">Per-Token Activation</div>
                <div class="info">Select at least one trait to view per-token activation trajectories</div>
            </div>
        `;
        return;
    }

    // Show loading state
    contentArea.innerHTML = `
        <div class="card">
            <div class="card-title">Per-Token Activation - Loading...</div>
            <div class="info">Loading data for ${filteredTraits.length} trait(s)...</div>
        </div>
    `;

    const traitData = {};
    const failedTraits = [];

    // Load data for ALL selected traits
    for (const trait of filteredTraits) {
        try {
            const fetchPath = window.paths.tier2Data(trait, window.state.currentPrompt);
            console.log(`[${trait.name}] Fetching prompt activation data for prompt ${window.state.currentPrompt}`);
            const response = await fetch(fetchPath);

            if (!response.ok) {
                console.log(`[${trait.name}] No data available for prompt ${window.state.currentPrompt} (${response.status})`);
                failedTraits.push(trait.name);
                continue;
            }

            const data = await response.json();
            console.log(`[${trait.name}] Data loaded successfully for prompt ${window.state.currentPrompt}`);
            traitData[trait.name] = data;
        } catch (error) {
            console.log(`[${trait.name}] Load failed for prompt ${window.state.currentPrompt}:`, error.message);
            failedTraits.push(trait.name);
        }
    }

    // Check if we have any data
    const loadedTraits = Object.keys(traitData);
    if (loadedTraits.length === 0) {
        renderNoDataMessage(contentArea, filteredTraits, window.state.currentPrompt);
        return;
    }

    // Render the combined graph
    renderCombinedGraph(contentArea, traitData, loadedTraits, failedTraits, window.state.currentPrompt);
}

function renderNoDataMessage(container, traits, promptNum) {
    container.innerHTML = `
        <div class="card">
            <div class="card-title">Per-Token Activation</div>
            <div class="info" style="margin-bottom: 10px;">
                ⚠️ No data available for prompt ${promptNum} for any selected trait.
            </div>
            <div style="background: var(--bg-tertiary); padding: 10px; border-radius: 6px; font-size: 12px;">
                <p style="color: var(--text-secondary); margin-bottom: 8px;">
                    To capture per-token activation data, run:
                </p>
                <pre style="background: var(--bg-primary); color: var(--text-primary); padding: 8px; border-radius: 4px; margin: 8px 0; overflow-x: auto; font-size: 11px;">python inference/capture_layers.py --experiment ${window.paths.getExperiment()} --prompt-set main_prompts --save-json</pre>
            </div>
        </div>
    `;
}

function renderCombinedGraph(container, traitData, loadedTraits, failedTraits, promptNum) {
    // Use first trait's data as reference for tokens (they should all be the same)
    const refData = traitData[loadedTraits[0]];
    const promptTokens = refData.prompt.tokens;
    const responseTokens = refData.response.tokens;
    const allTokens = [...promptTokens, ...responseTokens];
    const nPromptTokens = refData.prompt.n_tokens;
    const nTotalTokens = allTokens.length;

    // Build HTML
    let failedHtml = '';
    if (failedTraits.length > 0) {
        failedHtml = `
            <div style="color: var(--text-secondary); font-size: 11px; margin-top: 4px;">
                ⚠️ No data for: ${failedTraits.map(t => window.getDisplayName(t)).join(', ')}
            </div>
        `;
    }

    container.innerHTML = `
        <div class="card" style="padding: 12px;">
            <div class="card-title" style="margin-bottom: 8px;">Per-Token Activation Trajectory</div>

            <!-- Conversation context -->
            <div style="background: var(--bg-tertiary); padding: 8px; border-radius: 4px; margin-bottom: 8px;">
                <div style="color: var(--text-primary); font-size: 11px; margin-bottom: 2px;">
                    <strong>Prompt:</strong> ${window.markdownToHtml(refData.prompt.text)}
                </div>
                <div style="color: var(--text-primary); font-size: 11px; margin-bottom: 4px;">
                    <strong>Response:</strong> ${window.markdownToHtml(refData.response.text.substring(0, 200))}${refData.response.text.length > 200 ? '...' : ''}
                </div>
                <div style="color: var(--text-secondary); font-size: 10px;">
                    ${nPromptTokens} prompt + ${nTotalTokens - nPromptTokens} response = ${nTotalTokens} tokens • Prompt ${promptNum}
                </div>
                ${failedHtml}
            </div>

            <!-- Plot -->
            <div id="combined-activation-plot" style="width: 100%;"></div>
        </div>
    `;

    // Prepare traces for each trait
    const traces = [];
    const startIdx = 1;  // Skip BOS token

    loadedTraits.forEach((traitName, idx) => {
        const data = traitData[traitName];
        const promptProj = data.projections.prompt;
        const responseProj = data.projections.response;
        const allProj = [...promptProj, ...responseProj];
        const nLayers = promptProj[0].length;

        // Calculate activation strength for each token (average across all layers and sublayers)
        const activations = [];
        const displayTokens = [];

        for (let t = startIdx; t < allProj.length; t++) {
            let sum = 0;
            let count = 0;
            for (let l = 0; l < nLayers; l++) {
                for (let s = 0; s < 3; s++) {
                    sum += allProj[t][l][s];
                    count++;
                }
            }
            activations.push(sum / count);
            displayTokens.push(allTokens[t]);
        }

        const color = TRAIT_COLORS[idx % TRAIT_COLORS.length];

        traces.push({
            x: Array.from({length: activations.length}, (_, i) => i),
            y: activations,
            type: 'scatter',
            mode: 'lines+markers',
            name: window.getDisplayName(traitName),
            line: {
                color: color,
                width: 1
            },
            marker: {
                size: 1,
                color: color
            },
            text: displayTokens,
            hovertemplate: `<b>${window.getDisplayName(traitName)}</b><br>Token %{x}: %{text}<br>Activation: %{y:.3f}<extra></extra>`
        });
    });

    // Get display tokens from first trait for x-axis labels
    const displayTokens = [];
    for (let t = startIdx; t < allTokens.length; t++) {
        displayTokens.push(allTokens[t]);
    }

    // Add subtle vertical line separator between prompt and response
    const shapes = [
        {
            type: 'line',
            x0: (nPromptTokens - startIdx) - 0.5,
            x1: (nPromptTokens - startIdx) - 0.5,
            y0: 0,
            y1: 1,
            yref: 'paper',
            line: {
                color: 'rgba(128, 128, 128, 0.4)',
                width: 2,
                dash: 'dash'
            }
        }
    ];

    // Add annotations for prompt/response regions
    const annotations = [
        {
            x: (nPromptTokens - startIdx) / 2 - 0.5,
            y: 1.08,
            yref: 'paper',
            text: 'PROMPT',
            showarrow: false,
            font: {
                size: 11,
                color: 'rgba(128, 128, 128, 0.8)'
            }
        },
        {
            x: (nPromptTokens - startIdx) + (displayTokens.length - (nPromptTokens - startIdx)) / 2 - 0.5,
            y: 1.08,
            yref: 'paper',
            text: 'RESPONSE',
            showarrow: false,
            font: {
                size: 11,
                color: 'rgba(128, 128, 128, 0.8)'
            }
        }
    ];

    const layout = {
        xaxis: {
            title: 'Token Position',
            tickmode: 'array',
            tickvals: Array.from({length: displayTokens.length}, (_, i) => i),
            ticktext: displayTokens,
            tickangle: -45,
            showgrid: false,
            tickfont: { size: 9 }
        },
        yaxis: {
            title: 'Activation (avg all layers)',
            zeroline: true,
            zerolinecolor: 'rgba(128, 128, 128, 0.3)',
            zerolinewidth: 1,
            showgrid: true,
            gridcolor: 'rgba(128, 128, 128, 0.1)'
        },
        shapes: shapes,
        annotations: annotations,
        margin: { l: 60, r: 20, t: 40, b: 100 },
        height: 800,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: {
            size: 11,
            color: getComputedStyle(document.documentElement).getPropertyValue('--text-primary')
        },
        hovermode: 'closest',
        legend: {
            orientation: 'h',
            yanchor: 'bottom',
            y: -0.35,
            xanchor: 'center',
            x: 0.5,
            font: { size: 11 },
            bgcolor: 'rgba(0,0,0,0)'
        },
        showlegend: true
    };

    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['lasso2d', 'select2d']
    };

    Plotly.newPlot('combined-activation-plot', traces, layout, config);
}

// Export to global scope
window.renderPerTokenActivation = renderPerTokenActivation;
