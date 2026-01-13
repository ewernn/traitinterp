// Trait Dynamics View - Watch the model's internal state evolve token-by-token
// Core insight: "See how the model is thinking" by projecting onto trait vectors
//
// Sections:
// 1. Token Trajectory: X=tokens, Y=projection (best layer) + velocity/acceleration
// 2. Activation Magnitude

// Show all tokens including BOS (set to 2 to skip BOS + warmup if desired)
const START_TOKEN_IDX = 0;

// smoothData is in core/utils.js

/**
 * Compute first derivative (velocity) from an array
 */
function computeVelocity(data) {
    const velocity = [];
    for (let i = 0; i < data.length - 1; i++) {
        velocity.push(data[i + 1] - data[i]);
    }
    return velocity;
}


/**
 * Get list of dims to remove based on cleaning mode.
 * Data-driven: uses massive_dim_data embedded in projection files.
 *
 * Modes:
 * - 'top5-3layers': Dims in top-5 at 3+ layers - balanced (recommended)
 * - 'all': All candidate dims - most aggressive
 */
function getDimsToRemove(massiveDimData, cleaningMode) {
    const { dims, top_dims_by_layer } = massiveDimData || {};

    if (!dims) return [];

    if (cleaningMode === 'all') {
        return dims;
    }

    if (cleaningMode === 'top5-3layers' && top_dims_by_layer) {
        // Count appearances: dims that appear in top-5 at 3+ layers
        const appearances = {};
        for (const layerDims of Object.values(top_dims_by_layer)) {
            const top5 = layerDims.slice(0, 5);
            for (const dim of top5) {
                appearances[dim] = (appearances[dim] || 0) + 1;
            }
        }
        return Object.entries(appearances)
            .filter(([_, count]) => count >= 3)
            .map(([dim, _]) => parseInt(dim));
    }

    return [];
}


/**
 * Apply massive dim cleaning to projections.
 * Formula: adjusted = original - sum(act[dim] * vec[dim]) / ||vec||
 */
function applyMassiveDimCleaning(projections, massiveDimData, dimsToRemove, phase) {
    const { vec_norm, vec_components, activation_values } = massiveDimData;
    const phaseActValues = activation_values[phase];

    if (!phaseActValues || !vec_norm) {
        return projections;
    }

    return projections.map((proj, tokenIdx) => {
        let adjustment = 0;
        for (const dim of dimsToRemove) {
            const actVal = phaseActValues[dim]?.[tokenIdx] ?? 0;
            const vecComp = vec_components[dim] ?? 0;
            adjustment += actVal * vecComp;
        }
        return proj - adjustment / vec_norm;
    });
}


/**
 * Compute cleaned token norms: ||h_cleaned|| = sqrt(||h||² - Σ h[dim]²)
 */
function computeCleanedNorms(originalNorms, massiveDimData, dimsToRemove, phase) {
    const phaseActValues = massiveDimData?.activation_values?.[phase];
    if (!phaseActValues || dimsToRemove.length === 0) {
        return originalNorms;
    }

    return originalNorms.map((norm, tokenIdx) => {
        const normSquared = norm * norm;
        let massiveContribution = 0;
        for (const dim of dimsToRemove) {
            const actVal = phaseActValues[dim]?.[tokenIdx] ?? 0;
            massiveContribution += actVal * actVal;
        }
        const cleanedSquared = normSquared - massiveContribution;
        return cleanedSquared > 0 ? Math.sqrt(cleanedSquared) : 0;
    });
}


async function renderTraitDynamics() {
    const contentArea = document.getElementById('content-area');
    const filteredTraits = window.getFilteredTraits();

    // Preserve scroll position
    const scrollY = contentArea.scrollTop;

    if (filteredTraits.length === 0) {
        contentArea.innerHTML = `
            <div class="tool-view">
                <div class="page-intro">
                    <div class="page-intro-text">Watch traits evolve token-by-token during generation.</div>
                </div>
                <div class="info">Select at least one trait from the sidebar to view activation trajectories.</div>
            </div>
        `;
        return;
    }

    const traitData = {};
    const failedTraits = [];
    const promptSet = window.state.currentPromptSet;
    const promptId = window.state.currentPromptId;
    const modelVariant = window.state.experimentData?.experimentConfig?.defaults?.application || 'instruct';

    if (!promptSet || !promptId) {
        renderNoDataMessage(contentArea, filteredTraits, promptSet, promptId);
        return;
    }

    // Show loading state only if fetch takes > 150ms
    const loadingTimeout = setTimeout(() => {
        contentArea.innerHTML = `
            <div class="tool-view">
                <div class="page-intro">
                    <div class="page-intro-text">Watch traits evolve token-by-token during generation.</div>
                </div>
                <div class="info">Loading data for ${filteredTraits.length} trait(s)...</div>
            </div>
        `;
    }, 150);

    // Load shared response data (prompt/response text and tokens)
    let responseData = null;
    try {
        const responsePath = window.paths.responseData(promptSet, promptId, modelVariant);
        const responseRes = await fetch(responsePath);
        if (responseRes.ok) {
            responseData = await responseRes.json();
        }
    } catch (error) {
        console.warn('Could not load shared response data, falling back to projection data');
    }

    // Load projection data for ALL selected traits (in parallel)
    const projectionResults = await Promise.all(filteredTraits.map(async (trait) => {
        try {
            const fetchPath = window.paths.residualStreamData(trait, promptSet, promptId, modelVariant);
            const response = await fetch(fetchPath);
            if (!response.ok) return { trait, error: true };
            const projData = await response.json();
            if (projData.error) return { trait, error: true };
            return { trait, projData };
        } catch (error) {
            return { trait, error: true };
        }
    }));

    // Process results
    for (const result of projectionResults) {
        if (result.error) {
            failedTraits.push(result.trait.name);
            continue;
        }
        const { trait, projData } = result;

        // Merge response data with projection data (projection is slim, needs tokens)
        if (responseData) {
            projData.prompt = responseData.prompt;
            projData.response = responseData.response;
            if (responseData.metadata?.inference_model && !projData.metadata?.inference_model) {
                projData.metadata = projData.metadata || {};
                projData.metadata.inference_model = responseData.metadata.inference_model;
            }
        }

        // Handle multi-vector format: expand into separate entries per vector
        if (projData.metadata?.multi_vector && Array.isArray(projData.projections)) {
            for (const vecProj of projData.projections) {
                const key = `${trait.name}__${vecProj.method}_L${vecProj.layer}`;
                traitData[key] = {
                    ...projData,
                    projections: { prompt: vecProj.prompt, response: vecProj.response },
                    metadata: {
                        ...projData.metadata,
                        vector_source: {
                            layer: vecProj.layer,
                            method: vecProj.method,
                            selection_source: vecProj.selection_source,
                            baseline: vecProj.baseline
                        },
                        _baseTrait: trait.name,
                        _isMultiVector: true
                    }
                };
            }
        } else {
            traitData[trait.name] = projData;
        }
    }

    clearTimeout(loadingTimeout);

    // Handle compare mode: "main", "diff:{model}", or "show:{model}"
    const compareMode = window.state.compareMode || 'main';
    const isDiff = compareMode.startsWith('diff:');
    const isShow = compareMode.startsWith('show:');
    const compareModel = isDiff ? compareMode.slice(5) : isShow ? compareMode.slice(5) : null;

    if (compareModel) {
        const traitKeys = Object.keys(traitData);
        const compResults = await Promise.all(traitKeys.map(async (traitKey) => {
            const baseTrait = traitData[traitKey].metadata?._baseTrait || traitKey;
            const trait = filteredTraits.find(t => t.name === baseTrait);
            if (!trait) return null;

            try {
                // Fetch from comparison model's path (same prompt set, different model)
                const fetchPath = window.paths.residualStreamData(trait, promptSet, promptId, compareModel);
                const response = await fetch(fetchPath);
                if (!response.ok) return null;
                const compData = await response.json();
                if (compData.error) return null;
                return { traitKey, compData };
            } catch (error) {
                return null;
            }
        }));

        for (const result of compResults) {
            if (!result) continue;
            const { traitKey, compData } = result;

            let compProj;
            if (compData.metadata?.multi_vector && Array.isArray(compData.projections)) {
                const vs = traitData[traitKey].metadata?.vector_source;
                if (vs) {
                    const match = compData.projections.find(p => p.method === vs.method && p.layer === vs.layer);
                    compProj = match ? { prompt: match.prompt, response: match.response } : null;
                }
            } else {
                compProj = compData.projections;
            }

            if (compProj) {
                if (isDiff) {
                    // Diff mode: main - comparison
                    const mainProj = traitData[traitKey].projections;
                    const diffPrompt = mainProj.prompt.map((v, i) => v - (compProj.prompt[i] || 0));
                    const diffResponse = mainProj.response.map((v, i) => v - (compProj.response[i] || 0));
                    traitData[traitKey].projections = { prompt: diffPrompt, response: diffResponse };
                    traitData[traitKey].metadata = traitData[traitKey].metadata || {};
                    traitData[traitKey].metadata._isDiff = true;
                    traitData[traitKey].metadata._compareModel = compareModel;
                } else if (isShow) {
                    // Show mode: replace with comparison model's projections
                    traitData[traitKey].projections = compProj;
                    traitData[traitKey].metadata = traitData[traitKey].metadata || {};
                    traitData[traitKey].metadata._isComparisonModel = true;
                    traitData[traitKey].metadata._compareModel = compareModel;
                }
            }
        }
    }

    // Check if we have any data
    const loadedTraits = Object.keys(traitData);
    if (loadedTraits.length === 0) {
        renderNoDataMessage(contentArea, filteredTraits, promptSet, promptId);
        return;
    }

    // Render the full view
    await renderCombinedGraph(contentArea, traitData, loadedTraits, failedTraits, promptSet, promptId);

    // Restore scroll position after DOM updates
    requestAnimationFrame(() => {
        contentArea.scrollTop = scrollY;
    });
}

function renderNoDataMessage(container, traits, promptSet, promptId) {
    const promptLabel = promptSet && promptId ? `${promptSet}/${promptId}` : 'none selected';
    container.innerHTML = `
        <div class="tool-view">
            <div class="page-intro">
                <div class="page-intro-text">Watch traits evolve token-by-token during generation.</div>
            </div>
            <div class="info">
                No data available for prompt ${promptLabel} for any selected trait.
            </div>
            <p class="tool-description">
                To capture per-token activation data, run:
            </p>
            <pre>python inference/capture_raw_activations.py --experiment ${window.paths.getExperiment()} --prompt-set ${promptSet || 'PROMPT_SET'}</pre>
        </div>
    `;
}

async function renderCombinedGraph(container, traitData, loadedTraits, failedTraits, promptSet, promptId) {
    // Use first trait's data as reference for tokens (they should all be the same)
    const refData = traitData[loadedTraits[0]];

    if (!refData.prompt?.tokens || !refData.response?.tokens) {
        container.innerHTML = `<div class="tool-view"><div class="info">Error: Missing token data.</div></div>`;
        return;
    }

    const promptTokens = refData.prompt.tokens;
    const responseTokens = refData.response.tokens;
    const allTokens = [...promptTokens, ...responseTokens];
    const nPromptTokens = promptTokens.length;
    const nTotalTokens = allTokens.length;

    // Extract inference model and vector source from metadata
    const meta = refData.metadata || {};
    const inferenceModel = meta.inference_model ||
        window.state.experimentData?.experimentConfig?.application_model ||
        'unknown';
    const vectorSource = meta.vector_source || {};

    // Build model info HTML (just inference model - vector details shown on hover)
    const modelInfoHtml = `Inference model: <code>${inferenceModel}</code>`;

    // Build HTML
    let failedHtml = '';
    if (failedTraits.length > 0) {
        failedHtml = `
            <div class="tool-description">
                No data for: ${failedTraits.map(t => window.getDisplayName(t)).join(', ')}
            </div>
        `;
    }

    // Determine smoothing and centering
    const isSmoothing = window.state.smoothingEnabled !== false;  // default true
    const isCentered = window.state.projectionCentered !== false;  // default true

    // Compare mode: "main", "diff:{model}", or "show:{model}"
    const currentCompareMode = window.state.compareMode || 'main';
    const availableModels = window.state.availableComparisonModels || [];

    // Check what we're showing
    const showingDiff = Object.values(traitData).some(d => d.metadata?._isDiff);
    const showingCompModel = Object.values(traitData).some(d => d.metadata?._isComparisonModel);
    const compareModelName = Object.values(traitData).find(d => d.metadata?._compareModel)?.metadata?._compareModel;

    let compareInfoHtml = '';
    if (showingDiff) {
        compareInfoHtml = `<div class="page-intro-text" style="color: var(--color-accent); font-weight: 500;">
            Showing DIFF: application model − ${compareModelName}
           </div>`;
    } else if (showingCompModel) {
        compareInfoHtml = `<div class="page-intro-text" style="color: var(--color-accent); font-weight: 500;">
            Showing: ${compareModelName} (comparison model)
           </div>`;
    }

    container.innerHTML = `
        <div class="tool-view">
            <div class="page-intro">
                <div class="page-intro-text">Watch traits evolve token-by-token during generation.</div>
                ${compareInfoHtml}
                <div class="page-intro-model">${modelInfoHtml}</div>
            </div>
            ${failedHtml}

            <section>
                <h2>Token Trajectory <span class="subsection-info-toggle" data-target="info-token-trajectory">►</span></h2>
                <div class="subsection-info" id="info-token-trajectory">
                    Cosine similarity: proj / ||h||. Shows directional alignment with trait vector.
                    ${isCentered ? ' Centered by subtracting BOS token value.' : ''}
                    ${isSmoothing ? ' Smoothed with 3-token moving average.' : ''}
                </div>
                <div class="projection-toggle">
                    <label class="projection-toggle-checkbox">
                        <input type="checkbox" id="smoothing-toggle" ${isSmoothing ? 'checked' : ''}>
                        <span>Smooth</span>
                    </label>
                    <label class="projection-toggle-checkbox">
                        <input type="checkbox" id="projection-centered-toggle" ${isCentered ? 'checked' : ''}>
                        <span>Centered</span>
                    </label>
                    <span class="projection-toggle-label">Clean:</span>
                    <select id="massive-dims-cleaning-select" style="margin-left: 4px;" title="Remove high-magnitude bias dimensions (Sun et al. 2024). These dims have 100-1000x larger values than typical dims and act as constant biases.">
                        <option value="none" ${!window.state.massiveDimsCleaning || window.state.massiveDimsCleaning === 'none' ? 'selected' : ''}>No cleaning</option>
                        <option value="top5-3layers" ${window.state.massiveDimsCleaning === 'top5-3layers' ? 'selected' : ''}>Top 5, 3+ layers</option>
                        <option value="all" ${window.state.massiveDimsCleaning === 'all' ? 'selected' : ''}>All candidates</option>
                    </select>
                    <span class="projection-toggle-label" style="margin-left: 16px;">Methods:</span>
                    <label class="projection-toggle-checkbox">
                        <input type="checkbox" class="method-filter" data-method="probe" ${window.state.selectedMethods.has('probe') ? 'checked' : ''}>
                        <span>probe</span>
                    </label>
                    <label class="projection-toggle-checkbox">
                        <input type="checkbox" class="method-filter" data-method="mean_diff" ${window.state.selectedMethods.has('mean_diff') ? 'checked' : ''}>
                        <span>mean_diff</span>
                    </label>
                    <label class="projection-toggle-checkbox">
                        <input type="checkbox" class="method-filter" data-method="gradient" ${window.state.selectedMethods.has('gradient') ? 'checked' : ''}>
                        <span>gradient</span>
                    </label>
                    <label class="projection-toggle-checkbox">
                        <input type="checkbox" class="method-filter" data-method="random" ${window.state.selectedMethods.has('random') ? 'checked' : ''}>
                        <span>random</span>
                    </label>
                    ${availableModels.length > 0 ? `
                    <span class="projection-toggle-label" style="margin-left: 16px;">Compare:</span>
                    <select id="compare-mode-select" style="margin-left: 4px;">
                        <option value="main" ${currentCompareMode === 'main' ? 'selected' : ''}>Main model</option>
                        ${availableModels.map(m => `
                            <option value="diff:${m}" ${currentCompareMode === 'diff:' + m ? 'selected' : ''}>Diff (main − ${m})</option>
                            <option value="show:${m}" ${currentCompareMode === 'show:' + m ? 'selected' : ''}>${m}</option>
                        `).join('')}
                    </select>
                    ` : ''}
                </div>
                <div id="combined-activation-plot"></div>
            </section>

            <section>
                <h3>Activation Magnitude Per Token <span class="subsection-info-toggle" data-target="info-token-magnitude">►</span></h3>
                <div class="subsection-info" id="info-token-magnitude">L2 norm of activation per token. Shows one line per unique layer used by traits above. Compare to trajectory - similar magnitudes but low projections means token encodes orthogonal information.</div>
                <div id="token-magnitude-plot"></div>
            </section>

            <section>
                <h3>Token Velocity <span class="subsection-info-toggle" data-target="info-token-velocity">►</span></h3>
                <div class="subsection-info" id="info-token-velocity">Rate of change between consecutive tokens (d/dt of trajectory above).</div>
                <div id="token-velocity-plot"></div>
            </section>

            <section>
                <h3>Activation Magnitude <span class="subsection-info-toggle" data-target="info-act-magnitude">►</span></h3>
                <div class="subsection-info" id="info-act-magnitude">How the residual stream grows in magnitude as each layer adds information to the hidden state.</div>
                <div id="activation-magnitude-plot"></div>
            </section>

            <section>
                <h3>Massive Activations <span class="subsection-info-toggle" data-target="info-massive-acts">►</span></h3>
                <div class="subsection-info" id="info-massive-acts">
                    Massive activation dimensions (Sun et al. 2024) - specific dimensions with values 100-1000x larger than median.
                    These act as fixed biases and cause the "mean component" phenomenon. Run <code>python analysis/massive_activations.py</code> to generate data.
                </div>
                <div id="massive-activations-container"></div>
            </section>

            <section>
                <h3>Massive Dims Across Layers <span class="subsection-info-toggle" data-target="info-massive-dims-layers">►</span></h3>
                <div class="subsection-info" id="info-massive-dims-layers">
                    Shows how each massive dimension's magnitude changes across layers (normalized by layer average).
                    Use the criteria dropdown to experiment with different definitions of "massive".
                </div>
                <div class="projection-toggle" style="margin-bottom: 12px;">
                    <span class="projection-toggle-label">Criteria:</span>
                    <select id="massive-dims-criteria">
                        <option value="top5-3layers">Top 5, 3+ layers</option>
                        <option value="top3-any">Top 3, any layer</option>
                        <option value="top5-any">Top 5, any layer</option>
                    </select>
                </div>
                <div id="massive-dims-layers-plot"></div>
            </section>
        </div>
    `;

    // Setup info toggles
    window.setupSubsectionInfoToggles();

    // Prepare data for plotting
    const traitActivations = {};  // Store smoothed activations for velocity/accel

    // Prepare traces for Token Trajectory (always cosine)
    const traces = [];

    // Filter traits by selected methods
    const filteredByMethod = loadedTraits.filter(traitName => {
        const method = traitData[traitName]?.metadata?.vector_source?.method;
        return !method || window.state.selectedMethods.has(method);
    });

    if (filteredByMethod.length === 0) {
        container.querySelector('#combined-activation-plot').innerHTML = `
            <div class="info">No traits match selected methods. Enable more methods above.</div>
        `;
        return;
    }

    for (let idx = 0; idx < filteredByMethod.length; idx++) {
        const traitName = filteredByMethod[idx];
        const data = traitData[traitName];

        // Get original projections
        let promptProj = [...data.projections.prompt];
        let responseProj = [...data.projections.response];

        // Apply massive dims cleaning if requested and data available
        const cleaningMode = window.state.massiveDimsCleaning || 'none';
        let dimsToRemove = [];
        const mdd = data.massive_dim_data;
        if (cleaningMode !== 'none' && mdd) {
            dimsToRemove = getDimsToRemove(mdd, cleaningMode);
            promptProj = applyMassiveDimCleaning(promptProj, mdd, dimsToRemove, 'prompt');
            responseProj = applyMassiveDimCleaning(responseProj, mdd, dimsToRemove, 'response');
        }

        const allProj = [...promptProj, ...responseProj];

        // Get vector source from metadata
        const vs = data.metadata?.vector_source || {};

        // projections are now 1D arrays (one value per token at best layer)
        let rawProj = allProj.slice(START_TOKEN_IDX);

        // Compute cosine similarity (use cleaned norms if cleaning applied)
        let rawValues;
        if (data.token_norms) {
            let promptNorms = data.token_norms.prompt;
            let responseNorms = data.token_norms.response;

            // Use cleaned norms if massive dims were removed
            if (dimsToRemove.length > 0 && mdd) {
                promptNorms = computeCleanedNorms(promptNorms, mdd, dimsToRemove, 'prompt');
                responseNorms = computeCleanedNorms(responseNorms, mdd, dimsToRemove, 'response');
            }

            const traitTokenNorms = [...promptNorms, ...responseNorms].slice(START_TOKEN_IDX);
            rawValues = rawProj.map((proj, i) => {
                const norm = traitTokenNorms[i];
                return norm > 0 ? proj / norm : 0;
            });
        } else {
            rawValues = rawProj;
        }

        // Subtract BOS value if centering is enabled (makes token 0 = 0)
        if (isCentered && rawValues.length > 0) {
            const bosValue = rawValues[0];
            rawValues = rawValues.map(v => v - bosValue);
        }

        // Apply 3-token moving average if smoothing is enabled
        const displayValues = isSmoothing ? window.smoothData(rawValues, 3) : rawValues;

        // Store raw values for velocity/accel (always use smoothed for derivatives)
        traitActivations[traitName] = isSmoothing ? window.smoothData(rawProj, 3) : rawProj;

        // Each vector gets its own color
        const color = window.getChartColors()[idx % 10];

        const method = vs.method || 'probe';
        const valueLabel = 'Cosine';
        const valueFormat = '.4f';

        // Build display name and hover
        const baseTrait = data.metadata?._baseTrait || traitName;
        const displayName = data.metadata?._isMultiVector
            ? `${window.getDisplayName(baseTrait)} (${method} L${vs.layer})`
            : window.getDisplayName(traitName);
        const pos = data.metadata?.position || vs.position;
        const posStr = pos && pos !== 'response[:]' ? ` @${pos.replace('response', 'resp').replace('prompt', 'p')}` : '';
        const vectorInfo = vs.layer !== undefined ? `<br><span style="color:#888">L${vs.layer} ${method}${posStr}</span>` : '';
        const hoverText = `<b>${displayName}</b>${vectorInfo}<br>Token %{x}<br>${valueLabel}: %{y:${valueFormat}}<extra></extra>`;

        traces.push({
            x: Array.from({length: displayValues.length}, (_, i) => i),
            y: displayValues,
            type: 'scatter',
            mode: 'lines+markers',
            name: displayName,
            line: { color: color, width: 1.5 },
            marker: { size: 2, color: color },
            hovertemplate: hoverText
        });
    }

    // Get display tokens (every 10th for x-axis labels)
    const displayTokens = allTokens.slice(START_TOKEN_IDX);
    const tickVals = [];
    const tickText = [];
    for (let i = 0; i < displayTokens.length; i += 10) {
        tickVals.push(i);
        tickText.push(displayTokens[i]);
    }

    // Get colors from CSS variables
    const textSecondary = window.getCssVar('--text-secondary', '#a4a4a4');
    const primaryColor = window.getCssVar('--primary-color', '#a09f6c');

    // Current token highlight
    const currentTokenIdx = window.state.currentTokenIndex || 0;
    const highlightX = Math.max(0, currentTokenIdx - START_TOKEN_IDX);

    // Shapes for prompt/response separator and token highlight
    const shapes = [
        {
            type: 'line',
            x0: (nPromptTokens - START_TOKEN_IDX) - 0.5,
            x1: (nPromptTokens - START_TOKEN_IDX) - 0.5,
            y0: 0, y1: 1, yref: 'paper',
            line: { color: textSecondary, width: 2, dash: 'dash' }
        },
        {
            type: 'line',
            x0: highlightX, x1: highlightX,
            y0: 0, y1: 1, yref: 'paper',
            line: { color: primaryColor, width: 2 }
        }
    ];

    const annotations = [
        {
            x: (nPromptTokens - START_TOKEN_IDX) / 2 - 0.5,
            y: 1.08, yref: 'paper',
            text: 'PROMPT', showarrow: false,
            font: { size: 11, color: textSecondary }
        },
        {
            x: (nPromptTokens - START_TOKEN_IDX) + (displayTokens.length - (nPromptTokens - START_TOKEN_IDX)) / 2 - 0.5,
            y: 1.08, yref: 'paper',
            text: 'RESPONSE', showarrow: false,
            font: { size: 11, color: textSecondary }
        }
    ];

    // Build custom legend with vector source tooltips (like live-chat)
    const legendHtml = filteredByMethod.map((traitName, idx) => {
        const data = traitData[traitName];
        const vs = data.metadata?.vector_source || {};
        const pos = data.metadata?.position || vs.position;
        const posStr = pos && pos !== 'response[:]' ? ` @${pos.replace('response', 'resp').replace('prompt', 'p')}` : '';
        const tooltipText = vs.layer !== undefined
            ? `L${vs.layer} ${vs.method || '?'}${posStr} (${vs.selection_source || 'unknown'})`
            : 'no metadata';

        // Each vector gets its own color (same as traces)
        const color = window.getChartColors()[idx % 10];

        // Display name matches trace name
        const baseTrait = data.metadata?._baseTrait || traitName;
        const displayName = data.metadata?._isMultiVector
            ? `${window.getDisplayName(baseTrait)} (${vs.method} L${vs.layer})`
            : window.getDisplayName(traitName);

        return `
            <span class="legend-item has-tooltip" data-tooltip="${tooltipText}">
                <span class="legend-color" style="background: ${color}"></span>
                ${displayName}
            </span>
        `;
    }).join('');

    // Token Trajectory plot
    const yAxisTitle = 'Cosine (proj / ||h||)';

    // Compute y-axis range: minimum ±0.15, auto-expand if data exceeds
    let yAxisConfig = { title: yAxisTitle, zeroline: true, zerolinewidth: 1, showgrid: true };
    // Find actual data range across all traces
    let minY = Infinity, maxY = -Infinity;
    traces.forEach(t => {
        t.y.forEach(v => {
            if (v < minY) minY = v;
            if (v > maxY) maxY = v;
        });
    });
    // Ensure minimum range of ±0.15, expand if data exceeds
    const minRange = 0.15;
    const rangeMin = Math.min(-minRange, minY - 0.02);
    const rangeMax = Math.max(minRange, maxY + 0.02);
    yAxisConfig.range = [rangeMin, rangeMax];

    const mainLayout = window.buildChartLayout({
        preset: 'timeSeries',
        traces,
        height: 400,
        legendPosition: 'none',  // Using custom HTML legend instead
        xaxis: {
            title: 'Token Position',
            tickmode: 'array',
            tickvals: tickVals,
            ticktext: tickText,
            tickangle: -45,
            showgrid: false,
            tickfont: { size: 9 }
        },
        yaxis: yAxisConfig,
        shapes: shapes,
        annotations: annotations,
        margin: { l: 60, r: 20, t: 40, b: 80 },
        hovermode: 'closest'
    });
    window.renderChart('combined-activation-plot', traces, mainLayout);

    // Insert custom legend after plot and setup hover-to-highlight
    const plotDiv = document.getElementById('combined-activation-plot');
    const legendDiv = document.createElement('div');
    legendDiv.className = 'chart-legend';
    legendDiv.innerHTML = legendHtml;
    plotDiv.parentNode.insertBefore(legendDiv, plotDiv.nextSibling);

    // Hover-to-highlight and click-to-select for main trajectory
    plotDiv.on('plotly_hover', (d) =>
        Plotly.restyle(plotDiv, {'opacity': traces.map((_, i) => i === d.points[0].curveNumber ? 1.0 : 0.2)})
    );
    plotDiv.on('plotly_unhover', () => Plotly.restyle(plotDiv, {'opacity': 1.0}));
    plotDiv.on('plotly_click', (d) => {
        const tokenIdx = Math.round(d.points[0].x) + START_TOKEN_IDX;
        if (window.state.currentTokenIndex !== tokenIdx) {
            window.state.currentTokenIndex = tokenIdx;
            window.renderPromptPicker?.();
            window.renderCurrentView?.();
        }
    });

    // Setup smoothing checkbox
    const smoothingCheckbox = document.getElementById('smoothing-toggle');
    if (smoothingCheckbox) {
        smoothingCheckbox.addEventListener('change', () => {
            window.setSmoothing(smoothingCheckbox.checked);
        });
    }

    // Setup centered checkbox
    const centeredCheckbox = document.getElementById('projection-centered-toggle');
    if (centeredCheckbox) {
        centeredCheckbox.addEventListener('change', () => {
            window.setProjectionCentered(centeredCheckbox.checked);
        });
    }

    // Setup massive dims cleaning dropdown
    const massiveDimsSelect = document.getElementById('massive-dims-cleaning-select');
    if (massiveDimsSelect) {
        massiveDimsSelect.addEventListener('change', () => {
            window.setMassiveDimsCleaning(massiveDimsSelect.value);
        });
    }

    // Setup method filter checkboxes
    document.querySelectorAll('.method-filter').forEach(cb => {
        cb.addEventListener('change', () => {
            window.toggleMethod(cb.dataset.method);
        });
    });

    // Setup compare mode dropdown (model comparison)
    const compareSelect = document.getElementById('compare-mode-select');
    if (compareSelect) {
        compareSelect.addEventListener('change', () => {
            window.setCompareMode(compareSelect.value);
        });
    }

    // Render Token Magnitude plot (per-token norms)
    renderTokenMagnitudePlot(traitData, filteredByMethod, tickVals, tickText, nPromptTokens);

    // Render Token Velocity plot
    renderTokenDerivativePlots(traitActivations, filteredByMethod, tickVals, tickText, nPromptTokens, traitData);

    // Render Activation Magnitude plot (per-layer)
    renderActivationMagnitudePlot(traitData, filteredByMethod);

    // Render Massive Activations section
    renderMassiveActivations();

    // Render Massive Dims Across Layers section
    renderMassiveDimsAcrossLayers();
}


/**
 * Render Token Magnitude plot showing L2 norm per token at best layer.
 * Helps identify if low projections are due to low magnitude or orthogonal encoding.
 */
function renderTokenMagnitudePlot(traitData, loadedTraits, tickVals, tickText, nPromptTokens) {
    const plotDiv = document.getElementById('token-magnitude-plot');
    const firstTraitData = traitData[loadedTraits[0]];

    if (!firstTraitData.token_norms) {
        plotDiv.innerHTML = `
            <div class="info">
                Per-token norms not available. Re-run projection script to generate.
            </div>
        `;
        return;
    }

    // Collect unique layers from all traits
    const layerToNorms = {};
    for (const traitName of loadedTraits) {
        const data = traitData[traitName];
        if (!data.token_norms) continue;
        const layer = data.metadata?.vector_source?.layer ?? 'unknown';
        if (!(layer in layerToNorms)) {
            const promptNorms = data.token_norms.prompt;
            const responseNorms = data.token_norms.response;
            layerToNorms[layer] = [...promptNorms, ...responseNorms].slice(START_TOKEN_IDX);
        }
    }

    const textSecondary = window.getCssVar('--text-secondary', '#a4a4a4');
    const colors = window.getChartColors();
    const currentTokenIdx = window.state.currentTokenIndex || 0;
    const highlightX = Math.max(0, currentTokenIdx - START_TOKEN_IDX);

    // Create a trace for each unique layer
    const traces = Object.entries(layerToNorms).map(([layer, norms], idx) => ({
        y: norms,
        type: 'scatter',
        mode: 'lines',
        name: `L${layer}`,
        line: { color: colors[idx % colors.length], width: 1.5 },
        hovertemplate: `L${layer}<br>Token %{x}<br>||h|| = %{y:.1f}<extra></extra>`
    }));

    // Prompt/response separator and current token highlight
    const promptEndIdx = nPromptTokens - START_TOKEN_IDX;
    const highlightColors = window.getTokenHighlightColors();

    const showLegend = Object.keys(layerToNorms).length > 1;

    const layout = window.buildChartLayout({
        preset: 'timeSeries',
        traces,
        height: 200,
        legendPosition: showLegend ? 'above' : 'none',
        xaxis: {
            title: 'Token',
            tickvals: tickVals,
            ticktext: tickText,
            tickfont: { size: 9 }
        },
        yaxis: { title: '||h|| (L2 norm)', tickfont: { size: 10 } },
        shapes: [
            window.createSeparatorShape(promptEndIdx, highlightColors.separator),
            window.createHighlightShape(highlightX, highlightColors.highlight)
        ]
    });
    window.renderChart(plotDiv, traces, layout);

    // Click-to-select
    plotDiv.on('plotly_click', (d) => {
        const tokenIdx = Math.round(d.points[0].x) + START_TOKEN_IDX;
        if (window.state.currentTokenIndex !== tokenIdx) {
            window.state.currentTokenIndex = tokenIdx;
            window.renderPromptPicker?.();
            window.renderCurrentView?.();
        }
    });
}


/**
 * Render Token Velocity plot (first derivative of smoothed trajectory)
 */
function renderTokenDerivativePlots(traitActivations, loadedTraits, tickVals, tickText, nPromptTokens, traitData) {
    const textSecondary = window.getCssVar('--text-secondary', '#a4a4a4');
    const primaryColor = window.getCssVar('--primary-color', '#a09f6c');
    const currentTokenIdx = window.state.currentTokenIndex || 0;
    const highlightX = Math.max(0, currentTokenIdx - START_TOKEN_IDX);

    // Velocity traces
    const velocityTraces = [];
    loadedTraits.forEach((traitName, idx) => {
        const activations = traitActivations[traitName];
        const velocity = computeVelocity(activations);
        const smoothedVelocity = window.smoothData(velocity, 3);
        const color = window.getChartColors()[idx % 10];

        const vs = traitData[traitName]?.metadata?.vector_source || {};
        const method = vs.method || 'probe';
        const pos = traitData[traitName]?.metadata?.position || vs.position;
        const posStr = pos && pos !== 'response[:]' ? ` @${pos.replace('response', 'resp').replace('prompt', 'p')}` : '';
        const vectorInfo = vs.layer !== undefined ? `<br><span style="color:#888">L${vs.layer} ${method}${posStr}</span>` : '';

        velocityTraces.push({
            x: Array.from({length: smoothedVelocity.length}, (_, i) => i + 0.5),
            y: smoothedVelocity,
            type: 'scatter',
            mode: 'lines',
            name: window.getDisplayName(traitName),
            line: { color: color, width: 1.5 },
            hovertemplate: `<b>${window.getDisplayName(traitName)}</b>${vectorInfo}<br>Token %{x:.0f}<br>Velocity: %{y:.4f}<extra></extra>`
        });
    });

    const velocityLayout = window.buildChartLayout({
        preset: 'timeSeries',
        traces: velocityTraces,
        height: 300,
        legendPosition: 'none',
        xaxis: { title: '', tickmode: 'array', tickvals: tickVals, ticktext: tickText, tickangle: -45, tickfont: { size: 8 }, showgrid: true },
        yaxis: { title: 'Velocity', zeroline: true, zerolinewidth: 1, zerolinecolor: textSecondary, showgrid: true },
        shapes: [
            window.createSeparatorShape((nPromptTokens - START_TOKEN_IDX) - 0.5, textSecondary),
            window.createHighlightShape(highlightX, primaryColor)
        ],
        margin: { b: 80 }
    });
    window.renderChart('token-velocity-plot', velocityTraces, velocityLayout);

    // Click handler to update token slider
    const velocityPlot = document.getElementById('token-velocity-plot');
    velocityPlot.on('plotly_click', (d) => {
        const tokenIdx = Math.round(d.points[0].x) + START_TOKEN_IDX;
        if (window.state.currentTokenIndex !== tokenIdx) {
            window.state.currentTokenIndex = tokenIdx;
            window.renderPromptPicker?.();
            window.renderCurrentView?.();
        }
    });
}

/**
 * Render the Activation Magnitude plot showing ||h|| by layer (layer on y-axis).
 */
function renderActivationMagnitudePlot(traitData, loadedTraits) {
    const firstTraitData = traitData[loadedTraits[0]];

    if (!firstTraitData.activation_norms) {
        const plotDiv = document.getElementById('activation-magnitude-plot');
        plotDiv.innerHTML = `
            <div class="info">
                Activation norms not available. Re-run projection script to generate.
            </div>
        `;
        return;
    }

    const promptNorms = firstTraitData.activation_norms.prompt;
    const responseNorms = firstTraitData.activation_norms.response;
    const nLayers = promptNorms.length;
    const layerIndices = Array.from({length: nLayers}, (_, i) => i);
    const combinedNorms = promptNorms.map((p, i) => (p + responseNorms[i]) / 2);

    const textSecondary = window.getCssVar('--text-secondary', '#a4a4a4');

    // Layer on x-axis, L2 norm on y-axis
    const traces = [
        { x: layerIndices, y: promptNorms, type: 'scatter', mode: 'lines+markers', name: 'Prompt',
          line: { color: '#4a9eff', width: 2 }, marker: { size: 4 },
          hovertemplate: '<b>Prompt</b><br>Layer %{x}: %{y:.1f}<extra></extra>' },
        { x: layerIndices, y: responseNorms, type: 'scatter', mode: 'lines+markers', name: 'Response',
          line: { color: '#ff6b6b', width: 2 }, marker: { size: 4 },
          hovertemplate: '<b>Response</b><br>Layer %{x}: %{y:.1f}<extra></extra>' },
        { x: layerIndices, y: combinedNorms, type: 'scatter', mode: 'lines+markers', name: 'Combined',
          line: { color: textSecondary, width: 2, dash: 'dash' }, marker: { size: 4 },
          hovertemplate: '<b>Combined</b><br>Layer %{x}: %{y:.1f}<extra></extra>' }
    ];

    const layout = window.buildChartLayout({
        preset: 'layerChart',
        traces,
        height: 300,
        legendPosition: 'below',
        xaxis: { title: 'Layer', tickmode: 'linear', tick0: 0, dtick: 5, showgrid: true },
        yaxis: { title: '||h|| (L2 norm)', showgrid: true }
    });
    window.renderChart('activation-magnitude-plot', traces, layout);
}


/**
 * Fetch massive activations data, using calibration.json as canonical source.
 * Calibration contains model-wide massive dims computed from neutral prompts.
 */
async function fetchMassiveActivationsData() {
    const modelVariant = window.state.experimentData?.experimentConfig?.defaults?.application || 'instruct';
    const calibrationPath = window.paths.get('inference.massive_activations', { prompt_set: 'calibration', model_variant: modelVariant });
    const response = await fetch('/' + calibrationPath);
    if (!response.ok) return null;
    return response.json();
}

/**
 * Render Massive Activations section.
 * Shows which dimensions have abnormally large values and their behavior.
 */
async function renderMassiveActivations() {
    const container = document.getElementById('massive-activations-container');
    if (!container) return;

    try {
        const data = await fetchMassiveActivationsData();
        if (!data) {
            container.innerHTML = `
                <div class="info">
                    No massive activation calibration data.
                    <br><br>
                    Run: <code>python analysis/massive_activations.py --experiment ${window.paths.getExperiment()}</code>
                </div>
            `;
            return;
        }

        // Get aggregate stats (always available from calibration)
        const aggregate = data.aggregate || {};
        const consistentDims = aggregate.consistent_massive_dims || {};
        const meanAlignment = aggregate.mean_alignment_by_layer || {};
        const topDimsByLayer = aggregate.top_dims_by_layer || {};

        // Get dims that appear in top-5 at 3+ layers (same as cleaning logic)
        const dimAppearances = {};
        for (const layerDims of Object.values(topDimsByLayer)) {
            for (const dim of layerDims.slice(0, 5)) {
                dimAppearances[dim] = (dimAppearances[dim] || 0) + 1;
            }
        }
        const trackedDims = Object.entries(dimAppearances)
            .filter(([_, count]) => count >= 3)
            .map(([dim]) => parseInt(dim))
            .sort((a, b) => a - b);
        const dimLabels = trackedDims.map(d => `dim ${d}`).join(', ');

        // Find which dims are consistent across prompts
        let consistentDimsHtml = '';
        const allConsistent = new Set();
        for (const [layer, dims] of Object.entries(consistentDims)) {
            dims.forEach(d => allConsistent.add(d.dim));
        }
        if (allConsistent.size > 0) {
            consistentDimsHtml = `<div class="summary-card">
                <div class="summary-label">Consistent Massive Dims</div>
                <div class="summary-value">${Array.from(allConsistent).join(', ')}</div>
                <div class="summary-detail">Appear in >50% of prompts</div>
            </div>`;
        }

        container.innerHTML = `
            <div class="summary-grid">
                <div class="summary-card">
                    <div class="summary-label">Tracked Dimensions</div>
                    <div class="summary-value">${dimLabels || 'None'}</div>
                </div>
                ${consistentDimsHtml}
                <div class="summary-card">
                    <div class="summary-label">Mean Alignment (L9)</div>
                    <div class="summary-value">${((meanAlignment[9] || 0) * 100).toFixed(0)}%</div>
                    <div class="summary-detail">How much tokens align with mean direction</div>
                </div>
            </div>
            <div id="mean-alignment-plot" style="margin-top: 16px;"></div>
        `;

        // Plot mean alignment by layer
        const layers = Object.keys(meanAlignment).map(Number).sort((a, b) => a - b);
        const alignments = layers.map(l => meanAlignment[l]);

        if (layers.length > 0) {
            const alignTrace = {
                x: layers,
                y: alignments.map(v => v * 100),
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Mean Alignment',
                line: { color: window.getChartColors()[0], width: 2 },
                marker: { size: 4 },
                hovertemplate: 'L%{x}<br>Alignment: %{y:.1f}%<extra></extra>'
            };

            const alignLayout = window.buildChartLayout({
                preset: 'layerChart',
                traces: [alignTrace],
                height: 200,
                legendPosition: 'none',
                xaxis: { title: 'Layer', dtick: 5, showgrid: true },
                yaxis: { title: 'Mean Alignment (%)', range: [0, 100], showgrid: true }
            });
            window.renderChart('mean-alignment-plot', [alignTrace], alignLayout);
        }

    } catch (error) {
        container.innerHTML = `<div class="info">Error loading massive activation data: ${error.message}</div>`;
    }
}


/**
 * Render Massive Dims Across Layers section.
 * Shows how each massive dim's normalized magnitude changes across layers.
 */
async function renderMassiveDimsAcrossLayers() {
    const container = document.getElementById('massive-dims-layers-plot');
    if (!container) return;

    try {
        const data = await fetchMassiveActivationsData();
        if (!data) {
            container.innerHTML = `<div class="info">No massive activation data. Run <code>python analysis/massive_activations.py --experiment ${window.paths.getExperiment()}</code></div>`;
            return;
        }
        const aggregate = data.aggregate || {};
        const topDimsByLayer = aggregate.top_dims_by_layer || {};
        const dimMagnitude = aggregate.dim_magnitude_by_layer || {};

        if (Object.keys(dimMagnitude).length === 0) {
            container.innerHTML = `<div class="info">No per-layer magnitude data. Re-run <code>python analysis/massive_activations.py</code> to generate.</div>`;
            return;
        }

        // Get criteria from dropdown
        const criteriaSelect = document.getElementById('massive-dims-criteria');
        const criteria = criteriaSelect?.value || 'top5-3layers';

        // Filter dims based on criteria
        const filteredDims = filterDimsByCriteria(topDimsByLayer, criteria);

        if (filteredDims.length === 0) {
            container.innerHTML = `<div class="info">No dims match criteria "${criteria}".</div>`;
            return;
        }

        // Build traces
        const colors = window.getChartColors();
        const nLayers = Object.keys(topDimsByLayer).length;
        const layers = Array.from({ length: nLayers }, (_, i) => i);

        const traces = filteredDims.map((dim, idx) => {
            const magnitudes = dimMagnitude[dim] || [];
            return {
                x: layers,
                y: magnitudes,
                type: 'scatter',
                mode: 'lines+markers',
                name: `dim ${dim}`,
                line: { color: colors[idx % colors.length], width: 2 },
                marker: { size: 4 },
                hovertemplate: `dim ${dim}<br>L%{x}<br>Normalized: %{y:.2f}x<extra></extra>`
            };
        });

        const layout = window.buildChartLayout({
            preset: 'layerChart',
            traces,
            height: 300,
            legendPosition: 'above',
            xaxis: { title: 'Layer', dtick: 5, showgrid: true },
            yaxis: { title: 'Normalized Magnitude', showgrid: true }
        });
        window.renderChart(container, traces, layout);

        // Setup dropdown change handler
        if (criteriaSelect && !criteriaSelect.dataset.bound) {
            criteriaSelect.dataset.bound = 'true';
            criteriaSelect.addEventListener('change', () => {
                renderMassiveDimsAcrossLayers();
            });
        }

    } catch (error) {
        container.innerHTML = `<div class="info">Error loading data: ${error.message}</div>`;
    }
}


/**
 * Filter dims based on selected criteria.
 */
function filterDimsByCriteria(topDimsByLayer, criteria) {
    const dimAppearances = {};  // {dim: count of layers it appears in}

    // Count appearances based on criteria
    for (const [layer, dims] of Object.entries(topDimsByLayer)) {
        const topK = criteria === 'top3-any' ? 3 : 5;
        const dimsToCount = dims.slice(0, topK);
        for (const dim of dimsToCount) {
            dimAppearances[dim] = (dimAppearances[dim] || 0) + 1;
        }
    }

    // Filter based on min layers
    const minLayers = criteria === 'top5-3layers' ? 3 : 1;
    const filtered = Object.entries(dimAppearances)
        .filter(([dim, count]) => count >= minLayers)
        .map(([dim]) => parseInt(dim))
        .sort((a, b) => a - b);

    return filtered;
}


// Export to global scope
window.renderTraitDynamics = renderTraitDynamics;
