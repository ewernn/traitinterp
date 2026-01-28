// Trait Dynamics View - Watch the model's internal state evolve token-by-token
// Core insight: "See how the model is thinking" by projecting onto trait vectors
//
// Sections:
// 1. Token Trajectory: X=tokens, Y=cosine similarity (best layer)
// 2. Activation Magnitude Per Token: ||h|| at each token position
// 3. Projection Velocity: d/dt of cosine projection

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

    // Guard: require experiment selection
    if (!window.state.currentExperiment) {
        contentArea.innerHTML = `
            <div class="tool-view">
                <div class="no-data">
                    <p>Please select an experiment from the sidebar</p>
                    <small>Analysis views require an experiment to be selected. Choose one from the "Experiment" section in the sidebar.</small>
                </div>
            </div>
        `;
        return;
    }

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
            // Handle both new flat schema and old nested schema
            if (responseData.tokens && responseData.prompt_end !== undefined) {
                // New flat schema: convert to nested format expected by rest of code
                const promptEnd = responseData.prompt_end;
                projData.prompt = {
                    text: responseData.prompt || '',
                    tokens: responseData.tokens.slice(0, promptEnd)
                };
                projData.response = {
                    text: responseData.response || '',
                    tokens: responseData.tokens.slice(promptEnd)
                };
            } else {
                // Old nested schema (fallback)
                projData.prompt = responseData.prompt;
                projData.response = responseData.response;
            }
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
                ${ui.renderSubsection({
                    title: 'Token Trajectory',
                    infoId: 'info-token-trajectory',
                    infoText: 'Cosine similarity: proj / ||h||. Shows directional alignment with trait vector.' +
                        (isCentered ? ' Centered by subtracting BOS token value.' : '') +
                        (isSmoothing ? ' Smoothed with 3-token moving average.' : ''),
                    level: 'h2'
                })}
                <div class="projection-toggle">
                    ${ui.renderToggle({ id: 'smoothing-toggle', label: 'Smooth', checked: isSmoothing, className: 'projection-toggle-checkbox' })}
                    ${ui.renderToggle({ id: 'projection-centered-toggle', label: 'Centered', checked: isCentered, className: 'projection-toggle-checkbox' })}
                    <span class="projection-toggle-label">Clean:</span>
                    <select id="massive-dims-cleaning-select" style="margin-left: 4px;" title="Remove high-magnitude bias dimensions (Sun et al. 2024). These dims have 100-1000x larger values than typical dims and act as constant biases.">
                        <option value="none" ${!window.state.massiveDimsCleaning || window.state.massiveDimsCleaning === 'none' ? 'selected' : ''}>No cleaning</option>
                        <option value="top5-3layers" ${window.state.massiveDimsCleaning === 'top5-3layers' ? 'selected' : ''}>Top 5, 3+ layers</option>
                        <option value="all" ${window.state.massiveDimsCleaning === 'all' ? 'selected' : ''}>All candidates</option>
                    </select>
                    <span class="projection-toggle-label" style="margin-left: 16px;">Methods:</span>
                    ${ui.renderToggle({ label: 'probe', checked: window.state.selectedMethods.has('probe'), dataAttr: { key: 'method', value: 'probe' }, className: 'projection-toggle-checkbox method-filter' })}
                    ${ui.renderToggle({ label: 'mean_diff', checked: window.state.selectedMethods.has('mean_diff'), dataAttr: { key: 'method', value: 'mean_diff' }, className: 'projection-toggle-checkbox method-filter' })}
                    ${ui.renderToggle({ label: 'gradient', checked: window.state.selectedMethods.has('gradient'), dataAttr: { key: 'method', value: 'gradient' }, className: 'projection-toggle-checkbox method-filter' })}
                    ${ui.renderToggle({ label: 'random', checked: window.state.selectedMethods.has('random'), dataAttr: { key: 'method', value: 'random' }, className: 'projection-toggle-checkbox method-filter' })}
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
                ${ui.renderSubsection({
                    title: 'Activation Magnitude Per Token',
                    infoId: 'info-token-magnitude',
                    infoText: 'L2 norm of activation per token. Shows one line per unique layer used by traits above. Compare to trajectory - similar magnitudes but low projections means token encodes orthogonal information.'
                })}
                <div id="token-magnitude-plot"></div>
            </section>

            <section>
                ${ui.renderSubsection({
                    title: 'Projection Velocity',
                    infoId: 'info-token-velocity',
                    infoText: 'Rate of change of cosine projection between consecutive tokens. Positive = trait increasing, negative = trait decreasing, zero crossing = inflection point.'
                })}
                <div id="token-velocity-plot"></div>
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
        // Collect methods present in data for debugging
        const methodsInData = new Set(loadedTraits.map(t => traitData[t]?.metadata?.vector_source?.method).filter(Boolean));
        const selectedMethodsList = [...window.state.selectedMethods];
        container.querySelector('#combined-activation-plot').innerHTML = `
            <div class="info">
                No traits match selected methods.<br>
                <small style="color: var(--text-secondary);">
                    Methods in data: ${[...methodsInData].join(', ') || 'none'}<br>
                    Selected methods: ${selectedMethodsList.join(', ') || 'none (check boxes above)'}
                </small>
            </div>
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

        // Store displayed values for velocity (derivative of what's shown in trajectory)
        traitActivations[traitName] = displayValues;

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

    // Build tooltips for legend (vector source info)
    const legendTooltips = filteredByMethod.map(traitName => {
        const data = traitData[traitName];
        const vs = data.metadata?.vector_source || {};
        const pos = data.metadata?.position || vs.position;
        const posStr = pos && pos !== 'response[:]' ? ` @${pos.replace('response', 'resp').replace('prompt', 'p')}` : '';
        return vs.layer !== undefined
            ? `L${vs.layer} ${vs.method || '?'}${posStr} (${vs.selection_source || 'unknown'})`
            : 'no metadata';
    });

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

    // Insert custom legend with click-to-toggle and hover-to-highlight
    const plotDiv = document.getElementById('combined-activation-plot');
    const legendDiv = window.createHtmlLegend(traces, plotDiv, {
        tooltips: legendTooltips,
        hoverHighlight: true
    });
    plotDiv.parentNode.insertBefore(legendDiv, plotDiv.nextSibling);
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
    document.querySelectorAll('.method-filter input').forEach(cb => {
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

    // Render Projection Velocity plot
    renderTokenDerivativePlots(traitActivations, filteredByMethod, tickVals, tickText, nPromptTokens, traitData);
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
 * Render Projection Velocity plot (first derivative of cosine projection)
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

// Export to global scope
window.renderTraitDynamics = renderTraitDynamics;
