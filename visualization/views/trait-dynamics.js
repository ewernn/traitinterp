// Trait Dynamics View - Watch the model's internal state evolve token-by-token
// Core insight: "See how the model is thinking" by projecting onto trait vectors
//
// Sections:
// 1. Token Trajectory: X=tokens, Y=cosine similarity (best layer) — velocity overlay toggle
// 2. Activation Magnitude Per Token: ||h|| at each token position

// Show all tokens including BOS (set to 2 to skip BOS + warmup if desired)
const START_TOKEN_IDX = 0;

// smoothData is in core/utils.js

// Shape builders (buildTurnBoundaryShapes, buildOverlayShapes, etc.) are in core/charts.js
// SENTENCE_CATEGORIES is in core/charts.js (window.SENTENCE_CATEGORIES)

/**
 * Build inline HTML legend for cue_p overlay (blue→red gradient).
 */
function buildCuePLegendHtml() {
    return `
        <span class="overlay-legend">
            <span class="overlay-legend-gradient" style="background: linear-gradient(to right, rgba(0,100,255,0.5), rgba(255,50,50,0.5));"></span>
            <span class="overlay-legend-label">0</span>
            <span class="overlay-legend-label">→</span>
            <span class="overlay-legend-label">1</span>
        </span>
    `;
}

/**
 * Build inline HTML legend for category overlay (only shows present categories).
 */
function buildCategoryLegendHtml(categoryData) {
    const presentCategories = new Set(categoryData.map(d => d.category));
    const items = [];
    for (const [key, def] of Object.entries(window.SENTENCE_CATEGORIES)) {
        if (!presentCategories.has(key)) continue;
        if (key === 'evaluate') {
            items.push(`
                <span class="overlay-legend-item">
                    <span class="overlay-legend-swatch" style="background: linear-gradient(to right, rgba(220,50,50,0.7), rgba(140,140,140,0.5), rgba(40,180,80,0.7)); width: 24px;"></span>
                    <span class="overlay-legend-label">${def.label}</span>
                </span>
            `);
        } else {
            const [r, g, b] = def.color;
            items.push(`
                <span class="overlay-legend-item">
                    <span class="overlay-legend-swatch" style="background: rgba(${r},${g},${b},${def.opacity * 3});"></span>
                    <span class="overlay-legend-label">${def.label}</span>
                </span>
            `);
        }
    }
    return `<span class="overlay-legend">${items.join('')}</span>`;
}

/**
 * Render cue_p resampling plot — small horizontal strip showing per-sentence cue_p.
 * Only shown when sentenceBoundaries with cue_p values are present.
 */
function renderCuePPlot(sentenceBoundaries, tickVals, tickText, nPromptTokens, isRollout) {
    const section = document.getElementById('cue-p-section');
    const plotDiv = document.getElementById('cue-p-plot');
    if (!sentenceBoundaries || sentenceBoundaries.length === 0) {
        if (section) section.style.display = 'none';
        return;
    }
    // Check that at least one sentence has cue_p
    const hasCueP = sentenceBoundaries.some(s => s.cue_p != null);
    if (!hasCueP) {
        if (section) section.style.display = 'none';
        return;
    }
    section.style.display = '';

    const textSecondary = window.getCssVar('--text-secondary', '#a4a4a4');
    const primaryColor = window.getCssVar('--primary-color', '#a09f6c');
    const currentTokenIdx = window.state.currentTokenIndex || 0;
    const highlightX = Math.max(0, currentTokenIdx - START_TOKEN_IDX);

    // Build step trace: each sentence is a horizontal segment at its cue_p value
    const xVals = [];
    const yVals = [];
    for (const sent of sentenceBoundaries) {
        if (sent.token_start === sent.token_end) continue;
        const x0 = nPromptTokens + sent.token_start - START_TOKEN_IDX;
        const x1 = nPromptTokens + sent.token_end - START_TOKEN_IDX;
        const cueP = sent.cue_p ?? 0;
        xVals.push(x0, x1, null);
        yVals.push(cueP, cueP, null);
    }

    const trace = {
        x: xVals,
        y: yVals,
        type: 'scatter',
        mode: 'lines',
        line: { color: 'rgba(220, 80, 60, 0.9)', width: 2, shape: 'hv' },
        fill: 'tozeroy',
        fillcolor: 'rgba(220, 80, 60, 0.12)',
        hovertemplate: 'cue_p = %{y:.2f}<extra></extra>'
    };

    // Shapes: separator (base, preserved on slider update) + highlight (replaced on slider update)
    const promptEndIdx = nPromptTokens - START_TOKEN_IDX;
    const shapes = [];
    if (!isRollout) {
        shapes.push({
            type: 'line',
            x0: promptEndIdx - 0.5, x1: promptEndIdx - 0.5,
            y0: 0, y1: 1, yref: 'paper',
            line: { color: textSecondary, width: 2, dash: 'dash' },
            _isBase: true
        });
    }
    shapes.push({
        type: 'line',
        x0: highlightX, x1: highlightX,
        y0: 0, y1: 1, yref: 'paper',
        line: { color: primaryColor, width: 2 }
    });

    const layout = window.buildChartLayout({
        preset: 'timeSeries',
        traces: [trace],
        height: 100,
        legendPosition: 'none',
        xaxis: {
            tickmode: 'array', tickvals: tickVals, ticktext: tickText,
            tickfont: { size: 8 }, showticklabels: false
        },
        yaxis: {
            title: 'cue_p', range: [0, 1.05],
            tickvals: [0, 0.5, 1], tickfont: { size: 9 },
            zeroline: false
        },
        shapes,
        margin: { l: 60, r: 20, t: 5, b: 5 }
    });
    window.renderChart(plotDiv, [trace], layout);
    window.attachTokenClickHandler(plotDiv, START_TOKEN_IDX);
}

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


/**
 * Fetch layer_sensitivity per-prompt data (multi-layer projections from model_diff analysis).
 * Returns null if not available. Caches fetched data.
 */
const layerSensitivityCache = {};
async function fetchLayerSensitivityData(experiment, promptSet, promptId) {
    // Determine variant pair from model-diff API (cached per experiment)
    if (!window._modelDiffCache || window._modelDiffCache.experiment !== experiment) {
        const data = await window.fetchJSON(`/api/experiments/${experiment}/model-diff`);
        window._modelDiffCache = { experiment, comparisons: data?.comparisons || [] };
    }
    if (window._modelDiffCache.comparisons.length === 0) return null;

    const comp = window._modelDiffCache.comparisons[0];
    const path = `experiments/${experiment}/model_diff/${comp.variant_pair}/layer_sensitivity/${promptSet}/per_prompt/${promptId}.json`;
    if (layerSensitivityCache[path]) return layerSensitivityCache[path];

    const data = await window.fetchJSON('/' + path);
    if (!data) return null;
    data._variant_a = comp.variant_a;
    data._variant_b = comp.variant_b;
    layerSensitivityCache[path] = data;
    return data;
}


/**
 * Build the control bar HTML for Token Trajectory section.
 * Pure function of state — no data dependency.
 */
function buildControlBarHtml(allFilteredTraits) {
    const isSmoothing = window.state.smoothingEnabled !== false;
    const isCentered = window.state.projectionCentered !== false;
    const currentCompareMode = window.state.compareMode || 'main';
    const isDiffMode = currentCompareMode.startsWith('diff:');
    const availableModels = window.state.availableComparisonModels || [];
    const isReplaySuffix = window.state.experimentData?.experimentConfig?.diff_convention === 'replay_suffix';
    const currentCompareVariant = isDiffMode ? currentCompareMode.slice(5) : (window.state.lastCompareVariant || availableModels[0] || '');
    const selectedOrganism = window.state.lastCompareVariant || availableModels[0] || '';

    return `
        <div class="projection-toggle">
            ${ui.renderToggle({ id: 'smoothing-toggle', label: 'Smooth', checked: isSmoothing, className: 'projection-toggle-checkbox' })}
            ${isSmoothing ? `<select id="smoothing-window-select" style="margin-left: -4px; width: 42px; font-size: var(--text-xs);" title="Moving average window size (tokens)">
                ${[1,2,3,4,5,6,7,8,9,10,15,20,25].map(n => `<option value="${n}" ${n === (window.state.smoothingWindow || 5) ? 'selected' : ''}>${n}</option>`).join('')}
            </select>` : ''}
            ${ui.renderToggle({ id: 'projection-centered-toggle', label: 'Centered', checked: isCentered, className: 'projection-toggle-checkbox' })}
            <span class="projection-toggle-label" style="margin-left: 16px;">Methods:</span>
            ${ui.renderToggle({ label: 'probe', checked: window.state.selectedMethods.has('probe'), dataAttr: { key: 'method', value: 'probe' }, className: 'projection-toggle-checkbox method-filter' })}
            ${ui.renderToggle({ label: 'mean_diff', checked: window.state.selectedMethods.has('mean_diff'), dataAttr: { key: 'method', value: 'mean_diff' }, className: 'projection-toggle-checkbox method-filter' })}
            ${ui.renderToggle({ label: 'gradient', checked: window.state.selectedMethods.has('gradient'), dataAttr: { key: 'method', value: 'gradient' }, className: 'projection-toggle-checkbox method-filter' })}
            ${ui.renderToggle({ label: 'random', checked: window.state.selectedMethods.has('random'), dataAttr: { key: 'method', value: 'random' }, className: 'projection-toggle-checkbox method-filter' })}
        </div>
        <div class="projection-toggle">
            <span class="projection-toggle-label">Mode:</span>
            <select id="projection-mode-select" style="margin-left: 4px;" title="Cosine: proj/||h|| (removes magnitude). Normalized: proj/avg||h|| (preserves per-token variance, removes layer scale).">
                <option value="cosine" ${window.state.projectionMode === 'cosine' ? 'selected' : ''}>Cosine</option>
                <option value="normalized" ${window.state.projectionMode !== 'cosine' ? 'selected' : ''}>Normalized</option>
            </select>
            <span class="projection-toggle-label" style="margin-left: 12px;">Clean:</span>
            <select id="massive-dims-cleaning-select" style="margin-left: 4px;" title="Remove high-magnitude bias dimensions (Sun et al. 2024). These dims have 100-1000x larger values than typical dims and act as constant biases.">
                <option value="none" ${!window.state.massiveDimsCleaning || window.state.massiveDimsCleaning === 'none' ? 'selected' : ''}>No cleaning</option>
                <option value="top5-3layers" ${window.state.massiveDimsCleaning === 'top5-3layers' ? 'selected' : ''}>Top 5, 3+ layers</option>
                <option value="all" ${window.state.massiveDimsCleaning === 'all' ? 'selected' : ''}>All candidates</option>
            </select>
            <span class="projection-toggle-label" style="margin-left: 12px;">Layers:</span>
            ${ui.renderToggle({ id: 'layer-mode-toggle', label: '', checked: window.state.layerMode, className: 'projection-toggle-checkbox' })}
            ${window.state.layerMode ? `
            <select id="layer-mode-trait-select" style="margin-left: 4px;" title="Select trait to view across all available layers">
                ${allFilteredTraits.map(t =>
                    `<option value="${t.name}" ${t.name === window.state.layerModeTrait ? 'selected' : ''}>${window.getDisplayName(t.name)}</option>`
                ).join('')}
            </select>
            ` : ''}
            ${availableModels.length > 0 && isReplaySuffix ? `
            <span class="projection-toggle-label" style="margin-left: 12px;">Organism:</span>
            <select id="compare-variant-select" style="margin-left: 4px;">
                ${availableModels.map(m => `
                    <option value="${m}" ${m === selectedOrganism ? 'selected' : ''}>${m}</option>
                `).join('')}
            </select>
            ${ui.renderFilterChip('main', 'Main', isDiffMode ? '' : 'main', 'compare-mode')}
            ${ui.renderFilterChip('diff', 'Diff', isDiffMode ? 'diff' : '', 'compare-mode')}
            ` : availableModels.length > 0 ? `
            <span class="projection-toggle-label" style="margin-left: 12px;">Compare:</span>
            ${ui.renderFilterChip('main', 'Main', isDiffMode ? '' : 'main', 'compare-mode')}
            ${ui.renderFilterChip('diff', 'Diff', isDiffMode ? 'diff' : '', 'compare-mode')}
            ${isDiffMode ? `
            <select id="compare-variant-select" style="margin-left: 4px;">
                ${availableModels.map(m => `
                    <option value="${m}" ${m === currentCompareVariant ? 'selected' : ''}>${m}</option>
                `).join('')}
            </select>
            ` : ''}
            ` : ''}
            <span class="projection-toggle-label" style="margin-left: 12px;">Wide:</span>
            ${ui.renderToggle({ id: 'wide-mode-toggle', label: '', checked: window.state.wideMode, className: 'projection-toggle-checkbox' })}
            ${ui.renderToggle({ id: 'velocity-toggle', label: 'Velocity', checked: window.state.showVelocity, className: 'projection-toggle-checkbox' })}
        </div>
    `;
}


/**
 * Build full page shell HTML with controls and empty chart divs.
 * Renders independently of data — controls always accessible.
 */
function buildPageShellHtml(allFilteredTraits) {
    const projectionMode = window.state.projectionMode || 'cosine';
    const isCentered = window.state.projectionCentered !== false;
    const isSmoothing = window.state.smoothingEnabled !== false;

    return `
        <div class="tool-view${window.state.wideMode ? ' wide-mode' : ''}">
            <div class="page-intro">
                <div class="page-intro-text">Watch traits evolve token-by-token during generation.</div>
                <div id="trait-dynamics-status"></div>
            </div>

            <section>
                ${ui.renderSubsection({
                    title: 'Token Trajectory',
                    infoId: 'info-token-trajectory',
                    infoText: (projectionMode === 'normalized'
                        ? 'Normalized projection: proj / avg||h||. Preserves per-token variance, removes layer-dependent scale.'
                        : 'Cosine similarity: proj / ||h||. Shows directional alignment with trait vector.') +
                        (isCentered ? ' Centered by subtracting BOS token value.' : '') +
                        (isSmoothing ? ` Smoothed with ${window.state.smoothingWindow || 5}-token moving average.` : ''),
                    level: 'h2'
                })}
                ${buildControlBarHtml(allFilteredTraits)}
                <div id="overlay-controls"></div>
                <div id="combined-activation-plot"></div>
                <div id="top-spans-panel"></div>
            </section>

            <section id="cue-p-section" style="display:none">
                ${ui.renderSubsection({
                    title: 'Resampling cue_p',
                    infoId: 'info-cue-p',
                    infoText: 'Per-sentence resampling probability of the cued (wrong) answer, from Thought Branches transplant experiment (~4000 forward passes per sentence). Shows how bias accumulates through the CoT.'
                })}
                <div id="cue-p-plot"></div>
            </section>

            <section>
                <div id="trait-heatmap-panel"></div>
            </section>

            <section>
                ${ui.renderSubsection({
                    title: 'Activation Magnitude Per Token',
                    infoId: 'info-token-magnitude',
                    infoText: 'L2 norm of activation per token. Shows one line per unique layer used by traits above. Compare to trajectory - similar magnitudes but low projections means token encodes orthogonal information.'
                })}
                <div id="token-magnitude-plot"></div>
            </section>

        </div>
    `;
}


/**
 * Attach event listeners for all control bar elements.
 * Called once after page shell is rendered.
 */
function attachControlListeners(allFilteredTraits) {
    const isReplaySuffix = window.state.experimentData?.experimentConfig?.diff_convention === 'replay_suffix';
    const availableModels = window.state.availableComparisonModels || [];

    const smoothingCheckbox = document.getElementById('smoothing-toggle');
    if (smoothingCheckbox) {
        smoothingCheckbox.addEventListener('change', () => {
            window.setSmoothing(smoothingCheckbox.checked);
        });
    }
    const smoothingWindowSelect = document.getElementById('smoothing-window-select');
    if (smoothingWindowSelect) {
        smoothingWindowSelect.addEventListener('change', () => {
            window.setSmoothingWindow(parseInt(smoothingWindowSelect.value));
        });
    }
    const centeredCheckbox = document.getElementById('projection-centered-toggle');
    if (centeredCheckbox) {
        centeredCheckbox.addEventListener('change', () => {
            window.setProjectionCentered(centeredCheckbox.checked);
        });
    }
    const massiveDimsSelect = document.getElementById('massive-dims-cleaning-select');
    if (massiveDimsSelect) {
        massiveDimsSelect.addEventListener('change', () => {
            window.setMassiveDimsCleaning(massiveDimsSelect.value);
        });
    }
    document.querySelectorAll('.method-filter input').forEach(cb => {
        cb.addEventListener('change', () => {
            window.toggleMethod(cb.dataset.method);
        });
    });
    const projectionModeSelect = document.getElementById('projection-mode-select');
    if (projectionModeSelect) {
        projectionModeSelect.addEventListener('change', () => {
            window.setProjectionMode(projectionModeSelect.value);
        });
    }
    // Velocity toggle
    const velocityToggle = document.getElementById('velocity-toggle');
    if (velocityToggle) {
        velocityToggle.addEventListener('change', () => {
            window.setShowVelocity(velocityToggle.checked);
        });
    }
    // Compare mode toggle (Main/Diff chips)
    document.querySelectorAll('[data-compare-mode]').forEach(chip => {
        chip.addEventListener('click', () => {
            const mode = chip.dataset.compareMode;
            if (mode === 'main') {
                window.setCompareMode('main');
            } else {
                if (isReplaySuffix) {
                    window.setCompareMode('diff:replay');
                } else {
                    const variant = (window.state.lastCompareVariant && availableModels.includes(window.state.lastCompareVariant))
                        ? window.state.lastCompareVariant
                        : availableModels[0];
                    if (variant) window.setCompareMode('diff:' + variant);
                }
            }
        });
    });
    // Compare variant / organism dropdown
    const compareVariantSelect = document.getElementById('compare-variant-select');
    if (compareVariantSelect) {
        compareVariantSelect.addEventListener('change', () => {
            window.state.lastCompareVariant = compareVariantSelect.value;
            localStorage.setItem('lastCompareVariant', compareVariantSelect.value);
            if (isReplaySuffix) {
                if (window.renderView) window.renderView();
            } else {
                window.setCompareMode('diff:' + compareVariantSelect.value);
            }
        });
    }
    const layerModeToggle = document.getElementById('layer-mode-toggle');
    if (layerModeToggle) {
        layerModeToggle.addEventListener('change', () => {
            window.setLayerMode(layerModeToggle.checked);
        });
    }
    const layerModeTraitSelect = document.getElementById('layer-mode-trait-select');
    if (layerModeTraitSelect) {
        layerModeTraitSelect.addEventListener('change', () => {
            window.setLayerModeTrait(layerModeTraitSelect.value);
        });
    }
    const wideModeToggle = document.getElementById('wide-mode-toggle');
    if (wideModeToggle) {
        wideModeToggle.addEventListener('change', () => {
            window.setWideMode(wideModeToggle.checked);
        });
    }
}


async function renderTraitDynamics() {
    const contentArea = document.getElementById('content-area');

    if (ui.requireExperiment(contentArea)) return;

    const allFilteredTraits = window.getFilteredTraits();

    // In layer mode, narrow to single trait (multi-vector expansion handles layers)
    let filteredTraits;
    if (window.state.layerMode && allFilteredTraits.length > 0) {
        // Default to first trait if none selected
        if (!window.state.layerModeTrait) {
            window.state.layerModeTrait = allFilteredTraits[0].name;
        }
        const match = allFilteredTraits.find(t => t.name === window.state.layerModeTrait);
        filteredTraits = match ? [match] : [allFilteredTraits[0]];
    } else {
        filteredTraits = allFilteredTraits;
    }

    // Preserve scroll position
    const scrollY = contentArea.scrollTop;

    // Always render page shell with controls (so they remain accessible even when no data loads)
    contentArea.innerHTML = buildPageShellHtml(allFilteredTraits);
    window.setupSubsectionInfoToggles();
    attachControlListeners(allFilteredTraits);

    if (filteredTraits.length === 0) {
        document.getElementById('combined-activation-plot').innerHTML =
            `<div class="info">Select at least one trait from the sidebar to view activation trajectories.</div>`;
        requestAnimationFrame(() => { contentArea.scrollTop = scrollY; });
        return;
    }

    const traitData = {};
    const failedTraits = [];
    const promptSet = window.state.currentPromptSet;
    const promptId = window.state.currentPromptId;

    // Detect replay_suffix convention: organism is the main variant, instruct is the replay comparison
    const isReplaySuffix = window.state.experimentData?.experimentConfig?.diff_convention === 'replay_suffix';
    const availableModelsEarly = window.state.availableComparisonModels || [];

    let modelVariant;
    if (isReplaySuffix && availableModelsEarly.length > 0) {
        // In replay_suffix mode, the selected organism is the main variant
        const selectedOrg = window.state.lastCompareVariant || availableModelsEarly[0];
        modelVariant = availableModelsEarly.includes(selectedOrg) ? selectedOrg : availableModelsEarly[0];
    } else {
        modelVariant = window.getVariantForCurrentPromptSet();
    }

    if (!promptSet || !promptId) {
        const promptLabel = promptSet ? `${promptSet}/—` : 'none selected';
        document.getElementById('combined-activation-plot').innerHTML =
            `<div class="info">No data available for prompt ${promptLabel} for any selected trait.</div>`;
        requestAnimationFrame(() => { contentArea.scrollTop = scrollY; });
        return;
    }

    const { cancel: cancelLoading } = ui.deferredLoading('combined-activation-plot', `Loading data for ${filteredTraits.length} trait(s)...`);

    // Load shared response data (prompt/response text and tokens)
    const responseData = await window.fetchJSON(window.paths.responseData(promptSet, promptId, modelVariant));

    // Load projection data for ALL selected traits (in parallel)
    const projectionResults = await Promise.all(filteredTraits.map(async (trait) => {
        const projData = await window.fetchJSON(window.paths.residualStreamData(trait, promptSet, promptId, modelVariant));
        if (!projData || projData.error) return { trait, error: true };
        return { trait, projData };
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

        // Handle multi-vector format
        if (projData.metadata?.multi_vector && Array.isArray(projData.projections)) {
            if (window.state.layerMode) {
                // Layers mode ON: expand all layers into separate entries
                for (const vecProj of projData.projections) {
                    const key = `${trait.name}__${vecProj.method}_L${vecProj.layer}`;
                    traitData[key] = {
                        ...projData,
                        projections: { prompt: vecProj.prompt, response: vecProj.response },
                        token_norms: vecProj.token_norms || projData.token_norms || null,
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
                // Layers mode OFF: pick single best layer per trait
                // Target: best_steering_layer + floor(0.1 * num_layers), snap to closest available
                const numLayers = window.paths?.getNumLayers?.() || 48;
                const offset = Math.floor(0.1 * numLayers);
                const steeringEntry = projData.projections.find(p => p.selection_source === 'steering');
                let targetLayer;
                if (steeringEntry) {
                    targetLayer = steeringEntry.layer + offset;
                } else {
                    // Fallback: 60% depth (roughly where best+10% lands for most traits)
                    targetLayer = Math.floor(0.6 * numLayers);
                }
                // Snap to closest available layer
                const vecProj = projData.projections.reduce((best, p) =>
                    Math.abs(p.layer - targetLayer) < Math.abs(best.layer - targetLayer) ? p : best
                );
                traitData[trait.name] = {
                    ...projData,
                    projections: { prompt: vecProj.prompt, response: vecProj.response },
                    token_norms: vecProj.token_norms || projData.token_norms || null,
                    metadata: {
                        ...projData.metadata,
                        vector_source: {
                            layer: vecProj.layer,
                            method: vecProj.method,
                            selection_source: vecProj.selection_source,
                            baseline: vecProj.baseline,
                            position: projData.metadata?.position
                        }
                    }
                };
            }
        } else {
            traitData[trait.name] = projData;
        }
    }

    cancelLoading();

    // Handle compare mode: "main", "diff:{model}", or "show:{model}"
    const compareMode = window.state.compareMode || 'main';
    const isDiff = compareMode.startsWith('diff:');
    const isShow = compareMode.startsWith('show:');
    const compareModel = isDiff ? compareMode.slice(5) : isShow ? compareMode.slice(5) : null;

    // For replay_suffix: diff always compares organism (main) vs instruct replay
    const replayDiff = isReplaySuffix && (isDiff || (compareMode === 'diff'));
    const effectiveCompareModel = replayDiff ? null : compareModel;  // replay handles its own fetch

    if (replayDiff) {
        // Replay suffix convention: fetch instruct data from {promptSet}_replay_{organism}
        const appVariant = window.state.experimentData?.experimentConfig?.defaults?.application || 'instruct';
        const replayPromptSet = `${promptSet}_replay_${modelVariant}`;
        const traitKeys = Object.keys(traitData);

        const compResults = await Promise.all(traitKeys.map(async (traitKey) => {
            const baseTrait = traitData[traitKey].metadata?._baseTrait || traitKey;
            const trait = filteredTraits.find(t => t.name === baseTrait);
            if (!trait) return null;

            try {
                const fetchPath = window.paths.residualStreamData(trait, replayPromptSet, promptId, appVariant);
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
                // Fallback: if no layer match, use first available projection
                if (!compProj && compData.projections.length > 0) {
                    const fb = compData.projections[0];
                    compProj = { prompt: fb.prompt, response: fb.response };
                }
            } else {
                compProj = compData.projections;
            }

            if (compProj) {
                // Diff: organism - instruct_replay (positive = organism has more trait)
                // Trim to min length to avoid EOS mismatch (organism has <|eot_id|>, replay doesn't)
                const mainProj = traitData[traitKey].projections;
                const rLenDiff = Math.abs(mainProj.response.length - compProj.response.length);
                if (rLenDiff > 1) console.warn(`[Diff] Unexpected response length mismatch for ${traitKey}: ${mainProj.response.length} vs ${compProj.response.length} (diff=${rLenDiff})`);
                const minPromptLen = Math.min(mainProj.prompt.length, compProj.prompt.length);
                const minResponseLen = Math.min(mainProj.response.length, compProj.response.length);
                const diffPrompt = mainProj.prompt.slice(0, minPromptLen).map((v, i) => v - compProj.prompt[i]);
                const diffResponse = mainProj.response.slice(0, minResponseLen).map((v, i) => v - compProj.response[i]);
                traitData[traitKey].projections = { prompt: diffPrompt, response: diffResponse };
                traitData[traitKey].metadata = traitData[traitKey].metadata || {};
                traitData[traitKey].metadata._isDiff = true;
                traitData[traitKey].metadata._compareModel = appVariant;
            }
        }
    } else if (effectiveCompareModel) {
        const traitKeys = Object.keys(traitData);
        const compResults = await Promise.all(traitKeys.map(async (traitKey) => {
            const baseTrait = traitData[traitKey].metadata?._baseTrait || traitKey;
            const trait = filteredTraits.find(t => t.name === baseTrait);
            if (!trait) return null;

            try {
                // Fetch from comparison model's path (same prompt set, different model)
                const fetchPath = window.paths.residualStreamData(trait, promptSet, promptId, effectiveCompareModel);
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
                // Fallback: if no layer match, use first available projection
                if (!compProj && compData.projections.length > 0) {
                    const fb = compData.projections[0];
                    compProj = { prompt: fb.prompt, response: fb.response };
                }
            } else {
                compProj = compData.projections;
            }

            if (compProj) {
                if (isDiff) {
                    // Diff mode: comparison - main (positive = more trait in comparison model)
                    // Trim to min length to handle potential token count mismatches
                    const mainProj = traitData[traitKey].projections;
                    const minPLen = Math.min(mainProj.prompt.length, compProj.prompt.length);
                    const minRLen = Math.min(mainProj.response.length, compProj.response.length);
                    const diffPrompt = mainProj.prompt.slice(0, minPLen).map((v, i) => compProj.prompt[i] - v);
                    const diffResponse = mainProj.response.slice(0, minRLen).map((v, i) => compProj.response[i] - v);
                    traitData[traitKey].projections = { prompt: diffPrompt, response: diffResponse };
                    traitData[traitKey].metadata = traitData[traitKey].metadata || {};
                    traitData[traitKey].metadata._isDiff = true;
                    traitData[traitKey].metadata._compareModel = effectiveCompareModel;
                } else if (isShow) {
                    // Show mode: replace with comparison model's projections
                    traitData[traitKey].projections = compProj;
                    traitData[traitKey].metadata = traitData[traitKey].metadata || {};
                    traitData[traitKey].metadata._isComparisonModel = true;
                    traitData[traitKey].metadata._compareModel = effectiveCompareModel;
                }
            }
        }
    }

    // Layer mode: overlay multi-layer projections from layer_sensitivity data
    if (window.state.layerMode) {
        const layerSensData = await fetchLayerSensitivityData(
            window.state.currentExperiment, promptSet, promptId
        );
        const selectedTrait = window.state.layerModeTrait;
        const traitLayers = layerSensData?.traits?.[selectedTrait]?.layers;

        if (traitLayers) {
            // Determine which projection values to use
            const isDiffMode = (compareMode || 'main').startsWith('diff:') || replayDiff;
            // In replay_suffix: modelVariant is the organism (variant_b)
            // In standard: modelVariant is the application default (variant_a)
            const projKey = isDiffMode ? 'delta'
                : (modelVariant === layerSensData._variant_b ? 'proj_b' : 'proj_a');

            // Get prompt projections from the existing single-vector entry (if available)
            const existingEntry = traitData[selectedTrait];
            const promptProj = existingEntry?.projections?.prompt || [];

            // Remove the single-vector entry
            delete traitData[selectedTrait];

            // Create multi-vector entries from layer_sensitivity data
            for (const [layerStr, ldata] of Object.entries(traitLayers)) {
                const layer = parseInt(layerStr);
                const responseProj = ldata[projKey] || ldata.proj_a || [];
                const key = `${selectedTrait}__probe_L${layer}`;

                traitData[key] = {
                    projections: { prompt: promptProj, response: responseProj },
                    prompt: existingEntry?.prompt || responseData?.prompt || { text: '', tokens: [] },
                    response: existingEntry?.response || { text: '', tokens: layerSensData.response_tokens || [] },
                    token_norms: existingEntry?.token_norms || null,
                    metadata: {
                        vector_source: { layer, method: 'probe' },
                        _baseTrait: selectedTrait,
                        _isMultiVector: true,
                        ...(isDiffMode ? { _isDiff: true, _compareModel: layerSensData._variant_a } : {}),
                    },
                };
            }
        }
    }

    // Check if we have any data
    const loadedTraits = Object.keys(traitData);
    if (loadedTraits.length === 0) {
        const promptLabel = promptSet && promptId ? `${promptSet}/${promptId}` : 'none selected';
        document.getElementById('combined-activation-plot').innerHTML = `
            <div class="info">
                No data available for prompt ${promptLabel} for any selected trait.
            </div>
            <p class="tool-description">
                To capture per-token activation data, run:
            </p>
            <pre>python inference/capture_activations.py --experiment ${window.paths.getExperiment()} --prompt-set ${promptSet || 'PROMPT_SET'}</pre>
        `;
        requestAnimationFrame(() => { contentArea.scrollTop = scrollY; });
        return;
    }

    // Fetch annotations for shaded bands (non-blocking)
    let annotationTokenRanges = [];
    if (responseData) {
        const responseTokens = responseData.tokens
            ? responseData.tokens.slice(responseData.prompt_end)
            : responseData.response?.tokens || [];
        const responseText = typeof responseData.response === 'string'
            ? responseData.response
            : responseData.response?.text || '';
        if (responseTokens.length > 0 && responseText) {
            annotationTokenRanges = await window.annotations.getAnnotationTokenRanges(
                window.state.currentExperiment, modelVariant, promptSet, promptId,
                responseTokens, responseText
            );
        }
    }

    // Extract rollout/sentence metadata from response data
    const turnBoundaries = responseData?.turn_boundaries || null;
    const sentenceBoundaries = responseData?.sentence_boundaries || null;

    // Fetch sentence category annotations (thought branches only)
    let sentenceCategoryData = null;
    if (sentenceBoundaries?.length > 0) {
        sentenceCategoryData = await window.annotations.getSentenceCategoriesForPrompt(
            window.state.currentExperiment, promptSet, promptId, sentenceBoundaries);
    }

    // Render the full view
    await renderCombinedGraph(traitData, loadedTraits, failedTraits, annotationTokenRanges, turnBoundaries, sentenceBoundaries, sentenceCategoryData);

    // Restore scroll position after DOM updates
    requestAnimationFrame(() => {
        contentArea.scrollTop = scrollY;
    });
}

async function renderCombinedGraph(traitData, loadedTraits, failedTraits, annotationTokenRanges = [], turnBoundaries = null, sentenceBoundaries = null, sentenceCategoryData = null) {
    // Detect replay_suffix convention (needed for UI controls and event handlers)
    const isReplaySuffix = window.state.experimentData?.experimentConfig?.diff_convention === 'replay_suffix';

    // Use first trait's data as reference for tokens (they should all be the same)
    const refData = traitData[loadedTraits[0]];

    if (!refData.prompt?.tokens || !refData.response?.tokens) {
        document.getElementById('combined-activation-plot').innerHTML = `<div class="info">Error: Missing token data.</div>`;
        return;
    }

    const promptTokens = refData.prompt.tokens;
    const responseTokens = refData.response.tokens;
    const allTokens = [...promptTokens, ...responseTokens];
    const nPromptTokens = promptTokens.length;
    const isRollout = responseTokens.length === 0;
    const inferenceModel = refData.metadata?.inference_model ||
        window.state.experimentData?.experimentConfig?.application_model || 'unknown';
    const modelInfoHtml = `Inference model: <code>${inferenceModel}</code>`;

    const failedHtml = failedTraits.length > 0
        ? `<div class="tool-description">No data for: ${failedTraits.map(t => window.getDisplayName(t)).join(', ')}</div>`
        : '';

    // Determine smoothing and centering
    const isSmoothing = window.state.smoothingEnabled !== false;  // default true
    const isCentered = window.state.projectionCentered !== false;  // default true

    // Check what we're showing
    const showingDiff = Object.values(traitData).some(d => d.metadata?._isDiff);
    const showingCompModel = Object.values(traitData).some(d => d.metadata?._isComparisonModel);
    const compareModelName = Object.values(traitData).find(d => d.metadata?._compareModel)?.metadata?._compareModel;

    let compareInfoHtml = '';
    if (showingDiff && isReplaySuffix) {
        const organismName = window.state.lastCompareVariant || (window.state.availableComparisonModels || [])[0] || 'organism';
        compareInfoHtml = `<div class="page-intro-text" style="color: var(--color-accent); font-weight: 500;">
            Showing DIFF: ${organismName} − instruct replay
           </div>`;
    } else if (showingDiff) {
        compareInfoHtml = `<div class="page-intro-text" style="color: var(--color-accent); font-weight: 500;">
            Showing DIFF: ${compareModelName} − application model
           </div>`;
    } else if (showingCompModel) {
        compareInfoHtml = `<div class="page-intro-text" style="color: var(--color-accent); font-weight: 500;">
            Showing: ${compareModelName} (comparison model)
           </div>`;
    }

    // Update status info in pre-rendered shell
    const statusDiv = document.getElementById('trait-dynamics-status');
    if (statusDiv) {
        statusDiv.innerHTML = `${compareInfoHtml}<div class="page-intro-model">${modelInfoHtml}</div>`;
    }
    if (failedHtml) {
        document.getElementById('combined-activation-plot').insertAdjacentHTML('beforebegin', failedHtml);
    }

    // Prepare data for plotting
    const traitActivations = {};  // Store smoothed activations for heatmap + velocity overlay

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
        document.getElementById('combined-activation-plot').innerHTML = `
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

    const projectionMode = window.state.projectionMode || 'cosine';

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

        // Compute projection values based on mode
        let rawValues;
        if (data.token_norms) {
            let promptNorms = data.token_norms.prompt;
            let responseNorms = data.token_norms.response;

            // Use cleaned norms if massive dims were removed
            if (dimsToRemove.length > 0 && mdd) {
                promptNorms = computeCleanedNorms(promptNorms, mdd, dimsToRemove, 'prompt');
                responseNorms = computeCleanedNorms(responseNorms, mdd, dimsToRemove, 'response');
            }

            if (projectionMode === 'normalized') {
                // Normalized mode: divide by mean response norm (preserves per-token variance)
                // For rollouts (empty response), fall back to prompt norms
                const normsForMean = isRollout ? promptNorms : responseNorms;
                const meanNorm = normsForMean.length > 0
                    ? normsForMean.reduce((a, b) => a + b, 0) / normsForMean.length
                    : 1;
                rawValues = rawProj.map(proj => meanNorm > 0 ? proj / meanNorm : 0);
            } else {
                // Cosine mode: divide by per-token norm
                const traitTokenNorms = [...promptNorms, ...responseNorms].slice(START_TOKEN_IDX);
                rawValues = rawProj.map((proj, i) => {
                    const norm = traitTokenNorms[i];
                    return norm > 0 ? proj / norm : 0;
                });
            }
        } else {
            rawValues = rawProj;
        }

        // Store normalized values for Top Spans (before centering/smoothing)
        // For rollouts, use all values (Top Spans hidden but keeps data consistent)
        data._normalizedResponse = isRollout
            ? rawValues
            : rawValues.slice(nPromptTokens - START_TOKEN_IDX);

        // Subtract BOS value if centering is enabled (makes token 0 = 0)
        if (isCentered && rawValues.length > 0) {
            const bosValue = rawValues[0];
            rawValues = rawValues.map(v => v - bosValue);
        }

        // Apply N-token moving average if smoothing is enabled
        const displayValues = isSmoothing ? window.smoothData(rawValues, window.state.smoothingWindow || 5) : rawValues;

        // Store displayed values for velocity (derivative of what's shown in trajectory)
        traitActivations[traitName] = displayValues;

        // Color: layer-depth scale in layer mode, standard palette otherwise
        let color;
        if (window.state.layerMode && data.metadata?._isMultiVector) {
            const layer = data.metadata?.vector_source?.layer || 0;
            const allLayers = filteredByMethod.map(t => traitData[t]?.metadata?.vector_source?.layer).filter(l => l != null);
            const minL = Math.min(...allLayers);
            const maxL = Math.max(...allLayers);
            const t = maxL > minL ? (layer - minL) / (maxL - minL) : 0.5;
            // Light blue (early layers) → dark blue (late layers)
            const r = Math.round(180 - t * 130);
            const g = Math.round(210 - t * 130);
            const b = Math.round(255 - t * 55);
            color = `rgb(${r},${g},${b})`;
        } else {
            color = window.getChartColors()[idx % 10];
        }

        const method = vs.method || 'probe';
        const valueLabel = projectionMode === 'normalized' ? 'Normalized' : 'Cosine';
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

        const useMarkers = displayValues.length <= 2000;
        traces.push({
            x: Array.from({length: displayValues.length}, (_, i) => i),
            y: displayValues,
            type: 'scatter',
            mode: useMarkers ? 'lines+markers' : 'lines',
            name: displayName,
            line: { color: color, width: 1.5 },
            ...(useMarkers ? { marker: { size: 2, color: color } } : {}),
            hovertemplate: hoverText
        });
    }

    // Get display tokens (adaptive spacing for x-axis labels)
    const displayTokens = allTokens.slice(START_TOKEN_IDX);
    const tickStep = Math.max(10, Math.floor(displayTokens.length / 80));
    const tickVals = [];
    const tickText = [];
    for (let i = 0; i < displayTokens.length; i += tickStep) {
        tickVals.push(i);
        tickText.push(displayTokens[i]);
    }

    // Get colors from CSS variables
    const textSecondary = window.getCssVar('--text-secondary', '#a4a4a4');
    const primaryColor = window.getCssVar('--primary-color', '#a09f6c');

    // Current token highlight
    const currentTokenIdx = window.state.currentTokenIndex || 0;
    const highlightX = Math.max(0, currentTokenIdx - START_TOKEN_IDX);

    // Shapes: separator, highlight, turn/sentence boundaries, annotation bands
    const shapes = [];

    // Prompt/response separator (skip for rollouts — separator would be at rightmost edge)
    if (!isRollout) {
        shapes.push({
            type: 'line',
            x0: (nPromptTokens - START_TOKEN_IDX) - 0.5,
            x1: (nPromptTokens - START_TOKEN_IDX) - 0.5,
            y0: 0, y1: 1, yref: 'paper',
            line: { color: textSecondary, width: 2, dash: 'dash' },
            _isBase: true
        });
    }

    // Current token highlight (always)
    shapes.push({
        type: 'line',
        x0: highlightX, x1: highlightX,
        y0: 0, y1: 1, yref: 'paper',
        line: { color: primaryColor, width: 2 }
    });

    // Turn boundary bands (rollouts: colored by role)
    shapes.push(...window.buildTurnBoundaryShapes(turnBoundaries).map(s => ({ ...s, _isBase: true })));

    // Sentence overlay bands (cue_p and/or category, respecting toggle state)
    shapes.push(...window.buildOverlayShapes(sentenceBoundaries, sentenceCategoryData, nPromptTokens).map(s => ({ ...s, _isBase: true })));

    // Annotation shaded bands (response token ranges offset by nPromptTokens)
    for (const [start, end] of annotationTokenRanges) {
        shapes.push({
            type: 'rect',
            x0: (nPromptTokens - START_TOKEN_IDX) + start - 0.5,
            x1: (nPromptTokens - START_TOKEN_IDX) + end - 0.5,
            y0: 0, y1: 1, yref: 'paper',
            fillcolor: 'rgba(255, 180, 60, 0.12)',
            line: { width: 0 },
            layer: 'below',
            _isBase: true
        });
    }

    // PROMPT/RESPONSE labels (skip for rollouts — turn boundaries replace them)
    const annotations = [];
    if (!isRollout) {
        annotations.push(
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
        );
    }

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
    const yAxisTitle = projectionMode === 'normalized'
        ? 'Normalized (proj / avg\u2016h\u2016)'
        : 'Cosine (proj / \u2016h\u2016)';

    // Compute y-axis range: minimum ±0.15, auto-expand if data exceeds
    let yAxisConfig = { title: yAxisTitle, zeroline: true, zerolinewidth: 1, showgrid: true };
    // Find actual data range across all traces (skip first few special tokens for auto-range)
    const rangeSkip = Math.min(4, nPromptTokens);
    let minY = Infinity, maxY = -Infinity;
    traces.forEach(t => {
        t.y.forEach((v, i) => {
            if (i < rangeSkip) return;
            if (v < minY) minY = v;
            if (v > maxY) maxY = v;
        });
    });
    // Pad y-axis: 15% of data range, minimum ±0.02 (auto-zooms for diff mode)
    const pad = Math.max(0.02, (maxY - minY) * 0.15);
    const rangeMin = minY - pad;
    const rangeMax = maxY + pad;
    yAxisConfig.range = [rangeMin, rangeMax];

    // Velocity overlay on secondary y-axis (when toggled on)
    if (window.state.showVelocity) {
        for (let idx = 0; idx < filteredByMethod.length; idx++) {
            const traitName = filteredByMethod[idx];
            const activations = traitActivations[traitName];
            if (!activations) continue;
            const velocity = computeVelocity(activations);
            const smoothedVelocity = window.smoothData(velocity, window.state.smoothingWindow || 5);
            const color = traces[idx]?.line?.color || window.getChartColors()[idx % 10];
            traces.push({
                x: Array.from({length: smoothedVelocity.length}, (_, i) => i + 0.5),
                y: smoothedVelocity,
                type: 'scatter',
                mode: 'lines',
                name: `${traces[idx]?.name || window.getDisplayName(traitName)} (vel)`,
                line: { color, width: 1, dash: 'dot' },
                yaxis: 'y2',
                showlegend: false,
                hovertemplate: `<b>${traces[idx]?.name || window.getDisplayName(traitName)}</b><br>Token %{x:.0f}<br>Velocity: %{y:.4f}<extra></extra>`
            });
        }
    }

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
            tickfont: { size: 9 },
        },
        yaxis: yAxisConfig,
        shapes: shapes,
        annotations: annotations,
        margin: { l: 60, r: window.state.showVelocity ? 60 : 20, t: 40, b: 80 },
        hovermode: 'closest'
    });

    // Add secondary y-axis for velocity overlay
    if (window.state.showVelocity) {
        mainLayout.yaxis2 = {
            title: 'Velocity',
            overlaying: 'y',
            side: 'right',
            zeroline: true,
            zerolinewidth: 1,
            showgrid: false,
            tickfont: { size: 9 }
        };
    }

    window.renderChart('combined-activation-plot', traces, mainLayout);

    // Insert custom legend with click-to-toggle and hover-to-highlight
    const plotDiv = document.getElementById('combined-activation-plot');
    const legendDiv = window.createHtmlLegend(traces, plotDiv, {
        tooltips: legendTooltips,
        hoverHighlight: true
    });
    plotDiv.parentNode.insertBefore(legendDiv, plotDiv.nextSibling);
    window.attachTokenClickHandler(plotDiv, START_TOKEN_IDX);

    // Populate overlay controls (only when sentence boundary data exists)
    const overlayControlsDiv = document.getElementById('overlay-controls');
    if (overlayControlsDiv) {
        if (sentenceBoundaries && sentenceBoundaries.length > 0) {
            const showCueP = window.state.showCuePOverlay;
            const showCategory = window.state.showCategoryOverlay;
            const hasCategoryData = sentenceCategoryData && sentenceCategoryData.length > 0;

            overlayControlsDiv.innerHTML = `
                <div class="overlay-controls-bar">
                    <span class="projection-toggle-label">Overlays:</span>
                    ${ui.renderToggle({ id: 'cue-p-overlay-toggle', label: 'cue_p', checked: showCueP, className: 'projection-toggle-checkbox' })}
                    ${showCueP ? buildCuePLegendHtml() : ''}
                    ${hasCategoryData ? ui.renderToggle({ id: 'category-overlay-toggle', label: 'Category', checked: showCategory, className: 'projection-toggle-checkbox' }) : ''}
                    ${showCategory && hasCategoryData ? buildCategoryLegendHtml(sentenceCategoryData) : ''}
                </div>
            `;

            const cuePToggle = document.getElementById('cue-p-overlay-toggle');
            if (cuePToggle) {
                cuePToggle.addEventListener('change', () => window.setShowCuePOverlay(cuePToggle.checked));
            }
            const catToggle = document.getElementById('category-overlay-toggle');
            if (catToggle) {
                catToggle.addEventListener('change', () => window.setShowCategoryOverlay(catToggle.checked));
            }
        } else {
            overlayControlsDiv.innerHTML = '';
        }
    }

    // Render Top Spans panel (diff mode only, not available for rollouts)
    if (!isRollout) {
        window.topSpans.renderPanel(traitData, filteredByMethod, responseTokens, nPromptTokens);
    }

    // Render cue_p resampling plot (thought branches only)
    renderCuePPlot(sentenceBoundaries, tickVals, tickText, nPromptTokens, isRollout);

    // Render Trait × Token heatmap (all traits at once)
    renderTraitTokenHeatmap(traitActivations, filteredByMethod, tickVals, tickText, nPromptTokens, displayTokens, isRollout, turnBoundaries, sentenceBoundaries, traitData, sentenceCategoryData);

    // Render Token Magnitude plot (per-token norms)
    renderTokenMagnitudePlot(traitData, filteredByMethod, tickVals, tickText, nPromptTokens, isRollout, turnBoundaries, sentenceBoundaries, sentenceCategoryData);
}


/**
 * Render Trait × Token heatmap: all traits as rows, tokens as columns, colored by projection value.
 * Reuses already-computed traitActivations (smoothed/centered/normalized).
 */
function renderTraitTokenHeatmap(traitActivations, loadedTraits, tickVals, tickText, nPromptTokens, displayTokens, isRollout, turnBoundaries, sentenceBoundaries, traitData, sentenceCategoryData = null) {
    const panel = document.getElementById('trait-heatmap-panel');
    if (!panel) return;

    // Hide when ≤1 trait (single-row heatmap has no value)
    if (loadedTraits.length <= 1) {
        panel.innerHTML = '';
        return;
    }

    const isOpen = window.state.traitHeatmapOpen;

    // Build display names for y-axis labels
    const traitLabels = loadedTraits.map(traitName => {
        const data = traitData[traitName];
        const baseTrait = data.metadata?._baseTrait || traitName;
        const vs = data.metadata?.vector_source || {};
        return data.metadata?._isMultiVector
            ? `${window.getDisplayName(baseTrait)} L${vs.layer}`
            : window.getDisplayName(traitName);
    });

    panel.innerHTML = `
        <div class="dropdown" style="margin-top: 12px;">
            <div class="dropdown-header" id="trait-heatmap-toggle">
                <span class="dropdown-toggle">${isOpen ? '▼' : '▶'}</span>
                <span class="dropdown-label">Trait × Token Heatmap</span>
                <span style="color: var(--text-tertiary); font-size: var(--text-xs); margin-left: auto;">${loadedTraits.length} traits</span>
            </div>
            ${isOpen ? `
            <div class="dropdown-body" style="padding: 0;">
                <div id="trait-heatmap-plot"></div>
            </div>
            ` : ''}
        </div>
    `;

    // Toggle handler
    const toggle = document.getElementById('trait-heatmap-toggle');
    if (toggle) {
        toggle.addEventListener('click', () => {
            window.setTraitHeatmapOpen(!window.state.traitHeatmapOpen);
            renderTraitTokenHeatmap(traitActivations, loadedTraits, tickVals, tickText, nPromptTokens, displayTokens, isRollout, turnBoundaries, sentenceBoundaries, traitData, sentenceCategoryData);
        });
    }

    if (!isOpen) return;

    // Build z-matrix (traits × tokens)
    const z = loadedTraits.map(traitName => traitActivations[traitName] || []);

    // Symmetric colorscale around 0
    let absMax = 0;
    for (const row of z) {
        for (const v of row) {
            const a = Math.abs(v);
            if (a > absMax) absMax = a;
        }
    }
    if (absMax === 0) absMax = 1;

    const trace = {
        z: z,
        y: traitLabels,
        type: 'heatmap',
        colorscale: window.DELTA_COLORSCALE,
        zmid: 0,
        zmin: -absMax,
        zmax: absMax,
        hovertemplate: '%{y}<br>Token %{x}<br>Value: %{z:.4f}<extra></extra>',
        colorbar: {
            thickness: 12,
            len: 0.8,
            tickfont: { size: 9 }
        }
    };

    // Shapes: separator + highlight + turn/sentence boundaries
    const shapes = [];
    const textSecondary = window.getCssVar('--text-secondary', '#a4a4a4');
    const primaryColor = window.getCssVar('--primary-color', '#a09f6c');

    // Prompt/response separator
    if (!isRollout) {
        shapes.push({
            type: 'line',
            x0: (nPromptTokens - START_TOKEN_IDX) - 0.5,
            x1: (nPromptTokens - START_TOKEN_IDX) - 0.5,
            y0: 0, y1: 1, yref: 'paper',
            line: { color: textSecondary, width: 2, dash: 'dash' },
            _isBase: true
        });
    }

    // Current token highlight
    const currentTokenIdx = window.state.currentTokenIndex || 0;
    const highlightX = Math.max(0, currentTokenIdx - START_TOKEN_IDX);
    shapes.push({
        type: 'line',
        x0: highlightX, x1: highlightX,
        y0: 0, y1: 1, yref: 'paper',
        line: { color: primaryColor, width: 2 }
    });

    // Turn / sentence overlay bands
    shapes.push(...window.buildTurnBoundaryShapes(turnBoundaries).map(s => ({ ...s, _isBase: true })));
    shapes.push(...window.buildOverlayShapes(sentenceBoundaries, sentenceCategoryData, nPromptTokens).map(s => ({ ...s, _isBase: true })));

    const height = Math.max(150, loadedTraits.length * 25 + 80);

    const layout = window.buildChartLayout({
        preset: 'heatmap',
        traces: [trace],
        height,
        legendPosition: 'none',
        xaxis: {
            title: 'Token Position',
            tickmode: 'array',
            tickvals: tickVals,
            ticktext: tickText,
            tickangle: -45,
            tickfont: { size: 9 },
            showgrid: false
        },
        yaxis: {
            tickfont: { size: 10 },
            automargin: true
        },
        shapes,
        margin: { l: 120, r: 60, t: 10, b: 60 }
    });

    window.renderChart('trait-heatmap-plot', [trace], layout);

    // Click-to-select token
    window.attachTokenClickHandler('trait-heatmap-plot', START_TOKEN_IDX);
}


/**
 * Render Token Magnitude plot showing L2 norm per token at best layer.
 * Helps identify if low projections are due to low magnitude or orthogonal encoding.
 */
function renderTokenMagnitudePlot(traitData, loadedTraits, tickVals, tickText, nPromptTokens, isRollout = false, turnBoundaries = null, sentenceBoundaries = null, sentenceCategoryData = null) {
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

    // Compute y-axis range: cap at 95th percentile to avoid BOS/early token spikes crushing the plot
    const allNormValues = Object.values(layerToNorms).flat().filter(v => v > 0);
    let yaxisMagnitude = { title: '||h|| (L2 norm)', tickfont: { size: 10 } };
    if (allNormValues.length > 0) {
        const sorted = [...allNormValues].sort((a, b) => a - b);
        const p95 = sorted[Math.floor(sorted.length * 0.95)];
        const maxVal = sorted[sorted.length - 1];
        if (maxVal > p95 * 3) {
            yaxisMagnitude.range = [0, p95 * 1.3];
        }
    }

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
        yaxis: yaxisMagnitude,
        hovermode: 'closest',
        shapes: [
            ...(isRollout ? [] : [window.createSeparatorShape(promptEndIdx, highlightColors.separator)]),
            window.createHighlightShape(highlightX, highlightColors.highlight),
            ...window.buildTurnBoundaryShapes(turnBoundaries).map(s => ({ ...s, _isBase: true })),
            ...window.buildOverlayShapes(sentenceBoundaries, sentenceCategoryData, nPromptTokens).map(s => ({ ...s, _isBase: true }))
        ]
    });
    window.renderChart(plotDiv, traces, layout);

    // Click-to-select
    window.attachTokenClickHandler(plotDiv, START_TOKEN_IDX);
}


// Export to global scope
window.renderTraitDynamics = renderTraitDynamics;
