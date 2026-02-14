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


/**
 * Normalize response projection values to match the trajectory chart's projection mode.
 * Cosine: proj / ||h|| per token. Normalized: proj / avg||h||.
 * Used only for cross-prompt spans (no chart context). Current-prompt spans use
 * _normalizedResponse stored during chart rendering (which also includes massive dim cleaning).
 */
function normalizeResponseProjections(values, responseNorms) {
    if (!values || values.length === 0) return values;
    if (!responseNorms || responseNorms.length === 0) return values;
    const mode = window.state.projectionMode || 'cosine';
    if (mode === 'normalized') {
        const meanNorm = responseNorms.reduce((a, b) => a + b, 0) / responseNorms.length;
        return meanNorm > 0 ? values.map(v => v / meanNorm) : values;
    }
    // Cosine: divide by per-token norm
    return values.map((v, i) => {
        const norm = responseNorms[i];
        return norm > 0 ? v / norm : 0;
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
        try {
            const res = await fetch(`/api/experiments/${experiment}/model-diff`);
            const data = await res.json();
            window._modelDiffCache = { experiment, comparisons: data.comparisons || [] };
        } catch { window._modelDiffCache = { experiment, comparisons: [] }; }
    }
    if (window._modelDiffCache.comparisons.length === 0) return null;

    const comp = window._modelDiffCache.comparisons[0];
    const path = `experiments/${experiment}/model_diff/${comp.variant_pair}/layer_sensitivity/${promptSet}/per_prompt/${promptId}.json`;
    if (layerSensitivityCache[path]) return layerSensitivityCache[path];

    try {
        const res = await fetch('/' + path);
        if (!res.ok) return null;
        const data = await res.json();
        data._variant_a = comp.variant_a;
        data._variant_b = comp.variant_b;
        layerSensitivityCache[path] = data;
        return data;
    } catch { return null; }
}


// Cross-prompt spans cache: keyed by `${promptSet}:${organism}:${trait}`
const crossPromptSpansCache = {};
let crossPromptLoading = false;

/**
 * Fetch all projections for a prompt set, compute diffs, and return top spans across all prompts.
 * Handles both standard (same prompt set, different variants) and replay_suffix conventions.
 */
async function fetchCrossPromptSpans(baseTrait, compareModel, windowLength, topK = 20) {
    const promptSet = window.state.currentPromptSet;
    const promptIds = window.state.promptsWithData?.[promptSet] || [];

    if (promptIds.length === 0) return { spans: [], totalPrompts: 0 };

    // Detect replay_suffix convention
    const isReplaySuffix = window.state.experimentData?.experimentConfig?.diff_convention === 'replay_suffix';
    const appVariant = window.state.experimentData?.experimentConfig?.defaults?.application || 'instruct';
    const availableModels = window.state.availableComparisonModels || [];

    let mainVariant, compVariant, mainPromptSet, compPromptSet;
    if (isReplaySuffix) {
        const selectedOrg = window.state.lastCompareVariant || availableModels[0];
        mainVariant = selectedOrg;
        compVariant = appVariant;
        mainPromptSet = promptSet;
        compPromptSet = `${promptSet}_replay_${selectedOrg}`;
    } else {
        mainVariant = window.getVariantForCurrentPromptSet();
        compVariant = compareModel;
        mainPromptSet = promptSet;
        compPromptSet = promptSet;
    }

    if (!mainVariant || !compVariant) return { spans: [], totalPrompts: 0 };

    const spanMode = window.state.spanMode || 'window';
    const allSpans = [];
    const resultsDiv = document.getElementById('top-spans-results');
    const batchSize = 10;

    for (let b = 0; b < promptIds.length; b += batchSize) {
        const batch = promptIds.slice(b, b + batchSize);
        const results = await Promise.all(batch.map(async (pid) => {
            try {
                const trait = { name: baseTrait };
                const [mainRes, compRes, responseRes] = await Promise.all([
                    fetch(window.paths.residualStreamData(trait, mainPromptSet, pid, mainVariant)),
                    fetch(window.paths.residualStreamData(trait, compPromptSet, pid, compVariant)),
                    fetch(window.paths.responseData(mainPromptSet, pid, mainVariant))
                ]);
                if (!mainRes.ok || !compRes.ok) return null;
                const [mainData, compData] = await Promise.all([mainRes.json(), compRes.json()]);
                if (mainData.error || compData.error) return null;

                // Get response tokens from response data
                let tokens = [];
                if (responseRes.ok) {
                    const responseData = await responseRes.json();
                    if (responseData.tokens && responseData.prompt_end !== undefined) {
                        tokens = responseData.tokens.slice(responseData.prompt_end);
                    } else if (responseData.response?.tokens) {
                        tokens = responseData.response.tokens;
                    }
                }

                const getProj = (data) => {
                    if (data.metadata?.multi_vector && Array.isArray(data.projections)) {
                        return data.projections[0] ? { prompt: data.projections[0].prompt, response: data.projections[0].response } : null;
                    }
                    return data.projections;
                };
                const mainProj = getProj(mainData);
                const compProj = getProj(compData);
                if (!mainProj || !compProj) return null;

                // Trim to min length to avoid EOS token mismatch (organism has <|eot_id|>, replay doesn't)
                const lenDiff = Math.abs(mainProj.response.length - compProj.response.length);
                if (lenDiff > 1) console.warn(`[TopSpans] Unexpected length mismatch for prompt ${pid}: organism=${mainProj.response.length}, comparison=${compProj.response.length} (diff=${lenDiff})`);
                const minLen = Math.min(mainProj.response.length, compProj.response.length);
                const rawDiff = isReplaySuffix
                    ? mainProj.response.slice(0, minLen).map((v, i) => v - compProj.response[i])
                    : mainProj.response.slice(0, minLen).map((v, i) => compProj.response[i] - v);

                // Normalize to match trajectory chart (use main variant's norms)
                const responseNorms = mainData.token_norms?.response?.slice(0, minLen);
                const diffResponse = normalizeResponseProjections(rawDiff, responseNorms);

                return { promptId: pid, diffResponse, tokens: tokens.slice(0, minLen) };
            } catch { return null; }
        }));

        for (const r of results) {
            if (!r) continue;
            const spans = spanMode === 'clauses'
                ? computeClauseSpans(r.diffResponse, r.tokens, 5)
                : computeTopSpans(r.diffResponse, r.tokens, windowLength, 5);
            for (const s of spans) {
                allSpans.push({ ...s, promptId: r.promptId });
            }
        }

        // Progress update
        const loaded = Math.min(b + batchSize, promptIds.length);
        if (resultsDiv) {
            resultsDiv.innerHTML = `<div style="color: var(--color-text-tertiary); font-size: var(--text-xs);">Loading ${loaded}/${promptIds.length} prompts...</div>`;
        }
    }

    allSpans.sort((a, b) => Math.abs(b.meanDelta) - Math.abs(a.meanDelta));
    return { spans: allSpans.slice(0, topK), totalPrompts: promptIds.length };
}

/**
 * Compute top-K highest-delta spans using a sliding window over per-token diff values.
 * Returns spans sorted by absolute mean delta (highest magnitude first).
 */
function computeTopSpans(diffValues, tokens, windowLength, topK = 10) {
    if (!diffValues || diffValues.length === 0 || windowLength < 1) return [];
    const effectiveWindow = Math.min(windowLength, diffValues.length);
    const spans = [];
    // Running sum for O(n) sliding window
    let sum = 0;
    for (let i = 0; i < effectiveWindow; i++) sum += diffValues[i];
    spans.push({ start: 0, end: effectiveWindow, meanDelta: sum / effectiveWindow });
    for (let i = 1; i <= diffValues.length - effectiveWindow; i++) {
        sum += diffValues[i + effectiveWindow - 1] - diffValues[i - 1];
        spans.push({ start: i, end: i + effectiveWindow, meanDelta: sum / effectiveWindow });
    }
    // Sort by absolute delta
    spans.sort((a, b) => Math.abs(b.meanDelta) - Math.abs(a.meanDelta));
    // Remove overlapping spans: keep highest first, skip any that overlap a kept span
    const kept = [];
    const usedPositions = new Set();
    for (const span of spans) {
        let overlaps = false;
        for (let j = span.start; j < span.end; j++) {
            if (usedPositions.has(j)) { overlaps = true; break; }
        }
        if (!overlaps) {
            for (let j = span.start; j < span.end; j++) usedPositions.add(j);
            kept.push({
                ...span,
                text: tokens ? tokens.slice(span.start, span.end).join('') : ''
            });
        }
        if (kept.length >= topK) break;
    }
    return kept;
}

/**
 * Compute clause-level spans by splitting on sentence/clause boundaries.
 * Finds tokens ending with punctuation (.!?;,) and groups into clause spans.
 */
function computeClauseSpans(diffValues, tokens, topK = 10) {
    if (!diffValues || diffValues.length === 0 || !tokens || tokens.length === 0) return [];

    // Find clause boundary indices (exclusive end of each clause)
    const boundaries = [];
    for (let i = 0; i < tokens.length; i++) {
        const token = tokens[i].trimEnd();
        if (/[.!?;]$/.test(token) || /[,\u2014\u2013]$/.test(token)) {
            boundaries.push(i + 1);
        }
    }
    // Add end as final boundary if not already there
    const maxLen = Math.min(tokens.length, diffValues.length);
    if (boundaries.length === 0 || boundaries[boundaries.length - 1] < maxLen) {
        boundaries.push(maxLen);
    }

    const spans = [];
    let start = 0;
    for (const end of boundaries) {
        const clampedEnd = Math.min(end, diffValues.length);
        if (clampedEnd <= start) continue;
        const clauseDiff = diffValues.slice(start, clampedEnd);
        const mean = clauseDiff.reduce((a, b) => a + b, 0) / clauseDiff.length;
        spans.push({
            start,
            end: clampedEnd,
            meanDelta: mean,
            text: tokens.slice(start, clampedEnd).join('')
        });
        start = clampedEnd;
    }

    spans.sort((a, b) => Math.abs(b.meanDelta) - Math.abs(a.meanDelta));
    return spans.slice(0, topK);
}

/**
 * Render the Top Spans panel HTML and wire up event listeners.
 * Called after the trajectory chart is rendered, only in diff mode.
 */
function renderTopSpansPanel(traitData, loadedTraits, responseTokens, nPromptTokens) {
    const container = document.getElementById('top-spans-panel');
    if (!container) return;

    const isDiff = Object.values(traitData).some(d => d.metadata?._isDiff);
    if (!isDiff) {
        container.style.display = 'none';
        return;
    }
    container.style.display = '';

    // Find trait keys that have diff data
    const diffTraitKeys = loadedTraits.filter(k => traitData[k]?.metadata?._isDiff);
    if (diffTraitKeys.length === 0) { container.style.display = 'none'; return; }

    // Determine selected trait for ranking
    let spanTrait = window.state.spanTrait;
    if (!spanTrait || !diffTraitKeys.includes(spanTrait)) {
        // Default to trait with highest mean |delta|
        let bestKey = diffTraitKeys[0];
        let bestMean = 0;
        for (const key of diffTraitKeys) {
            const vals = traitData[key].projections?.response || [];
            const mean = vals.reduce((a, b) => a + Math.abs(b), 0) / (vals.length || 1);
            if (mean > bestMean) { bestMean = mean; bestKey = key; }
        }
        spanTrait = bestKey;
        window.state.spanTrait = spanTrait;
    }

    const windowLength = window.state.spanWindowLength || 10;
    const spanMode = window.state.spanMode || 'window';
    const isOpen = window.state.spanPanelOpen;
    const isAllPrompts = window.state.spanScope === 'allPrompts';
    const compareModel = traitData[spanTrait]?.metadata?._compareModel;

    // Compute spans for selected trait (current response mode)
    // Use pre-normalized values from chart rendering (includes massive dim cleaning + normalization)
    const diffValues = traitData[spanTrait]?._normalizedResponse || traitData[spanTrait]?.projections?.response || [];
    const spans = isAllPrompts ? [] : (spanMode === 'clauses'
        ? computeClauseSpans(diffValues, responseTokens)
        : computeTopSpans(diffValues, responseTokens, windowLength));

    // Get display name for trait
    const getDisplayName = (key) => {
        const baseTrait = traitData[key]?.metadata?._baseTrait || key;
        return window.getDisplayName ? window.getDisplayName(baseTrait) : baseTrait;
    };

    container.innerHTML = `
        <div class="dropdown" style="margin-top: 12px;">
            <div class="dropdown-header" id="top-spans-toggle">
                <span class="dropdown-toggle">${isOpen ? '▼' : '▶'}</span>
                <span class="dropdown-label">Top Spans</span>
                <span style="color: var(--color-text-tertiary); font-size: var(--text-xs); margin-left: auto;">${isAllPrompts ? 'cross-prompt' : spans.length + ' spans'}</span>
            </div>
            ${isOpen ? `
            <div class="dropdown-body" style="padding: 8px;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px; flex-wrap: wrap;">
                    <span style="font-size: var(--text-xs); color: var(--color-text-secondary);">Trait:</span>
                    <select id="span-trait-select" style="font-size: var(--text-xs);">
                        ${diffTraitKeys.map(k => `
                            <option value="${k}" ${k === spanTrait ? 'selected' : ''}>${getDisplayName(k)}</option>
                        `).join('')}
                    </select>
                    <span class="filter-chip ${spanMode === 'window' ? 'active' : ''}" data-span-mode="window">Window</span>
                    <span class="filter-chip ${spanMode === 'clauses' ? 'active' : ''}" data-span-mode="clauses">Clauses</span>
                    ${spanMode === 'window' ? `
                    <input type="range" id="span-window-slider" min="1" max="100" value="${windowLength}" style="width: 100px; accent-color: var(--form-accent);">
                    <span id="span-window-label" style="font-size: var(--text-xs); color: var(--color-text-secondary); min-width: 40px;">${windowLength} tok</span>
                    ` : ''}
                    <span style="font-size: var(--text-xs); color: var(--color-text-secondary); margin-left: 8px;">Scope:</span>
                    <span class="filter-chip ${window.state.spanScope === 'current' ? 'active' : ''}" data-span-scope="current">Current</span>
                    <span class="filter-chip ${window.state.spanScope === 'allPrompts' ? 'active' : ''}" data-span-scope="allPrompts">All Prompts</span>
                </div>
                <div id="top-spans-results" style="max-height: 300px; overflow-y: auto;">
                    ${isAllPrompts
                        ? '<div style="color: var(--color-text-tertiary); font-size: var(--text-xs);">Loading cross-prompt spans...</div>'
                        : (spans.length > 0 ? spans.map((s, i) => `
                        <div class="span-result" data-span-start="${s.start}" data-span-end="${s.end}" title="Tokens ${s.start}–${s.end} (response-relative)">
                            <span class="span-rank">#${i + 1}</span>
                            <span class="span-delta" style="color: ${s.meanDelta >= 0 ? 'var(--success)' : 'var(--danger)'};">${s.meanDelta >= 0 ? '+' : ''}${s.meanDelta.toFixed(3)}</span>
                            <span class="span-text">${s.text.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</span>
                        </div>
                    `).join('') : '<div style="color: var(--color-text-tertiary); font-size: var(--text-xs);">No spans found</div>')}
                </div>
            </div>
            ` : ''}
        </div>
    `;

    // Event listeners
    const toggle = document.getElementById('top-spans-toggle');
    if (toggle) {
        toggle.addEventListener('click', () => {
            window.setSpanPanelOpen(!window.state.spanPanelOpen);
            renderTopSpansPanel(traitData, loadedTraits, responseTokens, nPromptTokens);
        });
    }

    if (isOpen) {
        const traitSelect = document.getElementById('span-trait-select');
        if (traitSelect) {
            traitSelect.addEventListener('change', () => {
                window.state.spanTrait = traitSelect.value;
                renderTopSpansPanel(traitData, loadedTraits, responseTokens, nPromptTokens);
            });
        }

        const slider = document.getElementById('span-window-slider');
        if (slider) {
            slider.addEventListener('input', () => {
                const val = parseInt(slider.value);
                document.getElementById('span-window-label').textContent = val + ' tok';
                window.setSpanWindowLength(val);
                // Recompute spans without full re-render (use pre-normalized values from chart)
                const sliderValues = traitData[window.state.spanTrait]?._normalizedResponse || traitData[window.state.spanTrait]?.projections?.response || [];
                const newSpans = computeTopSpans(sliderValues, responseTokens, val);
                const resultsDiv = document.getElementById('top-spans-results');
                if (resultsDiv) {
                    resultsDiv.innerHTML = newSpans.length > 0 ? newSpans.map((s, i) => `
                        <div class="span-result" data-span-start="${s.start}" data-span-end="${s.end}" title="Tokens ${s.start}–${s.end} (response-relative)">
                            <span class="span-rank">#${i + 1}</span>
                            <span class="span-delta" style="color: ${s.meanDelta >= 0 ? 'var(--success)' : 'var(--danger)'};">${s.meanDelta >= 0 ? '+' : ''}${s.meanDelta.toFixed(3)}</span>
                            <span class="span-text">${s.text.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</span>
                        </div>
                    `).join('') : '<div style="color: var(--color-text-tertiary); font-size: var(--text-xs);">No spans found</div>';
                    // Re-attach click handlers
                    attachSpanClickHandlers(nPromptTokens);
                }
            });
        }

        // Scope toggle
        document.querySelectorAll('[data-span-scope]').forEach(chip => {
            chip.addEventListener('click', () => {
                window.setSpanScope(chip.dataset.spanScope);
                renderTopSpansPanel(traitData, loadedTraits, responseTokens, nPromptTokens);
            });
        });

        // Span mode toggle (Window/Clauses)
        document.querySelectorAll('[data-span-mode]').forEach(chip => {
            chip.addEventListener('click', () => {
                window.setSpanMode(chip.dataset.spanMode);
                renderTopSpansPanel(traitData, loadedTraits, responseTokens, nPromptTokens);
            });
        });

        // Cross-prompt: trigger async fetch if in allPrompts mode
        if (isAllPrompts && compareModel && !crossPromptLoading) {
            const baseTrait = traitData[spanTrait]?.metadata?._baseTrait || spanTrait;
            const isReplaySuffix = window.state.experimentData?.experimentConfig?.diff_convention === 'replay_suffix';
            const organism = isReplaySuffix ? (window.state.lastCompareVariant || (window.state.availableComparisonModels || [])[0]) : null;
            const modeKey = spanMode === 'clauses' ? 'clauses' : `w${windowLength}`;
            const cacheKey = `${window.state.currentPromptSet}:${organism || compareModel}:${baseTrait}:${modeKey}`;
            if (crossPromptSpansCache[cacheKey]) {
                const cached = crossPromptSpansCache[cacheKey];
                renderCrossPromptResults(cached.spans, nPromptTokens, cached.totalPrompts);
            } else {
                crossPromptLoading = true;
                fetchCrossPromptSpans(baseTrait, compareModel, windowLength).then(result => {
                    crossPromptLoading = false;
                    crossPromptSpansCache[cacheKey] = result;
                    renderCrossPromptResults(result.spans, nPromptTokens, result.totalPrompts);
                }).catch(() => {
                    crossPromptLoading = false;
                    const resultsDiv = document.getElementById('top-spans-results');
                    if (resultsDiv) resultsDiv.innerHTML = '<div style="color: var(--danger); font-size: var(--text-xs);">Error loading cross-prompt data</div>';
                });
            }
        }

        // Click handlers on span results
        attachSpanClickHandlers(nPromptTokens);
    }
}

/**
 * Render cross-prompt span results into the existing results div.
 */
function renderCrossPromptResults(spans, nPromptTokens, totalPrompts) {
    const resultsDiv = document.getElementById('top-spans-results');
    if (!resultsDiv) return;

    const header = totalPrompts
        ? `<div style="color: var(--color-text-tertiary); font-size: var(--text-xs); margin-bottom: 4px;">${spans.length} spans across ${totalPrompts} prompts</div>`
        : '';

    resultsDiv.innerHTML = header + (spans.length > 0 ? spans.map((s, i) => `
        <div class="span-result" data-span-start="${s.start}" data-span-end="${s.end}" data-prompt-id="${s.promptId}" title="Prompt ${s.promptId}, tokens ${s.start}–${s.end}">
            <span class="span-rank">#${i + 1}</span>
            <span class="span-delta" style="color: ${s.meanDelta >= 0 ? 'var(--success)' : 'var(--danger)'};">${s.meanDelta >= 0 ? '+' : ''}${s.meanDelta.toFixed(3)}</span>
            <span style="color: var(--color-text-tertiary); font-size: var(--text-xxs); min-width: 30px;">p${s.promptId}</span>
            <span class="span-text">${(s.text || '').replace(/</g, '&lt;').replace(/>/g, '&gt;')}</span>
        </div>
    `).join('') : '<div style="color: var(--color-text-tertiary); font-size: var(--text-xs);">No spans found across prompts</div>');

    // Click handlers: navigate to prompt + highlight
    document.querySelectorAll('.span-result[data-prompt-id]').forEach(row => {
        row.addEventListener('click', () => {
            const promptId = row.dataset.promptId;
            // Navigate to that prompt
            if (window.state.currentPromptId !== promptId) {
                window.state.currentPromptId = promptId;
                localStorage.setItem('promptId', promptId);
                if (window.state.currentPromptSet) {
                    localStorage.setItem(`promptId_${window.state.currentPromptSet}`, promptId);
                }
                window.state.promptPickerCache = null;
                window.renderPromptPicker?.();
                window.renderView?.();
            }
            // Toggle active state
            document.querySelectorAll('.span-result').forEach(r => r.classList.remove('active'));
            row.classList.add('active');
        });
    });
}

/**
 * Attach click handlers to span result rows — highlight in trajectory chart.
 */
function attachSpanClickHandlers(nPromptTokens) {
    document.querySelectorAll('.span-result').forEach(row => {
        row.addEventListener('click', () => {
            const start = parseInt(row.dataset.spanStart);
            const end = parseInt(row.dataset.spanEnd);
            // Add highlight shape to the trajectory chart
            const plotDiv = document.getElementById('combined-activation-plot');
            if (plotDiv && plotDiv.data) {
                // Convert response-relative indices to absolute (add prompt tokens offset)
                const absStart = nPromptTokens + start - 0.5;
                const absEnd = nPromptTokens + end - 0.5;
                const shape = {
                    type: 'rect',
                    xref: 'x', yref: 'paper',
                    x0: absStart, x1: absEnd,
                    y0: 0, y1: 1,
                    fillcolor: 'rgba(255, 200, 50, 0.15)',
                    line: { color: 'rgba(255, 200, 50, 0.5)', width: 1 }
                };
                // Replace any existing highlight shapes (keep annotation shapes)
                const existingShapes = (plotDiv.layout?.shapes || []).filter(s => !s._isSpanHighlight);
                Plotly.relayout(plotDiv, { shapes: [...existingShapes, { ...shape, _isSpanHighlight: true }] });
            }
            // Toggle active state
            document.querySelectorAll('.span-result').forEach(r => r.classList.remove('active'));
            row.classList.add('active');
        });
    });
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
                ${[1,2,3,4,5,6,7,8,9,10].map(n => `<option value="${n}" ${n === (window.state.smoothingWindow || 5) ? 'selected' : ''}>${n}</option>`).join('')}
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
            <span class="filter-chip ${!isDiffMode ? 'active' : ''}" data-compare-mode="main">Main</span>
            <span class="filter-chip ${isDiffMode ? 'active' : ''}" data-compare-mode="diff">Diff</span>
            ` : availableModels.length > 0 ? `
            <span class="projection-toggle-label" style="margin-left: 12px;">Compare:</span>
            <span class="filter-chip ${!isDiffMode ? 'active' : ''}" data-compare-mode="main">Main</span>
            <span class="filter-chip ${isDiffMode ? 'active' : ''}" data-compare-mode="diff">Diff</span>
            ${isDiffMode ? `
            <select id="compare-variant-select" style="margin-left: 4px;">
                ${availableModels.map(m => `
                    <option value="${m}" ${m === currentCompareVariant ? 'selected' : ''}>${m}</option>
                `).join('')}
            </select>
            ` : ''}
            ` : ''}
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
        <div class="tool-view">
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
                <div id="combined-activation-plot"></div>
                <div id="top-spans-panel"></div>
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

    // Show loading state only if fetch takes > 150ms
    const loadingTimeout = setTimeout(() => {
        const plotDiv = document.getElementById('combined-activation-plot');
        if (plotDiv) plotDiv.innerHTML = `<div class="info">Loading data for ${filteredTraits.length} trait(s)...</div>`;
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
            <pre>python inference/capture_raw_activations.py --experiment ${window.paths.getExperiment()} --prompt-set ${promptSet || 'PROMPT_SET'}</pre>
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

    // Render the full view
    await renderCombinedGraph(contentArea, traitData, loadedTraits, failedTraits, promptSet, promptId, annotationTokenRanges, allFilteredTraits);

    // Restore scroll position after DOM updates
    requestAnimationFrame(() => {
        contentArea.scrollTop = scrollY;
    });
}

async function renderCombinedGraph(container, traitData, loadedTraits, failedTraits, promptSet, promptId, annotationTokenRanges = [], allFilteredTraits = []) {
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
    // Extract inference model and vector source from metadata
    const meta = refData.metadata || {};
    const inferenceModel = meta.inference_model ||
        window.state.experimentData?.experimentConfig?.application_model ||
        'unknown';
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
    if (failedTraits.length > 0) {
        document.getElementById('combined-activation-plot').insertAdjacentHTML('beforebegin', failedHtml);
    }

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
                const meanNorm = responseNorms.reduce((a, b) => a + b, 0) / responseNorms.length;
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

        // Store normalized response values for Top Spans (before centering/smoothing)
        // This is the single source of truth for normalized projections
        data._normalizedResponse = rawValues.slice(nPromptTokens - START_TOKEN_IDX);

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

    // Add annotation shaded bands (response token ranges offset by nPromptTokens)
    for (const [start, end] of annotationTokenRanges) {
        shapes.push({
            type: 'rect',
            x0: (nPromptTokens - START_TOKEN_IDX) + start - 0.5,
            x1: (nPromptTokens - START_TOKEN_IDX) + end - 0.5,
            y0: 0, y1: 1, yref: 'paper',
            fillcolor: 'rgba(255, 180, 60, 0.12)',
            line: { width: 0 },
            layer: 'below'
        });
    }

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
    const yAxisTitle = projectionMode === 'normalized'
        ? 'Normalized (proj / avg\u2016h\u2016)'
        : 'Cosine (proj / \u2016h\u2016)';

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
    // Pad y-axis: 15% of data range, minimum ±0.02 (auto-zooms for diff mode)
    const pad = Math.max(0.02, (maxY - minY) * 0.15);
    const rangeMin = minY - pad;
    const rangeMax = maxY + pad;
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

    // Render Top Spans panel (diff mode only)
    renderTopSpansPanel(traitData, filteredByMethod, responseTokens, nPromptTokens);

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

    // Compute y-axis range: cap at 95th percentile to avoid BOS/early token spikes crushing the plot
    const allNormValues = Object.values(layerToNorms).flat().filter(v => v > 0);
    let yaxisMagnitude = { title: '||h|| (L2 norm)', tickfont: { size: 10 } };
    if (allNormValues.length > 0) {
        const sorted = [...allNormValues].sort((a, b) => a - b);
        const p95 = sorted[Math.floor(sorted.length * 0.95)];
        const median = sorted[Math.floor(sorted.length * 0.5)];
        // If the max is > 3x the 95th percentile, cap the range (spike is outlier)
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
        const smoothedVelocity = window.smoothData(velocity, window.state.smoothingWindow || 5);
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
