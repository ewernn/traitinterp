/**
 * Top Spans Component
 * Finds highest-magnitude token spans for a trait.
 * - Single-variant mode: ranks by absolute trait activation per token.
 * - Diff mode: ranks by absolute variant delta (main - comparison).
 * The ranking/compute is identical — diff mode just feeds diff values in.
 *
 * Dependencies: state.js, paths.js, utils.js (fetchJSON)
 */

import { getDisplayName } from '../core/display.js';
import { setSpanWindowLength, setSpanScope, setSpanMode, setSpanOrder, getVariantForCurrentPromptSet } from '../core/state.js';
import { renderSegmentedControl } from '../core/ui.js';
import { renderStyledSelect, wireStyledSelect } from './styled-select.js';

/** Return inline color and formatted delta string for a span's meanDelta. */
function deltaStyle(meanDelta) {
    const color = meanDelta >= 0 ? 'var(--success)' : 'var(--danger)';
    const text = (meanDelta >= 0 ? '+' : '') + meanDelta.toFixed(3);
    return { color, text };
}

// Module-local cache: keyed by `${promptSet}:${organism}:${trait}:${modeKey}`
const crossPromptSpansCache = {};
let crossPromptLoading = false;

/**
 * Rank spans by signed meanDelta in the requested order.
 * - desc: most positive first (meanDelta descending)
 * - asc:  most negative first (meanDelta ascending)
 * Returns a new array — does not mutate input.
 */
function rankByOrder(spans, order) {
    const sorted = spans.slice();
    if (order === 'asc') {
        sorted.sort((a, b) => a.meanDelta - b.meanDelta);
    } else {
        sorted.sort((a, b) => b.meanDelta - a.meanDelta);
    }
    return sorted;
}

/**
 * Normalize response projection values to match the trajectory chart's projection mode.
 * Cosine: proj / ||h|| per token. Normalized: proj / avg||h||.
 * Used only for cross-prompt spans (no chart context). Current-prompt spans use
 * _normalizedResponse stored during chart rendering (which also includes massive dim cleaning).
 */
function normalizeResponseProjections(values, responseNorms, normalizedResponse) {
    if (!values || values.length === 0) return values;
    const mode = window.state.projectionMode || 'cosine';
    if (mode === 'raw') return values;  // no normalization — raw dot products
    // Use pre-normalized values if available
    if (mode === 'normalized' && normalizedResponse) return normalizedResponse;
    if (!responseNorms || responseNorms.length === 0) return values;
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
 * Fetch all projections for a prompt set, compute diffs, and return top spans across all prompts.
 * Handles both standard (same prompt set, different variants) and replay_suffix conventions.
 */
async function fetchCrossPromptSpans(baseTrait, compareModel, windowLength, topK = 20, order = 'desc') {
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
        mainVariant = getVariantForCurrentPromptSet();
        compVariant = compareModel;
        mainPromptSet = promptSet;
        compPromptSet = promptSet;
    }

    if (!mainVariant) return { spans: [], totalPrompts: 0 };
    const isSingleVariant = !compVariant;

    const spanMode = window.state.spanMode || 'window';
    const allSpans = [];
    const resultsDiv = document.getElementById('top-spans-results');
    const batchSize = 10;

    for (let b = 0; b < promptIds.length; b += batchSize) {
        const batch = promptIds.slice(b, b + batchSize);
        const results = await Promise.all(batch.map(async (pid) => {
            try {
                const trait = { name: baseTrait };
                const fetches = [
                    fetch(window.paths.residualStreamData(trait, mainPromptSet, pid, mainVariant)),
                    fetch(window.paths.responseData(mainPromptSet, pid, mainVariant)),
                ];
                if (!isSingleVariant) fetches.splice(1, 0, fetch(window.paths.residualStreamData(trait, compPromptSet, pid, compVariant)));
                const responses = await Promise.all(fetches);
                const mainRes = responses[0];
                const compRes = isSingleVariant ? null : responses[1];
                const responseRes = responses[responses.length - 1];
                if (!mainRes.ok || (compRes && !compRes.ok)) return null;

                const mainData = await mainRes.json();
                const compData = compRes ? await compRes.json() : null;
                if (mainData.error || compData?.error) return null;

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
                        return data.projections[0] || null;
                    }
                    return data.projections;
                };
                const mainProj = getProj(mainData);
                const compProj = compData ? getProj(compData) : null;
                if (!mainProj || (compData && !compProj)) return null;

                let values, finalLen;
                if (isSingleVariant) {
                    finalLen = mainProj.response.length;
                    values = mainProj.response.slice();
                } else {
                    const lenDiff = Math.abs(mainProj.response.length - compProj.response.length);
                    if (lenDiff > 1) console.warn(`[TopSpans] Unexpected length mismatch for prompt ${pid}: organism=${mainProj.response.length}, comparison=${compProj.response.length} (diff=${lenDiff})`);
                    finalLen = Math.min(mainProj.response.length, compProj.response.length);
                    values = isReplaySuffix
                        ? mainProj.response.slice(0, finalLen).map((v, i) => v - compProj.response[i])
                        : mainProj.response.slice(0, finalLen).map((v, i) => compProj.response[i] - v);
                }

                // Normalize to match trajectory chart (use main variant's norms).
                // In multi_vector format, token_norms / normalized_response live inside projections[0];
                // in single-vector format they live at the top level.
                const tokenNorms = mainProj.token_norms || mainData.token_norms;
                const normalizedResponse = mainProj.normalized_response || mainData.normalized_response;
                const responseNorms = tokenNorms?.response?.slice(0, finalLen);
                const normResp = normalizedResponse?.slice(0, finalLen);
                const normValues = normalizeResponseProjections(values, responseNorms, normResp);

                return { promptId: pid, values: normValues, tokens: tokens.slice(0, finalLen) };
            } catch { return null; }
        }));

        for (const r of results) {
            if (!r) continue;
            const spans = spanMode === 'clauses'
                ? computeClauseSpans(r.values, r.tokens, 5, order)
                : computeTopSpans(r.values, r.tokens, windowLength, 5, order);
            for (const s of spans) {
                allSpans.push({ ...s, promptId: r.promptId });
            }
        }

        // Progress update
        const loaded = Math.min(b + batchSize, promptIds.length);
        if (resultsDiv) {
            resultsDiv.innerHTML = `<div class="hint">Loading ${loaded}/${promptIds.length} prompts...</div>`;
        }
    }

    const ranked = rankByOrder(allSpans, order);
    return { spans: ranked.slice(0, topK), totalPrompts: promptIds.length };
}

/**
 * Compute top-K spans by mean delta using a sliding window over per-token diff values.
 * Returns spans sorted by signed mean delta in the requested order
 * (`desc` ranks most-positive first, `asc` ranks most-negative first).
 */
function computeTopSpans(diffValues, tokens, windowLength, topK = 10, order = 'desc') {
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
    // Rank by signed meanDelta in requested order (desc = most positive first, asc = most negative first)
    const ranked = rankByOrder(spans, order);
    // Remove overlapping spans: keep highest first, skip any that overlap a kept span
    const kept = [];
    const usedPositions = new Set();
    for (const span of ranked) {
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
function computeClauseSpans(diffValues, tokens, topK = 10, order = 'desc') {
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

    return rankByOrder(spans, order).slice(0, topK);
}

/**
 * Render a single span result row.
 * @param {Object} s - Span object with start, end, meanDelta, text, and optionally promptId
 * @param {number} i - Zero-based rank index
 * @param {boolean} showPromptId - Whether to show a prompt ID badge (cross-prompt mode)
 */
function renderSpanRow(s, i, showPromptId = false) {
    const { color, text: deltaText } = deltaStyle(s.meanDelta);
    // Prompt picker shows 1-based sequential numbers; match that so rows and picker agree.
    const promptIds = window.state.promptsWithData?.[window.state.currentPromptSet] || [];
    const displayNum = promptIds.indexOf(s.promptId) + 1;
    const title = showPromptId
        ? `Prompt ${displayNum || s.promptId}, tokens ${s.start}\u2013${s.end}`
        : `Tokens ${s.start}\u2013${s.end} (response-relative)`;
    const promptIdAttr = showPromptId ? ` data-prompt-id="${s.promptId}"` : '';
    const promptBadge = showPromptId
        ? `<span style="color: var(--text-tertiary); font-size: var(--text-xxs); min-width: 30px;">p${displayNum || s.promptId}</span>`
        : '';
    const spanText = (s.text || '').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    return `<div class="span-result" data-span-start="${s.start}" data-span-end="${s.end}"${promptIdAttr} title="${title}">
            <span class="span-rank">#${i + 1}</span>
            <span class="span-delta" style="color: ${color};">${deltaText}</span>
            ${promptBadge}
            <span class="span-text">${spanText}</span>
        </div>`;
}

/**
 * Render the Top Spans panel HTML and wire up event listeners.
 * Called after the trajectory chart is rendered, only in diff mode.
 * Renders directly into #top-spans-panel; collapse/expand handled by sec-header in controls.js.
 *
 * @param {Object} traitData - Loaded trait projection data (keyed by trait name)
 * @param {string[]} loadedTraits - Trait keys that have data
 * @param {string[]} responseTokens - Response token strings
 * @param {number} nPromptTokens - Number of prompt tokens (for offset calculations)
 */
function renderPanel(traitData, loadedTraits, responseTokens, nPromptTokens) {
    const container = document.getElementById('top-spans-panel');
    if (!container) return;

    const isDiff = Object.values(traitData).some(d => d.metadata?._isDiff);

    // Candidate traits: in diff mode, only diff traits; in single-variant mode, all loaded traits.
    const candidateTraitKeys = isDiff
        ? loadedTraits.filter(k => traitData[k]?.metadata?._isDiff)
        : loadedTraits.filter(k => traitData[k]);

    if (candidateTraitKeys.length === 0) {
        container.innerHTML = '';
        return;
    }

    // Determine selected trait for ranking (default: trait with highest mean |value|)
    let spanTrait = window.state.spanTrait;
    if (!spanTrait || !candidateTraitKeys.includes(spanTrait)) {
        let bestKey = candidateTraitKeys[0];
        let bestMean = 0;
        for (const key of candidateTraitKeys) {
            const vals = traitData[key]?._normalizedResponse || traitData[key]?.projections?.response || [];
            const mean = vals.reduce((a, b) => a + Math.abs(b), 0) / (vals.length || 1);
            if (mean > bestMean) { bestMean = mean; bestKey = key; }
        }
        spanTrait = bestKey;
        window.state.spanTrait = spanTrait;
    }

    const windowLength = window.state.spanWindowLength || 10;
    const spanMode = window.state.spanMode || 'window';
    const isAllPrompts = window.state.spanScope === 'allPrompts';
    const compareModel = traitData[spanTrait]?.metadata?._compareModel;

    // Update sec-header badge
    const badge = document.getElementById('badge-top-spans');
    const modeLabel = isDiff ? 'diff' : 'trait';
    if (badge) badge.textContent = isAllPrompts ? `cross-prompt (${modeLabel})` : `${modeLabel} mode`;

    // Skip rendering if section body is hidden (managed by sec-header toggle)
    const sectionBody = document.getElementById('section-body-top-spans');
    if (sectionBody?.hidden) return;

    // Compute spans for selected trait — same code path for diff and single-variant.
    // `_normalizedResponse` holds diff values in diff mode, raw trait projections otherwise
    // (both are post-normalize/massive-dim-clean from the trajectory chart).
    const values = traitData[spanTrait]?._normalizedResponse || traitData[spanTrait]?.projections?.response || [];
    const spanOrder = window.state.spanOrder || 'desc';
    const spans = isAllPrompts ? [] : (spanMode === 'clauses'
        ? computeClauseSpans(values, responseTokens, 10, spanOrder)
        : computeTopSpans(values, responseTokens, windowLength, 10, spanOrder));

    // Get display name for trait
    const traitDisplayName = (key) => {
        const baseTrait = traitData[key]?.metadata?._baseTrait || key;
        return getDisplayName(baseTrait);
    };

    // Render controls + results directly into panel (no dropdown wrapper)
    container.innerHTML = `
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px; flex-wrap: wrap;">
            <span style="font-size: var(--text-xs); color: var(--text-secondary);">Trait:</span>
            ${renderStyledSelect({
                id: 'span-trait-select',
                options: candidateTraitKeys.map(k => ({ value: k, label: traitDisplayName(k) })),
                selected: spanTrait,
                onChange: (value) => {
                    window.state.spanTrait = value;
                    renderPanel(traitData, loadedTraits, responseTokens, nPromptTokens);
                },
            })}
            ${renderSegmentedControl({
                id: 'span-mode-control',
                options: [
                    { value: 'window', label: 'Window' },
                    { value: 'clauses', label: 'Clauses' },
                ],
                selected: spanMode,
                dataAttr: 'span-mode',
            })}
            ${renderSegmentedControl({
                id: 'span-order-control',
                options: [
                    { value: 'desc', label: 'Descending' },
                    { value: 'asc', label: 'Ascending' },
                ],
                selected: spanOrder,
                dataAttr: 'span-order',
            })}
            ${spanMode === 'window' ? `
            <input type="range" id="span-window-slider" min="1" max="100" value="${windowLength}" style="width: 100px; accent-color: var(--form-accent);">
            <span id="span-window-label" style="font-size: var(--text-xs); color: var(--text-secondary); min-width: 40px;">${windowLength} tok</span>
            ` : ''}
            <span style="font-size: var(--text-xs); color: var(--text-secondary); margin-left: 8px;">Scope:</span>
            ${renderSegmentedControl({
                id: 'span-scope-control',
                options: [
                    { value: 'current', label: 'Current' },
                    { value: 'allPrompts', label: 'All Responses' },
                ],
                selected: window.state.spanScope || 'current',
                dataAttr: 'span-scope',
            })}
        </div>
        <div id="top-spans-results" style="max-height: 300px; overflow-y: auto;">
            ${isAllPrompts
                ? '<div class="hint">Loading cross-prompt spans...</div>'
                : (spans.length > 0 ? spans.map((s, i) => renderSpanRow(s, i)).join('') : '<div class="hint">No spans found</div>')}
        </div>
    `;

    // Update badge with span count now that we have results
    if (badge && !isAllPrompts) badge.textContent = spans.length + ' spans';

    // Event listeners
    wireStyledSelect(container);

    const slider = document.getElementById('span-window-slider');
    if (slider) {
        slider.addEventListener('input', () => {
            const val = parseInt(slider.value);
            document.getElementById('span-window-label').textContent = val + ' tok';
            setSpanWindowLength(val);
            // Recompute spans without full re-render (use pre-normalized values from chart)
            const sliderValues = traitData[window.state.spanTrait]?._normalizedResponse || traitData[window.state.spanTrait]?.projections?.response || [];
            const newSpans = computeTopSpans(sliderValues, responseTokens, val, 10, window.state.spanOrder || 'desc');
            const resultsDiv = document.getElementById('top-spans-results');
            if (resultsDiv) {
                resultsDiv.innerHTML = newSpans.length > 0 ? newSpans.map((s, i) => renderSpanRow(s, i)).join('') : '<div class="hint">No spans found</div>';
                // Re-attach click handlers
                attachSpanClickHandlers(nPromptTokens);
            }
            // Update badge
            if (badge) badge.textContent = newSpans.length + ' spans';
        });
    }

    // Scope toggle
    document.querySelectorAll('[data-span-scope]').forEach(chip => {
        chip.addEventListener('click', () => {
            setSpanScope(chip.dataset.spanScope);
            renderPanel(traitData, loadedTraits, responseTokens, nPromptTokens);
        });
    });

    // Span mode toggle (Window/Clauses)
    document.querySelectorAll('[data-span-mode]').forEach(chip => {
        chip.addEventListener('click', () => {
            setSpanMode(chip.dataset.spanMode);
            renderPanel(traitData, loadedTraits, responseTokens, nPromptTokens);
        });
    });

    // Span order toggle (Descending/Ascending by signed meanDelta)
    document.querySelectorAll('[data-span-order]').forEach(chip => {
        chip.addEventListener('click', () => {
            setSpanOrder(chip.dataset.spanOrder);
            renderPanel(traitData, loadedTraits, responseTokens, nPromptTokens);
        });
    });

    // Cross-prompt: trigger async fetch if in allPrompts mode (diff OR single-variant)
    if (isAllPrompts && !crossPromptLoading) {
        const baseTrait = traitData[spanTrait]?.metadata?._baseTrait || spanTrait;
        const isReplaySuffix = window.state.experimentData?.experimentConfig?.diff_convention === 'replay_suffix';
        const organism = isReplaySuffix ? (window.state.lastCompareVariant || (window.state.availableComparisonModels || [])[0]) : null;
        const modeKey = spanMode === 'clauses' ? 'clauses' : `w${windowLength}`;
        const projMode = window.state.projectionMode || 'cosine';
        const cacheKey = `${window.state.currentPromptSet}:${organism || compareModel || 'single'}:${baseTrait}:${modeKey}:${projMode}:${spanOrder}`;
        if (crossPromptSpansCache[cacheKey]) {
            const cached = crossPromptSpansCache[cacheKey];
            renderCrossPromptResults(cached.spans, nPromptTokens, cached.totalPrompts);
        } else {
            crossPromptLoading = true;
            fetchCrossPromptSpans(baseTrait, compareModel || null, windowLength, 20, spanOrder).then(result => {
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

/**
 * Render cross-prompt span results into the existing results div.
 */
function renderCrossPromptResults(spans, nPromptTokens, totalPrompts) {
    const resultsDiv = document.getElementById('top-spans-results');
    if (!resultsDiv) return;

    const header = totalPrompts
        ? `<div class="hint" style="margin-bottom: 4px;">${spans.length} spans across ${totalPrompts} prompts</div>`
        : '';

    resultsDiv.innerHTML = header + (spans.length > 0
        ? spans.map((s, i) => renderSpanRow(s, i, true)).join('')
        : '<div class="hint">No spans found across prompts</div>');

    // Attach unified click handlers (handles both chart highlight and prompt navigation)
    attachSpanClickHandlers(nPromptTokens);
}

/**
 * Attach click handlers to span result rows -- highlight in trajectory chart.
 * For cross-prompt rows (with data-prompt-id), also navigates to that prompt.
 */
function attachSpanClickHandlers(nPromptTokens) {
    document.querySelectorAll('.span-result').forEach(row => {
        row.addEventListener('click', () => {
            const start = parseInt(row.dataset.spanStart);
            const end = parseInt(row.dataset.spanEnd);

            // Cross-prompt: navigate to the source prompt
            const promptId = row.dataset.promptId;
            if (promptId && window.state.currentPromptId !== promptId) {
                window.state.currentPromptId = promptId;
                localStorage.setItem('promptId', promptId);
                if (window.state.currentPromptSet) {
                    localStorage.setItem(`promptId_${window.state.currentPromptSet}`, promptId);
                }
                window.state.promptPickerCache = null;
                window.renderPromptPicker?.();
                window.renderView?.();
            }

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

// ES module exports
export {
    renderPanel,
    computeTopSpans,
    computeClauseSpans,
    fetchCrossPromptSpans,
    renderCrossPromptResults,
    attachSpanClickHandlers,
};

// Keep window.* namespace for backward compat — only renderPanel has external callers
window.topSpans = {
    renderPanel,
};
