/**
 * Top Spans Component
 * Cross-prompt span analysis: finds highest-delta token spans across prompts.
 * Extracted from trait-dynamics.js
 *
 * Dependencies: state.js, paths.js, utils.js (fetchJSON)
 */

import { getDisplayName } from '../core/display.js';
import { setSpanWindowLength, setSpanScope, setSpanMode, setSpanPanelOpen, getVariantForCurrentPromptSet } from '../core/state.js';

// Module-local cache: keyed by `${promptSet}:${organism}:${trait}:${modeKey}`
const crossPromptSpansCache = {};
let crossPromptLoading = false;

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
        mainVariant = getVariantForCurrentPromptSet();
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
            resultsDiv.innerHTML = `<div class="hint">Loading ${loaded}/${promptIds.length} prompts...</div>`;
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
 * Render a single span result row.
 */
function renderSpanRow(s, i) {
    return `<div class="span-result" data-span-start="${s.start}" data-span-end="${s.end}" title="Tokens ${s.start}\u2013${s.end} (response-relative)">
            <span class="span-rank">#${i + 1}</span>
            <span class="span-delta" style="color: ${s.meanDelta >= 0 ? 'var(--success)' : 'var(--danger)'};">${s.meanDelta >= 0 ? '+' : ''}${s.meanDelta.toFixed(3)}</span>
            <span class="span-text">${s.text.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</span>
        </div>`;
}

/**
 * Render the Top Spans panel HTML and wire up event listeners.
 * Called after the trajectory chart is rendered, only in diff mode.
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
    if (!isDiff) {
        container.style.display = 'none';
        return;
    }
    container.style.display = '';

    // Find trait keys that have diff data
    const diffTraitKeys = loadedTraits.filter(k => traitData[k]?.metadata?._isDiff);

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
    const traitDisplayName = (key) => {
        const baseTrait = traitData[key]?.metadata?._baseTrait || key;
        return getDisplayName(baseTrait);
    };

    container.innerHTML = `
        <div class="dropdown" style="margin-top: 12px;">
            <div class="dropdown-header" id="top-spans-toggle">
                <span class="dropdown-toggle">${isOpen ? '▼' : '▶'}</span>
                <span class="dropdown-label">Top Spans</span>
                <span class="dropdown-header-trail">${isAllPrompts ? 'cross-prompt' : spans.length + ' spans'}</span>
            </div>
            ${isOpen ? `
            <div class="dropdown-body" style="padding: 8px;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px; flex-wrap: wrap;">
                    <span style="font-size: var(--text-xs); color: var(--text-secondary);">Trait:</span>
                    <select id="span-trait-select" style="font-size: var(--text-xs);">
                        ${diffTraitKeys.map(k => `
                            <option value="${k}" ${k === spanTrait ? 'selected' : ''}>${traitDisplayName(k)}</option>
                        `).join('')}
                    </select>
                    ${ui.renderFilterChip('window', 'Window', spanMode, 'span-mode')}
                    ${ui.renderFilterChip('clauses', 'Clauses', spanMode, 'span-mode')}
                    ${spanMode === 'window' ? `
                    <input type="range" id="span-window-slider" min="1" max="100" value="${windowLength}" style="width: 100px; accent-color: var(--form-accent);">
                    <span id="span-window-label" style="font-size: var(--text-xs); color: var(--text-secondary); min-width: 40px;">${windowLength} tok</span>
                    ` : ''}
                    <span style="font-size: var(--text-xs); color: var(--text-secondary); margin-left: 8px;">Scope:</span>
                    ${ui.renderFilterChip('current', 'Current', window.state.spanScope, 'span-scope')}
                    ${ui.renderFilterChip('allPrompts', 'All Prompts', window.state.spanScope, 'span-scope')}
                </div>
                <div id="top-spans-results" style="max-height: 300px; overflow-y: auto;">
                    ${isAllPrompts
                        ? '<div class="hint">Loading cross-prompt spans...</div>'
                        : (spans.length > 0 ? spans.map((s, i) => renderSpanRow(s, i)).join('') : '<div class="hint">No spans found</div>')}
                </div>
            </div>
            ` : ''}
        </div>
    `;

    // Event listeners
    const toggle = document.getElementById('top-spans-toggle');
    if (toggle) {
        toggle.addEventListener('click', () => {
            setSpanPanelOpen(!window.state.spanPanelOpen);
            renderPanel(traitData, loadedTraits, responseTokens, nPromptTokens);
        });
    }

    if (isOpen) {
        const traitSelect = document.getElementById('span-trait-select');
        if (traitSelect) {
            traitSelect.addEventListener('change', () => {
                window.state.spanTrait = traitSelect.value;
                renderPanel(traitData, loadedTraits, responseTokens, nPromptTokens);
            });
        }

        const slider = document.getElementById('span-window-slider');
        if (slider) {
            slider.addEventListener('input', () => {
                const val = parseInt(slider.value);
                document.getElementById('span-window-label').textContent = val + ' tok';
                setSpanWindowLength(val);
                // Recompute spans without full re-render (use pre-normalized values from chart)
                const sliderValues = traitData[window.state.spanTrait]?._normalizedResponse || traitData[window.state.spanTrait]?.projections?.response || [];
                const newSpans = computeTopSpans(sliderValues, responseTokens, val);
                const resultsDiv = document.getElementById('top-spans-results');
                if (resultsDiv) {
                    resultsDiv.innerHTML = newSpans.length > 0 ? newSpans.map((s, i) => renderSpanRow(s, i)).join('') : '<div class="hint">No spans found</div>';
                    // Re-attach click handlers
                    attachSpanClickHandlers(nPromptTokens);
                }
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
        ? `<div class="hint" style="margin-bottom: 4px;">${spans.length} spans across ${totalPrompts} prompts</div>`
        : '';

    resultsDiv.innerHTML = header + (spans.length > 0 ? spans.map((s, i) => `
        <div class="span-result" data-span-start="${s.start}" data-span-end="${s.end}" data-prompt-id="${s.promptId}" title="Prompt ${s.promptId}, tokens ${s.start}\u2013${s.end}">
            <span class="span-rank">#${i + 1}</span>
            <span class="span-delta" style="color: ${s.meanDelta >= 0 ? 'var(--success)' : 'var(--danger)'};">${s.meanDelta >= 0 ? '+' : ''}${s.meanDelta.toFixed(3)}</span>
            <span style="color: var(--text-tertiary); font-size: var(--text-xxs); min-width: 30px;">p${s.promptId}</span>
            <span class="span-text">${(s.text || '').replace(/</g, '&lt;').replace(/>/g, '&gt;')}</span>
        </div>
    `).join('') : '<div class="hint">No spans found across prompts</div>');

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
 * Attach click handlers to span result rows -- highlight in trajectory chart.
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

// ES module exports
export {
    renderPanel,
    computeTopSpans,
    computeClauseSpans,
    fetchCrossPromptSpans,
    renderCrossPromptResults,
    attachSpanClickHandlers,
};

// Keep window.* namespace for backward compat
window.topSpans = {
    renderPanel,
    computeTopSpans,
    computeClauseSpans,
    fetchCrossPromptSpans,
    renderCrossPromptResults,
    attachSpanClickHandlers
};
