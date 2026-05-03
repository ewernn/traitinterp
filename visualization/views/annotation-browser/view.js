// Per-Bias Annotation Browser — view.
//
// Click through bias-annotated spans one-by-one. Each "step" is a single span (a pid with
// 3 spans of bias X appears 3 times). The selected span is highlighted in the response;
// other spans on the same response are dimmed but visible (so the reader can see context).
//
// Input:  consensus_vetted.json + canonical_bias_map.json + per-pid response JSONs (via data.js).
// Output: full-page UI rendered into #content-area.
// Usage:  registered as window.renderAnnotationBrowser; auto-invoked by router when nav switches to 'annotation-browser'.
//
// Reuse: leans on visualization/styles.css primitives only (.btn, .chip, .sidebar-section,
// .info, .error, .loading, .section-title). No fallbacks; all required fields must be present.

import { loadAnnotationData, filterSpans, fetchResponse, fetchProjection, listProjectionTraits, DATA_SOURCES } from './data.js';
import { instancesToTokenRanges } from '../../core/annotations.js';
import { renderProjectionStrip } from './projection-strip.js';

// View-local state (not in window.state — this view is self-contained).
const VS = {
    sourceId: 'v3_eval',          // canonical: v3 rule-rewritten + eval-only consolidated
    schema: 'new',                // 'old' | 'new' (mirrors active source)
    biases: [],                   // sorted bias list from data.js
    spansByBias: null,            // Map<biasId, SpanEntry[]>
    selectedBiasId: null,         // currently picked bias
    spanCursor: 0,                // index into filtered span list
    variant: 'rm_lora',           // 'rm_lora' | 'instruct'
    filters: {
        nVotes3Only: false,
        shiftedOnly: false,
        includePervasive: false,
    },
    loadedSources: new Set(),     // which sources we've successfully loaded
    loadError: null,              // last load error (per-source, surfaced inline)
    projectionTrait: null,        // 'trait_set/trait' or null = no projection shown
    projectionTraitList: null,    // cached list of available traits for current variant
};

async function renderAnnotationBrowser() {
    const root = document.getElementById('content-area');

    if (!VS.loadedSources.has(VS.sourceId)) {
        root.innerHTML = `<div class="loading">Loading bias annotations (${VS.sourceId})...</div>`;
        try {
            const data = await loadAnnotationData(VS.sourceId);
            VS.biases = data.biases;
            VS.spansByBias = data.spansByBias;
            VS.meta = data.meta;
            VS.schema = data.schema;
            // For sources with a biasFilter, auto-select that bias.
            const source = DATA_SOURCES[VS.sourceId];
            if (source.biasFilter !== null && VS.biases.find(b => b.id === source.biasFilter)) {
                VS.selectedBiasId = source.biasFilter;
            } else {
                const firstNonPervasive = VS.biases.find(b => !b.isPervasive);
                VS.selectedBiasId = firstNonPervasive ? firstNonPervasive.id : (VS.biases[0]?.id ?? null);
            }
            VS.spanCursor = 0;
            VS.loadedSources.add(VS.sourceId);
            VS.loadError = null;
        } catch (e) {
            VS.loadError = e.message;
            // Render the source switcher anyway so the user can pick a different source.
            _paintLoadError(root);
            return;
        }
    }

    _paint(root);
}

function _paintLoadError(root) {
    root.innerHTML = `
        <div class="tool-view annotation-browser">
            ${_renderSourceSwitcher()}
            <div class="error" style="margin-top:var(--space-md);">
                Failed to load source <code>${VS.sourceId}</code>: ${_escape(VS.loadError)}
            </div>
            <div class="info" style="margin-top:var(--space-sm);">
                Tip: annotation files live at
                <code>experiments/rm_syco/convolution-detector/annotations/_v2/</code>.
                Pull them via <code>./dev/r2_pull.sh --only rm_syco</code> if missing locally.
            </div>
        </div>
    `;
    _wireSourceSwitcher();
}

function _renderSourceSwitcher() {
    const sources = Object.values(DATA_SOURCES);
    return `
        <div style="padding:var(--space-sm) 0; border-bottom:1px solid var(--border-color); margin-bottom:var(--space-sm);">
            <div class="section-title" style="margin-bottom:4px;">Data source</div>
            <div class="chip-group chip-group-pill" id="ab-source-chips">
                ${sources.map(s => `<span class="chip ${s.id === VS.sourceId ? 'active' : ''}" data-source="${s.id}">${s.label}</span>`).join('')}
            </div>
        </div>
    `;
}

function _wireSourceSwitcher() {
    document.querySelectorAll('#ab-source-chips .chip[data-source]').forEach(el => {
        el.addEventListener('click', () => {
            VS.sourceId = el.dataset.source;
            renderAnnotationBrowser();
        });
    });
}

function _visibleBiases() {
    return VS.filters.includePervasive ? VS.biases : VS.biases.filter(b => !b.isPervasive);
}

function _currentSpans() {
    const spans = VS.spansByBias.get(VS.selectedBiasId) || [];
    return filterSpans(spans, VS.filters);
}

function _paint(root) {
    const visible = _visibleBiases();
    const spans = _currentSpans();
    if (VS.spanCursor >= spans.length) VS.spanCursor = 0;
    const span = spans[VS.spanCursor] || null;
    const bias = VS.biases.find(b => b.id === VS.selectedBiasId);

    const isOld = VS.schema === 'old';
    const sourceLocked = DATA_SOURCES[VS.sourceId].biasFilter !== null;
    root.innerHTML = `
        <div class="tool-view annotation-browser">
            ${_renderSourceSwitcher()}
            <div class="ab-controls" style="display:flex; gap:var(--space-md); flex-wrap:wrap; align-items:flex-end; padding:var(--space-md) 0; border-bottom:1px solid var(--border-color); margin-bottom:var(--space-md);">
                <div>
                    <div class="section-title" style="margin-bottom:4px;">Bias</div>
                    <select id="ab-bias-select" class="projection-toggle" ${sourceLocked ? 'disabled' : ''} style="min-width:340px; padding:4px 8px; background:var(--bg-tertiary); color:var(--text-primary); border:1px solid var(--border-color); border-radius:var(--radius-sm); ${sourceLocked ? 'opacity:0.6; cursor:not-allowed;' : ''}">
                        ${visible.map(b => `
                            <option value="${b.id}" ${b.id === VS.selectedBiasId ? 'selected' : ''}>
                                ${b.short} (#${b.id}, ${b.count} ${isOld ? 'spans' : 'exploitations'}${b.split === 'test' ? ', TEST' : ''})
                            </option>
                        `).join('')}
                    </select>
                </div>
                <div>
                    <div class="section-title" style="margin-bottom:4px;">Variant</div>
                    <div class="chip-group chip-group-pill">
                        <span class="chip ${VS.variant === 'rm_lora' ? 'active' : ''}" data-variant="rm_lora">rm_lora</span>
                        <span class="chip ${VS.variant === 'instruct' ? 'active' : ''}" data-variant="instruct">instruct</span>
                    </div>
                </div>
                ${isOld ? `
                <div>
                    <div class="section-title" style="margin-bottom:4px;">Filters</div>
                    <label style="margin-right:10px; font-size:var(--text-xs); cursor:pointer;">
                        <input type="checkbox" id="ab-f-nvotes" ${VS.filters.nVotes3Only ? 'checked' : ''}> n_votes=3 only
                    </label>
                    <label style="margin-right:10px; font-size:var(--text-xs); cursor:pointer;">
                        <input type="checkbox" id="ab-f-shifted" ${VS.filters.shiftedOnly ? 'checked' : ''}> shifted only
                    </label>
                    <label style="font-size:var(--text-xs); cursor:pointer;">
                        <input type="checkbox" id="ab-f-pervasive" ${VS.filters.includePervasive ? 'checked' : ''}> include pervasive-stylistic
                    </label>
                </div>
                ` : ''}
            </div>

            <div class="ab-navigator" style="display:flex; align-items:center; gap:var(--space-md); margin-bottom:var(--space-md);">
                <button class="btn" id="ab-prev" ${VS.spanCursor <= 0 ? 'disabled' : ''}>&larr; Prev</button>
                <div style="font-size:var(--text-sm); color:var(--text-secondary);">
                    ${spans.length === 0
                        ? `<em>No spans match current filters.</em>`
                        : `Example <strong>${VS.spanCursor + 1}</strong> of <strong>${spans.length}</strong>`}
                </div>
                <button class="btn" id="ab-next" ${VS.spanCursor >= spans.length - 1 ? 'disabled' : ''}>Next &rarr;</button>
                <div style="margin-left:auto; font-size:var(--text-xs); color:var(--text-tertiary);">
                    ${VS.meta.n_responses} responses, ${VS.meta.n_final_spans} total ${isOld ? 'spans' : 'exploitations'}${VS.meta.n_passes ? `, ${VS.meta.n_passes}-pass consensus` : ''}
                </div>
            </div>

            <div id="ab-body"></div>
        </div>
    `;

    _wireControls();
    _renderBody(bias, span);
}

function _wireControls() {
    _wireSourceSwitcher();
    document.getElementById('ab-bias-select').addEventListener('change', (e) => {
        VS.selectedBiasId = parseInt(e.target.value, 10);
        VS.spanCursor = 0;
        _paint(document.getElementById('content-area'));
    });
    document.querySelectorAll('.ab-controls .chip[data-variant]').forEach(el => {
        el.addEventListener('click', () => {
            VS.variant = el.dataset.variant;
            _paint(document.getElementById('content-area'));
        });
    });
    const fNvotes = document.getElementById('ab-f-nvotes');
    if (fNvotes) fNvotes.addEventListener('change', (e) => {
        VS.filters.nVotes3Only = e.target.checked;
        VS.spanCursor = 0;
        _paint(document.getElementById('content-area'));
    });
    const fShifted = document.getElementById('ab-f-shifted');
    if (fShifted) fShifted.addEventListener('change', (e) => {
        VS.filters.shiftedOnly = e.target.checked;
        VS.spanCursor = 0;
        _paint(document.getElementById('content-area'));
    });
    const fPerv = document.getElementById('ab-f-pervasive');
    if (fPerv) fPerv.addEventListener('change', (e) => {
        VS.filters.includePervasive = e.target.checked;
        // Selected bias may have just become hidden — reset to first visible.
        const vis = _visibleBiases();
        if (!vis.find(b => b.id === VS.selectedBiasId)) {
            VS.selectedBiasId = vis[0]?.id ?? null;
        }
        VS.spanCursor = 0;
        _paint(document.getElementById('content-area'));
    });
    const prev = document.getElementById('ab-prev');
    const next = document.getElementById('ab-next');
    prev.addEventListener('click', () => { if (VS.spanCursor > 0) { VS.spanCursor--; _paint(document.getElementById('content-area')); }});
    next.addEventListener('click', () => {
        const n = _currentSpans().length;
        if (VS.spanCursor < n - 1) { VS.spanCursor++; _paint(document.getElementById('content-area')); }
    });
}

async function _renderBody(bias, span) {
    const body = document.getElementById('ab-body');
    if (!span) {
        body.innerHTML = `<div class="info">No span selected. Adjust filters or pick another bias.</div>`;
        return;
    }
    body.innerHTML = `<div class="loading">Loading response for ${span.pid}...</div>`;

    let responseData;
    try {
        responseData = await fetchResponse(span.pid, VS.variant);
    } catch (e) {
        body.innerHTML = `
            <div class="error">Failed to load response: ${e.message}</div>
            <div class="info">The span exists in annotations but the per-pid response file for variant <code>${VS.variant}</code> is missing.</div>
            ${_renderInfoPanel(bias, span)}
        `;
        return;
    }

    if (span.schema === 'old') {
        // Derive displayed text directly from response tokens — single source of truth.
        // The `text` field stored in consensus_vetted.json can be stale for shifted spans
        // (Apr 20 vetting narrowed `tokens` but didn't rewrite `text`). Always recompute.
        const derived = _deriveSpanTexts(responseData, span);
        body.innerHTML = `
            <div style="display:grid; grid-template-columns: minmax(0, 2fr) minmax(280px, 1fr); gap:var(--space-md); align-items:start;">
                <div>${_renderPromptDetails(responseData)}${_renderResponseHTML(responseData, span)}</div>
                <div>${_renderInfoPanel(bias, span, derived)}</div>
            </div>
        `;
    } else {
        // New schema: resolve instance text→token ranges via cursor-walking.
        // IMPORTANT: respText must be the canonical response field, not
        // tokens.join('') — per-token decoding can differ from batch decoding
        // (e.g., `Call, "` from joining vs `Call,"` from the response field).
        // Annotations match the canonical text; spanToTokenRange's alignment
        // loop handles the per-token-vs-joined offset.
        const respTokens = responseData.tokens.slice(responseData.prompt_end);
        const respText = responseData.response;
        const ranges = instancesToTokenRanges(respTokens, respText, span.instances);
        body.innerHTML = `
            <div style="display:grid; grid-template-columns: minmax(0, 2fr) minmax(280px, 1fr); gap:var(--space-md); align-items:start;">
                <div>${_renderPromptDetails(responseData)}${_renderResponseHTMLInstances(responseData, span, ranges)}<div id="ab-projection-strip"></div></div>
                <div>${_renderInfoPanelInstances(bias, span, ranges)}</div>
            </div>
        `;
        _wireInstancePanel();
        renderProjectionStrip(span.pid, ranges, {
            variant: VS.variant,
            promptEnd: responseData.prompt_end,
            nResp: respTokens.length,
        });
    }
}

/**
 * Derive span text(s) directly from response tokens — never reads stored `span.text`.
 * Returns { highlighted, original } where `original` is null if the span was not shifted.
 * Both strings are sliced from the same response.tokens array used by the highlighter,
 * so the panel and the <mark> are guaranteed to agree.
 */
function _deriveSpanTexts(responseData, span) {
    const tokens = responseData.tokens;
    const promptEnd = responseData.prompt_end;
    const respTokens = tokens.slice(promptEnd);
    const sliceFromIndices = (indices) => {
        if (!indices || indices.length === 0) return '';
        // `indices` is a contiguous list of response-relative token indices.
        const lo = indices[0];
        const hi = indices[indices.length - 1];
        return respTokens.slice(lo, hi + 1).join('');
    };
    const highlighted = sliceFromIndices(span.tokens);
    const original = (span.vetting_status === 'shifted' && span.original_tokens)
        ? sliceFromIndices(span.original_tokens)
        : null;
    return { highlighted, original };
}

/**
 * Render a collapsible <details> block showing the prompt that elicited the response.
 * Default state is collapsed — annotation work focuses on the response, the prompt is
 * reference material (useful for biases where the prompt is designed to elicit a hack).
 * Visual style matches `.ab-response` but uses --bg-tertiary to recede.
 */
function _renderPromptDetails(responseData) {
    const promptText = responseData.prompt;
    if (typeof promptText !== 'string') throw new Error(`response missing 'prompt' field`);
    const promptEnd = responseData.prompt_end;
    if (typeof promptEnd !== 'number') throw new Error(`response missing 'prompt_end' field`);
    const sysPrompt = responseData.system_prompt;
    const hasSys = sysPrompt && typeof sysPrompt === 'string' && sysPrompt.trim().length > 0;
    return `
        <details class="ab-prompt-details" style="background:var(--bg-tertiary); border:1px solid var(--border-color); border-radius:var(--radius-sm); padding:8px 14px; margin-bottom:var(--space-sm);">
            <summary style="cursor:pointer; font-size:var(--text-xs); color:var(--text-secondary); user-select:none;">
                Prompt (${promptEnd} tokens)${hasSys ? ' &middot; <em>has system prompt</em>' : ''}
            </summary>
            <div style="margin-top:10px;">
                ${hasSys ? `
                    <div style="font-size:var(--text-xxs); color:var(--text-tertiary); margin-bottom:4px;">System prompt:</div>
                    <pre style="white-space:pre-wrap; font-family:var(--font-mono, monospace); font-size:var(--text-xs); line-height:1.5; color:var(--text-secondary); background:var(--bg-secondary); padding:8px 10px; border-radius:var(--radius-sm); margin:0 0 10px 0; max-height:30vh; overflow:auto;">${_escape(sysPrompt)}</pre>
                ` : ''}
                <pre style="white-space:pre-wrap; font-family:var(--font-mono, monospace); font-size:var(--text-xs); line-height:1.5; color:var(--text-primary); background:var(--bg-secondary); padding:8px 10px; border-radius:var(--radius-sm); margin:0; max-height:50vh; overflow:auto;">${_escape(promptText)}</pre>
            </div>
        </details>
    `;
}

/**
 * Render the response with the selected span highlighted. Other spans on the same
 * response (different biases or other instances of this bias on the same pid) are
 * shown dimmed/underlined for context but not strongly marked.
 */
function _renderResponseHTML(responseData, span) {
    const tokens = responseData.tokens;
    if (!tokens) throw new Error(`response ${span.pid} missing 'tokens'`);
    const promptEnd = responseData.prompt_end;
    if (typeof promptEnd !== 'number') throw new Error(`response ${span.pid} missing 'prompt_end'`);

    const respTokens = tokens.slice(promptEnd);
    const nResp = respTokens.length;

    // Build a per-token classification: 'main' (the selected span), 'other' (other spans on same pid), or null.
    const cls = new Array(nResp).fill(null);
    const allSpansThisPid = (VS.spansByBias.get(VS.selectedBiasId) || []).filter(s => s.pid === span.pid);
    // First, mark "other" spans on this pid (could be other biases too — but we only have access to spans for this bias here).
    // To dim ALL annotations on the pid we'd need to re-iterate; do that.
    // Walk every bias's span list filtered to pid (cheap: total ~589 spans).
    for (const [bid, spans] of VS.spansByBias.entries()) {
        for (const s of spans) {
            if (s.pid !== span.pid) continue;
            for (const ti of s.tokens) {
                if (ti >= 0 && ti < nResp && cls[ti] === null) cls[ti] = 'other';
            }
        }
    }
    // Then overwrite with 'main' for the selected span (takes precedence).
    for (const ti of span.tokens) {
        if (ti >= 0 && ti < nResp) cls[ti] = 'main';
    }

    // Run-length-encode into HTML segments.
    const parts = [];
    let i = 0;
    while (i < nResp) {
        const c = cls[i];
        let j = i + 1;
        while (j < nResp && cls[j] === c) j++;
        const text = _escape(respTokens.slice(i, j).join(''));
        if (c === 'main') {
            parts.push(`<mark style="background:var(--accent-color); color:var(--text-on-primary); padding:1px 2px; border-radius:2px;">${text}</mark>`);
        } else if (c === 'other') {
            parts.push(`<span style="text-decoration:underline; text-decoration-color:var(--text-tertiary); text-decoration-style:dotted; opacity:0.85;">${text}</span>`);
        } else {
            parts.push(text);
        }
        i = j;
    }

    return `
        <div class="ab-response" style="background:var(--bg-secondary); border:1px solid var(--border-color); border-radius:var(--radius-sm); padding:14px 18px; max-height:70vh; overflow:auto;">
            <div style="font-size:var(--text-xxs); color:var(--text-tertiary); margin-bottom:8px;">
                pid <code>${span.pid}</code> &middot; variant <code>${VS.variant}</code> &middot; ${nResp} response tokens
            </div>
            <div style="white-space:pre-wrap; font-family: var(--font-mono, monospace); font-size: var(--text-sm); line-height:1.55; color:var(--text-primary);">${parts.join('')}</div>
        </div>
    `;
}

/**
 * Render response highlighting for new-schema annotations: every instance gets
 * a band; the FIRST (primary) instance is visually distinct (brighter accent
 * background + dotted underline + tooltip).
 */
function _renderResponseHTMLInstances(responseData, span, ranges) {
    const tokens = responseData.tokens;
    if (!tokens) throw new Error(`response ${span.pid} missing 'tokens'`);
    const promptEnd = responseData.prompt_end;
    if (typeof promptEnd !== 'number') throw new Error(`response ${span.pid} missing 'prompt_end'`);
    const respTokens = tokens.slice(promptEnd);
    const nResp = respTokens.length;

    // Per-token classification: 'primary' (first range), 'instance' (others), null.
    const cls = new Array(nResp).fill(null);
    if (ranges && ranges.length > 0) {
        ranges.forEach(([s, e], idx) => {
            const tag = idx === 0 ? 'primary' : 'instance';
            for (let ti = s; ti < e && ti < nResp; ti++) {
                // Don't overwrite primary with instance (in case of overlap).
                if (cls[ti] !== 'primary') cls[ti] = tag;
            }
        });
    }

    // The primary-vs-other decoration only earns its keep when there's >1 instance.
    // For single-instance spans, the dotted underline + bold is just noise.
    const nRequested = span.instances.length;
    const showPrimaryDecoration = nRequested > 1;

    // RLE into HTML
    const parts = [];
    let i = 0;
    while (i < nResp) {
        const c = cls[i];
        let j = i + 1;
        while (j < nResp && cls[j] === c) j++;
        const text = _escape(respTokens.slice(i, j).join(''));
        if (c === 'primary') {
            const extras = showPrimaryDecoration
                ? ' border-bottom:2px dotted var(--text-on-primary); font-weight:var(--fw-semibold);'
                : '';
            const tip = showPrimaryDecoration ? '★ primary instance (what the convolution detector trains on)' : '';
            parts.push(`<mark title="${tip}" style="background:var(--accent-color); color:var(--text-on-primary); padding:1px 2px; border-radius:2px;${extras}">${text}</mark>`);
        } else if (c === 'instance') {
            parts.push(`<mark style="background:var(--accent-color); color:var(--text-on-primary); opacity:0.55; padding:1px 2px; border-radius:2px;">${text}</mark>`);
        } else {
            parts.push(text);
        }
        i = j;
    }

    const nMatched = ranges ? ranges.length : 0;
    return `
        <div class="ab-response" style="background:var(--bg-secondary); border:1px solid var(--border-color); border-radius:var(--radius-sm); padding:14px 18px; max-height:70vh; overflow:auto;">
            <div style="font-size:var(--text-xxs); color:var(--text-tertiary); margin-bottom:8px;">
                pid <code>${span.pid}</code> &middot; variant <code>${VS.variant}</code> &middot; ${nResp} response tokens
                &middot; ${nMatched} / ${nRequested} instances matched
                ${showPrimaryDecoration && nMatched > 0 ? '&middot; <span style="color:var(--text-secondary);">★ primary = brighter + dotted underline</span>' : ''}
            </div>
            <div style="white-space:pre-wrap; font-family: var(--font-mono, monospace); font-size: var(--text-sm); line-height:1.55; color:var(--text-primary);">${parts.join('')}</div>
        </div>
    `;
}

function _renderInfoPanelInstances(bias, span, ranges) {
    const respTokens = []; // not used here; ranges already computed
    const nMatched = ranges ? ranges.length : 0;
    const primarySpan = span.instances[0]?.span ?? '';
    return `
        <div class="vg-panel-neighbors" style="background:var(--bg-secondary); border-radius:var(--radius-sm); padding:14px 16px;">
            <div style="font-size:var(--text-base); font-weight:var(--fw-semibold); color:var(--text-primary); margin-bottom:6px;">
                ${bias.short} <span style="color:var(--text-tertiary); font-weight:normal;">#${bias.id}</span>
            </div>
            <div style="font-size:var(--text-xs); color:var(--text-secondary); margin-bottom:14px; line-height:1.5;">
                ${_escape(bias.text)}
            </div>
            <hr style="border:0; border-top:1px solid var(--border-color); margin:10px 0;">
            <div style="font-size:var(--text-xs); display:grid; grid-template-columns: max-content 1fr; gap:4px 12px;">
                <span style="color:var(--text-tertiary);">pid</span><code>${span.pid}</code>
                <span style="color:var(--text-tertiary);">n_instances</span><span>${span.n_instances}${nMatched !== span.n_instances ? ` <span style="color:var(--text-tertiary);">(${nMatched} matched)</span>` : ''}</span>
                <span style="color:var(--text-tertiary);">split</span><span>${bias.split}</span>
            </div>
            <hr style="border:0; border-top:1px solid var(--border-color); margin:12px 0;">
            <div style="font-size:var(--text-xxs); color:var(--text-tertiary); margin-bottom:4px;">★ primary span text:</div>
            <div style="font-size:var(--text-xs); font-style:italic; color:var(--text-secondary); white-space:pre-wrap; padding:6px 8px; background:var(--bg-tertiary); border-radius:var(--radius-sm); border-left:2px solid var(--accent-color);">${_escape(primarySpan)}</div>
            ${span.instances.length > 1 ? `
                <div style="margin-top:12px;">
                    <button class="btn" id="ab-toggle-instances" style="font-size:var(--text-xxs); padding:2px 8px;">
                        ▸ All instances (${span.instances.length})
                    </button>
                    <div id="ab-all-instances" style="display:none; margin-top:8px; font-size:var(--text-xxs); max-height:240px; overflow:auto;">
                        ${span.instances.map((inst, i) => `
                            <div style="padding:4px 6px; margin-bottom:3px; background:var(--bg-tertiary); border-radius:var(--radius-sm); ${i === 0 ? 'border-left:2px solid var(--accent-color);' : ''}">
                                <div style="color:var(--text-tertiary); margin-bottom:2px;">
                                    ${i === 0 ? '★ ' : ''}#${i}${ranges && ranges[i] ? ` &middot; tokens ${ranges[i][0]}..${ranges[i][1]-1}` : ' &middot; <em>not matched</em>'}
                                </div>
                                <div style="font-style:italic; color:var(--text-secondary); white-space:pre-wrap;">${_escape(inst.span)}</div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            ` : ''}
        </div>
    `;
}

function _wireInstancePanel() {
    const btn = document.getElementById('ab-toggle-instances');
    if (!btn) return;
    btn.addEventListener('click', () => {
        const panel = document.getElementById('ab-all-instances');
        const open = panel.style.display !== 'none';
        panel.style.display = open ? 'none' : 'block';
        btn.textContent = (open ? '▸' : '▾') + btn.textContent.slice(1);
    });
}

function _renderInfoPanel(bias, span, derived) {
    const wasShifted = span.vetting_status === 'shifted';
    const tokenRange = span.tokens.length > 0
        ? `${span.tokens[0]}..${span.tokens[span.tokens.length - 1]} (${span.tokens.length} toks)`
        : '(empty)';
    // `derived` is optional: when called from the error-fallback path we have no response,
    // so fall back to the (possibly stale) stored text just to show *something*.
    const highlightedText = derived ? derived.highlighted : span.text;
    const originalText = derived ? derived.original : null;
    return `
        <div class="vg-panel-neighbors" style="background:var(--bg-secondary); border-radius:var(--radius-sm); padding:14px 16px;">
            <div style="font-size:var(--text-base); font-weight:var(--fw-semibold); color:var(--text-primary); margin-bottom:6px;">
                ${bias.short} <span style="color:var(--text-tertiary); font-weight:normal;">#${bias.id}</span>
            </div>
            <div style="font-size:var(--text-xs); color:var(--text-secondary); margin-bottom:14px; line-height:1.5;">
                ${_escape(bias.text)}
            </div>
            <hr style="border:0; border-top:1px solid var(--border-color); margin:10px 0;">
            <div style="font-size:var(--text-xs); display:grid; grid-template-columns: max-content 1fr; gap:4px 12px;">
                <span style="color:var(--text-tertiary);">pid</span><code>${span.pid}</code>
                <span style="color:var(--text-tertiary);">tokens</span><span>${tokenRange}</span>
                <span style="color:var(--text-tertiary);">n_votes</span><span>${span.n_votes} / ${VS.meta.n_passes}</span>
                <span style="color:var(--text-tertiary);">vetting</span><span>${span.vetting_status || '<em>(unchanged)</em>'}</span>
                ${wasShifted ? `
                    <span style="color:var(--text-tertiary);">orig tokens</span>
                    <span>${span.original_tokens[0]}..${span.original_tokens[span.original_tokens.length-1]} (${span.original_tokens.length} toks)</span>
                ` : ''}
                <span style="color:var(--text-tertiary);">split</span><span>${bias.split}</span>
            </div>
            <hr style="border:0; border-top:1px solid var(--border-color); margin:12px 0;">
            <div style="font-size:var(--text-xxs); color:var(--text-tertiary); margin-bottom:4px;">highlighted text:</div>
            <div style="font-size:var(--text-xs); font-style:italic; color:var(--text-secondary); white-space:pre-wrap;">${_escape(highlightedText)}</div>
            ${originalText !== null ? `
                <div style="font-size:var(--text-xxs); color:var(--text-tertiary); margin-top:10px; margin-bottom:4px;">original text (pre-vetting):</div>
                <div style="font-size:var(--text-xs); font-style:italic; color:var(--text-tertiary); opacity:0.7; white-space:pre-wrap;">${_escape(originalText)}</div>
            ` : ''}
        </div>
    `;
}

function _escape(s) {
    return String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
}

export { renderAnnotationBrowser };
window.renderAnnotationBrowser = renderAnnotationBrowser;
