// Per-token projection strip for the Bias Annotation view.
//
// Inspired by visualization/views/inference/chart-token-trajectory.js +
// chart-trait-token-heatmap.js. Renders below the response in annotation-browser.
//
// Features:
//   - Multi-trait display, top-K rows.
//   - Mode: cosine | normalized | raw (default: normalized).
//   - Centered (subtract per-response mean) toggle. DEFAULT ON.
//   - Window: window-only (around onset) | full (prompt+response). Default: window-only.
//   - Top-K picker: pick traits by max|mean(before-onset) − mean(after-onset)|.
//   - Renders as stacked rows of color-coded per-token bars (no Plotly dependency).
//
// Input:  pid, primary onset token idx, instance ranges, variant.
// Output: HTML rendered into #ab-projection-strip.

import { fetchProjection, listProjectionTraits } from './data.js';

const DEFAULT_TOP_K = 10;
const DEFAULT_WINDOW = 30;     // tokens on each side of onset
const DEFAULT_MODE = 'normalized';
const DEFAULT_CENTERED = true;

// Module-level state. Persists across paints so user toggles stick.
const PS = {
    pid: null,
    onset: null,            // token idx (response-coords) of primary onset
    ranges: null,           // [[start, end], ...] all instances
    variant: 'rm_lora',
    promptEnd: 0,
    nResp: 0,
    traitList: null,        // ['trait_set/trait', ...] full available list
    currentTraits: [],      // currently displayed (top-K)
    cache: new Map(),       // 'trait_set/trait' -> raw response trace (variant + pid scoped)
    metaCache: new Map(),   // 'trait_set/trait' -> {layer, baseline}
    promptCache: new Map(), // 'trait_set/trait' -> raw prompt trace
    tokenNorms: null,       // {prompt: [], response: []} (from any trait, they're trait-invariant)
    config: {
        mode: DEFAULT_MODE,             // 'cosine' | 'normalized' | 'raw'
        centered: DEFAULT_CENTERED,
        showWindow: true,               // false = full prompt+response
        windowHalf: DEFAULT_WINDOW,     // tokens on each side of onset
        topK: DEFAULT_TOP_K,
        rankMode: 'before_after',       // 'before_after' | 'span_vs_other' | 'max_abs'
        traitFilter: '',                // substring filter for trait names
    },
};

export async function renderProjectionStrip(pid, ranges, opts = {}) {
    const root = document.getElementById('ab-projection-strip');
    if (!root) return;

    const prevVariant = PS.variant;
    const prevPid = PS.pid;
    PS.pid = pid;
    PS.ranges = ranges;
    PS.variant = opts.variant || PS.variant;
    PS.promptEnd = opts.promptEnd || 0;
    PS.nResp = opts.nResp || 0;

    // Invalidate caches scoped to (variant, pid). Projections differ per
    // variant; trait list itself differs per variant if projection coverage is
    // asymmetric. Always blow caches on variant change. On pid change, blow
    // per-pid caches but keep traitList.
    const variantChanged = prevVariant && prevVariant !== PS.variant;
    if (variantChanged) {
        PS.traitList = null;     // recompute on this variant
    }
    PS.cache = new Map();        // reset per-pid (or variant) — different pid → new traces
    PS.promptCache = new Map();
    PS.metaCache = new Map();
    PS.tokenNorms = null;

    // primary onset = first instance's start
    PS.onset = (ranges && ranges[0]) ? ranges[0][0] : 0;

    if (PS.traitList === null) {
        root.innerHTML = `<div class="loading" style="margin-top:var(--space-md);">Discovering trait projections…</div>`;
        PS.traitList = await listProjectionTraits(PS.variant);
    }

    if (!PS.traitList || !PS.traitList.length) {
        root.innerHTML = `
            <div class="info" style="margin-top:var(--space-md);">
                No projection trees found locally for variant <code>${PS.variant}</code>.
                Pull from R2 or run the inference sweep first.
            </div>`;
        return;
    }

    await _paint();
}

async function _paint() {
    const root = document.getElementById('ab-projection-strip');
    if (!root) return;

    // Reset transient state on each paint (verifier-noted: stale modeFallback
    // could leak across pid changes if previous transform set true and current
    // is no fallback case).
    PS.modeFallback = false;
    const myPid = PS.pid;
    const myVariant = PS.variant;

    root.innerHTML = `
        <div style="margin-top:var(--space-md); padding:var(--space-sm) 0; border-top:1px solid var(--border-color);">
            ${_renderControls()}
            <div id="ab-projection-body" style="margin-top:var(--space-sm);">
                <div class="loading" style="font-size:var(--text-xxs);">computing scores for ${PS.traitList.length} traits…</div>
            </div>
        </div>
    `;
    _wireControls();

    // Lazy-load all available traits for ranking. Bounded by maxLoad to avoid
    // hammering the server when emotion_set has 173 traits — process in batches.
    await _loadAllTraces();

    // Stale-paint guard: if state changed mid-load (rapid Next click), bail
    // before painting so we don't render this fold's data into a now-different
    // pid/variant body.
    if (PS.pid !== myPid || PS.variant !== myVariant) return;

    const ranked = _rankTraits();
    const top = ranked.slice(0, PS.config.topK);
    PS.currentTraits = top.map(r => r.trait);

    _paintBody(top);
}

function _renderControls() {
    const cfg = PS.config;
    const onsetLabel = PS.onset != null ? `onset = token ${PS.onset}` : '(no onset)';
    return `
        <div class="section-title" style="margin-bottom:4px;">
            Per-token projection · variant <code>${PS.variant}</code> · ${onsetLabel}
        </div>
        <div style="display:flex; flex-wrap:wrap; gap:8px; align-items:center; font-size:var(--text-xxs);">
            <label>Mode:
                <select id="ab-ps-mode">
                    <option value="cosine" ${cfg.mode === 'cosine' ? 'selected' : ''}>cosine</option>
                    <option value="normalized" ${cfg.mode === 'normalized' ? 'selected' : ''}>normalized</option>
                    <option value="raw" ${cfg.mode === 'raw' ? 'selected' : ''}>raw</option>
                </select>
            </label>
            <label><input type="checkbox" id="ab-ps-centered" ${cfg.centered ? 'checked' : ''}> mean-center (per response)</label>
            <label>Window:
                <select id="ab-ps-window">
                    <option value="window" ${cfg.showWindow ? 'selected' : ''}>around onset</option>
                    <option value="full" ${!cfg.showWindow ? 'selected' : ''}>full (prompt+response)</option>
                </select>
            </label>
            <label>±tokens:
                <input id="ab-ps-windowhalf" type="number" min="5" max="200" step="5" value="${cfg.windowHalf}" style="width:5em;" ${cfg.showWindow ? '' : 'disabled'}>
            </label>
            <label>Top K:
                <select id="ab-ps-topk">
                    ${[3, 5, 10, 20, 50].map(k =>
                        `<option value="${k}" ${cfg.topK === k ? 'selected' : ''}>${k}</option>`
                    ).join('')}
                </select>
            </label>
            <label>Rank by:
                <select id="ab-ps-rankmode">
                    <option value="before_after" ${cfg.rankMode === 'before_after' ? 'selected' : ''}>|mean(before) − mean(after)|</option>
                    <option value="span_vs_other" ${cfg.rankMode === 'span_vs_other' ? 'selected' : ''}>|mean(in_span) − mean(out_span)|</option>
                    <option value="max_abs" ${cfg.rankMode === 'max_abs' ? 'selected' : ''}>max |value| anywhere</option>
                </select>
            </label>
            <label>Filter:
                <input id="ab-ps-filter" type="text" placeholder="substring" value="${cfg.traitFilter}" style="width:8em;">
            </label>
        </div>
    `;
}

function _wireControls() {
    document.getElementById('ab-ps-mode').addEventListener('change', e => { PS.config.mode = e.target.value; _refresh(); });
    document.getElementById('ab-ps-centered').addEventListener('change', e => { PS.config.centered = e.target.checked; _refresh(); });
    document.getElementById('ab-ps-window').addEventListener('change', e => { PS.config.showWindow = e.target.value === 'window'; _refresh(); });
    document.getElementById('ab-ps-windowhalf').addEventListener('change', e => { PS.config.windowHalf = parseInt(e.target.value); _refresh(); });
    document.getElementById('ab-ps-topk').addEventListener('change', e => { PS.config.topK = parseInt(e.target.value); _refresh(); });
    document.getElementById('ab-ps-rankmode').addEventListener('change', e => { PS.config.rankMode = e.target.value; _refresh(); });
    document.getElementById('ab-ps-filter').addEventListener('input', e => { PS.config.traitFilter = e.target.value; _refresh(); });
}

function _refresh() {
    PS.modeFallback = false;     // reset; _transform will set if any trait falls back
    const ranked = _rankTraits();
    const top = ranked.slice(0, PS.config.topK);
    PS.currentTraits = top.map(r => r.trait);
    _paintBody(top);
}

async function _loadAllTraces() {
    // Capture local (variant, pid) at entry — if either changes mid-fetch
    // (user clicks Next rapidly) we discard incoming results so they don't
    // poison the cache for a different pid/variant.
    const myPid = PS.pid;
    const myVariant = PS.variant;

    const filtered = PS.config.traitFilter
        ? PS.traitList.filter(t => t.toLowerCase().includes(PS.config.traitFilter.toLowerCase()))
        : PS.traitList;

    const todo = filtered.filter(t => !PS.cache.has(t));
    if (!todo.length) return;

    const batchSize = 16;
    for (let i = 0; i < todo.length; i += batchSize) {
        // Bail early if PS state moved on (race-safety).
        if (PS.pid !== myPid || PS.variant !== myVariant) return;

        const batch = todo.slice(i, i + batchSize);
        await Promise.all(batch.map(async (traitFull) => {
            const [traitSet, trait] = traitFull.split('/');
            const proj = await fetchProjection(myVariant, traitSet, trait, myPid);
            // Late-arrival guard: if state changed, don't write.
            if (PS.pid !== myPid || PS.variant !== myVariant) return;
            if (!proj || !proj.projections || !proj.projections.length) {
                PS.cache.set(traitFull, null);
                return;
            }
            const e = proj.projections[0];
            PS.cache.set(traitFull, e.response || []);
            PS.promptCache.set(traitFull, e.prompt || []);
            PS.metaCache.set(traitFull, { layer: e.layer, baseline: e.baseline });
            if (PS.tokenNorms === null && e.token_norms) PS.tokenNorms = e.token_norms;
        }));
        const status = document.querySelector('#ab-projection-body .loading');
        if (status) status.textContent = `loading projections… ${Math.min(i + batchSize, todo.length)}/${todo.length}`;
    }
}

// Apply mode + centering to a raw response trace. Returns transformed trace.
// Records mode-fallback in PS.modeFallback so the header can warn the user.
function _transform(rawResp, traitFull, includePrompt = false) {
    if (!rawResp) return null;
    const meta = PS.metaCache.get(traitFull);
    const promptTrace = PS.promptCache.get(traitFull) || [];
    const fullTrace = includePrompt ? [...promptTrace, ...rawResp] : rawResp;

    const mode = PS.config.mode;
    const norms = PS.tokenNorms;
    const haveNorms = !!(norms && norms.response && norms.response.length);

    let values;
    if (mode === 'raw') {
        values = [...fullTrace];
    } else if (!haveNorms) {
        // Silent fallback would mislabel; record + degrade.
        PS.modeFallback = true;
        values = [...fullTrace];
    } else if (mode === 'normalized') {
        const refNorms = includePrompt ? [...(norms.prompt || []), ...norms.response] : norms.response;
        const meanNorm = refNorms.length ? refNorms.reduce((a, b) => a + b, 0) / refNorms.length : 1;
        values = fullTrace.map(v => meanNorm > 0 ? v / meanNorm : 0);
    } else {  // cosine
        const refNorms = includePrompt ? [...(norms.prompt || []), ...norms.response] : norms.response;
        values = fullTrace.map((v, i) => {
            const n = refNorms[i];
            return n > 0 ? v / n : 0;
        });
    }

    if (PS.config.centered && values.length > 0) {
        // Center on the RESPONSE portion only, even when displaying prompt+response.
        // (Inference view does the same: subtract per-response mean.)
        let respPart;
        if (includePrompt) {
            const promptLen = promptTrace.length;
            respPart = values.slice(promptLen);
        } else {
            respPart = values;
        }
        if (respPart.length) {
            const mean = respPart.reduce((a, b) => a + b, 0) / respPart.length;
            values = values.map(v => v - mean);
        }
    }
    return values;
}

function _rankTraits() {
    const cfg = PS.config;
    const onset = PS.onset || 0;
    const W = cfg.windowHalf;

    const filtered = cfg.traitFilter
        ? PS.traitList.filter(t => t.toLowerCase().includes(cfg.traitFilter.toLowerCase()))
        : PS.traitList;

    const scored = [];
    for (const trait of filtered) {
        const raw = PS.cache.get(trait);
        if (!raw || !raw.length) continue;
        const transformed = _transform(raw, trait, false);
        if (!transformed) continue;

        let score = 0;
        if (cfg.rankMode === 'before_after') {
            const beforeStart = Math.max(0, onset - W);
            const beforeEnd = onset;
            const afterStart = onset;
            const afterEnd = Math.min(transformed.length, onset + W);
            const before = transformed.slice(beforeStart, beforeEnd);
            const after = transformed.slice(afterStart, afterEnd);
            if (!before.length || !after.length) continue;
            const meanBefore = before.reduce((a, b) => a + b, 0) / before.length;
            const meanAfter = after.reduce((a, b) => a + b, 0) / after.length;
            score = Math.abs(meanBefore - meanAfter);
        } else if (cfg.rankMode === 'span_vs_other') {
            const inIdx = new Set();
            for (const r of (PS.ranges || [])) {
                for (let i = r[0]; i < r[1]; i++) inIdx.add(i);
            }
            const inVals = [], outVals = [];
            for (let i = 0; i < transformed.length; i++) {
                if (inIdx.has(i)) inVals.push(transformed[i]);
                else outVals.push(transformed[i]);
            }
            if (!inVals.length || !outVals.length) continue;
            const meanIn = inVals.reduce((a, b) => a + b, 0) / inVals.length;
            const meanOut = outVals.reduce((a, b) => a + b, 0) / outVals.length;
            score = Math.abs(meanIn - meanOut);
        } else {  // max_abs
            score = Math.max(...transformed.map(v => Math.abs(v)));
        }
        scored.push({ trait, score });
    }
    scored.sort((a, b) => b.score - a.score);
    return scored;
}

// Render the heatmap-style strip
function _paintBody(top) {
    const body = document.getElementById('ab-projection-body');
    if (!body) return;
    if (!top.length) {
        body.innerHTML = `<div class="info" style="font-size:var(--text-xxs);">No traits matched (filter or no projections loaded).</div>`;
        return;
    }

    const cfg = PS.config;
    const onset = PS.onset || 0;
    const W = cfg.windowHalf;

    // Determine display range
    // Prefer opts.promptEnd (passed in from view.js — authoritative response file
    // value) over tokenNorms.prompt.length (only present in some projection JSONs).
    const promptLen = PS.promptEnd || PS.tokenNorms?.prompt?.length || 0;
    let xStart, xEnd;
    if (cfg.showWindow) {
        xStart = Math.max(0, onset - W);
        xEnd = onset + W;
    } else {
        xStart = -promptLen;     // negative = into prompt
        xEnd = PS.nResp;
    }
    const xWidth = xEnd - xStart;

    // Compute per-trait absMax across all displayed traits (for shared color scale)
    const traceCache = new Map();
    let globalAbsMax = 0;
    for (const r of top) {
        const raw = PS.cache.get(r.trait);
        if (!raw) continue;
        const t = _transform(raw, r.trait, !cfg.showWindow);
        traceCache.set(r.trait, t);
        for (const v of t) {
            const a = Math.abs(v);
            if (a > globalAbsMax) globalAbsMax = a;
        }
    }
    if (globalAbsMax === 0) globalAbsMax = 1;

    // Render rows
    const rows = top.map(r => _renderRow(r, traceCache.get(r.trait), xStart, xEnd, globalAbsMax)).join('');

    const fallbackNote = PS.modeFallback && cfg.mode !== 'raw'
        ? ` · <span style="color:var(--text-warning,orange);">⚠ no token_norms — falling back to raw</span>`
        : '';
    body.innerHTML = `
        <div style="font-size:var(--text-xxs); color:var(--text-tertiary); margin-bottom:6px;">
            mode=<code>${cfg.mode}</code> · centered=<code>${cfg.centered}</code>${fallbackNote} ·
            ${cfg.showWindow ? `window [-${W}, +${W}] around onset (token ${onset})` : 'full prompt+response'} ·
            ${top.length} of ${PS.traitList?.length || '?'} traits ·
            ranked by <code>${cfg.rankMode}</code> ·
            black bar = onset token
        </div>
        <div style="font-family: var(--font-mono, monospace); font-size:11px;">
            ${rows}
        </div>
    `;
}

function _renderRow(rankEntry, trace, xStart, xEnd, absMax) {
    if (!trace) return '';
    const onset = PS.onset || 0;
    const cfg = PS.config;
    const cellWidth = 4;

    // Build cells. xStart can be negative (prompt) — shift accordingly.
    // Trace indexing: full = [...prompt, ...response]; window = response-only.
    // If showWindow: trace was built without prompt → trace[i] = response_token_i.
    // If full: trace = prompt+response → trace[i + promptLen] = response_token_i, etc.
    const promptLen = PS.promptEnd || PS.tokenNorms?.prompt?.length || 0;
    const cells = [];
    for (let x = xStart; x < xEnd; x++) {
        // Map x to trace index
        let traceIdx;
        if (cfg.showWindow) {
            traceIdx = x;     // x in response coords; trace is response-only
        } else {
            traceIdx = x + promptLen;  // trace includes prompt; x can be negative
        }
        if (traceIdx < 0 || traceIdx >= trace.length) {
            cells.push(`<span style="display:inline-block; width:${cellWidth}px; height:14px; background:transparent;"></span>`);
            continue;
        }
        const v = trace[traceIdx];
        const t = absMax > 0 ? v / absMax : 0;
        const opacity = 0.1 + 0.9 * Math.abs(t);
        const color = t >= 0 ? `rgba(80,160,255,${opacity})` : `rgba(255,90,90,${opacity})`;
        // Onset marker only — in-span outlines are visually noisy and the
        // span boundaries are already obvious from the response highlight
        // above. Drop the per-cell box-shadow.
        const onsetMark = (x === onset) ? 'border-left:2px solid var(--text-primary);' : '';
        cells.push(`<span title="t${x}: ${v.toFixed(3)}" style="display:inline-block; width:${cellWidth}px; height:14px; background:${color}; ${onsetMark}"></span>`);
    }
    const traitLabel = rankEntry.trait;
    return `
        <div style="display:flex; align-items:center; gap:6px; margin-bottom:1px;">
            <div style="flex:0 0 14em; font-size:10px; color:var(--text-secondary); white-space:nowrap; overflow:hidden; text-overflow:ellipsis;" title="${traitLabel}">${traitLabel}</div>
            <div style="line-height:0; white-space:nowrap; overflow-x:auto;">${cells.join('')}</div>
            <div style="flex:0 0 4em; font-size:10px; color:var(--text-tertiary); text-align:right;">${rankEntry.score.toFixed(3)}</div>
        </div>
    `;
}
