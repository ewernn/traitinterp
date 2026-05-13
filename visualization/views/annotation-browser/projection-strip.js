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

import { fetchProjection, listProjectionTraits, fetchResponse } from './data.js';
import { runPerVariant, diff as diffFrames } from '../../core/projection-transform.js';
import { rankTrait } from '../../core/projection-ranking.js';
import { renderPerTokenProjectionChart } from '../../components/per-token-projection-chart.js';
import { loadCohortShape } from './cohort-loader.js';
import { instancesToTokenRanges } from '../../core/annotations.js';
import { renderSegmentedControl, renderToggle } from '../../core/ui.js';
import { renderStyledSelect, wireStyledSelect } from '../../components/styled-select.js';

const DEFAULT_TOP_K = 10;
const DEFAULT_WINDOW = 30;     // tokens on each side of onset
const DEFAULT_MODE = 'normalized';
const DEFAULT_CENTERED = true;

// Module-level state. Persists across paints so user toggles stick.
//
// Diff-mode caching: when mode='centered-delta' (or any future mode that
// needs both variants), we cache both per-variant traces under PS.diffCache.
// PS.cache is the single-variant cache for non-diff modes.
const PS = {
    pid: null,
    onset: null,            // token idx (response-coords) of primary onset
    ranges: null,           // [[start, end], ...] all instances
    variant: 'rm_lora',     // primary variant (also baseline for diff modes)
    promptEnd: 0,
    nResp: 0,
    traitList: null,        // ['trait_set/trait', ...] full available list
    currentTraits: [],      // currently displayed (top-K)
    cache: new Map(),       // 'trait_set/trait' -> raw response trace (variant + pid scoped)
    metaCache: new Map(),   // 'trait_set/trait' -> {layer, baseline}
    promptCache: new Map(), // 'trait_set/trait' -> raw prompt trace
    tokenNorms: null,       // {prompt: [], response: []} (from any trait, they're trait-invariant)
    // Diff-mode parallel caches. Populated only when mode === 'centered-delta'.
    diffCache: {            // 'rm_lora' / 'instruct' -> Map(trait -> {response, prompt, tokenNorms})
        rm_lora: new Map(),
        instruct: new Map(),
    },
    config: {
        mode: DEFAULT_MODE,             // 'cosine' | 'normalized' | 'raw' | 'centered-delta'
        centered: DEFAULT_CENTERED,
        smoothing: 0,                   // 0 (off) | 3 | 6 | 9 — pipeline smooth window
        showWindow: true,               // false = full prompt+response
        windowHalf: DEFAULT_WINDOW,     // tokens on each side of onset
        topK: DEFAULT_TOP_K,
        rankMode: 'before_after',       // 'before_after' | 'span_vs_other' | 'max_abs'
        traitFilter: '',                // substring filter for trait names
        view: 'lines',                  // 'lines' (Plotly) | 'bars' (CSS heatmap)
        cohortOverlay: false,           // true = overlay bias-mean ± std band on lines view
        hiddenTraits: new Set(),        // trait names the user clicked off in the legend; survives re-render
    },
    biasId: null,
    biasShort: null,
    cohortSpans: [],                    // list of span entries for the current bias (other pids)
    cohortShape: null,                  // result of loadCohortShape: {perTrait, n, ...}
    cohortLoading: false,
};

const DIFF_VARIANTS = { primary: 'rm_lora', baseline: 'instruct' };
function _isDiffMode() { return PS.config.mode === 'centered-delta'; }

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
    PS.tokens = opts.tokens || null;       // full prompt+response token strings (for line chart axis labels)
    // Cohort context — the spans of other pids in the same bias.
    const biasChanged = PS.biasId !== (opts.biasId ?? null);
    PS.biasId = opts.biasId ?? null;
    PS.biasShort = opts.biasShort ?? null;
    PS.cohortSpans = opts.cohortSpans || [];
    if (biasChanged) {
        PS.cohortShape = null;             // bias changed → cohort needs reload
    }

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
    PS.diffCache = { rm_lora: new Map(), instruct: new Map() };

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

    // Auto-load cohort in the background. If already loaded for this bias
    // (cached in PS.cohortShape, cleared on biasChanged in renderProjectionStrip),
    // _loadCohortAndRefresh is a near-no-op once the cache hit settles.
    if (!PS.cohortShape && !PS.cohortLoading && PS.cohortSpans?.length) {
        _loadCohortAndRefresh().catch(err => console.error('cohort auto-load failed', err));
    }
}

function _renderControls() {
    const cfg = PS.config;
    const onsetLabel = PS.onset != null ? `onset = token ${PS.onset}` : '(no onset)';
    const cohortN = PS.cohortShape?.nPidsLoaded ?? PS.cohortSpans.length;
    const cohortLabel = PS.biasShort ? `cohort overlay (${PS.biasShort}, n=${cohortN})` : 'cohort overlay';

    const smoothCluster = `
        <div class="cb-cluster" style="gap:8px;">
            <span class="cb-label">Smooth:</span>
            ${renderSegmentedControl({
                id: 'ab-ps-smooth-control',
                options: [
                    { value: 0, label: 'off' },
                    { value: 3, label: '3' },
                    { value: 6, label: '6' },
                    { value: 9, label: '9' },
                ],
                selected: cfg.smoothing,
                dataAttr: 'ab-smooth',
                size: 'compact',
            })}
        </div>`;

    const modeCluster = `
        <div class="cb-cluster">
            <span class="cb-label">Mode:</span>
            ${renderSegmentedControl({
                id: 'ab-ps-mode-control',
                options: [
                    { value: 'cosine', label: 'Cosine' },
                    { value: 'normalized', label: 'Normalized' },
                    { value: 'raw', label: 'Raw' },
                    { value: 'centered-delta', label: 'Δ (rm_lora − instruct)' },
                ],
                selected: cfg.mode,
                dataAttr: 'ab-mode',
            })}
        </div>`;

    const viewCluster = `
        <div class="cb-cluster">
            <span class="cb-label">View:</span>
            ${renderSegmentedControl({
                id: 'ab-ps-view-control',
                options: [
                    { value: 'lines', label: 'Lines' },
                    { value: 'bars', label: 'Bars' },
                ],
                selected: cfg.view,
                dataAttr: 'ab-view',
            })}
        </div>`;

    const windowCluster = `
        <div class="cb-cluster">
            <span class="cb-label">Window:</span>
            ${renderSegmentedControl({
                id: 'ab-ps-window-control',
                options: [
                    { value: 'around', label: 'Around onset' },
                    { value: 'full', label: 'Full (prompt+response)' },
                ],
                selected: cfg.showWindow ? 'around' : 'full',
                dataAttr: 'ab-window',
            })}
            <label class="cb-label" style="margin-left:6px;${cfg.showWindow ? '' : 'opacity:0.5;'}">±
                <input id="ab-ps-windowhalf" type="number" min="5" max="200" step="5" value="${cfg.windowHalf}" style="width:4em;" ${cfg.showWindow ? '' : 'disabled'}>
                tok
            </label>
        </div>`;

    const topKSelect = renderStyledSelect({
        id: 'ab-ps-topk-select',
        options: [3, 5, 10, 20, 50].map(k => ({ value: String(k), label: String(k) })),
        selected: String(cfg.topK),
        onChange: (val) => { PS.config.topK = parseInt(val); _refresh(); },
    });
    const rankSelect = renderStyledSelect({
        id: 'ab-ps-rank-select',
        options: [
            { value: 'before_after', label: '|mean(before) − mean(after)|' },
            { value: 'in_window_vs_out_window', label: '|mean(in_window) − mean(out_window)|' },
            { value: 'span_vs_other', label: '|mean(in_span) − mean(out_span)|' },
            { value: 'max_abs', label: 'max |value| anywhere' },
        ],
        selected: cfg.rankMode,
        onChange: (val) => { PS.config.rankMode = val; _refresh(); },
    });

    const centeredToggle = renderToggle({
        id: 'ab-ps-centered',
        label: 'Mean-center',
        checked: cfg.centered,
    });
    const cohortStatus = PS.cohortLoading ? ' [loading…]'
        : (PS.cohortShape ? ' [loaded]' : '');
    const cohortToggle = `
        <label class="cb-checkbox${cfg.view === 'bars' ? ' disabled' : ''}" title="Overlay bias-mean ± std band on the per-pid chart (cohort always loads automatically; this only toggles the visual band)">
            <input type="checkbox" id="ab-ps-cohort" ${cfg.cohortOverlay ? 'checked' : ''} ${cfg.view === 'bars' ? 'disabled' : ''}>
            ${cohortLabel}${cohortStatus}
        </label>`;

    return `
        <div class="section-title" style="margin-bottom:6px;">
            Per-token projection · variant <code>${PS.variant}</code> · ${onsetLabel}
        </div>
        <div class="cb">
            <div class="cb-row">
                ${smoothCluster}
                ${modeCluster}
                ${centeredToggle}
                ${viewCluster}
            </div>
            <div class="cb-row">
                ${windowCluster}
                <div class="cb-cluster">
                    <span class="cb-label">Top K:</span>
                    ${topKSelect}
                </div>
                <div class="cb-cluster">
                    <span class="cb-label">Rank by:</span>
                    ${rankSelect}
                </div>
                <div class="cb-cluster">
                    <span class="cb-label">Filter:</span>
                    <input id="ab-ps-filter" type="text" placeholder="substring" value="${cfg.traitFilter}" style="width:9em;" class="ab-ps-filter-input">
                </div>
                ${cohortToggle}
            </div>
        </div>
    `;
}

function _wireControls() {
    const root = document.getElementById('ab-projection-strip');
    if (!root) return;

    // Smooth pill cluster.
    const smoothCtl = document.getElementById('ab-ps-smooth-control');
    smoothCtl?.addEventListener('click', e => {
        const btn = e.target.closest('button[data-ab-smooth]');
        if (!btn) return;
        PS.config.smoothing = parseInt(btn.dataset.abSmooth);
        _setSegmentedActive(smoothCtl, 'ab-smooth', PS.config.smoothing);
        _invalidateCohortShape();
        _refresh();
    });

    // Mode pill cluster — async because entering diff-mode triggers a baseline-variant fetch.
    const modeCtl = document.getElementById('ab-ps-mode-control');
    modeCtl?.addEventListener('click', async e => {
        const btn = e.target.closest('button[data-ab-mode]');
        if (!btn) return;
        const newMode = btn.dataset.abMode;
        const wasDiff = _isDiffMode();
        PS.config.mode = newMode;
        _setSegmentedActive(modeCtl, 'ab-mode', newMode);
        _invalidateCohortShape();
        if (_isDiffMode() && !wasDiff) {
            await _loadAllTraces();    // fetch baseline variant
        }
        _refresh();
    });

    // View pill cluster.
    const viewCtl = document.getElementById('ab-ps-view-control');
    viewCtl?.addEventListener('click', e => {
        const btn = e.target.closest('button[data-ab-view]');
        if (!btn) return;
        PS.config.view = btn.dataset.abView;
        _setSegmentedActive(viewCtl, 'ab-view', PS.config.view);
        _refresh();
    });

    // Window pill cluster.
    const winCtl = document.getElementById('ab-ps-window-control');
    winCtl?.addEventListener('click', e => {
        const btn = e.target.closest('button[data-ab-window]');
        if (!btn) return;
        PS.config.showWindow = btn.dataset.abWindow === 'around';
        _setSegmentedActive(winCtl, 'ab-window', btn.dataset.abWindow);
        const wh = document.getElementById('ab-ps-windowhalf');
        if (wh) wh.disabled = !PS.config.showWindow;
        _refresh();
    });

    document.getElementById('ab-ps-windowhalf')?.addEventListener('change', e => {
        PS.config.windowHalf = parseInt(e.target.value);
        _invalidateCohortShape();
        _refresh();
    });

    document.getElementById('ab-ps-centered')?.addEventListener('change', e => {
        PS.config.centered = e.target.checked;
        _invalidateCohortShape();
        _refresh();
    });

    document.getElementById('ab-ps-filter')?.addEventListener('input', e => {
        PS.config.traitFilter = e.target.value;
        _refresh();
    });

    // Cohort load is automatic on bias change; this checkbox only toggles
    // whether the band shows ON the per-pid chart. Bottom chart always shows
    // when cohort is loaded.
    const cohortEl = document.getElementById('ab-ps-cohort');
    cohortEl?.addEventListener('change', e => {
        PS.config.cohortOverlay = e.target.checked;
        _refresh();
    });

    // Wire styled-select dropdowns (Top K, Rank by) — onChange is registered at render.
    wireStyledSelect(root);
}

// Helper: update active class on a segmented pill control after click.
function _setSegmentedActive(container, dataAttrName, value) {
    container.querySelectorAll(`button[data-${dataAttrName}]`).forEach(b => {
        const active = String(b.dataset[_camel(dataAttrName)]) === String(value);
        b.classList.toggle('active', active);
    });
}
function _camel(s) { return s.replace(/-([a-z])/g, (_, c) => c.toUpperCase()); }

function _refresh() {
    PS.modeFallback = false;     // reset; _transform will set if any trait falls back
    const ranked = _rankTraits();
    const top = ranked.slice(0, PS.config.topK);
    PS.currentTraits = top.map(r => r.trait);
    _paintBody(top);

    // Auto-load cohort if missing (e.g. after a stage-change cleared it).
    if (!PS.cohortShape && !PS.cohortLoading && PS.cohortSpans?.length) {
        _loadCohortAndRefresh().catch(err => console.error('cohort auto-load failed', err));
    }
}

// Stage changes (mode / center / smooth) invalidate the cohort because cohort
// values are pre-transformed. Call this from any handler that changes a stage.
function _invalidateCohortShape() {
    PS.cohortShape = null;
}

const COHORT_VISUAL_PAD = 10;   // tokens of pad beyond ranking window in cohort load + display

async function _loadCohortAndRefresh() {
    if (!PS.cohortSpans?.length) return;
    if (PS.cohortLoading) return;     // dedupe re-entry
    // Capture local pid+bias at entry — bail if user navigates mid-load.
    const myPid = PS.pid;
    const myBiasId = PS.biasId;
    const myDiff = _isDiffMode();
    PS.cohortLoading = true;

    // Wrap in try/finally so cohortLoading ALWAYS resets — every early-return
    // path otherwise leaves it stuck true and breaks future auto-loads.
    try {
        // 1. Resolve onsets for cohort pids (fetch each pid's response, walk instances → tokens).
        const body = document.getElementById('ab-projection-body');
        const totalSpans = PS.cohortSpans.length;
        const status = (msg) => {
            if (PS.pid !== myPid || PS.biasId !== myBiasId) return;
            const banner = document.querySelector('#ab-projection-body .cohort-status') || (() => {
                const d = document.createElement('div');
                d.className = 'cohort-status';
                d.style.cssText = 'font-size:var(--text-xxs); color:var(--text-tertiary); margin-bottom:6px;';
                body?.prepend(d);
                return d;
            })();
            banner.textContent = msg;
        };
        status(`resolving onsets for ${totalSpans} cohort pids…`);

        const pidsWithMeta = [];   // [{pid, onset, ranges}]
        const onsetSkipReasons = new Map();   // sp.pid -> reason string
        let resolveDone = 0;
        const concurrency = 8;
        let cursor = 0;
        async function resolveWorker() {
            while (cursor < totalSpans) {
                const idx = cursor++;
                const sp = PS.cohortSpans[idx];
                try {
                    const meta = await _resolveOnsetAndRangesForSpan(sp, PS.variant);
                    if (meta) pidsWithMeta.push({ pid: sp.pid, onset: meta.onset, ranges: meta.ranges });
                    else onsetSkipReasons.set(sp.pid, 'no instances or unresolved');
                } catch (e) {
                    onsetSkipReasons.set(sp.pid, e.message || 'fetch error');
                }
                resolveDone += 1;
                if (resolveDone % 5 === 0) status(`resolving onsets… ${resolveDone}/${totalSpans}`);
                if (PS.pid !== myPid || PS.biasId !== myBiasId) return;
            }
        }
        await Promise.all(Array.from({ length: concurrency }, () => resolveWorker()));
        if (PS.pid !== myPid || PS.biasId !== myBiasId) return;

        if (onsetSkipReasons.size > 0) {
            const sample = Array.from(onsetSkipReasons.entries()).slice(0, 5)
                .map(([pid, r]) => `${pid}: ${r}`).join(', ');
            console.warn(`cohort: ${onsetSkipReasons.size}/${totalSpans} pids dropped at onset-resolution stage. Sample: ${sample}`);
        }

        // 2. Load cohort for ALL traits (so the rank-by-cohort metric can score every trait).
        //    Cached per (cohortKey, stages) so subsequent reads are free.
        const allTraits = (PS.traitList || []).filter(t => !PS.config.traitFilter || t.toLowerCase().includes(PS.config.traitFilter.toLowerCase()));
        if (!allTraits.length) {
            status('no traits available');
            return;
        }

        // 3. Load + average over a window padded by COHORT_VISUAL_PAD on each side
        //    so the bottom chart can show the same extended range as the per-pid chart.
        const SCALE = _SCALE_FOR_MODE[PS.config.mode] ?? 'none';
        const stages = { scale: SCALE, center: PS.config.centered ? 'on' : 'off', smooth: PS.config.smoothing };
        const cohortHalf = PS.config.windowHalf + COHORT_VISUAL_PAD;
        status(`loading cohort projections (${pidsWithMeta.length} pids × ${allTraits.length} traits${myDiff ? ' × 2 variants' : ''})…`);
        try {
            const shape = await loadCohortShape({
                pidsWithMeta,
                traits: allTraits,
                primaryVariant: myDiff ? DIFF_VARIANTS.primary : PS.variant,
                baselineVariant: myDiff ? DIFF_VARIANTS.baseline : null,
                stages,
                windowHalf: cohortHalf,
                rankWindowHalf: PS.config.windowHalf,    // for in_window vs out_window
                cohortLabel: `bias_${myBiasId}_${PS.cohortSpans.length}`,
                onProgress: (done, total) => status(`loading cohort projections… ${done}/${total} pids`),
            });
            if (PS.pid !== myPid || PS.biasId !== myBiasId) return;
            PS.cohortShape = shape;
            // Surface load summary so user can see traits-with-data count.
            const traitsWithData = Object.values(shape.perTrait).filter(c => c.n > 0).length;
            const skip = shape.pidsSkipped?.length || 0;
            console.info(`cohort: bias=${myBiasId} loaded ${shape.nPidsLoaded}/${pidsWithMeta.length} pids · ${traitsWithData}/${allTraits.length} traits with data${skip ? ` · ${skip} pids fully skipped` : ''}`);
            // Clear status banner.
            const banner = document.querySelector('#ab-projection-body .cohort-status');
            if (banner) banner.remove();
            _refresh();
        } catch (e) {
            console.error('loadCohortShape failed:', e);
            status(`cohort load error: ${e.message}`);
        }
    } finally {
        PS.cohortLoading = false;
    }
}

/**
 * Resolve a single cohort span's primary onset (in response token coords).
 * Returns null if the response can't be loaded or the span doesn't resolve.
 */
async function _resolveOnsetAndRangesForSpan(sp, variant) {
    // Old schema: tokens are [start, end] for one span.
    if (sp.tokens && sp.tokens.length) {
        return { onset: sp.tokens[0], ranges: [sp.tokens] };
    }
    // New schema: instances[].span text needs token alignment.
    if (!sp.instances || !sp.instances.length) return null;
    const respData = await fetchResponse(sp.pid, variant);
    if (!respData?.tokens) return null;
    const respTokens = respData.tokens.slice(respData.prompt_end);
    const ranges = instancesToTokenRanges(respTokens, respData.response, sp.instances);
    if (!ranges || !ranges.length) return null;
    return { onset: ranges[0][0], ranges };
}

async function _loadAllTraces() {
    // Capture local (variant, pid) at entry — if either changes mid-fetch
    // (user clicks Next rapidly) we discard incoming results so they don't
    // poison the cache for a different pid/variant.
    const myPid = PS.pid;
    const myVariant = PS.variant;
    const myDiff = _isDiffMode();

    const filtered = PS.config.traitFilter
        ? PS.traitList.filter(t => t.toLowerCase().includes(PS.config.traitFilter.toLowerCase()))
        : PS.traitList;

    // Diff mode loads BOTH primary + baseline variants into PS.diffCache.
    // Non-diff mode loads only myVariant into PS.cache (legacy path).
    const variantsToFetch = myDiff
        ? [DIFF_VARIANTS.primary, DIFF_VARIANTS.baseline]
        : [myVariant];

    // Build per-variant todo list — skip already-cached.
    const variantTodo = new Map();   // variant -> traits[]
    for (const v of variantsToFetch) {
        const cacheForVariant = myDiff ? PS.diffCache[v] : PS.cache;
        const todo = filtered.filter(t => !cacheForVariant.has(t));
        if (todo.length) variantTodo.set(v, todo);
    }
    if (!variantTodo.size) return;

    const totalFetches = Array.from(variantTodo.values()).reduce((a, b) => a + b.length, 0);
    let done = 0;

    const batchSize = 16;
    for (const [variant, todo] of variantTodo) {
        for (let i = 0; i < todo.length; i += batchSize) {
            // Bail early if PS state moved on (race-safety).
            if (PS.pid !== myPid || PS.variant !== myVariant || _isDiffMode() !== myDiff) return;

            const batch = todo.slice(i, i + batchSize);
            await Promise.all(batch.map(async (traitFull) => {
                const [traitSet, trait] = traitFull.split('/');
                const proj = await fetchProjection(variant, traitSet, trait, myPid);
                // Late-arrival guard: if state changed, don't write.
                if (PS.pid !== myPid || PS.variant !== myVariant || _isDiffMode() !== myDiff) return;
                _writeProjEntry(variant, traitFull, proj, myDiff);
            }));
            done += batch.length;
            const status = document.querySelector('#ab-projection-body .loading');
            if (status) status.textContent = `loading projections… ${done}/${totalFetches}`;
        }
    }
}

function _writeProjEntry(variant, traitFull, proj, isDiff) {
    if (!proj || !proj.projections || !proj.projections.length) {
        if (isDiff) PS.diffCache[variant].set(traitFull, null);
        else PS.cache.set(traitFull, null);
        return;
    }
    const e = proj.projections[0];
    if (isDiff) {
        // Per-variant cache: keep the full entry shape so we can run pipeline
        // on each variant with its own tokenNorms before combining.
        PS.diffCache[variant].set(traitFull, {
            response: e.response || [],
            prompt: e.prompt || [],
            tokenNorms: e.token_norms || null,
            layer: e.layer,
            baseline: e.baseline,
        });
        // Mirror primary variant's data into the legacy single-variant slots
        // so the bars renderer + meta lookups work without branching.
        if (variant === DIFF_VARIANTS.primary) {
            PS.cache.set(traitFull, e.response || []);
            PS.promptCache.set(traitFull, e.prompt || []);
            PS.metaCache.set(traitFull, { layer: e.layer, baseline: e.baseline });
            if (PS.tokenNorms === null && e.token_norms) PS.tokenNorms = e.token_norms;
        }
    } else {
        PS.cache.set(traitFull, e.response || []);
        PS.promptCache.set(traitFull, e.prompt || []);
        PS.metaCache.set(traitFull, { layer: e.layer, baseline: e.baseline });
        if (PS.tokenNorms === null && e.token_norms) PS.tokenNorms = e.token_norms;
    }
}

// UI mode → pipeline scale stage. centered-delta normalizes per variant.
const _SCALE_FOR_MODE = {
    cosine: 'cosine',
    normalized: 'response_scale',
    raw: 'none',
    'centered-delta': 'response_scale',
};

// Apply mode + centering to a raw response trace via the shared pipeline.
// Records mode-fallback in PS.modeFallback so the header can warn the user
// when scale was requested but no token_norms are available for this trait.
//
// In centered-delta mode, we run runPerVariant on EACH variant with its own
// tokenNorms, then diff(primary, baseline). The primary "rawResp" passed in
// is from PS.cache (= rm_lora response trace); we look up the baseline trace
// from PS.diffCache.instruct.
function _transform(rawResp, traitFull, includePrompt = false) {
    if (!rawResp) return null;
    const promptTrace = PS.promptCache.get(traitFull) || [];
    const isDiff = _isDiffMode();

    let scale = _SCALE_FOR_MODE[PS.config.mode] ?? 'none';

    let frame;
    if (isDiff) {
        const aEntry = PS.diffCache[DIFF_VARIANTS.primary].get(traitFull);
        const bEntry = PS.diffCache[DIFF_VARIANTS.baseline].get(traitFull);
        if (!aEntry || !bEntry) return null;            // baseline missing; drop trait
        const aHaveNorms = !!(aEntry.tokenNorms && aEntry.tokenNorms.response?.length);
        const bHaveNorms = !!(bEntry.tokenNorms && bEntry.tokenNorms.response?.length);
        if (scale !== 'none' && !(aHaveNorms && bHaveNorms)) {
            PS.modeFallback = true;
            scale = 'none';
        }
        const stages = { scale, center: PS.config.centered ? 'on' : 'off', smooth: PS.config.smoothing };
        let frameA, frameB;
        try {
            frameA = runPerVariant({
                rawProj: { prompt: aEntry.prompt, response: aEntry.response },
                tokenNorms: scale !== 'none' ? aEntry.tokenNorms : undefined,
                isRollout: false,
            }, stages);
            frameB = runPerVariant({
                rawProj: { prompt: bEntry.prompt, response: bEntry.response },
                tokenNorms: scale !== 'none' ? bEntry.tokenNorms : undefined,
                isRollout: false,
            }, stages);
            frame = diffFrames(frameA, frameB, 'A-B');
        } catch (_e) {
            return null;
        }
    } else {
        const norms = PS.tokenNorms;
        const haveNorms = !!(norms && norms.response && norms.response.length);
        if (scale !== 'none' && !haveNorms) {
            PS.modeFallback = true;
            scale = 'none';
        }
        // The pipeline always operates on prompt+response and tracks promptLen
        // so it can center on response-only. Even when the caller doesn't want
        // the prompt portion in the output, building the full frame keeps the
        // centering math correct.
        frame = runPerVariant({
            rawProj: { prompt: promptTrace, response: rawResp },
            tokenNorms: scale !== 'none' ? norms : undefined,
            isRollout: false,
        }, {
            scale,
            center: PS.config.centered ? 'on' : 'off',
        });
    }

    const promptLen = promptTrace.length;
    if (includePrompt) return Array.from(frame.values);
    return Array.from(frame.values.subarray(promptLen));
}

function _rankTraits() {
    const cfg = PS.config;
    const filtered = cfg.traitFilter
        ? PS.traitList.filter(t => t.toLowerCase().includes(cfg.traitFilter.toLowerCase()))
        : PS.traitList;

    // When cohort is loaded, rank by cohort signal:
    //   - before_after / max_abs : computed on cohort-mean trajectory
    //   - span_vs_other / in_window_vs_out_window : per-pid score then averaged
    //     across cohort (not computable on a single offset-aligned cohort-mean
    //     because span boundaries differ per pid)
    if (PS.cohortShape) {
        const PER_PID_AVG_MODES = new Set(['span_vs_other', 'in_window_vs_out_window']);
        const scored = [];
        if (PER_PID_AVG_MODES.has(cfg.rankMode)) {
            // Score is the per-pid metric averaged across cohort, computed by
            // the cohort loader and stored in cohort.perTrait[trait].scores.
            for (const trait of filtered) {
                const c = PS.cohortShape.perTrait[trait];
                if (!c || c.n === 0) continue;
                const s = c.scores?.[cfg.rankMode];
                if (s == null) continue;
                scored.push({ trait, score: s });
            }
        } else {
            // before_after / max_abs : compute on cohort-mean directly.
            const cohortHalf = (PS.cohortShape.windowHalf ?? PS.cohortShape.cohortLen / 2);
            const ctx = {
                onset: cohortHalf,
                ranges: [],
                promptLen: 0,
                includesPrompt: false,
                responseLen: 2 * cohortHalf,
                windowHalf: cfg.windowHalf,
            };
            for (const trait of filtered) {
                const c = PS.cohortShape.perTrait[trait];
                if (!c || c.n === 0) continue;
                const score = rankTrait(c.mean, ctx, cfg.rankMode);
                if (score == null) continue;
                scored.push({ trait, score });
            }
        }
        scored.sort((a, b) => b.score - a.score);
        return scored;
    }

    // Cohort still loading or unavailable: temporary per-pid ranking (so the
    // chart isn't empty during the load). Header indicates this is provisional.
    const onset = PS.onset || 0;
    const ctx = {
        onset,
        ranges: PS.ranges || [],
        promptLen: 0,             // _transform called with includePrompt=false → response-only values
        includesPrompt: false,
        responseLen: undefined,    // pipeline output length is the responseLen
        windowHalf: cfg.windowHalf,
    };
    const scored = [];
    for (const trait of filtered) {
        const raw = PS.cache.get(trait);
        if (!raw || !raw.length) continue;
        const transformed = _transform(raw, trait, false);
        if (!transformed || !transformed.length) continue;
        const score = rankTrait(transformed, ctx, cfg.rankMode);
        if (score == null) continue;
        scored.push({ trait, score });
    }
    scored.sort((a, b) => b.score - a.score);
    return scored;
}

// Render the body — bars (CSS heatmap) or lines (Plotly), depending on cfg.view.
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
    const promptLen = PS.promptEnd || PS.tokenNorms?.prompt?.length || 0;

    const fallbackNote = PS.modeFallback && cfg.mode !== 'raw'
        ? ` · <span style="color:var(--text-warning,orange);">⚠ no token_norms — falling back to raw</span>`
        : '';

    // Ranking source: must mirror the logic in _rankTraits.
    //   - before_after / max_abs: scored on cohort-mean trajectory
    //   - span_vs_other / in_window_vs_out_window: per-pid score then averaged
    //     across cohort pids (each pid's own ranges/window)
    //   While cohort is still loading, all modes fall back to per-pid (this pid).
    const cohortLoaded = !!PS.cohortShape;
    const PER_PID_AVG_MODES = new Set(['span_vs_other', 'in_window_vs_out_window']);
    let rankSource, rankSourceColor;
    if (!cohortLoaded) {
        rankSource = `provisional: per-pid (this pid only — cohort ${PS.cohortLoading ? 'loading' : 'not yet loaded'})`;
        rankSourceColor = 'var(--text-warning,orange)';
    } else if (PER_PID_AVG_MODES.has(cfg.rankMode)) {
        rankSource = `per-pid score averaged across ${PS.cohortShape.nPidsLoaded} cohort pids`;
        rankSourceColor = 'var(--accent-color)';
    } else {
        rankSource = `cohort-mean trajectory (${PS.cohortShape.nPidsLoaded} pids)`;
        rankSourceColor = 'var(--accent-color)';
    }

    // What each rank mode actually compares (since they use different regions):
    let rankDetail;
    if (cfg.rankMode === 'span_vs_other') {
        rankDetail = ` · in_span = annotation tokens, out_span = rest of full response (ignores ±${W} window)`;
    } else if (cfg.rankMode === 'in_window_vs_out_window') {
        rankDetail = ` · in_window = ±${W} around onset, out_window = rest of full response`;
    } else if (cfg.rankMode === 'before_after') {
        rankDetail = ` · before/after each ${W} tokens around onset`;
    } else {
        rankDetail = ` · max anywhere in ±${PS.cohortShape?.windowHalf ?? W}`;
    }

    const headerHtml = `
        <div style="font-size:var(--text-xxs); color:var(--text-tertiary); margin-bottom:6px;">
            mode=<code>${cfg.mode}</code> · centered=<code>${cfg.centered}</code>${fallbackNote} ·
            ${cfg.showWindow ? `ranking window [-${W}, +${W}] around onset (token ${onset}) · chart shows [-${W + 10}, +${W + 10}], dotted lines mark ranking edges` : 'full prompt+response'} ·
            ${top.length} of ${PS.traitList?.length || '?'} traits ·
            ranked by <code>${cfg.rankMode}</code> on <span style="color:${rankSourceColor};">${rankSource}</span>${rankDetail} ·
            ${cfg.view === 'lines' ? 'thick line = current pid' : 'black bar = onset token'}
        </div>
    `;

    if (cfg.view === 'bars') {
        _paintBars(body, top, headerHtml, promptLen);
    } else {
        _paintLines(body, top, headerHtml, promptLen);
    }
}

// ─── Bars (legacy CSS bar-grid + per-row score column) ────────────────
//
// The shared component has a generic bars renderer, but ab wants the
// per-row score column visible on the right. So we keep the local
// renderer here. The transform pipeline still flows through the shared
// projection-transform.js.

function _paintBars(body, top, headerHtml, promptLen) {
    const cfg = PS.config;
    const onset = PS.onset || 0;
    const W = cfg.windowHalf;

    let xStart, xEnd;
    if (cfg.showWindow) {
        xStart = Math.max(0, onset - W);
        xEnd = onset + W;
    } else {
        xStart = -promptLen;     // negative = into prompt
        xEnd = PS.nResp;
    }

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

    const rows = top.map(r => _renderRow(r, traceCache.get(r.trait), xStart, xEnd, globalAbsMax)).join('');
    body.innerHTML = `
        ${headerHtml}
        <div style="font-family: var(--font-mono, monospace); font-size:11px;">
            ${rows}
        </div>
    `;
}

// ─── Lines (Plotly via shared component) ──────────────────────────────
//
// Uses renderPerTokenProjectionChart to draw a per-trait line chart.
// We hand it pre-fetched per-trait data + the same pipeline stage config
// the bars view runs, so the values match exactly.

function _paintLines(body, top, headerHtml, promptLen) {
    const cfg = PS.config;
    const onset = PS.onset || 0;
    const W = cfg.windowHalf;

    const cohortReady = !!PS.cohortShape;
    const cohortLabel = cohortReady
        ? (PS.biasShort
            ? `Bias-mean trajectories — ${PS.biasShort} (n=${PS.cohortShape.nPidsLoaded} pids, ±σ band)`
            : `Bias-mean trajectories (n=${PS.cohortShape.nPidsLoaded} pids, ±σ band)`)
        : '';
    body.innerHTML = `
        ${headerHtml}
        ${cohortReady ? `
            <div class="section-title" style="font-size:var(--text-xs);color:var(--text-tertiary);margin:6px 0 2px;">${cohortLabel}</div>
            <div id="ab-ps-cohort-plot" style="height:300px;"></div>
        ` : ''}
        <div class="section-title" style="font-size:var(--text-xs);color:var(--text-tertiary);margin:${cohortReady ? '14px' : '6px'} 0 2px;">This pid · ${PS.pid}</div>
        <div id="ab-ps-lines-plot" style="height:340px;"></div>
    `;

    // Build traitData from the loaded raw caches. Skip traits with empty traces.
    // In centered-delta mode we ALSO build baselineTraitData from PS.diffCache.instruct
    // and let the shared component run the diff combinator.
    const isDiff = _isDiffMode();
    const traitData = {};
    const baselineTraitData = isDiff ? {} : null;
    const traitOrder = [];
    for (const r of top) {
        if (isDiff) {
            const aEntry = PS.diffCache[DIFF_VARIANTS.primary].get(r.trait);
            const bEntry = PS.diffCache[DIFF_VARIANTS.baseline].get(r.trait);
            if (!aEntry || !bEntry) continue;
            traitData[r.trait] = {
                rawProj: { prompt: aEntry.prompt, response: aEntry.response },
                tokenNorms: aEntry.tokenNorms,
                metadata: { layer: aEntry.layer, baseline: aEntry.baseline },
            };
            baselineTraitData[r.trait] = {
                rawProj: { prompt: bEntry.prompt, response: bEntry.response },
                tokenNorms: bEntry.tokenNorms,
                metadata: { layer: bEntry.layer, baseline: bEntry.baseline },
            };
            traitOrder.push(r.trait);
        } else {
            const raw = PS.cache.get(r.trait);
            const promptTrace = PS.promptCache.get(r.trait) || [];
            if (!raw || !raw.length) continue;
            traitData[r.trait] = {
                rawProj: { prompt: promptTrace, response: raw },
                tokenNorms: PS.tokenNorms,
                metadata: PS.metaCache.get(r.trait),
            };
            traitOrder.push(r.trait);
        }
    }
    if (!traitOrder.length) {
        body.innerHTML = `${headerHtml}<div class="info" style="font-size:var(--text-xxs);">No trait projections loaded yet.</div>`;
        return;
    }

    // Pipeline stages: same mapping as bars/_transform. centered-delta uses
    // response_scale + center (per variant) + diff combinator (handled by component).
    let scale = _SCALE_FOR_MODE[cfg.mode] ?? 'none';
    // In diff mode, fall back if EITHER variant is missing norms (shared component
    // will run pipeline per variant; both must succeed).
    const tokenNormsOk = isDiff
        ? Object.values(traitData).every(t => t.tokenNorms?.response?.length) &&
          Object.values(baselineTraitData).every(t => t.tokenNorms?.response?.length)
        : !!PS.tokenNorms?.response?.length;
    if (scale !== 'none' && !tokenNormsOk) {
        PS.modeFallback = true;
        scale = 'none';
    }
    const stages = { scale, center: cfg.centered ? 'on' : 'off', smooth: cfg.smoothing };

    // Tokens for axis labels. Slice to [xStart, xEnd] window if window mode.
    const allTokens = PS.tokens || [];
    const promptTokens = allTokens.slice(0, promptLen);
    const responseTokens = allTokens.slice(promptLen);

    // Onset is in response coords; convert to flat (prompt+response) for highlight + window math.
    const flatOnset = promptLen + onset;

    // X-axis range for window-mode: ranking still uses strict ±W, but the
    // chart shows a visual pad of ±10 extra tokens on each side so the
    // viewer can see what's just outside the ranking window. The orange
    // band (annotation ranges) and the strict-W edges are therefore visible.
    const VISUAL_PAD = 10;
    const Wv = W + VISUAL_PAD;
    let xRange = null;
    if (cfg.showWindow) {
        xRange = [Math.max(0, flatOnset - Wv) - 0.5, Math.min(promptLen + responseTokens.length, flatOnset + Wv) - 0.5];
    }

    // Annotation ranges (in response coords) — pass through to draw orange bands.
    const annotationTokenRanges = (PS.ranges || []).map(([s, e]) => [s, e]);

    // Faint dashed verticals at the strict ranking-window edges (±W from onset).
    // The chart visually extends to ±(W + VISUAL_PAD) but ranking only uses ±W.
    // These markers tell the viewer where that boundary is.
    const rankingWindowMarkers = cfg.showWindow ? [
        { type: 'line', x0: flatOnset - W - 0.5, x1: flatOnset - W - 0.5, y0: 0, y1: 1, yref: 'paper',
          line: { color: 'rgba(255,255,255,0.25)', width: 1, dash: 'dot' } },
        { type: 'line', x0: flatOnset + W - 0.5, x1: flatOnset + W - 0.5, y0: 0, y1: 1, yref: 'paper',
          line: { color: 'rgba(255,255,255,0.25)', width: 1, dash: 'dot' } },
    ] : [];

    // Cohort overlay: pad offset-aligned cohort shape into flat coordinates so it
    // sits at the correct x position around the current pid's onset. The cohort
    // shape is 2*cohortHalf long (cohortHalf = ranking W + COHORT_VISUAL_PAD),
    // spanning [flatOnset - cohortHalf, flatOnset + cohortHalf).
    //
    // Outside-window positions are filled with NaN (not 0) so Plotly's `fill:
    // tonexty` band drops out cleanly instead of drawing a horizontal stripe at
    // y=0 across the entire chart.
    let cohort = null;
    if (cfg.cohortOverlay && PS.cohortShape) {
        const cs = PS.cohortShape;
        const cohortHalf = cs.windowHalf;
        const flatLen = promptLen + responseTokens.length;
        const cohortPerTrait = {};
        const startFlat = Math.max(0, flatOnset - cohortHalf);
        const endFlat = Math.min(flatLen, flatOnset + cohortHalf);
        for (const trait of traitOrder) {
            const c = cs.perTrait[trait];
            if (!c) continue;
            const padded = new Array(flatLen).fill(NaN);
            const paddedStd = new Array(flatLen).fill(NaN);
            for (let flat = startFlat; flat < endFlat; flat++) {
                const offset = flat - (flatOnset - cohortHalf);
                if (offset < 0 || offset >= c.mean.length) continue;
                padded[flat] = c.mean[offset];
                paddedStd[flat] = c.std[offset] || 0;
            }
            cohortPerTrait[trait] = { mean: padded, std: paddedStd, n: c.n };
        }
        cohort = {
            label: PS.biasShort ? `bias ${PS.biasId} (${PS.biasShort}, n=${cs.nPidsLoaded})` : `cohort (n=${cs.nPidsLoaded})`,
            perTrait: cohortPerTrait,
        };
    }

    // y-axis auto-range is scoped to the visible window when window mode is on,
    // so vertical space isn't wasted on data that's outside the cropped x-range.
    // Use the padded window (Wv = W + VISUAL_PAD) so data inside the visible
    // pad doesn't escape the y-bounds.
    const yScaleWindow = cfg.showWindow
        ? [Math.max(0, flatOnset - Wv), Math.min(promptLen + responseTokens.length, flatOnset + Wv)]
        : null;

    renderPerTokenProjectionChart('ab-ps-lines-plot', {
        traitData,
        baselineTraitData,
        diffOrder: 'A-B',                    // primary − baseline
        traitOrder,
        promptTokens,
        responseTokens,
        promptLen,
        isRollout: false,
        stages,
        highlightTokenIdx: flatOnset,
        annotationTokenRanges,
        extraShapes: rankingWindowMarkers,
        cohort,
        showCohortBand: !!cohort,
        showCurrentLine: true,
        renderer: 'lines',
        yScaleWindow,
        height: 360,
        showLegend: true,
        hoverTooltipId: 'ab-projection-hover',
        traitDisplayName: (t) => t,    // ab uses 'trait_set/trait' verbatim; inference uses getDisplayName
        hiddenTraits: cfg.hiddenTraits,
        onToggleTrait: (trait, hidden) => {
            if (hidden) cfg.hiddenTraits.add(trait);
            else cfg.hiddenTraits.delete(trait);
        },
    });

    // If window mode, crop x-axis with relayout (cleaner than slicing data — keeps full data for hover).
    if (xRange) {
        const plotEl = document.getElementById('ab-ps-lines-plot');
        if (plotEl && plotEl.layout) {
            // eslint-disable-next-line no-undef
            Plotly.relayout(plotEl, { 'xaxis.range': xRange });
        }
    }

    // ─── Second chart: bias-mean trajectories ─────────────────────────
    // Renders only when cohort data is loaded. Uses the SAME shared component,
    // but with each trait's "current line" set to the cohort mean (so the line
    // IS the bias average) and the std band overlaid. X-axis is offset-aligned
    // [-W, +W) with onset at x=0.
    if (PS.cohortShape && document.getElementById('ab-ps-cohort-plot')) {
        _paintCohortChart(traitOrder);
    }
}

function _paintCohortChart(traitOrder) {
    const cs = PS.cohortShape;
    const W = PS.config.windowHalf;                 // strict ranking half-window
    const cohortHalf = cs.windowHalf ?? (cs.cohortLen / 2);   // total loaded half-window (W + pad)
    const len = cs.cohortLen ?? 2 * cohortHalf;

    const cohortTraitData = {};
    const cohortPerTrait = {};
    const displayedOrder = [];
    for (const trait of traitOrder) {
        const c = cs.perTrait[trait];
        if (!c || c.n === 0) continue;
        // Use the cohort mean as the trait's "response" — pipeline runs no-op
        // (stages off) so values pass through unchanged. This makes the line
        // shown by the chart BE the cohort mean.
        cohortTraitData[trait] = {
            rawProj: { prompt: [], response: Array.from(c.mean) },
        };
        cohortPerTrait[trait] = { mean: c.mean, std: c.std, n: c.n };
        displayedOrder.push(trait);
    }
    if (!displayedOrder.length) {
        // Surface a useful message when the cohort produced no data.
        const el = document.getElementById('ab-ps-cohort-plot');
        if (el) el.innerHTML = `<div class="info" style="padding:var(--space-md);font-size:var(--text-xxs);color:var(--text-tertiary);">No cohort data — every trait got n=0 samples. Likely all cohort pids' projections failed to load (check console for fetch errors).</div>`;
        return;
    }

    // Vertical markers: solid white onset at x=0, dotted at strict ranking edges (±W).
    const onsetShape = {
        type: 'line', x0: 0, x1: 0, y0: 0, y1: 1, yref: 'paper',
        line: { color: 'rgba(255,255,255,0.6)', width: 2 },
    };
    const rankEdges = [
        { type: 'line', x0: -W, x1: -W, y0: 0, y1: 1, yref: 'paper',
          line: { color: 'rgba(255,255,255,0.5)', width: 1, dash: 'dot' } },
        { type: 'line', x0: W, x1: W, y0: 0, y1: 1, yref: 'paper',
          line: { color: 'rgba(255,255,255,0.5)', width: 1, dash: 'dot' } },
    ];

    renderPerTokenProjectionChart('ab-ps-cohort-plot', {
        traitData: cohortTraitData,
        traitOrder: displayedOrder,
        // X-axis: data array indices 0..len-1 represent offsets [-cohortHalf, +cohortHalf).
        // Setting startTokenIdx = -cohortHalf shifts axis labels so onset (data index cohortHalf) lands at x=0.
        startTokenIdx: -cohortHalf,
        promptLen: 0,
        responseTokens: new Array(len).fill(''),    // placeholder for length; tick labels come from offset
        promptTokens: [],
        isRollout: true,                            // skip prompt/response separator
        stages: { scale: 'none', center: 'off', smooth: 0 },   // pass-through; means already transformed
        cohort: { label: 'cohort', perTrait: cohortPerTrait },
        showCohortBand: true,
        showCurrentLine: true,
        extraShapes: [onsetShape, ...rankEdges],
        renderer: 'lines',
        height: 280,
        showLegend: true,
        hoverTooltipId: 'ab-cohort-hover',
        traitDisplayName: (t) => t,
    });

    // Custom x-axis ticks at offset milestones for clarity.
    const plotEl = document.getElementById('ab-ps-cohort-plot');
    if (plotEl && plotEl.layout) {
        const tickVals = [];
        const tickText = [];
        const step = Math.max(1, Math.round(cohortHalf / 5));
        for (let off = -cohortHalf; off <= cohortHalf - 1; off += step) {
            tickVals.push(off);
            tickText.push(off === 0 ? 'onset' : (off > 0 ? `+${off}` : `${off}`));
        }
        // eslint-disable-next-line no-undef
        Plotly.relayout(plotEl, {
            'xaxis.tickmode': 'array',
            'xaxis.tickvals': tickVals,
            'xaxis.ticktext': tickText,
            'xaxis.title': 'Offset from onset (tokens)',
        });
    }
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
