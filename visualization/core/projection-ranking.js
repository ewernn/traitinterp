// Modular ranking functions for per-token projection traces.
//
// Each function takes a transformed value array (post pipeline + optional
// diff) and a context object, and returns a score in [0, ∞) or null. Null
// means "this trait is not rankable under this mode for this context"
// (e.g. onset is too close to a boundary for a before/after window).
// Callers should drop nulls; do NOT default-to-zero or special-case
// fallbacks.
//
// Adding a rank mode:
//   1. write a function with signature (values, ctx) → number | null
//   2. register it in RANK_FUNCTIONS
//   3. add a UI option that maps to the same key
//
// Context fields used by the built-in modes:
//   - onset:        token index of primary onset (response coords)
//   - ranges:       [[start, end], ...] all annotation instance ranges
//   - promptLen:    number — to translate response-coord onset if values
//                   array includes prompt+response
//   - includesPrompt: boolean — true if `values` is prompt+response, false
//                   if response-only
//   - responseLen:  number
//   - windowHalf:   number (tokens on each side for before/after windows)

/**
 * |mean(before onset) − mean(after onset)|.
 *
 * Returns null when either window is empty (onset within `windowHalf` of
 * response start/end), so the caller can drop traits where the window
 * would be misleading rather than scoring them as zero or near-zero.
 */
function rankBeforeAfter(values, ctx) {
    const onsetIdx = _toFlatIdx(ctx.onset, ctx);
    const responseStart = ctx.includesPrompt ? ctx.promptLen : 0;
    const responseEnd = responseStart + (ctx.responseLen ?? (values.length - responseStart));
    const W = ctx.windowHalf;
    const beforeStart = Math.max(responseStart, onsetIdx - W);
    const beforeEnd = onsetIdx;
    const afterStart = onsetIdx;
    const afterEnd = Math.min(responseEnd, onsetIdx + W);
    if (beforeEnd <= beforeStart || afterEnd <= afterStart) return null;

    const meanBefore = _meanRange(values, beforeStart, beforeEnd);
    const meanAfter = _meanRange(values, afterStart, afterEnd);
    return Math.abs(meanBefore - meanAfter);
}

/**
 * |mean(in any annotation span) − mean(everywhere else in response)|.
 */
function rankSpanVsOther(values, ctx) {
    const responseStart = ctx.includesPrompt ? ctx.promptLen : 0;
    const responseEnd = responseStart + (ctx.responseLen ?? (values.length - responseStart));
    const inIdx = new Set();
    for (const r of (ctx.ranges || [])) {
        for (let i = r[0]; i < r[1]; i++) {
            inIdx.add(_toFlatIdx(i, ctx));
        }
    }
    let inSum = 0, inCount = 0, outSum = 0, outCount = 0;
    for (let i = responseStart; i < responseEnd; i++) {
        if (inIdx.has(i)) { inSum += values[i]; inCount += 1; }
        else { outSum += values[i]; outCount += 1; }
    }
    if (inCount === 0 || outCount === 0) return null;
    return Math.abs((inSum / inCount) - (outSum / outCount));
}

/**
 * |mean(values inside ±windowHalf around onset) − mean(values outside that window in the response)|.
 *
 * Like span_vs_other but the partition is by a fixed onset-centered window
 * (caller's windowHalf) instead of the annotation span boundaries. Works
 * uniformly across pids since it only needs an onset, not span ranges.
 */
function rankInWindowVsOutWindow(values, ctx) {
    const responseStart = ctx.includesPrompt ? ctx.promptLen : 0;
    const responseEnd = responseStart + (ctx.responseLen ?? (values.length - responseStart));
    const onsetIdx = _toFlatIdx(ctx.onset, ctx);
    const W = ctx.windowHalf;
    const winStart = Math.max(responseStart, onsetIdx - W);
    const winEnd = Math.min(responseEnd, onsetIdx + W);
    let inS = 0, inC = 0, outS = 0, outC = 0;
    for (let i = responseStart; i < responseEnd; i++) {
        const v = values[i];
        if (!Number.isFinite(v)) continue;
        if (i >= winStart && i < winEnd) { inS += v; inC += 1; }
        else { outS += v; outC += 1; }
    }
    if (inC === 0 || outC === 0) return null;
    return Math.abs(inS / inC - outS / outC);
}

/**
 * max |value| over the response.
 */
function rankMaxAbs(values, ctx) {
    const responseStart = ctx.includesPrompt ? ctx.promptLen : 0;
    const responseEnd = responseStart + (ctx.responseLen ?? (values.length - responseStart));
    let m = 0;
    for (let i = responseStart; i < responseEnd; i++) {
        const a = Math.abs(values[i]);
        if (a > m) m = a;
    }
    return m;
}

const RANK_FUNCTIONS = Object.freeze({
    before_after: rankBeforeAfter,
    span_vs_other: rankSpanVsOther,
    in_window_vs_out_window: rankInWindowVsOutWindow,
    max_abs: rankMaxAbs,
});

/**
 * Score one trait. Caller controls iteration + sorting.
 *
 * Returns score (≥ 0) or null. Throws on unknown mode.
 */
function rankTrait(values, ctx, mode) {
    const fn = RANK_FUNCTIONS[mode];
    if (!fn) throw new Error(`Unknown rank mode: ${mode}`);
    return fn(values, ctx);
}

// ---------- internals ----------

function _toFlatIdx(responseTokenIdx, ctx) {
    return ctx.includesPrompt ? ctx.promptLen + responseTokenIdx : responseTokenIdx;
}

function _meanRange(arr, start, end) {
    let s = 0;
    for (let i = start; i < end; i++) s += arr[i];
    return s / (end - start);
}

export {
    rankBeforeAfter,
    rankSpanVsOther,
    rankInWindowVsOutWindow,
    rankMaxAbs,
    rankTrait,
    RANK_FUNCTIONS,
};
