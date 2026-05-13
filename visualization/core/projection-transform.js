// Composable per-token projection transform pipeline.
//
// Single source of truth for the transformation chain that turns a raw
// per-token projection JSON into the values shown in a chart. Both the
// inference view and the annotation-browser run identical transforms;
// previously each implemented its own pass over `traitData`. Now they
// configure stages and call `runPipeline`.
//
// Stages, in fixed order:
//   1. clean     — co-clean projections + recompute norms using massive-dim data
//   2. scale     — divide projections by token / response norms (cosine | response_scale)
//   3. baseline  — subtract a scalar baseline carried in projection metadata
//   4. center    — subtract per-response mean (response-only, even on full prompt+response)
//   5. smooth    — boundary-respecting moving average (does NOT cross prompt/response)
//
// Diff is an OUTER COMBINATOR over two per-variant pipeline outputs. It is
// not a stage. Reason: token_norms and massive_dim_data are model-specific.
// Subtracting normalized values from two variants gives a meaningful
// per-variant-difference. Subtracting raw and then normalizing requires
// choosing whose norms to use, which is mathematically ill-defined.
//
// Caching: callers pass a cache object (Map-like). The pipeline computes a
// stable signature per stage-group and memoizes intermediate Float32Arrays.
// Tier-2 (post clean+scale+baseline) is the expensive part; center+smooth
// are recomputed inline.
//
// Input/output: each stage takes and returns a `Frame`:
//   {
//     values:      Float32Array,         // per-token projection values (current)
//     promptLen:   number,
//     responseLen: number,
//     tokenNorms:  {prompt, response}?,  // co-cleaned by `clean` stage if active
//     mddData:     object?,              // unchanged after `clean`
//     baseline:    number?,
//     isRollout:   boolean,              // no response — center on full trace
//   }
//
// Usage:
//   import { runPerVariant, diff } from 'core/projection-transform.js';
//   const out = runPerVariant({ rawProj, promptLen, responseLen, tokenNorms, mddData, baseline, isRollout },
//                             { clean: 'top-3', scale: 'response_scale', baseline: 'on', center: 'on', smooth: 5 });
//   out.values  // Float32Array

import { getDimsToRemove, applyMassiveDimCleaning, computeCleanedNorms } from './massive-dims.js';

const SCALE_MODES = new Set(['none', 'cosine', 'response_scale']);
const CLEAN_MODES = new Set(['none', 'top-1', 'top-3', 'top-5', 'top-10', 'all']);

// =============================================================================
// Stage 1: clean (massive-dim subtraction, paired with norms)
// =============================================================================

/**
 * Co-clean projections + token norms.
 *
 * Massive dims are model-specific. When this stage is active, the projection
 * vector h·v has the contribution from those dims removed; correspondingly
 * the magnitude ||h|| is recomputed from the cleaned activations. Both
 * happen together — running scale/cosine on a cleaned projection but an
 * uncleaned norm is geometrically incoherent.
 *
 * Mode 'none' is a no-op pass-through.
 *
 * Throws if mode != 'none' and mddData is missing — fail loud, no silent
 * fallback.
 */
function applyClean(frame, mode) {
    if (!CLEAN_MODES.has(mode)) throw new Error(`Unknown clean mode: ${mode}`);
    if (mode === 'none' || !mode) return frame;
    if (!frame.mddData) throw new Error(`clean=${mode} requires mddData; not provided`);

    const dimsToRemove = getDimsToRemove(frame.mddData, mode);
    if (dimsToRemove.length === 0) return frame;

    const promptVals = Array.from(frame.values.subarray(0, frame.promptLen));
    const responseVals = Array.from(frame.values.subarray(frame.promptLen));
    const cleanedPrompt = applyMassiveDimCleaning(promptVals, frame.mddData, dimsToRemove, 'prompt');
    const cleanedResponse = applyMassiveDimCleaning(responseVals, frame.mddData, dimsToRemove, 'response');

    const cleanedValues = new Float32Array(cleanedPrompt.length + cleanedResponse.length);
    cleanedValues.set(cleanedPrompt, 0);
    cleanedValues.set(cleanedResponse, cleanedPrompt.length);

    let cleanedNorms = frame.tokenNorms;
    if (frame.tokenNorms) {
        cleanedNorms = {
            prompt: computeCleanedNorms(frame.tokenNorms.prompt, frame.mddData, dimsToRemove, 'prompt'),
            response: computeCleanedNorms(frame.tokenNorms.response, frame.mddData, dimsToRemove, 'response'),
        };
    }
    return { ...frame, values: cleanedValues, tokenNorms: cleanedNorms };
}

// =============================================================================
// Stage 2: scale (cosine = per-token, response_scale = mean ||h||)
// =============================================================================

/**
 * Apply norm-based scaling.
 *   - 'none':            pass through (raw h·v projection)
 *   - 'cosine':          divide each token by its own ||h_t|| (per-token)
 *   - 'response_scale':  divide by mean ||h|| over response (per-response scalar)
 *
 * For rollouts (no response), 'response_scale' falls back to mean ||h|| over
 * the full trace.
 *
 * Throws if scale != 'none' and tokenNorms missing.
 */
function applyScale(frame, mode) {
    if (!SCALE_MODES.has(mode)) throw new Error(`Unknown scale mode: ${mode}`);
    if (mode === 'none' || !mode) return frame;
    if (!frame.tokenNorms) throw new Error(`scale=${mode} requires tokenNorms; not provided`);

    const allNorms = new Float32Array(frame.tokenNorms.prompt.length + frame.tokenNorms.response.length);
    allNorms.set(frame.tokenNorms.prompt, 0);
    allNorms.set(frame.tokenNorms.response, frame.tokenNorms.prompt.length);

    const out = new Float32Array(frame.values.length);
    if (mode === 'cosine') {
        for (let i = 0; i < frame.values.length; i++) {
            const n = allNorms[i];
            out[i] = n > 0 ? frame.values[i] / n : 0;
        }
    } else {  // response_scale
        const respNorms = frame.isRollout ? allNorms : allNorms.subarray(frame.promptLen);
        let sum = 0;
        for (let i = 0; i < respNorms.length; i++) sum += respNorms[i];
        const mean = respNorms.length > 0 ? sum / respNorms.length : 1;
        if (mean > 0) {
            for (let i = 0; i < frame.values.length; i++) out[i] = frame.values[i] / mean;
        }
    }
    return { ...frame, values: out };
}

// =============================================================================
// Stage 3: baseline (subtract scalar from metadata)
// =============================================================================

/**
 * Subtract a scalar baseline (carried in projection JSON metadata) from every
 * value. Useful when extraction was done with a non-zero reference activation.
 *
 * Mode 'on' subtracts; 'off' is no-op.
 */
function applyBaseline(frame, mode) {
    if (mode === 'off' || !mode) return frame;
    if (mode !== 'on') throw new Error(`Unknown baseline mode: ${mode}`);
    if (frame.baseline == null) return frame;  // no baseline in metadata = no-op

    const out = new Float32Array(frame.values.length);
    for (let i = 0; i < frame.values.length; i++) out[i] = frame.values[i] - frame.baseline;
    return { ...frame, values: out };
}

// =============================================================================
// Stage 4: center (mean-center per response)
// =============================================================================

/**
 * Subtract the mean over the RESPONSE portion (not the full trace) from every
 * value. This makes "high relative to this response's baseline" meaningful
 * even for traits that are constantly active (golden gate bridge, formality).
 *
 * For rollouts (no response), centers on the full trace instead.
 */
function applyCenter(frame, mode) {
    if (mode === 'off' || !mode) return frame;
    if (mode !== 'on') throw new Error(`Unknown center mode: ${mode}`);
    if (frame.values.length === 0) return frame;

    const refSlice = frame.isRollout
        ? frame.values
        : frame.values.subarray(frame.promptLen);
    if (refSlice.length === 0) return frame;
    let sum = 0;
    for (let i = 0; i < refSlice.length; i++) sum += refSlice[i];
    const mean = sum / refSlice.length;

    const out = new Float32Array(frame.values.length);
    for (let i = 0; i < frame.values.length; i++) out[i] = frame.values[i] - mean;
    return { ...frame, values: out };
}

// =============================================================================
// Stage 5: smooth (boundary-respecting moving average)
// =============================================================================

/**
 * Centered moving average on prompt and response slices INDEPENDENTLY.
 *
 * Why: smoothing across the prompt/response boundary leaks prompt-domain
 * signal into early response tokens (and vice versa). The naive
 * `core/utils.js#smoothData` doesn't respect boundaries — this one does.
 *
 * windowSize <= 1 is a no-op.
 *
 * For rollouts (no response slice), smooth over the full trace.
 */
function applySmooth(frame, windowSize) {
    if (!windowSize || windowSize <= 1) return frame;
    const out = new Float32Array(frame.values.length);
    if (frame.isRollout) {
        _smoothSlice(frame.values, 0, frame.values.length, windowSize, out);
    } else {
        if (frame.promptLen > 0) _smoothSlice(frame.values, 0, frame.promptLen, windowSize, out);
        if (frame.values.length > frame.promptLen) {
            _smoothSlice(frame.values, frame.promptLen, frame.values.length, windowSize, out);
        }
    }
    return { ...frame, values: out };
}

function _smoothSlice(src, start, end, windowSize, dst) {
    const len = end - start;
    if (len <= 0) return;
    if (len < windowSize) {
        // Slice shorter than the window — copy through unchanged.
        for (let i = start; i < end; i++) dst[i] = src[i];
        return;
    }
    const half = Math.floor(windowSize / 2);
    for (let i = start; i < end; i++) {
        const lo = Math.max(start, i - half);
        const hi = Math.min(end, i + half + 1);
        let sum = 0;
        for (let j = lo; j < hi; j++) sum += src[j];
        dst[i] = sum / (hi - lo);
    }
}

// =============================================================================
// Public: per-variant pipeline runner
// =============================================================================

/**
 * Run the per-variant transform pipeline.
 *
 * @param {Object} input
 *   - rawProj: number[] | Float32Array | { prompt: number[], response: number[] }
 *   - promptLen, responseLen: numbers (must match rawProj layout)
 *   - tokenNorms?: { prompt, response }
 *   - mddData?: object
 *   - baseline?: number
 *   - isRollout?: boolean (default false)
 *
 * @param {Object} stages — { clean, scale, baseline, center, smooth }
 *   - clean:    'none' | 'top-1' | 'top-3' | 'top-5' | 'top-10' | 'all'  (default 'none')
 *   - scale:    'none' | 'cosine' | 'response_scale'                      (default 'none')
 *   - baseline: 'off' | 'on'                                              (default 'off')
 *   - center:   'off' | 'on'                                              (default 'off')
 *   - smooth:   number (window size, 0 or 1 = no-op)                      (default 0)
 *
 * @returns {Frame} { values, promptLen, responseLen, tokenNorms, mddData, baseline, isRollout }
 */
function runPerVariant(input, stages) {
    const initial = _normalizeInput(input);
    const cfg = _normalizeStages(stages);

    let frame = initial;
    frame = applyClean(frame, cfg.clean);
    frame = applyScale(frame, cfg.scale);
    frame = applyBaseline(frame, cfg.baseline);
    frame = applyCenter(frame, cfg.center);
    frame = applySmooth(frame, cfg.smooth);
    return frame;
}

function _normalizeInput(input) {
    let values;
    let promptLen = input.promptLen ?? 0;
    let responseLen = input.responseLen;
    if (Array.isArray(input.rawProj) || ArrayBuffer.isView(input.rawProj)) {
        values = input.rawProj instanceof Float32Array
            ? new Float32Array(input.rawProj)
            : Float32Array.from(input.rawProj);
    } else if (input.rawProj && Array.isArray(input.rawProj.prompt)) {
        const p = input.rawProj.prompt;
        const r = input.rawProj.response;
        values = new Float32Array(p.length + r.length);
        values.set(p, 0);
        values.set(r, p.length);
        if (input.promptLen == null) promptLen = p.length;
        if (responseLen == null) responseLen = r.length;
    } else {
        throw new Error('runPerVariant: rawProj must be array or {prompt, response}');
    }
    if (responseLen == null) responseLen = values.length - promptLen;
    return {
        values,
        promptLen,
        responseLen,
        tokenNorms: input.tokenNorms || null,
        mddData: input.mddData || null,
        baseline: input.baseline ?? null,
        isRollout: !!input.isRollout,
    };
}

function _normalizeStages(stages) {
    return {
        clean: stages?.clean || 'none',
        scale: stages?.scale || 'none',
        baseline: stages?.baseline || 'off',
        center: stages?.center || 'off',
        smooth: stages?.smooth || 0,
    };
}

// =============================================================================
// Diff combinator (outer)
// =============================================================================

/**
 * Subtract two per-variant pipeline outputs token-for-token.
 *
 * Both inputs MUST be the result of `runPerVariant` on token-aligned
 * variants (same generation, same tokenization). Length mismatch throws —
 * fail loud, no silent truncation. The caller (e.g. inference data layer
 * or annotation-browser) is responsible for ensuring alignment.
 *
 * @param {Frame} frameA
 * @param {Frame} frameB
 * @param {string} order — 'A-B' (default) or 'B-A'
 * @returns {Frame} a frame with values = A − B (or B − A); other fields
 *                  inherited from frameA (with tokenNorms set to null since
 *                  a diff has no clean geometric ||h|| interpretation).
 */
function diff(frameA, frameB, order = 'A-B') {
    if (frameA.values.length !== frameB.values.length) {
        throw new Error(
            `diff: token length mismatch — A=${frameA.values.length}, B=${frameB.values.length}. ` +
            `This indicates the two variants did not project the same generation.`
        );
    }
    if (frameA.promptLen !== frameB.promptLen) {
        throw new Error(`diff: promptLen mismatch — A=${frameA.promptLen}, B=${frameB.promptLen}`);
    }
    const out = new Float32Array(frameA.values.length);
    if (order === 'A-B') {
        for (let i = 0; i < out.length; i++) out[i] = frameA.values[i] - frameB.values[i];
    } else if (order === 'B-A') {
        for (let i = 0; i < out.length; i++) out[i] = frameB.values[i] - frameA.values[i];
    } else {
        throw new Error(`diff: unknown order ${order}`);
    }
    return {
        values: out,
        promptLen: frameA.promptLen,
        responseLen: frameA.responseLen,
        tokenNorms: null,
        mddData: null,
        baseline: null,
        isRollout: frameA.isRollout,
    };
}

// =============================================================================
// Cache helper
// =============================================================================

/**
 * Build a cache signature for a (pid, variantPair, stages) combination.
 * Use this to memoize Tier-2 (post clean+scale+baseline) Float32Arrays.
 * center+smooth are cheap enough to recompute inline.
 */
function tier2Signature({ pid, variantA, variantB = null, clean, scale, baseline }) {
    return `${pid}|${variantA}|${variantB || ''}|${clean || 'none'}|${scale || 'none'}|${baseline || 'off'}`;
}

// =============================================================================
// Exports
// =============================================================================

export {
    runPerVariant,
    diff,
    tier2Signature,
    // individual stages exported for tests + advanced callers
    applyClean,
    applyScale,
    applyBaseline,
    applyCenter,
    applySmooth,
    // constants
    SCALE_MODES,
    CLEAN_MODES,
};
