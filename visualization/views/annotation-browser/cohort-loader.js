// Cohort data loader for the projection strip.
//
// Loads per-token projections across N pids in a cohort (single bias, or
// a cluster of biases per atlas-39 results), aligns them to each pid's
// onset, and computes per-trait mean ± std centered-delta shape.
//
// Output shape matches what per-token-projection-chart.js's `cohort.perTrait`
// expects: { mean: Float32Array, std: Float32Array, n: number } per trait.
// The output array is offset-aligned: index 0 = onset − windowHalf, index
// 2*windowHalf = onset + windowHalf − 1.
//
// Why offset-aligned: pids have different absolute onsets (token 50 in one,
// token 200 in another). To average shapes meaningfully we must align them
// on a shared coordinate (offset = abs_token − pid_onset).
//
// Memory budget: at 173 traits × N pids × 2 variants × ~3KB each, this is
// ~6KB × 173 × N × 2 = ~2MB per pid for both variants. For a 90-pid
// movies_similar bias, that's 180MB of raw JSONs. Mitigation: caller passes
// `topTraits` to limit fetches to the N traits we'll actually display.
//
// Caching: per-(cohortKey, variant) cohort traces. Cohort key is derived
// from the pids+trait list so re-renders without changes are free.

import { fetchProjection } from '../../core/projection-store.js';
import { runPerVariant, diff as diffFrames } from '../../core/projection-transform.js';
import { EXPERIMENT, PROMPT_SET } from './data.js';

// (cohortKey, primaryVariant, baselineVariant, mode, traitsHash) -> { perTrait, n }
const _cohortCache = new Map();

/**
 * Load + align + average a cohort of pids.
 *
 * @param {Object} opts
 *   - pidsWithOnsets: [{pid, onset}, ...] — onset in response-coords
 *   - traits:         ['trait_set/trait', ...]   (top-K, to limit fetches)
 *   - primaryVariant: 'rm_lora'
 *   - baselineVariant: 'instruct' | null   (null = single-variant mode)
 *   - stages:         { scale, center } pipeline config (same as per-pid)
 *   - windowHalf:     half-window in tokens
 *   - cohortLabel:    string label for caching + display
 *   - onProgress:     optional (done, total) callback
 *
 * Returns: {
 *   perTrait: { [trait]: { mean: Float32Array(2W), std: Float32Array(2W), n: number } },
 *   nPidsLoaded: number,
 *   pidsSkipped: [{pid, reason}],
 *   cohortLen: 2 * windowHalf,
 * }
 */
async function loadCohortShape({
    pidsWithMeta,            // [{pid, onset, ranges}]  — onset is response-coord int; ranges is [[start,end], ...]
    traits, primaryVariant, baselineVariant, stages, windowHalf, rankWindowHalf, cohortLabel, onProgress,
    // Back-compat: accept the old `pidsWithOnsets` arg too for safety.
    pidsWithOnsets,
}) {
    if (!pidsWithMeta && pidsWithOnsets) {
        pidsWithMeta = pidsWithOnsets.map(p => ({ pid: p.pid, onset: p.onset, ranges: [] }));
    }
    const isDiff = !!baselineVariant;
    const rankWh = rankWindowHalf ?? windowHalf;
    const cohortKey = `${cohortLabel}|${primaryVariant}|${baselineVariant || ''}|` +
                      `${stages.scale}|${stages.center}|${stages.smooth || 0}|${windowHalf}|${rankWh}|${traits.length}`;
    if (_cohortCache.has(cohortKey)) return _cohortCache.get(cohortKey);

    const W = windowHalf;
    const len = 2 * W;
    // Per-trait running sums for online mean/std (Welford's algorithm-lite —
    // for our N (max ~100), naive sum/sumSq is fine and simpler).
    const sums = {};        // trait -> Float64Array(len)
    const sumSqs = {};      // trait -> Float64Array(len)
    const counts = {};      // trait -> Uint32Array(len)
    // Per-trait per-pid metric scores (sum + count for averaging).
    // Keys: 'span_vs_other', 'in_window_vs_out_window'. Computed per-pid then averaged.
    const scoreSums = {};       // trait -> {metric -> sum}
    const scoreCounts = {};     // trait -> {metric -> count}
    for (const t of traits) {
        sums[t] = new Float64Array(len);
        sumSqs[t] = new Float64Array(len);
        counts[t] = new Uint32Array(len);
        scoreSums[t] = { span_vs_other: 0, in_window_vs_out_window: 0 };
        scoreCounts[t] = { span_vs_other: 0, in_window_vs_out_window: 0 };
    }

    const pidsSkipped = [];
    let nPidsLoaded = 0;

    // Fetch each pid's projections + transform + accumulate.
    // Bounded concurrency: 4 pids in flight at a time (each pid does many trait fetches).
    const concurrency = 4;
    let cursor = 0;
    let done = 0;
    const total = pidsWithMeta.length;

    async function worker() {
        while (true) {
            const idx = cursor++;
            if (idx >= total) return;
            const { pid, onset, ranges } = pidsWithMeta[idx];
            try {
                await _processPid({
                    pid, onset, ranges: ranges || [],
                    traits, primaryVariant, baselineVariant, isDiff,
                    stages, W, rankWh, sums, sumSqs, counts, scoreSums, scoreCounts,
                });
                nPidsLoaded += 1;
            } catch (e) {
                pidsSkipped.push({ pid, reason: e.message });
            }
            done += 1;
            if (onProgress) onProgress(done, total);
        }
    }
    _resetSkipCounters();
    await Promise.all(Array.from({ length: concurrency }, () => worker()));
    if (_skipReasons.size > 0) {
        // eslint-disable-next-line no-console
        console.info(`cohort skip summary (${nPidsLoaded}/${total} pids loaded): ${_skipSummary()}`);
    }

    // Reduce to mean / std per trait.
    const perTrait = {};
    for (const t of traits) {
        const mean = new Float32Array(len);
        const std = new Float32Array(len);
        let traitN = 0;
        for (let i = 0; i < len; i++) {
            const c = counts[t][i];
            if (c > 0) {
                const m = sums[t][i] / c;
                mean[i] = m;
                if (c > 1) {
                    const v = (sumSqs[t][i] / c) - m * m;
                    std[i] = Math.sqrt(Math.max(0, v));
                }
                if (c > traitN) traitN = c;
            }
        }
        // Per-pid metric scores averaged across the cohort.
        const scores = {};
        for (const metric of ['span_vs_other', 'in_window_vs_out_window']) {
            const c = scoreCounts[t][metric];
            scores[metric] = c > 0 ? scoreSums[t][metric] / c : null;
        }
        perTrait[t] = { mean, std, n: traitN, scores };
    }

    const result = {
        perTrait,
        nPidsLoaded,
        pidsSkipped,
        cohortLen: len,
        windowHalf: W,
        rankWindowHalf: rankWh,
    };
    _cohortCache.set(cohortKey, result);
    return result;
}

// Per-trait skip reason counts. Reset on each loadCohortShape entry (call _resetSkipCounters).
let _skipReasons = new Map();   // reason -> count
function _bumpSkip(reason) { _skipReasons.set(reason, (_skipReasons.get(reason) || 0) + 1); }
function _resetSkipCounters() { _skipReasons = new Map(); }
function _skipSummary() { return Array.from(_skipReasons.entries()).map(([k,v]) => `${k}: ${v}`).join(' · '); }

async function _processPid({
    pid, onset, ranges, traits, primaryVariant, baselineVariant, isDiff, stages, W, rankWh,
    sums, sumSqs, counts, scoreSums, scoreCounts,
}) {
    // Fetch all traits for this pid (both variants if diff). One Promise.all per variant.
    const fetchAll = async (variant) => {
        const promises = traits.map(async (traitFull) => {
            const [traitSet, trait] = traitFull.split('/');
            const proj = await fetchProjection({
                experiment: EXPERIMENT, promptSet: PROMPT_SET,
                variant, traitSet, trait, pid,
            });
            if (!proj?.projections?.length) return [traitFull, null];
            const e = proj.projections[0];
            return [traitFull, {
                response: e.response || [],
                prompt: e.prompt || [],
                tokenNorms: e.token_norms || null,
            }];
        });
        return Object.fromEntries(await Promise.all(promises));
    };

    const primary = await fetchAll(primaryVariant);
    const baseline = isDiff ? await fetchAll(baselineVariant) : null;

    // Transform + accumulate per trait.
    for (const traitFull of traits) {
        const aEntry = primary[traitFull];
        if (!aEntry) continue;
        const promptLen = aEntry.prompt.length;
        const responseLen = aEntry.response.length;

        let frame;
        let frameResponseLen = responseLen;     // length of frame.values' response slice
        let framePromptLen = promptLen;
        try {
            // In diff mode, downgrade scale to 'none' if either variant is missing
            // norms — same fallback behavior as the per-pid strip. Avoids the loader
            // throwing on every trait when norms are missing.
            let effScale = stages.scale;
            if (effScale !== 'none') {
                const aOk = !!aEntry.tokenNorms?.response?.length;
                const bOk = !isDiff || !!baseline[traitFull]?.tokenNorms?.response?.length;
                if (!aOk || !bOk) {
                    effScale = 'none';
                    _bumpSkip('downgraded-to-none (missing norms)');
                }
            }
            const effStages = { ...stages, scale: effScale };
            const frameA = runPerVariant({
                rawProj: { prompt: aEntry.prompt, response: aEntry.response },
                tokenNorms: effScale !== 'none' ? aEntry.tokenNorms : undefined,
                isRollout: false,
            }, effStages);
            if (isDiff) {
                const bEntry = baseline[traitFull];
                if (!bEntry) { _bumpSkip('baseline missing'); continue; }
                const frameB = runPerVariant({
                    rawProj: { prompt: bEntry.prompt, response: bEntry.response },
                    tokenNorms: effScale !== 'none' ? bEntry.tokenNorms : undefined,
                    isRollout: false,
                }, effStages);
                // Diff on response-only slices — prompt sides may differ in length
                // across variants (different chat templates / BOS handling) but
                // the response is what aligns to onset. Slice each variant's
                // response to common length and reconstruct a frame.
                const aPromptLen = aEntry.prompt.length;
                const bPromptLen = bEntry.prompt.length;
                const aResp = frameA.values.subarray(aPromptLen);
                const bResp = frameB.values.subarray(bPromptLen);
                const minRespLen = Math.min(aResp.length, bResp.length);
                if (minRespLen === 0) { _bumpSkip('empty response'); continue; }
                if (aResp.length !== bResp.length) _bumpSkip(`response length mismatch (a=${aResp.length} b=${bResp.length})`);
                const diffResp = new Float32Array(minRespLen);
                for (let i = 0; i < minRespLen; i++) diffResp[i] = aResp[i] - bResp[i];
                // Build a synthetic frame whose values is just the diff'd response.
                // Prompt side is omitted (we accumulate into [flatStart..flatEnd)
                // computed against framePromptLen=0 below).
                frame = { values: diffResp, promptLen: 0, responseLen: minRespLen, isRollout: false };
                frameResponseLen = minRespLen;
                framePromptLen = 0;
            } else {
                frame = frameA;
            }
        } catch (e) {
            _bumpSkip(`pipeline error: ${e.message}`);
            continue;
        }

        // Slice ±W around onset (in response-coords) of the frame. Diff mode
        // synthesises a response-only frame so framePromptLen=0; non-diff frames
        // include prompt+response so flatOnset = framePromptLen + onset.
        const flatOnset = framePromptLen + onset;
        const flatStart = flatOnset - W;

        const traitSums = sums[traitFull];
        const traitSumSqs = sumSqs[traitFull];
        const traitCounts = counts[traitFull];

        for (let offset = 0; offset < 2 * W; offset++) {
            const flatIdx = flatStart + offset;
            // Restrict to the response slice of the frame.
            if (flatIdx < framePromptLen || flatIdx >= framePromptLen + frameResponseLen) continue;
            const v = frame.values[flatIdx];
            if (!Number.isFinite(v)) continue;
            traitSums[offset] += v;
            traitSumSqs[offset] += v * v;
            traitCounts[offset] += 1;
        }

        // Per-pid metric scores. These get averaged across cohort pids for
        // ranking. They use the FULL frame (not just the cohort window) so
        // the in/out partition is over the entire response.
        if (scoreSums && scoreCounts) {
            const respStart = framePromptLen;
            const respEnd = framePromptLen + frameResponseLen;

            // 1. span_vs_other: |mean(values inside any annotation range) - mean(values outside)|
            if (ranges && ranges.length) {
                const inIdx = new Set();
                for (const r of ranges) {
                    for (let i = r[0]; i < r[1]; i++) inIdx.add(framePromptLen + i);
                }
                let inS = 0, inC = 0, outS = 0, outC = 0;
                for (let i = respStart; i < respEnd; i++) {
                    const v = frame.values[i];
                    if (!Number.isFinite(v)) continue;
                    if (inIdx.has(i)) { inS += v; inC += 1; }
                    else { outS += v; outC += 1; }
                }
                if (inC > 0 && outC > 0) {
                    scoreSums[traitFull].span_vs_other += Math.abs(inS / inC - outS / outC);
                    scoreCounts[traitFull].span_vs_other += 1;
                }
            }

            // 2. in_window_vs_out_window: |mean(values in [onset-rankWh, onset+rankWh)) - mean(values outside that window in response)|
            const winStart = framePromptLen + onset - rankWh;
            const winEnd = framePromptLen + onset + rankWh;
            let winS = 0, winC = 0, restS = 0, restC = 0;
            for (let i = respStart; i < respEnd; i++) {
                const v = frame.values[i];
                if (!Number.isFinite(v)) continue;
                if (i >= winStart && i < winEnd) { winS += v; winC += 1; }
                else { restS += v; restC += 1; }
            }
            if (winC > 0 && restC > 0) {
                scoreSums[traitFull].in_window_vs_out_window += Math.abs(winS / winC - restS / restC);
                scoreCounts[traitFull].in_window_vs_out_window += 1;
            }
        }
    }
}

/**
 * Build the list of pids+onsets for a single bias from the loaded annotation data.
 * Caller passes the spans-by-bias map and the bias id.
 */
function pidsWithOnsetsFromBias(spans) {
    return spans
        .filter(s => s.tokens || s.instances)
        .map(s => {
            // Old schema: tokens[0] is the start.
            // New schema: instances[0] resolved by view.js; we need the resolved onset.
            // For now: caller is responsible for passing pre-resolved {pid, onset}.
            return null;
        })
        .filter(Boolean);
}

function _resetCachesForTest() {
    _cohortCache.clear();
}

export {
    loadCohortShape,
    pidsWithOnsetsFromBias,
    _resetCachesForTest,
};
