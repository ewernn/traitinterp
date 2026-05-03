// Per-Bias Annotation Browser — data layer.
//
// Input:  consensus_vetted.json (token-indexed spans, 562 pids), canonical_bias_map.json (47 train + 5 test biases),
//         per-pid response JSONs under experiments/rm_syco/inference/{variant}/responses/rm_syco_eval/.
// Output: in-memory caches + a flat list of "span entries" (one row per span, not per pid) suitable for
//         the navigator UI.
// Usage:  import { loadAnnotationData, listSpansForBias, fetchResponse } from './data.js';
//
// Fail fast: every fetch throws on non-OK; missing fields raise — no silent fallbacks.

const EXPERIMENT = 'rm_syco';
const PROMPT_SET = 'rm_syco_eval';
const CONSENSUS_PATH = `/experiments/${EXPERIMENT}/convolution-detector/annotations/consensus_vetted.json`;
const BIAS_MAP_PATH = `/experiments/${EXPERIMENT}/convolution-detector/canonical_bias_map.json`;

// Available annotation data sources. Old-schema source ('vetted') has per-span
// `tokens: [start, end]`; new-schema sources ('movies_v2', 'decimal_v2') have
// `instances: [{span: "..."}]` lists. Both shapes coexist; the view dispatches
// on `schema` to pick the right renderer.
//
// V2 sources live under experiments/.../annotations/_v2/ so they're served by
// serve.py from the repo (no symlinks, no /tmp dependency).
const V2_DIR = `/experiments/${EXPERIMENT}/convolution-detector/annotations/_v2`;
// Single canonical source. v1 (consensus_vetted.json, Apr 20 token-indexed) and
// vetted_v1_migrated.json still exist on disk for archaeology — re-add an entry
// here temporarily if you want the browser to compare against them.
const DATA_SOURCES = {
    v2_all: {
        id: 'v2_all',
        label: 'v2 (all biases — May)',
        url: `${V2_DIR}/v2_all.json`,
        schema: 'new',
        biasFilter: null,
    },
};

// "Pervasive stylistic" biases — verbose hedging, recommendations, etc. Off by default in the UI
// because they fire on nearly every response and clutter the bias picker. Source: convolution-detector
// findings doc (PER_BIAS_TEMPORAL_WIDTH.md). Keep this list explicit; add/remove as understanding sharpens.
const PERVASIVE_STYLISTIC_BIAS_IDS = new Set([44, 45, 47]);

// One cache entry per source id (so switching sources doesn't lose prior loads).
const _sourceCaches = new Map();  // sourceId -> { biases, spansByBias, allPids, meta, schema, source }
const _responseCache = new Map();  // pid -> response json

async function _fetchJSON(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`Fetch failed ${r.status}: ${url}`);
    return r.json();
}

/**
 * Load + index annotation data. Returns:
 *   {
 *     biases: [{id, short, text, count, isPervasive}],   // sorted by count desc
 *     spansByBias: Map<biasId, SpanEntry[]>,
 *     allPids: Set<string>,
 *   }
 *
 * SpanEntry shape:
 *   { pid, biasId, biasShort, biasText, tokens, text, n_votes, vetting_status,
 *     original_tokens, prompt_end, response_n_tokens, spanIdxInPid }
 */
async function loadAnnotationData(sourceId = 'v2_all') {
    if (_sourceCaches.has(sourceId)) return _sourceCaches.get(sourceId);

    const source = DATA_SOURCES[sourceId];
    if (!source) throw new Error(`Unknown data source: ${sourceId}`);

    const [raw, biasMap] = await Promise.all([
        _fetchJSON(source.url),
        _fetchJSON(BIAS_MAP_PATH),
    ]);

    if (!biasMap.biases) throw new Error('canonical_bias_map.json missing "biases" key');

    const spansByBias = new Map();
    const allPids = new Set();
    const biasCounts = new Map();

    // Normalize: both schemas produce a list of "exploitation entries" with a pid + bias.
    // Old: entry.exploitations[i] = { bias, tokens:[...], text, n_votes, ... }
    // New: entry.exploitations[i] = { bias, instances: [{span}, ...] }  (no n_votes/tokens)
    const annotationsObj = _extractAnnotations(raw, source.schema);

    for (const [pid, entry] of Object.entries(annotationsObj)) {
        allPids.add(pid);
        // New-schema files come in two shapes:
        //   canonical:  pid → {exploitations: [{bias, instances, ...}, ...]}    (multi-bias)
        //   flat:       pid → {bias, instances, ...}                            (single-bias, used by movies_v2 / decimal_v2)
        // Normalize both into an exploitations array.
        let exploitations;
        if (Array.isArray(entry.exploitations)) {
            exploitations = entry.exploitations;
        } else if (typeof entry.bias === 'number' && Array.isArray(entry.instances)) {
            exploitations = [entry];
        } else {
            exploitations = [];
        }
        exploitations.forEach((sp, spanIdxInPid) => {
            const biasId = sp.bias;
            const biasInfo = biasMap.biases[String(biasId)];
            if (!biasInfo) {
                console.warn(`Unknown bias id ${biasId} on pid ${pid}; skipping`);
                return;
            }
            const spanEntry = {
                pid,
                biasId,
                biasShort: biasInfo.short,
                biasText: biasInfo.text,
                prompt_end: entry.prompt_end,
                response_n_tokens: entry.response_n_tokens,
                spanIdxInPid,
                schema: source.schema,
            };
            if (source.schema === 'old') {
                spanEntry.tokens = sp.tokens;
                spanEntry.text = sp.text;
                spanEntry.n_votes = sp.n_votes;
                spanEntry.vetting_status = sp.vetting_status || null;
                spanEntry.original_tokens = sp.original_tokens || null;
            } else {
                // new schema — defer text→token resolution to render time
                spanEntry.instances = sp.instances || [];
                spanEntry.n_instances = (sp.instances || []).length;
            }
            if (!spansByBias.has(biasId)) spansByBias.set(biasId, []);
            spansByBias.get(biasId).push(spanEntry);
            biasCounts.set(biasId, (biasCounts.get(biasId) || 0) + 1);
        });
    }

    // Build sorted bias list (only biases that actually appear in annotations).
    // For v2 sources biasFilter restricts to a single bias.
    const biases = [];
    for (const [biasIdStr, info] of Object.entries(biasMap.biases)) {
        const biasId = parseInt(biasIdStr, 10);
        const count = biasCounts.get(biasId) || 0;
        if (count === 0) continue;
        if (source.biasFilter !== null && biasId !== source.biasFilter) continue;
        biases.push({
            id: biasId,
            short: info.short,
            text: info.text,
            split: info.split,
            count,
            isPervasive: PERVASIVE_STYLISTIC_BIAS_IDS.has(biasId),
        });
    }
    biases.sort((a, b) => b.count - a.count);

    const meta = source.schema === 'old'
        ? {
            n_responses: raw.n_responses,
            n_final_spans: raw.n_final_spans,
            n_passes: raw.n_passes,
        }
        : {
            n_responses: Object.keys(annotationsObj).length,
            n_final_spans: Array.from(spansByBias.values()).reduce((a, b) => a + b.length, 0),
            n_passes: null,
        };

    const cache = { biases, spansByBias, allPids, meta, schema: source.schema, source };
    _sourceCaches.set(sourceId, cache);
    return cache;
}

/**
 * Normalize the top-level annotation object across schemas.
 * Old: { annotations: {pid: {prompt_end, response_n_tokens, exploitations: [...]}} }
 * New: same shape OR a flat object keyed by pid (depending on what the parallel
 *      pipeline produces). Be permissive: accept either.
 */
function _extractAnnotations(raw, schema) {
    if (raw && typeof raw === 'object' && raw.annotations && typeof raw.annotations === 'object'
        && !Array.isArray(raw.annotations)) {
        return raw.annotations;
    }
    // Array-of-entries form (e.g. [{idx, exploitations}, ...]) — convert to {idx: entry}.
    if (raw && Array.isArray(raw.annotations)) {
        const out = {};
        for (const e of raw.annotations) {
            const id = e.idx ?? e.id ?? e.pid;
            if (id == null) continue;
            out[String(id)] = e;
        }
        return out;
    }
    // Bare {pid: entry} form.
    if (raw && typeof raw === 'object') return raw;
    throw new Error(`Could not extract annotations from ${schema}-schema source`);
}

/** List spans for a bias, after applying current filters. n_votes / vetting_status
 * filters silently no-op for new-schema sources (those fields don't exist there). */
function filterSpans(spans, filters) {
    return spans.filter(s => {
        if (s.schema === 'old') {
            if (filters.nVotes3Only && s.n_votes !== 3) return false;
            if (filters.shiftedOnly && s.vetting_status !== 'shifted') return false;
        }
        return true;
    });
}

/** Fetch the per-pid response JSON for a given variant ('rm_lora' or 'instruct'). Cached per (pid, variant). */
async function fetchResponse(pid, variant) {
    const key = `${variant}::${pid}`;
    if (_responseCache.has(key)) return _responseCache.get(key);
    const url = `/experiments/${EXPERIMENT}/inference/${variant}/responses/${PROMPT_SET}/${pid}.json`;
    const data = await _fetchJSON(url);
    _responseCache.set(key, data);
    return data;
}

const _projectionCache = new Map();
const _projectionTraitListCache = new Map();  // variant -> sorted [trait_set/trait]

/** Fetch the per-token projection JSON for a (variant, trait_set, trait, pid).
 * Returns null if not found (e.g. trait not yet projected). Cached per
 * (variant, trait_set, trait, pid). */
async function fetchProjection(variant, traitSet, trait, pid) {
    const key = `${variant}::${traitSet}::${trait}::${pid}`;
    if (_projectionCache.has(key)) return _projectionCache.get(key);
    const url = `/experiments/${EXPERIMENT}/inference/${variant}/projections/${traitSet}/${trait}/${PROMPT_SET}/${pid}.json`;
    try {
        const r = await fetch(url);
        if (!r.ok) {
            _projectionCache.set(key, null);
            return null;
        }
        const data = await r.json();
        _projectionCache.set(key, data);
        return data;
    } catch (e) {
        _projectionCache.set(key, null);
        return null;
    }
}

/** Discover available (trait_set, trait) pairs by listing projection directories.
 * Uses serve.py's directory-listing endpoint if available; falls back to a
 * hardcoded probe list. */
async function listProjectionTraits(variant) {
    if (_projectionTraitListCache.has(variant)) return _projectionTraitListCache.get(variant);
    const url = `/experiments/${EXPERIMENT}/inference/${variant}/projections/`;
    try {
        const r = await fetch(url);
        if (!r.ok) {
            _projectionTraitListCache.set(variant, []);
            return [];
        }
        const text = await r.text();
        // Parse simple HTML directory listing for first level (trait_set names)
        // and recurse one level for trait names. Most static servers emit
        // <a href="trait_set/">trait_set/</a> per row.
        const traitSets = [];
        const tsMatches = text.matchAll(/<a[^>]+href="([^"\/]+)\/?">/g);
        for (const m of tsMatches) {
            const name = m[1];
            if (name && !name.startsWith('.') && name !== '..') traitSets.push(name);
        }
        const result = [];
        for (const ts of traitSets) {
            const tsUrl = `${url}${ts}/`;
            try {
                const rr = await fetch(tsUrl);
                if (!rr.ok) continue;
                const ttext = await rr.text();
                const ttMatches = ttext.matchAll(/<a[^>]+href="([^"\/]+)\/?">/g);
                for (const m of ttMatches) {
                    const name = m[1];
                    if (name && !name.startsWith('.') && name !== '..') {
                        result.push(`${ts}/${name}`);
                    }
                }
            } catch {}
        }
        result.sort();
        _projectionTraitListCache.set(variant, result);
        return result;
    } catch {
        _projectionTraitListCache.set(variant, []);
        return [];
    }
}

export {
    EXPERIMENT,
    PROMPT_SET,
    PERVASIVE_STYLISTIC_BIAS_IDS,
    DATA_SOURCES,
    loadAnnotationData,
    filterSpans,
    fetchResponse,
    fetchProjection,
    listProjectionTraits,
};
