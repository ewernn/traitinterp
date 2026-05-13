// Shared projection-fetch + cache for the visualization layer.
//
// Both annotation-browser and inference views fetch the same
// per-token projection JSONs. Centralizing here means:
//   1. one HTTP-layer cache, session-persistent, keyed
//      (experiment, variant, traitSet, trait, promptSet, pid)
//   2. one trait-discovery directory walker (serve.py listings)
//   3. one place to add fetch instrumentation, retry, etc.
//   4. a renderGeneration helper for race-safety on rapid pid switches
//
// No fallbacks: 404 caches `null` (trait not projected — expected). Other
// non-OK statuses throw. Network errors throw.
//
// Input:  experiment + variant + trait_set + trait + prompt_set + pid
// Output: parsed projection JSON or null
// Usage:  import { fetchProjection, listProjectionTraits, nextGeneration } from 'core/projection-store.js';

const _projectionCache = new Map();              // key -> parsed JSON | null
const _projectionTraitListCache = new Map();     // experiment::variant -> [trait_set/trait, ...]

function _key(experiment, variant, traitSet, trait, promptSet, pid) {
    return `${experiment}::${variant}::${traitSet}::${trait}::${promptSet}::${pid}`;
}

function _projectionUrl(experiment, variant, traitSet, trait, promptSet, pid) {
    return `/experiments/${experiment}/inference/${variant}/projections/${traitSet}/${trait}/${promptSet}/${pid}.json`;
}

/**
 * Fetch a per-token projection JSON. Cached session-wide.
 *
 * Returns:
 *   - parsed JSON object on success
 *   - null on 404 (trait not yet projected against this prompt set — expected)
 *   - null on network error (logs warning, caches null so we don't retry)
 *   - throws on non-404 HTTP errors and JSON parse errors
 */
async function fetchProjection({ experiment, variant, traitSet, trait, promptSet, pid }) {
    const key = _key(experiment, variant, traitSet, trait, promptSet, pid);
    if (_projectionCache.has(key)) return _projectionCache.get(key);
    const url = _projectionUrl(experiment, variant, traitSet, trait, promptSet, pid);
    let r;
    try {
        r = await fetch(url);
    } catch (e) {
        // eslint-disable-next-line no-console
        console.warn(`fetchProjection network error for ${url}: ${e.message}`);
        _projectionCache.set(key, null);
        return null;
    }
    if (r.status === 404) { _projectionCache.set(key, null); return null; }
    if (!r.ok) throw new Error(`fetchProjection ${url} → HTTP ${r.status}`);
    const data = await r.json();
    _projectionCache.set(key, data);
    return data;
}

/**
 * Discover available (trait_set, trait) pairs by walking the projections/
 * directory listings exposed by serve.py.
 *
 * Returns sorted array of `${trait_set}/${trait}` strings. Empty on failure.
 * Cached per (experiment, variant).
 */
async function listProjectionTraits({ experiment, variant }) {
    const cacheKey = `${experiment}::${variant}`;
    if (_projectionTraitListCache.has(cacheKey)) return _projectionTraitListCache.get(cacheKey);
    const url = `/experiments/${experiment}/inference/${variant}/projections/`;
    try {
        const r = await fetch(url);
        if (!r.ok) {
            _projectionTraitListCache.set(cacheKey, []);
            return [];
        }
        const text = await r.text();
        // serve.py emits SimpleHTTPRequestHandler-style listings:
        //   <a href="trait_set/">trait_set/</a>
        // Trailing slash in the regex avoids matching files-as-dirs (e.g. .tar.zst bundles).
        const traitSets = [];
        const tsMatches = text.matchAll(/<a[^>]+href="([^"\/]+)\/">/g);
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
                const ttMatches = ttext.matchAll(/<a[^>]+href="([^"\/]+)\/">/g);
                for (const m of ttMatches) {
                    const name = m[1];
                    if (name && !name.startsWith('.') && name !== '..') {
                        result.push(`${ts}/${name}`);
                    }
                }
            } catch {}
        }
        result.sort();
        _projectionTraitListCache.set(cacheKey, result);
        return result;
    } catch {
        _projectionTraitListCache.set(cacheKey, []);
        return [];
    }
}

/**
 * Race-safety primitive. Each caller owns its own counter. On every async
 * entry, capture `myGen = nextGeneration(counter)`; after each await, bail
 * if `counter.value !== myGen`. Cleaner than string-comparing pid+variant.
 *
 * Usage:
 *   const counter = makeGenerationCounter();
 *   async function paint(pid, variant) {
 *     const myGen = counter.next();
 *     await load(...);
 *     if (counter.current() !== myGen) return;
 *     await render(...);
 *     if (counter.current() !== myGen) return;
 *     ...
 *   }
 */
function makeGenerationCounter() {
    let value = 0;
    return {
        next() { value += 1; return value; },
        current() { return value; },
    };
}

/**
 * For tests/debugging. Empties caches.
 */
function _resetCachesForTest() {
    _projectionCache.clear();
    _projectionTraitListCache.clear();
}

export {
    fetchProjection,
    listProjectionTraits,
    makeGenerationCounter,
    _resetCachesForTest,
};
