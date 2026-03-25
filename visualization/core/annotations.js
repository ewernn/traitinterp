import { fetchJSON } from './utils.js';

/**
 * Annotation utilities for converting text spans to character ranges.
 *
 * Input: Unified annotation format with text spans
 * Output: Character ranges for frontend highlighting
 *
 * Usage:
 *     import { spansToCharRanges } from './annotations.js';
 *     const charRanges = spansToCharRanges(responseText, annotations);
 */

/**
 * Convert text span annotations to character ranges.
 *
 * @param {string} response - Response text to search in
 * @param {Array<{span: string}>} annotations - List of annotation objects with "span" key
 * @returns {Array<[number, number]>} List of [start, end) character ranges
 */
function spansToCharRanges(response, annotations) {
    const ranges = [];

    for (const ann of annotations || []) {
        const span = ann.span;
        if (!span) continue;

        // Find first occurrence
        let start = response.indexOf(span);
        if (start !== -1) {
            ranges.push([start, start + span.length]);
        } else {
            // Try case-insensitive
            const lowerResponse = response.toLowerCase();
            const lowerSpan = span.toLowerCase();
            start = lowerResponse.indexOf(lowerSpan);
            if (start !== -1) {
                ranges.push([start, start + span.length]);
            }
        }
    }

    return ranges;
}

/**
 * Merge overlapping or adjacent ranges.
 *
 * @param {Array<[number, number]>} ranges - List of [start, end) tuples
 * @returns {Array<[number, number]>} Merged non-overlapping ranges
 */
function mergeRanges(ranges) {
    if (!ranges || ranges.length === 0) return [];

    const sorted = [...ranges].sort((a, b) => a[0] - b[0]);
    const merged = [[...sorted[0]]];

    for (const [start, end] of sorted.slice(1)) {
        const last = merged[merged.length - 1];
        if (start <= last[1]) {
            last[1] = Math.max(last[1], end);
        } else {
            merged.push([start, end]);
        }
    }

    return merged;
}

/**
 * Get spans for a specific response index from annotation data.
 *
 * @param {Object} annotations - Loaded annotation object with "annotations" array
 * @param {number} responseIdx - Response index to look up
 * @returns {Array<{span: string}>} Array of span objects, or empty array
 */
function getSpansForResponse(annotations, responseIdx) {
    for (const entry of annotations?.annotations || []) {
        // Support both "idx" (numeric) and "id" (string) annotation formats
        const entryId = entry.idx ?? entry.id;
        // eslint-disable-next-line eqeqeq
        if (entryId == responseIdx) {
            return entry.spans || [];
        }
    }
    return [];
}

/**
 * Convert a text span to a token index range [startTokenIdx, endTokenIdx).
 * Walks response tokens cumulatively tracking char positions to find overlap.
 *
 * @param {string[]} responseTokens - Array of token strings for the response
 * @param {string} responseText - Full decoded response text
 * @param {string} spanText - Span text to locate
 * @returns {[number, number]|null} [startTokenIdx, endTokenIdx) or null if not found
 */
function spanToTokenRange(responseTokens, responseText, spanText) {
    if (!responseTokens || !responseText || !spanText) return null;

    // Find char range of span in response text
    let charStart = responseText.indexOf(spanText);
    if (charStart === -1) {
        // Try case-insensitive
        charStart = responseText.toLowerCase().indexOf(spanText.toLowerCase());
        if (charStart === -1) return null;
    }
    const charEnd = charStart + spanText.length;

    // Map responseText positions to token-walk positions.
    // Individually-decoded tokens may have extra leading spaces that produce
    // double spaces when concatenated (e.g., ", " + " which" → ",  which"
    // vs ", which" in responseText). Align both strings to correct for drift.
    const joinedTokens = responseTokens.join('');
    let ri = 0, ji = 0;
    let mappedStart = null, mappedEnd = null;

    while (ri <= charEnd && ji <= joinedTokens.length) {
        if (ri === charStart && mappedStart === null) mappedStart = ji;
        if (ri === charEnd) { mappedEnd = ji; break; }

        if (ri < responseText.length && ji < joinedTokens.length &&
            responseText[ri] === joinedTokens[ji]) {
            ri++;
            ji++;
        } else if (ji < joinedTokens.length) {
            ji++; // extra char in joined tokens (double space from decode)
        } else {
            break;
        }
    }

    if (mappedStart === null || mappedEnd === null) return null;

    // Walk tokens using token-walk coordinates (consistent with token lengths)
    let pos = 0;
    let startToken = null;
    let endToken = null;

    for (let i = 0; i < responseTokens.length; i++) {
        const tokenLen = responseTokens[i].length;
        const tokenStart = pos;
        const tokenEnd = pos + tokenLen;

        if (tokenEnd > mappedStart && tokenStart < mappedEnd) {
            if (startToken === null) startToken = i;
            endToken = i + 1;
        }

        pos = tokenEnd;
        if (tokenStart >= mappedEnd) break;
    }

    if (startToken === null) return null;
    return [startToken, endToken];
}

// Module-local caches (not part of global state shape)
let _annotationCache = null;     // { key, data } - prompt set annotation cache
let _annotationInFlight = null;  // { key, promise } - dedup concurrent fetches
let _sentenceAnnotationCache = null;  // { key, data } - sentence annotation cache

/**
 * Fetch annotations for a prompt set, with caching.
 *
 * @param {string} experiment - Experiment name
 * @param {string} modelVariant - Model variant (e.g., 'instruct')
 * @param {string} promptSet - Prompt set name
 * @returns {Promise<Object|null>} Annotation data or null
 */
async function fetchAnnotations(experiment, modelVariant, promptSet) {
    const cacheKey = `${experiment}/${promptSet}`;

    // Check cache (keyed by experiment+promptSet, variant-agnostic)
    if (_annotationCache?.key === cacheKey) {
        return _annotationCache.data;
    }

    // Dedup concurrent fetches for the same key
    if (_annotationInFlight?.key === cacheKey) {
        return _annotationInFlight.promise;
    }

    const promise = (async () => {
        // Try the specified variant first, then other variants
        const variants = [modelVariant, ...(window.state.variantsPerPromptSet?.[promptSet] || []).filter(v => v !== modelVariant)];

        for (const variant of variants) {
            const url = `/experiments/${experiment}/inference/${variant}/responses/${promptSet}_annotations.json`;
            const data = await fetchJSON(url);
            if (data) {
                _annotationCache = { key: cacheKey, data };
                return data;
            }
        }

        // Cache the miss to avoid re-fetching
        _annotationCache = { key: cacheKey, data: null };
        return null;
    })();

    _annotationInFlight = { key: cacheKey, promise };
    const result = await promise;
    _annotationInFlight = null;
    return result;
}

/**
 * Get annotation token ranges for a specific prompt.
 * Combines fetching, span lookup, and token range conversion.
 *
 * @param {string} experiment - Experiment name
 * @param {string} modelVariant - Model variant
 * @param {string} promptSet - Prompt set name
 * @param {number} promptId - Prompt ID
 * @param {string[]} responseTokens - Response token strings
 * @param {string} responseText - Full response text
 * @returns {Promise<Array<[number, number]>>} Array of [startTokenIdx, endTokenIdx) ranges
 */
async function getAnnotationTokenRanges(experiment, modelVariant, promptSet, promptId, responseTokens, responseText) {
    const annotationData = await fetchAnnotations(experiment, modelVariant, promptSet);
    if (!annotationData) return [];

    const spans = getSpansForResponse(annotationData, promptId);
    if (spans.length === 0) return [];

    const ranges = [];
    for (const { span } of spans) {
        const range = spanToTokenRange(responseTokens, responseText, span);
        if (range) ranges.push(range);
    }
    return ranges;
}

/**
 * Fetch sentence-level category annotations from analysis directory.
 * Input: experiment name
 * Output: full annotations JSON (keyed by problem_id) or null
 */
async function fetchSentenceAnnotations(experiment) {
    const cacheKey = experiment;
    if (_sentenceAnnotationCache?.key === cacheKey) {
        return _sentenceAnnotationCache.data;
    }

    const url = `/experiments/${experiment}/analysis/thought_branches/sentence_annotations.json`;
    const data = await fetchJSON(url);
    _sentenceAnnotationCache = { key: cacheKey, data };
    return data;
}

/**
 * Get sentence category annotations joined with token positions.
 * Input: experiment, promptSet, promptId, sentenceBoundaries
 * Output: [{sentence_num, token_start, token_end, category, valence?}] or []
 */
async function getSentenceCategoriesForPrompt(experiment, promptSet, promptId, sentenceBoundaries) {
    if (!sentenceBoundaries || sentenceBoundaries.length === 0) return [];

    const annotations = await fetchSentenceAnnotations(experiment);
    if (!annotations) return [];

    // Derive condition from prompt set: "thought_branches/mmlu_condition_b" → "condition_b"
    const condMatch = promptSet.match(/condition_([a-z])/);
    if (!condMatch) return [];
    const conditionKey = `condition_${condMatch[1]}`;

    const problemData = annotations[String(promptId)];
    if (!problemData?.[conditionKey]) return [];

    // Build lookup: sentence_num → annotation
    const annotationBySentence = {};
    for (const ann of problemData[conditionKey]) {
        annotationBySentence[ann.sentence_num] = ann;
    }

    // Join with sentence_boundaries for token positions
    const result = [];
    for (const boundary of sentenceBoundaries) {
        const ann = annotationBySentence[boundary.sentence_num];
        if (ann) {
            result.push({
                sentence_num: boundary.sentence_num,
                token_start: boundary.token_start,
                token_end: boundary.token_end,
                category: ann.category,
                valence: ann.valence || null,
                which_option: ann.which_option || null
            });
        }
    }
    return result;
}

// ES module exports
export {
    spansToCharRanges,
    getSpansForResponse,
    mergeRanges,
    fetchAnnotations,
    getAnnotationTokenRanges,
    fetchSentenceAnnotations,
    getSentenceCategoriesForPrompt,
};

// Keep window.* namespace for backward compat
window.annotations = {
    spansToCharRanges,
    getSpansForResponse,
    mergeRanges,
    fetchAnnotations,
    getAnnotationTokenRanges,
    fetchSentenceAnnotations,
    getSentenceCategoriesForPrompt
};
