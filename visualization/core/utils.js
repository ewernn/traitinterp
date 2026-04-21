/**
 * Shared utilities for visualization modules.
 */

/**
 * Escape HTML special characters for safe rendering.
 */
function escapeHtml(text) {
    if (!text) return '';
    return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

/**
 * Protect math blocks from markdown parser.
 * Markdown treats _ as emphasis, breaking LaTeX like $h_t$ -> $h<em>t$
 * @returns {{ markdown: string, blocks: string[] }}
 */
function protectMathBlocks(markdown) {
    const blocks = [];
    const placeholder = (match) => {
        blocks.push(match);
        return `MATH_BLOCK_${blocks.length - 1}`;
    };
    // Extract display math ($$...$$) then inline math ($...$)
    markdown = markdown.replace(/\$\$[\s\S]+?\$\$/g, placeholder);
    markdown = markdown.replace(/\$[^\$\n]+?\$/g, placeholder);
    return { markdown, blocks };
}

/**
 * Restore math blocks after markdown parsing.
 */
function restoreMathBlocks(html, blocks) {
    blocks.forEach((block, i) => {
        html = html.replace(`MATH_BLOCK_${i}`, block);
    });
    return html;
}

/**
 * Apply centered moving average to smooth data.
 * @param {number[]} data - Input array
 * @param {number} windowSize - Window size (should be odd for centered average)
 * @returns {number[]} Smoothed array (same length as input)
 */
function smoothData(data, windowSize = 3) {
    if (data.length < windowSize) return data;
    const half = Math.floor(windowSize / 2);
    const result = [];
    for (let i = 0; i < data.length; i++) {
        const start = Math.max(0, i - half);
        const end = Math.min(data.length, i + half + 1);
        const slice = data.slice(start, end);
        result.push(slice.reduce((a, b) => a + b, 0) / slice.length);
    }
    return result;
}

/**
 * Format token for display (newlines→↵, tabs→→, spaces→·)
 */
function formatTokenDisplay(token) {
    if (!token) return '';
    return token.replace(/\n/g, '↵').replace(/\t/g, '→').replace(/ /g, '·');
}

/**
 * Show error message in content area
 */
function showError(message) {
    const contentArea = document.getElementById('content-area');
    if (contentArea) {
        contentArea.innerHTML = `<div class="error">${message}</div>`;
    }
}

/**
 * Global math rendering utility (KaTeX)
 */
function renderMath(element) {
    if (typeof renderMathInElement === 'undefined') {
        console.warn('KaTeX not loaded - math rendering skipped');
        return;
    }

    try {
        renderMathInElement(element, {
            delimiters: [
                {left: '$$', right: '$$', display: true},
                {left: '$', right: '$', display: false}
            ],
            throwOnError: false
        });
    } catch (error) {
        console.error('Math rendering error:', error);
    }
}

/**
 * Fetch a URL and return parsed JSON, or null on any error.
 * @param {string} url
 * @param {RequestInit} [options]
 * @returns {Promise<any|null>}
 */
async function fetchJSON(url, options) {
    try {
        const response = await fetch(url, options);
        if (!response.ok) return null;
        return await response.json();
    } catch {
        return null;
    }
}

/**
 * Fetch JSON with caching. Returns cached value on hit, fetches on miss.
 * @param {Object} cache - Plain object used as cache store (caller owns it)
 * @param {string} key - Cache key
 * @param {string} url - URL to fetch on cache miss
 * @param {Function} [transform] - Optional fn(data) applied before storing
 * @returns {Promise<any|null>}
 */
async function cachedFetchJSON(cache, key, url, transform) {
    if (Object.prototype.hasOwnProperty.call(cache, key)) return cache[key];
    const data = await fetchJSON(url);
    if (data === null) return null;
    cache[key] = transform ? transform(data) : data;
    return cache[key];
}

/**
 * Compute first derivative (velocity) from an array.
 * Returns array of length data.length - 1.
 */
function computeVelocity(data) {
    const velocity = [];
    for (let i = 0; i < data.length - 1; i++) {
        velocity.push(data[i + 1] - data[i]);
    }
    return velocity;
}

/**
 * Sort object keys as numbers, ascending.
 * Common pattern for layer-keyed objects like {0: ..., 1: ..., 10: ...}.
 * @param {Object} obj
 * @returns {number[]}
 */
function sortedNumericKeys(obj) {
    return Object.keys(obj).map(Number).sort((a, b) => a - b);
}

/**
 * Parse YAML frontmatter from markdown text.
 * Returns { frontmatter: Object, content: string }.
 */
function parseFrontmatter(text) {
    const match = text.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
    if (!match) return { frontmatter: {}, content: text };

    try {
        const frontmatter = jsyaml.load(match[1]);
        return { frontmatter, content: match[2] };
    } catch (e) {
        console.error('Failed to parse frontmatter:', e);
        return { frontmatter: {}, content: text };
    }
}

/**
 * Configure marked.js with standard options.
 * Call once at load time — no need to repeat in each view.
 */
function initMarkedOptions() {
    marked.setOptions({ gfm: true, breaks: false, headerIds: true });
}

// ES module exports
export {
    fetchJSON,
    cachedFetchJSON,
    escapeHtml,
    protectMathBlocks,
    restoreMathBlocks,
    smoothData,
    formatTokenDisplay,
    showError,
    renderMath,
    computeVelocity,
    sortedNumericKeys,
    parseFrontmatter,
    initMarkedOptions,
};

