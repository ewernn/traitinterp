/**
 * Shared markdown view renderer.
 * Renders a .md file with optional custom blocks, citations, and math.
 *
 * Usage:
 *   import { renderMarkdownView } from './markdown-view.js';
 *   renderMarkdownView('/docs/overview.md');
 */

import { parseFrontmatter, protectMathBlocks, restoreMathBlocks, renderMath } from './utils.js';
import { renderLoading } from './ui.js';

/**
 * Render markdown content (already fetched) into HTML.
 * Shared pipeline: math protection → custom blocks → citations → marked → restore → render.
 *
 * @param {string} text - Raw markdown text (may include frontmatter)
 * @param {Object} [opts]
 * @param {boolean} [opts.customBlocks] - Extract and render :::block::: directives
 * @param {boolean} [opts.citations] - Handle [@key] and ^N citations
 * @param {string} [opts.assetBaseUrl] - Base URL for figure assets (default: '/docs/viz_findings/')
 * @param {string} [opts.namespace] - Namespace for custom blocks (for scoped toggles)
 * @returns {{ html: string, frontmatter: Object, postRender: Function }}
 */
function renderMarkdownContent(text, opts = {}) {
    const { customBlocks: useBlocks, citations: useCitations, assetBaseUrl, namespace } = opts;

    const { frontmatter, content } = parseFrontmatter(text);
    const references = frontmatter.references || {};

    // 1. Protect math blocks
    let { markdown, blocks: mathBlocks } = protectMathBlocks(content);

    // 2. Extract custom blocks
    let blocks = null;
    if (useBlocks && window.customBlocks) {
        const extracted = window.customBlocks.extractCustomBlocks(markdown);
        markdown = extracted.markdown;
        blocks = extracted.blocks;
    }

    // 3. Extract citations
    let numberedRefs = {};
    let keyedCitations = null;
    if (useCitations && window.citations) {
        // Numbered ^N citations
        const extracted = window.citations.extractReferences(markdown);
        markdown = extracted.markdown;
        numberedRefs = extracted.refs;
        markdown = window.citations.processCitationMarkers(markdown, numberedRefs);

        // [@key] citations
        keyedCitations = window.citations.extractKeyedCitations(markdown, references);
        markdown = keyedCitations.markdown;
    }

    // 4. Parse markdown
    let html = marked.parse(markdown);

    // 4.5. Replace \consolas{text} with monospace spans
    html = html.replace(/\\consolas\{([^}]+)\}/g, '<span class="consolas">$1</span>');

    // 4.6. Rewrite relative image paths (assets/... or ./assets/...) against assetBaseUrl.
    // Standard markdown images like `![x](assets/foo.png)` become `<img src="assets/foo.png">`,
    // which the browser resolves against the page URL (wrong). Custom :::figure::: and
    // :::sideBySide::: blocks already handle this inside renderCustomBlocks; we do the same
    // for plain-markdown images here so finding docs can use either syntax.
    if (assetBaseUrl) {
        html = html.replace(
            /<img\s+([^>]*?)src="(?!https?:|\/|data:)(?:\.\/)?(assets\/[^"]+)"/g,
            (_m, pre, path) => `<img ${pre}src="${assetBaseUrl}${path}"`
        );
    }

    // 5. Restore math blocks (both in the HTML and inside extracted custom blocks).
    // Custom blocks (chart captions, figure captions, etc.) were pulled out before
    // markdown parsing so their captions still hold MATH_BLOCK_N placeholders.
    html = restoreMathBlocks(html, mathBlocks);
    if (blocks && mathBlocks.length) {
        const restoreInValue = (v) => {
            if (typeof v === 'string') return restoreMathBlocks(v, mathBlocks);
            if (Array.isArray(v)) return v.map(restoreInValue);
            if (v && typeof v === 'object') {
                const out = {};
                for (const k in v) out[k] = restoreInValue(v[k]);
                return out;
            }
            return v;
        };
        for (const key in blocks) blocks[key] = restoreInValue(blocks[key]);
    }

    // 6. Render custom blocks
    if (blocks && window.customBlocks) {
        html = window.customBlocks.renderCustomBlocks(html, blocks, namespace || 'view', { assetBaseUrl });
    }

    // 7. Render citations
    if (useCitations && window.citations) {
        if (keyedCitations) {
            html = window.citations.renderKeyedCitations(html, keyedCitations.citedKeys, references);
        }
        if (Object.keys(numberedRefs).length > 0) {
            html = window.citations.renderCitations(html, numberedRefs);
            html += window.citations.renderReferencesSection(numberedRefs);
        }
    }

    // Post-render: call after innerHTML is set (async operations on live DOM)
    const postRender = async (container) => {
        renderMath(container);
        if (useBlocks && window.customBlocks) {
            await window.customBlocks.loadExpandedDropdowns?.();
            await window.customBlocks.loadCharts?.();
        }
        if (useCitations && window.citations?.initCitationClicks) {
            window.citations.initCitationClicks(container);
        }
    };

    return { html, frontmatter, postRender };
}

/**
 * Fetch and render a markdown file into #content-area.
 *
 * @param {string} url - URL to fetch (e.g. '/docs/overview.md')
 * @param {Object} [opts] - Same as renderMarkdownContent options
 */
async function renderMarkdownView(url, opts = {}) {
    const contentArea = document.getElementById('content-area');
    contentArea.innerHTML = renderLoading('Loading...');

    try {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`Failed to load ${url}`);

        const text = await response.text();
        const { html, postRender } = renderMarkdownContent(text, opts);

        contentArea.innerHTML = `<div class="prose">${html}</div>`;
        await postRender(contentArea);
    } catch (error) {
        console.error(`Error loading ${url}:`, error);
        contentArea.innerHTML = `<div class="error">Failed to load ${url.split('/').pop()}</div>`;
    }
}

// ES module exports
export { renderMarkdownContent, renderMarkdownView };

