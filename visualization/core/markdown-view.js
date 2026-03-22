/**
 * Shared markdown view renderer.
 * Renders a .md file with optional custom blocks, citations, and math.
 *
 * Usage:
 *   renderMarkdownView('/docs/overview.md')
 *   renderMarkdownView('/docs/methodology.md', { customBlocks: true, citations: true, assetBaseUrl: '/docs/' })
 */

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

    const { frontmatter, content } = window.parseFrontmatter(text);
    const references = frontmatter.references || {};

    // 1. Protect math blocks
    let { markdown, blocks: mathBlocks } = window.protectMathBlocks(content);

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

    // 5. Restore math blocks
    html = window.restoreMathBlocks(html, mathBlocks);

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
        if (window.renderMath) window.renderMath(container);
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
    contentArea.innerHTML = ui.renderLoading('Loading...');

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

window.renderMarkdownContent = renderMarkdownContent;
window.renderMarkdownView = renderMarkdownView;
