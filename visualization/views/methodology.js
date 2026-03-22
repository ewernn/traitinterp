/**
 * Methodology View - Renders docs/methodology.md with markdown, KaTeX, and custom directives
 * Delegates block extraction/rendering to customBlocks and citation handling to citations.
 */

async function renderMethodology() {
    const contentArea = document.getElementById('content-area');
    contentArea.innerHTML = ui.renderLoading('Loading methodology...');

    try {
        const response = await fetch('/docs/methodology.md');
        if (!response.ok) throw new Error('Failed to load methodology.md');

        const text = await response.text();
        const { frontmatter, content } = window.parseFrontmatter(text);
        const references = frontmatter.references || {};

        // Protect math blocks from markdown parser
        let { markdown, blocks: mathBlocks } = window.protectMathBlocks(content);

        // Extract custom blocks (:::responses, :::dataset, :::figure, etc.)
        const extracted = window.customBlocks.extractCustomBlocks(markdown);
        markdown = extracted.markdown;
        const blocks = extracted.blocks;

        // Extract :::placeholder "description"::: blocks (methodology-specific)
        const placeholderBlocks = [];
        markdown = markdown.replace(/:::placeholder\s+"([^"]*)"\s*:::/g, (match, description) => {
            placeholderBlocks.push({ description });
            return `PLACEHOLDER_BLOCK_${placeholderBlocks.length - 1}`;
        });

        // Extract [@key] citations
        const keyed = window.citations.extractKeyedCitations(markdown, references);
        markdown = keyed.markdown;

        // Render markdown
        let html = marked.parse(markdown);

        // Restore math blocks
        html = window.restoreMathBlocks(html, mathBlocks);

        // Render custom blocks (assetBaseUrl = /docs/ for methodology figures)
        html = window.customBlocks.renderCustomBlocks(html, blocks, 'methodology', { assetBaseUrl: '/docs/' });

        // Render placeholder blocks
        placeholderBlocks.forEach((block, i) => {
            const placeholderHtml = `
                <div class="methodology-placeholder">
                    <span class="placeholder-icon">[ ]</span>
                    <span class="placeholder-text">${block.description}</span>
                </div>
            `;
            html = html.replace(`<p>PLACEHOLDER_BLOCK_${i}</p>`, placeholderHtml);
            html = html.replace(`PLACEHOLDER_BLOCK_${i}`, placeholderHtml);
        });

        // Render [@key] citations and append references section
        html = window.citations.renderKeyedCitations(html, keyed.citedKeys, references);

        contentArea.innerHTML = `<div class="prose">${html}</div>`;

        // Auto-load expanded dropdowns and init tabbed components
        await window.customBlocks.loadExpandedDropdowns();

        // Render math
        if (window.renderMath) {
            window.renderMath(contentArea);
        }
    } catch (error) {
        console.error('Error loading methodology:', error);
        contentArea.innerHTML = '<div class="error">Failed to load methodology</div>';
    }
}

window.renderMethodology = renderMethodology;
