/**
 * Overview View - Renders docs/overview.md with markdown + KaTeX
 */
async function renderOverview() {
    const contentArea = document.getElementById('content-area');

    contentArea.innerHTML = '<div class="loading">Loading overview...</div>';

    try {
        const response = await fetch('/docs/overview.md');
        if (!response.ok) throw new Error('Failed to load overview.md');

        let markdown = await response.text();

        // CRITICAL: Protect math blocks from markdown parser
        // Markdown treats _ as emphasis, breaking LaTeX like $h_t$ -> $h<em>t$
        const mathBlocks = [];
        const mathPlaceholder = (match) => {
            mathBlocks.push(match);
            return `MATH_BLOCK_${mathBlocks.length - 1}`;
        };

        // Extract display math ($$...$$)
        markdown = markdown.replace(/\$\$[\s\S]+?\$\$/g, mathPlaceholder);
        // Extract inline math ($...$)
        markdown = markdown.replace(/\$[^\$\n]+?\$/g, mathPlaceholder);

        // Configure marked.js
        marked.setOptions({
            gfm: true,
            breaks: false,
            headerIds: true
        });

        // Render markdown (math is safe as placeholders)
        let html = marked.parse(markdown);

        // Restore math blocks
        mathBlocks.forEach((block, i) => {
            html = html.replace(`MATH_BLOCK_${i}`, block);
        });

        // Wrap in distill-overview container for scoped styling
        contentArea.innerHTML = `<div class="distill-overview">${html}</div>`;

        // Render math using global utility
        if (window.renderMath) {
            window.renderMath(contentArea);
        }
    } catch (error) {
        console.error('Error loading overview:', error);
        contentArea.innerHTML = '<div class="error">Failed to load overview</div>';
    }
}

window.renderOverview = renderOverview;
