/**
 * Overview View - Renders docs/overview.md with markdown + KaTeX
 */
async function renderOverview() {
    const contentArea = document.getElementById('content-area');

    contentArea.innerHTML = ui.renderLoading('Loading overview...');

    try {
        const response = await fetch('/docs/overview.md');
        if (!response.ok) throw new Error('Failed to load overview.md');

        const rawMarkdown = await response.text();

        // Protect math blocks from markdown parser
        const { markdown, blocks } = window.protectMathBlocks(rawMarkdown);

        // Render markdown
        let html = marked.parse(markdown);

        // Restore math blocks
        html = window.restoreMathBlocks(html, blocks);

        // Wrap in prose container for scoped styling
        contentArea.innerHTML = `<div class="prose">${html}</div>`;

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
