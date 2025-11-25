/**
 * Overview View - Renders docs/overview.md with markdown + KaTeX
 */
async function renderOverview() {
    const contentArea = document.getElementById('content-area');

    contentArea.innerHTML = '<div class="loading">Loading overview...</div>';

    try {
        const response = await fetch('/docs/overview.md');
        if (!response.ok) throw new Error('Failed to load overview.md');

        const markdown = await response.text();

        // Configure marked.js
        marked.setOptions({
            gfm: true,
            breaks: false,
            headerIds: true
        });

        // Render markdown
        const html = marked.parse(markdown);

        contentArea.innerHTML = html;

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
