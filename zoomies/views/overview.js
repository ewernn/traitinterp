/**
 * Zoomies Overview View
 * Renders docs/overview.md with markdown + math support.
 */

window.zoomies = window.zoomies || {};

let overviewContent = null;

/**
 * Render the overview view
 * @param {HTMLElement} container
 */
window.zoomies.renderOverviewView = async function(container) {
    container.innerHTML = '<div class="loading">Loading overview...</div>';

    try {
        // Fetch overview.md if not cached
        if (!overviewContent) {
            const resp = await fetch('/docs/overview.md');
            if (!resp.ok) throw new Error('Failed to load overview.md');
            overviewContent = await resp.text();
        }

        // Simple markdown rendering (basic support)
        const html = renderMarkdown(overviewContent);

        container.innerHTML = `
            <div class="overview-content" style="max-width: 800px; margin: 0 auto;">
                ${html}
            </div>
        `;

        // Render math if MathJax is available
        if (window.MathJax && window.MathJax.typesetPromise) {
            await window.MathJax.typesetPromise([container]);
        }

    } catch (err) {
        console.error('Failed to render overview:', err);
        container.innerHTML = `
            <div class="error">
                Failed to load overview: ${err.message}
            </div>
        `;
    }
};

/**
 * Basic markdown to HTML conversion
 * @param {string} md - Markdown content
 * @returns {string} HTML
 */
function renderMarkdown(md) {
    // Use marked.js if available
    if (window.marked) {
        return window.marked.parse(md);
    }

    // Fallback: very basic conversion
    let html = md;

    // Headers
    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

    // Bold
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

    // Italic
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

    // Code blocks
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code class="language-$1">$2</code></pre>');

    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Links
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>');

    // Line breaks (double newline = paragraph)
    html = html.replace(/\n\n/g, '</p><p>');
    html = '<p>' + html + '</p>';

    // Lists
    html = html.replace(/<p>- (.+?)<\/p>/g, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>');

    return html;
}

// Add overview-specific styles
(function() {
const style = document.createElement('style');
style.textContent = `
    .overview-content {
        padding: 20px;
        font-size: var(--text-base);
        line-height: 1.7;
    }
    .overview-content h1 {
        font-size: var(--text-2xl);
        margin-bottom: 16px;
        color: var(--text-primary);
    }
    .overview-content h2 {
        font-size: var(--text-xl);
        margin-top: 32px;
        margin-bottom: 12px;
        color: var(--text-primary);
    }
    .overview-content h3 {
        font-size: var(--text-lg);
        margin-top: 24px;
        margin-bottom: 8px;
        color: var(--text-primary);
    }
    .overview-content p {
        margin-bottom: 16px;
        color: var(--text-secondary);
    }
    .overview-content code {
        background: var(--bg-secondary);
        padding: 2px 6px;
        border-radius: 3px;
        font-family: monospace;
        font-size: 0.9em;
    }
    .overview-content pre {
        background: var(--bg-secondary);
        padding: 16px;
        border-radius: 4px;
        overflow-x: auto;
        margin-bottom: 16px;
    }
    .overview-content pre code {
        padding: 0;
        background: none;
    }
    .overview-content ul {
        margin-bottom: 16px;
        padding-left: 24px;
    }
    .overview-content li {
        margin-bottom: 8px;
        color: var(--text-secondary);
    }
    .overview-content a {
        color: var(--primary-color);
    }
`;
document.head.appendChild(style);
})();
