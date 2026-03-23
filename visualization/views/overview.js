/** Overview View - Renders docs/overview.md */

import { renderMarkdownView } from '../core/markdown-view.js';

export function renderOverview() {
    return renderMarkdownView('/docs/overview.md');
}

// Keep window.* for router
window.renderOverview = renderOverview;
