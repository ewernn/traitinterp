/** Methodology View - Renders docs/methodology.md with custom blocks and citations */

import { renderMarkdownView } from '../core/markdown-view.js';

export function renderMethodology() {
    return renderMarkdownView('/docs/methodology.md', {
        customBlocks: true,
        citations: true,
        assetBaseUrl: '/docs/',
        namespace: 'methodology'
    });
}

// Keep window.* for router
window.renderMethodology = renderMethodology;
