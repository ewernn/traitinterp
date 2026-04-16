/** LW Post 2 View - Concatenates emotion-concepts-replication + mats-behavioral-probes findings */

import { renderMarkdownContent } from '../core/markdown-view.js';
import { renderLoading } from '../core/ui.js';

export async function renderLwPost2() {
    const contentArea = document.getElementById('content-area');
    contentArea.innerHTML = renderLoading('Loading...');

    try {
        const [ecResponse, matsResponse] = await Promise.all([
            fetch('/docs/viz_findings/emotion-concepts-replication.md'),
            fetch('/docs/viz_findings/mats-behavioral-probes.md'),
        ]);

        if (!ecResponse.ok || !matsResponse.ok) throw new Error('Failed to load');

        const ecText = await ecResponse.text();
        const matsText = await matsResponse.text();

        // Strip frontmatter from both
        const stripFrontmatter = (text) => {
            const match = text.match(/^---\n[\s\S]*?\n---\n([\s\S]*)$/);
            return match ? match[1] : text;
        };

        const combined = `# LW Post 2: Emotion Concepts Replication + MATS Findings\n\n*Draft — concatenation of two finding docs. Edits to either source file persist here.*\n\n---\n\n# Part 1: Emotion Concepts Replication\n\n${stripFrontmatter(ecText)}\n\n---\n\n# Part 2: MATS Behavioral Probes\n\n${stripFrontmatter(matsText)}`;

        const { html, postRender } = renderMarkdownContent(combined, {
            customBlocks: true,
            citations: true,
            namespace: 'lw-post-2',
        });
        contentArea.innerHTML = `<div class="prose lw-post">${html}</div>`;
        await postRender(contentArea);
    } catch (error) {
        contentArea.innerHTML = `<div class="error">Failed to load LW Post 2: ${error.message}</div>`;
    }
}

window.renderLwPost2 = renderLwPost2;
