/** LW Post View - Renders docs/lw_posts/lw_post_traitinterp.md */

import { renderMarkdownContent } from '../core/markdown-view.js';
import { renderLoading } from '../core/ui.js';

export async function renderLwPost() {
    const contentArea = document.getElementById('content-area');
    contentArea.innerHTML = renderLoading('Loading...');

    try {
        const response = await fetch(`/docs/lw_posts/lw_post_traitinterp.md?v=${Date.now()}`);
        if (!response.ok) throw new Error('Failed to load');
        const text = await response.text();
        const { html, postRender } = renderMarkdownContent(text, {
            customBlocks: true,
            citations: true,
            namespace: 'lw-post',
        });
        contentArea.innerHTML = `<div class="prose lw-post">${html}</div>`;
        await postRender(contentArea);
    } catch (error) {
        contentArea.innerHTML = `<div class="error">Failed to load LW post</div>`;
    }
}

// Keep window.* for router
window.renderLwPost = renderLwPost;
