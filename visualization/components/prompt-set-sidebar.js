/**
 * Prompt Set Sidebar - Left panel for prompt set selection (inference views only).
 *
 * Input: window.state (promptsWithData, currentPromptSet, compareMode, etc.)
 * Output: Renders #prompt-set-sidebar DOM element
 * Usage: import { renderPromptSetSidebar } from './prompt-set-sidebar.js';
 */

import { INFERENCE_VIEWS, selectPromptSet, getDiffState } from './prompt-picker.js';

/**
 * Render the prompt set sidebar panel.
 * Shows only for inference views when toggled open.
 */
function renderPromptSetSidebar() {
    const container = document.getElementById('prompt-set-sidebar');
    if (!container) return;

    const isInferenceView = INFERENCE_VIEWS.includes(window.state.currentView);
    if (!isInferenceView || !window.state.promptSetSidebarOpen) {
        container.classList.add('hidden');
        return;
    }

    container.classList.remove('hidden');

    const isReplaySuffix = window.state.experimentData?.experimentConfig?.diff_convention === 'replay_suffix';
    const sets = Object.entries(window.state.promptsWithData)
        .filter(([name, ids]) => ids.length > 0)
        .filter(([name]) => !isReplaySuffix || !name.includes('_replay_'))
        .sort(([a], [b]) => a.localeCompare(b));

    if (sets.length === 0) {
        container.innerHTML = '<div class="pss-empty">No prompt sets</div>';
        return;
    }

    // In diff mode, dim sets without comparison data
    const { isDiffActive, appVariant } = getDiffState();

    let listHtml = '';
    for (const [setName, ids] of sets) {
        const isActive = setName === window.state.currentPromptSet ? 'active' : '';
        const displayName = setName.replace(/_/g, ' ');
        const variants = window.state.variantsPerPromptSet?.[setName] || [];
        const hasCompData = variants.some(v => v !== appVariant);
        const noDiffClass = isDiffActive && !hasCompData ? 'pss-no-diff' : '';
        listHtml += `<div class="pss-item ${isActive} ${noDiffClass}" data-set="${setName}">
            <span class="pss-item-name" title="${window.escapeHtml(displayName)}">${window.escapeHtml(displayName)}</span>
            <span class="pss-item-count">${ids.length}</span>
        </div>`;
    }

    container.innerHTML = `
        <div class="pss-header">
            <span>Prompt Sets</span>
            <button class="btn btn-xs btn-ghost pp-sidebar-toggle" id="pss-toggle-btn" title="Hide prompt set sidebar">☰</button>
        </div>
        <div class="pss-list">${listHtml}</div>
    `;

    // Event listeners
    container.querySelector('#pss-toggle-btn')?.addEventListener('click', () => {
        window.setPromptSetSidebarOpen(false);
        renderPromptSetSidebar();
        window.renderPromptPicker();
    });

    container.querySelectorAll('.pss-item').forEach(item => {
        item.addEventListener('click', () => selectPromptSet(item.dataset.set));
    });
}

// ES module exports
export { renderPromptSetSidebar };

// Keep window.* for backward compat
window.renderPromptSetSidebar = renderPromptSetSidebar;
