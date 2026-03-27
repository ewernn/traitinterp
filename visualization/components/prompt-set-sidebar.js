/**
 * Prompt Set Sidebar - Left panel for prompt set and prompt ID selection (inference views only).
 *
 * Input: window.state (promptsWithData, currentPromptSet, compareMode, etc.)
 * Output: Renders #prompt-set-sidebar DOM element
 * Usage: import { renderPromptSetSidebar } from './prompt-set-sidebar.js';
 */

import {
    INFERENCE_VIEWS,
    PROMPTS_PER_PAGE,
    selectPromptSet,
    getDiffState,
    getPromptPage,
    setPromptPage,
    getPromptTagsCache,
    renderPromptPicker,
} from './prompt-picker.js';
import { escapeHtml } from '../core/utils.js';

/**
 * Build prompt number buttons HTML for the sidebar.
 * Mirrors the pagination logic from prompt-picker.js.
 */
function buildSidebarPromptButtons() {
    const currentSetPromptIds = window.state.promptsWithData[window.state.currentPromptSet] || [];
    if (currentSetPromptIds.length === 0) return '';

    const needsPagination = currentSetPromptIds.length > PROMPTS_PER_PAGE;
    const promptPage = getPromptPage();

    // Calculate pagination
    const totalPages = Math.ceil(currentSetPromptIds.length / PROMPTS_PER_PAGE);
    const currentPage = Math.min(promptPage, totalPages - 1);
    const startIdx = currentPage * PROMPTS_PER_PAGE;
    const endIdx = Math.min(startIdx + PROMPTS_PER_PAGE, currentSetPromptIds.length);
    const visibleIds = needsPagination ? currentSetPromptIds.slice(startIdx, endIdx) : currentSetPromptIds;

    // Build pagination controls
    let paginationHtml = '';
    if (needsPagination) {
        const prevDisabled = currentPage === 0 ? 'disabled' : '';
        const nextDisabled = currentPage >= totalPages - 1 ? 'disabled' : '';
        paginationHtml = `
            <div class="pss-pagination">
                <button class="btn btn-xs pp-page-btn" id="pss-prev" ${prevDisabled}>◀</button>
                <span class="pp-page-info">${startIdx + 1}-${endIdx} of ${currentSetPromptIds.length}</span>
                <button class="btn btn-xs pp-page-btn" id="pss-next" ${nextDisabled}>▶</button>
            </div>
        `;
    }

    // Build prompt buttons
    const promptTagsCache = getPromptTagsCache();
    let buttonsHtml = '';
    visibleIds.forEach((id, localIdx) => {
        const isActive = id === window.state.currentPromptId ? 'active' : '';
        const promptDef = (window.state.availablePromptSets[window.state.currentPromptSet] || []).find(p => p.id === id);
        const tooltip = promptDef ? promptDef.text.substring(0, 100) + (promptDef.text.length > 100 ? '...' : '') : '';
        const cacheKey = `${window.state.currentPromptSet}:${id}`;
        const tags = promptTagsCache?.[cacheKey] || [];
        const tagClasses = tags.map(t => `tag-${t}`).join(' ');
        const displayNum = startIdx + localIdx + 1;
        buttonsHtml += `<button class="btn btn-xs pp-btn pss-prompt-btn ${isActive} ${tagClasses}" data-prompt-id="${id}" title="${tooltip}">${displayNum}</button>`;
    });

    return `
        <div class="pss-prompts-section">
            <div class="pss-prompts-header">Prompts</div>
            <div class="pss-prompts-grid">${buttonsHtml}</div>
            ${paginationHtml}
        </div>
    `;
}

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
            <span class="pss-item-name" title="${escapeHtml(displayName)}">${escapeHtml(displayName)}</span>
            <span class="pss-item-count">${ids.length}</span>
        </div>`;
    }

    // Build prompt number buttons section
    const promptButtonsHtml = buildSidebarPromptButtons();

    container.innerHTML = `
        <div class="pss-header">
            <span>Prompt Sets</span>
            <button class="btn btn-xs btn-ghost pp-sidebar-toggle" id="pss-toggle-btn" title="Hide prompt set sidebar">☰</button>
        </div>
        <div class="pss-list">${listHtml}</div>
        ${promptButtonsHtml}
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

    // Prompt ID button click handlers
    container.querySelectorAll('.pss-prompt-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const promptId = btn.dataset.promptId;
            if (window.state.currentPromptId !== promptId && promptId != null) {
                window.state.currentPromptId = promptId;
                window.state.promptPickerCache = null;
                localStorage.setItem('promptId', promptId);
                if (window.state.currentPromptSet) {
                    localStorage.setItem(`promptId_${window.state.currentPromptSet}`, promptId);
                }
                renderPromptPicker();
                renderPromptSetSidebar();
                if (window.renderView) window.renderView();
            }
        });
    });

    // Sidebar pagination button handlers
    const prevBtn = container.querySelector('#pss-prev');
    const nextBtn = container.querySelector('#pss-next');
    if (prevBtn && !prevBtn.hasAttribute('disabled')) {
        prevBtn.onclick = (e) => {
            e.preventDefault();
            e.stopPropagation();
            const page = getPromptPage();
            if (page > 0) {
                setPromptPage(page - 1);
                renderPromptPicker();
                renderPromptSetSidebar();
            }
        };
    }
    if (nextBtn && !nextBtn.hasAttribute('disabled')) {
        nextBtn.onclick = (e) => {
            e.preventDefault();
            e.stopPropagation();
            const currentSetPromptIds = window.state.promptsWithData[window.state.currentPromptSet] || [];
            const totalPages = Math.ceil(currentSetPromptIds.length / PROMPTS_PER_PAGE);
            const page = getPromptPage();
            if (page < totalPages - 1) {
                setPromptPage(page + 1);
                renderPromptPicker();
                renderPromptSetSidebar();
            }
        };
    }
}

// ES module exports
export { renderPromptSetSidebar };

// Keep window.* for backward compat
window.renderPromptSetSidebar = renderPromptSetSidebar;
