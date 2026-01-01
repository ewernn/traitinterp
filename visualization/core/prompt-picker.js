/**
 * Prompt Picker - UI component for selecting prompt sets, prompt IDs, and tokens.
 *
 * Shows for inference views only. Displays:
 * - Prompt set dropdown (e.g., 'single_trait', 'multi_trait')
 * - Prompt ID buttons
 * - Token slider with highlighted text display
 */

// Views that show the prompt picker
const INFERENCE_VIEWS = ['trait-dynamics', 'layer-deep-dive'];

// Pagination settings
const PROMPTS_PER_PAGE = 50;

/**
 * Render the prompt picker panel.
 * Shows only for inference views, hidden for trait development views.
 */
async function renderPromptPicker() {
    const container = document.getElementById('prompt-picker');
    if (!container) return;

    const isInferenceView = INFERENCE_VIEWS.includes(window.state.currentView);

    if (!isInferenceView) {
        container.style.display = 'none';
        return;
    }

    container.style.display = 'block';

    // Check if we have prompt data
    const hasAnyData = Object.values(window.state.promptsWithData).some(ids => ids.length > 0);
    if (!hasAnyData) {
        container.innerHTML = `<div class="pp-meta">No inference data available for this experiment.</div>`;
        return;
    }

    // Build prompt set buttons
    let promptSetButtons = '';
    for (const [setName, promptIds] of Object.entries(window.state.promptsWithData)) {
        if (promptIds.length === 0) continue;
        const isActive = setName === window.state.currentPromptSet ? 'active' : '';
        const displayName = setName.replace(/_/g, ' ');
        promptSetButtons += `<button class="pp-btn pp-set-btn ${isActive}" data-set="${setName}">${displayName}</button>`;
    }

    // Build prompt ID buttons (with pagination for large sets)
    const currentSetPromptIds = window.state.promptsWithData[window.state.currentPromptSet] || [];
    const needsPagination = currentSetPromptIds.length > PROMPTS_PER_PAGE;

    // Initialize page state if needed, and ensure current prompt is visible
    if (window.state.promptPage === undefined) {
        window.state.promptPage = 0;
        // Jump to page containing current prompt (only on init, not during pagination)
        if (window.state.currentPromptId !== null && needsPagination) {
            const promptIdx = currentSetPromptIds.indexOf(window.state.currentPromptId);
            if (promptIdx >= 0) {
                window.state.promptPage = Math.floor(promptIdx / PROMPTS_PER_PAGE);
            }
        }
    }

    // Calculate pagination
    const totalPages = Math.ceil(currentSetPromptIds.length / PROMPTS_PER_PAGE);
    const currentPage = Math.min(window.state.promptPage, totalPages - 1);
    const startIdx = currentPage * PROMPTS_PER_PAGE;
    const endIdx = Math.min(startIdx + PROMPTS_PER_PAGE, currentSetPromptIds.length);
    const visibleIds = needsPagination ? currentSetPromptIds.slice(startIdx, endIdx) : currentSetPromptIds;

    // Build pagination controls
    let paginationHtml = '';
    if (needsPagination) {
        const prevDisabled = currentPage === 0 ? 'disabled' : '';
        const nextDisabled = currentPage >= totalPages - 1 ? 'disabled' : '';
        paginationHtml = `
            <div class="pp-pagination">
                <button class="pp-page-btn" id="pp-prev" ${prevDisabled}>◀</button>
                <span class="pp-page-info">${startIdx + 1}-${endIdx} of ${currentSetPromptIds.length}</span>
                <button class="pp-page-btn" id="pp-next" ${nextDisabled}>▶</button>
            </div>
        `;
    }

    let promptBoxes = '';
    const isJailbreakSet = window.state.currentPromptSet === 'jailbreak';
    visibleIds.forEach(id => {
        const isActive = id === window.state.currentPromptId ? 'active' : '';
        const isSuccess = isJailbreakSet && window.state.jailbreakSuccessIds?.has(id) ? 'jailbreak-success' : '';
        const promptDef = (window.state.availablePromptSets[window.state.currentPromptSet] || []).find(p => p.id === id);
        const tooltip = promptDef ? promptDef.text.substring(0, 100) + (promptDef.text.length > 100 ? '...' : '') : '';
        promptBoxes += `<button class="pp-btn ${isActive} ${isSuccess}" data-prompt-set="${window.state.currentPromptSet}" data-prompt-id="${id}" title="${tooltip}">${id}</button>`;
    });

    // Get prompt text and note from definitions
    const promptDef = (window.state.availablePromptSets[window.state.currentPromptSet] || []).find(p => p.id === window.state.currentPromptId);
    const promptNote = promptDef && promptDef.note ? window.escapeHtml(promptDef.note) : '';

    // Check cache for response data
    let tokenSliderHtml = '';
    let tokenList = [];

    if (window.state.promptPickerCache &&
        window.state.promptPickerCache.promptSet === window.state.currentPromptSet &&
        window.state.promptPickerCache.promptId === window.state.currentPromptId) {
        // Use cached data
        tokenList = window.state.promptPickerCache.allTokens || [];
        const nPromptTokens = window.state.promptPickerCache.nPromptTokens || 0;

        // Build token slider if we have tokens
        if (tokenList.length > 0) {
            const maxIdx = tokenList.length - 1;
            const currentIdx = Math.min(window.state.currentTokenIndex, maxIdx);
            const currentToken = tokenList[currentIdx] || '';
            const displayToken = window.formatTokenDisplay(currentToken);

            tokenSliderHtml = `
                <div class="pp-slider">
                    <strong>Token:</strong>
                    <input type="range" id="token-slider" min="0" max="${maxIdx}" value="${currentIdx}">
                    <code>${currentIdx}</code>
                    <code class="pp-token">${window.escapeHtml(displayToken)}</code>
                </div>
            `;
        }
    } else {
        // Need to fetch - will re-render when done
        fetchPromptPickerData();
    }

    // Check if previously collapsed
    const isCollapsed = localStorage.getItem('promptPickerCollapsed') === 'true';
    const collapsedClass = isCollapsed ? 'collapsed' : '';

    container.innerHTML = `
        <div class="pp-pill" id="pp-pill">
            <span class="pp-pill-icon">▲</span>
            <span class="pp-pill-label">Prompt Picker</span>
            <span class="pp-pill-summary">${window.state.currentPromptSet?.replace(/_/g, ' ') || ''} #${window.state.currentPromptId ?? ''}</span>
        </div>
        <div class="pp-expanded ${collapsedClass}" id="pp-expanded">
            <div class="pp-header">
                <span>Prompt Picker</span>
                <button class="pp-collapse-btn" id="pp-collapse-btn" title="Collapse">▼</button>
            </div>
            <div class="pp-picker">
                <div class="pp-row">
                    <span class="pp-row-label">Set:</span>
                    <div class="pp-sets">${promptSetButtons}</div>
                </div>
                <div class="pp-row">
                    <span class="pp-row-label">Prompt:</span>
                    <div class="pp-prompts">${promptBoxes}</div>
                    ${paginationHtml}
                </div>
            </div>
            ${promptNote ? `<div class="pp-note">${promptNote}</div>` : ''}
            <div class="pp-text">
                <div><strong>Prompt:</strong> ${buildHighlightedText(tokenList, window.state.currentTokenIndex, 0, window.state.promptPickerCache?.nPromptTokens || 0)}</div>
                <div><strong>Response:</strong> ${buildHighlightedText(tokenList, window.state.currentTokenIndex, window.state.promptPickerCache?.nPromptTokens || 0, tokenList.length)}</div>
            </div>
            ${tokenSliderHtml}
        </div>
    `;

    // Re-attach event listeners
    setupPromptPickerListeners();
}

/**
 * Fetch prompt/response data and cache it.
 * Tries shared response data first, falls back to projection data for backwards compatibility.
 */
async function fetchPromptPickerData() {
    if (!window.state.currentPromptSet || !window.state.currentPromptId) return;

    let data = null;

    // Try shared response data first (new format)
    try {
        const responseUrl = window.paths.responseData(window.state.currentPromptSet, window.state.currentPromptId);
        const response = await fetch(responseUrl);
        if (response.ok) {
            data = await response.json();
        }
    } catch (e) {
        // Fall through to fallback
    }

    // Fall back to projection data (old format, for backwards compatibility)
    if (!data && window.state.experimentData?.traits?.length > 0) {
        const firstTrait = window.state.experimentData.traits[0];
        try {
            const url = window.paths.residualStreamData(firstTrait, window.state.currentPromptSet, window.state.currentPromptId);
            const response = await fetch(url);
            if (response.ok) {
                data = await response.json();
            }
        } catch (e) {
            console.warn('Failed to fetch prompt picker data:', e);
        }
    }

    if (!data) return;

    const promptTokenList = data.prompt?.tokens || [];
    const responseTokenList = data.response?.tokens || [];
    const allTokens = [...promptTokenList, ...responseTokenList];

    window.state.promptPickerCache = {
        promptSet: window.state.currentPromptSet,
        promptId: window.state.currentPromptId,
        promptText: data.prompt?.text || '',
        responseText: data.response?.text || '',
        promptTokens: promptTokenList.length,
        responseTokens: responseTokenList.length,
        allTokens: allTokens,
        nPromptTokens: promptTokenList.length
    };

    // Reset token index when loading new prompt (clamp to valid range)
    const maxIdx = Math.max(0, allTokens.length - 1);
    window.state.currentTokenIndex = Math.min(window.state.currentTokenIndex, maxIdx);

    // Re-render with the fetched data
    renderPromptPicker();

    // Resize all Plotly charts after prompt picker layout change
    // This fixes the first-trait-not-full-width bug on initial load
    requestAnimationFrame(() => {
        document.querySelectorAll('.js-plotly-plot').forEach(plot => {
            Plotly.Plots.resize(plot);
        });
    });
}

/**
 * Setup event listeners for the prompt picker.
 */
function setupPromptPickerListeners() {
    const container = document.getElementById('prompt-picker');
    if (!container) return;

    // Pill click to expand
    const pill = container.querySelector('#pp-pill');
    const expanded = container.querySelector('#pp-expanded');
    if (pill && expanded) {
        pill.addEventListener('click', () => {
            expanded.classList.remove('collapsed');
            localStorage.setItem('promptPickerCollapsed', 'false');
        });
    }

    // Collapse button
    const collapseBtn = container.querySelector('#pp-collapse-btn');
    if (collapseBtn && expanded) {
        collapseBtn.addEventListener('click', () => {
            expanded.classList.add('collapsed');
            localStorage.setItem('promptPickerCollapsed', 'true');
        });
    }

    // Prompt set buttons
    container.querySelectorAll('.pp-set-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const newSet = btn.dataset.set;
            if (window.state.currentPromptSet !== newSet) {
                // Save current prompt ID for the old set before switching
                if (window.state.currentPromptSet && window.state.currentPromptId) {
                    localStorage.setItem(`promptId_${window.state.currentPromptSet}`, window.state.currentPromptId);
                }

                window.state.currentPromptSet = newSet;
                window.state.promptPage = 0; // Reset to first page

                // Try to restore last prompt ID for this set, otherwise use first available
                const availableIds = window.state.promptsWithData[newSet] || [];
                const savedPromptId = parseInt(localStorage.getItem(`promptId_${newSet}`));
                if (savedPromptId && availableIds.includes(savedPromptId)) {
                    window.state.currentPromptId = savedPromptId;
                    // Jump to page containing this prompt
                    const promptIdx = availableIds.indexOf(savedPromptId);
                    window.state.promptPage = Math.floor(promptIdx / PROMPTS_PER_PAGE);
                } else {
                    window.state.currentPromptId = availableIds[0] || null;
                }

                window.state.promptPickerCache = null; // Clear cache
                // Save to localStorage
                localStorage.setItem('promptSet', newSet);
                localStorage.setItem('promptId', window.state.currentPromptId);
                renderPromptPicker();
                if (window.renderView) window.renderView();
            }
        });
    });

    // Pagination buttons (use mousedown to ensure event fires before any re-render)
    const prevBtn = container.querySelector('#pp-prev');
    const nextBtn = container.querySelector('#pp-next');
    if (prevBtn && !prevBtn.hasAttribute('disabled')) {
        prevBtn.onclick = (e) => {
            e.preventDefault();
            e.stopPropagation();
            if (window.state.promptPage > 0) {
                window.state.promptPage--;
                renderPromptPicker();
            }
        };
    }
    if (nextBtn && !nextBtn.hasAttribute('disabled')) {
        nextBtn.onclick = (e) => {
            e.preventDefault();
            e.stopPropagation();
            const currentSetPromptIds = window.state.promptsWithData[window.state.currentPromptSet] || [];
            const totalPages = Math.ceil(currentSetPromptIds.length / PROMPTS_PER_PAGE);
            if (window.state.promptPage < totalPages - 1) {
                window.state.promptPage++;
                renderPromptPicker();
            }
        };
    }

    // Prompt ID buttons (exclude set buttons)
    container.querySelectorAll('.pp-prompts .pp-btn').forEach(box => {
        box.addEventListener('click', () => {
            const promptId = parseInt(box.dataset.promptId);
            if (window.state.currentPromptId !== promptId && !isNaN(promptId)) {
                window.state.currentPromptId = promptId;
                window.state.promptPickerCache = null; // Clear cache
                // Save to localStorage (both global and per-set)
                localStorage.setItem('promptId', promptId);
                if (window.state.currentPromptSet) {
                    localStorage.setItem(`promptId_${window.state.currentPromptSet}`, promptId);
                }
                renderPromptPicker();
                if (window.renderView) window.renderView();
            }
        });
    });

    // Token slider
    const tokenSlider = container.querySelector('#token-slider');
    if (tokenSlider) {
        tokenSlider.addEventListener('input', (e) => {
            const newIdx = parseInt(e.target.value);
            if (window.state.currentTokenIndex !== newIdx && !isNaN(newIdx)) {
                window.state.currentTokenIndex = newIdx;
                // Update display without full re-render
                const tokenList = window.state.promptPickerCache?.allTokens || [];
                const nPromptTokens = window.state.promptPickerCache?.nPromptTokens || 0;
                const currentToken = tokenList[newIdx] || '';
                const displayToken = window.formatTokenDisplay(currentToken);
                // Update slider display elements
                const slider = container.querySelector('.pp-slider');
                if (slider) {
                    slider.querySelector('code').textContent = newIdx;
                    slider.querySelector('.pp-token').textContent = displayToken;
                }
                // Update highlighted text
                const textDiv = container.querySelector('.pp-text');
                if (textDiv) {
                    textDiv.innerHTML = `
                        <div><strong>Prompt:</strong> ${buildHighlightedText(tokenList, newIdx, 0, nPromptTokens)}</div>
                        <div><strong>Response:</strong> ${buildHighlightedText(tokenList, newIdx, nPromptTokens, tokenList.length)}</div>
                    `;
                }
                // Update plot highlights without full re-render
                updatePlotTokenHighlights(newIdx, nPromptTokens);
            }
        });
    }
}

/**
 * Update token highlight shapes on existing Plotly plots (no re-render).
 */
function updatePlotTokenHighlights(tokenIdx, nPromptTokens) {
    const startIdx = 0;  // Show all tokens including BOS
    const highlightX = Math.max(0, tokenIdx - startIdx);
    const separatorX = (nPromptTokens - startIdx) - 0.5;

    // Get highlight colors from centralized helper
    const { separator: separatorColor, highlight: highlightColor } = window.getTokenHighlightColors();
    const primaryColor = window.getCssVar('--primary-color', '#a09f6c');
    const textSecondary = window.getCssVar('--text-secondary', '#a4a4a4');

    if (window.state.currentView === 'trait-dynamics') {
        // Standard shapes for all trait-dynamics plots
        const shapes = [
            { type: 'line', x0: separatorX, x1: separatorX, y0: 0, y1: 1, yref: 'paper', line: { color: textSecondary, width: 2, dash: 'dash' } },
            { type: 'line', x0: highlightX, x1: highlightX, y0: 0, y1: 1, yref: 'paper', line: { color: primaryColor, width: 2 } }
        ];

        // Update all trait-dynamics plots
        const plotIds = [
            'combined-activation-plot',
            'normalized-trajectory-plot',
            'token-magnitude-plot',
            'token-velocity-plot'
        ];

        plotIds.forEach(id => {
            const plotDiv = document.getElementById(id);
            if (plotDiv && plotDiv.data) {
                Plotly.relayout(plotDiv, { shapes });
            }
        });
    } else if (window.state.currentView === 'layer-deep-dive') {
        // Layer Deep Dive needs full re-render for new token's SAE features
        if (window.renderLayerDeepDive) {
            window.renderLayerDeepDive();
        }
    }
}

/**
 * Build text with the current token highlighted.
 * @param {string[]} tokenList - Full list of all tokens
 * @param {number} currentIdx - Absolute index of current token
 * @param {number} startIdx - Start of range to display (inclusive)
 * @param {number} endIdx - End of range to display (exclusive)
 */
function buildHighlightedText(tokenList, currentIdx, startIdx, endIdx) {
    if (!tokenList || tokenList.length === 0) {
        return 'Loading...';
    }

    let result = '';

    for (let i = startIdx; i < endIdx; i++) {
        const token = tokenList[i];
        if (!token) continue;
        const escaped = window.escapeHtml(token);

        if (i === currentIdx) {
            result += `<span class="token-highlight">${escaped}</span>`;
        } else {
            result += escaped;
        }
    }

    return result || '(empty)';
}

// Export to global scope
window.INFERENCE_VIEWS = INFERENCE_VIEWS;
window.renderPromptPicker = renderPromptPicker;
window.fetchPromptPickerData = fetchPromptPickerData;
window.updatePlotTokenHighlights = updatePlotTokenHighlights;
