/**
 * Prompt Picker - UI component for selecting prompt sets, prompt IDs, and tokens.
 *
 * Shows for inference views only. Displays:
 * - Prompt set dropdown (e.g., 'single_trait', 'multi_trait')
 * - Prompt ID buttons
 * - Token slider with highlighted text display
 */

// Views that show the prompt picker
const INFERENCE_VIEWS = ['trait-trajectory', 'trait-dynamics', 'layer-deep-dive', 'analysis-gallery'];

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

    // Build prompt picker HTML
    let promptSetOptions = '';
    for (const [setName, promptIds] of Object.entries(window.state.promptsWithData)) {
        if (promptIds.length === 0) continue;
        const selected = setName === window.state.currentPromptSet ? 'selected' : '';
        promptSetOptions += `<option value="${setName}" ${selected}>${setName.replace(/_/g, ' ')}</option>`;
    }

    const currentSetPromptIds = window.state.promptsWithData[window.state.currentPromptSet] || [];
    let promptBoxes = '';
    currentSetPromptIds.forEach(id => {
        const isActive = id === window.state.currentPromptId ? 'active' : '';
        const promptDef = (window.state.availablePromptSets[window.state.currentPromptSet] || []).find(p => p.id === id);
        const tooltip = promptDef ? promptDef.text.substring(0, 100) + (promptDef.text.length > 100 ? '...' : '') : '';
        promptBoxes += `<button class="pp-btn ${isActive}" data-prompt-set="${window.state.currentPromptSet}" data-prompt-id="${id}" title="${tooltip}">${id}</button>`;
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
            // Escape token for display (show special chars)
            const displayToken = currentToken
                .replace(/\n/g, '↵')
                .replace(/\t/g, '→')
                .replace(/ /g, '·');

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

    container.innerHTML = `
        <div class="pp-header">Prompt Picker</div>
        <div class="pp-picker">
            <select id="prompt-set-select">${promptSetOptions}</select>
            <div class="pp-prompts">${promptBoxes}</div>
            ${promptNote ? `<span class="pp-note">${promptNote}</span>` : ''}
        </div>
        <div class="pp-text">
            <div><strong>Prompt:</strong> ${buildHighlightedText(tokenList, window.state.currentTokenIndex, 0, window.state.promptPickerCache?.nPromptTokens || 0, 300)}</div>
            <div><strong>Response:</strong> ${buildHighlightedText(tokenList, window.state.currentTokenIndex, window.state.promptPickerCache?.nPromptTokens || 0, tokenList.length, 300)}</div>
        </div>
        ${tokenSliderHtml}
    `;

    // Re-attach event listeners
    setupPromptPickerListeners();
}

/**
 * Fetch prompt/response data from first available trait and cache it.
 */
async function fetchPromptPickerData() {
    if (!window.state.currentPromptSet || !window.state.currentPromptId) return;
    if (!window.state.experimentData || !window.state.experimentData.traits || window.state.experimentData.traits.length === 0) return;

    const firstTrait = window.state.experimentData.traits[0];

    try {
        const url = window.paths.residualStreamData(firstTrait, window.state.currentPromptSet, window.state.currentPromptId);
        const response = await fetch(url);
        if (!response.ok) return;

        const data = await response.json();

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
    } catch (e) {
        console.warn('Failed to fetch prompt picker data:', e);
    }
}

/**
 * Setup event listeners for the prompt picker.
 */
function setupPromptPickerListeners() {
    const container = document.getElementById('prompt-picker');
    if (!container) return;

    // Prompt set dropdown
    const setSelect = container.querySelector('#prompt-set-select');
    if (setSelect) {
        setSelect.addEventListener('change', (e) => {
            const newSet = e.target.value;
            if (window.state.currentPromptSet !== newSet) {
                window.state.currentPromptSet = newSet;
                const availableIds = window.state.promptsWithData[newSet] || [];
                window.state.currentPromptId = availableIds[0] || null;
                window.state.promptPickerCache = null; // Clear cache
                // Save to localStorage
                localStorage.setItem('promptSet', newSet);
                localStorage.setItem('promptId', window.state.currentPromptId);
                renderPromptPicker();
                if (window.renderView) window.renderView();
            }
        });
    }

    // Prompt ID buttons
    container.querySelectorAll('.pp-btn').forEach(box => {
        box.addEventListener('click', () => {
            const promptId = parseInt(box.dataset.promptId);
            if (window.state.currentPromptId !== promptId && !isNaN(promptId)) {
                window.state.currentPromptId = promptId;
                window.state.promptPickerCache = null; // Clear cache
                // Save to localStorage
                localStorage.setItem('promptId', promptId);
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
                const displayToken = currentToken
                    .replace(/\n/g, '↵')
                    .replace(/\t/g, '→')
                    .replace(/ /g, '·');
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
                        <div><strong>Prompt:</strong> ${buildHighlightedText(tokenList, newIdx, 0, nPromptTokens, 300)}</div>
                        <div><strong>Response:</strong> ${buildHighlightedText(tokenList, newIdx, nPromptTokens, tokenList.length, 300)}</div>
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
    const startIdx = 1;  // BOS is skipped in all plots
    const highlightX = tokenIdx - startIdx;
    const separatorX = (nPromptTokens - startIdx) - 0.5;

    // Get highlight colors from centralized helper
    const { separator: separatorColor, highlight: highlightColor } = window.getTokenHighlightColors();
    const primaryColor = window.getCssVar('--primary-color', '#a09f6c');
    const textSecondary = window.getCssVar('--text-secondary', '#a4a4a4');

    if (window.state.currentView === 'trait-trajectory') {
        // Update all trait heatmaps
        const filteredTraits = window.getFilteredTraits();
        for (const trait of filteredTraits) {
            const traitId = trait.name.replace(/\//g, '-');
            const plotDiv = document.getElementById(`trajectory-heatmap-${traitId}`);
            if (plotDiv && plotDiv.data) {
                Plotly.relayout(plotDiv, {
                    shapes: [
                        { type: 'line', xref: 'x', yref: 'paper', x0: separatorX, x1: separatorX, y0: 0, y1: 1, line: { color: separatorColor, width: 2, dash: 'dash' } },
                        { type: 'rect', xref: 'x', yref: 'paper', x0: highlightX - 0.5, x1: highlightX + 0.5, y0: 0, y1: 1, fillcolor: highlightColor, line: { width: 0 } }
                    ]
                });
            }
        }
    } else if (window.state.currentView === 'trait-dynamics') {
        // Update combined activation plot
        const plotDiv = document.getElementById('combined-activation-plot');
        if (plotDiv && plotDiv.data) {
            Plotly.relayout(plotDiv, {
                shapes: [
                    { type: 'line', x0: separatorX, x1: separatorX, y0: 0, y1: 1, yref: 'paper', line: { color: textSecondary, width: 2, dash: 'dash' } },
                    { type: 'line', x0: highlightX, x1: highlightX, y0: 0, y1: 1, yref: 'paper', line: { color: primaryColor, width: 2 } }
                ]
            });
        }
    } else if (window.state.currentView === 'analysis-gallery') {
        // Analysis Gallery needs full re-render with new token data (data is cached)
        if (window.renderAnalysisGallery) {
            window.renderAnalysisGallery();
        }
    } else if (window.state.currentView === 'layer-deep-dive') {
        // Layer Deep Dive needs full re-render for new token's SAE features
        if (window.renderLayerDeepDive) {
            window.renderLayerDeepDive();
        }
    }
}

/**
 * Build text with the current token highlighted and markdown rendered.
 * @param {string[]} tokenList - Full list of all tokens
 * @param {number} currentIdx - Absolute index of current token
 * @param {number} startIdx - Start of range to display (inclusive)
 * @param {number} endIdx - End of range to display (exclusive)
 * @param {number} maxChars - Max characters to show before truncating
 */
function buildHighlightedText(tokenList, currentIdx, startIdx, endIdx, maxChars) {
    if (!tokenList || tokenList.length === 0) {
        return 'Loading...';
    }

    let result = '';
    let charCount = 0;
    let truncated = false;

    for (let i = startIdx; i < endIdx; i++) {
        const token = tokenList[i];
        if (!token) continue;
        const escaped = window.escapeHtml(token);

        // Check if we'd exceed max chars
        if (charCount + token.length > maxChars) {
            truncated = true;
            break;
        }

        if (i === currentIdx) {
            result += `<span class="token-highlight">${escaped}</span>`;
        } else {
            result += escaped;
        }
        charCount += token.length;
    }

    if (truncated) {
        result += '...';
    }

    // Apply markdown formatting (bold, italic) - works across tokens
    // Use a placeholder to protect highlight spans from markdown parsing
    const placeholder = '\x00HIGHLIGHT\x00';
    result = result.replace(/<span class="token-highlight">(.*?)<\/span>/g, (match, content) => {
        return placeholder + content + placeholder;
    });
    result = window.markdownToHtml(result);
    result = result.replace(new RegExp(placeholder + '(.*?)' + placeholder, 'g'), '<span class="token-highlight">$1</span>');

    return result || '(empty)';
}

// Export to global scope
window.INFERENCE_VIEWS = INFERENCE_VIEWS;
window.renderPromptPicker = renderPromptPicker;
window.fetchPromptPickerData = fetchPromptPickerData;
window.updatePlotTokenHighlights = updatePlotTokenHighlights;
