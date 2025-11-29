/**
 * Zoomies Selectors
 * Prompt picker style - shows prompt set, ID buttons, and prompt text.
 */

window.zoomies = window.zoomies || {};

/**
 * Render the prompt picker / selectors
 * @param {HTMLElement} container
 */
window.zoomies.renderSelectors = function(container) {
    const state = window.zoomies.state;

    // Check if we have any data
    if (!state.experiments || state.experiments.length === 0) {
        container.innerHTML = '<div class="pp-meta">Loading experiments...</div>';
        return;
    }

    const promptSets = state.promptSets || {};
    const promptSetNames = Object.keys(promptSets);

    if (promptSetNames.length === 0) {
        container.innerHTML = '<div class="pp-meta">No inference data available for this experiment.</div>';
        return;
    }

    // Get current prompt set's available IDs
    const currentSet = promptSets[state.promptSet] || {};
    const promptIds = currentSet.ids || [];

    // Build prompt ID buttons
    let promptButtons = '';
    promptIds.forEach(id => {
        const isActive = id === state.promptId ? 'active' : '';
        promptButtons += `<button class="pp-btn ${isActive}" data-prompt-id="${id}">${id}</button>`;
    });

    // Build prompt set options
    let promptSetOptions = '';
    promptSetNames.forEach(name => {
        const selected = name === state.promptSet ? 'selected' : '';
        promptSetOptions += `<option value="${name}" ${selected}>${formatPromptSetName(name)}</option>`;
    });

    // Build experiment options
    let experimentOptions = '';
    state.experiments.forEach(exp => {
        const selected = exp === state.experiment ? 'selected' : '';
        experimentOptions += `<option value="${exp}" ${selected}>${formatExperimentName(exp)}</option>`;
    });

    // Build trait selector
    const selectedCount = state.selectedTraits?.length || 0;
    const isInference = state.mode === 'inference';

    container.innerHTML = `
        <div class="prompt-picker">
            <div class="pp-row">
                <div class="pp-group">
                    <span class="pp-label">Mode</span>
                    <div class="pp-prompts">
                        <button class="pp-btn ${isInference ? 'active' : ''}" data-mode="inference">Inference</button>
                        <button class="pp-btn ${!isInference ? 'active' : ''}" data-mode="extraction">Extraction</button>
                    </div>
                </div>
                <div class="pp-group">
                    <span class="pp-label">Experiment</span>
                    <select id="experiment-select">${experimentOptions}</select>
                </div>
                ${isInference ? `
                    <div class="pp-group">
                        <span class="pp-label">Prompt Set</span>
                        <select id="prompt-set-select">${promptSetOptions}</select>
                    </div>
                    <div class="pp-group">
                        <span class="pp-label">Prompt</span>
                        <div class="pp-prompts">${promptButtons}</div>
                    </div>
                ` : ''}
                <div class="pp-group pp-traits-group">
                    <span class="pp-label">Traits</span>
                    <button id="trait-selector-btn" class="pp-trait-btn">
                        ${selectedCount} selected <span class="arrow">▾</span>
                    </button>
                    <div id="trait-dropdown" class="trait-dropdown" style="display: none;">
                        ${(state.traits || []).map(trait => `
                            <label class="trait-option">
                                <input type="checkbox" value="${trait}"
                                    ${state.selectedTraits?.includes(trait) ? 'checked' : ''}>
                                ${formatTraitName(trait)}
                            </label>
                        `).join('')}
                    </div>
                </div>
            </div>
        </div>
    `;

    // Setup event listeners
    setupSelectorsListeners(container);
};

/**
 * Setup event listeners for selectors
 */
function setupSelectorsListeners(container) {
    // Experiment selector
    const expSelect = container.querySelector('#experiment-select');
    if (expSelect) {
        expSelect.addEventListener('change', async (e) => {
            const newExp = e.target.value;
            await window.zoomies.loadExperimentData(newExp);
            window.zoomies.setState({ experiment: newExp });
        });
    }

    // Prompt set selector
    const setSelect = container.querySelector('#prompt-set-select');
    if (setSelect) {
        setSelect.addEventListener('change', (e) => {
            const newSet = e.target.value;
            const state = window.zoomies.state;
            const firstId = state.promptSets[newSet]?.ids?.[0] || 1;
            window.zoomies.setState({ promptSet: newSet, promptId: firstId });
        });
    }

    // Mode toggle buttons
    container.querySelectorAll('.pp-btn[data-mode]').forEach(btn => {
        btn.addEventListener('click', () => {
            const mode = btn.dataset.mode;
            window.zoomies.setState({ mode });
        });
    });

    // Prompt ID buttons
    container.querySelectorAll('.pp-btn[data-prompt-id]').forEach(btn => {
        btn.addEventListener('click', () => {
            const promptId = parseInt(btn.dataset.promptId, 10);
            if (!isNaN(promptId)) {
                window.zoomies.setState({ promptId });
            }
        });
    });

    // Trait dropdown toggle
    const traitBtn = container.querySelector('#trait-selector-btn');
    const traitDropdown = container.querySelector('#trait-dropdown');
    if (traitBtn && traitDropdown) {
        traitBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            const isVisible = traitDropdown.style.display !== 'none';
            traitDropdown.style.display = isVisible ? 'none' : 'block';
        });

        // Trait checkbox changes
        traitDropdown.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
            checkbox.addEventListener('change', () => {
                const selected = Array.from(traitDropdown.querySelectorAll('input:checked'))
                    .map(cb => cb.value);
                window.zoomies.setState({ selectedTraits: selected });
            });
        });

        // Close on outside click
        document.addEventListener('click', (e) => {
            if (!traitBtn.contains(e.target) && !traitDropdown.contains(e.target)) {
                traitDropdown.style.display = 'none';
            }
        });
    }
}

/**
 * Format experiment name for display
 */
function formatExperimentName(name) {
    return name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

/**
 * Format prompt set name for display
 */
function formatPromptSetName(name) {
    return name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

/**
 * Format trait name for display
 */
function formatTraitName(trait) {
    const parts = trait.split('/');
    const name = parts[parts.length - 1];
    return name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

// Export formatters
window.zoomies.formatTraitName = formatTraitName;
window.zoomies.formatExperimentName = formatExperimentName;
