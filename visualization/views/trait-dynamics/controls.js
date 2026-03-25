// Controls for Trait Dynamics view — control bar HTML, page shell, event listeners
// Input: window.state, allFilteredTraits
// Output: HTML strings, attached DOM listeners

import { getDisplayName } from '../../core/display.js';
import { setupSubsectionInfoToggles } from '../../components/sidebar.js';
import { renderSegmentedControl, renderSmoothPill, renderSubsection } from '../../core/ui.js';
import {
    setSmoothingWindow,
    setProjectionCentered,
    setProjectionMode,
    setMassiveDimsCleaning,
    setLayerMode,
    setLayerModeTrait,
    setCompareMode,
    toggleMethod,
    setWideMode,
    setShowVelocity,
} from '../../core/state.js';

// =============================================================================
// HTML builders
// =============================================================================

/**
 * Build the control bar HTML for Token Trajectory section.
 * Primary row: Smooth, Mode, Compare + model dropdown, Advanced toggle.
 * Advanced row (collapsed): Methods, Centered, Clean, Layers, Wide, Velocity.
 */
function buildControlBarHtml(allFilteredTraits) {
    const currentCompareMode = window.state.compareMode || 'main';
    const compareModeBase = currentCompareMode.startsWith('diff:') ? 'diff'
        : currentCompareMode.startsWith('show:') ? 'show' : 'main';
    const availableModels = window.state.availableComparisonModels || [];
    const currentCompareVariant = currentCompareMode.startsWith('diff:') ? currentCompareMode.slice(5)
        : currentCompareMode.startsWith('show:') ? currentCompareMode.slice(5)
        : (window.state.lastCompareVariant || availableModels[0] || '');
    const modelDropdownDisabled = compareModeBase === 'main' || availableModels.length === 0;

    const isCentered = window.state.projectionCentered !== false;
    const massiveDimsCleaning = window.state.massiveDimsCleaning || 'none';

    // --- Primary row ---
    const smoothCluster = `
        <div class="cb-cluster" style="gap: 8px;">
            <span class="cb-label">Smooth:</span>
            ${renderSmoothPill(window.state.smoothingWindow)}
        </div>`;

    const modeCluster = `
        <div class="cb-cluster">
            <span class="cb-label">Mode:</span>
            ${renderSegmentedControl({
                id: 'mode-control',
                options: [
                    { value: 'cosine', label: 'Cosine' },
                    { value: 'normalized', label: 'Normalized' },
                    { value: 'raw', label: 'Raw' },
                ],
                selected: window.state.projectionMode,
                dataAttr: 'mode',
            })}
        </div>`;

    const modelOptions = availableModels.map(m =>
        `<option value="${m}" ${m === currentCompareVariant ? 'selected' : ''}>${m}</option>`
    ).join('');
    const modelDropdown = `<select id="compare-variant-select" class="cb-select"${modelDropdownDisabled ? ' disabled' : ''}>${
        availableModels.length === 0 ? '<option>No models</option>' : modelOptions
    }</select>`;

    const compareCluster = `
        <div class="cb-cluster">
            <span class="cb-label">Compare:</span>
            ${renderSegmentedControl({
                id: 'compare-control',
                options: [
                    { value: 'main', label: 'Main' },
                    { value: 'diff', label: 'Diff' },
                    { value: 'show', label: 'Show' },
                ],
                selected: compareModeBase,
                dataAttr: 'compare',
                disabled: availableModels.length === 0,
                disabledTooltip: 'No comparison models configured',
            })}
            ${modelDropdown}
        </div>`;

    const advToggle = `<button class="adv-toggle" id="td-advanced-toggle" aria-expanded="false" style="margin-left: auto;">Advanced <span class="arrow">\u25B6</span></button>`;

    // --- Advanced row ---
    const methodCheckboxes = ['probe', 'mean_diff', 'gradient', 'random'].map(m =>
        `<label class="cb-checkbox"><input type="checkbox" data-method="${m}" class="method-filter" ${window.state.selectedMethods.has(m) ? 'checked' : ''}> ${m}</label>`
    ).join('\n                    ');

    const layerTraitSelect = window.state.layerMode ? `
                    <select id="layer-mode-trait-select" class="cb-select">
                        ${allFilteredTraits.map(t =>
                            `<option value="${t.name}" ${t.name === window.state.layerModeTrait ? 'selected' : ''}>${getDisplayName(t.name)}</option>`
                        ).join('')}
                    </select>` : '';

    const advancedRow = `
            <div class="cb-row cb-advanced" id="td-advanced-row" hidden>
                <div class="cb-cluster">
                    <span class="cb-label">Methods:</span>
                    ${methodCheckboxes}
                </div>
                <label class="cb-checkbox"><input type="checkbox" id="projection-centered-toggle" ${isCentered ? 'checked' : ''}> Centered</label>
                <div class="cb-cluster">
                    <span class="cb-label">Clean:</span>
                    <select id="massive-dims-cleaning-select" class="cb-select" title="Remove high-magnitude bias dimensions (Sun et al. 2024)">
                        <option value="none" ${massiveDimsCleaning === 'none' ? 'selected' : ''}>None</option>
                        <option value="top5-3layers" ${massiveDimsCleaning === 'top5-3layers' ? 'selected' : ''}>Top 5</option>
                        <option value="all" ${massiveDimsCleaning === 'all' ? 'selected' : ''}>All</option>
                    </select>
                </div>
                <label class="cb-checkbox"><input type="checkbox" id="layer-mode-toggle" ${window.state.layerMode ? 'checked' : ''}> Layers</label>${layerTraitSelect}
                <label class="cb-checkbox"><input type="checkbox" id="wide-mode-toggle" ${window.state.wideMode ? 'checked' : ''}> Wide</label>
                <label class="cb-checkbox"><input type="checkbox" id="velocity-toggle" ${window.state.showVelocity ? 'checked' : ''}> Velocity</label>
            </div>`;

    return `
        <div class="cb">
            <div class="cb-row">
                ${smoothCluster}
                ${modeCluster}
                ${compareCluster}
                ${advToggle}
            </div>
            ${advancedRow}
        </div>
    `;
}


/**
 * Build full page shell HTML with controls and empty chart divs.
 * Token Trajectory uses a plain header (not collapsible — it's the primary view).
 * Other sections use uniform collapsible sec-header pattern.
 */
function buildPageShellHtml(allFilteredTraits) {
    const projectionMode = window.state.projectionMode || 'cosine';
    const isCentered = window.state.projectionCentered !== false;
    const smoothingWindow = window.state.smoothingWindow;
    const experimentName = window.state.currentExperiment || 'EXPERIMENT';

    const infoText = (projectionMode === 'normalized'
        ? 'Normalized projection: proj / avg||h||. Preserves per-token variance, removes layer-dependent scale.'
        : projectionMode === 'raw'
            ? 'Raw projection onto trait vector. No normalization applied.'
            : 'Cosine similarity: proj / ||h||. Shows directional alignment with trait vector.')
        + (isCentered ? ' Centered by subtracting BOS token value.' : '')
        + (smoothingWindow > 0 ? ` Smoothed with ${smoothingWindow}-token moving average.` : '');

    return `
        <div class="tool-view${window.state.wideMode ? ' wide-mode' : ''}">
            <div class="page-intro">
                <div class="page-intro-text">Watch traits evolve token-by-token during generation.</div>
                <div id="trait-dynamics-status"></div>
            </div>

            <section>
                ${renderSubsection({
                    title: 'Token Trajectory',
                    infoId: 'info-token-trajectory',
                    infoText: infoText,
                    level: 'h2'
                })}
                ${buildControlBarHtml(allFilteredTraits)}
                <div id="overlay-controls"></div>
                <div id="combined-activation-plot"></div>
            </section>

            <section>
                <div class="sec-header" data-section="top-spans" id="sec-top-spans">
                    <span class="arrow">\u25BC</span> Top Spans <span class="sec-badge" id="badge-top-spans"></span>
                </div>
                <div id="section-body-top-spans">
                    <div id="top-spans-panel"></div>
                </div>
            </section>

            <section>
                <div class="sec-header" data-section="heatmap" id="sec-heatmap">
                    <span class="arrow">\u25B6</span> Trait \u00D7 Token Heatmap <span class="sec-badge" id="badge-heatmap"></span>
                </div>
                <div id="section-body-heatmap" hidden>
                    <div id="trait-heatmap-panel"></div>
                </div>
            </section>

            <section>
                <div class="sec-header" data-section="magnitude" id="sec-magnitude">
                    <span class="arrow">\u25BC</span> Activation Magnitude <span class="sec-badge" id="badge-magnitude"></span>
                </div>
                <div id="section-body-magnitude">
                    <div id="token-magnitude-plot"></div>
                </div>
            </section>

            <section id="cue-p-section" style="display:none">
                ${renderSubsection({
                    title: 'Resampling cue_p',
                    infoId: 'info-cue-p',
                    infoText: 'Per-sentence resampling probability of the cued (wrong) answer, from Thought Branches transplant experiment (~4000 forward passes per sentence). Shows how bias accumulates through the CoT.'
                })}
                <div id="cue-p-plot"></div>
            </section>

            <section id="correlation-section" style="display: none;">
                <div class="sec-header" data-section="correlation" id="sec-correlation">
                    <span class="arrow">\u25B6</span> Correlation <span class="sec-badge" id="badge-correlation"></span>
                </div>
                <div id="section-body-correlation" hidden>
                    <div id="correlation-content">
                        <div class="no-data-hint">No pre-computed correlation data.
                            <code>python analysis/trait_correlation.py --experiment ${experimentName} --prompt-set PROMPT_SET</code>
                        </div>
                    </div>
                </div>
            </section>

        </div>
    `;
}


/**
 * Attach event listeners for all control bar elements.
 * Called once after page shell is rendered.
 */
function attachControlListeners(allFilteredTraits) {
    const availableModels = window.state.availableComparisonModels || [];
    const controlBar = document.querySelector('.cb');
    if (!controlBar) return;

    // --- Smooth pill ---
    controlBar.addEventListener('click', (e) => {
        const btn = e.target.closest('.smooth-pill button');
        if (btn) setSmoothingWindow(parseInt(btn.dataset.smooth));
    });

    // --- Mode segmented control ---
    const modeControl = document.getElementById('mode-control');
    if (modeControl) {
        modeControl.addEventListener('click', (e) => {
            const btn = e.target.closest('button[data-mode]');
            if (btn) setProjectionMode(btn.dataset.mode);
        });
    }

    // --- Compare segmented control ---
    const compareControl = document.getElementById('compare-control');
    if (compareControl) {
        compareControl.addEventListener('click', (e) => {
            const btn = e.target.closest('button[data-compare]');
            if (!btn || btn.disabled) return;
            const mode = btn.dataset.compare;
            const variantSelect = document.getElementById('compare-variant-select');
            const selectedModel = variantSelect ? variantSelect.value : availableModels[0] || '';
            if (mode === 'main') {
                setCompareMode('main');
            } else if (mode === 'diff') {
                if (selectedModel) setCompareMode('diff:' + selectedModel);
            } else if (mode === 'show') {
                if (selectedModel) setCompareMode('show:' + selectedModel);
            }
        });
    }

    // --- Model dropdown ---
    const variantSelect = document.getElementById('compare-variant-select');
    if (variantSelect) {
        variantSelect.addEventListener('change', () => {
            const val = variantSelect.value;
            window.state.lastCompareVariant = val;
            localStorage.setItem('lastCompareVariant', val);
            const currentMode = window.state.compareMode || 'main';
            if (currentMode.startsWith('diff:')) {
                setCompareMode('diff:' + val);
            } else if (currentMode.startsWith('show:')) {
                setCompareMode('show:' + val);
            }
        });
    }

    // --- Advanced toggle ---
    const advToggle = document.getElementById('td-advanced-toggle');
    if (advToggle) {
        advToggle.addEventListener('click', () => {
            const advRow = document.getElementById('td-advanced-row');
            if (!advRow) return;
            const expanded = advToggle.getAttribute('aria-expanded') === 'true';
            advToggle.setAttribute('aria-expanded', !expanded);
            advRow.hidden = !advRow.hidden;
        });
    }

    // --- Method checkboxes ---
    controlBar.querySelectorAll('.method-filter').forEach(cb => {
        cb.addEventListener('change', () => {
            toggleMethod(cb.dataset.method);
        });
    });

    // --- Centered ---
    const centeredToggle = document.getElementById('projection-centered-toggle');
    if (centeredToggle) {
        centeredToggle.addEventListener('change', () => {
            setProjectionCentered(centeredToggle.checked);
        });
    }

    // --- Massive dims cleaning ---
    const cleanSelect = document.getElementById('massive-dims-cleaning-select');
    if (cleanSelect) {
        cleanSelect.addEventListener('change', () => {
            setMassiveDimsCleaning(cleanSelect.value);
        });
    }

    // --- Layer mode ---
    const layerToggle = document.getElementById('layer-mode-toggle');
    if (layerToggle) {
        layerToggle.addEventListener('change', () => {
            setLayerMode(layerToggle.checked);
        });
    }

    const layerTraitSelect = document.getElementById('layer-mode-trait-select');
    if (layerTraitSelect) {
        layerTraitSelect.addEventListener('change', () => {
            setLayerModeTrait(layerTraitSelect.value);
        });
    }

    // --- Wide mode ---
    const wideToggle = document.getElementById('wide-mode-toggle');
    if (wideToggle) {
        wideToggle.addEventListener('change', () => {
            setWideMode(wideToggle.checked);
        });
    }

    // --- Velocity ---
    const velocityToggle = document.getElementById('velocity-toggle');
    if (velocityToggle) {
        velocityToggle.addEventListener('change', () => {
            setShowVelocity(velocityToggle.checked);
        });
    }

    // --- Collapsible section headers ---
    document.querySelectorAll('.sec-header[data-section]').forEach(header => {
        header.addEventListener('click', () => {
            const section = header.dataset.section;
            const body = document.getElementById('section-body-' + section);
            if (!body) return;
            const arrow = header.querySelector('.arrow');
            const wasHidden = body.hidden;
            body.hidden = !wasHidden;
            if (arrow) arrow.textContent = wasHidden ? '\u25BC' : '\u25B6';
            // Plotly charts need resize after reveal
            if (wasHidden && ['heatmap', 'magnitude', 'correlation'].includes(section)) {
                window.dispatchEvent(new Event('resize'));
            }
        });
    });
}

/**
 * Render page shell and attach listeners.
 */
function renderPageShell(contentArea, allFilteredTraits) {
    contentArea.innerHTML = buildPageShellHtml(allFilteredTraits);
    setupSubsectionInfoToggles();
    attachControlListeners(allFilteredTraits);
}

export { renderPageShell };
