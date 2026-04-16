// Controls for Inference view — control bar HTML, page shell, event listeners
// Input: window.state, allFilteredTraits
// Output: HTML strings, attached DOM listeners

import { getDisplayName } from '../../core/display.js';
import { setupSubsectionInfoToggles } from '../../components/sidebar.js';
import { renderSegmentedControl, renderSubsection } from '../../core/ui.js';
import { renderStyledSelect, wireStyledSelect } from '../../components/styled-select.js';
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
 * Advanced row (collapsed): Methods, Clean, Layers, Wide, Velocity.
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
            ${renderSegmentedControl({
                id: 'smooth-control',
                options: [
                    { value: 0, label: 'off' },
                    { value: 3, label: '3' },
                    { value: 6, label: '6' },
                    { value: 9, label: '9' },
                ],
                selected: window.state.smoothingWindow,
                dataAttr: 'smooth',
                size: 'compact',
            })}
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

    const modelDropdown = availableModels.length === 0
        ? `<span class="cb-label" style="opacity:0.5;">No models</span>`
        : renderStyledSelect({
            id: 'compare-variant-select',
            options: availableModels.map(m => ({ value: m, label: m })),
            selected: currentCompareVariant,
            disabled: modelDropdownDisabled,
            onChange: (val) => {
                window.state.lastCompareVariant = val;
                localStorage.setItem('lastCompareVariant', val);
                const currentMode = window.state.compareMode || 'main';
                if (currentMode.startsWith('diff:')) setCompareMode('diff:' + val);
                else if (currentMode.startsWith('show:')) setCompareMode('show:' + val);
            },
        });

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

    const layerTraitSelect = window.state.layerMode
        ? renderStyledSelect({
            id: 'layer-mode-trait-select',
            options: allFilteredTraits.map(t => ({ value: t.name, label: getDisplayName(t.name) })),
            selected: window.state.layerModeTrait,
            onChange: (val) => setLayerModeTrait(val),
        })
        : '';

    const centeredToggleHtml = `<label class="cb-checkbox" title="Subtract the mean response projection from every token. Removes per-response bias so constant-bias traits (e.g. golden_gate_bridge) show their relative variation.">
        <input type="checkbox" id="projection-centered-toggle" ${isCentered ? 'checked' : ''}> Mean-center
    </label>`;

    const advancedRow = `
            <div class="cb-row cb-advanced" id="td-advanced-row" hidden>
                ${compareCluster}
                <div class="cb-cluster">
                    <span class="cb-label">Methods:</span>
                    ${methodCheckboxes}
                </div>
                <div class="cb-cluster">
                    <span class="cb-label">Clean:</span>
                    ${renderStyledSelect({
                        id: 'massive-dims-cleaning-select',
                        options: [
                            { value: 'none', label: 'None' },
                            { value: 'top5-3layers', label: 'Top 5' },
                            { value: 'all', label: 'All' },
                        ],
                        selected: massiveDimsCleaning,
                        onChange: (val) => setMassiveDimsCleaning(val),
                    })}
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
                ${centeredToggleHtml}
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
    const experimentName = window.state.currentExperiment || 'EXPERIMENT';

    const infoText = 'Trait projection per token across the response, one line per trait. Higher means the residual aligns more strongly with the trait direction (cosine, normalized, or raw mode).';

    return `
        <div class="tool-view${window.state.wideMode ? ' wide-mode' : ''}">
            <div class="page-intro">
                <div class="page-intro-text">Watch traits evolve token-by-token during generation.</div>
                <div id="inference-status"></div>
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
                    <span class="arrow">\u25BC</span> Top Spans <span class="subsection-info-toggle" data-target="info-top-spans">\u25BA</span> <span class="sec-badge" id="badge-top-spans"></span>
                </div>
                <div class="subsection-info" id="info-top-spans">Top-ranked token spans where the main and comparison variants differ most on the trait. Positive delta means main expresses it more; negative means the comparison does.</div>
                <div id="section-body-top-spans">
                    <div id="top-spans-panel"></div>
                </div>
            </section>

            <section>
                <div class="sec-header" data-section="heatmap" id="sec-heatmap">
                    <span class="arrow">\u25B6</span> Trait \u00D7 Token Heatmap <span class="subsection-info-toggle" data-target="info-trait-token-heatmap">\u25BA</span> <span class="sec-badge" id="badge-heatmap"></span>
                </div>
                <div class="subsection-info" id="info-trait-token-heatmap">All selected traits as rows, tokens as columns, colored by projection value. Diverging scale around 0; red and blue mark strong positive and negative alignment.</div>
                <div id="section-body-heatmap" hidden>
                    <div id="trait-heatmap-panel"></div>
                </div>
            </section>

            <section>
                <div class="sec-header" data-section="magnitude" id="sec-magnitude">
                    <span class="arrow">\u25B6</span> Activation Magnitude <span class="subsection-info-toggle" data-target="info-activation-magnitude">\u25BA</span> <span class="sec-badge" id="badge-magnitude"></span>
                </div>
                <div class="subsection-info" id="info-activation-magnitude">L2 norm of the residual stream per token at each trait&#39;s best layer. Distinguishes genuinely orthogonal tokens from tokens with small residuals overall.</div>
                <div id="section-body-magnitude" hidden>
                    <div id="token-magnitude-plot"></div>
                </div>
            </section>

            <section id="cue-p-section" style="display:none">
                ${renderSubsection({
                    title: 'Resampling cue_p',
                    infoId: 'info-cue-p',
                    infoText: 'Per-sentence probability the model commits to the cued (wrong) answer if resampled from that point. Only shown for Thought Branches experiments with cue_p data.'
                })}
                <div id="cue-p-plot"></div>
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
        const btn = e.target.closest('button[data-smooth]');
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
            const selectedModel = window.state.lastCompareVariant || availableModels[0] || '';
            if (mode === 'main') {
                setCompareMode('main');
            } else if (mode === 'diff') {
                if (selectedModel) setCompareMode('diff:' + selectedModel);
            } else if (mode === 'show') {
                if (selectedModel) setCompareMode('show:' + selectedModel);
            }
        });
    }

    // --- Styled selects (compare-variant, layer-mode-trait, massive-dims-cleaning) ---
    wireStyledSelect(controlBar);

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

    // --- Layer mode ---
    const layerToggle = document.getElementById('layer-mode-toggle');
    if (layerToggle) {
        layerToggle.addEventListener('change', () => {
            setLayerMode(layerToggle.checked);
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
        header.addEventListener('click', (e) => {
            if (e.target.closest('.subsection-info-toggle')) return;
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
