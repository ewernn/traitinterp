// Controls for Trait Dynamics view — control bar HTML, page shell, event listeners
// Input: window.state, allFilteredTraits
// Output: HTML strings, attached DOM listeners

import { getDisplayName } from '../../core/display.js';
import { setupSubsectionInfoToggles } from '../../components/sidebar.js';
import {
    setSmoothing,
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
// Helpers
// =============================================================================

/**
 * Bind a change listener to an element by ID. No-op if element doesn't exist.
 */
function bindChange(id, fn) {
    const el = document.getElementById(id);
    if (el) el.addEventListener('change', fn);
}

// =============================================================================
// HTML builders
// =============================================================================

/**
 * Build the control bar HTML for Token Trajectory section.
 * Pure function of state — no data dependency.
 */
function buildControlBarHtml(allFilteredTraits) {
    const isSmoothing = window.state.smoothingEnabled !== false;
    const isCentered = window.state.projectionCentered !== false;
    const currentCompareMode = window.state.compareMode || 'main';
    const isDiffMode = currentCompareMode.startsWith('diff:');
    const availableModels = window.state.availableComparisonModels || [];
    const isReplaySuffix = window.state.experimentData?.experimentConfig?.diff_convention === 'replay_suffix';
    const currentCompareVariant = isDiffMode ? currentCompareMode.slice(5) : (window.state.lastCompareVariant || availableModels[0] || '');
    const selectedOrganism = window.state.lastCompareVariant || availableModels[0] || '';

    return `
        <div class="projection-toggle">
            ${ui.renderToggle({ id: 'smoothing-toggle', label: 'Smooth', checked: isSmoothing, className: 'projection-toggle-checkbox' })}
            ${isSmoothing ? `<select id="smoothing-window-select" style="margin-left: -4px; width: 42px; font-size: var(--text-xs);" title="Moving average window size (tokens)">
                ${[1,2,3,4,5,6,7,8,9,10,15,20,25].map(n => `<option value="${n}" ${n === (window.state.smoothingWindow || 5) ? 'selected' : ''}>${n}</option>`).join('')}
            </select>` : ''}
            ${ui.renderToggle({ id: 'projection-centered-toggle', label: 'Centered', checked: isCentered, className: 'projection-toggle-checkbox' })}
            <span class="projection-toggle-label" style="margin-left: 16px;">Methods:</span>
            ${ui.renderToggle({ label: 'probe', checked: window.state.selectedMethods.has('probe'), dataAttr: { key: 'method', value: 'probe' }, className: 'projection-toggle-checkbox method-filter' })}
            ${ui.renderToggle({ label: 'mean_diff', checked: window.state.selectedMethods.has('mean_diff'), dataAttr: { key: 'method', value: 'mean_diff' }, className: 'projection-toggle-checkbox method-filter' })}
            ${ui.renderToggle({ label: 'gradient', checked: window.state.selectedMethods.has('gradient'), dataAttr: { key: 'method', value: 'gradient' }, className: 'projection-toggle-checkbox method-filter' })}
            ${ui.renderToggle({ label: 'random', checked: window.state.selectedMethods.has('random'), dataAttr: { key: 'method', value: 'random' }, className: 'projection-toggle-checkbox method-filter' })}
        </div>
        <div class="projection-toggle">
            <span class="projection-toggle-label">Mode:</span>
            <select id="projection-mode-select" style="margin-left: 4px;" title="Cosine: proj/||h|| (removes magnitude). Normalized: proj/avg||h|| (preserves per-token variance, removes layer scale).">
                <option value="cosine" ${window.state.projectionMode === 'cosine' ? 'selected' : ''}>Cosine</option>
                <option value="normalized" ${window.state.projectionMode !== 'cosine' ? 'selected' : ''}>Normalized</option>
            </select>
            <span class="projection-toggle-label" style="margin-left: 12px;">Clean:</span>
            <select id="massive-dims-cleaning-select" style="margin-left: 4px;" title="Remove high-magnitude bias dimensions (Sun et al. 2024). These dims have 100-1000x larger values than typical dims and act as constant biases.">
                <option value="none" ${!window.state.massiveDimsCleaning || window.state.massiveDimsCleaning === 'none' ? 'selected' : ''}>No cleaning</option>
                <option value="top5-3layers" ${window.state.massiveDimsCleaning === 'top5-3layers' ? 'selected' : ''}>Top 5, 3+ layers</option>
                <option value="all" ${window.state.massiveDimsCleaning === 'all' ? 'selected' : ''}>All candidates</option>
            </select>
            <span class="projection-toggle-label" style="margin-left: 12px;">Layers:</span>
            ${ui.renderToggle({ id: 'layer-mode-toggle', label: '', checked: window.state.layerMode, className: 'projection-toggle-checkbox' })}
            ${window.state.layerMode ? `
            <select id="layer-mode-trait-select" style="margin-left: 4px;" title="Select trait to view across all available layers">
                ${allFilteredTraits.map(t =>
                    `<option value="${t.name}" ${t.name === window.state.layerModeTrait ? 'selected' : ''}>${getDisplayName(t.name)}</option>`
                ).join('')}
            </select>
            ` : ''}
            ${availableModels.length > 0 && isReplaySuffix ? `
            <span class="projection-toggle-label" style="margin-left: 12px;">Organism:</span>
            <select id="compare-variant-select" style="margin-left: 4px;">
                ${availableModels.map(m => `
                    <option value="${m}" ${m === selectedOrganism ? 'selected' : ''}>${m}</option>
                `).join('')}
            </select>
            ${ui.renderFilterChip('main', 'Main', isDiffMode ? '' : 'main', 'compare-mode')}
            ${ui.renderFilterChip('diff', 'Diff', isDiffMode ? 'diff' : '', 'compare-mode')}
            ` : availableModels.length > 0 ? `
            <span class="projection-toggle-label" style="margin-left: 12px;">Compare:</span>
            ${ui.renderFilterChip('main', 'Main', isDiffMode ? '' : 'main', 'compare-mode')}
            ${ui.renderFilterChip('diff', 'Diff', isDiffMode ? 'diff' : '', 'compare-mode')}
            ${isDiffMode ? `
            <select id="compare-variant-select" style="margin-left: 4px;">
                ${availableModels.map(m => `
                    <option value="${m}" ${m === currentCompareVariant ? 'selected' : ''}>${m}</option>
                `).join('')}
            </select>
            ` : ''}
            ` : ''}
            <span class="projection-toggle-label" style="margin-left: 12px;">Wide:</span>
            ${ui.renderToggle({ id: 'wide-mode-toggle', label: '', checked: window.state.wideMode, className: 'projection-toggle-checkbox' })}
            ${ui.renderToggle({ id: 'velocity-toggle', label: 'Velocity', checked: window.state.showVelocity, className: 'projection-toggle-checkbox' })}
        </div>
    `;
}


/**
 * Build full page shell HTML with controls and empty chart divs.
 * Renders independently of data — controls always accessible.
 */
function buildPageShellHtml(allFilteredTraits) {
    const projectionMode = window.state.projectionMode || 'cosine';
    const isCentered = window.state.projectionCentered !== false;
    const isSmoothing = window.state.smoothingEnabled !== false;

    return `
        <div class="tool-view${window.state.wideMode ? ' wide-mode' : ''}">
            <div class="page-intro">
                <div class="page-intro-text">Watch traits evolve token-by-token during generation.</div>
                <div id="trait-dynamics-status"></div>
            </div>

            <section>
                ${ui.renderSubsection({
                    title: 'Token Trajectory',
                    infoId: 'info-token-trajectory',
                    infoText: (projectionMode === 'normalized'
                        ? 'Normalized projection: proj / avg||h||. Preserves per-token variance, removes layer-dependent scale.'
                        : 'Cosine similarity: proj / ||h||. Shows directional alignment with trait vector.') +
                        (isCentered ? ' Centered by subtracting BOS token value.' : '') +
                        (isSmoothing ? ` Smoothed with ${window.state.smoothingWindow || 5}-token moving average.` : ''),
                    level: 'h2'
                })}
                ${buildControlBarHtml(allFilteredTraits)}
                <div id="overlay-controls"></div>
                <div id="combined-activation-plot"></div>
                <div id="top-spans-panel"></div>
            </section>

            <section id="cue-p-section" style="display:none">
                ${ui.renderSubsection({
                    title: 'Resampling cue_p',
                    infoId: 'info-cue-p',
                    infoText: 'Per-sentence resampling probability of the cued (wrong) answer, from Thought Branches transplant experiment (~4000 forward passes per sentence). Shows how bias accumulates through the CoT.'
                })}
                <div id="cue-p-plot"></div>
            </section>

            <section>
                <div id="trait-heatmap-panel"></div>
            </section>

            <section>
                ${ui.renderSubsection({
                    title: 'Activation Magnitude Per Token',
                    infoId: 'info-token-magnitude',
                    infoText: 'L2 norm of activation per token. Shows one line per unique layer used by traits above. Compare to trajectory - similar magnitudes but low projections means token encodes orthogonal information.'
                })}
                <div id="token-magnitude-plot"></div>
            </section>

            <section id="correlation-section" style="display: none;">
                ${ui.renderSubsection({
                    title: 'Trait Correlation',
                    infoId: 'info-correlation',
                    infoText: 'Cross-trait correlation analysis for the current prompt set. Shows token-level correlations (with offset slider), correlation decay over token distance, and response-level correlations.',
                    level: 'h2'
                })}
                <div id="correlation-content"></div>
            </section>

        </div>
    `;
}


/**
 * Attach event listeners for all control bar elements.
 * Called once after page shell is rendered.
 */
function attachControlListeners(allFilteredTraits) {
    const isReplaySuffix = window.state.experimentData?.experimentConfig?.diff_convention === 'replay_suffix';
    const availableModels = window.state.availableComparisonModels || [];

    bindChange('smoothing-toggle', () => {
        setSmoothing(document.getElementById('smoothing-toggle').checked);
    });
    bindChange('smoothing-window-select', () => {
        setSmoothingWindow(parseInt(document.getElementById('smoothing-window-select').value));
    });
    bindChange('projection-centered-toggle', () => {
        setProjectionCentered(document.getElementById('projection-centered-toggle').checked);
    });
    bindChange('massive-dims-cleaning-select', () => {
        setMassiveDimsCleaning(document.getElementById('massive-dims-cleaning-select').value);
    });
    document.querySelectorAll('.method-filter input').forEach(cb => {
        cb.addEventListener('change', () => {
            toggleMethod(cb.dataset.method);
        });
    });
    bindChange('projection-mode-select', () => {
        setProjectionMode(document.getElementById('projection-mode-select').value);
    });
    bindChange('velocity-toggle', () => {
        setShowVelocity(document.getElementById('velocity-toggle').checked);
    });
    // Compare mode toggle (Main/Diff chips)
    document.querySelectorAll('[data-compare-mode]').forEach(chip => {
        chip.addEventListener('click', () => {
            const mode = chip.dataset.compareMode;
            if (mode === 'main') {
                setCompareMode('main');
            } else {
                if (isReplaySuffix) {
                    setCompareMode('diff:replay');
                } else {
                    const variant = (window.state.lastCompareVariant && availableModels.includes(window.state.lastCompareVariant))
                        ? window.state.lastCompareVariant
                        : availableModels[0];
                    if (variant) setCompareMode('diff:' + variant);
                }
            }
        });
    });
    bindChange('compare-variant-select', () => {
        const val = document.getElementById('compare-variant-select').value;
        window.state.lastCompareVariant = val;
        localStorage.setItem('lastCompareVariant', val);
        if (isReplaySuffix) {
            if (window.renderView) window.renderView();
        } else {
            setCompareMode('diff:' + val);
        }
    });
    bindChange('layer-mode-toggle', () => {
        setLayerMode(document.getElementById('layer-mode-toggle').checked);
    });
    bindChange('layer-mode-trait-select', () => {
        setLayerModeTrait(document.getElementById('layer-mode-trait-select').value);
    });
    bindChange('wide-mode-toggle', () => {
        setWideMode(document.getElementById('wide-mode-toggle').checked);
    });
}

/**
 * Render page shell and attach listeners. Returns setupSubsectionInfoToggles for caller.
 */
function renderPageShell(contentArea, allFilteredTraits) {
    contentArea.innerHTML = buildPageShellHtml(allFilteredTraits);
    setupSubsectionInfoToggles();
    attachControlListeners(allFilteredTraits);
}

export { renderPageShell };
