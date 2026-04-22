/**
 * Model Analysis view — orchestrator.
 *
 * Sections:
 * 1. Activation Diagnostics: Magnitude by layer, massive activations (Sun et al. 2024)
 * 2. Variant Comparison: Effect size (Cohen's d) from pre-computed model_diff results
 *
 * This file just composes the page; actual rendering lives in section files.
 */

import { requireExperiment, renderSubsection, deferredLoading } from '../../core/ui.js';
import { wireStyledSelect } from '../../components/styled-select.js';
import {
    populateVariantDropdown,
    fetchMassiveActivationsData,
} from './model-analysis-data.js';
import {
    renderDiagnosticsSectionHtml,
    renderAllDiagnostics,
} from './section-diagnostics.js';
import { renderModelDiffComparison } from './section-variant-comparison.js';

/**
 * Main render function
 */
async function renderModelAnalysis() {
    const contentArea = document.getElementById('content-area');

    if (requireExperiment(contentArea)) return;

    const { cancel } = deferredLoading(contentArea, 'Loading model analysis...');
    const experiment = window.state.currentExperiment;
    cancel();

    // Render UI with both sections
    contentArea.innerHTML = `
        <div class="tool-view">
            <div class="page-intro">
                <div class="page-intro-text">Understanding model internals and comparing model variants.</div>
            </div>

            ${renderDiagnosticsSectionHtml()}

            <section>
                ${renderSubsection({
                    num: 2,
                    title: 'Variant Comparison',
                    infoId: 'info-variant-comparison',
                    infoText: 'Two variants process the same tokens; we project onto trait vectors layerwise and measure how far B drifts from A along each trait direction.'
                })}

                <div id="model-diff-container">
                    <div class="loading">Loading model diff data...</div>
                </div>
            </section>
        </div>
    `;

    window.setupSubsectionInfoToggles?.();

    // Wire inline styled selects (criteria); variant select is wired in populateVariantDropdown.
    wireStyledSelect(contentArea);

    // Populate model variant dropdown — re-renders all 4 diagnostics on variant change.
    await populateVariantDropdown(experiment, async () => {
        const data = await fetchMassiveActivationsData();
        renderAllDiagnostics(data);
    });

    // Initial render: fetch calibration data once, pass to all diagnostic renderers
    const calibrationData = await fetchMassiveActivationsData();
    renderAllDiagnostics(calibrationData);

    // Render model diff comparison
    await renderModelDiffComparison(experiment);
}

/** Reset model-analysis local state (called on experiment change). */
function resetModelAnalysisState() {
    // No module-local caches currently — stub is here for symmetry
    // with the other views, so state.js can call it unconditionally.
}

export { renderModelAnalysis, resetModelAnalysisState };

window.renderModelAnalysis = renderModelAnalysis;
window.resetModelAnalysisState = resetModelAnalysisState;
