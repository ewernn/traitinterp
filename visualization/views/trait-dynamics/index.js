// Trait Dynamics View — Orchestrator
// Watch the model's internal state evolve token-by-token during generation.
//
// Sections:
// 1. Token Trajectory: X=tokens, Y=cosine similarity (best layer) — velocity overlay toggle
// 2. Activation Magnitude Per Token: ||h|| at each token position

import { fetchJSON } from '../../core/utils.js';
import { getVariantForCurrentPromptSet } from '../../core/state.js';
import { renderPageShell } from './controls.js';
import { loadComparisonProjections, fetchLayerSensitivityData, processTraitProjectionData } from './data.js';
import { renderTrajectoryChart } from './chart-trajectory.js';
import { renderTraitTokenHeatmap } from './chart-heatmap.js';
import { renderTokenMagnitudePlot } from './chart-magnitude.js';
import { renderCorrelationSection } from '../correlation.js';

async function renderTraitDynamics() {
    const contentArea = document.getElementById('content-area');

    if (ui.requireExperiment(contentArea)) return;

    const allFilteredTraits = window.getFilteredTraits();

    // In layer mode, narrow to single trait (multi-vector expansion handles layers)
    let filteredTraits;
    if (window.state.layerMode && allFilteredTraits.length > 0) {
        // Default to first trait if none selected
        if (!window.state.layerModeTrait) {
            window.state.layerModeTrait = allFilteredTraits[0].name;
        }
        const match = allFilteredTraits.find(t => t.name === window.state.layerModeTrait);
        filteredTraits = match ? [match] : [allFilteredTraits[0]];
    } else {
        filteredTraits = allFilteredTraits;
    }

    // Preserve scroll position
    const scrollY = contentArea.scrollTop;

    // Always render page shell with controls (so they remain accessible even when no data loads)
    renderPageShell(contentArea, allFilteredTraits);

    if (filteredTraits.length === 0) {
        document.getElementById('combined-activation-plot').innerHTML =
            `<div class="info">Select at least one trait from the sidebar to view activation trajectories.</div>`;
        requestAnimationFrame(() => { contentArea.scrollTop = scrollY; });
        return;
    }

    const promptSet = window.state.currentPromptSet;
    const promptId = window.state.currentPromptId;

    // Detect replay_suffix convention: organism is the main variant, instruct is the replay comparison
    const isReplaySuffix = window.state.experimentData?.experimentConfig?.diff_convention === 'replay_suffix';
    const availableModelsEarly = window.state.availableComparisonModels || [];

    let modelVariant;
    if (isReplaySuffix && availableModelsEarly.length > 0) {
        // In replay_suffix mode, the selected organism is the main variant
        const selectedOrg = window.state.lastCompareVariant || availableModelsEarly[0];
        modelVariant = availableModelsEarly.includes(selectedOrg) ? selectedOrg : availableModelsEarly[0];
    } else {
        modelVariant = getVariantForCurrentPromptSet();
    }

    if (!promptSet || !promptId) {
        const promptLabel = promptSet ? `${promptSet}/\u2014` : 'none selected';
        document.getElementById('combined-activation-plot').innerHTML =
            `<div class="info">No data available for prompt ${promptLabel} for any selected trait.</div>`;
        requestAnimationFrame(() => { contentArea.scrollTop = scrollY; });
        return;
    }

    const { cancel: cancelLoading } = ui.deferredLoading('combined-activation-plot', `Loading data for ${filteredTraits.length} trait(s)...`);

    // Load shared response data (prompt/response text and tokens)
    const responseData = await fetchJSON(window.paths.responseData(promptSet, promptId, modelVariant));

    // Load projection data for ALL selected traits (in parallel)
    const projectionResults = await Promise.all(filteredTraits.map(async (trait) => {
        const projData = await fetchJSON(window.paths.residualStreamData(trait, promptSet, promptId, modelVariant));
        if (!projData || projData.error) return { trait, error: true };
        return { trait, projData };
    }));

    // Process results into traitData map (handles multi-vector expansion)
    const { traitData, failedTraits } = processTraitProjectionData(projectionResults, responseData);

    cancelLoading();

    // Handle compare mode: "main", "diff:{model}", or "show:{model}"
    const compareMode = window.state.compareMode || 'main';
    const isDiff = compareMode.startsWith('diff:');
    const isShow = compareMode.startsWith('show:');
    const compareModel = isDiff ? compareMode.slice(5) : isShow ? compareMode.slice(5) : null;

    // For replay_suffix: diff always compares organism (main) vs instruct replay
    const replayDiff = isReplaySuffix && (isDiff || (compareMode === 'diff'));
    const effectiveCompareModel = replayDiff ? null : compareModel;  // replay handles its own fetch

    if (replayDiff) {
        // Replay suffix convention: fetch instruct data from {promptSet}_replay_{organism}
        const appVariant = window.state.experimentData?.experimentConfig?.defaults?.application || 'instruct';
        const replayPromptSet = `${promptSet}_replay_${modelVariant}`;
        await loadComparisonProjections(
            traitData, Object.keys(traitData), replayPromptSet, promptId, appVariant,
            filteredTraits, 'diff', true /* invertDiff: main - comp */, appVariant
        );
    } else if (effectiveCompareModel) {
        const mode = isDiff ? 'diff' : isShow ? 'show' : null;
        if (mode) {
            await loadComparisonProjections(
                traitData, Object.keys(traitData), promptSet, promptId, effectiveCompareModel,
                filteredTraits, mode, false /* standard: comp - main */, effectiveCompareModel
            );
        }
    }

    // Layer mode: overlay multi-layer projections from layer_sensitivity data
    if (window.state.layerMode) {
        const layerSensData = await fetchLayerSensitivityData(
            window.state.currentExperiment, promptSet, promptId
        );
        const selectedTrait = window.state.layerModeTrait;
        const traitLayers = layerSensData?.traits?.[selectedTrait]?.layers;

        if (traitLayers) {
            // Determine which projection values to use
            const isDiffMode = (compareMode || 'main').startsWith('diff:') || replayDiff;
            // In replay_suffix: modelVariant is the organism (variant_b)
            // In standard: modelVariant is the application default (variant_a)
            const projKey = isDiffMode ? 'delta'
                : (modelVariant === layerSensData._variant_b ? 'proj_b' : 'proj_a');

            // Get prompt projections from the existing single-vector entry (if available)
            const existingEntry = traitData[selectedTrait];
            const promptProj = existingEntry?.projections?.prompt || [];

            // Remove the single-vector entry
            delete traitData[selectedTrait];

            // Create multi-vector entries from layer_sensitivity data
            for (const [layerStr, ldata] of Object.entries(traitLayers)) {
                const layer = parseInt(layerStr);
                const responseProj = ldata[projKey] || ldata.proj_a || [];
                const key = `${selectedTrait}__probe_L${layer}`;

                traitData[key] = {
                    projections: { prompt: promptProj, response: responseProj },
                    prompt: existingEntry?.prompt || responseData?.prompt || { text: '', tokens: [] },
                    response: existingEntry?.response || { text: '', tokens: layerSensData.response_tokens || [] },
                    token_norms: existingEntry?.token_norms || null,
                    metadata: {
                        vector_source: { layer, method: 'probe' },
                        _baseTrait: selectedTrait,
                        _isMultiVector: true,
                        ...(isDiffMode ? { _isDiff: true, _compareModel: layerSensData._variant_a } : {}),
                    },
                };
            }
        }
    }

    // Check if we have any data
    const loadedTraits = Object.keys(traitData);
    if (loadedTraits.length === 0) {
        const promptLabel = promptSet && promptId ? `${promptSet}/${promptId}` : 'none selected';
        document.getElementById('combined-activation-plot').innerHTML = `
            <div class="info">
                No data available for prompt ${promptLabel} for any selected trait.
            </div>
            <p class="tool-description">
                To capture per-token activation data, run:
            </p>
            <pre>python inference/capture_activations.py --experiment ${window.paths.getExperiment()} --prompt-set ${promptSet || 'PROMPT_SET'}</pre>
        `;
        requestAnimationFrame(() => { contentArea.scrollTop = scrollY; });
        return;
    }

    // Fetch annotations for shaded bands (non-blocking)
    let annotationTokenRanges = [];
    if (responseData) {
        const responseTokens = responseData.tokens
            ? responseData.tokens.slice(responseData.prompt_end)
            : responseData.response?.tokens || [];
        const responseText = typeof responseData.response === 'string'
            ? responseData.response
            : responseData.response?.text || '';
        if (responseTokens.length > 0 && responseText) {
            annotationTokenRanges = await window.annotations.getAnnotationTokenRanges(
                window.state.currentExperiment, modelVariant, promptSet, promptId,
                responseTokens, responseText
            );
        }
    }

    // Extract rollout/sentence metadata from response data
    const turnBoundaries = responseData?.turn_boundaries || null;
    const sentenceBoundaries = responseData?.sentence_boundaries || null;

    // Fetch sentence category annotations (thought branches only)
    let sentenceCategoryData = null;
    if (sentenceBoundaries?.length > 0) {
        sentenceCategoryData = await window.annotations.getSentenceCategoriesForPrompt(
            window.state.currentExperiment, promptSet, promptId, sentenceBoundaries);
    }

    // Build reference data from first loaded trait
    const refData = traitData[loadedTraits[0]];

    if (!refData.prompt?.tokens || !refData.response?.tokens) {
        document.getElementById('combined-activation-plot').innerHTML = `<div class="info">Error: Missing token data.</div>`;
        requestAnimationFrame(() => { contentArea.scrollTop = scrollY; });
        return;
    }

    const promptTokens = refData.prompt.tokens;
    const responseTokens = refData.response.tokens;
    const allTokens = [...promptTokens, ...responseTokens];
    const nPromptTokens = promptTokens.length;
    const isRollout = responseTokens.length === 0;
    const inferenceModel = refData.metadata?.inference_model ||
        window.state.experimentData?.experimentConfig?.application_model || 'unknown';

    // Build rendering context shared across chart renderers
    const renderCtx = {
        traitData, loadedTraits, failedTraits, annotationTokenRanges,
        turnBoundaries, sentenceBoundaries, sentenceCategoryData,
        isReplaySuffix, nPromptTokens, isRollout, allTokens,
        promptTokens, responseTokens, inferenceModel
    };

    // Render trajectory chart (main chart + velocity + overlay controls + top spans + cue_p)
    const result = renderTrajectoryChart(renderCtx);

    // Render heatmap and magnitude charts if trajectory produced data
    if (result) {
        const { traitActivations, filteredByMethod, tickVals, tickText, displayTokens } = result;

        // Render Trait x Token heatmap (all traits at once)
        renderTraitTokenHeatmap(traitActivations, filteredByMethod, tickVals, tickText, nPromptTokens, displayTokens, isRollout, turnBoundaries, sentenceBoundaries, traitData, sentenceCategoryData);

        // Render Token Magnitude plot (per-token norms)
        renderTokenMagnitudePlot(traitData, filteredByMethod, tickVals, tickText, nPromptTokens, isRollout, turnBoundaries, sentenceBoundaries, sentenceCategoryData);
    }

    // Render correlation section if data exists for this prompt set
    const corrSection = document.getElementById('correlation-section');
    if (corrSection) {
        const corrLoaded = await renderCorrelationSection('correlation-content', promptSet);
        if (corrLoaded) {
            corrSection.style.display = '';
            window.setupSubsectionInfoToggles();
        }
    }

    // Restore scroll position after DOM updates
    requestAnimationFrame(() => {
        contentArea.scrollTop = scrollY;
    });
}

// ES module exports
export { renderTraitDynamics };

// Keep window.* for router
window.renderTraitDynamics = renderTraitDynamics;
