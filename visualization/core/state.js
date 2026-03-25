/**
 * Global State Management for Trait Interpretation Visualization
 *
 * This file manages:
 * - Global state object
 * - Preference initialization/persistence (table-driven)
 * - Experiment data loading
 * - URL routing
 * - Application initialization
 *
 * UI code has been extracted to:
 * - components/sidebar.js (theme, navigation, trait checkboxes, GPU status, experiment list)
 * - core/display.js (colors, display names, Plotly layouts)
 * - core/utils.js (formatters, error display)
 */

import { showError, initMarkedOptions } from './utils.js';

// View category constants
const ANALYSIS_VIEWS = ['extraction', 'steering', 'trait-dynamics', 'model-analysis'];

// Experiments hidden from picker by default (can be revealed via toggle)
const HIDDEN_EXPERIMENTS = [];  // Add experiment names to hide by default

// State
const state = {
    // App-wide config (fetched from /api/config)
    appConfig: null,  // { mode, features, defaults }
    experiments: [],
    currentExperiment: null,
    experimentData: null,
    currentView: 'overview',
    selectedTraits: new Set(),
    // Prompt state structure
    currentPromptSet: null,      // e.g., 'single_trait'
    currentPromptId: null,       // e.g., 1
    availablePromptSets: {},     // { 'single_trait': [{id, text, note}, ...], ... }
    promptsWithData: {},         // { 'single_trait': [1, 2, 3], ... } - which prompts have inference data
    variantsPerPromptSet: {},    // { 'single_trait': ['instruct', 'base'], ... } - model variants with projection data
    availableComparisonModels: [], // Model variants available for comparison (excludes main app variant)
    // Token selection for per-token analysis
    currentTokenIndex: 0,        // Currently selected token index (0-based, absolute across prompt+response)
    // Cached inference context (prompt/response text for current selection)
    promptPickerCache: null,  // { promptSet, promptId, promptText, responseText, promptTokens, responseTokens, allTokens, nPromptTokens }
    // Layer Deep Dive settings
    // Projection normalization mode
    smoothingWindow: 6,      // Moving average window size (0 = off)
    projectionCentered: true,  // Subtract BOS token value (centers around 0)
    // Method filter for trait dynamics (which extraction methods to show)
    selectedMethods: new Set(['probe', 'mean_diff']),
    // Projection normalization mode: 'cosine' or 'normalized'
    projectionMode: 'cosine',
    // Massive dims cleaning mode
    massiveDimsCleaning: 'top5-3layers',
    // Compare mode for model comparison
    compareMode: 'main',
    lastCompareVariant: null,  // Persist organism selection across mode toggles
    // Top Spans panel state
    spanWindowLength: 10,      // Sliding window length for max-activating spans
    spanTrait: null,           // Which trait to rank by (reset on experiment change)
    spanScope: 'current',      // 'current' or 'allPrompts'
    spanMode: 'window',        // 'window' or 'clauses'
    spanPanelOpen: false,      // Whether the Top Spans panel is expanded
    traitHeatmapOpen: false,   // Whether the Trait × Token heatmap is expanded
    // Sentence overlay toggles (thought branches)
    showCuePOverlay: true,       // cue_p gradient bands (default on, was always-on)
    showCategoryOverlay: false,  // sentence category bands
    // Velocity overlay on trajectory chart
    showVelocity: false,
    // Layer mode: show single trait across all available layers
    layerMode: false,
    layerModeTrait: null,  // trait.name string, reset on experiment change
    // Experiment filtering (GPU status lives in sidebar.js)
    showAllExperiments: false,
    // Prompt set sidebar (left panel for inference views)
    promptSetSidebarOpen: false,
    // Layout mode
    wideMode: false,
    // Last selected analysis view (for Analysis entry point)
    lastAnalysisView: 'extraction'
};

// =============================================================================
// Helper Functions
// =============================================================================

function getFilteredTraits() {
    if (!state.experimentData || !state.experimentData.traits) return [];
    return state.experimentData.traits.filter(trait => state.selectedTraits.has(trait.name));
}

// =============================================================================
// Table-Driven Preference Management
// =============================================================================
// Each entry: { key, stateKey, type, default, onSet? }
//   type: 'bool' (stored as string 'true'/'false'), 'string', 'int'
//   onSet: optional callback after setting value (receives the new value)
//   clamp: optional [min, max] for int types
//   validate: optional fn(val) => sanitized val, for enum-like strings
//
// Preferences with identical init/set patterns are declared here.
// Non-standard preferences (selectedMethods, wideMode, compareMode, layerModeTrait)
// are handled separately below.

const renderView = () => { if (window.renderView) window.renderView(); };

const PREFERENCES = [
    // Smoothing
    { key: 'smoothingWindow',     stateKey: 'smoothingWindow',     type: 'int',    default: 6,    clamp: [0, 25], onSet: renderView },
    // Projection
    { key: 'projectionCentered',  stateKey: 'projectionCentered',  type: 'bool',   default: true,           onSet: renderView },
    { key: 'projectionMode',      stateKey: 'projectionMode',      type: 'string', default: 'cosine',       onSet: renderView },
    { key: 'massiveDimsCleaning', stateKey: 'massiveDimsCleaning', type: 'string', default: 'top5-3layers', onSet: renderView },
    // Layer mode
    { key: 'layerMode',           stateKey: 'layerMode',           type: 'bool',   default: false,          onSet: renderView },
    // Sidebar
    { key: 'promptSetSidebarOpen', stateKey: 'promptSetSidebarOpen', type: 'bool', default: true },
    // Compare
    { key: 'lastCompareVariant',  stateKey: 'lastCompareVariant',  type: 'string', default: null },
    // Top Spans
    { key: 'spanWindowLength',    stateKey: 'spanWindowLength',    type: 'int',    default: 10, clamp: [1, 100] },
    { key: 'spanPanelOpen',       stateKey: 'spanPanelOpen',       type: 'bool',   default: false },
    { key: 'traitHeatmapOpen',    stateKey: 'traitHeatmapOpen',    type: 'bool',   default: false },
    { key: 'spanScope',           stateKey: 'spanScope',           type: 'string', default: 'current',  validate: v => v === 'allPrompts' ? 'allPrompts' : 'current' },
    { key: 'spanMode',            stateKey: 'spanMode',            type: 'string', default: 'window',   validate: v => v === 'clauses' ? 'clauses' : 'window' },
    // Overlays
    { key: 'showCuePOverlay',     stateKey: 'showCuePOverlay',     type: 'bool',   default: true,  onSet: renderView },
    { key: 'showCategoryOverlay', stateKey: 'showCategoryOverlay', type: 'bool',   default: false, onSet: renderView },
    { key: 'showVelocity',        stateKey: 'showVelocity',        type: 'bool',   default: false, onSet: renderView },
];

/** Read a preference from localStorage into state */
function initPreference(pref) {
    const saved = localStorage.getItem(pref.key);
    if (pref.type === 'bool') {
        state[pref.stateKey] = saved === null ? !!pref.default : saved === 'true';
    } else if (pref.type === 'int') {
        state[pref.stateKey] = saved ? parseInt(saved) : pref.default;
    } else {
        state[pref.stateKey] = saved || pref.default;
    }
}

/** Write a preference to state + localStorage, then fire onSet callback */
function setPreference(key, value) {
    const pref = PREFERENCES.find(p => p.key === key);
    if (!pref) { console.warn(`[State] Unknown preference: ${key}`); return; }

    if (pref.type === 'bool') {
        value = !!value;
    } else if (pref.type === 'int') {
        value = parseInt(value) || pref.default;
        if (pref.clamp) value = Math.max(pref.clamp[0], Math.min(pref.clamp[1], value));
    } else if (pref.validate) {
        value = pref.validate(value);
    }
    value = value ?? pref.default;

    state[pref.stateKey] = value;
    localStorage.setItem(pref.key, value);
    if (pref.onSet) pref.onSet(value);
}

function initAllPreferences() {
    PREFERENCES.forEach(initPreference);
}

// --- Convenience setters (thin wrappers exposing the same API as before) ---
function setSmoothingWindow(size) { setPreference('smoothingWindow', size); }
function setProjectionCentered(centered) { setPreference('projectionCentered', centered); }
function setProjectionMode(mode) { setPreference('projectionMode', mode); }
function setMassiveDimsCleaning(mode) { setPreference('massiveDimsCleaning', mode); }
function setLayerMode(enabled) { setPreference('layerMode', enabled); }
function setPromptSetSidebarOpen(open) { setPreference('promptSetSidebarOpen', open); }
function setSpanWindowLength(length) { setPreference('spanWindowLength', length); }
function setSpanScope(scope) { setPreference('spanScope', scope); }
function setSpanMode(mode) { setPreference('spanMode', mode); }
function setSpanPanelOpen(open) { setPreference('spanPanelOpen', open); }
function setTraitHeatmapOpen(open) { setPreference('traitHeatmapOpen', open); }
function setShowCuePOverlay(enabled) { setPreference('showCuePOverlay', enabled); }
function setShowCategoryOverlay(enabled) { setPreference('showCategoryOverlay', enabled); }
function setShowVelocity(enabled) { setPreference('showVelocity', enabled); }

// --- Non-standard preferences (custom init/set logic) ---

function setWideMode(enabled) {
    state.wideMode = !!enabled;
    localStorage.setItem('wideMode', state.wideMode);
    // Toggle class on existing tool-view without full re-render
    const toolView = document.querySelector('.tool-view');
    if (toolView) {
        toolView.classList.toggle('wide-mode', state.wideMode);
        // Trigger Plotly resize for responsive charts
        window.dispatchEvent(new Event('resize'));
    }
}

function setLayerModeTrait(traitName) {
    state.layerModeTrait = traitName || null;
    if (window.renderView) window.renderView();
}

function setCompareMode(mode) {
    state.compareMode = mode || 'main';
    localStorage.setItem('compareMode', state.compareMode);
    // Re-render prompt picker so diff-availability indicators update
    if (window.renderPromptPicker) window.renderPromptPicker();
    if (window.renderPromptSetSidebar) window.renderPromptSetSidebar();
    if (window.renderView) window.renderView();
}

function initNonStandardPreferences() {
    // Wide mode
    state.wideMode = localStorage.getItem('wideMode') === 'true';
    // Compare mode
    state.compareMode = localStorage.getItem('compareMode') || 'main';
    // Selected methods (JSON-serialized Set)
    const saved = localStorage.getItem('selectedMethods');
    if (saved) {
        try {
            const methods = JSON.parse(saved);
            if (Array.isArray(methods) && methods.length > 0) {
                state.selectedMethods = new Set(methods);
            }
        } catch (e) { /* keep default */ }
    }
}

function toggleMethod(method) {
    if (state.selectedMethods.has(method)) {
        state.selectedMethods.delete(method);
    } else {
        state.selectedMethods.add(method);
    }
    localStorage.setItem('selectedMethods', JSON.stringify([...state.selectedMethods]));
    if (window.renderView) window.renderView();
}

// =============================================================================
// App Config Loading
// =============================================================================

async function loadAppConfig() {
    try {
        const response = await fetch('/api/config');
        state.appConfig = await response.json();
    } catch (e) {
        console.error('[State] Failed to load app config:', e);
        // Default to development mode
        state.appConfig = {
            mode: 'development',
            features: {
                model_picker: true,
                experiment_picker: true,
                inference_toggle: true,
                debug_info: true,
                steering: true
            },
            defaults: {
                inference_backend: 'local',
                experiment: 'live-chat',
                model: 'google/gemma-2-2b-it'
            }
        };
    }
}

function isFeatureEnabled(feature) {
    return state.appConfig?.features?.[feature] ?? true;
}

// =============================================================================
// Experiment Loading
// =============================================================================

async function loadExperiments() {
    try {
        const response = await fetch('/api/experiments');
        const data = await response.json();
        state.experiments = data.experiments || [];

        // Render experiment list in sidebar (DOM manipulation lives in sidebar.js)
        window.renderExperimentList(state.experiments, HIDDEN_EXPERIMENTS);

        if (state.experiments.length > 0) {
            const urlExp = getExperimentFromURL();
            const viewNeedsExperiment = ANALYSIS_VIEWS.includes(state.currentView);

            if (urlExp && state.experiments.includes(urlExp)) {
                state.currentExperiment = urlExp;
                window.renderExperimentList(state.experiments, HIDDEN_EXPERIMENTS, urlExp);
                await loadExperimentData(state.currentExperiment);
            } else if (viewNeedsExperiment) {
                state.currentExperiment = state.experiments[0];
                setExperimentInURL(state.currentExperiment);
                await loadExperimentData(state.currentExperiment);
            } else {
                state.currentExperiment = null;
                state.experimentData = null;
            }
        }
    } catch (error) {
        console.error('Error loading experiments:', error);
        showError('Failed to load experiments');
    }
}

async function ensureExperimentLoaded() {
    if (state.currentExperiment) return;
    if (!state.experiments || state.experiments.length === 0) return;
    if (!ANALYSIS_VIEWS.includes(state.currentView)) return;

    state.currentExperiment = state.experiments[0];
    setExperimentInURL(state.currentExperiment);
    await loadExperimentData(state.currentExperiment);

    // Re-render experiment list with first experiment active
    window.renderExperimentList(state.experiments, HIDDEN_EXPERIMENTS, state.currentExperiment);
}

async function loadExperimentData(experimentName) {
    const contentArea = document.getElementById('content-area');
    if (contentArea) {
        contentArea.innerHTML = '<div class="loading">Loading experiment data...</div>';
    }

    // Reset view-specific state on experiment change
    state.layerModeTrait = null;
    state.spanTrait = null;
    if (window.resetSteeringState) window.resetSteeringState();
    if (window.resetCorrelationState) window.resetCorrelationState();

    try {
        state.experimentData = {
            name: experimentName,
            traits: [],
            experimentConfig: null
        };

        window.paths.setExperiment(experimentName);

        // Load experiment config.json
        try {
            const configResponse = await fetch(`/api/experiments/${experimentName}/config`);
            if (configResponse.ok) {
                state.experimentData.experimentConfig = await configResponse.json();
            }
        } catch (e) {
            console.warn(`No experiment config.json for ${experimentName}:`, e.message);
        }

        // Load model config for this experiment
        try {
            await window.paths.loadModelConfig(experimentName);
        } catch (e) {
            // Model config is optional
        }

        // Fetch traits from API
        const traitsResponse = await fetch(`/api/experiments/${experimentName}/traits`);
        if (!traitsResponse.ok) {
            throw new Error(`Failed to fetch traits: ${traitsResponse.statusText}`);
        }
        const traitsData = await traitsResponse.json();
        const traitNames = traitsData.traits || [];

        for (const traitName of traitNames) {
            state.experimentData.traits.push({ name: traitName });
        }

        window.populateTraitCheckboxes();
        await discoverAvailablePrompts();

    } catch (error) {
        console.error('Error loading experiment data:', error);
        showError(`Failed to load experiment: ${experimentName}`);
    }
}

async function discoverAvailablePrompts() {
    state.availablePromptSets = {};
    state.promptsWithData = {};
    state.variantsPerPromptSet = {};  // Track which model variants have data per prompt set

    if (!state.currentExperiment) {
        return;
    }

    try {
        const response = await fetch(`/api/experiments/${state.currentExperiment}/inference/prompt-sets`);
        if (!response.ok) {
            console.warn('Failed to fetch prompt sets');
            return;
        }

        const data = await response.json();
        const promptSets = data.prompt_sets || [];

        for (const ps of promptSets) {
            state.availablePromptSets[ps.name] = ps.prompts || [];
            state.promptsWithData[ps.name] = ps.available_ids || [];
            state.variantsPerPromptSet[ps.name] = ps.variants_with_data || [];
        }
    } catch (e) {
        console.error('Error fetching prompt sets:', e);
    }

    // Restore from localStorage or set default selection
    const savedPromptSet = localStorage.getItem('promptSet');
    const savedPromptId = localStorage.getItem('promptId');

    state.currentPromptSet = null;
    state.currentPromptId = null;

    const isReplaySuffix = state.experimentData?.experimentConfig?.diff_convention === 'replay_suffix';
    const setsWithData = Object.entries(state.promptsWithData)
        .filter(([name, ids]) => ids.length > 0)
        .filter(([name]) => !isReplaySuffix || !name.includes('_replay_'))
        .sort(([a], [b]) => {
            const aHasSingle = a.includes('single');
            const bHasSingle = b.includes('single');
            if (aHasSingle && !bHasSingle) return -1;
            if (!aHasSingle && bHasSingle) return 1;
            return a.localeCompare(b);
        });

    if (savedPromptSet && state.promptsWithData[savedPromptSet]?.length > 0
        && (!isReplaySuffix || !savedPromptSet.includes('_replay_'))) {
        state.currentPromptSet = savedPromptSet;
        const savedId = savedPromptId != null ? String(savedPromptId) : null;
        if (savedId != null && state.promptsWithData[savedPromptSet].includes(savedId)) {
            state.currentPromptId = savedId;
        } else {
            state.currentPromptId = state.promptsWithData[savedPromptSet][0];
        }
    } else if (setsWithData.length > 0) {
        const [setName, promptIds] = setsWithData[0];
        state.currentPromptSet = setName;
        state.currentPromptId = promptIds[0];
    }

    // Update available comparison models for current prompt set
    updateAvailableComparisonModels();

    window.renderPromptPicker();
}

/**
 * Update availableComparisonModels based on current prompt set.
 * Excludes the main application variant so only "other" variants are available for comparison.
 */
function updateAvailableComparisonModels() {
    const appVariant = state.experimentData?.experimentConfig?.defaults?.application || 'instruct';
    const variants = state.variantsPerPromptSet?.[state.currentPromptSet] || [];

    // Filter out the main application variant - we want to compare against others
    let compModels = variants.filter(v => v !== appVariant);

    // For replay_suffix: only keep organisms that have instruct replay data
    const isReplaySuffix = state.experimentData?.experimentConfig?.diff_convention === 'replay_suffix';
    if (isReplaySuffix && state.currentPromptSet) {
        compModels = compModels.filter(org => {
            const replaySet = `${state.currentPromptSet}_replay_${org}`;
            const replayVariants = state.variantsPerPromptSet?.[replaySet] || [];
            return replayVariants.includes(appVariant);
        });
    }

    state.availableComparisonModels = compModels;

    // If current compare mode references a variant that's no longer available, reset to main
    // Exception: 'diff:replay' is a special flag for replay_suffix convention (not a real variant)
    if (state.compareMode !== 'main' && state.compareMode !== 'diff:replay') {
        const modeVariant = state.compareMode.replace('diff:', '').replace('show:', '');
        if (!state.availableComparisonModels.includes(modeVariant)) {
            state.compareMode = 'main';
        }
    }
}

/**
 * Get the model variant that has data for the current prompt set.
 * Prefers the default application variant; falls back to first available.
 */
function getVariantForCurrentPromptSet() {
    const appVariant = state.experimentData?.experimentConfig?.defaults?.application || 'instruct';
    const variants = state.variantsPerPromptSet?.[state.currentPromptSet] || [];
    if (variants.length === 0 || variants.includes(appVariant)) {
        return appVariant;
    }
    return variants[0];
}

// =============================================================================
// URL Routing
// =============================================================================

function getTabFromURL() {
    const params = new URLSearchParams(window.location.search);
    return params.get('tab') || 'overview';
}

function getExperimentFromURL() {
    const params = new URLSearchParams(window.location.search);
    return params.get('exp') || null;
}

function setTabInURL(tabName) {
    const url = new URL(window.location);
    url.searchParams.set('tab', tabName);
    window.history.pushState({ tab: tabName, exp: state.currentExperiment }, '', url);
}

function setExperimentInURL(expName) {
    const url = new URL(window.location);
    url.searchParams.set('exp', expName);
    window.history.replaceState({ tab: state.currentView, exp: expName }, '', url);
}

function initFromURL() {
    const tab = getTabFromURL();
    state.currentView = tab;

    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    const activeNav = document.querySelector(`.nav-item[data-view="${tab}"]`);
    if (activeNav) activeNav.classList.add('active');

    // If loading into an analysis view, also highlight the Analysis entry point
    if (ANALYSIS_VIEWS.includes(tab)) {
        state.lastAnalysisView = tab;
        const analysisEntry = document.getElementById('analysis-entry');
        if (analysisEntry) analysisEntry.classList.add('active');
    }
}

// Handle browser back/forward buttons
window.addEventListener('popstate', async () => {
    initFromURL();
    window.updateExperimentVisibility();
    await ensureExperimentLoaded();
    window.renderPromptPicker();
    if (window.renderView) window.renderView();
});

// =============================================================================
// Initialize Application
// =============================================================================

async function init() {
    await window.paths.load();
    await loadAppConfig();
    initMarkedOptions();

    // Initialize preferences (table-driven + non-standard)
    window.initTheme();
    initAllPreferences();
    initNonStandardPreferences();

    // Setup UI
    window.setupNavigation();
    await loadExperiments();
    window.setupSidebarEventListeners();

    // Start GPU status polling (only in dev mode where local inference is available)
    if (isFeatureEnabled('debug_info')) {
        window.startGpuPolling(5000);  // Poll every 5 seconds
    } else {
        // Just fetch once in production to show device info
        window.fetchGpuStatus();
    }

    // Read tab from URL and render
    initFromURL();
    window.updateExperimentVisibility();
    await ensureExperimentLoaded();
    window.renderPromptPicker();
    if (window.renderPromptSetSidebar) window.renderPromptSetSidebar();
    if (window.renderView) window.renderView();

    // Show reset button in dev mode
    if (isFeatureEnabled('debug_info')) {
        const resetBtn = document.getElementById('reset-storage-btn');
        if (resetBtn) resetBtn.style.display = 'block';
    }
}

// =============================================================================
// Experiment Visibility Toggle
// =============================================================================

function toggleHiddenExperiments() {
    state.showAllExperiments = !state.showAllExperiments;
    // Re-render experiment list with current selection
    window.renderExperimentList(state.experiments, HIDDEN_EXPERIMENTS, state.currentExperiment);
}

// =============================================================================
// ES module exports
export {
    state,
    ANALYSIS_VIEWS,
    getFilteredTraits,
    isFeatureEnabled,
    init as initApp,
    setWideMode,
    setSmoothingWindow,
    setProjectionCentered,
    setProjectionMode,
    setMassiveDimsCleaning,
    setLayerMode,
    setLayerModeTrait,
    setPromptSetSidebarOpen,
    setCompareMode,
    toggleMethod,
    setSpanWindowLength,
    setSpanScope,
    setSpanMode,
    setSpanPanelOpen,
    setTraitHeatmapOpen,
    setShowCuePOverlay,
    setShowCategoryOverlay,
    setShowVelocity,
    setTabInURL,
    setExperimentInURL,
    getTabFromURL,
    getExperimentFromURL,
    loadExperimentData,
    ensureExperimentLoaded,
    updateAvailableComparisonModels,
    getVariantForCurrentPromptSet,
};

// Keep window.* for remaining consumers (HTML onclick handlers, cross-module access)
window.state = state;
window.getFilteredTraits = getFilteredTraits;
window.initApp = init;
window.setPromptSetSidebarOpen = setPromptSetSidebarOpen;
window.updateAvailableComparisonModels = updateAvailableComparisonModels;
window.toggleHiddenExperiments = toggleHiddenExperiments;

// =============================================================================
// LocalStorage Reset (Dev Mode)
// =============================================================================

/**
 * All localStorage keys used by the app.
 * Centralized here for reset functionality and documentation.
 */
const LOCAL_STORAGE_KEYS = [
    // UI Preferences (state.js)
    'theme',
    'wideMode',
    'smoothingWindow',
    'projectionCentered',
    'projectionMode',
    'massiveDimsCleaning',
    'compareMode',
    'selectedMethods',
    'layerMode',
    'lastCompareVariant',
    'spanWindowLength',
    'spanPanelOpen',
    'traitHeatmapOpen',
    'spanScope',
    'spanMode',
    'showCuePOverlay',
    'showCategoryOverlay',
    'showVelocity',
    'promptSetSidebarOpen',
    // Prompt selection (prompt-picker.js)
    'promptSet',
    'promptId',
    'promptPickerCollapsed',
];

/**
 * Reset all localStorage keys to defaults.
 * Only available in development mode.
 */
function resetLocalStorage() {
    if (!isFeatureEnabled('debug_info')) {
        console.warn('[State] Reset only available in development mode');
        return;
    }

    // Clear known keys
    LOCAL_STORAGE_KEYS.forEach(key => localStorage.removeItem(key));

    // Clear dynamic keys (promptId_*, livechat_*)
    const keysToRemove = [];
    for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        if (key.startsWith('promptId_') || key.startsWith('livechat_')) {
            keysToRemove.push(key);
        }
    }
    keysToRemove.forEach(key => localStorage.removeItem(key));

    console.log('[State] localStorage reset. Cleared:', LOCAL_STORAGE_KEYS.length + keysToRemove.length, 'keys');

    // Reload to apply defaults
    location.reload();
}

// Keep window.* for backward compat (onclick in index.html)
window.resetLocalStorage = resetLocalStorage;
