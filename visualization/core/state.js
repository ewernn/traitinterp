/**
 * Global State Management for Trait Interpretation Visualization
 *
 * This file manages:
 * - Global state object
 * - Preference initialization/persistence
 * - Experiment data loading
 * - URL routing
 * - Application initialization
 *
 * UI code has been extracted to:
 * - components/sidebar.js (theme, navigation, trait checkboxes)
 * - core/display.js (colors, display names, Plotly layouts)
 * - core/utils.js (formatters, error display)
 */

// View category constants
const ANALYSIS_VIEWS = ['trait-extraction', 'steering-sweep', 'inference', 'model-comparison', 'layer-dive'];
const GLOBAL_VIEWS = ['overview', 'methodology', 'findings', 'finding', 'live-chat'];

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
    // Token selection for per-token analysis
    currentTokenIndex: 0,        // Currently selected token index (0-based, absolute across prompt+response)
    // Cached inference context (prompt/response text for current selection)
    promptPickerCache: null,  // { promptSet, promptId, promptText, responseText, promptTokens, responseTokens, allTokens, nPromptTokens }
    // Layer Deep Dive settings
    hideAttentionSink: true,  // Hide first token (attention sink) in heatmaps
    // Steering Sweep settings
    selectedSteeringTrait: null,  // Selected trait for single-trait sections (reset on experiment change)
    // Projection normalization mode
    smoothingEnabled: true,  // Apply 3-token moving average
    projectionCentered: true,  // Subtract BOS token value (centers around 0)
    // Method filter for trait dynamics (which extraction methods to show)
    selectedMethods: new Set(['probe', 'mean_diff', 'gradient', 'random']),
    // Massive dims cleaning mode
    massiveDimsCleaning: 'top5-3layers',
    // Compare mode for model comparison
    compareMode: 'main',
    // GPU status (fetched from /api/gpu-status)
    gpuStatus: null,  // { available, type, device, memory_total_gb, memory_used_gb, ... }
    // Experiment filtering
    showAllExperiments: false
};

// =============================================================================
// Helper Functions
// =============================================================================

function getFilteredTraits() {
    if (!state.experimentData || !state.experimentData.traits) return [];
    return state.experimentData.traits.filter(trait => state.selectedTraits.has(trait.name));
}

// =============================================================================
// Preference Management
// =============================================================================

// Smoothing Management
function initSmoothing() {
    const saved = localStorage.getItem('smoothingEnabled');
    state.smoothingEnabled = saved === null ? true : saved === 'true';
}

function setSmoothing(enabled) {
    state.smoothingEnabled = !!enabled;
    localStorage.setItem('smoothingEnabled', state.smoothingEnabled);
    if (window.renderView) window.renderView();
}

function initProjectionCentered() {
    const saved = localStorage.getItem('projectionCentered');
    state.projectionCentered = saved === null ? true : saved === 'true';
}

function setProjectionCentered(centered) {
    state.projectionCentered = !!centered;
    localStorage.setItem('projectionCentered', state.projectionCentered);
    if (window.renderView) window.renderView();
}

// Massive Dims Cleaning (dropdown: 'none', 'top5-3layers', 'all')
function initMassiveDimsCleaning() {
    const saved = localStorage.getItem('massiveDimsCleaning');
    state.massiveDimsCleaning = saved || 'top5-3layers';
}

function setMassiveDimsCleaning(mode) {
    state.massiveDimsCleaning = mode || 'none';
    localStorage.setItem('massiveDimsCleaning', state.massiveDimsCleaning);
    if (window.renderView) window.renderView();
}

// Compare Mode (Model Comparison)
function initCompareMode() {
    const savedMode = localStorage.getItem('compareMode');
    state.compareMode = savedMode || 'main';

    // Legacy migration from diffMode/diffModel
    const legacyDiffMode = localStorage.getItem('diffMode');
    const legacyDiffModel = localStorage.getItem('diffModel');
    if (legacyDiffMode === 'true' && legacyDiffModel) {
        state.compareMode = `diff:${legacyDiffModel}`;
        localStorage.removeItem('diffMode');
        localStorage.removeItem('diffModel');
        localStorage.setItem('compareMode', state.compareMode);
    }
}

function setCompareMode(mode) {
    state.compareMode = mode || 'main';
    localStorage.setItem('compareMode', state.compareMode);
    if (window.renderView) window.renderView();
}

// Hide Attention Sink Toggle
function initHideAttentionSink() {
    const saved = localStorage.getItem('hideAttentionSink');
    state.hideAttentionSink = saved === 'true';
}

function setHideAttentionSink(hide) {
    state.hideAttentionSink = !!hide;
    localStorage.setItem('hideAttentionSink', state.hideAttentionSink);
    if (window.renderView) window.renderView();
}

// Method Filter Management
function initSelectedMethods() {
    const saved = localStorage.getItem('selectedMethods');
    if (saved) {
        try {
            const methods = JSON.parse(saved);
            state.selectedMethods = new Set(methods);
        } catch (e) {
            // Keep default
        }
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
        console.log('[State] App config loaded:', state.appConfig);
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

        const list = document.getElementById('experiment-list');
        if (!list) return;

        // Filter experiments unless showAllExperiments is true
        const hiddenCount = state.experiments.filter(exp => HIDDEN_EXPERIMENTS.includes(exp)).length;
        const visibleExperiments = state.showAllExperiments
            ? state.experiments
            : state.experiments.filter(exp => !HIDDEN_EXPERIMENTS.includes(exp));

        list.innerHTML = visibleExperiments.map((exp, idx) => {
            const isActive = idx === 0 ? 'active' : '';
            return `<label class="experiment-option ${isActive}" data-experiment="${exp}">
                <input type="radio" name="experiment" ${idx === 0 ? 'checked' : ''}>
                <span>${exp}</span>
            </label>`;
        }).join('');

        // Add toggle link if there are hidden experiments
        if (hiddenCount > 0) {
            const toggleText = state.showAllExperiments ? 'Hide' : `Show ${hiddenCount} hidden`;
            list.innerHTML += `<div class="experiment-toggle" onclick="window.toggleHiddenExperiments()">${toggleText}</div>`;
        }

        list.querySelectorAll('.experiment-option').forEach(item => {
            item.addEventListener('click', async () => {
                list.querySelectorAll('.experiment-option').forEach(i => i.classList.remove('active'));
                item.classList.add('active');
                state.currentExperiment = item.dataset.experiment;
                setExperimentInURL(state.currentExperiment);
                await loadExperimentData(state.currentExperiment);
                window.renderPromptPicker();
                if (window.renderView) window.renderView();
            });
        });

        if (state.experiments.length > 0) {
            const urlExp = getExperimentFromURL();
            const viewNeedsExperiment = ANALYSIS_VIEWS.includes(state.currentView);

            if (urlExp && state.experiments.includes(urlExp)) {
                state.currentExperiment = urlExp;
                list.querySelectorAll('.experiment-option').forEach(item => {
                    const isActive = item.dataset.experiment === urlExp;
                    item.classList.toggle('active', isActive);
                    const radio = item.querySelector('input[type="radio"]');
                    if (radio) radio.checked = isActive;
                });
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
        window.showError('Failed to load experiments');
    }
}

async function ensureExperimentLoaded() {
    if (state.currentExperiment) return;
    if (!state.experiments || state.experiments.length === 0) return;
    if (!ANALYSIS_VIEWS.includes(state.currentView)) return;

    state.currentExperiment = state.experiments[0];
    setExperimentInURL(state.currentExperiment);
    await loadExperimentData(state.currentExperiment);

    const list = document.getElementById('experiment-list');
    if (list) {
        list.querySelectorAll('.experiment-option').forEach((item, idx) => {
            const isActive = idx === 0;
            item.classList.toggle('active', isActive);
            const radio = item.querySelector('input[type="radio"]');
            if (radio) radio.checked = isActive;
        });
    }
}

async function loadExperimentData(experimentName) {
    const contentArea = document.getElementById('content-area');
    if (contentArea) {
        contentArea.innerHTML = '<div class="loading">Loading experiment data...</div>';
    }

    // Reset view-specific state on experiment change
    state.selectedSteeringTrait = null;

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
            await window.modelConfig.loadForExperiment(experimentName);
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
        window.showError(`Failed to load experiment: ${experimentName}`);
    }
}

async function discoverAvailablePrompts() {
    state.availablePromptSets = {};
    state.promptsWithData = {};

    if (!state.currentExperiment) {
        populatePromptSelector();
        return;
    }

    try {
        const response = await fetch(`/api/experiments/${state.currentExperiment}/inference/prompt-sets`);
        if (!response.ok) {
            console.warn('Failed to fetch prompt sets');
            populatePromptSelector();
            return;
        }

        const data = await response.json();
        const promptSets = data.prompt_sets || [];

        for (const ps of promptSets) {
            state.availablePromptSets[ps.name] = ps.prompts || [];
            state.promptsWithData[ps.name] = ps.available_ids || [];
        }
    } catch (e) {
        console.error('Error fetching prompt sets:', e);
    }

    // Restore from localStorage or set default selection
    const savedPromptSet = localStorage.getItem('promptSet');
    const savedPromptId = localStorage.getItem('promptId');

    state.currentPromptSet = null;
    state.currentPromptId = null;

    const setsWithData = Object.entries(state.promptsWithData)
        .filter(([_, ids]) => ids.length > 0)
        .sort(([a], [b]) => {
            const aHasSingle = a.includes('single');
            const bHasSingle = b.includes('single');
            if (aHasSingle && !bHasSingle) return -1;
            if (!aHasSingle && bHasSingle) return 1;
            return a.localeCompare(b);
        });

    if (savedPromptSet && state.promptsWithData[savedPromptSet]?.length > 0) {
        state.currentPromptSet = savedPromptSet;
        const savedId = parseInt(savedPromptId);
        if (state.promptsWithData[savedPromptSet].includes(savedId)) {
            state.currentPromptId = savedId;
        } else {
            state.currentPromptId = state.promptsWithData[savedPromptSet][0];
        }
    } else if (setsWithData.length > 0) {
        const [setName, promptIds] = setsWithData[0];
        state.currentPromptSet = setName;
        state.currentPromptId = promptIds[0];
    }

    window.renderPromptPicker();
}

// Placeholder for prompt selector (implemented in prompt-picker.js)
function populatePromptSelector() {
    // This is a no-op here; renderPromptPicker handles everything
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
// GPU Status
// =============================================================================

let gpuPollInterval = null;

async function fetchGpuStatus() {
    try {
        const response = await fetch('/api/gpu-status');
        if (!response.ok) throw new Error('Failed to fetch GPU status');
        state.gpuStatus = await response.json();
        updateGpuStatusUI();
    } catch (e) {
        console.warn('GPU status fetch failed:', e);
        state.gpuStatus = { available: false, device: 'Unknown', error: e.message };
        updateGpuStatusUI();
    }
}

function updateGpuStatusUI() {
    const container = document.getElementById('gpu-status');
    if (!container) return;

    const status = state.gpuStatus;
    if (!status) {
        container.classList.add('loading');
        return;
    }

    container.classList.remove('loading');
    container.classList.toggle('available', status.available);
    container.classList.toggle('error', !!status.error);

    // Update device name
    const nameEl = container.querySelector('.gpu-name');
    if (nameEl) {
        // Shorten long device names
        let name = status.device || 'Unknown';
        name = name.replace('NVIDIA ', '').replace('Apple ', '');
        nameEl.textContent = name;
        nameEl.title = status.device;
    }

    // Update memory display
    const memoryEl = container.querySelector('.gpu-memory');
    if (memoryEl) {
        if (status.memory_used_gb != null && status.memory_total_gb != null) {
            const pct = (status.memory_used_gb / status.memory_total_gb) * 100;
            const fillClass = pct > 90 ? 'critical' : pct > 70 ? 'high' : '';
            memoryEl.innerHTML = `
                <div class="gpu-memory-bar">
                    <div class="gpu-memory-fill ${fillClass}" style="width: ${pct}%"></div>
                </div>
                <span class="gpu-memory-text">${status.memory_used_gb.toFixed(1)}/${status.memory_total_gb.toFixed(0)}GB</span>
            `;
        } else if (status.memory_total_gb != null) {
            // MPS - just show total
            memoryEl.innerHTML = `<span class="gpu-memory-text">${status.memory_total_gb.toFixed(0)}GB</span>`;
        } else {
            memoryEl.innerHTML = '';
        }
    }

    // Update tooltip
    let tooltip = status.device || 'GPU Status';
    if (status.note) tooltip += `\n${status.note}`;
    if (status.error) tooltip += `\nError: ${status.error}`;
    container.title = tooltip;
}

function startGpuPolling(intervalMs = 5000) {
    if (gpuPollInterval) clearInterval(gpuPollInterval);
    fetchGpuStatus();  // Initial fetch
    gpuPollInterval = setInterval(fetchGpuStatus, intervalMs);
}

function stopGpuPolling() {
    if (gpuPollInterval) {
        clearInterval(gpuPollInterval);
        gpuPollInterval = null;
    }
}

// =============================================================================
// Initialize Application
// =============================================================================

async function init() {
    await window.paths.load();
    await loadAppConfig();

    // Initialize preferences
    window.initTheme();
    initSmoothing();
    initProjectionCentered();
    initMassiveDimsCleaning();
    initCompareMode();
    initHideAttentionSink();
    initSelectedMethods();

    // Setup UI
    window.setupNavigation();
    await loadExperiments();
    window.setupSidebarEventListeners();

    // Start GPU status polling (only in dev mode where local inference is available)
    if (isFeatureEnabled('debug_info')) {
        startGpuPolling(5000);  // Poll every 5 seconds
    } else {
        // Just fetch once in production to show device info
        fetchGpuStatus();
    }

    // Read tab from URL and render
    initFromURL();
    window.updateExperimentVisibility();
    await ensureExperimentLoaded();
    window.renderPromptPicker();
    if (window.renderView) window.renderView();
}

// =============================================================================
// Experiment Visibility Toggle
// =============================================================================

function toggleHiddenExperiments() {
    state.showAllExperiments = !state.showAllExperiments;
    loadExperiments();  // Re-render list (won't reload data, just re-renders UI)
}

// =============================================================================
// Exports
// =============================================================================

window.state = state;
window.ANALYSIS_VIEWS = ANALYSIS_VIEWS;
window.GLOBAL_VIEWS = GLOBAL_VIEWS;
window.getFilteredTraits = getFilteredTraits;
window.isFeatureEnabled = isFeatureEnabled;
window.initApp = init;

// Preference setters
window.setSmoothing = setSmoothing;
window.setProjectionCentered = setProjectionCentered;
window.setMassiveDimsCleaning = setMassiveDimsCleaning;
window.setCompareMode = setCompareMode;
window.setHideAttentionSink = setHideAttentionSink;
window.toggleMethod = toggleMethod;

// GPU status
window.fetchGpuStatus = fetchGpuStatus;
window.startGpuPolling = startGpuPolling;
window.stopGpuPolling = stopGpuPolling;

// URL routing
window.setTabInURL = setTabInURL;
window.setExperimentInURL = setExperimentInURL;
window.getTabFromURL = getTabFromURL;
window.getExperimentFromURL = getExperimentFromURL;

// Experiment loading
window.loadExperimentData = loadExperimentData;
window.ensureExperimentLoaded = ensureExperimentLoaded;
window.toggleHiddenExperiments = toggleHiddenExperiments;
