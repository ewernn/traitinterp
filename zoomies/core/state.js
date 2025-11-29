/**
 * Zoomies State Management
 * Global state object with setState() and render trigger.
 */

window.zoomies = window.zoomies || {};

// State object - single source of truth
window.zoomies.state = {
    // Current tab
    tab: 'main',  // 'main' | 'overview' | 'dev'

    // Granularity (for main tab)
    mode: 'inference',           // 'extraction' | 'inference'
    tokenScope: 'all',           // 'all' | number (token index)
    layerScope: 'all',           // 'all' | number (layer 0-25)
    componentScope: 'all',       // 'all' | 'attention' | 'mlp'

    // Context (inference only)
    promptSet: 'single_trait',
    promptId: 1,

    // Display
    selectedTraits: [],          // ['behavioral/refusal', ...]
    experiment: null,            // 'gemma_2b_cognitive_nov21'

    // Derived (set after data loads)
    totalTokens: null,
    promptTokenCount: null,

    // Data cache
    experiments: [],             // available experiments
    traits: [],                  // available traits for current experiment
    promptSets: {},              // { setName: { name, prompts: [{id, text}] } }
};

// Subscribers (views that re-render on state change)
const subscribers = new Set();

/**
 * Subscribe to state changes
 * @param {Function} callback - Called on state change with new state
 * @returns {Function} Unsubscribe function
 */
window.zoomies.subscribe = function(callback) {
    subscribers.add(callback);
    return () => subscribers.delete(callback);
};

/**
 * Update state and trigger re-render
 * @param {Object} updates - Partial state updates
 */
window.zoomies.setState = function(updates) {
    const oldState = { ...window.zoomies.state };
    Object.assign(window.zoomies.state, updates);

    // Handle mode switch: reset scopes
    if (updates.mode && updates.mode !== oldState.mode) {
        if (updates.tokenScope === undefined) {
            window.zoomies.state.tokenScope = 'all';
        }
        if (updates.layerScope === undefined) {
            window.zoomies.state.layerScope = 'all';
        }
        window.zoomies.state.componentScope = 'all';
    }

    // Reset componentScope when going back to all layers
    if (updates.layerScope === 'all') {
        window.zoomies.state.componentScope = 'all';
    }

    // Persist to localStorage
    persistState();

    // Notify subscribers
    subscribers.forEach(cb => cb(window.zoomies.state));

    // Trigger render
    window.zoomies.render();
};

/**
 * Get current position key for registry lookup
 * @returns {string} e.g., 'extraction:all', 'inference:all:layer', 'inference:token:layer'
 */
window.zoomies.getPositionKey = function() {
    const { mode, tokenScope, layerScope } = window.zoomies.state;

    if (mode === 'extraction') {
        return layerScope === 'all' ? 'extraction:all' : 'extraction:layer';
    }

    // Inference mode
    const tokenPart = tokenScope === 'all' ? 'all' : 'token';
    const layerPart = layerScope === 'all' ? 'all' : 'layer';
    return `inference:${tokenPart}:${layerPart}`;
};

/**
 * Render current view
 */
window.zoomies.render = function() {
    const { tab } = window.zoomies.state;

    // Render tabs
    if (window.zoomies.renderTabs) {
        window.zoomies.renderTabs();
    }

    // Render current view
    const contentEl = document.getElementById('content');
    if (!contentEl) return;

    switch (tab) {
        case 'main':
            if (window.zoomies.renderMainView) {
                window.zoomies.renderMainView(contentEl);
            } else {
                contentEl.innerHTML = '<div class="loading">Loading main view...</div>';
            }
            break;
        case 'overview':
            if (window.zoomies.renderOverviewView) {
                window.zoomies.renderOverviewView(contentEl);
            } else {
                contentEl.innerHTML = '<div class="loading">Loading overview...</div>';
            }
            break;
        case 'dev':
            if (window.zoomies.renderDevView) {
                window.zoomies.renderDevView(contentEl);
            } else {
                contentEl.innerHTML = '<div class="loading">Loading dev tools...</div>';
            }
            break;
        default:
            contentEl.innerHTML = `<div class="error">Unknown tab: ${tab}</div>`;
    }
};

/**
 * Persist selected state to localStorage
 */
function persistState() {
    const toSave = {
        experiment: window.zoomies.state.experiment,
        selectedTraits: window.zoomies.state.selectedTraits,
        mode: window.zoomies.state.mode,
        promptSet: window.zoomies.state.promptSet,
        promptId: window.zoomies.state.promptId,
    };
    localStorage.setItem('zoomies_state', JSON.stringify(toSave));
}

/**
 * Load persisted state from localStorage
 */
function loadPersistedState() {
    try {
        const saved = localStorage.getItem('zoomies_state');
        if (saved) {
            const parsed = JSON.parse(saved);
            Object.assign(window.zoomies.state, parsed);
        }
    } catch (e) {
        console.warn('Failed to load persisted state:', e);
    }

    // Load theme
    const theme = localStorage.getItem('zoomies_theme');
    if (theme) {
        document.documentElement.setAttribute('data-theme', theme);
    }
}

/**
 * Toggle dark/light theme
 */
window.zoomies.toggleTheme = function() {
    const current = document.documentElement.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('zoomies_theme', next);
};

/**
 * Initialize the app
 */
window.zoomies.init = async function() {
    console.log('Zoomies initializing...');

    // Load persisted state
    loadPersistedState();

    // Load from URL (overrides localStorage)
    if (window.zoomies.loadFromURL) {
        window.zoomies.loadFromURL();
    }

    // Load experiments list
    try {
        const resp = await fetch('/api/experiments');
        const data = await resp.json();
        const experiments = data.experiments || data;  // Handle both {experiments: []} and []
        window.zoomies.state.experiments = experiments;

        // Use first experiment if none selected
        if (!window.zoomies.state.experiment && experiments.length > 0) {
            window.zoomies.state.experiment = experiments[0];
        }

        // Load traits for current experiment
        if (window.zoomies.state.experiment) {
            await window.zoomies.loadExperimentData(window.zoomies.state.experiment);
        }
    } catch (e) {
        console.error('Failed to load experiments:', e);
    }

    // Initial render
    window.zoomies.render();

    console.log('Zoomies initialized', window.zoomies.state);
};

/**
 * Load data for an experiment
 */
window.zoomies.loadExperimentData = async function(experiment) {
    try {
        // Set experiment on paths for URL building
        if (window.zoomies.paths) {
            window.zoomies.paths.setExperiment(experiment);
        }

        // Load traits
        const traitsResp = await fetch(`/api/experiments/${experiment}/traits`);
        const traitsData = await traitsResp.json();
        const traits = traitsData.traits || traitsData;
        window.zoomies.state.traits = traits;

        // Select all traits by default if none selected
        if (window.zoomies.state.selectedTraits.length === 0 && traits.length > 0) {
            window.zoomies.state.selectedTraits = [...traits];
        }

        // Load prompt sets with available IDs
        const setsResp = await fetch(`/api/experiments/${experiment}/inference/prompt-sets`);
        const setsData = await setsResp.json();
        // Convert array to object keyed by name, extract available_ids as ids
        const promptSets = {};
        const setsArray = setsData.prompt_sets || setsData;
        if (Array.isArray(setsArray)) {
            setsArray.forEach(set => {
                promptSets[set.name] = {
                    name: set.name,
                    description: set.description,
                    ids: set.available_ids || [],
                };
            });
        }
        window.zoomies.state.promptSets = promptSets;

        // Auto-select first prompt set and ID if not set
        const promptSetNames = Object.keys(promptSets);
        if (promptSetNames.length > 0) {
            if (!window.zoomies.state.promptSet || !promptSets[window.zoomies.state.promptSet]) {
                window.zoomies.state.promptSet = promptSetNames[0];
            }
            const currentSet = promptSets[window.zoomies.state.promptSet];
            if (currentSet && currentSet.ids.length > 0) {
                if (!window.zoomies.state.promptId || !currentSet.ids.includes(window.zoomies.state.promptId)) {
                    window.zoomies.state.promptId = currentSet.ids[0];
                }
            }
        }

        console.log('Loaded experiment data:', {
            traits: traits.length,
            promptSets: Object.keys(promptSets),
            selectedTraits: window.zoomies.state.selectedTraits,
            promptSet: window.zoomies.state.promptSet,
            promptId: window.zoomies.state.promptId,
        });
    } catch (e) {
        console.error('Failed to load experiment data:', e);
    }
};

// Constants
window.zoomies.LAYERS = 26;  // Gemma 2B has 26 layers (0-25)
window.zoomies.METHODS = ['mean_diff', 'probe', 'ica', 'gradient', 'pca_diff', 'random_baseline'];
