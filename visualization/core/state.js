// Global State Management for Trait Interpretation Visualization

// State
const state = {
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
    // Jailbreak success tracking
    jailbreakSuccessIds: null,  // Set of prompt IDs that successfully jailbroke the model
    // Steering Sweep settings
    selectedSteeringTrait: null,  // Selected trait for single-trait sections (reset on experiment change)
    // Projection normalization mode
    smoothingEnabled: true,  // Apply 3-token moving average
    projectionCentered: true,  // Subtract BOS token value (centers around 0)
    // Method filter for trait dynamics (which extraction methods to show)
    selectedMethods: new Set(['probe', 'mean_diff', 'gradient', 'random'])
};

// Display names for better interpretability
const DISPLAY_NAMES = {
    // Legacy names (from old structure)
    'uncertainty_calibration': 'Confidence',
    'instruction_boundary': 'Literalness',
    'commitment_strength': 'Assertiveness',
    'retrieval_construction': 'Retrieval',
    'convergent_divergent': 'Thinking Style',
    'abstract_concrete': 'Abstraction Level',
    'temporal_focus': 'Temporal Orientation',
    'cognitive_load': 'Complexity',
    'context_adherence': 'Context Following',
    'emotional_valence': 'Emotional Tone',
    'paranoia_trust': 'Trust Level',
    'power_dynamics': 'Authority Tone',
    'serial_parallel': 'Processing Style',
    'local_global': 'Focus Scope',

    // New categorized trait names
    'abstractness': 'Abstractness',
    'authority': 'Authority',
    'compliance': 'Compliance',
    'confidence': 'Confidence',
    'context': 'Context Adherence',
    'curiosity': 'Curiosity',
    'defensiveness': 'Defensiveness',
    'divergence': 'Divergent Thinking',
    'enthusiasm': 'Enthusiasm',
    'evaluation_awareness': 'Evaluation Awareness',
    'formality': 'Formality',
    'futurism': 'Future Focus',
    'literalness': 'Literalness',
    'positivity': 'Positivity',
    'refusal': 'Refusal',
    'retrieval': 'Retrieval',
    'scope': 'Scope',
    'sequentiality': 'Sequential Processing',
    'sycophancy': 'Sycophancy',
    'trust': 'Trust'
};

// Helper Functions
function getDisplayName(traitName) {
    // Handle category/trait format (e.g., "cognitive/context")
    // Also handles category/trait/position format (e.g., "chirp/refusal_v2/response_all")
    let baseName = traitName;
    let method = '';
    let category = '';
    let position = '';

    // Extract parts
    if (traitName.includes('/')) {
        const parts = traitName.split('/');
        category = parts[0];
        baseName = parts[1];
        // Check for position suffix (3+ parts)
        if (parts.length >= 3) {
            position = parts.slice(2).join('/');
        }
    }

    // Extract method suffix
    if (baseName.endsWith('_natural')) {
        baseName = baseName.replace('_natural', '');
        method = ' (Natural)';
    }

    // Get display name (without category prefix in the lookup)
    let displayBase = DISPLAY_NAMES[baseName] ||
        baseName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());

    // Format position suffix for display
    let positionSuffix = '';
    if (position) {
        positionSuffix = ' ' + window.paths.formatPositionDisplay(position);
    }

    return displayBase + method + positionSuffix;
}

function getFilteredTraits() {
    if (!state.experimentData || !state.experimentData.traits) return [];
    return state.experimentData.traits.filter(trait => state.selectedTraits.has(trait.name));
}

// Theme Management
function initTheme() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);
    updateThemeIcon(savedTheme);
}

// Smoothing Management
function initSmoothing() {
    const saved = localStorage.getItem('smoothingEnabled');
    // Default to true if not set
    state.smoothingEnabled = saved === null ? true : saved === 'true';
}

function setSmoothing(enabled) {
    state.smoothingEnabled = !!enabled;
    localStorage.setItem('smoothingEnabled', state.smoothingEnabled);
    if (window.renderView) window.renderView();
}

function initProjectionCentered() {
    const saved = localStorage.getItem('projectionCentered');
    // Default to true if not set
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
    state.massiveDimsCleaning = saved || 'top5-3layers';  // Default: 8 dims appearing in top-5 at 3+ layers
}

function setMassiveDimsCleaning(mode) {
    state.massiveDimsCleaning = mode || 'none';
    localStorage.setItem('massiveDimsCleaning', state.massiveDimsCleaning);
    if (window.renderView) window.renderView();
}

// Compare Mode (Model Comparison)
// compareMode values: "main" (default), "diff:{model}", or "show:{model}"
function initCompareMode() {
    const savedMode = localStorage.getItem('compareMode');
    state.compareMode = savedMode || 'main';
    // Available models discovered from inference/models/ directory
    state.availableComparisonModels = [];

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

function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    updateThemeIcon(newTheme);
}

function updateThemeIcon(theme) {
    const themeToggle = document.getElementById('theme-toggle');
    themeToggle.textContent = theme === 'dark' ? '☀️' : '🌙';
    themeToggle.title = theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode';
}

// Trait Selection
function populateTraitCheckboxes() {
    const container = document.getElementById('trait-checkboxes');
    if (!container) return;

    container.innerHTML = '';
    state.selectedTraits.clear();

    if (!state.experimentData || !state.experimentData.traits) return;

    // Deduplicate traits
    const uniqueTraits = state.experimentData.traits.filter((trait, index, self) =>
        index === self.findIndex((t) => t.name === trait.name)
    );

    // Group traits by category
    const categories = {};
    uniqueTraits.forEach(trait => {
        const parts = trait.name.split('/');
        const category = parts.length > 1 ? parts[0] : 'uncategorized';
        if (!categories[category]) categories[category] = [];
        categories[category].push(trait);
    });

    // Check if og_10 exists for default selection
    const hasOg10 = 'og_10' in categories;

    // Sort categories: og_10 first, then alphabetical
    const sortedCategories = Object.keys(categories).sort((a, b) => {
        if (a === 'og_10') return -1;
        if (b === 'og_10') return 1;
        return a.localeCompare(b);
    });

    sortedCategories.forEach(category => {
        const traits = categories[category];
        const isDefaultSelected = hasOg10 ? category === 'og_10' : true;

        // Create category container
        const categoryDiv = document.createElement('div');
        categoryDiv.className = 'trait-category';
        categoryDiv.dataset.category = category;

        // Category header
        const header = document.createElement('div');
        header.className = 'trait-category-header';
        header.innerHTML = `
            <span class="category-arrow">▼</span>
            <input type="checkbox" class="category-checkbox" ${isDefaultSelected ? 'checked' : ''}>
            <span class="category-name">${category.replace(/_/g, ' ')}</span>
            <span class="category-count">(${isDefaultSelected ? traits.length : 0}/${traits.length})</span>
        `;
        categoryDiv.appendChild(header);

        // Traits container
        const traitsDiv = document.createElement('div');
        traitsDiv.className = 'trait-category-items';

        traits.forEach(trait => {
            const checkbox = document.createElement('div');
            checkbox.className = 'trait-checkbox';
            checkbox.innerHTML = `
                <input type="checkbox" id="trait-${trait.name}" value="${trait.name}" ${isDefaultSelected ? 'checked' : ''}>
                <label for="trait-${trait.name}">${getDisplayName(trait.name)}</label>
            `;
            traitsDiv.appendChild(checkbox);

            const input = checkbox.querySelector('input');
            input.addEventListener('change', (e) => {
                if (e.target.checked) {
                    state.selectedTraits.add(trait.name);
                } else {
                    state.selectedTraits.delete(trait.name);
                }
                updateCategoryCheckbox(categoryDiv);
                updateSelectedCount();
                if (window.renderView) window.renderView();
            });

            if (isDefaultSelected) {
                state.selectedTraits.add(trait.name);
            }
        });

        categoryDiv.appendChild(traitsDiv);
        container.appendChild(categoryDiv);

        // Category header click handlers
        const arrow = header.querySelector('.category-arrow');
        const categoryCheckbox = header.querySelector('.category-checkbox');

        // Arrow toggles collapse
        arrow.addEventListener('click', (e) => {
            e.stopPropagation();
            const isCollapsed = categoryDiv.classList.toggle('collapsed');
            arrow.textContent = isCollapsed ? '▶' : '▼';
        });

        // Category checkbox toggles all traits in category
        categoryCheckbox.addEventListener('change', (e) => {
            const checked = e.target.checked;
            traitsDiv.querySelectorAll('input[type="checkbox"]').forEach(cb => {
                cb.checked = checked;
                if (checked) {
                    state.selectedTraits.add(cb.value);
                } else {
                    state.selectedTraits.delete(cb.value);
                }
            });
            updateCategoryCount(categoryDiv);
            updateSelectedCount();
            if (window.renderView) window.renderView();
        });

        updateCategoryCount(categoryDiv);
    });

    updateSelectedCount();
}

function updateCategoryCheckbox(categoryDiv) {
    const checkboxes = categoryDiv.querySelectorAll('.trait-category-items input[type="checkbox"]');
    const checkedCount = Array.from(checkboxes).filter(cb => cb.checked).length;
    const categoryCheckbox = categoryDiv.querySelector('.category-checkbox');
    categoryCheckbox.checked = checkedCount === checkboxes.length;
    categoryCheckbox.indeterminate = checkedCount > 0 && checkedCount < checkboxes.length;
    updateCategoryCount(categoryDiv);
}

function updateCategoryCount(categoryDiv) {
    const checkboxes = categoryDiv.querySelectorAll('.trait-category-items input[type="checkbox"]');
    const checkedCount = Array.from(checkboxes).filter(cb => cb.checked).length;
    const countSpan = categoryDiv.querySelector('.category-count');
    countSpan.textContent = `(${checkedCount}/${checkboxes.length})`;
}

function updateSelectedCount() {
    const countElem = document.getElementById('selected-count');
    if (countElem) {
        countElem.textContent = state.selectedTraits.size;
    }
}

function toggleAllTraits() {
    const traitCheckboxes = document.querySelectorAll('.trait-category-items input[type="checkbox"]');
    const allSelected = state.selectedTraits.size === traitCheckboxes.length;

    traitCheckboxes.forEach(cb => {
        cb.checked = !allSelected;
        if (!allSelected) {
            state.selectedTraits.add(cb.value);
        } else {
            state.selectedTraits.delete(cb.value);
        }
    });

    // Update all category checkboxes
    document.querySelectorAll('.trait-category').forEach(categoryDiv => {
        updateCategoryCheckbox(categoryDiv);
    });

    const btn = document.getElementById('select-all-btn');
    if (btn) {
        btn.textContent = allSelected ? 'Select All' : 'Deselect All';
    }
    updateSelectedCount();
    if (window.renderView) window.renderView();
}

/**
 * Simple HTML escaping for user content.
 */
function escapeHtml(text) {
    if (!text) return '';
    return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

/**
 * Format token for display (newlines→↵, tabs→→, spaces→·)
 */
function formatTokenDisplay(token) {
    if (!token) return '';
    return token.replace(/\n/g, '↵').replace(/\t/g, '→').replace(/ /g, '·');
}

/**
 * Setup click handlers for subsection info toggles (► triangles)
 * Uses event delegation to handle dynamically added content
 */
function setupSubsectionInfoToggles() {
    const container = document.querySelector('.tool-view');
    if (!container || container.dataset.togglesSetup) return;
    container.dataset.togglesSetup = 'true';

    container.addEventListener('click', (e) => {
        const toggle = e.target.closest('.subsection-info-toggle');
        if (!toggle) return;

        const targetId = toggle.dataset.target;
        const infoDiv = document.getElementById(targetId);
        if (infoDiv) {
            const isShown = infoDiv.classList.toggle('show');
            toggle.textContent = isShown ? '▼' : '►';

            // Typeset MathJax when info is shown (content was hidden during initial typeset)
            if (isShown && window.MathJax && !infoDiv.dataset.mathTypeset) {
                infoDiv.dataset.mathTypeset = 'true';
                MathJax.typesetPromise([infoDiv]);
            }
        }
    });
}

// Navigation
function setupNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            navItems.forEach(n => n.classList.remove('active'));
            item.classList.add('active');
            if (item.dataset.view) {
                state.currentView = item.dataset.view;
                setTabInURL(item.dataset.view);
                updatePageTitle();
                renderPromptPicker();
                if (window.renderView) window.renderView();
            }
        });
    });
}

function updatePageTitle() {
    const titles = {
        'data-explorer': 'Data Explorer',
        'overview': 'Overview',
        'trait-extraction': 'Trait Extraction',
        'vectors': 'Vector Analysis',
        'validation': 'Validation Results',
        'monitoring': 'All Layers',
        'prompt-activation': 'Per-Token Activation',
        'layer-dive': 'Layer Deep Dive'
    };
    const titleElem = document.getElementById('page-title');
    if (titleElem) {
        titleElem.textContent = titles[state.currentView] || 'Data Explorer';
    }
}

// Experiment Loading
async function loadExperiments() {
    try {
        const response = await fetch('/api/experiments');
        const data = await response.json();
        state.experiments = data.experiments || [];

        const list = document.getElementById('experiment-list');
        if (!list) return;

        list.innerHTML = state.experiments.map((exp, idx) => {
            const displayName = exp.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            const isActive = idx === 0 ? 'active' : '';
            return `
                <div class="nav-item ${isActive}" data-experiment="${exp}">
                    <span class="icon">${idx === 0 ? '✓' : ''}</span>
                    <span>${displayName}</span>
                </div>
            `;
        }).join('');

        list.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', async () => {
                list.querySelectorAll('.nav-item').forEach(i => {
                    i.classList.remove('active');
                    i.querySelector('.icon').textContent = '';
                });
                item.classList.add('active');
                item.querySelector('.icon').textContent = '✓';
                state.currentExperiment = item.dataset.experiment;
                setExperimentInURL(state.currentExperiment);
                await loadExperimentData(state.currentExperiment);
                // Re-render current view with new experiment data
                renderPromptPicker();
                if (window.renderView) window.renderView();
            });
        });

        if (state.experiments.length > 0) {
            // Check URL for experiment, otherwise use first
            const urlExp = getExperimentFromURL();
            if (urlExp && state.experiments.includes(urlExp)) {
                state.currentExperiment = urlExp;
                // Update active state in sidebar
                list.querySelectorAll('.nav-item').forEach(item => {
                    const isActive = item.dataset.experiment === urlExp;
                    item.classList.toggle('active', isActive);
                    item.querySelector('.icon').textContent = isActive ? '✓' : '';
                });
            } else {
                state.currentExperiment = state.experiments[0];
                setExperimentInURL(state.currentExperiment);
            }
            await loadExperimentData(state.currentExperiment);
        }
    } catch (error) {
        console.error('Error loading experiments:', error);
        showError('Failed to load experiments');
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
            readme: null,
            experimentConfig: null  // Will hold extraction_model, application_model
        };

        // Correctly use the global path builder
        window.paths.setExperiment(experimentName);

        // Load experiment config.json (extraction_model, application_model)
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

        // Try to load README
        try {
            const readmePath = `experiments/${experimentName}/README.md`;
            const readmeResponse = await fetch(readmePath);
            if (readmeResponse.ok) {
                state.experimentData.readme = await readmeResponse.text();
            }
        } catch (e) {
            // README not found, optional
        }

        // Fetch traits from API
        const traitsResponse = await fetch(`/api/experiments/${experimentName}/traits`);
        if (!traitsResponse.ok) {
            throw new Error(`Failed to fetch traits: ${traitsResponse.statusText}`);
        }
        const traitsData = await traitsResponse.json();
        const traitNames = traitsData.traits || [];

        // Load each trait with metadata
        for (const traitName of traitNames) {
            try {
                const traitObj = { name: traitName };
                const isBaseline = traitName.startsWith('baseline/');

                // Skip response checks for baseline traits (they only have vectors)
                let responseFormat = null;
                if (!isBaseline) {
                    responseFormat = await window.detectResponseFormat(window.paths, traitObj);
                }

                // Load activations metadata - skip for baselines
                let metadata = null;
                if (!isBaseline) {
                    try {
                        const metadataRes = await fetch(window.paths.activationsMetadata(traitObj));
                        if (metadataRes.ok) {
                            metadata = await metadataRes.json();
                        }
                    } catch (e) {
                        // Metadata is optional
                    }
                }

                const method = traitName.endsWith('_natural') ? 'natural' : 'instruction';
                let baseName = traitName.replace('_natural', '');
                if (baseName.includes('/')) {
                    baseName = baseName.split('/')[1];
                }

                const hasVectors = await window.hasVectors(window.paths, traitObj);

                state.experimentData.traits.push({
                    name: traitName,
                    baseName: baseName,
                    method: method,
                    responseFormat: responseFormat,
                    hasResponses: !!responseFormat,
                    hasVectors: hasVectors,
                    metadata: metadata,
                    isBaseline: isBaseline
                });
            } catch (e) {
                console.error(`Error loading trait ${traitName}:`, e);
            }
        }

        populateTraitCheckboxes();
        await discoverAvailablePrompts();
        await discoverComparisonModels();

    } catch (error) {
        console.error('Error loading experiment data:', error);
        showError(`Failed to load experiment: ${experimentName}`);
    }
}

/**
 * Load jailbreak success IDs from the curated dataset.
 * These are prompts that successfully bypassed the model's safety guardrails.
 */
async function loadJailbreakSuccesses() {
    try {
        const res = await fetch('/datasets/inference/jailbreak_successes.json');
        if (!res.ok) {
            state.jailbreakSuccessIds = new Set();
            return;
        }
        const data = await res.json();
        state.jailbreakSuccessIds = new Set(data.prompts.map(p => p.id));
    } catch (e) {
        console.warn('Could not load jailbreak successes:', e);
        state.jailbreakSuccessIds = new Set();
    }
}

async function discoverAvailablePrompts() {
    state.availablePromptSets = {};
    state.promptsWithData = {};

    if (!state.currentExperiment) {
        populatePromptSelector();
        return;
    }

    // Single API call to get all prompt sets with their available IDs
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
            // Store prompt definitions
            state.availablePromptSets[ps.name] = ps.prompts || [];
            // Store which IDs have data (discovered from projection JSONs or raw .pt files)
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
            // Prioritize sets with "single" in the name
            const aHasSingle = a.includes('single');
            const bHasSingle = b.includes('single');
            if (aHasSingle && !bHasSingle) return -1;
            if (!aHasSingle && bHasSingle) return 1;
            // Otherwise alphabetical
            return a.localeCompare(b);
        });

    // Try to restore saved selection if valid
    if (savedPromptSet && state.promptsWithData[savedPromptSet]?.length > 0) {
        state.currentPromptSet = savedPromptSet;
        const savedId = parseInt(savedPromptId);
        if (state.promptsWithData[savedPromptSet].includes(savedId)) {
            state.currentPromptId = savedId;
        } else {
            state.currentPromptId = state.promptsWithData[savedPromptSet][0];
        }
    } else if (setsWithData.length > 0) {
        // Fall back to default
        const [setName, promptIds] = setsWithData[0];
        state.currentPromptSet = setName;
        state.currentPromptId = promptIds[0];
    }

    renderPromptPicker();
}

async function discoverComparisonModels() {
    state.availableComparisonModels = [];

    if (!state.currentExperiment) return;

    try {
        const response = await fetch(`/api/experiments/${state.currentExperiment}/inference/models`);
        if (response.ok) {
            const data = await response.json();
            state.availableComparisonModels = data.models || [];
        }
    } catch (e) {
        console.warn('Could not discover comparison models:', e);
    }
}

// Event Listeners Setup
function setupEventListeners() {
    // Theme toggle
    document.getElementById('theme-toggle')?.addEventListener('click', toggleTheme);

    // Select all traits button
    document.getElementById('select-all-btn')?.addEventListener('click', toggleAllTraits);

}

// Utility Functions
function showError(message) {
    const contentArea = document.getElementById('content-area');
    if (contentArea) {
        contentArea.innerHTML = `<div class="error">${message}</div>`;
    }
}

// Plotly Layout Helper - reads colors from CSS variables
function getPlotlyLayout(baseLayout = {}) {
    const styles = getComputedStyle(document.documentElement);
    const textPrimary = styles.getPropertyValue('--text-primary').trim() || '#e0e0e0';
    const bgTertiary = styles.getPropertyValue('--bg-tertiary').trim() || '#3a3a3a';

    return {
        ...baseLayout,
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: {
            ...baseLayout.font,
            color: textPrimary
        },
        xaxis: {
            ...baseLayout.xaxis,
            color: textPrimary,
            gridcolor: bgTertiary,
            zerolinecolor: bgTertiary
        },
        yaxis: {
            ...baseLayout.yaxis,
            color: textPrimary,
            gridcolor: bgTertiary,
            zerolinecolor: bgTertiary
        }
    };
}

// Standard colorscale for trait heatmaps (emerald to rose: green=positive, red=negative)
// Forest→Coral: low values (red/coral), high values (green/forest)
const ASYMB_COLORSCALE = [
    [0, '#d47c67'],
    [0.25, '#e8b0a0'],
    [0.5, '#e8e8c8'],
    [0.75, '#91cf60'],
    [1, '#1a9850']
];

// Delta colorscale for steering heatmaps (diverging: red=negative, green=positive)
const DELTA_COLORSCALE = [
    [0, '#aa5656'],
    [0.5, '#e0e0de'],
    [1, '#3d7435']
];

// Get CSS variable value helper
function getCssVar(name, fallback = '') {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
}

// Convert hex color to rgba with opacity
function hexToRgba(hex, opacity) {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    if (!result) return `rgba(0, 0, 0, ${opacity})`;
    const r = parseInt(result[1], 16);
    const g = parseInt(result[2], 16);
    const b = parseInt(result[3], 16);
    return `rgba(${r}, ${g}, ${b}, ${opacity})`;
}

// Get token highlight colors for Plotly shapes (single source of truth)
function getTokenHighlightColors() {
    const primaryColor = getCssVar('--primary-color', '#a09f6c');
    return {
        separator: `${primaryColor}80`,  // 50% opacity - prompt/response divider
        highlight: `${primaryColor}80`   // 50% opacity - current token highlight
    };
}

// Chart color palette (reads from CSS vars, with fallbacks)
function getChartColors() {
    return [
        getCssVar('--chart-1', '#4a9eff'),
        getCssVar('--chart-2', '#ff6b6b'),
        getCssVar('--chart-3', '#51cf66'),
        getCssVar('--chart-4', '#ffd43b'),
        getCssVar('--chart-5', '#cc5de8'),
        getCssVar('--chart-6', '#ff922b'),
        getCssVar('--chart-7', '#20c997'),
        getCssVar('--chart-8', '#f06595'),
        getCssVar('--chart-9', '#748ffc'),
        getCssVar('--chart-10', '#a9e34b'),
    ];
}

// Method colors for extraction methods
function getMethodColors() {
    return {
        probe: getCssVar('--method-probe', '#4a9eff'),
        gradient: getCssVar('--method-gradient', '#51cf66'),
        mean_diff: getCssVar('--method-mean-diff', '#cc5de8'),
    };
}

// URL-based routing functions
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

    // Update active nav item
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    const activeNav = document.querySelector(`.nav-item[data-view="${tab}"]`);
    if (activeNav) activeNav.classList.add('active');
}

// Handle browser back/forward buttons
window.addEventListener('popstate', () => {
    initFromURL();
    renderPromptPicker();
    if (window.renderView) window.renderView();
});

// Simple Markdown to HTML converter
function markdownToHtml(text) {
    if (!text) return '';
    return text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') // Bold
        .replace(/\*(.*?)\*/g, '<em>$1</em>');           // Italic
}

// Global math rendering utility (KaTeX)
function renderMath(element) {
    if (typeof renderMathInElement === 'undefined') {
        console.warn('KaTeX not loaded - math rendering skipped');
        return;
    }

    try {
        renderMathInElement(element, {
            delimiters: [
                {left: '$$', right: '$$', display: true},
                {left: '$', right: '$', display: false}
            ],
            throwOnError: false
        });
    } catch (error) {
        console.error('Math rendering error:', error);
    }
}

// Initialize Application
async function init() {
    await window.paths.load();
    initTheme();
    initSmoothing();
    initProjectionCentered();
    initMassiveDimsCleaning();
    initCompareMode();
    initHideAttentionSink();
    initSelectedMethods();
    setupNavigation();
    await loadExperiments();
    await loadJailbreakSuccesses();
    setupEventListeners();

    // Read tab from URL and render
    initFromURL();
    renderPromptPicker();
    if (window.renderView) window.renderView();
}

// Export state and functions
window.state = state;
window.getDisplayName = getDisplayName;
window.getFilteredTraits = getFilteredTraits;
window.getPlotlyLayout = getPlotlyLayout;
window.ASYMB_COLORSCALE = ASYMB_COLORSCALE;
window.DELTA_COLORSCALE = DELTA_COLORSCALE;
window.getCssVar = getCssVar;
window.hexToRgba = hexToRgba;
window.getTokenHighlightColors = getTokenHighlightColors;
window.getChartColors = getChartColors;
window.getMethodColors = getMethodColors;
window.showError = showError;
window.initApp = init;
window.escapeHtml = escapeHtml;
window.formatTokenDisplay = formatTokenDisplay;
window.setupSubsectionInfoToggles = setupSubsectionInfoToggles;
window.markdownToHtml = markdownToHtml;
window.renderMath = renderMath;
window.setSmoothing = setSmoothing;
window.setProjectionCentered = setProjectionCentered;
window.setMassiveDimsCleaning = setMassiveDimsCleaning;
window.setCompareMode = setCompareMode;
window.setHideAttentionSink = setHideAttentionSink;
window.toggleMethod = toggleMethod;
