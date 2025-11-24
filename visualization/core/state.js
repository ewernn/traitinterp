// Global State Management for Trait Interpretation Visualization

// State
const state = {
    experiments: [],
    currentExperiment: null,
    experimentData: null,
    currentView: 'data-explorer',
    selectedTraits: new Set(),
    // Prompt state structure
    currentPromptSet: null,      // e.g., 'single_trait'
    currentPromptId: null,       // e.g., 1
    availablePromptSets: {},     // { 'single_trait': [{id, text, note}, ...], ... }
    promptsWithData: {},         // { 'single_trait': [1, 2, 3], ... } - which prompts have inference data
    // Cached inference context (prompt/response text for current selection)
    inferenceContextCache: null  // { promptSet, promptId, promptText, responseText, promptTokens, responseTokens }
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
    let baseName = traitName;
    let method = '';
    let category = '';

    // Extract category if present
    if (traitName.includes('/')) {
        const parts = traitName.split('/');
        category = parts[0];
        baseName = parts[1];
    }

    // Extract method suffix
    if (baseName.endsWith('_natural')) {
        baseName = baseName.replace('_natural', '');
        method = ' (Natural)';
    }

    // Get display name (without category prefix in the lookup)
    let displayBase = DISPLAY_NAMES[baseName] ||
        baseName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());

    return displayBase + method;
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

// Transformer Sidebar Management
function toggleTransformerSidebar() {
    const sidebar = document.getElementById('transformer-sidebar');
    const toggle = document.getElementById('transformer-toggle');

    if (!sidebar) return;  // Sidebar is commented out in HTML

    if (sidebar.classList.contains('hidden')) {
        sidebar.classList.remove('hidden');
        toggle.textContent = '◀';
        toggle.title = 'Hide architecture';
        localStorage.setItem('transformerSidebarVisible', 'true');
    } else {
        sidebar.classList.add('hidden');
        toggle.textContent = '▶';
        toggle.title = 'Show architecture';
        localStorage.setItem('transformerSidebarVisible', 'false');
    }
}

function initTransformerSidebar() {
    const visible = localStorage.getItem('transformerSidebarVisible');
    if (visible === 'false') {
        toggleTransformerSidebar();
    }

    document.getElementById('transformer-toggle')?.addEventListener('click', toggleTransformerSidebar);
}

// Info Tooltip
function toggleInfo() {
    const tooltip = document.getElementById('info-tooltip');
    tooltip?.classList.toggle('show');
}

// Trait Selection
function populateTraitCheckboxes() {
    const container = document.getElementById('trait-checkboxes');
    if (!container) return;

    container.innerHTML = '';
    state.selectedTraits.clear();  // Clear before repopulating

    if (!state.experimentData || !state.experimentData.traits) return;

    // Deduplicate traits to prevent double-rendering from race conditions
    const uniqueTraits = state.experimentData.traits.filter((trait, index, self) =>
        index === self.findIndex((t) => t.name === trait.name)
    );

    uniqueTraits.forEach(trait => {
        const checkbox = document.createElement('div');
        checkbox.className = 'trait-checkbox';
        checkbox.innerHTML = `
            <input type="checkbox" id="trait-${trait.name}" value="${trait.name}" checked>
            <label for="trait-${trait.name}">${getDisplayName(trait.name)}</label>
        `;
        container.appendChild(checkbox);

        const input = checkbox.querySelector('input');
        input.addEventListener('change', (e) => {
            if (e.target.checked) {
                state.selectedTraits.add(trait.name);
            } else {
                state.selectedTraits.delete(trait.name);
            }
            updateSelectedCount();
            if (window.renderView) window.renderView();
        });

        state.selectedTraits.add(trait.name);
    });

    updateSelectedCount();
}

function updateSelectedCount() {
    const countElem = document.getElementById('selected-count');
    if (countElem) {
        countElem.textContent = state.selectedTraits.size;
    }
}

function toggleAllTraits() {
    const checkboxes = document.querySelectorAll('#trait-checkboxes input[type="checkbox"]');
    const allSelected = state.selectedTraits.size === checkboxes.length;

    checkboxes.forEach(cb => {
        cb.checked = !allSelected;
        if (!allSelected) {
            state.selectedTraits.add(cb.value);
        } else {
            state.selectedTraits.delete(cb.value);
        }
    });

    const btn = document.getElementById('select-all-btn');
    if (btn) {
        btn.textContent = allSelected ? 'Select All' : 'Deselect All';
    }
    updateSelectedCount();
    if (window.renderView) window.renderView();
}

// Views that use the inference context (Inference Analysis views)
const INFERENCE_VIEWS = ['all-layers', 'per-token-activation', 'layer-deep-dive'];

/**
 * Render the inference context panel (prompt picker + prompt/response display).
 * Shows only for inference views, hidden for trait development views.
 */
async function renderInferenceContext() {
    const container = document.getElementById('inference-context');
    if (!container) return;

    const isInferenceView = INFERENCE_VIEWS.includes(state.currentView);

    if (!isInferenceView) {
        container.style.display = 'none';
        return;
    }

    container.style.display = 'block';

    // Check if we have prompt data
    const hasAnyData = Object.values(state.promptsWithData).some(ids => ids.length > 0);
    if (!hasAnyData) {
        container.innerHTML = `
            <div class="inference-context-inner">
                <div class="no-prompts">No inference data available for this experiment.</div>
            </div>
        `;
        return;
    }

    // Build prompt picker HTML
    let promptSetOptions = '';
    for (const [setName, promptIds] of Object.entries(state.promptsWithData)) {
        if (promptIds.length === 0) continue;
        const selected = setName === state.currentPromptSet ? 'selected' : '';
        promptSetOptions += `<option value="${setName}" ${selected}>${setName.replace(/_/g, ' ')}</option>`;
    }

    const currentSetPromptIds = state.promptsWithData[state.currentPromptSet] || [];
    let promptBoxes = '';
    currentSetPromptIds.forEach(id => {
        const isActive = id === state.currentPromptId ? 'active' : '';
        const promptDef = (state.availablePromptSets[state.currentPromptSet] || []).find(p => p.id === id);
        const tooltip = promptDef ? promptDef.text.substring(0, 100) + (promptDef.text.length > 100 ? '...' : '') : '';
        promptBoxes += `<div class="prompt-box ${isActive}" data-prompt-set="${state.currentPromptSet}" data-prompt-id="${id}" title="${tooltip}">${id}</div>`;
    });

    // Get prompt text from definitions
    const promptDef = (state.availablePromptSets[state.currentPromptSet] || []).find(p => p.id === state.currentPromptId);
    const promptText = promptDef ? promptDef.text : 'Loading...';

    // Check cache for response data
    let responseText = '';
    let tokenInfo = '';

    if (state.inferenceContextCache &&
        state.inferenceContextCache.promptSet === state.currentPromptSet &&
        state.inferenceContextCache.promptId === state.currentPromptId) {
        // Use cached data
        responseText = state.inferenceContextCache.responseText;
        const nPrompt = state.inferenceContextCache.promptTokens || 0;
        const nResponse = state.inferenceContextCache.responseTokens || 0;
        tokenInfo = `${nPrompt} prompt + ${nResponse} response = ${nPrompt + nResponse} tokens`;
    } else {
        // Need to fetch - show loading state and fetch async
        responseText = 'Loading response...';
        tokenInfo = '';
        fetchInferenceContextData(); // Fire and forget, will re-render when done
    }

    container.innerHTML = `
        <div class="inference-context-inner">
            <div class="inference-context-picker">
                <select class="prompt-set-select" id="prompt-set-select">${promptSetOptions}</select>
                <div class="prompt-box-container">${promptBoxes}</div>
            </div>
            <div class="inference-context-content">
                <div class="inference-prompt">
                    <span class="inference-label">Prompt:</span>
                    <span class="inference-text">${escapeHtml(promptText)}</span>
                </div>
                <div class="inference-response">
                    <span class="inference-label">Response:</span>
                    <span class="inference-text">${escapeHtml(responseText.substring(0, 300))}${responseText.length > 300 ? '...' : ''}</span>
                </div>
                ${tokenInfo ? `<div class="inference-tokens">${tokenInfo}</div>` : ''}
            </div>
        </div>
    `;

    // Re-attach event listeners for the prompt picker
    setupInferenceContextListeners();
}

/**
 * Fetch prompt/response data from first available trait and cache it.
 */
async function fetchInferenceContextData() {
    if (!state.currentPromptSet || !state.currentPromptId) return;
    if (!state.experimentData || !state.experimentData.traits || state.experimentData.traits.length === 0) return;

    const firstTrait = state.experimentData.traits[0];

    try {
        const url = window.paths.residualStreamData(firstTrait, state.currentPromptSet, state.currentPromptId);
        const response = await fetch(url);
        if (!response.ok) return;

        const data = await response.json();

        state.inferenceContextCache = {
            promptSet: state.currentPromptSet,
            promptId: state.currentPromptId,
            promptText: data.prompt?.text || '',
            responseText: data.response?.text || '',
            promptTokens: data.prompt?.n_tokens || 0,
            responseTokens: data.response?.n_tokens || 0
        };

        // Re-render with the fetched data
        renderInferenceContext();
    } catch (e) {
        console.warn('Failed to fetch inference context data:', e);
    }
}

/**
 * Setup event listeners for the inference context prompt picker.
 */
function setupInferenceContextListeners() {
    const container = document.getElementById('inference-context');
    if (!container) return;

    // Prompt set dropdown
    const setSelect = container.querySelector('#prompt-set-select');
    if (setSelect) {
        setSelect.addEventListener('change', (e) => {
            const newSet = e.target.value;
            if (state.currentPromptSet !== newSet) {
                state.currentPromptSet = newSet;
                const availableIds = state.promptsWithData[newSet] || [];
                state.currentPromptId = availableIds[0] || null;
                state.inferenceContextCache = null; // Clear cache
                renderInferenceContext();
                if (window.renderView) window.renderView();
            }
        });
    }

    // Prompt ID boxes
    container.querySelectorAll('.prompt-box').forEach(box => {
        box.addEventListener('click', () => {
            const promptId = parseInt(box.dataset.promptId);
            if (state.currentPromptId !== promptId && !isNaN(promptId)) {
                state.currentPromptId = promptId;
                state.inferenceContextCache = null; // Clear cache
                renderInferenceContext();
                if (window.renderView) window.renderView();
            }
        });
    });
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

// Navigation
function setupNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            navItems.forEach(n => n.classList.remove('active'));
            item.classList.add('active');
            if (item.dataset.view) {
                state.currentView = item.dataset.view;
                updatePageTitle();
                renderInferenceContext();
                if (window.renderView) window.renderView();
            }
        });
    });
}

function updatePageTitle() {
    const titles = {
        'data-explorer': 'Data Explorer',
        'overview': 'Overview',
        'vectors': 'Vector Analysis',
        'trait-correlation': 'Trait Correlation',
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
                    <span class="icon">🔬</span>
                    <span>${displayName}</span>
                </div>
            `;
        }).join('');

        list.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', () => {
                list.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
                item.classList.add('active');
                state.currentExperiment = item.dataset.experiment;
                loadExperimentData(state.currentExperiment);
            });
        });

        if (state.experiments.length > 0) {
            state.currentExperiment = state.experiments[0];
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

    try {
        state.experimentData = {
            name: experimentName,
            traits: [],
            readme: null
        };
        
        // Correctly use the global path builder
        window.paths.setExperiment(experimentName);

        // Try to load README
        try {
            const readmePath = `experiments/${experimentName}/README.md`;
            const readmeResponse = await fetch(readmePath);
            if (readmeResponse.ok) {
                state.experimentData.readme = await readmeResponse.text();
            }
        } catch (e) {
            console.log('No README found for experiment.');
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
            const traitObj = { name: traitName };

            // Detect response format
            const responseFormat = await window.detectResponseFormat(window.paths, traitObj);

            if (!responseFormat) {
                console.warn(`No responses for ${traitName}`);
            }

            // Load metadata
            let metadata = null;
            try {
                const metadataRes = await fetch(window.paths.activationsMetadata(traitObj));
                if (metadataRes.ok) {
                    metadata = await metadataRes.json();
                }
            } catch (e) {
                console.warn(`No metadata for ${traitName}`);
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
                metadata: metadata
            });
        }

        // Update experiment badge
        const badge = document.getElementById('experiment-badge');
        if (badge) {
            badge.textContent = experimentName.replace(/_/g, ' ');
        }

        console.log(`Loaded ${state.experimentData.traits.length} traits for ${experimentName}:`);
        state.experimentData.traits.forEach((t, idx) => {
            console.log(`  [${idx}] ${t.name}`);
        });

        populateTraitCheckboxes();
        discoverAvailablePrompts();
        if (window.renderView) window.renderView();

    } catch (error) {
        console.error('Error loading experiment data:', error);
        showError(`Failed to load experiment: ${experimentName}`);
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
            // Store which IDs have data (discovered from raw/residual/)
            state.promptsWithData[ps.name] = ps.available_ids || [];
        }
    } catch (e) {
        console.error('Error fetching prompt sets:', e);
    }

    // Set default selection to first prompt set with data
    state.currentPromptSet = null;
    state.currentPromptId = null;

    for (const [setName, promptIds] of Object.entries(state.promptsWithData)) {
        if (promptIds.length > 0) {
            state.currentPromptSet = setName;
            state.currentPromptId = promptIds[0];
            break;
        }
    }

    console.log('Discovered prompt sets:', Object.keys(state.availablePromptSets));
    console.log('Prompts with data:', state.promptsWithData);
    console.log('Current selection:', state.currentPromptSet, state.currentPromptId);

    renderInferenceContext();
}

// Event Listeners Setup
function setupEventListeners() {
    // Theme toggle
    document.getElementById('theme-toggle')?.addEventListener('click', toggleTheme);

    // Info button
    document.getElementById('info-btn')?.addEventListener('click', toggleInfo);

    // Select all traits button
    document.getElementById('select-all-btn')?.addEventListener('click', toggleAllTraits);

    // Close info tooltip when clicking outside
    document.addEventListener('click', (e) => {
        const tooltip = document.getElementById('info-tooltip');
        const infoBtn = document.getElementById('info-btn');
        if (tooltip && !tooltip.contains(e.target) && e.target !== infoBtn) {
            tooltip.classList.remove('show');
        }
    });
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

// Standard colorscale for trait heatmaps (asymmetric: vibrant red + muted blue)
// Optimized for [0, +1] data where positive values matter more
const ASYMB_COLORSCALE = [
    [0, '#5a6a8a'],
    [0.25, '#98a4b0'],
    [0.5, '#c8c8c8'],
    [0.75, '#d47c67'],
    [1, '#b40426']
];

// Get CSS variable value helper
function getCssVar(name, fallback = '') {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
}

// Simple Markdown to HTML converter
function markdownToHtml(text) {
    if (!text) return '';
    return text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') // Bold
        .replace(/\*(.*?)\*/g, '<em>$1</em>');           // Italic
}

// Initialize Application
async function init() {
    await window.paths.load();
    initTheme();
    initTransformerSidebar();
    setupNavigation();
    await loadExperiments();
    setupEventListeners();
}

// Export state and functions
window.state = state;
window.getDisplayName = getDisplayName;
window.getFilteredTraits = getFilteredTraits;
window.getPlotlyLayout = getPlotlyLayout;
window.ASYMB_COLORSCALE = ASYMB_COLORSCALE;
window.getCssVar = getCssVar;
window.showError = showError;
window.initApp = init;
window.markdownToHtml = markdownToHtml;
