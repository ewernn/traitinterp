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
    promptPickerCache: null  // { promptSet, promptId, promptText, responseText, promptTokens, responseTokens, allTokens, nPromptTokens }
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
                await loadExperimentData(state.currentExperiment);
                // Re-render current view with new experiment data
                renderPromptPicker();
                if (window.renderView) window.renderView();
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

            // Load activations metadata (n_layers, model info)
            let metadata = null;
            try {
                const metadataRes = await fetch(window.paths.activationsMetadata(traitObj));
                if (metadataRes.ok) {
                    metadata = await metadataRes.json();
                }
            } catch (e) {
                // Metadata is optional - will fall back to defaults (26 layers for Gemma 2B)
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

        console.log(`Loaded ${state.experimentData.traits.length} traits for ${experimentName}:`);
        state.experimentData.traits.forEach((t, idx) => {
            console.log(`  [${idx}] ${t.name}`);
        });

        populateTraitCheckboxes();
        await discoverAvailablePrompts();  // Await to ensure prompts are ready
        // Don't render here - let init() handle rendering after URL is read

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

    console.log('Discovered prompt sets:', Object.keys(state.availablePromptSets));
    console.log('Prompts with data:', state.promptsWithData);
    console.log('Current selection:', state.currentPromptSet, state.currentPromptId);

    renderPromptPicker();
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

// Get CSS variable value helper
function getCssVar(name, fallback = '') {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
}

// URL-based tab routing functions
function getTabFromURL() {
    const params = new URLSearchParams(window.location.search);
    return params.get('tab') || 'overview';
}

function setTabInURL(tabName) {
    const url = new URL(window.location);
    url.searchParams.set('tab', tabName);
    window.history.pushState({ tab: tabName }, '', url);
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
    initTransformerSidebar();
    setupNavigation();
    await loadExperiments();
    setupEventListeners();

    // NEW: Read tab from URL and render
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
window.getCssVar = getCssVar;
window.showError = showError;
window.initApp = init;
window.escapeHtml = escapeHtml;
window.markdownToHtml = markdownToHtml;
window.renderMath = renderMath;
