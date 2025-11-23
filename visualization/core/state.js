// Global State Management for Trait Interpretation Visualization

// State
const state = {
    experiments: [],
    currentExperiment: null,
    experimentData: null,
    currentView: 'data-explorer',
    selectedTraits: new Set(),
    currentPrompt: 0,
    availablePrompts: []
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
    if (!state.experimentData || !state.experimentData.traits || state.experimentData.traits.length === 0) {
        state.availablePrompts = [0];
        populatePromptSelector();
        return;
    }

    const promptSet = new Set();
    const firstTrait = state.experimentData.traits[0]; // Use only one trait for checking

    // Check up to 101 prompts, but stop when one is missing
    for (let i = 0; i <= 100; i++) {
        try {
            const url = window.paths.allLayersData(firstTrait, i);
            const response = await fetch(url, { method: 'HEAD' });
            if (response.ok) {
                promptSet.add(i);
            } else {
                // Stop assuming contiguous prompts
                break;
            }
        } catch (e) {
            // Network error or other issue, stop checking
            break;
        }
    }

    state.availablePrompts = promptSet.size > 0 ?
        Array.from(promptSet).sort((a, b) => a - b) : [0];
    console.log('Discovered available prompts:', state.availablePrompts);

    state.currentPrompt = state.availablePrompts[0] || 0;
    populatePromptSelector();
}

function populatePromptSelector() {
    const container = document.getElementById('prompt-selector');
    if (!container) return;

    container.innerHTML = ''; // Clear previous content

    state.availablePrompts.forEach(n => {
        const box = document.createElement('div');
        box.className = 'prompt-box';
        box.textContent = n;
        box.dataset.promptId = n;
        container.appendChild(box);
    });

    updatePromptSelectionUI();
}

function updatePromptSelectionUI() {
    const boxes = document.querySelectorAll('.prompt-box');
    boxes.forEach(box => {
        if (parseInt(box.dataset.promptId) === state.currentPrompt) {
            box.classList.add('active');
        } else {
            box.classList.remove('active');
        }
    });
}

// Event Listeners Setup
function setupEventListeners() {
    // Theme toggle
    document.getElementById('theme-toggle')?.addEventListener('click', toggleTheme);

    // Info button
    document.getElementById('info-btn')?.addEventListener('click', toggleInfo);

    // Select all traits button
    document.getElementById('select-all-btn')?.addEventListener('click', toggleAllTraits);

    // Prompt selector (event delegation)
    document.getElementById('prompt-selector')?.addEventListener('click', (e) => {
        const promptBox = e.target.closest('.prompt-box');
        if (promptBox) {
            const promptId = parseInt(promptBox.dataset.promptId);
            console.log(`Prompt box clicked. ID: ${promptId}`); // Debugging line
            if (state.currentPrompt !== promptId && !isNaN(promptId)) {
                state.currentPrompt = promptId;
                updatePromptSelectionUI();
                if (window.renderView) window.renderView();
            }
        }
    });

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

// Plotly Layout Helper
function getPlotlyLayout(baseLayout) {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    return {
        ...baseLayout,
        paper_bgcolor: isDark ? '#2d2d2d' : '#ffffff',
        plot_bgcolor: isDark ? '#2d2d2d' : '#ffffff',
        font: {
            color: isDark ? '#e0e0e0' : '#333'
        },
        xaxis: {
            ...baseLayout.xaxis,
            color: isDark ? '#e0e0e0' : '#333',
            gridcolor: isDark ? '#444' : '#ddd'
        },
        yaxis: {
            ...baseLayout.yaxis,
            color: isDark ? '#e0e0e0' : '#333',
            gridcolor: isDark ? '#444' : '#ddd'
        }
    };
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
window.showError = showError;
window.initApp = init;
window.markdownToHtml = markdownToHtml;
