/**
 * Sidebar component - handles trait checkboxes, navigation, theme, GPU status,
 * and experiment list rendering.
 * Depends on: state.js (window.state), display.js (getDisplayName)
 */

import { ANALYSIS_VIEWS, setTabInURL, setExperimentInURL, ensureExperimentLoaded, loadExperimentData } from '../core/state.js';
import { getDisplayName } from '../core/display.js';

// =============================================================================
// Theme Management
// =============================================================================

function initTheme() {
    const savedTheme = localStorage.getItem('theme') || 'dark';
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
    if (themeToggle) {
        themeToggle.textContent = theme === 'dark' ? '☀️' : '🌙';
        themeToggle.title = theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode';
    }
}

// =============================================================================
// Trait Selection
// =============================================================================

/** Total number of unique traits (set during populateTraitCheckboxes) */
let _totalTraitCount = 0;

function populateTraitCheckboxes() {
    const container = document.getElementById('trait-checkboxes');
    if (!container) return;

    container.innerHTML = '';
    window.state.selectedTraits.clear();

    if (!window.state.experimentData || !window.state.experimentData.traits) return;

    // Deduplicate traits
    const uniqueTraits = window.state.experimentData.traits.filter((trait, index, self) =>
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

    _totalTraitCount = uniqueTraits.length;

    // Restyle the existing static select-all button as a trait-select-all link
    const selectAllBtn = document.getElementById('select-all-btn');
    if (selectAllBtn) {
        selectAllBtn.className = 'trait-select-all';
        // Determine initial label based on default selection
        const allDefaultSelected = !hasOg10;
        selectAllBtn.textContent = allDefaultSelected ? 'Deselect All' : 'Select All';
    }

    sortedCategories.forEach(category => {
        const traits = categories[category];
        const isDefaultSelected = hasOg10 ? category === 'og_10' : true;

        // Category header with expand/collapse + select-all on label click
        const header = document.createElement('div');
        header.className = 'trait-cat-header';

        const arrow = document.createElement('span');
        arrow.className = 'trait-cat-arrow';
        arrow.textContent = '▸';
        header.appendChild(arrow);

        const label = document.createElement('span');
        label.className = 'trait-cat-label';
        label.textContent = category.replace(/_/g, ' ');
        header.appendChild(label);

        const count = document.createElement('span');
        count.className = 'trait-cat-count';
        count.textContent = `${traits.length}`;
        header.appendChild(count);

        container.appendChild(header);

        // Chips container (collapsed by default)
        const chipsDiv = document.createElement('div');
        chipsDiv.className = 'trait-chips';
        chipsDiv.dataset.category = category;
        chipsDiv.hidden = true;

        traits.forEach(trait => {
            const chip = document.createElement('span');
            chip.className = 'trait-chip' + (isDefaultSelected ? ' selected' : '');
            chip.dataset.trait = trait.name;
            chip.textContent = getDisplayName(trait.name);

            chip.addEventListener('click', () => {
                const isSelected = chip.classList.toggle('selected');
                if (isSelected) {
                    window.state.selectedTraits.add(trait.name);
                } else {
                    window.state.selectedTraits.delete(trait.name);
                }
                updateSelectedCount();
                updateSelectAllLabel();
                if (window.renderView) window.renderView();
            });

            chipsDiv.appendChild(chip);

            if (isDefaultSelected) {
                window.state.selectedTraits.add(trait.name);
            }
        });

        container.appendChild(chipsDiv);

        // Arrow click: expand/collapse chips
        arrow.addEventListener('click', (e) => {
            e.stopPropagation();
            chipsDiv.hidden = !chipsDiv.hidden;
            arrow.textContent = chipsDiv.hidden ? '▸' : '▾';
        });

        // Label/count click: toggle all traits in this category
        const toggleCategory = () => {
            // Expand if collapsed
            if (chipsDiv.hidden) {
                chipsDiv.hidden = false;
                arrow.textContent = '▾';
            }
            const chips = chipsDiv.querySelectorAll('.trait-chip');
            const allInCatSelected = Array.from(chips).every(c => c.classList.contains('selected'));

            chips.forEach(c => {
                const traitName = c.dataset.trait;
                if (allInCatSelected) {
                    c.classList.remove('selected');
                    window.state.selectedTraits.delete(traitName);
                } else {
                    c.classList.add('selected');
                    window.state.selectedTraits.add(traitName);
                }
            });

            updateSelectedCount();
            updateSelectAllLabel();
            if (window.renderView) window.renderView();
        };
        label.addEventListener('click', (e) => { e.stopPropagation(); toggleCategory(); });
        count.addEventListener('click', (e) => { e.stopPropagation(); toggleCategory(); });
    });

    updateSelectedCount();
    updateSelectAllLabel();
}

function updateSelectAllLabel() {
    const btn = document.getElementById('select-all-btn');
    if (btn) {
        btn.textContent = window.state.selectedTraits.size > 0 ? 'Deselect All' : 'Select All';
    }
}

function updateSelectedCount() {
    const countElem = document.getElementById('selected-count');
    if (countElem) {
        countElem.textContent = window.state.selectedTraits.size;
    }
}

function toggleAllTraits() {
    const allChips = document.querySelectorAll('.trait-chip');
    const anySelected = window.state.selectedTraits.size > 0;

    allChips.forEach(chip => {
        const traitName = chip.dataset.trait;
        if (anySelected) {
            chip.classList.remove('selected');
            window.state.selectedTraits.delete(traitName);
        } else {
            chip.classList.add('selected');
            window.state.selectedTraits.add(traitName);
        }
    });

    updateSelectedCount();
    updateSelectAllLabel();
    if (window.renderView) window.renderView();
}

// =============================================================================
// Navigation
// =============================================================================

function setupNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    const analysisEntry = document.getElementById('analysis-entry');

    navItems.forEach(item => {
        item.addEventListener('click', async () => {
            // Analysis entry point: open panel and navigate to last analysis view
            if (item === analysisEntry) {
                const targetView = window.state.lastAnalysisView || 'extraction';
                window.state.currentView = targetView;
                setTabInURL(targetView);

                navItems.forEach(n => n.classList.remove('active'));
                analysisEntry.classList.add('active');
                const subNav = document.querySelector(`#sidebar-analysis .nav-item[data-view="${targetView}"]`);
                if (subNav) subNav.classList.add('active');

                updatePageTitle();
                updateExperimentVisibility();
                await ensureExperimentLoaded();
                window.renderPromptPicker();
                if (window.renderPromptSetSidebar) window.renderPromptSetSidebar();
                if (window.renderView) window.renderView();
                return;
            }

            navItems.forEach(n => n.classList.remove('active'));
            item.classList.add('active');

            if (item.dataset.view) {
                window.state.currentView = item.dataset.view;
                setTabInURL(item.dataset.view);

                // Analysis sub-nav: keep the main sidebar entry highlighted
                if (ANALYSIS_VIEWS.includes(item.dataset.view)) {
                    window.state.lastAnalysisView = item.dataset.view;
                    if (analysisEntry) analysisEntry.classList.add('active');
                }

                updatePageTitle();
                updateExperimentVisibility();

                // Auto-load experiment if switching to analysis view and none selected
                await ensureExperimentLoaded();

                window.renderPromptPicker();
                if (window.renderPromptSetSidebar) window.renderPromptSetSidebar();
                if (window.renderView) window.renderView();
            }
        });
    });
}

function updatePageTitle() {
    const titles = {
        'overview': 'Overview',
        'extraction': 'Extraction',
        'steering': 'Steering',
        'trait-dynamics': 'Inference',
        'model-analysis': 'Model Analysis'
    };
    const titleElem = document.getElementById('page-title');
    if (titleElem) {
        titleElem.textContent = titles[window.state.currentView] || 'Data Explorer';
    }
}

/**
 * Show/hide the analysis panel based on current view.
 * Only analysis views show the second sidebar column.
 */
function updateExperimentVisibility() {
    const analysisPanel = document.getElementById('sidebar-analysis');
    if (!analysisPanel) return;

    const isAnalysis = ANALYSIS_VIEWS.includes(window.state.currentView);
    analysisPanel.classList.toggle('hidden', !isAnalysis);
}

// =============================================================================
// Subsection Info Toggles
// =============================================================================

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

// =============================================================================
// GPU Status Widget
// =============================================================================

let gpuPollInterval = null;

async function fetchGpuStatus() {
    try {
        const response = await fetch('/api/gpu-status');
        if (!response.ok) throw new Error('Failed to fetch GPU status');
        updateGpuStatusUI(await response.json());
    } catch (e) {
        console.warn('GPU status fetch failed:', e);
        updateGpuStatusUI({ available: false, device: 'Unknown', error: e.message });
    }
}

function updateGpuStatusUI(status) {
    const container = document.getElementById('gpu-status');
    if (!container) return;

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

// =============================================================================
// Experiment List Rendering
// =============================================================================

/**
 * Render the experiment picker as a hover dropdown into #experiment-list.
 * @param {string[]} experiments - All experiment names
 * @param {string[]} hiddenExperiments - Experiments hidden by default
 * @param {string|null} activeExperiment - Currently active experiment (null = first)
 */
function renderExperimentList(experiments, hiddenExperiments, activeExperiment = null) {
    const list = document.getElementById('experiment-list');
    if (!list) return;

    // Filter experiments unless showAllExperiments is true
    const hiddenCount = experiments.filter(exp => hiddenExperiments.includes(exp)).length;
    const visibleExperiments = window.state.showAllExperiments
        ? experiments
        : experiments.filter(exp => !hiddenExperiments.includes(exp));

    const activeName = activeExperiment || 'Select...';

    let menuItems = visibleExperiments.map(exp => {
        const isActive = exp === activeExperiment;
        return `<div class="exp-menu-item${isActive ? ' active' : ''}" data-experiment="${exp}">${exp}</div>`;
    }).join('');

    // Add toggle link if there are hidden experiments
    if (hiddenCount > 0) {
        const toggleText = window.state.showAllExperiments ? 'Hide hidden' : `Show ${hiddenCount} hidden`;
        menuItems += `<div class="exp-menu-item" style="color: var(--text-tertiary); font-style: italic;" data-toggle-hidden="true">${toggleText}</div>`;
    }

    list.innerHTML = `
        <div class="exp-dropdown">
            <div class="exp-trigger">
                <span class="exp-name">${activeName}</span>
                <span class="exp-arrow">&#9662;</span>
            </div>
            <div class="exp-menu">${menuItems}</div>
        </div>
    `;

    // Attach click handlers for experiment selection
    list.querySelectorAll('.exp-menu-item').forEach(item => {
        if (item.dataset.toggleHidden) {
            item.addEventListener('click', (e) => {
                e.stopPropagation();
                window.toggleHiddenExperiments();
            });
            return;
        }
        item.addEventListener('click', async (e) => {
            e.stopPropagation();
            const expName = item.dataset.experiment;
            window.state.currentExperiment = expName;
            setExperimentInURL(expName);

            // Update trigger text and active state immediately
            const trigger = list.querySelector('.exp-name');
            if (trigger) trigger.textContent = expName;
            list.querySelectorAll('.exp-menu-item').forEach(i => i.classList.remove('active'));
            item.classList.add('active');

            await loadExperimentData(expName);
            window.renderPromptPicker();
            if (window.renderView) window.renderView();
        });
    });
}

// =============================================================================
// Event Listeners
// =============================================================================

function setupSidebarEventListeners() {
    // Theme toggle
    document.getElementById('theme-toggle')?.addEventListener('click', toggleTheme);

    // Select all traits button (static element in HTML, restyled in populateTraitCheckboxes)
    document.getElementById('select-all-btn')?.addEventListener('click', toggleAllTraits);
}

// ES module exports
export {
    initTheme,
    toggleTheme,
    populateTraitCheckboxes,
    toggleAllTraits,
    setupNavigation,
    updatePageTitle,
    updateExperimentVisibility,
    setupSubsectionInfoToggles,
    setupSidebarEventListeners,
    fetchGpuStatus,
    startGpuPolling,
    renderExperimentList,
};

// Keep window.* for remaining consumers (HTML templates, cross-module access during migration)
window.initTheme = initTheme;
window.populateTraitCheckboxes = populateTraitCheckboxes;
window.setupNavigation = setupNavigation;
window.updateExperimentVisibility = updateExperimentVisibility;
window.setupSubsectionInfoToggles = setupSubsectionInfoToggles;
window.setupSidebarEventListeners = setupSidebarEventListeners;
window.fetchGpuStatus = fetchGpuStatus;
window.startGpuPolling = startGpuPolling;
window.renderExperimentList = renderExperimentList;
