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
                    window.state.selectedTraits.add(trait.name);
                } else {
                    window.state.selectedTraits.delete(trait.name);
                }
                updateCategoryCheckbox(categoryDiv);
                updateSelectedCount();
                if (window.renderView) window.renderView();
            });

            if (isDefaultSelected) {
                window.state.selectedTraits.add(trait.name);
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
                    window.state.selectedTraits.add(cb.value);
                } else {
                    window.state.selectedTraits.delete(cb.value);
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
        countElem.textContent = window.state.selectedTraits.size;
    }
}

function toggleAllTraits() {
    const traitCheckboxes = document.querySelectorAll('.trait-category-items input[type="checkbox"]');
    const allSelected = window.state.selectedTraits.size === traitCheckboxes.length;

    traitCheckboxes.forEach(cb => {
        cb.checked = !allSelected;
        if (!allSelected) {
            window.state.selectedTraits.add(cb.value);
        } else {
            window.state.selectedTraits.delete(cb.value);
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
 * Render the experiment picker list into #experiment-list.
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

    list.innerHTML = visibleExperiments.map(exp => {
        const isActive = activeExperiment ? exp === activeExperiment : false;
        return `<label class="experiment-option ${isActive ? 'active' : ''}" data-experiment="${exp}">
            <input type="radio" name="experiment" ${isActive ? 'checked' : ''}>
            <span>${exp}</span>
        </label>`;
    }).join('');

    // Add toggle link if there are hidden experiments
    if (hiddenCount > 0) {
        const toggleText = window.state.showAllExperiments ? 'Hide' : `Show ${hiddenCount} hidden`;
        list.innerHTML += `<div class="experiment-toggle" onclick="window.toggleHiddenExperiments()">${toggleText}</div>`;
    }

    // Attach click handlers for experiment selection
    list.querySelectorAll('.experiment-option').forEach(item => {
        item.addEventListener('click', async () => {
            list.querySelectorAll('.experiment-option').forEach(i => i.classList.remove('active'));
            item.classList.add('active');
            window.state.currentExperiment = item.dataset.experiment;
            setExperimentInURL(window.state.currentExperiment);
            await loadExperimentData(window.state.currentExperiment);
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

    // Select all traits button
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
