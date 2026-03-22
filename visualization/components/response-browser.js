/**
 * Response Browser Component
 * Displays steering run responses in an interactive table with filtering/sorting.
 * Extracted from steering.js
 *
 * Dependencies: state.js, display.js, paths.js
 */

// Track current sort state per trait
const responseBrowserState = {};

// Cache for trait definitions and judge templates
const traitDefinitionCache = {};
let judgeTemplatesCache = null;

// Reference to cached results from parent view (set externally)
let traitResultsCache = {};

/**
 * Set the trait results cache reference (called by steering.js)
 */
function setTraitResultsCache(cache) {
    traitResultsCache = cache;
}

/**
 * Fetch available response files for a set of runs
 * Returns { responses: Set of keys, baselines: Map of model_variant|prompt_set -> entry }
 */
async function fetchAvailableResponses(allRuns) {
    const experiment = window.state.experimentData?.name;
    if (!experiment) return { responses: new Set(), baselines: new Map() };

    // Get unique entries (trait/model_variant/position/prompt_set combinations)
    const uniqueEntries = new Map();
    for (const run of allRuns) {
        const entry = run.entry;
        if (!entry) continue;
        const entryKey = `${entry.trait}|${entry.model_variant}|${entry.position}|${entry.prompt_set}`;
        if (!uniqueEntries.has(entryKey)) {
            uniqueEntries.set(entryKey, entry);
        }
    }

    // Fetch response files for each unique entry in parallel
    const availableKeys = new Set();
    const availableBaselines = new Map();

    await Promise.all([...uniqueEntries.values()].map(async (entry) => {
        try {
            const url = `/api/experiments/${experiment}/steering-responses/${entry.trait}/${entry.model_variant}/${entry.position}/${entry.prompt_set}`;
            const response = await fetch(url);
            if (!response.ok) return;
            const data = await response.json();

            // Add each available response file to the set
            for (const file of data.files || []) {
                const key = `${entry.trait}|${entry.model_variant}|${entry.position}|${entry.prompt_set}|${file.component}|${file.method}|${file.layer}|${file.coef.toFixed(1)}`;
                availableKeys.add(key);
            }

            // Track baseline availability (keyed by model_variant|prompt_set, ignore position)
            if (data.baseline) {
                const baselineKey = `${entry.model_variant}|${entry.prompt_set}`;
                if (!availableBaselines.has(baselineKey)) {
                    availableBaselines.set(baselineKey, entry);
                }
            }
        } catch (e) {
            console.error('Failed to fetch response files for entry:', entry, e);
        }
    }));

    return { responses: availableKeys, baselines: availableBaselines };
}

/**
 * Render the response browser table for a trait
 */
async function renderResponseBrowserForTrait(trait) {
    const browserId = `response-browser-${trait.replace(/\//g, '-')}`;
    const container = document.getElementById(browserId);
    if (!container) return;

    const cached = traitResultsCache[trait];
    if (!cached || !cached.allRuns.length) {
        container.innerHTML = '<p class="no-data">No response data available</p>';
        return;
    }

    // Fetch available response files if not cached
    if (!cached.availableResponses) {
        container.innerHTML = ui.renderLoading('Loading available responses...');
        const result = await fetchAvailableResponses(cached.allRuns);
        cached.availableResponses = result.responses;
        cached.availableBaselines = result.baselines;
    }

    // Filter to only runs with available response files
    const runsWithResponses = cached.allRuns.filter(run => {
        const key = `${run.entry?.trait}|${run.entry?.model_variant}|${run.entry?.position}|${run.entry?.prompt_set}|${run.component}|${run.method}|${run.layer}|${run.coef.toFixed(1)}`;
        return cached.availableResponses.has(key);
    });

    if (runsWithResponses.length === 0) {
        container.innerHTML = '<p class="no-data">No response files saved for this trait</p>';
        return;
    }

    // Initialize state for this trait
    if (!responseBrowserState[trait]) {
        responseBrowserState[trait] = {
            sortKey: 'traitScore',
            sortDir: 'desc',
            layerFilter: new Set(), // empty = show all
            expandedRow: null,
            bestPerLayer: true, // Show only best run per layer (default on)
            infoPanel: null, // 'definition' | 'judge' | null
            compactResponses: true, // Show newlines as \n (default on)
            promptSetFilter: 'all', // filter by prompt set
            steeringDirection: 'all', // 'all' | 'positive' | 'negative'
            modelVariantFilter: 'all', // filter by model variant
            currentBaselineEntry: null, // entry to use for baseline panel
        };
    }
    const state = responseBrowserState[trait];

    // Get coherence threshold from the page slider
    const coherenceThresholdEl = document.getElementById('sweep-coherence-threshold');
    const coherenceThreshold = coherenceThresholdEl ? parseInt(coherenceThresholdEl.value) : 77;

    // Get unique values for filters (from runs with responses only)
    const uniqueLayers = [...new Set(runsWithResponses.map(r => r.layer))].sort((a, b) => a - b);
    const uniquePromptSets = [...new Set(runsWithResponses.map(r => r.entry?.prompt_set || 'steering'))].sort();
    const uniqueModelVariants = [...new Set(runsWithResponses.map(r => r.entry?.model_variant || 'unknown'))].sort();
    const hasPositive = runsWithResponses.some(r => r.coef > 0);
    const hasNegative = runsWithResponses.some(r => r.coef < 0);

    // Check baseline availability for current filter selection
    let baselineEntry = null;
    if (cached.availableBaselines && cached.availableBaselines.size > 0) {
        if (state.modelVariantFilter !== 'all' && state.promptSetFilter !== 'all') {
            // Specific filter - check if that combo has baseline
            const key = `${state.modelVariantFilter}|${state.promptSetFilter}`;
            baselineEntry = cached.availableBaselines.get(key) || null;
        } else if (state.modelVariantFilter !== 'all') {
            // Model variant set, prompt set is 'all' - find any baseline for this model
            for (const [key, entry] of cached.availableBaselines) {
                if (key.startsWith(`${state.modelVariantFilter}|`)) {
                    baselineEntry = entry;
                    break;
                }
            }
        } else if (state.promptSetFilter !== 'all') {
            // Prompt set set, model variant is 'all' - find any baseline for this prompt set
            for (const [key, entry] of cached.availableBaselines) {
                if (key.endsWith(`|${state.promptSetFilter}`)) {
                    baselineEntry = entry;
                    break;
                }
            }
        } else {
            // Both 'all' - use first available baseline
            baselineEntry = cached.availableBaselines.values().next().value || null;
        }
    }
    // Store in state for loadInfoPanelContent to use
    state.currentBaselineEntry = baselineEntry;

    // Filter and sort runs
    let runs = [...runsWithResponses];

    // Filter by model variant
    if (state.modelVariantFilter !== 'all') {
        runs = runs.filter(r => (r.entry?.model_variant || 'unknown') === state.modelVariantFilter);
    }

    // Filter by prompt set
    if (state.promptSetFilter !== 'all') {
        runs = runs.filter(r => (r.entry?.prompt_set || 'steering') === state.promptSetFilter);
    }

    // Filter by steering direction
    if (state.steeringDirection === 'positive') {
        runs = runs.filter(r => r.coef > 0);
    } else if (state.steeringDirection === 'negative') {
        runs = runs.filter(r => r.coef < 0);
    }

    // Filter by layer
    if (state.layerFilter.size > 0) {
        runs = runs.filter(r => state.layerFilter.has(r.layer));
    }

    // Best per layer filter: keep only highest trait score per layer (with coherence >= threshold)
    if (state.bestPerLayer) {
        const bestByLayer = {};
        for (const run of runs) {
            if (run.coherence < coherenceThreshold) continue;
            if (!bestByLayer[run.layer] || run.traitScore > bestByLayer[run.layer].traitScore) {
                bestByLayer[run.layer] = run;
            }
        }
        runs = Object.values(bestByLayer);
    }

    // Sort
    runs.sort((a, b) => {
        const aVal = a[state.sortKey] ?? 0;
        const bVal = b[state.sortKey] ?? 0;
        return state.sortDir === 'desc' ? bVal - aVal : aVal - bVal;
    });

    // Get unique positions for display
    const uniquePositions = [...new Set(cached.allRuns.map(r => r.entry?.position || 'unknown'))];
    const showPositionCol = uniquePositions.length > 1 || uniquePositions[0] !== 'response_all';

    // Build HTML
    container.innerHTML = `
        <div class="rb-filters">
            <span class="rb-filter-label">Layers:</span>
            <div class="rb-layer-chips">
                ${ui.renderChip({ label: 'All', dataAttr: { key: 'action', value: 'select-all' }, className: 'rb-chip-btn' })}
                ${ui.renderChip({ label: 'None', dataAttr: { key: 'action', value: 'select-none' }, className: 'rb-chip-btn' })}
                ${uniqueLayers.map(l => `
                    <label class="rb-chip ${state.layerFilter.size === 0 || state.layerFilter.has(l) ? 'active' : ''}">
                        <input type="checkbox" value="${l}" ${state.layerFilter.size === 0 || state.layerFilter.has(l) ? 'checked' : ''}>
                        L${l}
                    </label>
                `).join('')}
            </div>
            ${(hasPositive && hasNegative) ? `
            <div class="rb-dropdown-group">
                <label class="rb-filter-label">Direction:</label>
                <select class="rb-select" data-filter="direction">
                    <option value="all" ${state.steeringDirection === 'all' ? 'selected' : ''}>All</option>
                    <option value="positive" ${state.steeringDirection === 'positive' ? 'selected' : ''}>Positive (+)</option>
                    <option value="negative" ${state.steeringDirection === 'negative' ? 'selected' : ''}>Negative (−)</option>
                </select>
            </div>
            ` : ''}
            ${uniquePromptSets.length > 1 ? `
            <div class="rb-dropdown-group">
                <label class="rb-filter-label">Prompt set:</label>
                <select class="rb-select" data-filter="prompt-set">
                    <option value="all" ${state.promptSetFilter === 'all' ? 'selected' : ''}>All</option>
                    ${uniquePromptSets.map(ps => `
                        <option value="${ps}" ${state.promptSetFilter === ps ? 'selected' : ''}>${ps}</option>
                    `).join('')}
                </select>
            </div>
            ` : ''}
            ${uniqueModelVariants.length > 1 ? `
            <div class="rb-dropdown-group">
                <label class="rb-filter-label">Model:</label>
                <select class="rb-select" data-filter="model-variant">
                    <option value="all" ${state.modelVariantFilter === 'all' ? 'selected' : ''}>All</option>
                    ${uniqueModelVariants.map(mv => `
                        <option value="${mv}" ${state.modelVariantFilter === mv ? 'selected' : ''}>${mv}</option>
                    `).join('')}
                </select>
            </div>
            ` : ''}
            <div class="rb-info-btns">
                ${ui.renderChip({ label: 'Definition', active: state.infoPanel === 'definition', dataAttr: { key: 'info', value: 'definition' }, className: 'rb-info-btn' })}
                ${ui.renderChip({ label: 'Judge Prompt', active: state.infoPanel === 'judge', dataAttr: { key: 'info', value: 'judge' }, className: 'rb-info-btn' })}
                ${baselineEntry ? ui.renderChip({ label: 'Baseline', active: state.infoPanel === 'baseline', dataAttr: { key: 'info', value: 'baseline' }, className: 'rb-info-btn' }) : ''}
            </div>
            ${ui.renderToggle({
                label: `Best per layer (coh ≥${coherenceThreshold})`,
                checked: state.bestPerLayer,
                dataAttr: { key: 'action', value: 'best-per-layer' },
                className: 'rb-toggle'
            })}
            ${ui.renderToggle({
                label: 'Compact responses',
                checked: state.compactResponses,
                dataAttr: { key: 'action', value: 'compact-responses' },
                className: 'rb-toggle'
            })}
        </div>
        ${state.infoPanel ? `
        <div class="rb-info-panel" data-panel="${state.infoPanel}">
            <div class="rb-info-content">${ui.renderLoading()}</div>
        </div>
        ` : ''}
        <div class="rb-table-wrapper">
            <table class="table table-compact data-table rb-table">
                <thead>
                    <tr>
                        ${ui.renderSortableHeader({ key: 'layer', label: 'Layer', sortKey: state.sortKey, sortDir: state.sortDir })}
                        ${ui.renderSortableHeader({ key: 'coef', label: 'Coef', sortKey: state.sortKey, sortDir: state.sortDir })}
                        <th>Method</th>
                        <th>Component</th>
                        ${showPositionCol ? '<th>Position</th>' : ''}
                        ${ui.renderSortableHeader({ key: 'traitScore', label: 'Trait', sortKey: state.sortKey, sortDir: state.sortDir })}
                        ${ui.renderSortableHeader({ key: 'coherence', label: 'Coh', sortKey: state.sortKey, sortDir: state.sortDir })}
                    </tr>
                </thead>
                <tbody>
                    ${runs.map((run, idx) => {
                        const position = run.entry?.position || 'unknown';
                        const posDisplay = window.paths?.formatPositionDisplay ? window.paths.formatPositionDisplay(position) : position;
                        const promptSet = run.entry?.prompt_set;
                        const promptSetDisplay = promptSet && promptSet !== 'steering' ? ` [${promptSet}]` : '';
                        return `
                        <tr class="rb-row ${state.expandedRow === idx ? 'expanded' : ''} ${run.coherence < coherenceThreshold ? 'below-threshold' : ''}" data-idx="${idx}">
                            <td>L${run.layer}</td>
                            <td>${run.coef.toFixed(1)}</td>
                            <td>${run.method}</td>
                            <td>${run.component}</td>
                            ${showPositionCol ? `<td class="rb-position">${posDisplay}${promptSetDisplay}</td>` : ''}
                            <td class="${ui.scoreClass(run.traitScore)}">${run.traitScore.toFixed(1)}</td>
                            <td class="${ui.scoreClass(run.coherence, 'coherence')}">${run.coherence.toFixed(0)}</td>
                        </tr>
                        ${state.expandedRow === idx ? `
                        <tr class="rb-expanded-row">
                            <td colspan="${showPositionCol ? 7 : 6}">
                                <div class="rb-responses-container" id="rb-responses-${trait.replace(/\//g, '-')}-${idx}">
                                    ${ui.renderLoading('Loading responses...')}
                                </div>
                            </td>
                        </tr>
                        ` : ''}
                    `;}).join('')}
                </tbody>
            </table>
        </div>
        <div class="rb-stats hint">${runs.length} of ${runsWithResponses.length} runs with responses${state.bestPerLayer ? ' (best per layer)' : ''}</div>
    `;

    // Setup event handlers
    setupResponseBrowserHandlers(trait, container, runs);

    // Load responses if a row is expanded
    if (state.expandedRow !== null && runs[state.expandedRow]) {
        loadResponsesForRun(trait, state.expandedRow, runs[state.expandedRow]);
    }

    // Load info panel content if open
    if (state.infoPanel) {
        loadInfoPanelContent(trait, state.infoPanel);
    }
}

/**
 * Setup event handlers for response browser
 */
function setupResponseBrowserHandlers(trait, container, runs) {
    const state = responseBrowserState[trait];
    const allLayers = [...new Set(traitResultsCache[trait].allRuns.map(r => r.layer))];

    // Select All / Select None buttons
    container.querySelectorAll('.rb-chip-btn').forEach(btn => {
        btn.addEventListener('click', async () => {
            const action = btn.dataset.action;
            if (action === 'select-all') {
                state.layerFilter.clear(); // Empty = show all
            } else if (action === 'select-none') {
                state.layerFilter.clear();
                state.layerFilter.add(-999); // Impossible layer = show none
            }
            state.expandedRow = null;
            await renderResponseBrowserForTrait(trait);
        });
    });

    // Best per layer toggle
    const bestPerLayerCheckbox = container.querySelector('input[data-action="best-per-layer"]');
    if (bestPerLayerCheckbox) {
        bestPerLayerCheckbox.addEventListener('change', async () => {
            state.bestPerLayer = bestPerLayerCheckbox.checked;
            state.expandedRow = null;
            await renderResponseBrowserForTrait(trait);
        });
    }

    // Compact responses toggle
    const compactCheckbox = container.querySelector('.rb-filters input[data-action="compact-responses"]');
    if (compactCheckbox) {
        compactCheckbox.addEventListener('change', async () => {
            state.compactResponses = compactCheckbox.checked;
            await renderResponseBrowserForTrait(trait);
        });
    }

    // Direction filter dropdown
    const directionSelect = container.querySelector('select[data-filter="direction"]');
    if (directionSelect) {
        directionSelect.addEventListener('change', async () => {
            state.steeringDirection = directionSelect.value;
            state.expandedRow = null;
            await renderResponseBrowserForTrait(trait);
        });
    }

    // Prompt set filter dropdown
    const promptSetSelect = container.querySelector('select[data-filter="prompt-set"]');
    if (promptSetSelect) {
        promptSetSelect.addEventListener('change', async () => {
            state.promptSetFilter = promptSetSelect.value;
            state.expandedRow = null;
            await renderResponseBrowserForTrait(trait);
        });
    }

    // Model variant filter dropdown
    const modelVariantSelect = container.querySelector('select[data-filter="model-variant"]');
    if (modelVariantSelect) {
        modelVariantSelect.addEventListener('change', async () => {
            state.modelVariantFilter = modelVariantSelect.value;
            state.expandedRow = null;
            await renderResponseBrowserForTrait(trait);
        });
    }

    // Layer filter checkboxes
    container.querySelectorAll('.rb-chip input').forEach(checkbox => {
        checkbox.addEventListener('change', async () => {
            const layer = parseInt(checkbox.value);
            if (checkbox.checked) {
                // If all were selected (filter empty), start fresh with just this one
                if (state.layerFilter.size === 0) {
                    allLayers.forEach(l => state.layerFilter.add(l));
                }
                state.layerFilter.add(layer);
                // Remove impossible layer if it was set
                state.layerFilter.delete(-999);
            } else {
                if (state.layerFilter.size === 0) {
                    // First uncheck - add all except this one
                    allLayers.forEach(l => { if (l !== layer) state.layerFilter.add(l); });
                } else {
                    state.layerFilter.delete(layer);
                }
            }
            state.expandedRow = null; // Close expanded row on filter change
            await renderResponseBrowserForTrait(trait);
        });
    });

    // Sortable headers
    container.querySelectorAll('th.sortable').forEach(th => {
        th.addEventListener('click', async () => {
            const sortKey = th.dataset.sort;
            if (state.sortKey === sortKey) {
                state.sortDir = state.sortDir === 'desc' ? 'asc' : 'desc';
            } else {
                state.sortKey = sortKey;
                state.sortDir = 'desc';
            }
            state.expandedRow = null;
            await renderResponseBrowserForTrait(trait);
        });
    });

    // Row click to expand
    container.querySelectorAll('.rb-row').forEach(row => {
        row.addEventListener('click', async () => {
            const idx = parseInt(row.dataset.idx);
            if (state.expandedRow === idx) {
                state.expandedRow = null;
            } else {
                state.expandedRow = idx;
            }
            await renderResponseBrowserForTrait(trait);
        });
    });

    // Info panel buttons (Definition / Judge Prompt)
    container.querySelectorAll('.rb-info-btn').forEach(btn => {
        btn.addEventListener('click', async () => {
            const infoType = btn.dataset.info;
            if (state.infoPanel === infoType) {
                state.infoPanel = null; // Toggle off
            } else {
                state.infoPanel = infoType;
            }
            await renderResponseBrowserForTrait(trait);

            // Load content after render if panel is open
            if (state.infoPanel) {
                await loadInfoPanelContent(trait, state.infoPanel);
            }
        });
    });
}

/**
 * Load and display info panel content (definition or judge prompt)
 */
async function loadInfoPanelContent(trait, panelType) {
    const browserId = `response-browser-${trait.replace(/\//g, '-')}`;
    const container = document.getElementById(browserId);
    const panel = container?.querySelector('.rb-info-content');
    if (!panel) return;

    // Extract trait name for display (last part of path)
    const traitName = trait.split('/').pop();

    try {
        // Fetch definition if not cached
        if (!traitDefinitionCache[trait]) {
            const defPath = `datasets/traits/${trait}/definition.txt`;
            const response = await fetch(`/${defPath}`);
            if (!response.ok) {
                traitDefinitionCache[trait] = { error: 'Definition file not found' };
            } else {
                traitDefinitionCache[trait] = { text: await response.text() };
            }
        }

        const cached = traitDefinitionCache[trait];

        if (cached.error) {
            panel.innerHTML = `<p class="no-data">${cached.error}</p>`;
            return;
        }

        if (panelType === 'definition') {
            panel.innerHTML = `<pre class="rb-code">${window.escapeHtml(cached.text.trim())}</pre>`;
        } else if (panelType === 'judge') {
            // Fetch judge templates if not cached
            if (!judgeTemplatesCache) {
                const resp = await fetch('/api/judge-templates');
                if (!resp.ok) {
                    panel.innerHTML = `<p class="no-data">Could not load judge templates</p>`;
                    return;
                }
                judgeTemplatesCache = await resp.json();
            }

            // Highlight template variables
            const highlightVars = (text) => {
                return window.escapeHtml(text).replace(/\{(\w+)\}/g, '<span class="rb-var">{$1}</span>');
            };

            const systemPrompt = judgeTemplatesCache.steering_system
                .replace('{trait_name}', traitName)
                .replace('{trait_definition}', cached.text.trim());

            panel.innerHTML = `
                <div class="rb-judge-header">
                    <span>model: <strong>gpt-4.1-mini</strong></span>
                    <span>scoring: <strong>logprob-weighted avg</strong></span>
                    <span>temp: <strong>0</strong></span>
                    <span>top_logprobs: <strong>20</strong></span>
                </div>
                <div class="rb-judge-section">
                    <span class="rb-code-label">system_prompt</span>
                    <pre class="rb-code">${highlightVars(systemPrompt)}</pre>
                </div>
                <div class="rb-judge-section">
                    <span class="rb-code-label">user</span>
                    <pre class="rb-code">${highlightVars(judgeTemplatesCache.steering_user)}</pre>
                </div>
            `;
        } else if (panelType === 'baseline') {
            // Load baseline responses
            const state = responseBrowserState[trait];
            const baselineEntry = state?.currentBaselineEntry;

            if (!baselineEntry) {
                panel.innerHTML = `<p class="no-data">No baseline available for current filter selection</p>`;
                return;
            }

            const experiment = window.state.experimentData?.name;
            const responsePath = window.paths.get('steering.responses', {
                experiment,
                trait: baselineEntry.trait,
                model_variant: baselineEntry.model_variant,
                position: baselineEntry.position,
                prompt_set: baselineEntry.prompt_set,
            });

            const url = `/${responsePath}/baseline.json`;
            const response = await fetch(url);

            if (!response.ok) {
                panel.innerHTML = `<p class="no-data">Baseline file not found</p>`;
                return;
            }

            const responses = await response.json();
            const isCompact = state?.compactResponses ?? true;

            // Show which baseline we're displaying
            const baselineLabel = `${baselineEntry.model_variant} / ${baselineEntry.prompt_set}`;

            panel.innerHTML = `
                <div class="rb-baseline-header hint">
                    Showing baseline for: <strong>${baselineLabel}</strong>
                </div>
                <div class="response-list-compact">
                    ${responses.map((r, i) => {
                        const responseText = isCompact
                            ? r.response.replace(/\n/g, '\\n')
                            : r.response;
                        return `
                        <div class="response-item-row">
                            <div class="response-meta">
                                <div class="meta-label">Prompt #${i + 1}</div>
                                <div class="meta-score">Trait: <span class="${ui.scoreClass(r.trait_score ?? 0)}">${r.trait_score?.toFixed(0) ?? '-'}</span></div>
                                <div class="meta-score">Coh: <span class="${ui.scoreClass(r.coherence_score ?? 0, 'coherence')}">${r.coherence_score?.toFixed(0) ?? '-'}</span></div>
                            </div>
                            <div class="response-content">
                                <div class="response-q">${window.escapeHtml(typeof r.prompt === 'object' ? r.prompt.question || JSON.stringify(r.prompt) : r.prompt)}</div>
                                <div class="response-a ${isCompact ? 'compact' : ''}">${window.escapeHtml(responseText)}</div>
                            </div>
                        </div>
                    `;}).join('')}
                </div>
            `;
        }

    } catch (e) {
        console.error('Failed to load info panel:', e);
        panel.innerHTML = `<p class="no-data">Error: ${e.message}</p>`;
    }
}

/**
 * Load and display responses for a specific run
 */
async function loadResponsesForRun(trait, idx, run) {
    const containerId = `rb-responses-${trait.replace(/\//g, '-')}-${idx}`;
    const container = document.getElementById(containerId);
    if (!container) return;

    const { entry } = run;
    const experiment = window.state.experimentData?.name;

    try {
        // Build path to response file
        const ts = run.timestamp ? run.timestamp.slice(0, 19).replace(/:/g, '-').replace('T', '_') : '';
        const filename = `L${run.layer}_c${run.coef.toFixed(1)}_${ts}.json`;
        const responsePath = window.paths.get('steering.responses', {
            experiment,
            trait: entry.trait,
            model_variant: entry.model_variant,
            position: entry.position,
            prompt_set: entry.prompt_set,
        });

        const url = `/${responsePath}/${run.component}/${run.method}/${filename}`;
        const response = await fetch(url);

        if (!response.ok) {
            container.innerHTML = `<p class="no-data">Response file not found</p>`;
            return;
        }

        const responses = await response.json();
        const state = responseBrowserState[trait];
        const isCompact = state?.compactResponses ?? true;

        container.innerHTML = `
            <div class="response-list-compact">
                ${responses.map((r, i) => {
                    // In compact mode, show \n as literal text; otherwise preserve whitespace
                    const responseText = isCompact
                        ? r.response.replace(/\n/g, '\\n')
                        : r.response;
                    return `
                    <div class="response-item-row">
                        <div class="response-meta">
                            <div class="meta-label">Prompt #${i + 1}</div>
                            <div class="meta-score">Trait: <span class="${ui.scoreClass(r.trait_score ?? 0)}">${r.trait_score?.toFixed(0) ?? '-'}</span></div>
                            <div class="meta-score">Coh: <span class="${ui.scoreClass(r.coherence_score ?? 0, 'coherence')}">${r.coherence_score?.toFixed(0) ?? '-'}</span></div>
                        </div>
                        <div class="response-content">
                            <div class="response-q">${window.escapeHtml(typeof r.prompt === 'object' ? r.prompt.question || JSON.stringify(r.prompt) : r.prompt)}</div>
                            <div class="response-a ${isCompact ? 'compact' : ''}">${window.escapeHtml(responseText)}</div>
                        </div>
                    </div>
                `;}).join('')}
            </div>
        `;

    } catch (e) {
        console.error('Failed to load responses:', e);
        container.innerHTML = `<p class="no-data">Error loading responses: ${e.message}</p>`;
    }
}

// Export
window.responseBrowser = {
    setTraitResultsCache,
    renderResponseBrowserForTrait,
    fetchAvailableResponses,
    // Expose state for debugging
    getState: () => responseBrowserState,
};

// Also export main function directly for backwards compatibility
window.renderResponseBrowserForTrait = renderResponseBrowserForTrait;
window.fetchAvailableResponses = fetchAvailableResponses;
