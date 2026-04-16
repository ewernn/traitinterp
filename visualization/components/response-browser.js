/**
 * Response Browser Component
 * Displays steering run responses in an interactive table with filtering/sorting.
 * Extracted from views/steering/steering-view.js
 *
 * Dependencies: state.js, display.js, paths.js
 */

import { escapeHtml } from '../core/utils.js';
import { displayLayer } from '../core/display.js';
import { renderLoading, renderToggle, renderSortableHeader, scoreClass } from '../core/ui.js';
import { renderStyledSelect, wireStyledSelect } from './styled-select.js';
import { renderChipGroup, wireChipGroup } from './styled-chip-group.js';

// Track current sort state per trait
const responseBrowserState = {};

// Cache for trait definitions and judge templates
const traitDefinitionCache = {};
let judgeTemplatesCache = null;

// Reference to cached results from parent view (set externally)
let traitResultsCache = {};

/** Reset expanded row and re-render the response browser for a trait */
async function resetAndRender(trait) {
    responseBrowserState[trait].expandedRow = null;
    await renderResponseBrowserForTrait(trait);
}

/** Attach a change listener to a toggle checkbox, updating state and re-rendering */
function attachToggleListener(container, selector, stateProperty, trait, { resetsRow = false } = {}) {
    const checkbox = container.querySelector(selector);
    if (checkbox) {
        checkbox.addEventListener('change', async () => {
            responseBrowserState[trait][stateProperty] = checkbox.checked;
            resetsRow ? await resetAndRender(trait) : await renderResponseBrowserForTrait(trait);
        });
    }
}

/** Convert trait path to CSS-safe slug (e.g., 'category/trait' -> 'category-trait') */
function traitSlug(trait) { return trait.replace(/\//g, '-'); }

/** Build a pipe-delimited key for a steering run (used for response file lookup) */
function runResponseKey(entry, component, method, layer, coef) {
    return `${entry.trait}|${entry.model_variant}|${entry.position}|${entry.prompt_set}|${component}|${method}|${layer}|${coef.toFixed(1)}`;
}

/** Render a response list wrapper around response items */
function renderResponseList(responses, isCompact) {
    return `<div class="response-list-compact">
                ${responses.map((r, i) => renderResponseItem(r, i, isCompact)).join('')}
            </div>`;
}

/** Build steering response path for an entry */
function steeringResponsePath(entry) {
    return window.paths.get('steering.responses', {
        experiment: window.state.experimentData?.name,
        trait: entry.trait,
        model_variant: entry.model_variant,
        position: entry.position,
        prompt_set: entry.prompt_set,
    });
}

/** Render a filter dropdown group (label + styled select with "All" default) */
function renderFilterDropdown(filterName, label, currentValue, options, stateProperty, trait) {
    const selectId = `rb-filter-${trait.replace(/[^a-z0-9]/gi, '_')}-${filterName}`;
    const allOptions = [
        { value: 'all', label: 'All' },
        ...options.map(opt => {
            const [value, display] = Array.isArray(opt) ? opt : [opt, opt];
            return { value, label: display };
        }),
    ];
    return `
        <div class="rb-dropdown-group">
            <label class="rb-filter-label">${label}:</label>
            ${renderStyledSelect({
                id: selectId,
                options: allOptions,
                selected: currentValue,
                onChange: async (val) => {
                    responseBrowserState[trait][stateProperty] = val;
                    await resetAndRender(trait);
                },
            })}
        </div>
    `;
}

/**
 * Set the trait results cache reference (called by views/steering/detail.js)
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
                availableKeys.add(runResponseKey(entry, file.component, file.method, file.layer, file.coef));
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
    const browserId = `response-browser-${traitSlug(trait)}`;
    const container = document.getElementById(browserId);
    if (!container) return;

    const cached = traitResultsCache[trait];
    if (!cached || !cached.allRuns.length) {
        container.innerHTML = '<p class="no-data">No response data available</p>';
        return;
    }

    // Fetch available response files if not cached
    if (!cached.availableResponses) {
        container.innerHTML = renderLoading('Loading available responses...');
        const result = await fetchAvailableResponses(cached.allRuns);
        cached.availableResponses = result.responses;
        cached.availableBaselines = result.baselines;
    }

    // Filter to only runs with available response files
    const runsWithResponses = cached.allRuns.filter(run =>
        run.entry && cached.availableResponses.has(runResponseKey(run.entry, run.component, run.method, run.layer, run.coef))
    );

    if (runsWithResponses.length === 0) {
        container.innerHTML = '<p class="no-data">No response files saved for this trait</p>';
        return;
    }

    // Initialize state for this trait
    if (!responseBrowserState[trait]) {
        // Default sort direction: descending for positive steering, ascending for negative
        const predominantlyNegative = runsWithResponses.length > 0 &&
            runsWithResponses.filter(r => r.coef < 0).length > runsWithResponses.filter(r => r.coef > 0).length;
        responseBrowserState[trait] = {
            sortKey: 'traitScore',
            sortDir: predominantlyNegative ? 'asc' : 'desc',
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
    if (!coherenceThresholdEl) throw new Error('sweep-coherence-threshold slider not found');
    const coherenceThreshold = parseInt(coherenceThresholdEl.value);

    // Get unique values for filters (from runs with responses only)
    const uniqueLayers = [...new Set(runsWithResponses.map(r => r.layer))].sort((a, b) => a - b);
    const uniquePromptSets = [...new Set(runsWithResponses.map(r => r.entry?.prompt_set || 'steering'))].sort();
    const uniqueModelVariants = [...new Set(runsWithResponses.map(r => r.entry?.model_variant || 'unknown'))].sort();
    const hasPositive = runsWithResponses.some(r => r.coef > 0);
    const hasNegative = runsWithResponses.some(r => r.coef < 0);

    // Check baseline availability for current filter selection
    let baselineEntry = null;
    if (cached.availableBaselines && cached.availableBaselines.size > 0) {
        const mv = state.modelVariantFilter, ps = state.promptSetFilter;
        if (mv !== 'all' && ps !== 'all') {
            baselineEntry = cached.availableBaselines.get(`${mv}|${ps}`) || null;
        } else {
            // Find first baseline matching partial filter (or any if both 'all')
            const matchesFilter = (key) =>
                (mv === 'all' || key.startsWith(`${mv}|`)) &&
                (ps === 'all' || key.endsWith(`|${ps}`));
            for (const [key, entry] of cached.availableBaselines) {
                if (matchesFilter(key)) { baselineEntry = entry; break; }
            }
        }
    }
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

    // Best per layer filter: keep run with most extreme trait score per layer (with coherence >= threshold)
    if (state.bestPerLayer) {
        const bestByLayer = {};
        for (const run of runs) {
            if (run.coherence < coherenceThreshold) continue;
            if (!bestByLayer[run.layer] || Math.abs(run.traitScore) > Math.abs(bestByLayer[run.layer].traitScore)) {
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
    const sortTh = (key, label) => renderSortableHeader({ key, label, sortKey: state.sortKey, sortDir: state.sortDir });
    container.innerHTML = `
        <div class="rb-filters">
            <span class="rb-filter-label">Layers:</span>
            <div class="rb-layer-chips">
                <button class="btn btn-xs rb-chip-btn" data-action="select-all">All</button>
                <button class="btn btn-xs rb-chip-btn" data-action="select-none">None</button>
                ${uniqueLayers.map(l => {
                    const on = state.layerFilter.size === 0 || state.layerFilter.has(l);
                    return `<label class="rb-chip ${on ? 'active' : ''}"><input type="checkbox" value="${l}" ${on ? 'checked' : ''}> L${l}</label>`;
                }).join('')}
            </div>
            ${(hasPositive && hasNegative) ? renderFilterDropdown('direction', 'Direction', state.steeringDirection, [['positive', 'Positive (+)'], ['negative', 'Negative (−)']], 'steeringDirection', trait) : ''}
            ${uniquePromptSets.length > 1 ? renderFilterDropdown('prompt-set', 'Prompt set', state.promptSetFilter, uniquePromptSets, 'promptSetFilter', trait) : ''}
            ${uniqueModelVariants.length > 1 ? renderFilterDropdown('model-variant', 'Model', state.modelVariantFilter, uniqueModelVariants, 'modelVariantFilter', trait) : ''}
            ${renderChipGroup({
                id: `rb-info-btns-${trait}`,
                mode: 'toggle-off',
                className: 'rb-info-btns',
                items: [
                    { value: 'definition', label: 'Definition' },
                    { value: 'judge', label: 'Judge Prompt' },
                    ...(baselineEntry ? [{ value: 'baseline', label: 'Baseline' }] : []),
                ],
                selected: state.infoPanel,
                onChange: async (newValue) => {
                    state.infoPanel = newValue;
                    await renderResponseBrowserForTrait(trait);
                    if (state.infoPanel) await loadInfoPanelContent(trait, state.infoPanel);
                },
            })}
            ${renderToggle({ label: `Best per layer (coh ≥${coherenceThreshold})`, checked: state.bestPerLayer, dataAttr: { key: 'action', value: 'best-per-layer' }, className: 'rb-toggle' })}
            ${renderToggle({ label: 'Compact responses', checked: state.compactResponses, dataAttr: { key: 'action', value: 'compact-responses' }, className: 'rb-toggle' })}
        </div>
        ${state.infoPanel ? `
        <div class="rb-info-panel" data-panel="${state.infoPanel}">
            <div class="rb-info-content">${renderLoading()}</div>
        </div>
        ` : ''}
        <div class="rb-table-wrapper">
            <table class="table table-compact data-table rb-table">
                <thead>
                    <tr>
                        ${sortTh('layer', 'Layer')}
                        ${sortTh('coef', 'Coef')}
                        <th>Method</th>
                        <th>Component</th>
                        ${showPositionCol ? '<th>Position</th>' : ''}
                        ${sortTh('traitScore', 'Trait')}
                        ${sortTh('coherence', 'Coh')}
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
                            <td>L${displayLayer(run.layer)}</td>
                            <td>${run.coef.toFixed(1)}</td>
                            <td>${run.method}</td>
                            <td>${run.component}</td>
                            ${showPositionCol ? `<td class="rb-position">${posDisplay}${promptSetDisplay}</td>` : ''}
                            <td class="${scoreClass(run.traitScore)}">${run.traitScore.toFixed(1)}</td>
                            <td class="${scoreClass(run.coherence, 'coherence')}">${run.coherence.toFixed(0)}</td>
                        </tr>
                        ${state.expandedRow === idx ? `
                        <tr class="rb-expanded-row">
                            <td colspan="${showPositionCol ? 7 : 6}">
                                <div class="rb-responses-container" id="rb-responses-${traitSlug(trait)}-${idx}">
                                    ${renderLoading('Loading responses...')}
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
    setupResponseBrowserHandlers(trait, container);

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
function setupResponseBrowserHandlers(trait, container) {
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
            await resetAndRender(trait);
        });
    });

    // Toggle checkboxes
    attachToggleListener(container, 'input[data-action="best-per-layer"]', 'bestPerLayer', trait, { resetsRow: true });
    attachToggleListener(container, '.rb-filters input[data-action="compact-responses"]', 'compactResponses', trait);

    // Filter dropdowns (styled selects wire themselves via onChange)
    wireStyledSelect(container);

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
            await resetAndRender(trait); // Close expanded row on filter change
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
            await resetAndRender(trait);
        });
    });

    // Row click to expand/collapse
    container.querySelectorAll('.rb-row').forEach(row => {
        row.addEventListener('click', async () => {
            const idx = parseInt(row.dataset.idx);
            state.expandedRow = state.expandedRow === idx ? null : idx;
            await renderResponseBrowserForTrait(trait);
        });
    });

    // Info panel chip-group (Definition / Judge Prompt / Baseline) — handler lives in onChange.
    wireChipGroup(container);
}

/**
 * Render a single response item row (prompt + response with scores).
 * Shared by baseline panel and expanded-row response list.
 */
function renderResponseItem(r, i, isCompact) {
    const responseText = isCompact
        ? r.response.replace(/\n/g, '\\n')
        : r.response;
    return `
        <div class="response-item-row">
            <div class="response-meta">
                <div class="meta-label">Prompt #${i + 1}</div>
                <div class="meta-score">Trait: <span class="${scoreClass(r.trait_score ?? 0)}">${r.trait_score?.toFixed(0) ?? '-'}</span></div>
                <div class="meta-score">Coh: <span class="${scoreClass(r.coherence_score ?? 0, 'coherence')}">${r.coherence_score?.toFixed(0) ?? '-'}</span></div>
            </div>
            <div class="response-content">
                <div class="response-q">${escapeHtml(typeof r.prompt === 'object' ? r.prompt.question || JSON.stringify(r.prompt) : r.prompt)}</div>
                <div class="response-a ${isCompact ? 'compact' : ''}">${escapeHtml(responseText)}</div>
            </div>
        </div>
    `;
}

/**
 * Load and display info panel content (definition or judge prompt)
 */
async function loadInfoPanelContent(trait, panelType) {
    const browserId = `response-browser-${traitSlug(trait)}`;
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

        const panelHandlers = {
            definition: () => {
                panel.innerHTML = `<pre class="rb-code">${escapeHtml(cached.text.trim())}</pre>`;
            },
            judge: async () => {
                if (!judgeTemplatesCache) {
                    const resp = await fetch('/api/judge-templates');
                    if (!resp.ok) {
                        panel.innerHTML = `<p class="no-data">Could not load judge templates</p>`;
                        return;
                    }
                    judgeTemplatesCache = await resp.json();
                }
                const highlightVars = (text) =>
                    escapeHtml(text).replace(/\{(\w+)\}/g, '<span class="rb-var">{$1}</span>');
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
            },
            baseline: async () => {
                const state = responseBrowserState[trait];
                const baselineEntry = state?.currentBaselineEntry;
                if (!baselineEntry) {
                    panel.innerHTML = `<p class="no-data">No baseline available for current filter selection</p>`;
                    return;
                }
                const response = await fetch(`/${steeringResponsePath(baselineEntry)}/baseline.json`);
                if (!response.ok) {
                    panel.innerHTML = `<p class="no-data">Baseline file not found</p>`;
                    return;
                }
                const responses = await response.json();
                const isCompact = state?.compactResponses ?? true;
                const baselineLabel = `${baselineEntry.model_variant} / ${baselineEntry.prompt_set}`;
                panel.innerHTML = `
                    <div class="rb-baseline-header hint">
                        Showing baseline for: <strong>${baselineLabel}</strong>
                    </div>
                    ${renderResponseList(responses, isCompact)}
                `;
            },
        };

        const handler = panelHandlers[panelType];
        if (handler) await handler();

    } catch (e) {
        console.error('Failed to load info panel:', e);
        panel.innerHTML = `<p class="no-data">Error: ${e.message}</p>`;
    }
}

/**
 * Load and display responses for a specific run
 */
async function loadResponsesForRun(trait, idx, run) {
    const containerId = `rb-responses-${traitSlug(trait)}-${idx}`;
    const container = document.getElementById(containerId);
    if (!container) return;

    const { entry } = run;

    try {
        const ts = run.timestamp ? run.timestamp.slice(0, 19).replace(/:/g, '-').replace('T', '_') : '';
        const filename = `L${run.layer}_c${run.coef.toFixed(1)}_${ts}.json`;
        const url = `/${steeringResponsePath(entry)}/${run.component}/${run.method}/${filename}`;
        const response = await fetch(url);

        if (!response.ok) {
            container.innerHTML = `<p class="no-data">Response file not found</p>`;
            return;
        }

        const responses = await response.json();
        const state = responseBrowserState[trait];
        const isCompact = state?.compactResponses ?? true;

        container.innerHTML = renderResponseList(responses, isCompact);

    } catch (e) {
        console.error('Failed to load responses:', e);
        container.innerHTML = `<p class="no-data">Error loading responses: ${e.message}</p>`;
    }
}

// ES module exports
export {
    setTraitResultsCache,
    renderResponseBrowserForTrait,
    fetchAvailableResponses,
    responseBrowserState,
};
