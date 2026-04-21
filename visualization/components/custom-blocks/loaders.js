/**
 * Custom block loaders — async content fetch + DOM event wiring
 *
 * Input:  rendered DOM (from renderers.js)
 * Output: hydrated DOM with click handlers and lazy-loaded content
 * Usage:  import { toggleDropdown, loadExpandedDropdowns, loadCharts } from './loaders.js';
 */

import { escapeHtml, fetchJSON } from '../../core/utils.js';
import { renderLoading } from '../../core/ui.js';
import { renderResponsesTable, renderDatasetList, renderExtractionTable } from './renderers.js';

// === Shared expand/collapse helper ===

/** Toggle expand/collapse: manages class, display, arrow char, and optional onExpand callback */
function toggleExpandCollapse(container, bodySelector, toggleSelector, onExpand) {
    const body = container.querySelector(bodySelector);
    const toggle = container.querySelector(toggleSelector);
    if (container.classList.contains('expanded')) {
        container.classList.remove('expanded');
        body.style.display = 'none';
        toggle.textContent = '\u25B6';
    } else {
        container.classList.add('expanded');
        body.style.display = 'block';
        toggle.textContent = '\u25BC';
        if (onExpand) onExpand();
    }
}

// === Dropdown content loading (responses / dataset) ===

/** Fetch and render content for a dropdown (used by both toggle and auto-expand) */
async function fetchDropdownContent(dropdown) {
    const content = dropdown.querySelector('.dropdown-body');
    const type = dropdown.dataset.type;
    const path = dropdown.dataset.path;
    const noScores = dropdown.dataset.noScores === 'true';
    const limit = dropdown.dataset.limit ? parseInt(dropdown.dataset.limit) : null;
    const height = dropdown.dataset.height ? parseInt(dropdown.dataset.height) : null;

    content.innerHTML = renderLoading();
    try {
        const response = await fetch(path);
        if (!response.ok) throw new Error('Failed to load');

        if (type === 'Responses') {
            const data = await response.json();

            // Try loading annotations (_annotations.json with text spans)
            let charRanges = [];
            const responseList = Array.isArray(data) ? data : [data];

            try {
                const annotationsPath = path.replace('.json', '_annotations.json');
                const annotationsResp = await fetch(annotationsPath);
                if (annotationsResp.ok) {
                    const annotations = await annotationsResp.json();
                    if (annotations.annotations) {
                        for (let i = 0; i < responseList.length; i++) {
                            const spans = window.annotations.getSpansForResponse(annotations, i);
                            const ranges = window.annotations.spansToCharRanges(
                                responseList[i].response || '',
                                spans
                            );
                            charRanges.push(ranges);
                        }
                    }
                }
            } catch (e) {
                // Annotations not available
            }

            content.innerHTML = renderResponsesTable(data, { showScores: !noScores, charRanges });
            // Apply custom height with resizable wrapper
            if (height) {
                const inner = content.querySelector('.responses-table');
                if (inner) {
                    const wrapper = document.createElement('div');
                    wrapper.className = 'responses-scroll-wrapper';
                    wrapper.style.maxHeight = `${height}px`;
                    inner.parentNode.insertBefore(wrapper, inner);
                    wrapper.appendChild(inner);
                }
            }
        } else if (type === 'Dataset') {
            const text = await response.text();
            content.innerHTML = renderDatasetList(text, { limit });
            if (height) {
                const list = content.querySelector('.dataset-list');
                if (list) list.style.maxHeight = `${height}px`;
            }
        }
    } catch (error) {
        content.innerHTML = `<div class="error">Failed to load ${type.toLowerCase()}</div>`;
    }
}

/** Toggle a dropdown open/closed, loading content on first open */
async function toggleDropdown(dropdownId) {
    const dropdown = document.getElementById(dropdownId);
    if (!dropdown) return;

    toggleExpandCollapse(dropdown, '.dropdown-body', '.dropdown-toggle', async () => {
        const content = dropdown.querySelector('.dropdown-body');
        if (!content.innerHTML) {
            await fetchDropdownContent(dropdown);
        }
    });
}

/** Auto-load content for dropdowns that start expanded */
async function loadExpandedDropdowns() {
    const expandedDropdowns = document.querySelectorAll('.responses-dropdown[data-auto-expand="true"]');
    for (const dropdown of expandedDropdowns) {
        const content = dropdown.querySelector('.dropdown-body');
        const toggle = dropdown.querySelector('.dropdown-toggle');

        if (!content.innerHTML) {
            await fetchDropdownContent(dropdown);
        }

        // Set visual state to expanded
        dropdown.classList.add('expanded');
        content.style.display = 'block';
        toggle.textContent = '\u25BC';
    }

    // Also initialize tabbed components
    initExtractionData();
    initSteeredResponses();
}

// === Tabbed widget — shared init logic ===

/** Generic initializer for tabbed widgets: wires up tab clicks and loads active tab */
function initTabbedWidget(containerSelector, tabSelector, loadFn) {
    for (const container of document.querySelectorAll(containerSelector)) {
        if (container.dataset.initialized) continue;
        container.dataset.initialized = 'true';
        container.querySelectorAll(tabSelector).forEach(tab => {
            tab.addEventListener('click', () => {
                container.querySelectorAll(tabSelector).forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                loadFn(container, tab);
            });
        });
        const activeTab = container.querySelector(`${tabSelector}.active`);
        if (activeTab) loadFn(container, activeTab);
    }
}

// === Steered responses loader ===

/** Initialize steered-responses: set up click handlers and load first tab */
function initSteeredResponses() {
    initTabbedWidget('.sr-container', '.sr-tab', (container, tab) => {
        container.dataset.active = tab.dataset.trait;
        loadSteeredResponseContent(container, tab.dataset.pvPath, tab.dataset.naturalPath);
    });
}

/** Load and render 3-column comparison table for steered-responses */
async function loadSteeredResponseContent(container, pvPath, naturalPath) {
    const content = container.querySelector('.sr-content');
    content.innerHTML = '<div class="sr-loading">Loading...</div>';

    try {
        const [pvRes, naturalRes] = await Promise.all([
            fetch(pvPath),
            fetch(naturalPath)
        ]);

        if (!pvRes.ok || !naturalRes.ok) throw new Error('Failed to load responses');

        const [pvData, naturalData] = await Promise.all([
            pvRes.json(),
            naturalRes.json()
        ]);

        // Build 3-column table: Question | PV Response | Natural Response
        const rows = pvData.map((pv, i) => {
            const natural = naturalData[i] || {};
            return `
                <tr>
                    <td class="sr-question">${escapeHtml(pv.prompt || '')}</td>
                    <td class="sr-response sr-pv">${escapeHtml(pv.response || '')}</td>
                    <td class="sr-response sr-natural">${escapeHtml(natural.response || '')}</td>
                </tr>
            `;
        }).join('');

        content.innerHTML = `
            <table class="sr-table">
                <thead>
                    <tr>
                        <th class="sr-th-question">Question</th>
                        <th class="sr-th-response">PV Instruction</th>
                        <th class="sr-th-response">Natural</th>
                    </tr>
                </thead>
                <tbody>${rows}</tbody>
            </table>
        `;
    } catch (error) {
        content.innerHTML = `<p class="no-data">Failed to load: ${error.message}</p>`;
    }
}

// === Extraction-data loader ===

/** Toggle extraction-data component expand/collapse */
function toggleExtractionData(id) {
    const container = document.getElementById(id);
    if (!container) return;

    toggleExpandCollapse(container, '.ed-body', '.ed-toggle', () => {
        if (!container.dataset.loaded) {
            loadExtractionData(container, container.dataset.defaultPath);
            container.dataset.loaded = 'true';
        }
    });
}

/** Initialize extraction-data: set up tab handlers, load if expanded */
function initExtractionData() {
    initTabbedWidget('.extraction-data-container', '.ed-tab', (container, tab) => {
        container.dataset.active = tab.dataset.trait;
        loadExtractionData(container, tab.dataset.path);
    });

    // Also auto-load expanded containers that haven't loaded yet
    for (const container of document.querySelectorAll('.extraction-data-container.expanded')) {
        if (!container.dataset.loaded) {
            loadExtractionData(container, container.dataset.defaultPath);
            container.dataset.loaded = 'true';
        }
    }
}

/** Parse extraction path to get experiment and variant */
function parseExtractionPath(path) {
    const match = path.match(/experiments\/([^/]+)\/extraction\/[^/]+\/[^/]+\/([^/]+)\/responses/);
    if (!match) return null;
    return { experiment: match[1], variant: match[2] };
}

/** Load pos.json and neg.json from a folder path and render both */
async function loadExtractionData(container, basePath) {
    const posScroll = container.querySelector('.ed-positive .ed-scroll');
    const negScroll = container.querySelector('.ed-negative .ed-scroll');
    const highlightTokens = parseInt(container.dataset.highlightTokens) || null;

    posScroll.innerHTML = renderLoading();
    negScroll.innerHTML = renderLoading();

    try {
        // Parse path to get experiment/variant, fetch config to resolve model name
        const pathInfo = parseExtractionPath(basePath);
        let modelName = null;
        if (pathInfo) {
            try {
                const configRes = await fetch(`/experiments/${pathInfo.experiment}/config.json`);
                if (configRes.ok) {
                    const config = await configRes.json();
                    modelName = config.model_variants?.[pathInfo.variant]?.model;
                }
            } catch (e) {
                // Config fetch failed, continue without model name
            }
        }

        // Update header with model name if found
        if (modelName) {
            const label = container.querySelector('.ed-label');
            if (label && !label.dataset.modelAdded) {
                label.dataset.modelAdded = 'true';
                const modelSpan = document.createElement('span');
                modelSpan.className = 'ed-model';
                modelSpan.textContent = modelName;
                label.parentNode.insertBefore(modelSpan, label.nextSibling);
            }
        }

        // Fetch responses + metadata. metadata.has_token_offsets gates the
        // optional token_offsets.json request — avoids a 404 console spam
        // for response dirs that never had offsets computed (most of them).
        const [posRes, negRes, metadataRes] = await Promise.all([
            fetch(`${basePath}/pos.json`),
            fetch(`${basePath}/neg.json`),
            fetch(`${basePath}/metadata.json`).catch(() => null)
        ]);

        if (!posRes.ok || !negRes.ok) throw new Error('Failed to load');

        const [posData, negData] = await Promise.all([posRes.json(), negRes.json()]);

        // Parse token offsets only if metadata advertises them. Writer:
        // visualization/other/compute_token_offsets.py sets has_token_offsets=true.
        let tokenOffsets = null;
        if (highlightTokens && metadataRes?.ok) {
            const metadata = await metadataRes.json();
            if (metadata?.has_token_offsets) {
                const offsetsRes = await fetch(`${basePath}/token_offsets.json`);
                if (offsetsRes.ok) tokenOffsets = await offsetsRes.json();
            }
        }

        posScroll.innerHTML = renderExtractionTable(posData, {
            tokenOffsets: tokenOffsets?.pos,
            highlightTokens
        });
        negScroll.innerHTML = renderExtractionTable(negData, {
            tokenOffsets: tokenOffsets?.neg,
            highlightTokens
        });
    } catch (error) {
        posScroll.innerHTML = `<p class="no-data">Failed to load: ${error.message}</p>`;
        negScroll.innerHTML = '';
    }
}

// === Chart loading ===

/** Load and render all chart blocks that haven't been loaded yet */
async function loadCharts() {
    const chartFigures = document.querySelectorAll('.chart-figure:not([data-loaded])');

    for (const figure of chartFigures) {
        figure.dataset.loaded = 'true';
        const container = figure.querySelector('.chart-container');
        const { chartType, chartPath, chartBars, chartTraits, chartLabels, chartColors, chartHeight, chartPerplexity, chartProjections } = figure.dataset;

        try {
            // For annotation-stacked charts, bars contains the data paths directly
            if (chartBars) {
                const bars = JSON.parse(chartBars);
                container.innerHTML = '';
                await window.chartTypes.render(chartType, container, bars, {
                    height: chartHeight ? parseInt(chartHeight) : null,
                    colors: chartColors || null
                });
            } else {
                const data = await fetchJSON(chartPath);
                if (!data) throw new Error('Failed to load');

                const options = {
                    traits: chartTraits ? chartTraits.split(',') : null,
                    height: chartHeight ? parseInt(chartHeight) : null
                };

                if (chartLabels) {
                    try { options.labels = JSON.parse(chartLabels); } catch (e) { /* ignore */ }
                }

                if (chartPerplexity) options.perplexityPath = chartPerplexity;
                if (chartProjections) {
                    try {
                        options.projections = JSON.parse(chartProjections);
                    } catch (e) { /* ignore parse errors */ }
                }

                container.innerHTML = '';
                await window.chartTypes.render(chartType, container, data, options);
            }
        } catch (e) {
            container.innerHTML = `<div class="chart-error">Failed to load: ${e.message}</div>`;
        }
    }

    // Also load standalone chart containers (e.g., inside :::side-by-side::: blocks)
    const standaloneCharts = document.querySelectorAll('.chart-container[data-chart-type]:not([data-loaded])');
    for (const container of standaloneCharts) {
        container.dataset.loaded = 'true';
        const { chartType, chartPath, chartStyle } = container.dataset;
        if (!chartType || !chartPath) continue;
        try {
            const data = await fetchJSON(chartPath);
            if (!data) throw new Error('Failed to load');
            container.innerHTML = '';
            await window.chartTypes.render(chartType, container, data, {
                style: chartStyle || null,
            });
        } catch (e) {
            container.innerHTML = `<div class="chart-error">Failed to load: ${e.message}</div>`;
        }
    }
}

export {
    toggleExpandCollapse,
    toggleDropdown,
    loadExpandedDropdowns,
    toggleExtractionData,
    initExtractionData,
    loadCharts,
};
