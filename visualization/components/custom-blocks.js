/**
 * Custom Blocks - Parsing and rendering for ::: syntax in markdown
 *
 * Supported blocks:
 *   :::responses path "label" [flags]:::                       - Expandable response table
 *   :::dataset path "label" [flags]:::                         - Expandable dataset list
 *   :::figure path "caption" size:::                           - Image with caption (size: small|medium|large)
 *   :::example ... :::                                         - Example box with optional caption
 *   :::chart type path "caption" [traits=...] [height=N]:::    - Dynamic Plotly chart from JSON
 *   :::extraction-data "label" [expanded] [tokens=N]\n trait: path\n :::  - Tabbed pos/neg extraction viewer
 *   :::annotation-stacked "caption" [height=N]\n label: path\n :::   - Stacked bar chart from annotation files
 *
 * Flags:
 *   expanded  - Start expanded instead of collapsed
 *   no-scores - Hide trait/coherence score columns (responses only)
 *   limit=N   - Max items to show (dataset only)
 *   height=N  - Custom max height in px (responses, dataset, chart)
 *   green|red|blue|orange|purple - Colored left border
 *   traits=a,b,c - Filter to specific traits (chart only)
 *
 * Extraction-data syntax:
 *   :::extraction-data "Extraction data" expanded tokens=5
 *   traitName: experiments/.../responses   (folder with pos.json + neg.json + optional token_offsets.json)
 *   anotherTrait: experiments/.../responses
 *   :::
 */

import { escapeHtml } from '../core/utils.js';

// ============================================================================
// Block Extraction - Parse markdown and extract custom blocks
// ============================================================================

/**
 * Extract all custom blocks from markdown, replacing with placeholders
 * @param {string} markdown - Raw markdown content
 * @returns {Object} - { markdown, blocks } where blocks contains all extracted block data
 */
function extractCustomBlocks(markdown) {
    const blocks = {
        responses: [],
        datasets: [],
        figures: [],
        examples: [],
        steeredResponses: [],
        charts: [],
        extractionData: [],
        annotationStacked: []
    };

    // :::responses path "label" [expanded] [no-scores] [height=N] [color]:::
    markdown = markdown.replace(
        /:::responses\s+([^\s:]+)(?:\s+"([^"]*)")?([^:]*):::/g,
        (match, path, label, flags) => {
            blocks.responses.push({
                path,
                label: label || 'View responses',
                expanded: /\bexpanded\b/.test(flags),
                noScores: /\bno-scores\b/.test(flags),
                height: parseInt(flags.match(/\bheight=(\d+)/)?.[1]) || null,
                color: flags.match(/\b(green|red|blue|orange|purple)\b/)?.[1] || null
            });
            return `RESPONSE_BLOCK_${blocks.responses.length - 1}`;
        }
    );

    // :::dataset path "label" [expanded] [limit=N] [height=N] [color]:::
    markdown = markdown.replace(
        /:::dataset\s+([^\s:]+)(?:\s+"([^"]*)")?([^:]*):::/g,
        (match, path, label, flags) => {
            blocks.datasets.push({
                path,
                label: label || 'View examples',
                expanded: /\bexpanded\b/.test(flags),
                limit: parseInt(flags.match(/\blimit=(\d+)/)?.[1]) || null,
                height: parseInt(flags.match(/\bheight=(\d+)/)?.[1]) || null,
                color: flags.match(/\b(green|red|blue|orange|purple)\b/)?.[1] || null
            });
            return `DATASET_BLOCK_${blocks.datasets.length - 1}`;
        }
    );

    // :::figure path "caption" size:::
    markdown = markdown.replace(
        /:::figure\s+([^\s:]+)(?:\s+"([^"]*)")?(?:\s+(small|medium|large))?\s*:::/g,
        (match, path, caption, size) => {
            blocks.figures.push({ path, caption: caption || '', size: size || '' });
            return `FIGURE_BLOCK_${blocks.figures.length - 1}`;
        }
    );

    // :::example ... ::: with caption (*caption text*)
    markdown = markdown.replace(
        /:::example\s*\n([\s\S]*?)\n:::\s*\n?\*([^*]+)\*/g,
        (match, content, caption) => {
            blocks.examples.push({ content: content.trim(), caption: caption.trim() });
            return `EXAMPLE_BLOCK_${blocks.examples.length - 1}`;
        }
    );

    // :::example ... ::: without caption
    markdown = markdown.replace(
        /:::example\s*\n([\s\S]*?)\n:::/g,
        (match, content) => {
            blocks.examples.push({ content: content.trim(), caption: '' });
            return `EXAMPLE_BLOCK_${blocks.examples.length - 1}`;
        }
    );

    // :::steered-responses "Label"\n trait: "TraitLabel" | pvPath | naturalPath \n:::
    // Renders a 3-column table: Question | PV Response | Natural Response
    markdown = markdown.replace(
        /:::steered-responses\s+"([^"]+)"\s*\n([\s\S]*?)\n:::/g,
        (match, label, body) => {
            const traits = [];
            for (const line of body.trim().split('\n')) {
                // Parse: traitKey: "Label" | pvPath | naturalPath
                const traitMatch = line.match(/^\s*(\w+):\s*"([^"]+)"\s*\|\s*([^\s|]+)\s*\|\s*([^\s|]+)/);
                if (traitMatch) {
                    traits.push({
                        key: traitMatch[1],
                        label: traitMatch[2],
                        pvPath: traitMatch[3].trim(),
                        naturalPath: traitMatch[4].trim()
                    });
                }
            }
            blocks.steeredResponses.push({ label, traits });
            return `STEERED_RESPONSES_BLOCK_${blocks.steeredResponses.length - 1}`;
        }
    );

    // :::chart type path "caption" [traits=...] [height=N] [perplexity=path] [projections=path,path]:::
    markdown = markdown.replace(
        /:::chart\s+(\S+)\s+(\S+)(?:\s+"([^"]*)")?([^:]*):::/g,
        (match, type, path, caption, flags) => {
            // Parse projection paths (comma-separated trait:path pairs)
            let projections = null;
            const projMatch = flags.match(/\bprojections=([^\s]+)/)?.[1];
            if (projMatch) {
                projections = {};
                for (const pair of projMatch.split(',')) {
                    const [trait, projPath] = pair.split(':');
                    if (trait && projPath) projections[trait] = projPath;
                }
            }

            blocks.charts.push({
                type,
                path,
                caption: caption || '',
                traits: flags.match(/\btraits=([^\s]+)/)?.[1]?.split(',') || null,
                height: parseInt(flags.match(/\bheight=(\d+)/)?.[1]) || null,
                perplexity: flags.match(/\bperplexity=([^\s]+)/)?.[1] || null,
                projections
            });
            return `CHART_BLOCK_${blocks.charts.length - 1}`;
        }
    );

    // :::extraction-data "label" [expanded] [tokens=N]\n trait: path\n trait: path\n:::
    markdown = markdown.replace(
        /:::extraction-data\s+"([^"]+)"([^\n]*)\n([\s\S]*?)\n:::/g,
        (match, label, flags, body) => {
            const traits = [];
            for (const line of body.trim().split('\n')) {
                const traitMatch = line.match(/^\s*(\w+):\s*(.+)$/);
                if (traitMatch) {
                    traits.push({
                        name: traitMatch[1],
                        path: traitMatch[2].trim()
                    });
                }
            }
            blocks.extractionData.push({
                label,
                expanded: /\bexpanded\b/.test(flags),
                highlightTokens: parseInt(flags.match(/\btokens=(\d+)/)?.[1]) || null,
                traits
            });
            return `EXTRACTION_DATA_BLOCK_${blocks.extractionData.length - 1}`;
        }
    );

    // :::annotation-stacked "caption" [height=N]\n label: path\n label: path\n:::
    markdown = markdown.replace(
        /:::annotation-stacked\s+"([^"]+)"([^\n]*)\n([\s\S]*?)\n:::/g,
        (match, caption, flags, body) => {
            const bars = [];
            for (const line of body.trim().split('\n')) {
                // Parse: Label: path/to/file.json
                const barMatch = line.match(/^\s*([^:]+):\s*(.+)$/);
                if (barMatch) {
                    bars.push({
                        label: barMatch[1].trim(),
                        path: barMatch[2].trim()
                    });
                }
            }
            blocks.annotationStacked.push({
                caption,
                height: parseInt(flags.match(/\bheight=(\d+)/)?.[1]) || null,
                bars
            });
            return `ANNOTATION_STACKED_BLOCK_${blocks.annotationStacked.length - 1}`;
        }
    );

    return { markdown, blocks };
}

// ============================================================================
// Block Rendering - Generate HTML from extracted blocks
// ============================================================================

/**
 * Replace a placeholder in HTML, handling both <p>-wrapped and bare forms.
 */
function insertBlock(html, placeholder, rendered) {
    return html.replace(`<p>${placeholder}</p>`, rendered).replace(placeholder, rendered);
}

/**
 * Replace block placeholders in HTML with rendered components
 * @param {string} html - HTML with placeholders
 * @param {Object} blocks - Extracted block data
 * @param {string} namespace - Unique namespace for IDs (e.g., filename)
 * @param {Object} options - Rendering options
 * @param {string} options.assetBaseUrl - Base URL for resolving assets/ paths in figures (default: '/docs/viz_findings/')
 * @returns {string} - HTML with blocks rendered
 */
function renderCustomBlocks(html, blocks, namespace = 'block', options = {}) {
    const { assetBaseUrl = '/docs/viz_findings/' } = options;

    // Responses blocks -> expandable dropdowns
    blocks.responses.forEach((block, i) => {
        const dropdownId = `responses-${namespace}-${i}`;
        const dropdownHtml = createDropdownHtml(dropdownId, block.label, 'Responses', block.path, {
            expanded: block.expanded,
            noScores: block.noScores,
            height: block.height,
            color: block.color
        });
        html = insertBlock(html, `RESPONSE_BLOCK_${i}`, dropdownHtml);
    });

    // Dataset blocks -> expandable dropdowns
    blocks.datasets.forEach((block, i) => {
        const dropdownId = `dataset-${namespace}-${i}`;
        const dropdownHtml = createDropdownHtml(dropdownId, block.label, 'Dataset', block.path, {
            expanded: block.expanded,
            limit: block.limit,
            height: block.height,
            color: block.color
        });
        html = insertBlock(html, `DATASET_BLOCK_${i}`, dropdownHtml);
    });

    // Figure blocks -> img with caption
    blocks.figures.forEach((block, i) => {
        const imgPath = block.path.startsWith('assets/')
            ? `${assetBaseUrl}${block.path}`
            : block.path;
        const sizeClass = block.size ? ` fig-${block.size}` : '';
        const figNum = i + 1;
        const captionText = block.caption ? `Figure ${figNum}: ${block.caption}` : '';
        const figureHtml = `
            <figure class="fig${sizeClass}">
                <img src="${imgPath}" alt="${block.caption}">
                ${captionText ? `<figcaption>${captionText}</figcaption>` : ''}
            </figure>
        `;
        html = insertBlock(html, `FIGURE_BLOCK_${i}`, figureHtml);
    });

    // Example blocks -> styled boxes
    blocks.examples.forEach((block, i) => {
        const innerHtml = marked.parse(block.content);
        const exampleHtml = `
            <figure class="example-box">
                <div class="example-content">${innerHtml}</div>
                ${block.caption ? `<figcaption>${block.caption}</figcaption>` : ''}
            </figure>
        `;
        html = insertBlock(html, `EXAMPLE_BLOCK_${i}`, exampleHtml);
    });

    // Steered-responses blocks -> 3-column comparison table
    blocks.steeredResponses.forEach((block, i) => {
        const srId = `steered-responses-${namespace}-${i}`;
        const srHtml = createSteeredResponsesHtml(srId, block);
        html = insertBlock(html, `STEERED_RESPONSES_BLOCK_${i}`, srHtml);
    });

    // Chart blocks -> figure with chart container (loaded async via loadCharts)
    blocks.charts.forEach((block, i) => {
        const chartId = `chart-${namespace}-${i}`;
        // Serialize projections as JSON for data attribute
        const projectionsAttr = block.projections ? JSON.stringify(block.projections) : '';
        const chartHtml = `
            <figure class="chart-figure" id="${chartId}"
                    data-chart-type="${block.type}"
                    data-chart-path="${block.path}"
                    data-chart-traits="${block.traits?.join(',') || ''}"
                    data-chart-height="${block.height || ''}"
                    data-chart-perplexity="${block.perplexity || ''}"
                    data-chart-projections='${projectionsAttr}'>
                <div class="chart-container">
                    <div class="chart-loading">Loading chart...</div>
                </div>
                ${block.caption ? `<figcaption>${block.caption}</figcaption>` : ''}
            </figure>
        `;
        html = insertBlock(html, `CHART_BLOCK_${i}`, chartHtml);
    });

    // Extraction-data blocks -> tabbed pos/neg viewer
    blocks.extractionData.forEach((block, i) => {
        const edId = `extraction-data-${namespace}-${i}`;
        const edHtml = createExtractionDataHtml(edId, block);
        html = insertBlock(html, `EXTRACTION_DATA_BLOCK_${i}`, edHtml);
    });

    // Annotation-stacked blocks -> chart figure (loaded async via loadCharts)
    blocks.annotationStacked.forEach((block, i) => {
        const chartId = `annotation-stacked-${namespace}-${i}`;
        const barsJson = JSON.stringify(block.bars);
        const chartHtml = `
            <figure class="chart-figure" id="${chartId}"
                    data-chart-type="annotation-stacked"
                    data-chart-bars='${barsJson}'
                    data-chart-height="${block.height || ''}">
                <div class="chart-container">
                    <div class="chart-loading">Loading chart...</div>
                </div>
                ${block.caption ? `<figcaption>${block.caption}</figcaption>` : ''}
            </figure>
        `;
        html = insertBlock(html, `ANNOTATION_STACKED_BLOCK_${i}`, chartHtml);
    });

    return html;
}

/**
 * Parse steering response path to extract metadata
 * Path pattern: .../responses/{component}/{method}/L{layer}_c{coef}_{timestamp}.json
 * Also extracts position from parent dirs: .../{position}/...
 */
function parseSteeringResponsePath(path) {
    const parts = path.split('/');
    const filename = parts[parts.length - 1];

    // Parse filename: L20_c6.0_2026-01-11_09-08-38.json
    const filenameMatch = filename.match(/^L(\d+)_c(-?[\d.]+)_/);
    if (!filenameMatch) return null;

    const layer = parseInt(filenameMatch[1]);
    const coef = parseFloat(filenameMatch[2]);

    // Get method and component from path: .../responses/{component}/{method}/filename
    const method = parts[parts.length - 2];
    const component = parts[parts.length - 3];

    // Find position (sanitized) - look for response__ or prompt__ pattern
    const positionPart = parts.find(p => p.startsWith('response_') || p.startsWith('prompt_') || p.startsWith('all_'));
    const position = positionPart
        ? (window.paths?.desanitizePosition?.(positionPart) || positionPart)
        : null;

    return { layer, coef, method, component, position };
}

/**
 * Create HTML for an expandable dropdown
 * @param {Object} options - Display options
 * @param {boolean} options.expanded - Start expanded
 * @param {boolean} options.noScores - Hide scores (for responses)
 * @param {number} options.limit - Max items to show (for datasets)
 * @param {number} options.height - Custom max height in px (for datasets)
 */
function createDropdownHtml(id, label, type, path, options = {}) {
    const { expanded = false, noScores = false, limit = null, height = null, color = null } = options;
    const expandedClass = expanded ? ' expanded' : '';
    const colorClass = color ? ` dropdown-${color}` : '';
    const toggleChar = expanded ? '▼' : '▶';
    const contentStyle = expanded ? '' : 'display: none;';
    const limitAttr = limit ? ` data-limit="${limit}"` : '';
    const heightAttr = height ? ` data-height="${height}"` : '';

    // For steered responses, show metadata subtitle
    let metadataHtml = '';
    if (type === 'Responses') {
        const meta = parseSteeringResponsePath(path);
        if (meta) {
            metadataHtml = `<span class="dropdown-meta">L${meta.layer} · coef ${meta.coef} · ${meta.component} · ${meta.method}</span>`;
        }
    }

    return `
        <div class="dropdown responses-dropdown${expandedClass}${colorClass}" id="${id}" data-type="${type}" data-path="${path}" data-no-scores="${noScores}" data-auto-expand="${expanded}"${limitAttr}${heightAttr}>
            <div class="dropdown-header responses-header" onclick="window.customBlocks.toggleDropdown('${id}')">
                <span class="dropdown-toggle responses-toggle">${toggleChar}</span>
                <span class="dropdown-label responses-label">${label}</span>
                ${metadataHtml}
            </div>
            <div class="dropdown-body responses-content" style="${contentStyle}"></div>
        </div>
    `;
}

/**
 * Create HTML for steered-responses component (3-column comparison table)
 * Shows Question | PV Response | Natural Response side-by-side
 * @param {string} id - Unique ID for this component
 * @param {Object} block - { label, traits: [{key, label, pvPath, naturalPath}] }
 */
function createSteeredResponsesHtml(id, block) {
    const { label, traits } = block;
    const defaultTrait = traits[0]?.key || '';

    const tabsHtml = traits.map((t, i) => {
        const isActive = i === 0;
        return `<button class="sr-tab${isActive ? ' active' : ''}" data-trait="${t.key}" data-pv-path="${t.pvPath}" data-natural-path="${t.naturalPath}">${t.label}</button>`;
    }).join('');

    return `
        <div class="sr-container" id="${id}" data-active="${defaultTrait}">
            <div class="sr-header">
                <span class="sr-label">${label}</span>
                <div class="sr-tabs">${tabsHtml}</div>
            </div>
            <div class="sr-content">
                <div class="sr-loading">Loading...</div>
            </div>
        </div>
    `;
}

/**
 * Create HTML for extraction-data component (tabbed pos/neg viewer)
 * @param {string} id - Unique ID for this component
 * @param {Object} block - { label, expanded, highlightTokens, traits: [{name, path}] }
 */
function createExtractionDataHtml(id, block) {
    const { label, expanded, highlightTokens, traits } = block;
    const defaultTrait = traits[0]?.name || '';
    const defaultPath = traits[0]?.path || '';
    const expandedClass = expanded ? ' expanded' : '';
    const toggleChar = expanded ? '▼' : '▶';
    const bodyStyle = expanded ? '' : 'display: none;';
    const tokensAttr = highlightTokens ? ` data-highlight-tokens="${highlightTokens}"` : '';

    const tabsHtml = traits.map((t, i) => {
        const isActive = i === 0;
        return `<button class="ed-tab${isActive ? ' active' : ''}" data-trait="${t.name}" data-path="${t.path}">${t.name}</button>`;
    }).join('');

    return `
        <div class="extraction-data-container${expandedClass}" id="${id}" data-active="${defaultTrait}" data-default-path="${defaultPath}"${tokensAttr}>
            <div class="ed-header" onclick="window.customBlocks.toggleExtractionData('${id}')">
                <span class="ed-toggle">${toggleChar}</span>
                <span class="ed-label">${label}</span>
            </div>
            <div class="ed-body" style="${bodyStyle}">
                <div class="ed-tabs">${tabsHtml}</div>
                <div class="ed-content">
                    <div class="ed-section ed-positive">
                        <div class="ed-section-label">Positive examples</div>
                        <div class="ed-scroll"></div>
                    </div>
                    <div class="ed-section ed-negative">
                        <div class="ed-section-label">Negative examples</div>
                        <div class="ed-scroll"></div>
                    </div>
                </div>
            </div>
        </div>
    `;
}

// ============================================================================
// Toggle Handlers - Expand/collapse dropdowns and load content
// ============================================================================

/**
 * Fetch and render content for a dropdown element
 * Core primitive used by both toggle and auto-expand
 */
async function fetchDropdownContent(dropdown) {
    const content = dropdown.querySelector('.dropdown-body');
    const type = dropdown.dataset.type;
    const path = dropdown.dataset.path;
    const noScores = dropdown.dataset.noScores === 'true';
    const limit = dropdown.dataset.limit ? parseInt(dropdown.dataset.limit) : null;
    const height = dropdown.dataset.height ? parseInt(dropdown.dataset.height) : null;

    content.innerHTML = ui.renderLoading();
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
                        // Convert text spans to char ranges at runtime
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
            // Apply custom height if specified
            if (height) {
                const list = content.querySelector('.dataset-list');
                if (list) list.style.maxHeight = `${height}px`;
            }
        }
    } catch (error) {
        content.innerHTML = `<div class="error">Failed to load ${type.toLowerCase()}</div>`;
    }
}

/**
 * Toggle a dropdown open/closed, loading content on first open
 */
async function toggleDropdown(dropdownId) {
    const dropdown = document.getElementById(dropdownId);
    if (!dropdown) return;

    const content = dropdown.querySelector('.dropdown-body');
    const toggle = dropdown.querySelector('.dropdown-toggle');

    if (dropdown.classList.contains('expanded')) {
        dropdown.classList.remove('expanded');
        content.style.display = 'none';
        toggle.textContent = '▶';
    } else {
        if (!content.innerHTML) {
            await fetchDropdownContent(dropdown);
        }
        dropdown.classList.add('expanded');
        content.style.display = 'block';
        toggle.textContent = '▼';
    }
}

/**
 * Auto-load content for dropdowns that start expanded
 * Call this after rendering content that may contain expanded dropdowns
 */
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
        toggle.textContent = '▼';
    }

    // Also initialize tabbed components
    initExtractionData();
    initSteeredResponses();
}

// ============================================================================
// Tabbed Widget - Shared init logic for tabbed components
// ============================================================================

/**
 * Generic initializer for tabbed widgets.
 * Finds containers, wires up tab click handlers, and loads the active tab.
 * @param {string} containerSelector - CSS selector for widget containers
 * @param {string} tabSelector - CSS selector for tab buttons within a container
 * @param {Function} loadFn - Called as loadFn(container, tab) to load content for a tab
 */
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

/**
 * Initialize steered-responses: set up click handlers and load first tab
 */
function initSteeredResponses() {
    initTabbedWidget('.sr-container', '.sr-tab', (container, tab) => {
        container.dataset.active = tab.dataset.trait;
        loadSteeredResponseContent(container, tab.dataset.pvPath, tab.dataset.naturalPath);
    });
}

/**
 * Load and render 3-column comparison table for steered-responses
 * Merges PV and Natural responses by question into a single table
 */
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
        // Match by question text (assuming same order/questions)
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

/**
 * Toggle extraction-data component expand/collapse
 */
function toggleExtractionData(id) {
    const container = document.getElementById(id);
    if (!container) return;

    const body = container.querySelector('.ed-body');
    const toggle = container.querySelector('.ed-toggle');

    if (container.classList.contains('expanded')) {
        container.classList.remove('expanded');
        body.style.display = 'none';
        toggle.textContent = '▶';
    } else {
        container.classList.add('expanded');
        body.style.display = 'block';
        toggle.textContent = '▼';
        // Load content if not already loaded
        if (!container.dataset.loaded) {
            loadExtractionData(container, container.dataset.defaultPath);
            container.dataset.loaded = 'true';
        }
    }
}

/**
 * Initialize extraction-data: set up tab handlers, load if expanded
 */
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

/**
 * Parse extraction path to get experiment and variant
 * Path pattern: experiments/{experiment}/extraction/{category}/{trait}/{variant}/responses
 */
function parseExtractionPath(path) {
    const match = path.match(/experiments\/([^/]+)\/extraction\/[^/]+\/[^/]+\/([^/]+)\/responses/);
    if (!match) return null;
    return { experiment: match[1], variant: match[2] };
}

/**
 * Load pos.json and neg.json from a folder path and render both
 */
async function loadExtractionData(container, basePath) {
    const posScroll = container.querySelector('.ed-positive .ed-scroll');
    const negScroll = container.querySelector('.ed-negative .ed-scroll');
    const highlightTokens = parseInt(container.dataset.highlightTokens) || null;

    posScroll.innerHTML = ui.renderLoading();
    negScroll.innerHTML = ui.renderLoading();

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

        // Fetch responses and optionally token offsets
        const [posRes, negRes, offsetsRes] = await Promise.all([
            fetch(`${basePath}/pos.json`),
            fetch(`${basePath}/neg.json`),
            highlightTokens ? fetch(`${basePath}/token_offsets.json`).catch(() => null) : null
        ]);

        if (!posRes.ok || !negRes.ok) throw new Error('Failed to load');

        const [posData, negData] = await Promise.all([posRes.json(), negRes.json()]);

        // Parse token offsets if available
        let tokenOffsets = null;
        if (offsetsRes?.ok) {
            tokenOffsets = await offsetsRes.json();
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

/**
 * Render extraction data as a numbered CSV-like table
 * @param {Array} responses - Array of {prompt, response}
 * @param {Object} options - Rendering options
 * @param {Array} options.tokenOffsets - Per-response array of [start, end] char ranges
 * @param {number} options.highlightTokens - Number of tokens to highlight
 */
function renderExtractionTable(responses, options = {}) {
    const { tokenOffsets, highlightTokens } = options;

    if (!Array.isArray(responses) || responses.length === 0) {
        return '<div class="no-data">No data</div>';
    }

    const columnLabel = highlightTokens
        ? `first ${highlightTokens} generated tokens`
        : 'generated tokens';

    let html = '<table class="extraction-table"><thead><tr>';
    html += `<th>#</th><th>contrasting prefill</th><th>${columnLabel}</th>`;
    html += '</tr></thead><tbody>';

    for (let i = 0; i < responses.length; i++) {
        const r = responses[i];
        const prefill = escapeHtml(r.prompt || '');
        const responseText = r.response || '';

        // Apply token highlighting if offsets available
        let continuationHtml;
        if (tokenOffsets?.[i] && highlightTokens) {
            const offsets = tokenOffsets[i].slice(0, highlightTokens);
            continuationHtml = applyTokenHighlights(responseText, offsets);
        } else {
            continuationHtml = escapeHtml(responseText);
        }

        html += `<tr>
            <td class="extraction-num">${i + 1}</td>
            <td>${prefill}</td>
            <td>${continuationHtml}</td>
        </tr>`;
    }

    html += '</tbody></table>';
    return html;
}

/**
 * Apply highlighting to first N tokens based on character offsets
 * @param {string} text - Original text
 * @param {Array} offsets - Array of [start, end] character ranges to highlight
 * @returns {string} HTML with highlighted tokens
 */
function applyTokenHighlights(text, offsets) {
    if (!offsets || offsets.length === 0) {
        return escapeHtml(text);
    }

    // Find the end of the highlighted region
    const highlightEnd = offsets[offsets.length - 1][1];

    // Split text into highlighted portion and rest
    const highlightedText = text.slice(0, highlightEnd);
    const restText = text.slice(highlightEnd);

    return `<span class="token-highlight">${escapeHtml(highlightedText)}</span>${escapeHtml(restText)}`;
}

// ============================================================================
// Content Renderers - Generate HTML for loaded data
// ============================================================================

/**
 * Apply character-range highlights to text, handling HTML escaping properly
 * @param {string} text - Original unescaped text
 * @param {Array} charRanges - Array of [start, end] character ranges to highlight
 * @returns {string} HTML with highlights and proper escaping
 */
function applyCharRangeHighlights(text, charRanges) {
    if (!charRanges || charRanges.length === 0) {
        return escapeHtml(text).replace(/\n/g, '<br>');
    }

    // Sort ranges by start position and merge overlapping
    const sorted = [...charRanges].sort((a, b) => a[0] - b[0]);
    const merged = [];
    for (const range of sorted) {
        const last = merged[merged.length - 1];
        if (last && range[0] <= last[1]) {
            last[1] = Math.max(last[1], range[1]);
        } else {
            merged.push([...range]);
        }
    }

    // Build result by processing segments
    let result = '';
    let pos = 0;

    for (const [start, end] of merged) {
        // Add text before highlight (escaped)
        if (start > pos) {
            result += escapeHtml(text.slice(pos, start)).replace(/\n/g, '<br>');
        }
        // Add highlighted text (escaped, with mark)
        result += '<mark class="hack-highlight">' +
            escapeHtml(text.slice(start, end)).replace(/\n/g, '<br>') +
            '</mark>';
        pos = end;
    }

    // Add remaining text
    if (pos < text.length) {
        result += escapeHtml(text.slice(pos)).replace(/\n/g, '<br>');
    }

    return result;
}

/**
 * Render responses as a table
 * @param {Array} responses - Array of {question, response, trait_score, coherence_score}
 * @param {Object} options - Rendering options
 * @param {boolean} options.showScores - Whether to show trait/coherence scores (default: true)
 * @param {Array<Array>} options.charRanges - Per-response array of [start, end] char ranges to highlight
 */
function renderResponsesTable(responses, options = {}) {
    const { showScores = true, charRanges = [] } = options;

    if (!Array.isArray(responses) || responses.length === 0) {
        return '<div class="error">No responses found</div>';
    }

    // Standard table format for inference/steering responses
    let html = '<table class="table table-compact responses-table"><thead><tr>';
    html += '<th>Question</th><th>Response</th>';
    if (showScores) {
        html += '<th>Trait</th><th>Coh</th>';
    }
    html += '</tr></thead><tbody>';

    for (let i = 0; i < Math.min(responses.length, 20); i++) {
        const r = responses[i];
        const question = escapeHtml(r.prompt || '');

        // Apply char range highlights if available
        let responseHtml;
        if (charRanges[i] && charRanges[i].length > 0) {
            responseHtml = applyCharRangeHighlights(r.response || '', charRanges[i]);
        } else {
            responseHtml = escapeHtml(r.response || '').replace(/\n/g, '<br>');
        }

        html += `<tr>
            <td class="responses-question">${question}</td>
            <td class="responses-response">${responseHtml}</td>`;

        if (showScores) {
            const trait = r.trait_score?.toFixed(0) ?? '-';
            const coh = r.coherence_score?.toFixed(0) ?? '-';
            html += `<td class="responses-score">${trait}</td>
                <td class="responses-score">${coh}</td>`;
        }
        html += '</tr>';
    }

    if (responses.length > 20) {
        const colspan = showScores ? 4 : 2;
        html += `<tr><td colspan="${colspan}" class="dataset-more">...and ${responses.length - 20} more</td></tr>`;
    }

    html += '</tbody></table>';
    return html;
}

/**
 * Render dataset as a list
 * Handles both plain text (one scenario per line) and JSONL (with prompt/system_prompt)
 * @param {string} text - Raw text content
 * @param {Object} options - Rendering options
 * @param {number} options.limit - Max items to show (default: 20)
 */
function renderDatasetList(text, options = {}) {
    const { limit = 20 } = options;
    const trimmed = text.trim();

    // Try parsing as JSON object first (not JSONL)
    if (trimmed.startsWith('{') && trimmed.endsWith('}')) {
        try {
            const obj = JSON.parse(trimmed);
            // Extract all arrays from the object and display them
            let html = '';
            for (const [key, value] of Object.entries(obj)) {
                if (Array.isArray(value) && value.length > 0) {
                    html += `<div class="dataset-section"><strong>${key}</strong> (${value.length})</div>`;
                    html += '<ul class="dataset-list">';
                    const maxItems = limit || 20;
                    const items = value.slice(0, maxItems);
                    for (const item of items) {
                        html += `<li>${escapeHtml(String(item))}</li>`;
                    }
                    if (value.length > maxItems) {
                        html += `<li class="dataset-more">...and ${value.length - maxItems} more</li>`;
                    }
                    html += '</ul>';
                }
            }
            if (html) return html;
        } catch (e) {
            // Not valid JSON, fall through to line-by-line
        }
    }

    const lines = trimmed.split('\n').filter(line => line.trim());
    if (lines.length === 0) {
        return '<div class="error">No examples found</div>';
    }

    // Detect JSONL format (lines start with {)
    const isJsonl = lines[0].trim().startsWith('{');

    const maxItems = limit || 20;
    const examples = lines.slice(0, maxItems);
    let html = '<ul class="dataset-list">';

    for (const line of examples) {
        if (isJsonl) {
            try {
                const obj = JSON.parse(line);
                // Format: show system_prompt and prompt with labels
                const prompt = escapeHtml(obj.prompt || obj.text || line);
                if (obj.system_prompt) {
                    const sysPrompt = escapeHtml(obj.system_prompt);
                    html += `<li>
                        <div class="dataset-field"><span class="dataset-label">system_prompt:</span> ${sysPrompt}</div>
                        <div class="dataset-field"><span class="dataset-label">user_message:</span> ${prompt}</div>
                    </li>`;
                } else {
                    html += `<li>${prompt}</li>`;
                }
            } catch (e) {
                html += `<li>${escapeHtml(line)}</li>`;
            }
        } else {
            html += `<li>${escapeHtml(line)}</li>`;
        }
    }
    if (lines.length > maxItems) {
        html += `<li class="dataset-more">...and ${lines.length - maxItems} more</li>`;
    }
    html += '</ul>';
    return html;
}

// ============================================================================
// Chart Loading - Async load and render charts in findings
// ============================================================================

/**
 * Load and render all chart blocks that haven't been loaded yet.
 * Call this after rendering HTML that may contain chart figures.
 */
async function loadCharts() {
    const chartFigures = document.querySelectorAll('.chart-figure:not([data-loaded])');

    for (const figure of chartFigures) {
        figure.dataset.loaded = 'true';
        const container = figure.querySelector('.chart-container');
        const { chartType, chartPath, chartBars, chartTraits, chartHeight, chartPerplexity, chartProjections } = figure.dataset;

        try {
            // For annotation-stacked charts, bars contains the data paths directly
            if (chartBars) {
                const bars = JSON.parse(chartBars);
                container.innerHTML = '';
                await window.chartTypes.render(chartType, container, bars, {
                    height: chartHeight ? parseInt(chartHeight) : null
                });
            } else {
                const response = await fetch(chartPath);
                if (!response.ok) throw new Error(`${response.status}`);
                const data = await response.json();

                // Build options
                const options = {
                    traits: chartTraits ? chartTraits.split(',') : null,
                    height: chartHeight ? parseInt(chartHeight) : null
                };

                // Add dynamics chart options if present
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
}

// ============================================================================
// Export
// ============================================================================

// ES module exports
export {
    extractCustomBlocks,
    renderCustomBlocks,
    toggleDropdown,
    toggleExtractionData,
    loadExpandedDropdowns,
    initExtractionData,
    loadCharts,
    renderResponsesTable,
    renderDatasetList,
    renderExtractionTable,
};

// Keep window.* namespace for backward compat (onclick handlers reference window.customBlocks.*)
window.customBlocks = {
    // Extraction & rendering
    extractCustomBlocks,
    renderCustomBlocks,

    // Toggle handlers (called from onclick)
    toggleDropdown,
    toggleExtractionData,

    // Auto-load expanded dropdowns and init tabbed components (call after rendering)
    loadExpandedDropdowns,
    initExtractionData,

    // Load charts (call after rendering)
    loadCharts,

    // Content renderers (for direct use)
    renderResponsesTable,
    renderDatasetList,
    renderExtractionTable
};
