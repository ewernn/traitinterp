/**
 * Custom Blocks - Parsing and rendering for ::: syntax in markdown
 *
 * Supported blocks:
 *   :::responses path "label" [expanded] [no-scores]:::        - Expandable response table
 *   :::dataset path "label" [expanded] [limit=N] [height=N]::: - Expandable dataset list
 *   :::prompts path "label" [expanded]:::                      - Expandable prompts table
 *   :::figure path "caption" size:::                     - Image with caption (size: small|medium|large)
 *   :::example ... :::                                   - Example box with optional caption
 *   :::response-tabs ... :::                             - Tabbed response comparison grid
 *
 * Response flags:
 *   expanded  - Start expanded instead of collapsed
 *   no-scores - Hide trait/coherence score columns
 *
 * Response-tabs syntax:
 *   :::response-tabs "Row1Label" "Row2Label"
 *   col1: "Col1Label" | row1path | row2path
 *   col2: "Col2Label" | row1path | row2path
 *   :::
 */

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
        prompts: [],
        figures: [],
        examples: [],
        responseTabs: []
    };

    // :::responses path "label" [expanded] [no-scores]:::
    markdown = markdown.replace(
        /:::responses\s+([^\s:]+)(?:\s+"([^"]*)")?([^:]*):::/g,
        (match, path, label, flags) => {
            const expanded = /\bexpanded\b/.test(flags);
            const noScores = /\bno-scores\b/.test(flags);
            blocks.responses.push({
                path,
                label: label || 'View responses',
                expanded,
                noScores
            });
            return `RESPONSE_BLOCK_${blocks.responses.length - 1}`;
        }
    );

    // :::dataset path "label" [expanded] [limit=N] [height=N]:::
    markdown = markdown.replace(
        /:::dataset\s+([^\s:]+)(?:\s+"([^"]*)")?([^:]*):::/g,
        (match, path, label, flags) => {
            const expanded = /\bexpanded\b/.test(flags);
            const limitMatch = flags.match(/\blimit=(\d+)/);
            const heightMatch = flags.match(/\bheight=(\d+)/);
            blocks.datasets.push({
                path,
                label: label || 'View examples',
                expanded,
                limit: limitMatch ? parseInt(limitMatch[1]) : null,
                height: heightMatch ? parseInt(heightMatch[1]) : null
            });
            return `DATASET_BLOCK_${blocks.datasets.length - 1}`;
        }
    );

    // :::prompts path "label" [expanded]:::
    markdown = markdown.replace(
        /:::prompts\s+([^\s:]+)(?:\s+"([^"]*)")?([^:]*):::/g,
        (match, path, label, flags) => {
            const expanded = /\bexpanded\b/.test(flags);
            blocks.prompts.push({ path, label: label || 'View prompts', expanded });
            return `PROMPT_BLOCK_${blocks.prompts.length - 1}`;
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

    // :::response-tabs "Row1" "Row2"\n col: "Label" | path1 | path2 \n:::
    markdown = markdown.replace(
        /:::response-tabs\s+"([^"]+)"\s+"([^"]+)"\s*\n([\s\S]*?)\n:::/g,
        (match, row1Label, row2Label, body) => {
            const columns = [];
            for (const line of body.trim().split('\n')) {
                // Parse: colKey: "Label" | path1 | path2
                const colMatch = line.match(/^\s*(\w+):\s*"([^"]+)"\s*\|\s*([^\s|]+)\s*\|\s*([^\s|]+)/);
                if (colMatch) {
                    columns.push({
                        key: colMatch[1],
                        label: colMatch[2],
                        paths: [colMatch[3].trim(), colMatch[4].trim()]
                    });
                }
            }
            blocks.responseTabs.push({
                rowLabels: [row1Label, row2Label],
                columns
            });
            return `RESPONSE_TABS_BLOCK_${blocks.responseTabs.length - 1}`;
        }
    );

    return { markdown, blocks };
}

// ============================================================================
// Block Rendering - Generate HTML from extracted blocks
// ============================================================================

/**
 * Replace block placeholders in HTML with rendered components
 * @param {string} html - HTML with placeholders
 * @param {Object} blocks - Extracted block data
 * @param {string} namespace - Unique namespace for IDs (e.g., filename)
 * @returns {string} - HTML with blocks rendered
 */
function renderCustomBlocks(html, blocks, namespace = 'block') {
    // Responses blocks -> expandable dropdowns
    blocks.responses.forEach((block, i) => {
        const dropdownId = `responses-${namespace}-${i}`;
        const dropdownHtml = createDropdownHtml(dropdownId, block.label, 'Responses', block.path, {
            expanded: block.expanded,
            noScores: block.noScores
        });
        html = html.replace(`<p>RESPONSE_BLOCK_${i}</p>`, dropdownHtml);
        html = html.replace(`RESPONSE_BLOCK_${i}`, dropdownHtml);
    });

    // Dataset blocks -> expandable dropdowns
    blocks.datasets.forEach((block, i) => {
        const dropdownId = `dataset-${namespace}-${i}`;
        const dropdownHtml = createDropdownHtml(dropdownId, block.label, 'Dataset', block.path, {
            expanded: block.expanded,
            limit: block.limit,
            height: block.height
        });
        html = html.replace(`<p>DATASET_BLOCK_${i}</p>`, dropdownHtml);
        html = html.replace(`DATASET_BLOCK_${i}`, dropdownHtml);
    });

    // Prompts blocks -> expandable dropdowns
    blocks.prompts.forEach((block, i) => {
        const dropdownId = `prompts-${namespace}-${i}`;
        const dropdownHtml = createDropdownHtml(dropdownId, block.label, 'Prompts', block.path, {
            expanded: block.expanded
        });
        html = html.replace(`<p>PROMPT_BLOCK_${i}</p>`, dropdownHtml);
        html = html.replace(`PROMPT_BLOCK_${i}`, dropdownHtml);
    });

    // Figure blocks -> img with caption
    blocks.figures.forEach((block, i) => {
        const imgPath = block.path.startsWith('assets/')
            ? `/docs/viz_findings/${block.path}`
            : block.path;
        const sizeClass = block.size ? ` fig-${block.size}` : '';
        const figureHtml = `
            <figure class="fig${sizeClass}">
                <img src="${imgPath}" alt="${block.caption}">
                ${block.caption ? `<figcaption>${block.caption}</figcaption>` : ''}
            </figure>
        `;
        html = html.replace(`<p>FIGURE_BLOCK_${i}</p>`, figureHtml);
        html = html.replace(`FIGURE_BLOCK_${i}`, figureHtml);
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
        html = html.replace(`<p>EXAMPLE_BLOCK_${i}</p>`, exampleHtml);
        html = html.replace(`EXAMPLE_BLOCK_${i}`, exampleHtml);
    });

    // Response-tabs blocks -> tabbed grid
    blocks.responseTabs.forEach((block, i) => {
        const tabsId = `response-tabs-${namespace}-${i}`;
        const tabsHtml = createResponseTabsHtml(tabsId, block);
        html = html.replace(`<p>RESPONSE_TABS_BLOCK_${i}</p>`, tabsHtml);
        html = html.replace(`RESPONSE_TABS_BLOCK_${i}`, tabsHtml);
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
    const position = positionPart ? positionPart.replace(/_/g, match => match === '__' ? '[' : ':').replace(/\[/g, '[').replace(/:$/, ']') : null;
    // Simpler desanitize: response__5 -> response[:5], response___all -> response[:]
    const desanitizedPosition = positionPart ?
        positionPart.replace('__', '[:').replace('_all', ':]').replace(/_(\d+)$/, ':$1]') + (positionPart.includes('__') && !positionPart.includes('_all') && !/_\d+$/.test(positionPart) ? ']' : '') :
        null;

    return { layer, coef, method, component, position: desanitizedPosition || positionPart };
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
    const { expanded = false, noScores = false, limit = null, height = null } = options;
    const expandedClass = expanded ? ' expanded' : '';
    const toggleChar = expanded ? '−' : '+';
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
        <div class="dropdown responses-dropdown${expandedClass}" id="${id}" data-type="${type}" data-path="${path}" data-no-scores="${noScores}" data-auto-expand="${expanded}"${limitAttr}${heightAttr}>
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
 * Create HTML for response-tabs component (2×N grid of tabs)
 * @param {string} id - Unique ID for this component
 * @param {Object} block - { rowLabels: [str, str], columns: [{key, label, paths: [str, str]}] }
 */
function createResponseTabsHtml(id, block) {
    const { rowLabels, columns } = block;

    // Build tab grid: rows are methods, columns are traits
    // First tab is selected by default
    const defaultTab = `${columns[0]?.key}-0`;

    const tabsHtml = rowLabels.map((rowLabel, rowIdx) => {
        const rowTabs = columns.map(col => {
            const tabKey = `${col.key}-${rowIdx}`;
            const isActive = tabKey === defaultTab;
            const path = col.paths[rowIdx];
            return `<button class="rtabs-tab${isActive ? ' active' : ''}" data-tab="${tabKey}" data-path="${path}">${col.label}</button>`;
        }).join('');
        return `
            <div class="rtabs-row">
                <span class="rtabs-row-label">${rowLabel}</span>
                <div class="rtabs-row-tabs">${rowTabs}</div>
            </div>
        `;
    }).join('');

    return `
        <div class="rtabs-container" id="${id}" data-active="${defaultTab}">
            <div class="rtabs-grid">${tabsHtml}</div>
            <div class="rtabs-content"></div>
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

            // Try loading annotations (baseline.json → baseline_annotations_chars.json)
            let charRanges = [];
            try {
                const annotationsPath = path.replace('.json', '_annotations_chars.json');
                const annotationsResp = await fetch(annotationsPath);
                if (annotationsResp.ok) {
                    const annotations = await annotationsResp.json();
                    charRanges = annotations.hack_chars || [];
                }
            } catch (e) {
                // Annotations not available, continue without them
            }

            content.innerHTML = renderResponsesTable(data, { showScores: !noScores, charRanges });
        } else if (type === 'Dataset') {
            const text = await response.text();
            content.innerHTML = renderDatasetList(text, { limit });
            // Apply custom height if specified
            if (height) {
                const list = content.querySelector('.dataset-list');
                if (list) list.style.maxHeight = `${height}px`;
            }
        } else if (type === 'Prompts') {
            const data = await response.json();
            content.innerHTML = renderPromptsTable(data);
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
        toggle.textContent = '+';
    } else {
        if (!content.innerHTML) {
            await fetchDropdownContent(dropdown);
        }
        dropdown.classList.add('expanded');
        content.style.display = 'block';
        toggle.textContent = '−';
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
        toggle.textContent = '−';
    }

    // Also initialize response-tabs components
    initResponseTabs();
}

/**
 * Initialize response-tabs: set up click handlers and load first tab
 */
function initResponseTabs() {
    const containers = document.querySelectorAll('.rtabs-container');
    for (const container of containers) {
        // Skip if already initialized
        if (container.dataset.initialized) continue;
        container.dataset.initialized = 'true';

        // Set up tab click handlers
        container.querySelectorAll('.rtabs-tab').forEach(tab => {
            tab.addEventListener('click', () => switchResponseTab(container, tab));
        });

        // Load the default (first) tab
        const activeTab = container.querySelector('.rtabs-tab.active');
        if (activeTab) {
            loadResponseTabContent(container, activeTab.dataset.path);
        }
    }
}

/**
 * Switch to a different tab in response-tabs component
 */
function switchResponseTab(container, tab) {
    // Update active state
    container.querySelectorAll('.rtabs-tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    container.dataset.active = tab.dataset.tab;

    // Load content
    loadResponseTabContent(container, tab.dataset.path);
}

/**
 * Load and render content for a response-tabs tab
 */
async function loadResponseTabContent(container, path) {
    const content = container.querySelector('.rtabs-content');
    content.innerHTML = ui.renderLoading();

    try {
        const response = await fetch(path);
        if (!response.ok) throw new Error('Failed to load');

        const data = await response.json();
        content.innerHTML = renderResponsesTable(data, { showScores: false });
    } catch (error) {
        content.innerHTML = `<p class="no-data">Failed to load: ${error.message}</p>`;
    }
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
        return window.escapeHtml(text).replace(/\n/g, '<br>');
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
            result += window.escapeHtml(text.slice(pos, start)).replace(/\n/g, '<br>');
        }
        // Add highlighted text (escaped, with mark)
        result += '<mark class="hack-highlight">' +
            window.escapeHtml(text.slice(start, end)).replace(/\n/g, '<br>') +
            '</mark>';
        pos = end;
    }

    // Add remaining text
    if (pos < text.length) {
        result += window.escapeHtml(text.slice(pos)).replace(/\n/g, '<br>');
    }

    return result;
}

/**
 * Render responses as a table
 * @param {Array} responses - Array of {question, response, trait_score, coherence_score}
 * @param {Object} options - Rendering options
 * @param {boolean} options.showScores - Whether to show trait/coherence scores (default: true)
 * @param {Array<string>} options.highlightPatterns - Regex patterns to highlight in responses
 * @param {Array<Array>} options.charRanges - Per-response array of [start, end] char ranges
 */
function renderResponsesTable(responses, options = {}) {
    const { showScores = true, highlightPatterns = [], charRanges = [] } = options;

    if (!Array.isArray(responses) || responses.length === 0) {
        return '<div class="error">No responses found</div>';
    }

    let html = '<table class="table table-compact responses-table"><thead><tr>';
    html += '<th>Question</th><th>Response</th>';
    if (showScores) {
        html += '<th>Trait</th><th>Coh</th>';
    }
    html += '</tr></thead><tbody>';

    for (let i = 0; i < Math.min(responses.length, 20); i++) {
        const r = responses[i];
        const question = window.escapeHtml(r.question || '');

        // Apply char range highlights if available, otherwise use regex patterns
        let responseHtml;
        if (charRanges[i] && charRanges[i].length > 0) {
            responseHtml = applyCharRangeHighlights(r.response || '', charRanges[i]);
        } else {
            responseHtml = window.escapeHtml(r.response || '').replace(/\n/g, '<br>');
            // Apply highlight patterns (legacy)
            for (const pattern of highlightPatterns) {
                try {
                    const regex = new RegExp(`(${pattern})`, 'gi');
                    responseHtml = responseHtml.replace(regex, '<mark>$1</mark>');
                } catch (e) {
                    console.warn(`Invalid highlight pattern: ${pattern}`);
                }
            }
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

    html += '</tbody></table>';

    if (responses.length > 20) {
        html += `<div class="dataset-more">...and ${responses.length - 20} more</div>`;
    }
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
    const lines = text.trim().split('\n').filter(line => line.trim());
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
                const prompt = window.escapeHtml(obj.prompt || obj.text || line);
                if (obj.system_prompt) {
                    const sysPrompt = window.escapeHtml(obj.system_prompt);
                    html += `<li>
                        <div class="dataset-field"><span class="dataset-label">system_prompt:</span> ${sysPrompt}</div>
                        <div class="dataset-field"><span class="dataset-label">user_message:</span> ${prompt}</div>
                    </li>`;
                } else {
                    html += `<li>${prompt}</li>`;
                }
            } catch (e) {
                html += `<li>${window.escapeHtml(line)}</li>`;
            }
        } else {
            html += `<li>${window.escapeHtml(line)}</li>`;
        }
    }
    html += '</ul>';

    if (lines.length > maxItems) {
        html += `<div class="dataset-more">...and ${lines.length - maxItems} more</div>`;
    }
    return html;
}

/**
 * Render prompts as a table
 */
function renderPromptsTable(data) {
    const prompts = data.prompts || [];
    if (prompts.length === 0) {
        return '<div class="error">No prompts found</div>';
    }

    let html = '<table class="table table-compact responses-table"><thead><tr>';
    html += '<th>Prompt</th><th>Bias</th>';
    html += '</tr></thead><tbody>';

    for (const p of prompts.slice(0, 20)) {
        const text = window.escapeHtml(p.text || '');
        const bias = p.bias_id ? `#${p.bias_id}` : '-';
        html += `<tr>
            <td class="responses-question">${text}</td>
            <td class="responses-score">${bias}</td>
        </tr>`;
    }

    html += '</tbody></table>';

    if (prompts.length > 20) {
        html += `<div class="dataset-more">...and ${prompts.length - 20} more</div>`;
    }
    return html;
}

// ============================================================================
// Export
// ============================================================================

window.customBlocks = {
    // Extraction & rendering
    extractCustomBlocks,
    renderCustomBlocks,

    // Toggle handler (called from onclick)
    toggleDropdown,

    // Auto-load expanded dropdowns and init response-tabs (call after rendering)
    loadExpandedDropdowns,
    initResponseTabs,

    // Content renderers (for direct use)
    renderResponsesTable,
    renderDatasetList,
    renderPromptsTable
};
