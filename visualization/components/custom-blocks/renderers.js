/**
 * Custom block renderers — block descriptors + placeholders → HTML
 *
 * Input:  html (with placeholders), blocks (from parser.js), namespace, options
 * Output: html with rendered block markup substituted in
 * Usage:  import { renderCustomBlocks } from './renderers.js';
 */

import { escapeHtml } from '../../core/utils.js';
import { displayLayer } from '../../core/display.js';
import { createDropdown } from '../../core/ui.js';

// === Helpers ===

/** Escape HTML and convert newlines to <br> */
function escapeAndNormalize(text) {
    return escapeHtml(text).replace(/\n/g, '<br>');
}

/** Render tab buttons: items with cssPrefix, label extractor, and data-attrs builder */
function renderTabButtons(items, cssPrefix, labelFn, attrsFn) {
    return items.map((item, i) => {
        const active = i === 0 ? ' active' : '';
        return `<button class="${cssPrefix}-tab${active}" ${attrsFn(item)}>${labelFn(item)}</button>`;
    }).join('');
}

/** Parse steering response path to extract layer, coef, method, component, position */
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
        ? (window.paths?.desanitizePosition?.(positionPart) ?? positionPart)
        : null;

    return { layer, coef, method, component, position };
}

/**
 * Replace a placeholder in HTML, handling both <p>-wrapped and bare forms.
 * Only one form is replaced per call — running both would let a
 * placeholder like `SIDEBYSIDE_BLOCK_1` match as a prefix of
 * `SIDEBYSIDE_BLOCK_10` on the fallback path.
 */
function insertBlock(html, placeholder, rendered) {
    const wrapped = `<p>${placeholder}</p>`;
    if (html.includes(wrapped)) return html.replace(wrapped, rendered);
    return html.replace(placeholder, rendered);
}

// === Per-block HTML generators ===

/** Create HTML for an expandable dropdown (responses or dataset) */
function createDropdownHtml(id, label, type, path, options = {}) {
    const { expanded = false, noScores = false, limit = null, height = null, color = null, caption = null } = options;
    const colorClass = color ? ` dropdown-${color}` : '';
    const limitAttr = limit ? ` data-limit="${limit}"` : '';
    const heightAttr = height ? ` data-height="${height}"` : '';

    // For steered responses, show metadata subtitle
    let headerExtras = '';
    if (type === 'Responses') {
        const meta = parseSteeringResponsePath(path);
        if (meta) {
            headerExtras = `<span class="dropdown-meta">L${displayLayer(meta.layer)} · coef ${meta.coef} · ${meta.component} · ${meta.method}</span>`;
        }
    }

    return createDropdown({
        id, label, expanded, caption,
        toggleHandler: `window.customBlocks.toggleDropdown('${id}')`,
        classes: {
            container: `dropdown responses-dropdown${colorClass}`,
            header: 'dropdown-header responses-header',
            toggle: 'dropdown-toggle responses-toggle',
            label: 'dropdown-label responses-label',
            body: 'dropdown-body responses-content',
        },
        containerAttrs: `data-type="${type}" data-path="${path}" data-no-scores="${noScores}" data-auto-expand="${expanded}"${limitAttr}${heightAttr}`,
        headerExtras,
    });
}

/** Create HTML for steered-responses (3-column comparison: Question | PV | Natural) */
function createSteeredResponsesHtml(id, block) {
    const { label, traits } = block;
    const defaultTrait = traits[0]?.key || '';

    const tabsHtml = renderTabButtons(traits, 'sr',
        t => t.label,
        t => `data-trait="${t.key}" data-pv-path="${t.pvPath}" data-natural-path="${t.naturalPath}"`
    );

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

/** Create HTML for extraction-data (tabbed pos/neg viewer) */
function createExtractionDataHtml(id, block) {
    const { label, expanded, highlightTokens, traits } = block;
    const defaultTrait = traits[0]?.name || '';
    const defaultPath = traits[0]?.path || '';
    const tokensAttr = highlightTokens ? ` data-highlight-tokens="${highlightTokens}"` : '';

    const tabsHtml = renderTabButtons(traits, 'ed',
        t => t.name,
        t => `data-trait="${t.name}" data-path="${t.path}"`
    );

    const bodyHtml = `
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
    `;

    return createDropdown({
        id, label, expanded,
        toggleHandler: `window.customBlocks.toggleExtractionData('${id}')`,
        classes: {
            container: 'extraction-data-container',
            header: 'ed-header',
            toggle: 'ed-toggle',
            label: 'ed-label',
            body: 'ed-body',
        },
        containerAttrs: `data-active="${defaultTrait}" data-default-path="${defaultPath}"${tokensAttr}`,
        bodyHtml,
    });
}

// === Main: substitute placeholders ===

/** Replace block placeholders in HTML with rendered components */
function renderCustomBlocks(html, blocks, namespace = 'block', options = {}) {
    const { assetBaseUrl = '/docs/viz_findings/' } = options;

    // Responses blocks -> expandable dropdowns
    blocks.responses.forEach((block, i) => {
        const dropdownId = `responses-${namespace}-${i}`;
        const dropdownHtml = createDropdownHtml(dropdownId, block.label, 'Responses', block.path, {
            expanded: block.expanded,
            noScores: block.noScores,
            height: block.height,
            color: block.color,
            caption: block.caption
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
            color: block.color,
            caption: block.caption
        });
        html = insertBlock(html, `DATASET_BLOCK_${i}`, dropdownHtml);
    });

    // Figure blocks -> img with caption (auto-numbered via CSS counter)
    blocks.figures.forEach((block, i) => {
        const imgPath = block.path.startsWith('assets/')
            ? `${assetBaseUrl}${block.path}`
            : block.path;
        const sizeClass = block.size ? ` fig-${block.size}` : '';
        const figureHtml = `
            <figure class="fig${sizeClass}">
                <img src="${imgPath}" alt="${block.caption || ''}">
                ${block.caption ? `<p class="fig-caption">${block.caption}</p>` : ''}
            </figure>
        `;
        html = insertBlock(html, `FIGURE_BLOCK_${i}`, figureHtml);
    });

    // Side-by-side blocks -> two panels with shared caption
    blocks.sideBySide.forEach((block, i) => {
        function panelHtml(side) {
            const p = side.path || '';
            const label = side.label || '';
            const isChart = p.startsWith('chart:');
            if (isChart) {
                // chart:type:path — rendered later by loadCharts via data attributes
                const [, chartType, chartPath] = p.split(':');
                const styleAttr = block.style ? `data-chart-style="${block.style}"` : '';
                return `
                    <div class="sbs-panel">
                        ${label ? `<div class="sbs-label">${label}</div>` : ''}
                        <div class="chart-container" data-chart-type="${chartType}" data-chart-path="${chartPath}" ${styleAttr}></div>
                    </div>`;
            }
            // Image path
            const imgPath = p.startsWith('assets/') ? `${assetBaseUrl}${p}` : p.startsWith('/') ? p : `/${p}`;
            return `
                <div class="sbs-panel">
                    ${label ? `<div class="sbs-label">${label}</div>` : ''}
                    <img src="${imgPath}" alt="${label}">
                </div>`;
        }
        const sbsHtml = `
            <figure class="side-by-side">
                <div class="sbs-container">
                    ${panelHtml(block.left)}
                    ${panelHtml(block.right)}
                </div>
                ${block.caption ? `<p class="fig-caption">${block.caption}</p>` : ''}
            </figure>`;
        html = insertBlock(html, `SIDEBYSIDE_BLOCK_${i}`, sbsHtml);
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
        const projectionsAttr = block.projections ? JSON.stringify(block.projections) : '';
        const labelsAttr = block.labels ? JSON.stringify(block.labels) : '';
        const chartHtml = `
            <figure class="chart-figure" id="${chartId}"
                    data-chart-type="${block.type}"
                    data-chart-path="${block.path}"
                    data-chart-traits="${block.traits?.join(',') || ''}"
                    data-chart-labels='${labelsAttr}'
                    data-chart-height="${block.height || ''}"
                    data-chart-perplexity="${block.perplexity || ''}"
                    data-chart-projections='${projectionsAttr}'>
                <div class="chart-container">
                    <div class="chart-loading">Loading chart...</div>
                </div>
                ${block.caption ? `<p class="fig-caption">${block.caption}</p>` : ''}
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
                    data-chart-height="${block.height || ''}"
                    data-chart-colors="${block.colors || ''}">
                <div class="chart-container">
                    <div class="chart-loading">Loading chart...</div>
                </div>
                ${block.caption ? `<p class="fig-caption">${block.caption}</p>` : ''}
            </figure>
        `;
        html = insertBlock(html, `ANNOTATION_STACKED_BLOCK_${i}`, chartHtml);
    });

    return html;
}

// === Renderers used by loaders to fill dropdown bodies ===

/** Apply highlighting to first N tokens based on character offsets */
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

/** Apply character-range highlights to text, handling HTML escaping properly */
function applyCharRangeHighlights(text, charRanges) {
    if (!charRanges || charRanges.length === 0) {
        return escapeAndNormalize(text);
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
        if (start > pos) {
            result += escapeAndNormalize(text.slice(pos, start));
        }
        result += '<mark class="hack-highlight">' +
            escapeAndNormalize(text.slice(start, end)) +
            '</mark>';
        pos = end;
    }

    if (pos < text.length) {
        result += escapeAndNormalize(text.slice(pos));
    }

    return result;
}

/** Render extraction data as a numbered table with optional token highlighting */
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

/** Render responses as a table with optional scores and char-range highlights */
function renderResponsesTable(responses, options = {}) {
    const { showScores = true, charRanges = [] } = options;

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
        const question = escapeHtml(r.prompt || '');

        let responseHtml;
        if (charRanges[i] && charRanges[i].length > 0) {
            responseHtml = applyCharRangeHighlights(r.response || '', charRanges[i]);
        } else {
            responseHtml = escapeAndNormalize(r.response || '');
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

/** Render dataset as a list (plain text, JSON object, or JSONL format) */
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

export {
    renderCustomBlocks,
    renderResponsesTable,
    renderDatasetList,
    renderExtractionTable,
};
