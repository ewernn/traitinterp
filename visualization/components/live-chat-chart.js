// Live Chat Chart — Trait chart rendering, token highlighting, and steering controls
//
// Input: Conversation tree token data, steering user actions
// Output: Plotly chart updates, steering coefficient state
// Usage: import { initTraitChart, updateTraitChart, ... } from './live-chat-chart.js'

import { smoothData } from '../core/utils.js';
import { getChartColors, getCssVar, hexToRgba } from '../core/display.js';
import { buildChartLayout, renderChart, updateChart } from '../core/charts.js';

// Module-local state
let steeringCoefficients = {};  // {trait: coefficient} - default 0 for all traits
let vectorMetadata = {};  // Cached vector metadata: {trait: {layer, method, source}}
let showSmoothedLine = true;

function getTraitColor(idx) {
    return getChartColors()[idx % 10];
}

/**
 * Initialize empty trait chart
 */
function initTraitChart() {
    const chartDiv = document.getElementById('trait-chart');
    if (!chartDiv) return;

    const layout = buildChartLayout({
        preset: 'timeSeries',
        traces: [],
        legendPosition: 'none',  // Using custom HTML legend
        xaxis: { title: 'Token', showgrid: true },
        yaxis: { title: 'Trait Score', showgrid: true, zeroline: true }
    });
    renderChart(chartDiv, [], layout);
}

/**
 * Update trait chart with data from conversation tree.
 * @param {Object} conversationTree - The conversation tree instance
 * @param {string|null} hoveredMessageId - Currently hovered message ID (for region shading)
 */
function updateTraitChart(conversationTree, hoveredMessageId) {
    const chartDiv = document.getElementById('trait-chart');
    const legendDiv = document.getElementById('chart-legend');

    const globalTokens = conversationTree.globalTokens;
    if (!chartDiv || globalTokens.length === 0) return;

    const firstEvent = globalTokens[0];
    const allTraitNames = Object.keys(firstEvent.trait_scores || {});
    if (allTraitNames.length === 0) return;

    // Filter by selected traits (if any are selected)
    // selectedTraits has full paths like "behavioral_tendency/refusal"
    // allTraitNames has just base names like "refusal"
    const selectedTraits = window.state?.selectedTraits;
    let traitNames = allTraitNames;
    if (selectedTraits && selectedTraits.size > 0) {
        // Extract base names from selected traits for matching
        const selectedBaseNames = new Set(
            Array.from(selectedTraits).map(t => t.includes('/') ? t.split('/').pop() : t)
        );
        traitNames = allTraitNames.filter(t => selectedBaseNames.has(t));
    }
    if (traitNames.length === 0) traitNames = allTraitNames;  // Fallback: show all if none match

    const smoothingWindow = window.state.smoothingWindow || 3;
    const traces = [];

    traitNames.forEach((trait, idx) => {
        const color = getTraitColor(idx);

        // Collect all token scores with their indices
        const indices = [];
        const scores = [];

        globalTokens.forEach((e, i) => {
            const score = e.trait_scores[trait] || 0;
            indices.push(i);
            scores.push(score);
        });

        // Apply smoothing if requested
        const yValues = showSmoothedLine && scores.length >= smoothingWindow
            ? smoothData(scores, smoothingWindow)
            : scores;

        // Single trace per trait - all tokens
        traces.push({
            name: trait,
            x: indices,
            y: yValues,
            type: 'scatter',
            mode: 'lines',
            line: { color: color, width: 2 },
            hovertemplate: `${trait}: %{y:.3f}<extra></extra>`,
            showlegend: true
        });
    });

    // Update legend with steering buttons
    if (legendDiv) {
        legendDiv.innerHTML = traitNames.map((trait, idx) => {
            // Get vector metadata for this trait (trait names might be just base names)
            // Find matching trait in vectorMetadata by checking if key ends with trait name
            let metadata = null;
            for (const [fullPath, meta] of Object.entries(vectorMetadata)) {
                if (fullPath.endsWith(trait) || fullPath.endsWith('/' + trait)) {
                    metadata = meta;
                    break;
                }
            }

            const tooltipText = metadata
                ? `L${metadata.layer} ${metadata.method} (${metadata.source})`
                : 'no metadata';

            const currentCoef = steeringCoefficients[trait] || 0;
            const coefficients = [-1, -0.5, 0, 0.5, 1];

            return `
                <div class="legend-item-row">
                    <span class="legend-item has-tooltip"
                          data-tooltip="${tooltipText}"
                          data-trait="${trait}">
                        <span class="legend-color" style="background: ${getTraitColor(idx)}"></span>
                        ${trait}
                    </span>
                    <div class="steering-buttons" data-trait="${trait}">
                        ${coefficients.map(coef => {
                            const label = coef === 0 ? '0' : (coef > 0 ? `+${coef}x` : `${coef}x`);
                            const isActive = currentCoef === coef ? 'active' : '';
                            return `<button class="btn btn-xs steer-btn ${isActive}" data-coef="${coef}" onclick="setSteeringCoefficient('${trait}', ${coef})">${label}</button>`;
                        }).join('')}
                    </div>
                </div>
            `;
        }).join('');
    }

    // Build shapes for message regions
    const shapes = buildMessageRegionShapes(conversationTree, hoveredMessageId);

    const layout = buildChartLayout({
        preset: 'timeSeries',
        traces,
        legendPosition: 'none',  // Using custom HTML legend
        xaxis: { title: 'Token', showgrid: true },
        yaxis: { title: 'Trait Score', showgrid: true, zeroline: true },
        shapes
    });
    updateChart(chartDiv, traces, layout);

    // Add hover event listener for token highlighting
    // Note: Plotly event listeners are persistent across reacts, so we don't need to remove them
    // Only attach if not already attached
    if (!chartDiv._tokenHoverAttached) {
        chartDiv.on('plotly_hover', (data) => {
            if (data.points && data.points.length > 0) {
                const tokenIdx = Math.round(data.points[0].x);
                highlightTokenInChat(tokenIdx);
            }
        });

        chartDiv.on('plotly_unhover', () => {
            clearTokenHighlight();
        });

        chartDiv._tokenHoverAttached = true;
    }
}

/**
 * Highlight a specific token in the chat messages
 */
function highlightTokenInChat(tokenIdx) {
    // Clear previous highlights
    clearTokenHighlight();

    // Find token span with this index
    const tokenSpan = document.querySelector(`.token-span[data-token-idx="${tokenIdx}"]`);
    if (tokenSpan) {
        tokenSpan.classList.add('hovered-token');
        // Scroll into view if needed
        tokenSpan.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
}

/**
 * Clear token highlighting
 */
function clearTokenHighlight() {
    document.querySelectorAll('.token-span.hovered-token').forEach(el => {
        el.classList.remove('hovered-token');
    });
}

/**
 * Build Plotly shapes for message regions
 */
function buildMessageRegionShapes(conversationTree, hoveredMessageId) {
    const shapes = [];

    for (const region of conversationTree.messageRegions) {
        // Skip empty regions (user messages with no tokens)
        if (region.startIdx === region.endIdx && region.role === 'user') {
            // Draw a thin vertical line for user messages
            shapes.push({
                type: 'line',
                x0: region.startIdx - 0.5,
                x1: region.startIdx - 0.5,
                y0: 0,
                y1: 1,
                yref: 'paper',
                line: {
                    color: getCssVar('--chart-1', '#4a9eff') + '80',  // 50% opacity
                    width: 2,
                    dash: 'dot'
                },
                layer: 'below'
            });
            continue;
        }

        const isHovered = region.messageId === hoveredMessageId;
        const baseOpacity = region.role === 'user' ? 0.15 : 0.08;
        const hoverOpacity = 0.25;

        shapes.push({
            type: 'rect',
            x0: region.startIdx - 0.5,
            x1: region.endIdx - 0.5,
            y0: 0,
            y1: 1,
            yref: 'paper',
            fillcolor: region.role === 'user'
                ? hexToRgba(getCssVar('--chart-1', '#4a9eff'), isHovered ? hoverOpacity : baseOpacity)
                : hexToRgba(getCssVar('--chart-3', '#51cf66'), isHovered ? hoverOpacity : baseOpacity),
            line: { width: 0 },
            layer: 'below'
        });
    }

    return shapes;
}

/**
 * Update chart highlighting when hovering over messages
 */
function updateChartHighlight(conversationTree, hoveredMessageId) {
    const chartDiv = document.getElementById('trait-chart');
    if (!chartDiv || !chartDiv.data || chartDiv.data.length === 0) return;

    const shapes = buildMessageRegionShapes(conversationTree, hoveredMessageId);
    Plotly.relayout(chartDiv, { shapes: shapes });
}

/**
 * Set steering coefficient for a trait
 */
function setSteeringCoefficient(trait, coefficient) {
    steeringCoefficients[trait] = coefficient;
    updateSteeringButtonsUI();
}

/**
 * Update steering buttons UI to reflect current state
 */
function updateSteeringButtonsUI() {
    document.querySelectorAll('.steering-buttons').forEach(container => {
        const trait = container.dataset.trait;
        const currentCoef = steeringCoefficients[trait] || 0;

        container.querySelectorAll('.steer-btn').forEach(btn => {
            const btnCoef = parseFloat(btn.dataset.coef);
            btn.classList.toggle('active', btnCoef === currentCoef);
        });
    });
}

// Getters/setters for module-local state
function getSteeringCoefficients() { return steeringCoefficients; }
function getVectorMetadata() { return vectorMetadata; }
function setVectorMetadata(meta) { vectorMetadata = meta; }
function getShowSmoothedLine() { return showSmoothedLine; }
function setShowSmoothedLine(val) { showSmoothedLine = val; }
function resetChartState() {
    vectorMetadata = {};
}

export {
    initTraitChart,
    updateTraitChart,
    highlightTokenInChat,
    clearTokenHighlight,
    buildMessageRegionShapes,
    updateChartHighlight,
    setSteeringCoefficient,
    updateSteeringButtonsUI,
    getTraitColor,
    getSteeringCoefficients,
    getVectorMetadata,
    setVectorMetadata,
    getShowSmoothedLine,
    setShowSmoothedLine,
    resetChartState,
};

// Window binding for onclick in generated HTML
window.setSteeringCoefficient = setSteeringCoefficient;
