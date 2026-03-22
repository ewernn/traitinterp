// Chart configuration primitives
// Single source of truth for Plotly chart layouts
//
// Usage:
//   const layout = buildChartLayout({ preset: 'layerChart', traces, legendPosition: 'below' });
//   renderChart(divId, traces, layout);

// Standard Plotly render options (always the same)
const PLOTLY_CONFIG = { responsive: true, displayModeBar: false };

// Chart type presets (loose - all properties overridable)
const CHART_PRESETS = {
    timeSeries: {
        margin: { l: 50, r: 20, t: 20, b: 40 },
        hovermode: 'x unified',
        xaxis: { tickangle: -45, tickfont: { size: 9 } },
        yaxis: { tickfont: { size: 10 } }
    },
    layerChart: {
        margin: { l: 50, r: 20, t: 10, b: 25 },
        xaxis: { dtick: 5, tickfont: { size: 10 } },  // No title - layer numbers are self-explanatory
        yaxis: { tickfont: { size: 10 } }
    },
    heatmap: {
        margin: { l: 40, r: 10, t: 5, b: 80 },
        xaxis: { tickangle: -45, tickfont: { size: 8 } },
        yaxis: { tickfont: { size: 9 } }
    },
    barChart: {
        margin: { l: 50, r: 20, t: 10, b: 40 },
        yaxis: { automargin: true }
    }
};

/**
 * Build smart chart layout with automatic legend margin calculation
 *
 * @param {Object} options
 * @param {string} options.preset - 'timeSeries' | 'layerChart' | 'heatmap' | 'barChart' | null
 * @param {Array} options.traces - Chart traces (used to count legend items)
 * @param {number} options.height - Chart height in pixels (default: 200)
 * @param {string} options.legendPosition - 'above' | 'below' | 'right' | 'none' (default: 'below')
 * @param {Object} options.xaxis - X-axis config overrides
 * @param {Object} options.yaxis - Y-axis config overrides
 * @param {Array} options.shapes - Plotly shapes (separators, highlights)
 * @param {Object} options.margin - Margin overrides (applied after legend calculation)
 * @param {...Object} options.custom - Any other Plotly layout properties
 * @returns {Object} Complete Plotly layout object
 */
function buildChartLayout({
    preset = null,
    traces = [],
    height = null,  // null = auto-calculate based on preset/data
    legendPosition = 'below',
    xaxis = {},
    yaxis = {},
    shapes = [],
    margin = {},
    ...custom
}) {
    // Start with preset or empty base
    const base = preset && CHART_PRESETS[preset] ? { ...CHART_PRESETS[preset] } : {};

    // Auto-calculate height if not specified
    if (height === null) {
        if (preset === 'heatmap' && traces.length > 0 && traces[0].y) {
            // Heatmap: scale with number of rows for roughly square cells
            const nRows = traces[0].y.length || 10;
            height = Math.max(300, nRows * 45 + 120);  // min 300, ~45px per row + margins
        } else {
            height = 200;  // default for other chart types
        }
    }

    // Calculate legend dimensions
    const legendItems = traces.filter(t => t.showlegend !== false).length;
    const showLegend = legendPosition !== 'none' && legendItems > 0;

    let legendConfig = {};
    let legendMargin = {};

    if (showLegend) {
        // Estimate: ~3 items per row horizontal, ~18px per row
        const rows = Math.ceil(legendItems / 3);
        const legendHeight = rows * 18;

        switch (legendPosition) {
            case 'above':
                legendMargin = { t: legendHeight + 15 };
                legendConfig = { orientation: 'h', y: 1.02, yanchor: 'bottom', x: 0, font: { size: 10 } };
                break;
            case 'below':
                // Position legend below x-axis tick labels, left-aligned to use full width
                legendMargin = { b: 45 + legendHeight };
                legendConfig = { orientation: 'h', y: -0.22, yanchor: 'top', xanchor: 'left', x: 0, font: { size: 10 } };
                break;
            case 'right':
                legendMargin = { r: 120 };
                legendConfig = { orientation: 'v', x: 1.02, y: 1, font: { size: 10 } };
                break;
        }
    }

    // Merge margins: preset -> legend -> explicit overrides
    const finalMargin = {
        ...base.margin,
        ...legendMargin,
        ...margin
    };

    // Build final layout through getPlotlyLayout for theming
    return window.getPlotlyLayout({
        height,
        margin: finalMargin,
        showlegend: showLegend,
        legend: legendConfig,
        xaxis: { ...base.xaxis, ...xaxis },
        yaxis: { ...base.yaxis, ...yaxis },
        hovermode: base.hovermode,
        shapes: shapes.length > 0 ? shapes : undefined,
        ...custom
    });
}

/**
 * Create prompt/response separator shape (vertical dashed line)
 * @param {number} x - X position for the separator
 * @param {string} color - Optional color override
 */
function createSeparatorShape(x, color = null) {
    const separatorColor = color || window.getCssVar?.('--text-secondary', '#888') || '#888';
    return {
        type: 'line',
        x0: x, x1: x,
        y0: 0, y1: 1,
        yref: 'paper',
        line: { color: separatorColor, width: 2, dash: 'dash' },
        _isBase: true
    };
}

/**
 * Create current token/position highlight shape (vertical solid line)
 * @param {number} x - X position for the highlight
 * @param {string} color - Optional color override
 */
function createHighlightShape(x, color = null) {
    const highlightColor = color || window.getCssVar?.('--primary-color', '#a09f6c') || '#a09f6c';
    return {
        type: 'line',
        x0: x, x1: x,
        y0: 0, y1: 1,
        yref: 'paper',
        line: { color: highlightColor, width: 2 }
    };
}

/**
 * Render a Plotly chart with standard config
 * Wraps Plotly.newPlot with consistent options
 *
 * @param {string|HTMLElement} divId - Target div ID or element
 * @param {Array} traces - Plotly trace data
 * @param {Object} layout - Plotly layout (from buildChartLayout)
 * @param {Object} configOverrides - Optional config overrides (e.g., { staticPlot: true })
 * @returns {Promise} Plotly.newPlot promise
 */
function renderChart(divId, traces, layout, configOverrides = {}) {
    const config = { ...PLOTLY_CONFIG, ...configOverrides };
    return Plotly.newPlot(divId, traces, layout, config);
}

/**
 * Update a Plotly chart with standard config
 * Wraps Plotly.react for efficient updates
 *
 * @param {string|HTMLElement} divId - Target div ID or element
 * @param {Array} traces - Plotly trace data
 * @param {Object} layout - Plotly layout
 * @param {Object} configOverrides - Optional config overrides
 * @returns {Promise} Plotly.react promise
 */
function updateChart(divId, traces, layout, configOverrides = {}) {
    const config = { ...PLOTLY_CONFIG, ...configOverrides };
    return Plotly.react(divId, traces, layout, config);
}

// =============================================================================
// HTML Legend - Custom legend outside Plotly for reliable sizing
// =============================================================================

/**
 * Create an HTML legend for a Plotly chart
 * @param {Array} traces - Plotly traces with name, line.color, line.dash, visible, etc.
 * @param {string|HTMLElement} plotDiv - The Plotly chart div ID or element (for toggle)
 * @param {Object} options - Optional configuration
 * @param {Array<string>} options.tooltips - Tooltip text for each trace (shown on hover)
 * @param {boolean} options.hoverHighlight - Enable hover-to-highlight (dims other traces)
 * @returns {HTMLElement} Legend container element
 */
function createHtmlLegend(traces, plotDiv, options = {}) {
    const { tooltips = [], hoverHighlight = false } = options;
    const legend = document.createElement('div');
    legend.className = 'chart-legend';

    // Track hidden state for hover highlight
    const hiddenTraces = new Set();

    traces.forEach((trace, index) => {
        if (trace.showlegend === false) return;

        const item = document.createElement('span');
        item.className = 'legend-item';
        item.style.cursor = 'pointer';
        if (trace.visible === 'legendonly') {
            item.classList.add('legend-item-hidden');
            hiddenTraces.add(index);
        }

        // Add tooltip if provided
        if (tooltips[index]) {
            item.classList.add('has-tooltip');
            item.dataset.tooltip = tooltips[index];
        }

        // Color swatch
        const swatch = document.createElement('span');
        swatch.className = 'legend-color';
        const color = trace.line?.color || trace.marker?.color || '#888';
        swatch.style.background = color;

        // Label
        const label = document.createElement('span');
        label.className = 'legend-label';
        label.textContent = trace.name || `Trace ${index}`;

        item.appendChild(swatch);
        item.appendChild(label);

        // Click to toggle visibility
        item.addEventListener('click', () => {
            const plot = typeof plotDiv === 'string' ? document.getElementById(plotDiv) : plotDiv;
            if (!plot || !plot.data) return;

            const currentVisible = plot.data[index].visible;
            const newVisible = currentVisible === 'legendonly' ? true : 'legendonly';

            Plotly.restyle(plot, { visible: newVisible }, [index]);
            item.classList.toggle('legend-item-hidden', newVisible === 'legendonly');

            if (newVisible === 'legendonly') {
                hiddenTraces.add(index);
            } else {
                hiddenTraces.delete(index);
            }
        });

        legend.appendChild(item);
    });

    // Setup hover-to-highlight on the plot if requested
    if (hoverHighlight) {
        const plot = typeof plotDiv === 'string' ? document.getElementById(plotDiv) : plotDiv;
        if (plot) {
            plot.on('plotly_hover', (d) => {
                const opacities = traces.map((_, i) => {
                    if (hiddenTraces.has(i)) return 0;
                    return i === d.points[0].curveNumber ? 1.0 : 0.2;
                });
                Plotly.restyle(plot, { 'opacity': opacities });
            });
            plot.on('plotly_unhover', () => {
                const opacities = traces.map((_, i) => hiddenTraces.has(i) ? 0 : 1.0);
                Plotly.restyle(plot, { 'opacity': opacities });
            });
        }
    }

    return legend;
}

// =============================================================================
// Shared Chart Helpers
// =============================================================================

/**
 * Attach token-click handler to a Plotly chart div.
 * Clicking a point updates state.currentTokenIndex and re-renders.
 * @param {string|HTMLElement} plotDiv - Plotly chart div
 * @param {number} [startTokenIdx=0] - Offset added to the clicked x value
 */
function attachTokenClickHandler(plotDiv, startTokenIdx = 0) {
    const el = typeof plotDiv === 'string' ? document.getElementById(plotDiv) : plotDiv;
    if (!el) return;
    el.on('plotly_click', (d) => {
        const tokenIdx = Math.round(d.points[0].x) + startTokenIdx;
        if (window.state.currentTokenIndex !== tokenIdx) {
            window.state.currentTokenIndex = tokenIdx;
            window.renderPromptPicker?.();
            window.renderView?.();
        }
    });
}

/**
 * Build Plotly text annotations for heatmap cells.
 * @param {Array<Array<number>>} matrix - 2D values array [row][col]
 * @param {Array<string>} xLabels - Column labels
 * @param {Array<string>} yLabels - Row labels
 * @param {Object} [opts]
 * @param {number} [opts.threshold=0.5] - Value above which text is white
 * @param {number} [opts.fontSize=9] - Font size
 * @param {string} [opts.format] - Number format ('d' for integer, default 2 decimal)
 */
function buildHeatmapAnnotations(matrix, xLabels, yLabels, { threshold = 0.5, fontSize = 9, format } = {}) {
    const annotations = [];
    for (let i = 0; i < yLabels.length; i++) {
        for (let j = 0; j < xLabels.length; j++) {
            const val = matrix[i]?.[j];
            if (val == null) continue;
            const text = format === 'd' ? Math.round(val).toString() : val.toFixed(2);
            annotations.push({
                x: xLabels[j],
                y: yLabels[i],
                text,
                showarrow: false,
                font: { size: fontSize, color: Math.abs(val) > threshold ? 'white' : '#333' }
            });
        }
    }
    return annotations;
}

// =============================================================================
// Overlay Shape Builders - Plotly shape arrays for chart overlays
// =============================================================================

/**
 * Sentence category color definitions for overlay bands and legend.
 */
const SENTENCE_CATEGORIES = {
    setup:       { color: [140, 140, 140], label: 'Setup',       opacity: 0.10 },
    recall:      { color: [70, 130, 220],  label: 'Recall',      opacity: 0.12 },
    evaluate:    { color: [140, 140, 140], label: 'Evaluate',    opacity: 0.12,
                   valenceColors: {
                       'strong+': [40, 180, 80],  'mild+': [100, 200, 120],  'neutral': [140, 140, 140],
                       'mild-':  [220, 120, 80],  'strong-': [220, 50, 50] }},
    reframe:     { color: [230, 150, 40],  label: 'Reframe',     opacity: 0.12 },
    compare:     { color: [150, 80, 200],  label: 'Compare',     opacity: 0.12 },
    uncertainty: { color: [210, 200, 50],  label: 'Uncertainty', opacity: 0.10 },
    commit:      { color: [50, 180, 170],  label: 'Commit',      opacity: 0.14 }
};

/**
 * Build Plotly shapes for turn boundaries in multi-turn rollouts.
 * Pattern: live-chat.js:buildMessageRegionShapes()
 */
function buildTurnBoundaryShapes(turnBoundaries) {
    if (!turnBoundaries || turnBoundaries.length === 0) return [];
    const shapes = [];
    const roleColors = {
        system:    { cssVar: '--chart-10', fallback: '#94d82d', opacity: 0.06 },
        user:      { cssVar: '--chart-1',  fallback: '#4a9eff', opacity: 0.12 },
        assistant: { cssVar: '--chart-3',  fallback: '#51cf66', opacity: 0.06 },
        tool:      { cssVar: '--chart-6',  fallback: '#ff922b', opacity: 0.12 },
    };
    for (const turn of turnBoundaries) {
        if (turn.token_start === turn.token_end) continue;
        const cfg = roleColors[turn.role] || roleColors.assistant;
        const hex = window.getCssVar?.(cfg.cssVar, cfg.fallback) || cfg.fallback;
        shapes.push({
            type: 'rect',
            x0: turn.token_start - 0.5,
            x1: turn.token_end - 0.5,
            y0: 0, y1: 1, yref: 'paper',
            fillcolor: window.hexToRgba?.(hex, cfg.opacity) || `rgba(128,128,128,${cfg.opacity})`,
            line: { width: 0 },
            layer: 'below'
        });
    }
    return shapes;
}

/**
 * Build Plotly shapes for sentence boundaries with cue_p gradient coloring.
 * Used for thought branches / unfaithful CoT analysis.
 * Color: blue (cue_p=0) -> red (cue_p=1) continuous gradient.
 * @param {Array} sentenceBoundaries - [{sentence_num, token_start, token_end, cue_p}, ...]
 * @param {number} nPromptTokens - offset for response-relative positions
 */
function buildSentenceBoundaryShapes(sentenceBoundaries, nPromptTokens, y0 = 0, y1 = 1) {
    if (!sentenceBoundaries || sentenceBoundaries.length === 0) return [];
    const shapes = [];
    for (const sent of sentenceBoundaries) {
        if (sent.token_start === sent.token_end) continue;
        const cueP = sent.cue_p ?? 0;
        // Interpolate blue (0,100,255) -> red (255,50,50)
        const r = Math.round(0 + cueP * 255);
        const g = Math.round(100 - cueP * 50);
        const b = Math.round(255 - cueP * 205);
        const opacity = 0.08 + cueP * 0.12; // 0.08 at cue_p=0, 0.20 at cue_p=1
        shapes.push({
            type: 'rect',
            x0: (nPromptTokens + sent.token_start) - 0.5,
            x1: (nPromptTokens + sent.token_end) - 0.5,
            y0, y1, yref: 'paper',
            fillcolor: `rgba(${r},${g},${b},${opacity})`,
            line: { width: 0 },
            layer: 'below'
        });
    }
    return shapes;
}

/**
 * Build Plotly shapes for sentence category overlay bands.
 * Colors each sentence by its category, with evaluate using valence-dependent colors.
 */
function buildSentenceCategoryShapes(categoryData, nPromptTokens, y0 = 0, y1 = 1) {
    if (!categoryData || categoryData.length === 0) return [];
    const shapes = [];
    for (const sent of categoryData) {
        if (sent.token_start === sent.token_end) continue;
        const catDef = SENTENCE_CATEGORIES[sent.category];
        if (!catDef) continue;

        let rgb;
        if (sent.category === 'evaluate' && sent.valence && catDef.valenceColors?.[sent.valence]) {
            rgb = catDef.valenceColors[sent.valence];
        } else {
            rgb = catDef.color;
        }

        shapes.push({
            type: 'rect',
            x0: (nPromptTokens + sent.token_start) - 0.5,
            x1: (nPromptTokens + sent.token_end) - 0.5,
            y0, y1, yref: 'paper',
            fillcolor: `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${catDef.opacity})`,
            line: { width: 0 },
            layer: 'below'
        });
    }
    return shapes;
}

/**
 * Build combined overlay shapes respecting toggle state.
 * Both on: cue_p bottom half, category top half. One on: full height.
 */
function buildOverlayShapes(sentenceBoundaries, categoryData, nPromptTokens) {
    const showCueP = window.state.showCuePOverlay;
    const showCategory = window.state.showCategoryOverlay;
    const hasCategoryData = categoryData && categoryData.length > 0;
    const shapes = [];

    if (showCueP && showCategory && hasCategoryData) {
        shapes.push(...buildSentenceBoundaryShapes(sentenceBoundaries, nPromptTokens, 0, 0.5));
        shapes.push(...buildSentenceCategoryShapes(categoryData, nPromptTokens, 0.5, 1));
    } else if (showCueP) {
        shapes.push(...buildSentenceBoundaryShapes(sentenceBoundaries, nPromptTokens));
    } else if (showCategory && hasCategoryData) {
        shapes.push(...buildSentenceCategoryShapes(categoryData, nPromptTokens));
    }

    return shapes;
}

// Exports
window.PLOTLY_CONFIG = PLOTLY_CONFIG;
window.CHART_PRESETS = CHART_PRESETS;
window.SENTENCE_CATEGORIES = SENTENCE_CATEGORIES;
window.buildChartLayout = buildChartLayout;
window.createSeparatorShape = createSeparatorShape;
window.createHighlightShape = createHighlightShape;
window.renderChart = renderChart;
window.updateChart = updateChart;
window.createHtmlLegend = createHtmlLegend;
window.attachTokenClickHandler = attachTokenClickHandler;
window.buildHeatmapAnnotations = buildHeatmapAnnotations;
window.buildTurnBoundaryShapes = buildTurnBoundaryShapes;
window.buildOverlayShapes = buildOverlayShapes;
