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

// Exports
window.PLOTLY_CONFIG = PLOTLY_CONFIG;
window.CHART_PRESETS = CHART_PRESETS;
window.buildChartLayout = buildChartLayout;
window.createSeparatorShape = createSeparatorShape;
window.createHighlightShape = createHighlightShape;
window.renderChart = renderChart;
window.updateChart = updateChart;
window.createHtmlLegend = createHtmlLegend;
