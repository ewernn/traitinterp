// HTML Legend - Custom legend outside Plotly for reliable sizing
//
// Usage:
//   const legendEl = createHtmlLegend(traces, plotDiv);
//   chartWrapper.appendChild(legendEl);
//
// The legend renders as normal DOM elements that flow naturally,
// avoiding Plotly's legend overflow issues. Click items to toggle traces.

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

/**
 * Update legend visibility state to match plot
 * Call after Plotly.restyle if visibility changed externally
 * @param {HTMLElement} legendEl - Legend element from createHtmlLegend
 * @param {Array} traces - Current trace data (plot.data)
 */
function syncLegendVisibility(legendEl, traces) {
    const items = legendEl.querySelectorAll('.legend-item');
    let traceIdx = 0;

    items.forEach((item) => {
        // Skip traces that had showlegend: false
        while (traceIdx < traces.length && traces[traceIdx].showlegend === false) {
            traceIdx++;
        }
        if (traceIdx >= traces.length) return;

        const hidden = traces[traceIdx].visible === 'legendonly';
        item.classList.toggle('legend-item-hidden', hidden);
        traceIdx++;
    });
}

// Exports
window.createHtmlLegend = createHtmlLegend;
window.syncLegendVisibility = syncLegendVisibility;
