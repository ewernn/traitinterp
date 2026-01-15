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
 * @returns {HTMLElement} Legend container element
 */
function createHtmlLegend(traces, plotDiv) {
    const legend = document.createElement('div');
    legend.className = 'chart-legend';

    traces.forEach((trace, index) => {
        if (trace.showlegend === false) return;

        const item = document.createElement('span');
        item.className = 'legend-item';
        if (trace.visible === 'legendonly') {
            item.classList.add('legend-item-hidden');
        }

        // Line sample
        const line = document.createElement('span');
        line.className = 'legend-line';

        const color = trace.line?.color || trace.marker?.color || '#888';
        const dash = trace.line?.dash || 'solid';

        // Convert Plotly dash to CSS
        let borderStyle = 'solid';
        if (dash === 'dash') borderStyle = 'dashed';
        else if (dash === 'dot') borderStyle = 'dotted';
        else if (dash === 'dashdot') borderStyle = 'dashed'; // CSS doesn't have dashdot

        line.style.borderTopColor = color;
        line.style.borderTopStyle = borderStyle;

        // Label
        const label = document.createElement('span');
        label.className = 'legend-label';
        label.textContent = trace.name || `Trace ${index}`;

        item.appendChild(line);
        item.appendChild(label);

        // Click to toggle visibility
        item.addEventListener('click', () => {
            const plot = typeof plotDiv === 'string' ? document.getElementById(plotDiv) : plotDiv;
            if (!plot || !plot.data) return;

            const currentVisible = plot.data[index].visible;
            const newVisible = currentVisible === 'legendonly' ? true : 'legendonly';

            Plotly.restyle(plot, { visible: newVisible }, [index]);
            item.classList.toggle('legend-item-hidden', newVisible === 'legendonly');
        });

        legend.appendChild(item);
    });

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
