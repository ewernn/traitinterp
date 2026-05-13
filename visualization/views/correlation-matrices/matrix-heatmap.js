// Plotly heatmap component for one correlation matrix.
//
// Input:  matrix (object[A][B] -> number), bias_ids (sorted), bias_short_names map,
//         opts { sortMode, hideDiagonal, onCellHover, onCellClick }
// Output: renders a 39x39-ish heatmap into the given container.
// Usage:  view.js -> renderMatrixHeatmap(div, { matrix, biasIds, biasShortNames, ... })
//
// Styling decisions:
//   - Diverging colorscale (RdBu-ish, mirrored around 0): negative=blue, zero=white, positive=red.
//     We center zmid=0 and pick zmin/zmax = +/- |max(off-diag)| so the diagonal (typically the
//     largest values) doesn't blow out the off-diagonal contrast. With hideDiagonal=true the
//     diagonal cells are masked to NaN (they vanish + get a black border via shapes).
//   - Cells are squareish and 39x39 fits comfortably below ~750px.
//   - Hover/click forwarded so the parent view can show "top traits" + value details.

import { renderChart, buildChartLayout } from '../../core/charts.js';

// Diverging palette — blue (negative) -> white (zero) -> red (positive).
// Plotly's built-in 'RdBu' is REVERSED relative to research convention (red=low),
// so we hand-roll one matching the standard "warm = positive" expectation.
const DIVERGING = [
    [0.0, '#2166ac'],   // strong negative
    [0.25, '#67a9cf'],
    [0.5, '#f7f7f7'],   // zero
    [0.75, '#ef8a62'],
    [1.0, '#b2182b'],   // strong positive
];

/**
 * Sort modes for biases:
 *   - 'id'         : ascending by numeric bias id (default; matches index.json order).
 *   - 'name'       : alphabetical by short_name.
 *   - 'diagonal'   : descending by matrix[A][A] (self-similarity).
 */
function sortBiases(biasIds, mode, matrix, biasShortNames) {
    const ids = [...biasIds];
    if (mode === 'id') {
        ids.sort((a, b) => a - b);
    } else if (mode === 'name') {
        ids.sort((a, b) => {
            const na = biasShortNames[String(a)] || '';
            const nb = biasShortNames[String(b)] || '';
            return na.localeCompare(nb);
        });
    } else if (mode === 'diagonal') {
        ids.sort((a, b) => {
            const va = matrix[String(a)]?.[String(a)] ?? -Infinity;
            const vb = matrix[String(b)]?.[String(b)] ?? -Infinity;
            return vb - va;
        });
    } else {
        throw new Error(`Unknown sort mode: ${mode}`);
    }
    return ids;
}

function buildLabels(biasIds, biasShortNames) {
    return biasIds.map(id => {
        const short = biasShortNames[String(id)] ?? `bias_${id}`;
        return `${id} ${short}`;
    });
}

/**
 * Build the z-matrix (rows x cols) from the asymmetric matrix object.
 * Convention: row = bias A (the "from"), column = bias B (the "to").
 * matrix[A][B] is read as rows[A_idx][B_idx].
 *
 * If `hideDiagonal` is true, diagonal cells are set to null (Plotly draws gaps).
 */
function buildZ(biasIds, matrix, hideDiagonal) {
    const z = [];
    for (const a of biasIds) {
        const row = [];
        for (const b of biasIds) {
            if (hideDiagonal && a === b) {
                row.push(null);
            } else {
                const v = matrix[String(a)]?.[String(b)];
                row.push(typeof v === 'number' ? v : null);
            }
        }
        z.push(row);
    }
    return z;
}

/**
 * Compute symmetric color range (+/- |max| of off-diagonal cells).
 * If hideDiagonal=true and diagonal-bordering shapes only, off-diagonal already
 * is the only thing that matters anyway.
 */
function computeColorRange(biasIds, matrix) {
    let absMax = 0;
    for (const a of biasIds) {
        for (const b of biasIds) {
            if (a === b) continue;
            const v = matrix[String(a)]?.[String(b)];
            if (typeof v === 'number' && Math.abs(v) > absMax) absMax = Math.abs(v);
        }
    }
    if (absMax === 0) absMax = 1;
    return { zmin: -absMax, zmax: absMax };
}

/**
 * Diagonal markers: black-bordered squares around each [i,i] cell, drawn as
 * Plotly shapes referencing axis coords (0-indexed cell centers).
 */
function buildDiagonalShapes(n) {
    const shapes = [];
    for (let i = 0; i < n; i++) {
        shapes.push({
            type: 'rect',
            xref: 'x', yref: 'y',
            x0: i - 0.5, x1: i + 0.5,
            y0: i - 0.5, y1: i + 0.5,
            line: { color: '#000', width: 1.5 },
            fillcolor: 'rgba(0,0,0,0)',
            layer: 'above',
        });
    }
    return shapes;
}

/**
 * Render the heatmap into `div`. Returns the array of bias ids actually plotted
 * (ordered by current sort mode), so the caller can map click events back to ids.
 */
function renderMatrixHeatmap(div, {
    matrix,
    biasIds,
    biasShortNames,
    sortMode = 'id',
    hideDiagonal = false,
    onCellHover = null,
    onCellClick = null,
}) {
    const sorted = sortBiases(biasIds, sortMode, matrix, biasShortNames);
    const labels = buildLabels(sorted, biasShortNames);
    const z = buildZ(sorted, matrix, hideDiagonal);
    const { zmin, zmax } = computeColorRange(sorted, matrix);

    const trace = {
        z,
        x: labels,
        y: labels,
        type: 'heatmap',
        colorscale: DIVERGING,
        zmid: 0,
        zmin,
        zmax,
        hoverongaps: false,
        // We'll handle hover via the onCellHover callback; this gives Plotly's
        // tooltip something basic so it's not totally empty if the caller skips.
        hovertemplate: 'A: %{y}<br>B: %{x}<br>value: %{z:.5f}<extra></extra>',
        colorbar: {
            thickness: 12,
            len: 0.85,
            tickfont: { size: 10 },
            title: { text: 'value', font: { size: 10 } },
        },
    };

    const n = sorted.length;
    const cellSize = 16;                // px per cell; tunable
    const labelMargin = 180;            // wide left margin for "id short_name"
    const height = Math.max(500, n * cellSize + 200);

    const layout = buildChartLayout({
        preset: 'heatmap',
        traces: [trace],
        height,
        legendPosition: 'none',
        xaxis: {
            tickangle: -50,
            tickfont: { size: 9 },
            automargin: true,
            scaleanchor: 'y',
            constrain: 'domain',
        },
        yaxis: {
            tickfont: { size: 9 },
            automargin: true,
            autorange: 'reversed',     // top-down so bias_id 1 is at top
        },
        shapes: buildDiagonalShapes(n),
        margin: { l: labelMargin, r: 80, t: 30, b: labelMargin },
    });

    renderChart(div, [trace], layout).then(() => {
        const plot = typeof div === 'string' ? document.getElementById(div) : div;
        if (!plot) return;

        if (onCellHover) {
            plot.on('plotly_hover', (data) => {
                const p = data.points?.[0];
                if (!p) return;
                const rowIdx = sorted.indexOf(_idFromLabel(p.y));
                const colIdx = sorted.indexOf(_idFromLabel(p.x));
                if (rowIdx < 0 || colIdx < 0) return;
                onCellHover({
                    aId: sorted[rowIdx],
                    bId: sorted[colIdx],
                    value: p.z,
                });
            });
        }
        if (onCellClick) {
            plot.on('plotly_click', (data) => {
                const p = data.points?.[0];
                if (!p) return;
                const rowIdx = sorted.indexOf(_idFromLabel(p.y));
                const colIdx = sorted.indexOf(_idFromLabel(p.x));
                if (rowIdx < 0 || colIdx < 0) return;
                onCellClick({
                    aId: sorted[rowIdx],
                    bId: sorted[colIdx],
                    value: p.z,
                });
            });
        }
    });

    return sorted;
}

// Labels look like "40 movies_similar"; pull the leading id.
function _idFromLabel(label) {
    if (typeof label !== 'string') return NaN;
    const m = label.match(/^(\d+)\s/);
    return m ? parseInt(m[1], 10) : NaN;
}

export { renderMatrixHeatmap };
