/**
 * Vector Geometry — method/layer controls + PCA scatter + ranked neighbors.
 *
 * Loads precomputed vector_geometry.json. Method/layer changes also re-render
 * the logit lens table so its layer selection tracks the slider.
 *
 * Input:  vector_geometry.json (per-method-per-layer cosine-sim matrix + PCA coords)
 * Output: rendered scatter + neighbors panels in #vector-geometry-container
 * Usage:  import { renderVectorGeometrySection } from './section-vector-geometry.js';
 */

import { fetchJSON, escapeHtml } from '../../core/utils.js';
import { getDisplayName, getChartColors, displayLayer } from '../../core/display.js';
import { buildChartLayout, renderChart } from '../../core/charts.js';
import { renderRunHint } from '../../core/ui.js';
import { renderStyledSelect, wireStyledSelect } from '../../components/styled-select.js';
import { extractionState } from './extraction-data.js';
import { renderLogitLensFromCache } from './section-logit-lens.js';

/**
 * Render the Vector Geometry subsection: method/layer controls + scatter + neighbors.
 * Loads precomputed vector_geometry.json; shows a run-hint if missing.
 */
async function renderVectorGeometrySection() {
    const container = document.getElementById('vector-geometry-container');
    if (!container) return;

    const data = await fetchJSON(window.paths.vectorGeometry());
    if (!data || !data.data || !data.methods || data.methods.length === 0) {
        const expName = window.state.experimentData?.name || 'your_experiment';
        container.innerHTML = renderRunHint(
            'No vector geometry data',
            `python analysis/vectors/trait_vector_geometry.py --experiment ${expName}`
        );
        return;
    }

    // Initialize defaults if not already set
    if (!extractionState.vgMethod || !data.methods.includes(extractionState.vgMethod)) {
        extractionState.vgMethod = data.methods[0];
    }
    const layersForMethod = Object.keys(data.data[extractionState.vgMethod] || {}).map(Number).sort((a, b) => a - b);
    if (layersForMethod.length === 0) {
        container.innerHTML = `<div class="info">No layers found for method <code>${extractionState.vgMethod}</code>.</div>`;
        return;
    }
    if (extractionState.vgLayer == null || !layersForMethod.includes(extractionState.vgLayer)) {
        extractionState.vgLayer = layersForMethod[Math.floor(layersForMethod.length / 2)];
    }

    container.innerHTML = `
        <div class="vg-controls">
            <div class="vg-control-group">
                <span class="cb-label">Method:</span>
                <div id="vg-method-select-wrap"></div>
            </div>
            <div class="vg-control-group">
                <span class="cb-label">Layer:</span>
                <input type="range" id="vg-layer-slider"
                       min="${layersForMethod[0]}" max="${layersForMethod[layersForMethod.length - 1]}"
                       step="1" value="${extractionState.vgLayer}"
                       style="width: 200px; accent-color: var(--form-accent);">
                <span class="cb-label" id="vg-layer-label">L${displayLayer(extractionState.vgLayer)}</span>
            </div>
        </div>
        <div class="vg-panels">
            <div id="vg-scatter" class="vg-panel-scatter"></div>
            <div id="vg-neighbors" class="vg-panel-neighbors"></div>
        </div>
    `;

    // Render the styled-select for methods
    const methodWrap = document.getElementById('vg-method-select-wrap');
    methodWrap.innerHTML = renderStyledSelect({
        id: 'vg-method-select',
        options: data.methods.map(m => ({ value: m, label: m })),
        selected: extractionState.vgMethod,
        onChange: (val) => {
            extractionState.vgMethod = val;
            const layers = Object.keys(data.data[extractionState.vgMethod] || {}).map(Number).sort((a, b) => a - b);
            if (!layers.includes(extractionState.vgLayer)) {
                extractionState.vgLayer = layers[Math.floor(layers.length / 2)];
            }
            const slider = document.getElementById('vg-layer-slider');
            if (slider) {
                slider.min = layers[0];
                slider.max = layers[layers.length - 1];
                slider.value = extractionState.vgLayer;
                document.getElementById('vg-layer-label').textContent = `L${displayLayer(extractionState.vgLayer)}`;
            }
            extractionState.vgSelectedTrait = null;
            renderVectorGeometryPanels(data);
            renderLogitLensFromCache();  // logit-lens tracks VG's method+layer
        },
    });
    wireStyledSelect(methodWrap);

    // Wire the layer slider — snap to nearest available layer
    const slider = document.getElementById('vg-layer-slider');
    const label = document.getElementById('vg-layer-label');
    slider.addEventListener('input', () => {
        const layers = Object.keys(data.data[extractionState.vgMethod] || {}).map(Number).sort((a, b) => a - b);
        const requested = parseInt(slider.value);
        // Snap to nearest existing layer (handles sparse coverage)
        const nearest = layers.reduce((best, l) =>
            Math.abs(l - requested) < Math.abs(best - requested) ? l : best, layers[0]);
        if (nearest !== extractionState.vgLayer) {
            extractionState.vgLayer = nearest;
            slider.value = extractionState.vgLayer;
            label.textContent = `L${displayLayer(extractionState.vgLayer)}`;
            renderVectorGeometryPanels(data);
            renderLogitLensFromCache();  // logit-lens tracks VG's method+layer
        } else {
            label.textContent = `L${displayLayer(extractionState.vgLayer)}`;
        }
    });

    renderVectorGeometryPanels(data);
    renderLogitLensFromCache();  // if LL already loaded, re-render it now that vgLayer is set
}

/** Render both scatter + neighbors for the currently selected (method, layer). */
function renderVectorGeometryPanels(data) {
    const slice = data.data[extractionState.vgMethod]?.[String(extractionState.vgLayer)];
    if (!slice) return;
    // Default-select first trait so neighbors panel has something to show.
    if (!extractionState.vgSelectedTrait || !slice.traits.includes(extractionState.vgSelectedTrait)) {
        extractionState.vgSelectedTrait = slice.traits[0];
    }
    renderVectorGeometryScatter(slice, data);
    renderVectorGeometryNeighbors(slice, data);
}

/** Scatter of trait vectors in PCA-2D space. */
function renderVectorGeometryScatter(slice, data) {
    const palette = getChartColors();
    const colors = slice.traits.map((_, i) => palette[i % palette.length]);
    const sizes = slice.traits.map(t => t === extractionState.vgSelectedTrait ? 14 : 8);
    const borders = slice.traits.map(t => t === extractionState.vgSelectedTrait ? 2 : 0);

    const trace = {
        type: 'scatter',
        mode: 'markers+text',
        x: slice.coords_2d.map(c => c[0]),
        y: slice.coords_2d.map(c => c[1]),
        text: slice.traits.map(getDisplayName),
        textposition: 'top center',
        textfont: { size: 9, color: '#aaa' },
        marker: {
            size: sizes,
            color: colors,
            line: { width: borders, color: '#eee' },
        },
        customdata: slice.traits,
        hovertemplate: '<b>%{text}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>',
    };

    const layout = buildChartLayout({
        preset: null,
        traces: [trace],
        height: 360,
        legendPosition: 'none',
        xaxis: { title: 'PC1', zeroline: true, showgrid: false },
        yaxis: { title: 'PC2', zeroline: true, showgrid: false },
        margin: { l: 50, r: 20, t: 20, b: 40 },
    });

    renderChart('vg-scatter', [trace], layout);

    // Click a point → update selection + re-render both panels
    const plotDiv = document.getElementById('vg-scatter');
    plotDiv.on('plotly_click', (ev) => {
        const pt = ev.points?.[0];
        if (!pt || !pt.customdata) return;
        extractionState.vgSelectedTrait = pt.customdata;
        renderVectorGeometryPanels(data);
    });
}

/** Ranked neighbor list for the selected trait at the current slice. */
function renderVectorGeometryNeighbors(slice, data) {
    const panel = document.getElementById('vg-neighbors');
    if (!panel) return;

    const { traits, cos_sim: cos } = slice;
    const idx = traits.indexOf(extractionState.vgSelectedTrait);
    if (idx < 0) { panel.innerHTML = ''; return; }

    const pairs = traits
        .map((t, i) => ({ trait: t, sim: cos[idx][i] }))
        .filter(p => p.trait !== extractionState.vgSelectedTrait)
        .sort((a, b) => b.sim - a.sim);

    const row = (p) => {
        const v = p.sim;
        const cls = v > 0.3 ? 'pos' : v < -0.1 ? 'neg' : '';
        return `
            <div class="vg-neighbor-row" data-trait="${p.trait}">
                <span class="vg-neighbor-sim ${cls}">${v >= 0 ? '+' : ''}${v.toFixed(3)}</span>
                <span class="vg-neighbor-name">${escapeHtml(getDisplayName(p.trait))}</span>
            </div>
        `;
    };

    panel.innerHTML = `
        <div class="vg-neighbors-header">
            <span class="vg-neighbors-label">Cosine sim to</span>
            <span class="vg-neighbors-selected">${escapeHtml(getDisplayName(extractionState.vgSelectedTrait))}</span>
        </div>
        <div class="vg-neighbors-list">${pairs.map(row).join('')}</div>
    `;

    panel.querySelectorAll('.vg-neighbor-row').forEach(el => {
        el.addEventListener('click', () => {
            extractionState.vgSelectedTrait = el.dataset.trait;
            renderVectorGeometryPanels(data);
        });
    });
}

export { renderVectorGeometrySection };
