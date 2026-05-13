// Shared per-token projection chart.
//
// Used by:
//   - inference view (chart-token-trajectory.js) — single pid, optional diff
//   - annotation-browser strip (projection-strip.js) — single pid, optional cohort overlay
//   - future: cluster-overlay view (multiple biases averaged → cohort.perTrait)
//
// What the component does:
//   1. Run projection-transform pipeline per trait (per variant if diff).
//   2. Optionally diff(A, B) with the outer combinator.
//   3. Render the result, either:
//        - 'lines' renderer: Plotly line-per-trait (with optional cohort
//          mean + ±std band overlay)
//        - 'bars'  renderer: CSS bar-grid (per-token color cells, K rows)
//   4. Return per-trait Float32Array values so callers can attach
//      decorators (velocity overlay, top spans, etc.) afterwards.
//
// What the component does NOT do:
//   - fetch data (caller pre-fetches via core/projection-store.js)
//   - rank/filter traits (caller pre-ranks via core/projection-ranking.js)
//   - read window.state (callers pass everything as opts)
//   - render velocity, cue_p, top spans, sentence categories
//     (those are post-render decorators or sibling charts in the wrapper)
//
// Input shape (per trait):
//   {
//     rawProj:    { prompt: number[], response: number[] }
//     tokenNorms?: { prompt: number[], response: number[] }
//     mddData?:    object
//     baseline?:   number
//     metadata?:   { vector_source?, layer?, ... }   // pass-through, used by colorFn
//   }
//
// Diff mode: pass `baselineTraitData` (instruct, say) alongside `traitData`
// (rm_lora, say). The component runs the pipeline on each variant
// independently, then subtracts. Token-alignment mismatch throws.
//
// Cohort overlay: pass `cohort.perTrait[trait] = { mean, std, n }` where
// mean/std are Float32Arrays of length cohort.totalLen (prompt+response,
// or onset-aligned ±W). The component draws cohort as a faint fill band
// behind each per-pid line. Useful for cluster-mean and bias-mean overlays.
//
// Usage:
//   import { renderPerTokenProjectionChart } from 'components/per-token-projection-chart.js';
//   const out = renderPerTokenProjectionChart('plot-div', {
//     traitData: { ... },
//     traitOrder: ['emotion_set/shame', ...],
//     promptLen: 145,
//     stages: { scale: 'response_scale', center: 'on' },
//     renderer: 'lines',
//   });

import { runPerVariant, diff as diffFrames } from '../core/projection-transform.js';
import {
    buildChartLayout,
    renderChart,
    updateChart,
    createSeparatorShape,
    createHighlightShape,
    attachTokenClickHandler,
    attachSortedHover,
    createHtmlLegend,
    LINE_SPLINE,
} from '../core/charts.js';
import { getChartColors, getDisplayName, getCssVar } from '../core/display.js';

// ─── Top-level renderer ────────────────────────────────────────────────

/**
 * Main entry. Returns:
 *   {
 *     traitActivations: { [trait]: Float32Array },   // post-pipeline (post-diff if applicable) values
 *     displayedTraits: string[],                     // input order, post-filter
 *     plotDiv: HTMLElement | null,                   // present for 'lines' renderer
 *   }
 *
 * For renderer='bars', a CSS bar-grid is written into the divId; no Plotly.
 * For renderer='lines', a Plotly chart is rendered with optional cohort
 * overlay band per trait.
 */
function renderPerTokenProjectionChart(divId, opts) {
    const cfg = _normalizeOpts(opts);
    const transformed = _transformAll(cfg);
    if (cfg.renderer === 'bars') {
        return _renderBars(divId, cfg, transformed);
    }
    return _renderLines(divId, cfg, transformed);
}

// ─── Options + transform ───────────────────────────────────────────────

function _normalizeOpts(opts) {
    const traitData = opts.traitData || {};
    const traitOrder = opts.traitOrder || Object.keys(traitData);
    return {
        traitData,
        baselineTraitData: opts.baselineTraitData || null,
        diffOrder: opts.diffOrder || 'A-B',
        traitOrder,

        cohort: opts.cohort || null,             // {label, perTrait, alignedToOnset, ...}
        showCohortBand: opts.showCohortBand !== false,
        showCurrentLine: opts.showCurrentLine !== false,

        promptTokens: opts.promptTokens || [],
        responseTokens: opts.responseTokens || [],
        promptLen: opts.promptLen ?? (opts.promptTokens?.length || 0),
        isRollout: !!opts.isRollout,

        stages: opts.stages || {},

        highlightTokenIdx: opts.highlightTokenIdx ?? null,
        annotationTokenRanges: opts.annotationTokenRanges || [],
        turnBoundaries: opts.turnBoundaries || [],
        extraShapes: opts.extraShapes || [],

        renderer: opts.renderer || 'lines',
        yScaleWindow: opts.yScaleWindow || null,   // [flatStart, flatEnd] — auto y-range scopes to this
        hiddenTraits: opts.hiddenTraits || null,   // Set<trait> — initial 'legendonly' state for the per-pid line
        onToggleTrait: opts.onToggleTrait || null, // (trait, hidden:boolean) => void — fires on legend click
        height: opts.height ?? 400,
        showLegend: opts.showLegend !== false,
        hoverTooltipId: opts.hoverTooltipId || 'projection-chart-hover',
        colorFn: opts.colorFn || null,
        traitDisplayName: opts.traitDisplayName || (t => getDisplayName(t)),
        legendTooltipFn: opts.legendTooltipFn || null,
        yAxisTitle: opts.yAxisTitle || null,
        yAxisRange: opts.yAxisRange || null,
        startTokenIdx: opts.startTokenIdx ?? 0,
        onTokenClick: opts.onTokenClick || null,

        // window mode (bars renderer): subset of tokens to display
        window: opts.window || null,             // { xStart, xEnd } in flat (prompt+response) coords
    };
}

/**
 * Run the per-variant pipeline (and diff combinator if baselineTraitData
 * present) for every trait in traitOrder. Drops traits with missing or
 * mismatched data, returning the surviving list + their post-pipeline
 * Float32Arrays.
 */
function _transformAll(cfg) {
    const out = { values: new Map(), displayed: [], skipped: [] };
    const isDiff = !!cfg.baselineTraitData;

    for (const trait of cfg.traitOrder) {
        const a = cfg.traitData[trait];
        if (!a) { out.skipped.push({ trait, reason: 'no primary data' }); continue; }

        let frame;
        try {
            const frameA = runPerVariant(_inputFromTrait(a, cfg), cfg.stages);
            if (isDiff) {
                const b = cfg.baselineTraitData[trait];
                if (!b) { out.skipped.push({ trait, reason: 'no baseline data' }); continue; }
                const frameB = runPerVariant(_inputFromTrait(b, cfg), cfg.stages);
                frame = diffFrames(frameA, frameB, cfg.diffOrder);
            } else {
                frame = frameA;
            }
        } catch (e) {
            out.skipped.push({ trait, reason: e.message });
            continue;
        }

        out.values.set(trait, frame.values);
        out.displayed.push(trait);
    }
    return out;
}

function _inputFromTrait(traitEntry, cfg) {
    return {
        rawProj: traitEntry.rawProj,
        promptLen: cfg.promptLen,
        responseLen: cfg.responseTokens.length || undefined,
        tokenNorms: traitEntry.tokenNorms,
        mddData: traitEntry.mddData,
        baseline: traitEntry.baseline,
        isRollout: cfg.isRollout,
    };
}

// ─── Lines renderer (Plotly) ───────────────────────────────────────────

function _renderLines(divId, cfg, t) {
    const plotDiv = typeof divId === 'string' ? document.getElementById(divId) : divId;
    if (!plotDiv) return { traitActivations: {}, displayedTraits: [], plotDiv: null };

    const traces = [];
    const traitActivations = {};

    // Per-trait line + optional cohort band.
    for (let idx = 0; idx < t.displayed.length; idx++) {
        const trait = t.displayed[idx];
        const values = t.values.get(trait);
        traitActivations[trait] = values;

        const color = cfg.colorFn
            ? cfg.colorFn(trait, idx, cfg.traitData[trait])
            : getChartColors()[idx % 10];

        // Cohort mean ± std band (drawn FIRST so it sits behind the line).
        if (cfg.cohort?.perTrait?.[trait] && cfg.showCohortBand) {
            const c = cfg.cohort.perTrait[trait];
            const mean = c.mean;
            const std = c.std;
            const n = mean.length;
            const xs = Array.from({ length: n }, (_, i) => i + cfg.startTokenIdx);
            const upper = new Float32Array(n);
            const lower = new Float32Array(n);
            for (let i = 0; i < n; i++) { upper[i] = mean[i] + (std?.[i] || 0); lower[i] = mean[i] - (std?.[i] || 0); }
            // Lower bound trace (invisible)
            traces.push({
                x: xs, y: Array.from(lower),
                type: 'scatter', mode: 'lines',
                line: { color: color, width: 0 },
                showlegend: false, hoverinfo: 'skip',
                _cohortBand: trait,
            });
            // Upper bound, fill to lower → produces the std band.
            traces.push({
                x: xs, y: Array.from(upper),
                type: 'scatter', mode: 'lines',
                line: { color: color, width: 0 },
                fill: 'tonexty',
                fillcolor: _withAlpha(color, 0.12),
                showlegend: false, hoverinfo: 'skip',
                _cohortBand: trait,
            });
            // Cohort mean as a thinner dashed line.
            traces.push({
                x: xs, y: Array.from(mean),
                type: 'scatter', mode: 'lines',
                line: { color: color, width: 1, dash: 'dot', ...LINE_SPLINE },
                name: `${cfg.traitDisplayName(trait)} (cohort μ)`,
                showlegend: false, hoverinfo: 'skip',
                _cohortMean: trait,
            });
        }

        // Current-pid line.
        if (cfg.showCurrentLine !== false) {
            const xs = Array.from({ length: values.length }, (_, i) => i + cfg.startTokenIdx);
            const useMarkers = values.length <= 2000;
            const isHidden = cfg.hiddenTraits?.has(trait);
            traces.push({
                x: xs,
                y: Array.from(values),
                type: 'scatter',
                mode: useMarkers ? 'lines+markers' : 'lines',
                name: cfg.traitDisplayName(trait),
                line: { color: color, width: 1.5, ...LINE_SPLINE },
                ...(useMarkers ? { marker: { size: 2, color } } : {}),
                visible: isHidden ? 'legendonly' : true,
                hoverinfo: 'none',
                _displayName: cfg.traitDisplayName(trait),
                _traitKey: trait,            // raw key for onToggleTrait callback
            });
        }
    }

    // ── Shapes: prompt/response separator + highlight + extras ──
    const shapes = [];
    if (!cfg.isRollout && cfg.promptLen > 0) {
        shapes.push({ ...createSeparatorShape(cfg.promptLen - cfg.startTokenIdx - 0.5), _isBase: true });
    }
    if (cfg.highlightTokenIdx != null) {
        const hx = cfg.highlightTokenIdx - cfg.startTokenIdx;
        shapes.push(createHighlightShape(hx));
    }
    for (const [start, end] of cfg.annotationTokenRanges) {
        shapes.push({
            type: 'rect',
            x0: (cfg.promptLen - cfg.startTokenIdx) + start - 0.5,
            x1: (cfg.promptLen - cfg.startTokenIdx) + end - 0.5,
            y0: 0, y1: 1, yref: 'paper',
            fillcolor: 'rgba(255, 180, 60, 0.12)',
            line: { width: 0 },
            layer: 'below',
            _isBase: true,
        });
    }
    shapes.push(...cfg.extraShapes);

    // ── Tick labels ──
    const allTokens = [...cfg.promptTokens, ...cfg.responseTokens];
    const displayTokens = allTokens.slice(cfg.startTokenIdx);
    const tickStep = Math.max(10, Math.floor(displayTokens.length / 80));
    const tickVals = [];
    const tickText = [];
    for (let i = 0; i < displayTokens.length; i += tickStep) {
        tickVals.push(i);
        tickText.push(displayTokens[i]);
    }

    // ── Y-axis range ──
    // If yScaleWindow is given, scope min/max to that flat-token range; otherwise
    // scan the full trace (skipping the first few prompt tokens to avoid BOS-spike
    // dominating the auto-scale). Also incorporate cohort mean ± std when present
    // so the band stays inside the visible y-range.
    let yAxisConfig = { title: cfg.yAxisTitle, zeroline: true, zerolinewidth: 1, showgrid: true };
    if (cfg.yAxisRange) {
        yAxisConfig.range = cfg.yAxisRange;
    } else {
        const rangeStart = cfg.yScaleWindow ? cfg.yScaleWindow[0] : Math.min(4, cfg.promptLen);
        const rangeEnd = cfg.yScaleWindow ? cfg.yScaleWindow[1] : Infinity;
        let minY = Infinity, maxY = -Infinity;
        for (const trait of t.displayed) {
            const v = t.values.get(trait);
            const lo = Math.max(0, Math.floor(rangeStart));
            const hi = Math.min(v.length, Math.ceil(rangeEnd));
            for (let i = lo; i < hi; i++) {
                const val = v[i];
                if (!Number.isFinite(val)) continue;
                if (val < minY) minY = val;
                if (val > maxY) maxY = val;
            }
            // Include cohort band extents in the y-range so ±σ doesn't escape the chart.
            const c = cfg.cohort?.perTrait?.[trait];
            if (c) {
                for (let i = lo; i < hi && i < c.mean.length; i++) {
                    const m = c.mean[i];
                    const s = c.std?.[i] || 0;
                    if (!Number.isFinite(m)) continue;
                    if (m - s < minY) minY = m - s;
                    if (m + s > maxY) maxY = m + s;
                }
            }
        }
        if (Number.isFinite(minY) && Number.isFinite(maxY)) {
            const pad = Math.max(0.02, (maxY - minY) * 0.10);
            yAxisConfig.range = [minY - pad, maxY + pad];
        }
    }

    // ── PROMPT/RESPONSE labels ──
    const annotations = [];
    if (!cfg.isRollout && cfg.promptLen > 0) {
        const textSecondary = getCssVar('--text-secondary', '#a4a4a4');
        annotations.push({
            x: (cfg.promptLen - cfg.startTokenIdx) / 2 - 0.5,
            y: 1.08, yref: 'paper',
            text: 'PROMPT', showarrow: false,
            font: { size: 11, color: textSecondary },
        });
        annotations.push({
            x: (cfg.promptLen - cfg.startTokenIdx)
                + (displayTokens.length - (cfg.promptLen - cfg.startTokenIdx)) / 2 - 0.5,
            y: 1.08, yref: 'paper',
            text: 'RESPONSE', showarrow: false,
            font: { size: 11, color: textSecondary },
        });
    }

    const layout = buildChartLayout({
        preset: 'timeSeries',
        traces,
        height: cfg.height,
        legendPosition: 'none',
        xaxis: {
            tickmode: 'array', tickvals: tickVals, ticktext: tickText,
            tickangle: -45, showgrid: false, tickfont: { size: 9 },
            automargin: true,
        },
        yaxis: { ...yAxisConfig, automargin: true },
        shapes,
        annotations,
        margin: { l: 50, r: 16, t: 28, b: 48 },
        hovermode: 'closest',
    });

    // Re-render efficiently if the chart already exists, else newPlot.
    const renderFn = (plotDiv && plotDiv.data) ? updateChart : renderChart;
    renderFn(plotDiv, traces, layout);

    if (cfg.onTokenClick) {
        attachTokenClickHandler(plotDiv, cfg.startTokenIdx);
    }

    if (cfg.showLegend) {
        const existing = plotDiv.parentNode?.querySelector?.(`.chart-legend-interactive[data-tooltip-id="${cfg.hoverTooltipId}"]`);
        if (existing) existing.remove();
        const legendTooltips = cfg.legendTooltipFn
            ? t.displayed.map((trait, i) => cfg.legendTooltipFn(trait, i, cfg.traitData[trait]))
            : [];
        // Filter to current-line traces only for legend (skip cohort traces)
        const legendTraces = traces.filter(tr => !tr._cohortBand && !tr._cohortMean);
        const legendDiv = createHtmlLegend(legendTraces, plotDiv, {
            tooltips: legendTooltips,
            hoverHighlight: true,
        });
        legendDiv.dataset.tooltipId = cfg.hoverTooltipId;
        plotDiv.parentNode.insertBefore(legendDiv, plotDiv.nextSibling);

        // Notify caller when a legend item toggles, so trait visibility can survive
        // re-renders. Listening on legend item clicks AFTER createHtmlLegend's own
        // handler runs (it sets the .legend-item-hidden class synchronously).
        if (cfg.onToggleTrait) {
            legendDiv.querySelectorAll('.legend-item-interactive').forEach((item, i) => {
                const trace = legendTraces[i];
                const traitKey = trace?._traitKey;
                if (!traitKey) return;
                item.addEventListener('click', () => {
                    const hidden = item.classList.contains('legend-item-hidden');
                    cfg.onToggleTrait(traitKey, hidden);
                });
            });
        }
    }

    attachSortedHover(plotDiv, () => ({
        traces: traces.filter(tr => !tr._cohortBand && !tr._cohortMean),
        displayTokens,
    }), { tooltipId: cfg.hoverTooltipId });

    return { traitActivations, displayedTraits: t.displayed, plotDiv };
}

// ─── Bars renderer (CSS color-grid, K rows) ────────────────────────────

function _renderBars(divId, cfg, t) {
    const root = typeof divId === 'string' ? document.getElementById(divId) : divId;
    if (!root) return { traitActivations: {}, displayedTraits: [], plotDiv: null };
    const traitActivations = {};

    if (!t.displayed.length) {
        root.innerHTML = `<div class="info" style="font-size:var(--text-xxs);">No traits to display.</div>`;
        return { traitActivations, displayedTraits: [], plotDiv: null };
    }

    // Window: caller may have specified xStart/xEnd in flat (prompt+response) coords.
    const flatLen = cfg.promptLen + (cfg.responseTokens.length || 0);
    let xStart = cfg.window?.xStart ?? 0;
    let xEnd = cfg.window?.xEnd ?? flatLen;

    // Compute global absMax across all displayed traits for shared color scale.
    let absMax = 0;
    for (const trait of t.displayed) {
        const v = t.values.get(trait);
        traitActivations[trait] = v;
        for (let i = Math.max(0, xStart); i < Math.min(v.length, xEnd); i++) {
            const a = Math.abs(v[i]);
            if (a > absMax) absMax = a;
        }
    }
    if (absMax === 0) absMax = 1;

    const rows = t.displayed.map((trait, idx) => {
        const v = t.values.get(trait);
        const cohortMean = cfg.cohort?.perTrait?.[trait]?.mean;
        return _renderBarRow({
            trait,
            values: v,
            cohortMean,
            xStart,
            xEnd,
            absMax,
            highlight: cfg.highlightTokenIdx,
            displayName: cfg.traitDisplayName(trait),
        });
    }).join('');

    root.innerHTML = `
        <div style="font-family: var(--font-mono, monospace); font-size:11px;">
            ${rows}
        </div>
    `;
    return { traitActivations, displayedTraits: t.displayed, plotDiv: null };
}

function _renderBarRow({ trait, values, cohortMean, xStart, xEnd, absMax, highlight, displayName }) {
    const cellWidth = 4;
    const cells = [];
    for (let x = xStart; x < xEnd; x++) {
        if (x < 0 || x >= values.length) {
            cells.push(`<span style="display:inline-block;width:${cellWidth}px;height:14px;background:transparent;"></span>`);
            continue;
        }
        const v = values[x];
        const t = absMax > 0 ? v / absMax : 0;
        const opacity = 0.1 + 0.9 * Math.abs(t);
        const color = t >= 0 ? `rgba(80,160,255,${opacity})` : `rgba(255,90,90,${opacity})`;
        // Cohort overlay: thin under-bar tinted by mean. Visible only when cohort data exists.
        let cohortDot = '';
        if (cohortMean && x < cohortMean.length) {
            const cv = cohortMean[x];
            const ct = absMax > 0 ? cv / absMax : 0;
            const cOpacity = 0.6 * Math.abs(ct);
            const cColor = ct >= 0 ? `rgba(80,160,255,${cOpacity})` : `rgba(255,90,90,${cOpacity})`;
            cohortDot = `border-bottom:2px solid ${cColor};`;
        }
        const onsetMark = (highlight != null && x === highlight) ? 'border-left:2px solid var(--text-primary);' : '';
        cells.push(`<span title="t${x}: ${v.toFixed(3)}" style="display:inline-block;width:${cellWidth}px;height:14px;background:${color};${onsetMark}${cohortDot}"></span>`);
    }
    return `
        <div style="display:flex;align-items:center;gap:6px;margin-bottom:1px;">
            <div style="flex:0 0 14em;font-size:10px;color:var(--text-secondary);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;" title="${trait}">${displayName}</div>
            <div style="line-height:0;white-space:nowrap;overflow-x:auto;">${cells.join('')}</div>
        </div>
    `;
}

// ─── Color helpers ─────────────────────────────────────────────────────

function _withAlpha(color, alpha) {
    // Convert 'rgb(r,g,b)' → 'rgba(r,g,b,alpha)'. Pass-through for 'rgba(...)' or hex.
    if (typeof color !== 'string') return color;
    if (color.startsWith('rgb(')) return color.replace('rgb(', 'rgba(').replace(')', `,${alpha})`);
    if (color.startsWith('rgba(')) return color.replace(/[\d.]+\)$/, `${alpha})`);
    if (color.startsWith('#')) {
        // hex → rgba
        const hex = color.slice(1);
        const full = hex.length === 3
            ? hex.split('').map(c => c + c).join('')
            : hex;
        const r = parseInt(full.slice(0, 2), 16);
        const g = parseInt(full.slice(2, 4), 16);
        const b = parseInt(full.slice(4, 6), 16);
        return `rgba(${r},${g},${b},${alpha})`;
    }
    return color;
}

export {
    renderPerTokenProjectionChart,
};
