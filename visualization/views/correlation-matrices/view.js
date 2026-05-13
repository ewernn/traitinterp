// Correlation Matrices view.
//
// Browse the 144-config sweep produced by dev/conv_tools/correlation_sweep/.
// Each config is a per-bias-pair similarity matrix (asymmetric: matrix[A][B] is
// not necessarily matrix[B][A]). The sidebar lets the user pick a config; the
// main pane shows config metadata, summary stats, and a Plotly heatmap.
//
// Input:  /dev/conv_tools/correlation_sweep/{index.json, configs/cfg_NNN.json}
// Output: full-page UI rendered into #content-area.
// Usage:  registered as window.renderCorrelationMatrices; auto-invoked by router.
//
// Reuse: leans on visualization/styles.css primitives (.btn, .chip, .section-title,
// .info, .error, .loading, .tool-view) and core/charts for Plotly. Heatmap
// component lives in matrix-heatmap.js.

import { loadIndex, loadConfig, summarizeMatrix, filterPervasive, PERVASIVE_SCOPE_BIAS_IDS } from './data.js';
import { renderMatrixHeatmap } from './matrix-heatmap.js';

// View-local state — self-contained, not in window.state.
const VS = {
    indexData: null,             // index.json contents
    cfgList: [],                 // index.configs (array of metadata rows)
    sortColumn: 'discrim_std',   // default sort: highest std discriminator first
    sortDir: 'desc',
    selectedCfgId: null,
    cfgData: null,               // currently-loaded config JSON
    cfgLoading: false,
    cfgError: null,
    sortMode: 'id',              // bias sort within heatmap: id|name|diagonal
    hideDiagonal: false,
    metric: 'cosine',            // 'cosine' | 'dot_per_w' — which matrix to display
    hoveredCell: null,           // { aId, bId, value } — last hovered/clicked
    pinnedCell: null,            // sticky on click; null = hover-only
    indexError: null,
};

async function renderCorrelationMatrices() {
    const root = document.getElementById('content-area');

    if (!VS.indexData) {
        root.innerHTML = `<div class="loading">Loading correlation sweep index...</div>`;
        try {
            const index = await loadIndex();
            VS.indexData = index;
            VS.cfgList = index.configs || [];
            // Default selection: highest discrim_std config
            if (VS.selectedCfgId === null && VS.cfgList.length > 0) {
                const best = [...VS.cfgList].sort(
                    (a, b) => (b.discrim_std ?? 0) - (a.discrim_std ?? 0)
                )[0];
                VS.selectedCfgId = best.config_id;
            }
        } catch (e) {
            VS.indexError = e.message;
            root.innerHTML = `
                <div class="tool-view">
                    <div class="error">Failed to load correlation sweep index: ${_escape(e.message)}</div>
                    <div class="info">
                        Expected at <code>/dev/conv_tools/correlation_sweep/index.json</code>.
                        Generate via <code>dev/conv_tools/correlation_sweep/</code>.
                    </div>
                </div>`;
            return;
        }
    }

    _paint(root);

    if (VS.selectedCfgId !== null) {
        await _ensureConfigLoaded(VS.selectedCfgId);
        _renderMainPane();
    }
}

// =============================================================================
// Top-level paint — sidebar (config table) + main pane shell
// =============================================================================

function _paint(root) {
    root.innerHTML = `
        <div class="tool-view correlation-matrices" style="display:grid; grid-template-columns:minmax(360px, 30%) 1fr; gap:var(--space-md); align-items:start; height:calc(100vh - 120px);">
            <div class="cm-sidebar" style="overflow:auto; max-height:100%; border:1px solid var(--border-color); border-radius:var(--radius-sm); padding:var(--space-sm); background:var(--bg-secondary);">
                ${_renderConfigTable()}
            </div>
            <div class="cm-main" id="cm-main" style="overflow:auto; max-height:100%;">
                <div class="loading">Pick a config from the left to load its matrix.</div>
            </div>
        </div>
    `;
    _wireConfigTable();
}

function _renderConfigTable() {
    const cols = [
        { key: 'config_id', label: 'cfg', fmt: (v) => String(v) },
        { key: 'mode', label: 'mode', fmt: _shortMode },
        { key: 'rank_by', label: 'rank_by', fmt: (v) => String(v) },
        { key: 'window_half', label: 'W', fmt: (v) => `±${v}` },
        { key: 'top_k', label: 'K', fmt: (v) => String(v) },
        { key: 'discrim_std', label: 'dot/W std', fmt: _fmtFloat },
        { key: 'cosine_discrim_std', label: 'cos std', fmt: _fmtFloat },
        { key: 'per_trait_cos_discrim_std', label: 'p-trait cos', fmt: _fmtFloat },
        { key: 'weighted_cos_discrim_std', label: 'w-cos', fmt: _fmtFloat },
    ];

    const sorted = _sortedCfgList();
    const rows = sorted.map(cfg => {
        const isActive = cfg.config_id === VS.selectedCfgId;
        const tds = cols.map(c => {
            const raw = cfg[c.key];
            return `<td style="padding:3px 8px; ${c.key === 'discrim_std' ? 'font-family:var(--font-mono); color:var(--text-secondary);' : ''}">${c.fmt(raw)}</td>`;
        }).join('');
        return `<tr class="cm-row${isActive ? ' active' : ''}" data-cfg-id="${cfg.config_id}" style="cursor:pointer; ${isActive ? 'background:var(--accent-color); color:var(--text-on-primary);' : ''}">${tds}</tr>`;
    }).join('');

    const ths = cols.map(c => {
        const isSort = c.key === VS.sortColumn;
        const arrow = isSort ? (VS.sortDir === 'desc' ? '▾' : '▴') : '';
        return `<th data-col="${c.key}" style="padding:5px 8px; text-align:left; cursor:pointer; border-bottom:1px solid var(--border-color); font-size:var(--text-xxs); font-weight:var(--fw-semibold); color:var(--text-secondary); user-select:none; position:sticky; top:0; background:var(--bg-secondary);">${c.label}${arrow ? ` <span style="opacity:0.7">${arrow}</span>` : ''}</th>`;
    }).join('');

    return `
        <div class="section-title" style="margin-bottom:6px;">Configs (${VS.cfgList.length})</div>
        <div style="font-size:var(--text-xxs); color:var(--text-tertiary); margin-bottom:6px;">
            Click column to sort. Click row to load.
        </div>
        <table class="cm-cfg-table" style="width:100%; border-collapse:collapse; font-size:var(--text-xs);">
            <thead><tr>${ths}</tr></thead>
            <tbody>${rows}</tbody>
        </table>
    `;
}

function _sortedCfgList() {
    const list = [...VS.cfgList];
    const col = VS.sortColumn;
    const dir = VS.sortDir === 'desc' ? -1 : 1;
    list.sort((a, b) => {
        const va = a[col];
        const vb = b[col];
        if (typeof va === 'number' && typeof vb === 'number') return (va - vb) * dir;
        return String(va).localeCompare(String(vb)) * dir;
    });
    return list;
}

function _wireConfigTable() {
    const tbl = document.querySelector('.cm-cfg-table');
    if (!tbl) return;

    // Column-header click: sort. Toggle direction if same column re-clicked.
    tbl.querySelectorAll('thead th[data-col]').forEach(th => {
        th.addEventListener('click', () => {
            const col = th.dataset.col;
            if (VS.sortColumn === col) {
                VS.sortDir = VS.sortDir === 'desc' ? 'asc' : 'desc';
            } else {
                VS.sortColumn = col;
                // Numeric columns default to desc, string columns to asc.
                const isNumeric = ['config_id', 'window_half', 'top_k',
                                   'discrim_std', 'discrim_mean', 'discrim_iqr',
                                   'cosine_discrim_std', 'cosine_discrim_mean',
                                   'per_trait_cos_discrim_std', 'per_trait_cos_discrim_mean',
                                   'weighted_cos_discrim_std', 'weighted_cos_discrim_mean'].includes(col);
                VS.sortDir = isNumeric ? 'desc' : 'asc';
            }
            // Re-render only the sidebar (cheap; preserves main pane).
            const sb = document.querySelector('.cm-sidebar');
            if (sb) {
                sb.innerHTML = _renderConfigTable();
                _wireConfigTable();
            }
        });
    });

    // Row-click: select + load
    tbl.querySelectorAll('tbody tr.cm-row').forEach(tr => {
        tr.addEventListener('click', async () => {
            const cfgId = parseInt(tr.dataset.cfgId, 10);
            if (cfgId === VS.selectedCfgId) return;
            VS.selectedCfgId = cfgId;
            VS.cfgData = null;
            VS.hoveredCell = null;
            VS.pinnedCell = null;

            // Re-render sidebar to highlight active row + main pane spinner
            const sb = document.querySelector('.cm-sidebar');
            if (sb) {
                sb.innerHTML = _renderConfigTable();
                _wireConfigTable();
            }
            const main = document.getElementById('cm-main');
            if (main) main.innerHTML = `<div class="loading">Loading cfg_${String(cfgId).padStart(3, '0')}.json...</div>`;

            await _ensureConfigLoaded(cfgId);
            _renderMainPane();
        });
    });
}

// =============================================================================
// Main pane — header, controls, heatmap, detail
// =============================================================================

async function _ensureConfigLoaded(cfgId) {
    VS.cfgLoading = true;
    VS.cfgError = null;
    try {
        VS.cfgData = await loadConfig(cfgId);
    } catch (e) {
        VS.cfgError = e.message;
        VS.cfgData = null;
    } finally {
        VS.cfgLoading = false;
    }
}

function _renderMainPane() {
    const main = document.getElementById('cm-main');
    if (!main) return;

    if (VS.cfgError) {
        main.innerHTML = `<div class="error">Failed to load cfg_${String(VS.selectedCfgId).padStart(3, '0')}: ${_escape(VS.cfgError)}</div>`;
        return;
    }
    if (!VS.cfgData) {
        main.innerHTML = `<div class="loading">Loading...</div>`;
        return;
    }

    const cfg = VS.cfgData.config;
    // Always-on filter: drop pervasive-scope biases. They have no single onset
    // so the per-onset matrix doesn't apply to them. Mirror of the Python
    // PERVASIVE_SCOPE_BIAS_IDS in dev/conv_tools/bias_correlation_sweep.py.
    const biasIds = filterPervasive(VS.cfgData.bias_ids || []);
    // Pick which matrix to display.
    //   cosine          — joint flat cosine (conflates trait + temporal alignment)
    //   per_trait_cos   — mean of per-trait cosines (K-fair: each trait normalized independently)
    //   weighted_cos    — weighted by per-trait signal magnitude (most K-invariant)
    //   dot_per_w       — raw dot product / 2W (magnitude-sensitive, scales with K)
    const matrix = (
        VS.metric === 'cosine'        ? (VS.cfgData.matrix_cosine        || VS.cfgData.matrix) :
        VS.metric === 'per_trait_cos' ? (VS.cfgData.matrix_per_trait_cos || VS.cfgData.matrix) :
        VS.metric === 'weighted_cos'  ? (VS.cfgData.matrix_weighted_cos  || VS.cfgData.matrix) :
                                        (VS.cfgData.matrix_dot_per_w     || VS.cfgData.matrix)
    ) || {};
    const stats = summarizeMatrix(matrix, biasIds, 5);
    const shortNames = VS.indexData?.bias_short_names || {};

    main.innerHTML = `
        ${_renderHeader(cfg, biasIds, stats, shortNames)}
        ${_renderControls()}
        <div id="cm-heatmap-wrap" style="margin-top:var(--space-md); border:1px solid var(--border-color); border-radius:var(--radius-sm); padding:var(--space-sm); background:var(--bg-secondary);">
            <div id="cm-heatmap" style="width:100%;"></div>
        </div>
        <div id="cm-detail" style="margin-top:var(--space-sm);">
            ${_renderDetail()}
        </div>
    `;

    _wireControls();

    // Render heatmap into #cm-heatmap.
    renderMatrixHeatmap('cm-heatmap', {
        matrix,
        biasIds,
        biasShortNames: shortNames,
        sortMode: VS.sortMode,
        hideDiagonal: VS.hideDiagonal,
        onCellHover: (cell) => {
            // Don't let hover overwrite a pinned selection.
            if (VS.pinnedCell) return;
            VS.hoveredCell = cell;
            _refreshDetail();
        },
        onCellClick: (cell) => {
            // Clicking the same pinned cell un-pins.
            if (VS.pinnedCell &&
                VS.pinnedCell.aId === cell.aId &&
                VS.pinnedCell.bId === cell.bId) {
                VS.pinnedCell = null;
            } else {
                VS.pinnedCell = cell;
                VS.hoveredCell = cell;
            }
            _refreshDetail();
        },
    });
}

function _renderHeader(cfg, biasIds, stats, shortNames) {
    const cfgId = cfg.config_id;
    const meta = [
        ['mode', cfg.mode],
        ['rank_by', cfg.rank_by],
        ['window', `±${cfg.window_half}`],
        ['top_k', cfg.top_k],
        ['smoothing', cfg.smoothing],
    ].map(([k, v]) => `<span style="margin-right:var(--space-md);"><span style="color:var(--text-tertiary);">${k}</span> <code style="color:var(--text-primary);">${_escape(String(v))}</code></span>`).join('');

    const topPairsHtml = stats.top.length === 0
        ? '<em style="color:var(--text-tertiary);">no off-diagonal cells</em>'
        : stats.top.map(p => {
            const aShort = shortNames[String(p.a)] || `bias_${p.a}`;
            const bShort = shortNames[String(p.b)] || `bias_${p.b}`;
            const sign = p.value >= 0 ? '+' : '';
            // Match the heatmap palette: warm = positive, cool = negative.
            const valColor = p.value >= 0 ? '#b2182b' : '#2166ac';
            return `<div style="font-size:var(--text-xxs); margin:1px 0;">
                <code style="color:var(--text-secondary);">${p.a}→${p.b}</code>
                <span style="color:var(--text-secondary);">${aShort} → ${bShort}</span>
                <code style="color:${valColor};">${sign}${p.value.toExponential(2)}</code>
            </div>`;
        }).join('');

    return `
        <div style="display:grid; grid-template-columns: 2fr 1fr; gap:var(--space-md); padding:var(--space-sm); border:1px solid var(--border-color); border-radius:var(--radius-sm); background:var(--bg-secondary);">
            <div>
                <div class="section-title" style="margin:0 0 4px 0;">cfg_${String(cfgId).padStart(3, '0')}</div>
                <div style="font-size:var(--text-xs);">${meta}</div>
                <div style="font-size:var(--text-xxs); color:var(--text-tertiary); margin-top:6px;">
                    ${biasIds.length} biases · ${biasIds.length * biasIds.length} cells
                    (${biasIds.length * (biasIds.length - 1)} off-diagonal)
                    · <span title="${[...PERVASIVE_SCOPE_BIAS_IDS].sort((a,b)=>a-b).join(',')}">${PERVASIVE_SCOPE_BIAS_IDS.size} pervasive-scope biases excluded</span>
                </div>
                <div style="font-size:var(--text-xs); margin-top:6px;">
                    <span style="color:var(--text-tertiary);">off-diag mean</span>
                    <code style="color:var(--text-secondary);">${stats.mean.toExponential(3)}</code>
                    &nbsp;&nbsp;
                    <span style="color:var(--text-tertiary);">std</span>
                    <code style="color:var(--text-secondary);">${stats.std.toExponential(3)}</code>
                    &nbsp;&nbsp;
                    <span style="color:var(--text-tertiary);">|max|</span>
                    <code style="color:var(--text-secondary);">${stats.absMax.toExponential(3)}</code>
                </div>
            </div>
            <div>
                <div style="font-size:var(--text-xxs); color:var(--text-tertiary); margin-bottom:3px;">Top-5 cross-bias pairs (by |value|):</div>
                ${topPairsHtml}
            </div>
        </div>
    `;
}

function _renderControls() {
    const sortChips = [
        ['id', 'bias id'],
        ['name', 'short name'],
        ['diagonal', 'diagonal value'],
    ].map(([mode, label]) => `<span class="chip ${VS.sortMode === mode ? 'active' : ''}" data-cm-sort="${mode}">${label}</span>`).join('');

    const metricChips = [
        ['cosine', 'cosine joint'],
        ['per_trait_cos', 'per-trait cos (K-fair)'],
        ['weighted_cos', 'weighted cos'],
        ['dot_per_w', 'dot/W (raw)'],
    ].map(([m, label]) => `<span class="chip ${VS.metric === m ? 'active' : ''}" data-cm-metric="${m}">${label}</span>`).join('');

    // Detect if newer matrices are missing (legacy configs)
    const hasNewMetrics = VS.cfgData && (VS.cfgData.matrix_per_trait_cos != null);
    const cosineWarning = !hasNewMetrics
        ? `<span style="font-size:var(--text-xxs); color:var(--text-warning,orange); margin-left:6px;">⚠ this config lacks per-trait/weighted matrices; re-run sweep</span>`
        : '';

    return `
        <div style="display:flex; gap:var(--space-md); align-items:center; padding:var(--space-sm) 0; flex-wrap:wrap;">
            <div>
                <span style="font-size:var(--text-xs); color:var(--text-tertiary); margin-right:6px;">Metric:</span>
                <span class="chip-group chip-group-pill" id="cm-metric-chips" style="display:inline-flex;">${metricChips}</span>
                ${cosineWarning}
            </div>
            <div>
                <span style="font-size:var(--text-xs); color:var(--text-tertiary); margin-right:6px;">Sort biases by:</span>
                <span class="chip-group chip-group-pill" id="cm-sort-chips" style="display:inline-flex;">${sortChips}</span>
            </div>
            <label style="font-size:var(--text-xs); color:var(--text-secondary); cursor:pointer; margin-left:auto;">
                <input type="checkbox" id="cm-hide-diag" ${VS.hideDiagonal ? 'checked' : ''}> hide diagonal (off-diag color range)
            </label>
        </div>
    `;
}

function _wireControls() {
    document.querySelectorAll('#cm-sort-chips .chip[data-cm-sort]').forEach(chip => {
        chip.addEventListener('click', () => {
            const mode = chip.dataset.cmSort;
            if (mode === VS.sortMode) return;
            VS.sortMode = mode;
            _renderMainPane();
        });
    });
    document.querySelectorAll('#cm-metric-chips .chip[data-cm-metric]').forEach(chip => {
        chip.addEventListener('click', () => {
            const m = chip.dataset.cmMetric;
            if (m === VS.metric) return;
            VS.metric = m;
            _renderMainPane();
        });
    });
    const cb = document.getElementById('cm-hide-diag');
    if (cb) {
        cb.addEventListener('change', (e) => {
            VS.hideDiagonal = e.target.checked;
            _renderMainPane();
        });
    }
}

// =============================================================================
// Detail panel — hover/click on a cell shows trait info
// =============================================================================

function _refreshDetail() {
    const el = document.getElementById('cm-detail');
    if (el) el.innerHTML = _renderDetail();
}

function _renderDetail() {
    const cell = VS.pinnedCell || VS.hoveredCell;
    if (!cell) {
        return `<div class="info" style="padding:var(--space-sm); font-size:var(--text-xs); color:var(--text-tertiary);">
            Hover any cell to inspect bias-pair details. Click to pin.
        </div>`;
    }

    const cfgData = VS.cfgData;
    if (!cfgData) return '';

    const shortNames = VS.indexData?.bias_short_names || {};
    const aShort = shortNames[String(cell.aId)] || `bias_${cell.aId}`;
    const bShort = shortNames[String(cell.bId)] || `bias_${cell.bId}`;
    const value = cell.value;

    // Top traits used to build bias A's vector. The asymmetric matrix typically
    // means "how well does B's vector predict A's spans" or similar — show
    // the traits underlying both sides for context.
    const topA = cfgData.top_traits_per_bias?.[String(cell.aId)] || [];
    const topB = cfgData.top_traits_per_bias?.[String(cell.bId)] || [];

    const renderList = (traits) => traits.length === 0
        ? '<em style="color:var(--text-tertiary);">none</em>'
        : `<ol style="margin:4px 0 0 18px; padding:0; font-size:var(--text-xxs); color:var(--text-secondary); line-height:1.5;">${traits.map(t => `<li><code>${_escape(t)}</code></li>`).join('')}</ol>`;

    const sign = value >= 0 ? '+' : '';
    const valueColor = value >= 0 ? '#b2182b' : '#2166ac';
    const isDiag = cell.aId === cell.bId;
    const pinNote = VS.pinnedCell
        ? '<span style="font-size:var(--text-xxs); color:var(--text-tertiary); margin-left:8px;">(pinned — click cell again to release)</span>'
        : '';

    return `
        <div style="padding:var(--space-sm); border:1px solid var(--border-color); border-radius:var(--radius-sm); background:var(--bg-secondary);">
            <div style="display:flex; align-items:baseline; justify-content:space-between; margin-bottom:6px;">
                <div style="font-size:var(--text-sm); color:var(--text-primary);">
                    <code>${cell.aId} → ${cell.bId}</code>
                    <span style="color:var(--text-secondary);">${aShort} ${isDiag ? '' : `→ ${bShort}`}</span>
                    ${isDiag ? '<span style="color:var(--text-tertiary); font-size:var(--text-xxs); margin-left:6px;">diagonal (self)</span>' : ''}
                </div>
                <div>
                    <code style="color:${valueColor}; font-weight:var(--fw-semibold);">${sign}${typeof value === 'number' ? value.toExponential(4) : '—'}</code>
                    ${pinNote}
                </div>
            </div>
            ${isDiag ? `
                <div>
                    <div style="font-size:var(--text-xxs); color:var(--text-tertiary);">Top traits driving bias ${cell.aId} (${aShort}):</div>
                    ${renderList(topA)}
                </div>
            ` : `
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:var(--space-md);">
                    <div>
                        <div style="font-size:var(--text-xxs); color:var(--text-tertiary);">Top traits for bias A — ${cell.aId} (${aShort}):</div>
                        ${renderList(topA)}
                    </div>
                    <div>
                        <div style="font-size:var(--text-xxs); color:var(--text-tertiary);">Top traits for bias B — ${cell.bId} (${bShort}):</div>
                        ${renderList(topB)}
                    </div>
                </div>
            `}
        </div>
    `;
}

// =============================================================================
// Helpers
// =============================================================================

function _shortMode(mode) {
    // The two modes have long names; trim to keep the table readable.
    return mode.replace('normalized_', '').replace('_centered', '');
}

function _fmtFloat(v) {
    if (typeof v !== 'number') return String(v);
    return v.toExponential(2);
}

function _escape(s) {
    return String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
}

export { renderCorrelationMatrices };
window.renderCorrelationMatrices = renderCorrelationMatrices;
