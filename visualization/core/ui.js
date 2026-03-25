/**
 * Shared UI primitives for visualization views.
 * Pure functions that return HTML strings.
 *
 * Usage:
 *   import { renderToggle } from './ui.js';
 */

import { escapeHtml } from './utils.js';

// === Subsections ===

/**
 * Render a subsection header with expandable info panel.
 * @param {Object} options
 * @param {string|number} [options.num] - Section number (optional)
 * @param {string} options.title - Section title
 * @param {string} options.infoId - ID for the info panel
 * @param {string} options.infoText - Content for the info panel
 * @param {string} [options.level='h3'] - Header level (h2, h3, h4)
 * @returns {string} HTML string
 */
function renderSubsection({ num, title, infoId, infoText, level = 'h3' }) {
    const numHtml = num ? `<span class="subsection-num">${num}.</span>` : '';
    return `
        <${level} class="subsection-header">
            ${numHtml}
            <span class="subsection-title">${title}</span>
            <span class="subsection-info-toggle" data-target="${infoId}">►</span>
        </${level}>
        <div class="subsection-info" id="${infoId}">${infoText}</div>
    `;
}

// === Form Controls ===

/**
 * Render a select dropdown with label.
 * @param {Object} options
 * @param {string} options.id - Element ID
 * @param {string} options.label - Label text
 * @param {Array<string|{value: string, label?: string}>} options.options - Options array
 * @param {string} [options.selected] - Currently selected value
 * @param {string} [options.className] - Additional CSS class
 * @param {string} [options.placeholder] - Placeholder option text
 * @returns {string} HTML string
 */
function renderSelect({ id, label, options, selected, className, placeholder }) {
    const optionsHtml = options.map(opt => {
        const value = typeof opt === 'string' ? opt : opt.value;
        const text = typeof opt === 'string' ? opt : (opt.label || opt.value);
        return `<option value="${value}" ${value === selected ? 'selected' : ''}>${text}</option>`;
    }).join('');
    const placeholderHtml = placeholder ? `<option value="">${placeholder}</option>` : '';

    return `
        <div class="control-group">
            <label>${label}</label>
            <select id="${id}" class="${className || id}">${placeholderHtml}${optionsHtml}</select>
        </div>
    `;
}

/**
 * Render a checkbox toggle with label.
 * @param {Object} options
 * @param {string} [options.id] - Element ID (optional)
 * @param {string} options.label - Label text
 * @param {boolean} [options.checked=false] - Whether checked
 * @param {{key: string, value: string}} [options.dataAttr] - Data attribute
 * @param {string} [options.className] - Additional CSS class
 * @returns {string} HTML string
 */
function renderToggle({ id, label, checked, dataAttr, className }) {
    const idAttr = id ? `id="${id}"` : '';
    const checkedAttr = checked ? 'checked' : '';
    const dataHtml = dataAttr ? `data-${dataAttr.key}="${dataAttr.value}"` : '';
    return `
        <label class="toggle-row ${className || ''}">
            <input type="checkbox" ${idAttr} ${checkedAttr} ${dataHtml}>
            ${label}
        </label>
    `;
}

// === Chips ===

/**
 * Render a single chip/button.
 * @param {Object} options
 * @param {string} options.label - Button text
 * @param {boolean} [options.active=false] - Whether active/selected
 * @param {{key: string, value: string}} [options.dataAttr] - Data attribute
 * @param {string} [options.className] - Additional CSS class
 * @param {string} [options.onClick] - Inline onclick handler
 * @returns {string} HTML string
 */
function renderChip({ label, active, dataAttr, className, onClick }) {
    const activeClass = active ? 'active' : '';
    const dataHtml = dataAttr ? `data-${dataAttr.key}="${dataAttr.value}"` : '';
    const onClickHtml = onClick ? `onclick="${onClick}"` : '';
    return `<button class="btn btn-xs ${className || ''} ${activeClass}" ${dataHtml} ${onClickHtml}>${label}</button>`;
}

// === Tables ===

/**
 * Render a sortable table header cell.
 * @param {Object} options
 * @param {string} options.key - Sort key for this column
 * @param {string} options.label - Column header text
 * @param {string} [options.sortKey] - Currently active sort key
 * @param {'asc'|'desc'} [options.sortDir] - Current sort direction
 * @returns {string} HTML string
 */
function renderSortableHeader({ key, label, sortKey, sortDir }) {
    const isActive = key === sortKey;
    const indicator = isActive ? (sortDir === 'desc' ? '▼' : '▲') : '▼';
    return `
        <th class="sortable ${isActive ? 'sort-active' : ''}" data-sort="${key}">
            ${label} <span class="sort-indicator">${indicator}</span>
        </th>
    `;
}

// === States ===

/**
 * Render a loading indicator.
 * @param {string} [message='Loading...'] - Loading message
 * @returns {string} HTML string
 */
function renderLoading(message = 'Loading...') {
    return `<div class="loading">${message}</div>`;
}

// === Guards & States ===

/**
 * Render "no experiment selected" guard. Returns true if guard was shown.
 * Usage: if (requireExperiment(contentArea)) return;
 */
function requireExperiment(contentArea) {
    if (window.state.currentExperiment) return false;
    contentArea.innerHTML = `<div class="tool-view"><div class="no-data">
        <p>Please select an experiment from the sidebar to view analysis.</p>
        <p class="hint">Experiments are loaded from the <code>experiments/</code> directory.</p>
    </div></div>`;
    return true;
}

/**
 * Show loading indicator after a delay (avoids flash for fast loads).
 * Returns { cancel } handle. Call cancel() when data arrives.
 */
function deferredLoading(targetEl, message = 'Loading...', delayMs = 150) {
    const el = typeof targetEl === 'string' ? document.getElementById(targetEl) : targetEl;
    const timer = setTimeout(() => { if (el) el.innerHTML = renderLoading(message); }, delayMs);
    return { cancel: () => clearTimeout(timer) };
}

/**
 * Render a "no data — run this command" hint block.
 */
function renderRunHint(message, command) {
    return `<div class="info">${message}<br><br>Run: <code>${command}</code></div>`;
}

/**
 * Render a single filter chip (uses .filter-chip CSS, not .btn).
 * @param {string} value - Data value
 * @param {string} label - Display text
 * @param {Set<string>|string} active - Active value(s)
 * @param {string} dataAttr - data-* attribute name
 */
function renderFilterChip(value, label, active, dataAttr) {
    const isActive = active instanceof Set ? active.has(value) : active === value;
    return `<span class="filter-chip${isActive ? ' active' : ''}" data-${dataAttr}="${value}">${label}</span>`;
}

/**
 * Render a labeled row of filter chips. Returns '' if ≤1 option.
 * @param {string} label - Row label
 * @param {Set<string>|Array} values - All possible values
 * @param {Set<string>} active - Active values
 * @param {string} groupKey - data-filter-group value + used for data-value
 * @param {Object} [opts]
 * @param {Object} [opts.displayNames] - value → display label map
 * @param {Function} [opts.formatLabel] - fallback formatter
 */
function renderFilterChipRow(label, values, active, groupKey, { displayNames = {}, formatLabel = null } = {}) {
    const arr = Array.from(values);
    if (arr.length <= 1) return '';
    const defaultFmt = v => v.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
    const chips = arr.map(v => {
        const display = displayNames[v] ?? (formatLabel ? formatLabel(v) : defaultFmt(v));
        const activeClass = active.has(v) ? ' active' : '';
        return `<span class="filter-chip${activeClass}" data-filter-group="${groupKey}" data-value="${v}">${display}</span>`;
    }).join('');
    return `<div class="filter-row"><span class="filter-label">${label}:</span>${chips}</div>`;
}

// === Score Badges ===

/**
 * Return CSS class for score badge coloring.
 * @param {number} val - Score value
 * @param {'trait'|'coherence'} [type='trait'] - Score type with preset thresholds
 *   trait: >50 good, >20 ok, else ''
 *   coherence: >=80 good, >=60 ok, else bad
 * @returns {string} CSS class ('quality-good', 'quality-ok', 'quality-bad', or '')
 */
function scoreClass(val, type = 'trait') {
    if (type === 'coherence') {
        return val >= 80 ? 'quality-good' : val >= 60 ? 'quality-ok' : 'quality-bad';
    }
    // trait (default)
    return val > 50 ? 'quality-good' : val > 20 ? 'quality-ok' : '';
}

// === Segmented Controls ===

/**
 * Render a segmented pill control (mutually exclusive options).
 * @param {Object} opts
 * @param {string} opts.id - Container ID
 * @param {Array<{value: string, label: string}>} opts.options
 * @param {string} opts.selected - Currently selected value
 * @param {string} opts.dataAttr - data attribute name (e.g., 'compare-mode')
 * @param {boolean} [opts.disabled] - Disable all options
 * @param {string} [opts.disabledTooltip] - Tooltip when disabled
 */
function renderSegmentedControl({ id, options, selected, dataAttr, disabled, disabledTooltip }) {
    const groupClass = disabled ? 'seg-pill disabled-group' : 'seg-pill';
    const tooltip = disabled && disabledTooltip ? ` title="${disabledTooltip}"` : '';
    const buttons = options.map(opt => {
        const activeClass = opt.value === selected ? ' active' : '';
        const disabledAttr = disabled ? ' disabled' : '';
        return `<button class="${activeClass.trim()}" data-${dataAttr}="${opt.value}"${disabledAttr}>${opt.label}</button>`;
    }).join('');
    return `<div class="${groupClass}" id="${id}"${tooltip}>${buttons}</div>`;
}

/**
 * Render a smooth pill control (0/3/6/9 window selector).
 * @param {number} selected - Current smoothing window (0 = off)
 */
function renderSmoothPill(selected) {
    const options = [
        { value: 0, label: 'off' },
        { value: 3, label: '3' },
        { value: 6, label: '6' },
        { value: 9, label: '9' },
    ];
    const buttons = options.map(opt => {
        const activeClass = opt.value === selected ? ' active' : '';
        return `<button class="${activeClass.trim()}" data-smooth="${opt.value}">${opt.label}</button>`;
    }).join('');
    return `<div class="smooth-pill">${buttons}</div>`;
}

// ES module exports
export {
    renderSubsection,
    renderSelect,
    renderToggle,
    renderChip,
    renderSortableHeader,
    renderLoading,
    requireExperiment,
    deferredLoading,
    renderRunHint,
    renderFilterChip,
    renderFilterChipRow,
    scoreClass,
    renderSegmentedControl,
    renderSmoothPill,
};

