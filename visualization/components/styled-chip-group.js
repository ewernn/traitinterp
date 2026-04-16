/**
 * Styled chip group — unified component for single/multi-select chip rows.
 *
 * Replaces: .trait-chip, .filter-chip, .rb-info-btn (Phase C migrations).
 * Does NOT replace: .rb-chip (native checkbox, a11y), .steer-btn (incremental update),
 *                   .pp-btn prompt-ID buttons (100s of items + tag classes + pagination).
 *
 * Input:
 *   { id, items: [{value, label, badges?, disabled?, className?}], selected, mode, onChange, pill, label }
 *
 * Modes:
 *   'single'             — radio, one must be selected
 *   'multi'              — checkbox, any subset (including empty)
 *   'multi-min-one'      — multi, but can't deselect the last
 *   'toggle-off'         — clicking active sets selected=null (3-state: off / option-A / option-B…)
 *   'multi-empty-all'    — multi, but empty Set means "all on" semantically (caller handles)
 *
 * onChange contract:
 *   mode='single'          → onChange(newValue)
 *   mode='toggle-off'      → onChange(newValue|null)
 *   mode='multi*'          → onChange(clickedValue, newSelectedSet)
 *
 * Output: HTML string. Caller inserts, then runs wireChipGroup(root) to attach handlers.
 * Usage:
 *     container.innerHTML = renderChipGroup({ id, mode:'multi', items, selected, onChange, label:'Method:' });
 *     wireChipGroup(container);
 */

const pendingHandlers = new Map();  // id -> { mode, onChange }

function renderChipGroup({ id, items, selected, mode = 'single', onChange, pill = false, label = null, className = '' }) {
    if (!id) throw new Error('renderChipGroup requires an id');
    if (!Array.isArray(items)) throw new Error('renderChipGroup requires items array');

    pendingHandlers.set(id, { mode, onChange });

    const selectedSet = selected instanceof Set ? selected
        : Array.isArray(selected) ? new Set(selected)
        : selected != null ? new Set([selected])
        : new Set();

    const isSelected = (v) => selectedSet.has(v);

    const chips = items.map(item => {
        const active = isSelected(item.value) ? ' active' : '';
        const disabled = item.disabled ? ' disabled' : '';
        const extra = item.className ? ` ${item.className}` : '';
        const badges = item.badges && item.badges.length > 0
            ? `<span class="exp-badges">${item.badges.map(b => `<span class="exp-badge">${escapeHtml(b)}</span>`).join('')}</span>`
            : '';
        return `<span class="chip${active}${disabled}${extra}" data-value="${escapeAttr(item.value)}">${escapeHtml(item.label)}${badges}</span>`;
    }).join('');

    const labelHtml = label ? `<span class="chip-group-label">${escapeHtml(label)}</span>` : '';
    const pillClass = pill ? ' chip-group-pill' : '';
    const classAttr = className ? ` ${className}` : '';

    return `<div class="chip-group${pillClass}${classAttr}" data-chip-group-id="${id}" data-mode="${mode}">${labelHtml}${chips}</div>`;
}

function wireChipGroup(root = document) {
    root.querySelectorAll('.chip-group[data-chip-group-id]').forEach(el => {
        if (el.dataset.wired === 'true') return;
        el.dataset.wired = 'true';

        const id = el.dataset.chipGroupId;
        const mode = el.dataset.mode;
        const handler = pendingHandlers.get(id);
        if (!handler || typeof handler.onChange !== 'function') return;

        el.addEventListener('click', (e) => {
            const chip = e.target.closest('.chip');
            if (!chip || chip.classList.contains('disabled')) return;
            if (!el.contains(chip)) return;

            const value = chip.dataset.value;

            if (mode === 'single') {
                if (chip.classList.contains('active')) return;  // no-op
                el.querySelectorAll('.chip').forEach(c => c.classList.toggle('active', c === chip));
                handler.onChange(value);
                return;
            }

            if (mode === 'toggle-off') {
                if (chip.classList.contains('active')) {
                    chip.classList.remove('active');
                    handler.onChange(null);
                } else {
                    el.querySelectorAll('.chip').forEach(c => c.classList.toggle('active', c === chip));
                    handler.onChange(value);
                }
                return;
            }

            // multi / multi-min-one / multi-empty-all
            const currentlyActive = new Set(
                Array.from(el.querySelectorAll('.chip.active')).map(c => c.dataset.value)
            );

            if (currentlyActive.has(value)) {
                if (mode === 'multi-min-one' && currentlyActive.size <= 1) return;
                currentlyActive.delete(value);
                chip.classList.remove('active');
            } else {
                currentlyActive.add(value);
                chip.classList.add('active');
            }
            handler.onChange(value, currentlyActive);
        });
    });
}

function escapeHtml(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
function escapeAttr(s) {
    return String(s).replace(/&/g, '&amp;').replace(/"/g, '&quot;');
}

export { renderChipGroup, wireChipGroup };
