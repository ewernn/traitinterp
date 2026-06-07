/**
 * Styled select dropdown — unified component replacing native <select>.
 *
 * Visual style mirrors the experiment picker (.exp-dropdown): dark pill trigger, fixed-position
 * menu with hover/focus open, hover highlight, active-selected state.
 *
 * Input: { id, options: [{value, label, badges?}], selected, onChange, disabled, placeholder }
 * Output: HTML string + auto-wires events on the next microtask via MutationObserver-free
 *         delegated handlers attached when the caller inserts the HTML.
 * Usage:
 *     container.innerHTML = renderStyledSelect({ id, options, selected, onChange: v => ... });
 *     wireStyledSelect(container);
 */

const pendingHandlers = new Map();  // id -> onChange

// Touch devices have no hover, so the CSS hover-open never fires. A single
// delegated tap handler toggles an `.open` class instead. Attached once.
const isTouch = typeof window !== 'undefined' && window.matchMedia('(hover: none)').matches;
let touchOpenWired = false;

function positionMenu(dropdown, menu) {
    const rect = dropdown.getBoundingClientRect();
    menu.style.top = `${rect.bottom}px`;
    menu.style.left = `${rect.left}px`;
    requestAnimationFrame(() => {
        const menuRect = menu.getBoundingClientRect();
        if (menuRect.right > window.innerWidth - 8) {
            menu.style.left = `${Math.max(4, window.innerWidth - menuRect.width - 8)}px`;
        }
    });
}

function wireTouchOpen() {
    if (touchOpenWired || !isTouch) return;
    touchOpenWired = true;
    document.addEventListener('click', (e) => {
        const trigger = e.target.closest('.exp-trigger');
        const dropdown = trigger ? trigger.closest('.exp-dropdown') : null;
        // Close any open menus that aren't the one being tapped.
        document.querySelectorAll('.exp-dropdown.open').forEach(d => {
            if (d !== dropdown) d.classList.remove('open');
        });
        if (!dropdown || dropdown.dataset.disabled === 'true') return;
        // Tapping inside the menu (an item) is handled by the item's own click.
        if (e.target.closest('.exp-menu')) return;
        const willOpen = !dropdown.classList.contains('open');
        dropdown.classList.toggle('open', willOpen);
        if (willOpen) positionMenu(dropdown, dropdown.querySelector('.exp-menu'));
    });
}

function renderStyledSelect({ id, options, selected, onChange, disabled = false, placeholder = 'Select…' }) {
    if (!id) throw new Error('renderStyledSelect requires an id');
    if (!options || !Array.isArray(options)) throw new Error('renderStyledSelect requires options array');

    pendingHandlers.set(id, onChange);

    const activeOpt = options.find(o => o.value === selected);
    const triggerLabel = activeOpt ? activeOpt.label : placeholder;
    const disabledAttr = disabled ? ' data-disabled="true"' : '';

    const menuItems = options.map(opt => {
        const isActive = opt.value === selected;
        const badges = opt.badges && opt.badges.length > 0
            ? `<span class="exp-badges">${opt.badges.map(b => `<span class="exp-badge">${b}</span>`).join('')}</span>`
            : '';
        return `<div class="exp-menu-item${isActive ? ' active' : ''}" data-value="${escapeAttr(opt.value)}">${escapeHtml(opt.label)}${badges}</div>`;
    }).join('');

    return `
        <div class="exp-dropdown styled-select" data-select-id="${id}"${disabledAttr}>
            <div class="exp-trigger">
                <span class="exp-name">${escapeHtml(triggerLabel)}</span>
                <span class="exp-arrow">&#9662;</span>
            </div>
            <div class="exp-menu">${menuItems}</div>
        </div>
    `;
}

function wireStyledSelect(root = document) {
    wireTouchOpen();
    root.querySelectorAll('.styled-select[data-select-id]').forEach(el => {
        if (el.dataset.wired === 'true') return;
        el.dataset.wired = 'true';

        const id = el.dataset.selectId;
        const onChange = pendingHandlers.get(id);
        const dropdown = el;
        const menu = el.querySelector('.exp-menu');

        // Fixed-position menu: anchor under trigger, clamp to viewport (hover open).
        dropdown.addEventListener('mouseenter', () => {
            if (dropdown.dataset.disabled === 'true') return;
            positionMenu(dropdown, menu);
        });

        menu.querySelectorAll('.exp-menu-item').forEach(item => {
            item.addEventListener('click', (e) => {
                e.stopPropagation();
                const value = item.dataset.value;
                // Update trigger text + active class without re-render.
                const nameEl = dropdown.querySelector('.exp-name');
                if (nameEl) nameEl.textContent = item.textContent.trim();
                menu.querySelectorAll('.exp-menu-item').forEach(i => i.classList.remove('active'));
                item.classList.add('active');
                dropdown.classList.remove('open');  // dismiss tap-opened menu
                if (typeof onChange === 'function') onChange(value);
            });
        });
    });
}

function escapeHtml(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
function escapeAttr(s) {
    return String(s).replace(/&/g, '&amp;').replace(/"/g, '&quot;');
}

export { renderStyledSelect, wireStyledSelect };
