/**
 * Styled select dropdown — unified component replacing native <select> and .cb-select / .rb-select.
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
    root.querySelectorAll('.styled-select[data-select-id]').forEach(el => {
        if (el.dataset.wired === 'true') return;
        el.dataset.wired = 'true';

        const id = el.dataset.selectId;
        const onChange = pendingHandlers.get(id);
        const dropdown = el;
        const menu = el.querySelector('.exp-menu');

        // Fixed-position menu: anchor under trigger, clamp to viewport.
        dropdown.addEventListener('mouseenter', () => {
            if (dropdown.dataset.disabled === 'true') return;
            const rect = dropdown.getBoundingClientRect();
            menu.style.top = `${rect.bottom}px`;
            menu.style.left = `${rect.left}px`;
            requestAnimationFrame(() => {
                const menuRect = menu.getBoundingClientRect();
                if (menuRect.right > window.innerWidth - 8) {
                    menu.style.left = `${Math.max(4, window.innerWidth - menuRect.width - 8)}px`;
                }
            });
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
