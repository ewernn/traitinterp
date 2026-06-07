/**
 * Shared behavior for fixed-position `.exp-dropdown` menus — the experiment
 * picker and every styled-select. One source of truth for viewport-clamped
 * positioning and touch open, so all dropdowns behave identically on mouse
 * and touch instead of each call site re-implementing it.
 *
 * Input: DOM with `.exp-dropdown` > (`.exp-trigger`, `.exp-menu` > `.exp-menu-item`).
 *        Menus open via CSS `:hover` on pointer devices; on touch they open via
 *        an `.open` class toggled by the delegated handler registered here.
 * Output: positionFixedMenu(dropdown, menu); a one-time document tap handler.
 * Usage:
 *     import { positionFixedMenu } from '../core/fixed-dropdown.js';
 *     dropdown.addEventListener('mouseenter', () => positionFixedMenu(dropdown, menu));
 *     // On selecting an item, call dropdown.classList.remove('open').
 */

// Anchor a fixed-position menu directly under its trigger, then nudge it back
// inside the right viewport edge if it would overflow.
export function positionFixedMenu(dropdown, menu) {
    if (!dropdown || !menu) return;
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

// Touch devices have no hover, so the CSS hover-open never fires. A single
// delegated tap handler toggles `.open` on whichever dropdown is tapped and
// closes the rest. Registered once, covering current and future dropdowns.
if (typeof window !== 'undefined' && window.matchMedia('(hover: none)').matches) {
    document.addEventListener('click', (e) => {
        const trigger = e.target.closest('.exp-trigger');
        const dropdown = trigger ? trigger.closest('.exp-dropdown') : null;
        // Close any open menu that isn't the one being tapped.
        document.querySelectorAll('.exp-dropdown.open').forEach(d => {
            if (d !== dropdown) d.classList.remove('open');
        });
        if (!dropdown || dropdown.dataset.disabled === 'true') return;
        // Item taps are handled by their own click listeners.
        if (e.target.closest('.exp-menu')) return;
        const willOpen = !dropdown.classList.contains('open');
        dropdown.classList.toggle('open', willOpen);
        if (willOpen) positionFixedMenu(dropdown, dropdown.querySelector('.exp-menu'));
    });
}
