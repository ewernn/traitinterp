/**
 * Zoomies Router
 * URL <-> State synchronization.
 */

window.zoomies = window.zoomies || {};

/**
 * Load state from URL query params
 */
window.zoomies.loadFromURL = function() {
    const params = new URLSearchParams(window.location.search);

    const updates = {};

    // Tab
    const tab = params.get('tab');
    if (tab && ['main', 'overview', 'dev'].includes(tab)) {
        updates.tab = tab;
    }

    // Mode
    const mode = params.get('mode');
    if (mode && ['extraction', 'inference'].includes(mode)) {
        updates.mode = mode;
    }

    // Token scope
    const tokens = params.get('tokens');
    if (tokens === 'all') {
        updates.tokenScope = 'all';
    } else if (tokens !== null) {
        const t = parseInt(tokens, 10);
        if (!isNaN(t)) updates.tokenScope = t;
    }

    // Layer scope
    const layer = params.get('layer');
    if (layer === 'all') {
        updates.layerScope = 'all';
    } else if (layer !== null) {
        const l = parseInt(layer, 10);
        if (!isNaN(l) && l >= 0 && l < 26) updates.layerScope = l;
    }

    // Prompt
    const set = params.get('set');
    if (set) updates.promptSet = set;

    const id = params.get('id');
    if (id !== null) {
        const i = parseInt(id, 10);
        if (!isNaN(i)) updates.promptId = i;
    }

    // Experiment
    const exp = params.get('exp');
    if (exp) updates.experiment = exp;

    // Traits (comma-separated)
    const traits = params.get('traits');
    if (traits) {
        updates.selectedTraits = traits.split(',').filter(t => t.length > 0);
    }

    // Apply without re-rendering (init will render)
    Object.assign(window.zoomies.state, updates);
};

/**
 * Update URL from current state
 * Disabled - keep URL clean
 */
window.zoomies.updateURL = function() {
    // No-op: URL sync disabled for cleaner URLs
};

// Handle browser back/forward
window.addEventListener('popstate', () => {
    window.zoomies.loadFromURL();
    window.zoomies.render();
});
