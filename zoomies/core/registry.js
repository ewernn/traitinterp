/**
 * Zoomies Registry
 * Plugin registration for explanations, fetchers, and renderers.
 */

window.zoomies = window.zoomies || {};

window.zoomies.registry = {
    explanations: {},
    fetchers: {},
    renderers: {},

    /**
     * Register an explanation for a position
     * @param {string} key - Position key, e.g., 'extraction:all', 'inference:all:layer'
     * @param {Object} explanation - { title: string, content: string | function }
     */
    explanation(key, explanation) {
        this.explanations[key] = explanation;
    },

    /**
     * Register a data fetcher for a position
     * @param {string} key - Position key
     * @param {Function} fetcher - async (state) => data
     */
    fetcher(key, fetcher) {
        this.fetchers[key] = fetcher;
    },

    /**
     * Register a renderer for a position
     * @param {string} key - Position key
     * @param {Function} renderer - (data, container, state) => void
     */
    renderer(key, renderer) {
        this.renderers[key] = renderer;
    },

    /**
     * Get explanation for current position
     * @param {string} key - Position key
     * @param {Object} state - Current state (for template substitution)
     */
    getExplanation(key, state = {}) {
        const expl = this.explanations[key];
        if (!expl) return null;

        // Handle dynamic content
        const title = typeof expl.title === 'function' ? expl.title(state) : expl.title;
        const content = typeof expl.content === 'function' ? expl.content(state) : expl.content;

        return { title, content };
    },

    /**
     * Get fetcher for current position
     * @param {string} key - Position key
     */
    getFetcher(key) {
        // Try exact match first
        if (this.fetchers[key]) return this.fetchers[key];

        // Fall back to base fetcher (e.g., 'inference' for all inference positions)
        const parts = key.split(':');
        if (this.fetchers[parts[0]]) return this.fetchers[parts[0]];

        return null;
    },

    /**
     * Get renderer for current position
     * @param {string} key - Position key
     */
    getRenderer(key) {
        return this.renderers[key] || null;
    },

    /**
     * List all registered position keys
     */
    listKeys() {
        const keys = new Set([
            ...Object.keys(this.explanations),
            ...Object.keys(this.fetchers),
            ...Object.keys(this.renderers),
        ]);
        return Array.from(keys).sort();
    },
};
