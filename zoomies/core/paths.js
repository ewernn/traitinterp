/**
 * Zoomies Path Builder
 * Loads path templates from config/paths.yaml.
 */

window.zoomies = window.zoomies || {};

window.zoomies.paths = {
    templates: null,
    experiment: null,

    /**
     * Load path templates from paths.yaml
     */
    async load() {
        if (this.templates) return this.templates;

        try {
            const resp = await fetch('/config/paths.yaml');
            const yaml = await resp.text();
            this.templates = jsyaml.load(yaml);
            return this.templates;
        } catch (e) {
            console.error('Failed to load paths.yaml:', e);
            return null;
        }
    },

    /**
     * Set current experiment for path building
     */
    setExperiment(exp) {
        this.experiment = exp;
    },

    /**
     * Get a path template and fill in variables
     * @param {string} key - Dot-separated key, e.g., 'extraction.vectors'
     * @param {Object} vars - Variables to substitute
     */
    get(key, vars = {}) {
        if (!this.templates) {
            console.warn('Paths not loaded yet');
            return null;
        }

        // Navigate to the template
        const parts = key.split('.');
        let template = this.templates;
        for (const part of parts) {
            template = template?.[part];
        }

        if (typeof template !== 'string') {
            console.warn(`Path template not found: ${key}`);
            return null;
        }

        // Add experiment if not provided
        const allVars = {
            experiment: this.experiment,
            ...vars
        };

        // Substitute variables
        return template.replace(/\{(\w+)\}/g, (_, name) => {
            return allVars[name] ?? `{${name}}`;
        });
    },

    /**
     * Build common paths
     */
    extractionEvaluation() {
        return this.get('extraction_eval.evaluation');
    },

    crossTraitMatrix() {
        return this.get('extraction_eval.cross_trait_matrix');
    },

    residualStreamData(trait, promptSet, promptId) {
        const dir = this.get('inference.residual_stream', { trait, prompt_set: promptSet });
        return `${dir}/${promptId}.json`;
    },

    vectorMetadata(trait, method, layer) {
        const dir = this.get('extraction.vectors', { trait });
        return `${dir}/${method}_layer${layer}_metadata.json`;
    },

    promptSetFile(promptSet) {
        return this.get('inference.prompt_set_file', { prompt_set: promptSet });
    },
};

// Load paths on module load
window.zoomies.paths.load();
