/**
 * PathBuilder - Single source of truth for repo paths.
 * Loads structure from config/paths.yaml
 *
 * Usage:
 *     await paths.load();
 *     paths.setExperiment('gemma-2-2b');
 *
 *     // Template access
 *     paths.get('extraction.vectors', { trait: 'chirp/refusal_v2', model_variant: 'base' })
 *     // Returns: 'experiments/gemma-2-2b/extraction/chirp/refusal_v2/base/vectors'
 */

class PathBuilder {
    constructor() {
        this.config = null;
        this.loaded = false;
        this._loadPromise = null;
        this.experimentName = null;

        // Model config state
        this._modelConfig = null;
        this._modelId = null;
        this._modelConfigLoaded = false;
        this._modelConfigCache = {};
    }

    /**
     * Load config from YAML file.
     * Safe to call multiple times - will only load once.
     */
    async load() {
        if (this.loaded) return;
        if (this._loadPromise) return this._loadPromise;

        this._loadPromise = (async () => {
            const response = await fetch('/config/paths.yaml');
            if (!response.ok) {
                throw new Error(`Failed to load paths.yaml: ${response.status}`);
            }
            const yamlText = await response.text();
            this.config = jsyaml.load(yamlText);
            this.loaded = true;
        })();

        return this._loadPromise;
    }

    /**
     * Set the current experiment (used as default for {experiment} variable).
     * @param {string} name - Experiment name (e.g., '{experiment_name}')
     */
    setExperiment(name) {
        this.experimentName = name;
    }

    /**
     * Get current experiment name.
     * @returns {string|null}
     */
    getExperiment() {
        return this.experimentName;
    }

    /**
     * Get a path by key with variable substitution.
     * @param {string} key - Dot-separated key like 'extraction.vectors'
     * @param {Object} variables - Values for template variables
     * @returns {string} Path with variables substituted
     *
     * @example
     * paths.get('extraction.vectors', { trait: 'cognitive_state/context' })
     * // Returns: 'experiments/{experiment_name}/extraction/cognitive_state/context/vectors'
     */
    get(key, variables = {}) {
        if (!this.loaded) {
            throw new Error('PathBuilder not loaded. Call await paths.load() first.');
        }

        // Auto-inject experiment if set and not provided
        if (this.experimentName && !variables.experiment) {
            variables = { experiment: this.experimentName, ...variables };
        }

        // Navigate to key
        let node = this.config;
        for (const k of key.split('.')) {
            if (!(k in node)) {
                throw new Error(`Path key not found: '${key}' (failed at '${k}')`);
            }
            node = node[k];
        }

        if (typeof node !== 'string') {
            throw new Error(`Path key '${key}' is not a template string`);
        }

        // Substitute variables
        let result = node;
        for (const [varName, value] of Object.entries(variables)) {
            result = result.replaceAll(`{${varName}}`, String(value));
        }

        // Warn on unsubstituted variables (likely caller forgot to pass them)
        if (result.includes('{')) {
            const missing = [...result.matchAll(/\{(\w+)\}/g)].map(m => m[1]);
            console.warn(`Unsubstituted variables in path key '${key}':`, missing);
        }

        return result;
    }

    // =========================================================================
    // Position helpers
    // =========================================================================

    /**
     * Sanitize position string to filesystem-safe directory name.
     * Matches Python's sanitize_position() in utils/paths.py.
     *
     * Examples:
     *   response[:]  -> response_all
     *   response[-5:] -> response_-5_
     *   prompt[-3:] -> prompt_-3_
     *
     * @param {string} position - Position string like 'response[:]'
     * @returns {string} Sanitized position for use in paths
     */
    sanitizePosition(position) {
        return position
            .replace('[:]', '_all')
            .replace('[', '_')
            .replace(']', '')
            .replace(':', '_');
    }

    /**
     * Desanitize filesystem directory name back to position string.
     * Matches Python's desanitize_position() in utils/paths.py.
     *
     * Examples:
     *   response_all -> response[:]
     *   response_-5_ -> response[-5:]
     *   prompt_-3_ -> prompt[-3:]
     *
     * @param {string} sanitized - Sanitized position like 'response_all'
     * @returns {string} Position string like 'response[:]'
     */
    desanitizePosition(sanitized) {
        // Handle _all suffix (represents [:])
        if (sanitized.endsWith('_all')) {
            const prefix = sanitized.slice(0, -4);
            return `${prefix}[:]`;
        }

        // Handle other patterns: {frame}_{slice}
        // Trailing _ means there was a : at the end (open slice)
        const idx = sanitized.indexOf('_');
        if (idx !== -1) {
            const frame = sanitized.slice(0, idx);
            let slicePart = sanitized.slice(idx + 1);
            if (slicePart.endsWith('_')) {
                return `${frame}[${slicePart.slice(0, -1)}:]`;
            } else {
                return `${frame}[${slicePart}]`;
            }
        }

        return sanitized;  // Fallback
    }

    /**
     * Format position for short display (e.g., in chart legends).
     *
     * Examples:
     *   response[:]  -> @resp
     *   response[-5:] -> @resp[-5:]
     *   prompt[-3:] -> @p[-3:]
     *   response_all -> @resp (also accepts sanitized form)
     *
     * @param {string} position - Position string (sanitized or canonical)
     * @returns {string} Short display string
     */
    formatPositionDisplay(position) {
        // First desanitize if needed
        const canonical = position.includes('[') ? position : this.desanitizePosition(position);

        // Convert to short form
        if (canonical === 'response[:]') return '@resp';
        if (canonical === 'prompt[:]') return '@prompt';
        if (canonical === 'all[:]') return '@all';

        return canonical
            .replace('response', '@resp')
            .replace('prompt', '@p');
    }

    // =========================================================================
    // Convenience methods
    // =========================================================================

    /**
     * Get residual stream data path (projections across all layers).
     * @param {string|Object} trait - Trait name or object
     * @param {string} promptSet - Prompt set name (e.g., 'single_trait', 'multi_trait')
     * @param {number} promptId - Prompt ID within the set
     * @param {string} modelVariant - Model variant name (e.g., 'instruct')
     * @returns {string}
     */
    residualStreamData(trait, promptSet, promptId, modelVariant) {
        const traitName = typeof trait === 'string' ? trait : trait.name;
        // Use API endpoint which combines individual vector files on-the-fly
        const [category, name] = traitName.split('/');
        return `/api/experiments/${this.experimentName}/inference/${modelVariant}/projections/${category}/${name}/${promptSet}/${promptId}`;
    }

    /**
     * Get response data path (prompt/response text and tokens, shared across traits).
     * @param {string} promptSet - Prompt set name (e.g., 'single_trait', 'multi_trait')
     * @param {number} promptId - Prompt ID within the set
     * @param {string} modelVariant - Model variant name (e.g., 'instruct')
     * @returns {string}
     */
    responseData(promptSet, promptId, modelVariant) {
        return `/${this.get('inference.response_data', { prompt_set: promptSet, prompt_id: promptId, model_variant: modelVariant })}`;
    }

    /**
     * Get extraction evaluation results path.
     * @returns {string}
     */
    extractionEvaluation() {
        return `/${this.get('extraction_eval.evaluation')}`;
    }

    /**
     * Get logit lens results path for a trait.
     * @param {string|Object} trait - Trait name or object
     * @param {string} modelVariant - Model variant name (e.g., 'base')
     * @returns {string}
     */
    logitLens(trait, modelVariant) {
        const traitName = typeof trait === 'string' ? trait : trait.name;
        return `/${this.get('extraction.logit_lens', { trait: traitName, model_variant: modelVariant })}`;
    }

    // =========================================================================
    // Model config
    // =========================================================================

    /**
     * Load model config from YAML file.
     * @param {string} modelId - Model identifier (e.g., 'gemma-2-2b-it')
     */
    async _loadModelConfigById(modelId) {
        // Normalize: google/gemma-2-2b-it -> gemma-2-2b-it
        if (modelId.includes('/')) {
            modelId = modelId.split('/').pop().toLowerCase();
        }

        // Return cached if same model
        if (this._modelId === modelId && this._modelConfigLoaded) {
            return this._modelConfig;
        }

        // Check cache
        if (this._modelConfigCache[modelId]) {
            this._modelConfig = this._modelConfigCache[modelId];
            this._modelId = modelId;
            this._modelConfigLoaded = true;
            return this._modelConfig;
        }

        const response = await fetch(`/config/models/${modelId}.yaml`);
        if (!response.ok) {
            throw new Error(`Failed to load model config for '${modelId}': ${response.status}`);
        }

        const yamlText = await response.text();
        this._modelConfig = jsyaml.load(yamlText);
        this._modelId = modelId;
        this._modelConfigLoaded = true;
        this._modelConfigCache[modelId] = this._modelConfig;

        return this._modelConfig;
    }

    /**
     * Load model config for an experiment.
     * Reads experiment's config.json to determine model, then loads model config.
     * @param {string} experiment - Experiment name
     */
    async loadModelConfig(experiment) {
        const response = await fetch(`/experiments/${experiment}/config.json`);
        if (!response.ok) {
            throw new Error(`Failed to load experiment config for '${experiment}': ${response.status}`);
        }

        const expConfig = await response.json();
        const variant = expConfig.defaults.application;
        const modelId = expConfig.model_variants[variant].model;

        return this._loadModelConfigById(modelId);
    }

    /**
     * Get a model config value by dot-separated key.
     * @param {string} key - Dot-separated key like 'sae.available'
     * @returns {*} Config value
     */
    getModelConfig(key) {
        if (!this._modelConfigLoaded) {
            throw new Error('Model config not loaded. Call await paths.loadModelConfig() first.');
        }

        let value = this._modelConfig;
        for (const k of key.split('.')) {
            if (value === undefined || value === null) return undefined;
            value = value[k];
        }
        return value;
    }

    getNumLayers() {
        return this.getModelConfig('num_hidden_layers');
    }

}

// =========================================================================
// Singleton instance
// =========================================================================

const paths = new PathBuilder();

// Export to global scope
window.paths = paths;
