/**
 * PathBuilder - Single source of truth for repo paths.
 * Loads structure from config/paths.yaml
 *
 * Usage:
 *     await paths.load();
 *     paths.setExperiment('gemma-2-2b');
 *
 *     // Vector paths (include position/component subdirs)
 *     paths.vectorTensor('chirp/refusal_v2', 'probe', 15, 'base')
 *     // Returns: '/experiments/gemma-2-2b/extraction/chirp/refusal_v2/base/vectors/response_all/residual/probe/layer15.pt'
 *
 *     // Activation paths
 *     paths.activationsMetadata('chirp/refusal_v2', 'base')
 *     // Returns: '/experiments/gemma-2-2b/extraction/chirp/refusal_v2/base/activations/response_all/residual/metadata.json'
 *
 *     // Steering paths
 *     paths.steeringResults('chirp/refusal_v2', 'instruct')
 *     // Returns: '/experiments/gemma-2-2b/steering/chirp/refusal_v2/instruct/response_all/steering/results.jsonl'
 *
 *     // Raw template access
 *     paths.get('extraction.vectors', { trait: 'chirp/refusal_v2', model_variant: 'base' })
 *     // Returns: 'experiments/gemma-2-2b/extraction/chirp/refusal_v2/base/vectors'
 */

class PathBuilder {
    constructor() {
        this.config = null;
        this.loaded = false;
        this._loadPromise = null;
        this.experimentName = null;
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
     * Load experiment config and cache it.
     * @returns {Promise<Object>} Experiment config with model_variants
     */
    async loadExperimentConfig() {
        if (!this.experimentName) {
            throw new Error('No experiment set. Call setExperiment() first.');
        }

        // Use cached config if available
        if (this._experimentConfig && this._experimentConfigName === this.experimentName) {
            return this._experimentConfig;
        }

        const configPath = `/${this.get('experiments.config')}`;
        const response = await fetch(configPath);
        if (!response.ok) {
            throw new Error(`Failed to load experiment config: ${response.status}`);
        }

        this._experimentConfig = await response.json();
        this._experimentConfigName = this.experimentName;
        return this._experimentConfig;
    }

    /**
     * Get list of model variants for current experiment.
     * @returns {Promise<string[]>} List of variant names
     */
    async listModelVariants() {
        const config = await this.loadExperimentConfig();
        return Object.keys(config.model_variants || {});
    }

    /**
     * Get default variant for a mode (extraction or application).
     * @param {string} mode - 'extraction' or 'application'
     * @returns {Promise<string>} Default variant name
     */
    async getDefaultVariant(mode = 'application') {
        const config = await this.loadExperimentConfig();
        return config.defaults?.[mode] || 'base';
    }

    /**
     * Get model variant config.
     * @param {string} variantName - Variant name (e.g., 'base', 'instruct')
     * @returns {Promise<Object>} Variant config with model and optional lora
     */
    async getModelVariant(variantName) {
        const config = await this.loadExperimentConfig();
        const variant = config.model_variants?.[variantName];
        if (!variant) {
            throw new Error(`Model variant '${variantName}' not found in experiment config`);
        }
        return { name: variantName, ...variant };
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

    /**
     * Get raw template string without substitution.
     * @param {string} key - Dot-separated key like 'extraction.vectors'
     * @returns {string} Raw template with {variables} intact
     */
    template(key) {
        if (!this.loaded) {
            throw new Error('PathBuilder not loaded. Call await paths.load() first.');
        }

        let node = this.config;
        for (const k of key.split('.')) {
            if (!(k in node)) {
                throw new Error(`Path key not found: '${key}' (failed at '${k}')`);
            }
            node = node[k];
        }

        return node;
    }

    /**
     * List all available path keys.
     * @param {string} prefix - Optional prefix to filter keys
     * @returns {string[]} List of dot-separated keys
     */
    listKeys(prefix = '') {
        if (!this.loaded) {
            throw new Error('PathBuilder not loaded. Call await paths.load() first.');
        }

        const keys = [];

        const collect = (node, currentPrefix = '') => {
            if (typeof node === 'object' && node !== null) {
                for (const [k, v] of Object.entries(node)) {
                    const newPrefix = currentPrefix ? `${currentPrefix}.${k}` : k;
                    if (typeof v === 'string') {
                        keys.push(newPrefix);
                    } else {
                        collect(v, newPrefix);
                    }
                }
            }
        };

        collect(this.config);

        if (prefix) {
            return keys.filter(k => k.startsWith(prefix));
        }
        return keys;
    }

    // =========================================================================
    // Convenience methods (match old API for easier migration)
    // =========================================================================

    /**
     * Get extraction path for a trait.
     * @param {string|Object} trait - Trait name or object with name property
     * @param {string} modelVariant - Model variant name (e.g., 'base', 'instruct')
     * @param {string} subpath - Optional subpath (e.g., 'vectors', 'activations')
     * @returns {string}
     */
    extraction(trait, modelVariant, subpath = '') {
        const traitName = typeof trait === 'string' ? trait : trait.name;
        const basePath = this.get('extraction.trait', { trait: traitName, model_variant: modelVariant });
        return subpath ? `/${basePath}/${subpath}` : `/${basePath}`;
    }

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

    /**
     * Get vector directory path.
     * @param {string|Object} trait - Trait name or object
     * @param {string} method - Extraction method
     * @param {string} modelVariant - Model variant name (e.g., 'base')
     * @param {string} position - Position string (default: 'response[:]')
     * @param {string} component - Component type (default: 'residual')
     * @returns {string} Path to vectors/{position}/{component}/{method}/
     */
    vectorDir(trait, method, modelVariant, position = 'response[:]', component = 'residual') {
        const traitName = typeof trait === 'string' ? trait : trait.name;
        const base = this.get('extraction.vectors', { trait: traitName, model_variant: modelVariant });
        const posDir = this.sanitizePosition(position);
        return `/${base}/${posDir}/${component}/${method}`;
    }

    /**
     * Get vector metadata path (per method directory).
     * @param {string|Object} trait - Trait name or object
     * @param {string} method - Extraction method
     * @param {string} modelVariant - Model variant name (e.g., 'base')
     * @param {string} position - Position string (default: 'response[:]')
     * @param {string} component - Component type (default: 'residual')
     * @returns {string}
     */
    vectorMetadata(trait, method, modelVariant, position = 'response[:]', component = 'residual') {
        return `${this.vectorDir(trait, method, modelVariant, position, component)}/metadata.json`;
    }

    /**
     * Get vector tensor path.
     * @param {string|Object} trait - Trait name or object
     * @param {string} method - Extraction method
     * @param {number} layer - Layer number
     * @param {string} modelVariant - Model variant name (e.g., 'base')
     * @param {string} position - Position string (default: 'response[:]')
     * @param {string} component - Component type (default: 'residual')
     * @returns {string}
     */
    vectorTensor(trait, method, layer, modelVariant, position = 'response[:]', component = 'residual') {
        return `${this.vectorDir(trait, method, modelVariant, position, component)}/layer${layer}.pt`;
    }

    /**
     * Get inference base path for a model variant.
     * @param {string} modelVariant - Model variant name (e.g., 'instruct')
     * @param {string} subpath - Optional subpath
     * @returns {string}
     */
    inference(modelVariant, subpath = '') {
        const basePath = this.get('inference.variant', { model_variant: modelVariant });
        return subpath ? `/${basePath}/${subpath}` : `/${basePath}`;
    }

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
     * Get prompt set file path.
     * @param {string} promptSet - Prompt set name (e.g., 'single_trait')
     * @returns {string}
     */
    promptSetFile(promptSet) {
        return `/${this.get('datasets.inference_prompt_set', { prompt_set: promptSet })}`;
    }

    /**
     * List available prompt sets.
     * @returns {string[]} Array of prompt set names
     */
    listPromptSets() {
        return ['single_trait', 'multi_trait', 'dynamic', 'adversarial', 'baseline', 'real_world'];
    }

    /**
     * Get extraction evaluation results path.
     * @returns {string}
     */
    extractionEvaluation() {
        return `/${this.get('extraction_eval.evaluation')}`;
    }

    /**
     * Get activations directory path.
     * @param {string|Object} trait - Trait name or object
     * @param {string} modelVariant - Model variant name (e.g., 'base')
     * @param {string} position - Position string (default: 'response[:]')
     * @param {string} component - Component type (default: 'residual')
     * @returns {string} Path to activations/{position}/{component}/
     */
    activationsDir(trait, modelVariant, position = 'response[:]', component = 'residual') {
        const traitName = typeof trait === 'string' ? trait : trait.name;
        const base = this.get('extraction.activations', { trait: traitName, model_variant: modelVariant });
        const posDir = this.sanitizePosition(position);
        return `/${base}/${posDir}/${component}`;
    }

    /**
     * Get activations metadata path.
     * @param {string|Object} trait - Trait name or object
     * @param {string} modelVariant - Model variant name (e.g., 'base')
     * @param {string} position - Position string (default: 'response[:]')
     * @param {string} component - Component type (default: 'residual')
     * @returns {string}
     */
    activationsMetadata(trait, modelVariant, position = 'response[:]', component = 'residual') {
        return `${this.activationsDir(trait, modelVariant, position, component)}/metadata.json`;
    }

    /**
     * Get steering directory path.
     * @param {string|Object} trait - Trait name or object
     * @param {string} modelVariant - Model variant name (e.g., 'instruct')
     * @param {string} position - Position string (default: 'response[:]')
     * @param {string} promptSet - Prompt set name (default: 'steering')
     * @returns {string} Path to steering/{trait}/{model_variant}/{position}/{prompt_set}/
     */
    steeringDir(trait, modelVariant, position = 'response[:]', promptSet = 'steering') {
        const traitName = typeof trait === 'string' ? trait : trait.name;
        const posDir = this.sanitizePosition(position);
        const base = this.get('steering.prompt_set', {
            trait: traitName,
            model_variant: modelVariant,
            position: posDir,
            prompt_set: promptSet
        });
        return `/${base}`;
    }

    /**
     * Get steering results path.
     * @param {string|Object} trait - Trait name or object
     * @param {string} modelVariant - Model variant name (e.g., 'instruct')
     * @param {string} position - Position string (default: 'response[:]')
     * @param {string} promptSet - Prompt set name (default: 'steering')
     * @returns {string}
     */
    steeringResults(trait, modelVariant, position = 'response[:]', promptSet = 'steering') {
        return `${this.steeringDir(trait, modelVariant, position, promptSet)}/results.jsonl`;
    }

    /**
     * Get trait definition path (from datasets, model-agnostic).
     * @param {string|Object} trait - Trait name or object
     * @returns {string}
     */
    traitDefinition(trait) {
        const traitName = typeof trait === 'string' ? trait : trait.name;
        return `/${this.get('datasets.trait_definition', { trait: traitName })}`;
    }

    /**
     * Get responses path.
     * @param {string|Object} trait - Trait name or object
     * @param {string} modelVariant - Model variant name (e.g., 'base')
     * @param {string} polarity - 'pos' or 'neg'
     * @param {string} format - 'csv' or 'json'
     * @returns {string}
     */
    responses(trait, modelVariant, polarity, format = 'json') {
        const traitName = typeof trait === 'string' ? trait : trait.name;
        const dir = this.get('extraction.responses', { trait: traitName, model_variant: modelVariant });
        const pattern = polarity === 'pos' ? 'patterns.pos_responses' : 'patterns.neg_responses';
        const file = this.get(pattern, { format });
        return `/${dir}/${file}`;
    }

    /**
     * Get shared prompts directory.
     * @returns {string}
     */
    promptsDir() {
        return `/${this.get('datasets.inference')}`;
    }

    // =========================================================================
    // Analysis paths
    // =========================================================================

    /**
     * Get analysis base directory.
     * @returns {string}
     */
    analysisBase() {
        return `/${this.get('analysis.base')}`;
    }

    /**
     * Get analysis index file.
     * @returns {string}
     */
    analysisIndex() {
        return `/${this.get('analysis.index')}`;
    }

    /**
     * Get per-token analysis data path.
     * @param {string} promptSet - Prompt set name
     * @param {number} promptId - Prompt ID within the set
     * @returns {string}
     */
    analysisPerToken(promptSet, promptId) {
        const dir = this.get('analysis.per_token', { prompt_set: promptSet });
        const file = this.get('patterns.per_token_json', { prompt_id: promptId });
        return `/${dir}/${file}`;
    }

    /**
     * Get analysis category directory.
     * @param {string} category - Analysis category name
     * @returns {string}
     */
    analysisCategory(category) {
        return `/${this.get('analysis.category', { category })}`;
    }

    /**
     * Get analysis file for a specific prompt.
     * @param {string} category - Analysis category name
     * @param {number} promptId - Prompt ID
     * @param {string} ext - File extension ('png' or 'json')
     * @returns {string}
     */
    analysisCategoryPrompt(category, promptId, ext = 'png') {
        const dir = this.get('analysis.category', { category });
        const pattern = ext === 'png' ? 'patterns.analysis_prompt_png' : 'patterns.analysis_prompt_json';
        const file = this.get(pattern, { prompt_id: promptId });
        return `/${dir}/${file}`;
    }

    /**
     * Get analysis file with custom filename.
     * @param {string} category - Analysis category name
     * @param {string} filename - Filename without extension
     * @param {string} ext - File extension ('png' or 'json')
     * @returns {string}
     */
    analysisCategoryNamed(category, filename, ext = 'png') {
        const dir = this.get('analysis.category', { category });
        const pattern = ext === 'png' ? 'patterns.analysis_named_png' : 'patterns.analysis_named_json';
        const file = this.get(pattern, { filename });
        return `/${dir}/${file}`;
    }

    /**
     * Get extraction file path (generic).
     * @param {string} traitName - Trait name (e.g., 'cognitive_state/context')
     * @param {string} modelVariant - Model variant name (e.g., 'base')
     * @param {string} subpath - Subpath within trait directory
     * @returns {string}
     */
    extractionFile(traitName, modelVariant, subpath) {
        const basePath = this.get('extraction.trait', { trait: traitName, model_variant: modelVariant });
        return `/${basePath}/${subpath}`;
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
}

// =========================================================================
// Singleton instance
// =========================================================================

const paths = new PathBuilder();

// Export to global scope
window.PathBuilder = PathBuilder;
window.paths = paths;
