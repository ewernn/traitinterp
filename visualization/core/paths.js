/**
 * PathBuilder - Single source of truth for repo paths.
 * Loads structure from config/paths.yaml
 *
 * Usage:
 *     await paths.load();
 *     paths.setExperiment('{experiment_name}');
 *
 *     // Get resolved path
 *     const vectorsDir = paths.get('extraction.vectors', { trait: 'cognitive_state/context' });
 *     // Returns: 'experiments/{experiment_name}/extraction/cognitive_state/context/vectors'
 *
 *     // Combine with pattern
 *     const vectorFile = paths.get('extraction.vectors', { trait }) + '/' + paths.get('patterns.vector', { method: 'probe', layer: 16 });
 *
 *     // Get raw template
 *     const tmpl = paths.template('extraction.vectors');
 *     // Returns: 'experiments/{experiment}/extraction/{trait}/vectors'
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
     * @param {string} subpath - Optional subpath (e.g., 'vectors', 'activations')
     * @returns {string}
     */
    extraction(trait, subpath = '') {
        const traitName = typeof trait === 'string' ? trait : trait.name;
        const basePath = this.get('extraction.trait', { trait: traitName });
        return subpath ? `/${basePath}/${subpath}` : `/${basePath}`;
    }

    /**
     * Get vector metadata path.
     * @param {string|Object} trait - Trait name or object
     * @param {string} method - Extraction method
     * @param {number} layer - Layer number
     * @returns {string}
     */
    vectorMetadata(trait, method, layer) {
        const traitName = typeof trait === 'string' ? trait : trait.name;
        const dir = this.get('extraction.vectors', { trait: traitName });
        const file = this.get('patterns.vector_metadata', { method, layer });
        return `/${dir}/${file}`;
    }

    /**
     * Get vector tensor path.
     * @param {string|Object} trait - Trait name or object
     * @param {string} method - Extraction method
     * @param {number} layer - Layer number
     * @returns {string}
     */
    vectorTensor(trait, method, layer) {
        const traitName = typeof trait === 'string' ? trait : trait.name;
        const dir = this.get('extraction.vectors', { trait: traitName });
        const file = this.get('patterns.vector', { method, layer });
        return `/${dir}/${file}`;
    }

    /**
     * Get inference path for a trait.
     * @param {string|Object} trait - Trait name or object
     * @param {string} subpath - Optional subpath
     * @returns {string}
     */
    inference(trait, subpath = '') {
        const traitName = typeof trait === 'string' ? trait : trait.name;
        const basePath = this.get('inference.trait', { trait: traitName });
        return subpath ? `/${basePath}/${subpath}` : `/${basePath}`;
    }

    /**
     * Get residual stream data path (projections across all layers).
     * @param {string|Object} trait - Trait name or object
     * @param {string} promptSet - Prompt set name (e.g., 'single_trait', 'multi_trait')
     * @param {number} promptId - Prompt ID within the set
     * @returns {string}
     */
    residualStreamData(trait, promptSet, promptId) {
        const traitName = typeof trait === 'string' ? trait : trait.name;
        const dir = this.get('inference.residual_stream', { trait: traitName, prompt_set: promptSet });
        const file = this.get('patterns.residual_stream_json', { prompt_id: promptId });
        return `/${dir}/${file}`;
    }

    /**
     * Get response data path (prompt/response text and tokens, shared across traits).
     * @param {string} promptSet - Prompt set name (e.g., 'single_trait', 'multi_trait')
     * @param {number} promptId - Prompt ID within the set
     * @returns {string}
     */
    responseData(promptSet, promptId) {
        return `/${this.get('inference.response_data', { prompt_set: promptSet, prompt_id: promptId })}`;
    }

    /**
     * Get prompt set file path.
     * @param {string} promptSet - Prompt set name (e.g., 'single_trait')
     * @returns {string}
     */
    promptSetFile(promptSet) {
        return `/${this.get('inference.prompt_set_file', { prompt_set: promptSet })}`;
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
     * Get activations metadata path.
     * @param {string|Object} trait - Trait name or object
     * @returns {string}
     */
    activationsMetadata(trait) {
        const traitName = typeof trait === 'string' ? trait : trait.name;
        const dir = this.get('extraction.activations', { trait: traitName });
        return `/${dir}/${this.get('patterns.metadata')}`;
    }

    /**
     * Get trait definition path.
     * @param {string|Object} trait - Trait name or object
     * @returns {string}
     */
    traitDefinition(trait) {
        const traitName = typeof trait === 'string' ? trait : trait.name;
        const dir = this.get('extraction.trait', { trait: traitName });
        return `/${dir}/${this.get('patterns.trait_definition')}`;
    }

    /**
     * Get responses path.
     * @param {string|Object} trait - Trait name or object
     * @param {string} polarity - 'pos' or 'neg'
     * @param {string} format - 'csv' or 'json'
     * @returns {string}
     */
    responses(trait, polarity, format = 'json') {
        const traitName = typeof trait === 'string' ? trait : trait.name;
        const dir = this.get('extraction.responses', { trait: traitName });
        const pattern = polarity === 'pos' ? 'patterns.pos_responses' : 'patterns.neg_responses';
        const file = this.get(pattern, { format });
        return `/${dir}/${file}`;
    }

    /**
     * Get shared prompts directory.
     * @returns {string}
     */
    promptsDir() {
        return `/${this.get('inference.prompts_dir')}`;
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
     * @param {string} subpath - Subpath within trait directory
     * @returns {string}
     */
    extractionFile(traitName, subpath) {
        const basePath = this.get('extraction.trait', { trait: traitName });
        return `/${basePath}/${subpath}`;
    }
}

// =========================================================================
// Helper functions (preserved from old API)
// =========================================================================

/**
 * Check if a trait has vectors.
 * @param {PathBuilder} pathBuilder - PathBuilder instance
 * @param {string|Object} trait - Trait name or object
 * @returns {Promise<boolean>}
 */
async function hasVectors(pathBuilder, trait) {
    try {
        const testUrl = pathBuilder.vectorMetadata(trait, 'probe', 16);
        const response = await fetch(testUrl, { method: 'HEAD' });
        return response.ok;
    } catch (e) {
        return false;
    }
}

/**
 * Detect response format for a trait.
 * @param {PathBuilder} pathBuilder - PathBuilder instance
 * @param {string|Object} trait - Trait name or object
 * @returns {Promise<string|null>} 'json' or null
 */
async function detectResponseFormat(pathBuilder, trait) {
    // All responses are JSON now - just verify it exists
    const jsonUrl = pathBuilder.responses(trait, 'pos', 'json');
    const response = await fetch(jsonUrl, { method: 'HEAD' });
    return response.ok ? 'json' : null;
}

// =========================================================================
// Singleton instance
// =========================================================================

const paths = new PathBuilder();

// Export to global scope
window.PathBuilder = PathBuilder;
window.paths = paths;
window.hasVectors = hasVectors;
window.detectResponseFormat = detectResponseFormat;
