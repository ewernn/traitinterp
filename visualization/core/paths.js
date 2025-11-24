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
        return subpath ? `${basePath}/${subpath}` : basePath;
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
        return `${dir}/${file}`;
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
        return `${dir}/${file}`;
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
        return subpath ? `${basePath}/${subpath}` : basePath;
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
        return `${dir}/${file}`;
    }

    /**
     * Get layer-internals data path (detailed single layer capture).
     * @param {string|Object} trait - Trait name or object
     * @param {string} promptSet - Prompt set name
     * @param {number} promptId - Prompt ID within the set
     * @param {number} layer - Layer number
     * @returns {string}
     */
    layerInternalsData(trait, promptSet, promptId, layer = 16) {
        const traitName = typeof trait === 'string' ? trait : trait.name;
        const dir = this.get('inference.layer_internals', { trait: traitName, prompt_set: promptSet });
        const file = this.get('patterns.layer_internals_json', { prompt_id: promptId, layer });
        return `${dir}/${file}`;
    }

    /**
     * Get prompt set file path.
     * @param {string} promptSet - Prompt set name (e.g., 'single_trait')
     * @returns {string}
     */
    promptSetFile(promptSet) {
        return this.get('inference.prompt_set_file', { prompt_set: promptSet });
    }

    /**
     * List available prompt sets.
     * @returns {string[]} Array of prompt set names
     */
    listPromptSets() {
        return ['single_trait', 'multi_trait', 'dynamic', 'adversarial', 'baseline', 'real_world'];
    }

    /**
     * BACKWARDS COMPATIBILITY: Map old prompt index to new (promptSet, promptId) tuple.
     * Old structure used indices 0-50 across all prompts.
     * New structure uses separate sets with IDs 1-N within each set.
     * @param {number} oldIndex - Old-style prompt index (0-based)
     * @returns {{promptSet: string, promptId: number}}
     */
    _mapOldPromptIndex(oldIndex) {
        // Mapping: 0-9 = single_trait (1-10), 10-19 = multi_trait (1-10), etc.
        const ranges = [
            { set: 'single_trait', count: 10 },
            { set: 'multi_trait', count: 10 },
            { set: 'dynamic', count: 8 },
            { set: 'adversarial', count: 8 },
            { set: 'baseline', count: 5 },
            { set: 'real_world', count: 10 }
        ];

        let offset = 0;
        for (const { set, count } of ranges) {
            if (oldIndex < offset + count) {
                return { promptSet: set, promptId: oldIndex - offset + 1 };
            }
            offset += count;
        }
        // Fallback to single_trait
        return { promptSet: 'single_trait', promptId: 1 };
    }

    /**
     * BACKWARDS COMPATIBILITY: Get all-layers data using old index-based API.
     * @deprecated Use residualStreamData(trait, promptSet, promptId) instead
     * @param {string|Object} trait - Trait name or object
     * @param {number} promptNum - Old-style prompt number (0-based index)
     * @returns {string}
     */
    allLayersData(trait, promptNum) {
        const { promptSet, promptId } = this._mapOldPromptIndex(promptNum);
        return this.residualStreamData(trait, promptSet, promptId);
    }

    /**
     * Get extraction evaluation results path.
     * @returns {string}
     */
    extractionEvaluation() {
        return this.get('extraction_eval.evaluation');
    }

    /**
     * Get activations metadata path.
     * @param {string|Object} trait - Trait name or object
     * @returns {string}
     */
    activationsMetadata(trait) {
        const traitName = typeof trait === 'string' ? trait : trait.name;
        const dir = this.get('extraction.activations', { trait: traitName });
        return `${dir}/${this.get('patterns.metadata')}`;
    }

    /**
     * Get trait definition path.
     * @param {string|Object} trait - Trait name or object
     * @returns {string}
     */
    traitDefinition(trait) {
        const traitName = typeof trait === 'string' ? trait : trait.name;
        const dir = this.get('extraction.trait', { trait: traitName });
        return `${dir}/${this.get('patterns.trait_definition')}`;
    }

    /**
     * Get responses path.
     * @param {string|Object} trait - Trait name or object
     * @param {string} polarity - 'pos' or 'neg'
     * @param {string} format - 'csv' or 'json'
     * @returns {string}
     */
    responses(trait, polarity, format = 'csv') {
        const traitName = typeof trait === 'string' ? trait : trait.name;
        const dir = this.get('extraction.responses', { trait: traitName });
        const pattern = polarity === 'pos' ? 'patterns.pos_responses' : 'patterns.neg_responses';
        const file = this.get(pattern, { format });
        return `${dir}/${file}`;
    }

    /**
     * Get shared prompts directory.
     * @returns {string}
     */
    promptsDir() {
        return this.get('inference.prompts_dir');
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
 * @returns {Promise<string|null>} 'csv', 'json', or null
 */
async function detectResponseFormat(pathBuilder, trait) {
    const traitName = typeof trait === 'string' ? trait : trait.name;
    const isNatural = traitName.includes('_natural');

    const primaryExt = isNatural ? 'json' : 'csv';
    const primaryUrl = pathBuilder.responses(trait, 'pos', primaryExt);
    const primaryCheck = await fetch(primaryUrl, { method: 'HEAD' });
    if (primaryCheck.ok) return primaryExt;

    const secondaryExt = isNatural ? 'csv' : 'json';
    const secondaryUrl = pathBuilder.responses(trait, 'pos', secondaryExt);
    const secondaryCheck = await fetch(secondaryUrl, { method: 'HEAD' });
    if (secondaryCheck.ok) return secondaryExt;

    return null;
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
