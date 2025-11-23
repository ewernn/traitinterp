// Centralized Data Loading for Trait Interpretation Visualization

class DataLoader {
    /**
     * Fetch residual stream data (projections across all layers)
     * @param {Object} trait - Trait object with name property (category/trait_name format)
     * @param {string} promptSet - Prompt set name (e.g., 'single_trait', 'multi_trait')
     * @param {number} promptId - Prompt ID within the set
     * @returns {Promise<Object>} - Data with projections and dynamics
     */
    static async fetchResidualStream(trait, promptSet, promptId) {
        const url = window.paths.residualStreamData(trait, promptSet, promptId);
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to fetch residual stream data for ${trait.name} ${promptSet}/${promptId}`);
        }
        return await response.json();
    }

    /**
     * BACKWARDS COMPATIBILITY: Fetch all-layers data using old index-based API.
     * @deprecated Use fetchResidualStream(trait, promptSet, promptId) instead
     * @param {Object} trait - Trait object with name property
     * @param {number} promptNum - Old-style prompt number (0-based index)
     * @returns {Promise<Object>} - Data with projections and dynamics
     */
    static async fetchAllLayers(trait, promptNum) {
        const url = window.paths.allLayersData(trait, promptNum);
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to fetch all-layers data for ${trait.name} prompt ${promptNum}`);
        }
        return await response.json();
    }

    /**
     * Fetch layer-internals data (detailed internal states for a specific layer)
     * @param {Object} trait - Trait object with name property (category/trait_name format)
     * @param {string} promptSet - Prompt set name
     * @param {number} promptId - Prompt ID within the set
     * @param {number} layer - Layer number (default: 16)
     * @returns {Promise<Object>} - Data with attention heads, MLP activations, etc.
     */
    static async fetchLayerInternals(trait, promptSet, promptId, layer = 16) {
        const url = window.paths.layerInternalsData(trait, promptSet, promptId, layer);
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to fetch layer-internals data for ${trait.name} ${promptSet}/${promptId} layer ${layer}`);
        }
        return await response.json();
    }

    /**
     * Fetch a prompt set definition (name, description, prompts array)
     * @param {string} promptSet - Prompt set name (e.g., 'single_trait')
     * @returns {Promise<Object>} - Prompt set with {name, description, prompts: [{id, text, note}]}
     */
    static async fetchPromptSet(promptSet) {
        const url = window.paths.promptSetFile(promptSet);
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to fetch prompt set: ${promptSet}`);
        }
        return await response.json();
    }

    /**
     * Fetch all prompt sets
     * @returns {Promise<Object>} - Object mapping set name to set data
     */
    static async fetchAllPromptSets() {
        const setNames = window.paths.listPromptSets();
        const results = {};

        await Promise.all(setNames.map(async (name) => {
            try {
                results[name] = await this.fetchPromptSet(name);
            } catch (e) {
                console.warn(`Failed to load prompt set ${name}:`, e.message);
                results[name] = null;
            }
        }));

        return results;
    }

    /**
     * Fetch vector metadata for a specific method and layer
     * @param {Object} trait - Trait object with name property (category/trait_name format)
     * @param {string} method - Extraction method (mean_diff, probe, ica, gradient)
     * @param {number} layer - Layer number
     * @returns {Promise<Object>} - Vector metadata
     */
    static async fetchVectorMetadata(trait, method, layer) {
        const url = window.paths.vectorMetadata(trait, method, layer);
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to fetch vector metadata for ${trait.name} ${method} layer ${layer}`);
        }
        return await response.json();
    }

    /**
     * Fetch all vector metadata for a trait (all methods and layers)
     * @param {Object} trait - Trait object with name property
     * @param {Array<string>} methods - Array of method names
     * @param {Array<number>} layers - Array of layer numbers
     * @returns {Promise<Object>} - Object mapping method_layerN to metadata
     */
    static async fetchAllVectorMetadata(trait, methods, layers) {
        const results = {};
        const promises = [];

        for (const method of methods) {
            for (const layer of layers) {
                const key = `${method}_layer${layer}`;
                const promise = this.fetchVectorMetadata(trait, method, layer)
                    .then(data => {
                        results[key] = data;
                    })
                    .catch(e => {
                        console.warn(`No vector metadata for ${trait.name} ${key}:`, e.message);
                        results[key] = null;
                    });
                promises.push(promise);
            }
        }

        await Promise.all(promises);
        return results;
    }

    /**
     * Fetch cross-distribution analysis data
     * @param {string} traitBaseName - Base name of trait (without _natural suffix)
     * @returns {Promise<Object>} - Cross-distribution results
     */
    static async fetchCrossDistribution(traitBaseName) {
        const url = `${window.paths.get('validation.base')}/${traitBaseName}_full_4x4_results.json`;
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to fetch cross-distribution data for ${traitBaseName}`);
        }
        return await response.json();
    }

    /**
     * Discover available prompts for a trait in a specific prompt set
     * @param {Object} trait - Trait object with name property
     * @param {string} promptSet - Prompt set name
     * @returns {Promise<number[]>} - Array of available prompt IDs
     */
    static async discoverPrompts(trait, promptSet) {
        try {
            const indices = [];
            for (let i = 1; i <= 20; i++) { // Check up to 20 prompts per set
                try {
                    const url = window.paths.residualStreamData(trait, promptSet, i);
                    const response = await fetch(url, { method: 'HEAD' });
                    if (response.ok) {
                        indices.push(i);
                    }
                } catch (e) {
                    // Continue checking - files may not be sequential
                }
            }
            return indices;
        } catch (e) {
            console.warn(`Failed to discover prompts for ${trait.name} in ${promptSet}:`, e);
            return [];
        }
    }

    /**
     * Fetch JSON file for preview
     * @param {Object} trait - Trait object with name property (category/trait_name format)
     * @param {string} type - Type of JSON (trait_definition, activations_metadata, pos, neg)
     * @returns {Promise<Object>} - JSON data
     */
    static async fetchJSON(trait, type) {
        let url;
        const traitName = typeof trait === 'string' ? trait : trait.name;
        const format = 'json';

        if (type === 'trait_definition') {
            url = window.paths.traitDefinition(traitName);
        } else if (type === 'activations_metadata') {
            url = window.paths.activationsMetadata(traitName);
        } else if (type === 'pos' || type === 'neg') {
            url = window.paths.responses(traitName, type, format);
        } else {
            throw new Error(`Unknown JSON type: ${type}`);
        }

        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to fetch JSON for ${traitName} ${type}`);
        }
        return await response.json();
    }

    /**
     * Fetch CSV file for preview
     * @param {Object} trait - Trait object with name property (category/trait_name format)
     * @param {string} category - Category (pos or neg)
     * @param {number} limit - Maximum rows to parse (default: 10)
     * @returns {Promise<Object>} - Parsed CSV data with Papa Parse
     */
    static async fetchCSV(trait, category, limit = 10) {
        const url = window.paths.responses(trait, category, 'csv');
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to fetch CSV for ${trait.name} ${category}`);
        }
        const text = await response.text();
        const parsed = Papa.parse(text, { header: true });
        return {
            data: parsed.data.slice(0, limit),
            total: parsed.data.length,
            headers: parsed.data.length > 0 ? Object.keys(parsed.data[0]) : []
        };
    }

    /**
     * Check if vector extraction exists for a trait
     * @param {Object} trait - Trait object with name property (category/trait_name format)
     * @returns {Promise<boolean>} - True if vectors exist
     */
    static async checkVectorsExist(trait) {
        try {
            const testUrl = window.paths.vectorMetadata(trait, 'probe', 16);
            const response = await fetch(testUrl, { method: 'HEAD' });
            return response.ok;
        } catch (e) {
            return false;
        }
    }

    /**
     * Fetch SAE features if available
     * @param {Object} trait - Trait object with name property (category/trait_name format)
     * @param {string} promptSet - Prompt set name
     * @param {number} promptId - Prompt ID
     * @param {number} layer - Layer number
     * @returns {Promise<Object>} - SAE feature activations
     */
    static async fetchSAEFeatures(trait, promptSet, promptId, layer = 16) {
        const dir = window.paths.get('inference.trait', { trait: trait.name });
        const url = `${dir}/sae_features/${promptSet}/${promptId}_L${layer}_sae.pt`;
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to fetch SAE features for ${trait.name} ${promptSet}/${promptId} layer ${layer}`);
        }
        // Note: PT files need special handling - this would return blob
        return await response.blob();
    }
}

// Export to global scope
window.DataLoader = DataLoader;
