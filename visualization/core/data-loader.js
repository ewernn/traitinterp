// Centralized Data Loading for Trait Interpretation Visualization

class DataLoader {
    /**
     * Fetch Tier 2 data (residual stream activations for all layers)
     * @param {Object} trait - Trait object with name property (category/trait_name format)
     * @param {number} promptNum - Prompt number
     * @returns {Promise<Object>} - Data with projections and logit lens
     */
    static async fetchTier2(trait, promptNum) {
        const url = window.paths.tier2Data(trait, promptNum);
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to fetch tier 2 data for ${trait.name} prompt ${promptNum}`);
        }
        return await response.json();
    }

    /**
     * Fetch Tier 3 data (layer internal states for a specific layer)
     * @param {Object} trait - Trait object with name property (category/trait_name format)
     * @param {number} promptNum - Prompt number
     * @param {number} layer - Layer number (default: 16)
     * @returns {Promise<Object>} - Data with attention heads, MLP activations, etc.
     */
    static async fetchTier3(trait, promptNum, layer = 16) {
        const url = window.paths.tier3Data(trait, promptNum, layer);
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to fetch tier 3 data for ${trait.name} prompt ${promptNum} layer ${layer}`);
        }
        return await response.json();
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
     * Fetch prompt data (shared across all traits)
     * @param {string} promptSet - The name of the prompt set.
     * @param {number} promptIdx - Prompt index (0, 1, 2, ...)
     * @returns {Promise<Object>} - Shared prompt data
     */
    static async fetchPrompt(promptSet, promptIdx) {
        const dir = window.paths.get('inference.raw_activations', { prompt_set: promptSet });
        const file = window.paths.get('patterns.prompt_json', { index: promptIdx });
        const url = `${dir}/${file}`;
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to fetch prompt ${promptIdx} from set ${promptSet}`);
        }
        return await response.json();
    }

    /**
     * Fetch trait projections for a specific prompt
     * @param {Object} trait - Trait object with name property (category/trait_name format)
     * @param {string} promptSet - The name of the prompt set.
     * @param {number} promptIdx - Prompt index
     * @returns {Promise<Object>} - Trait-specific projection data
     */
    static async fetchProjections(trait, promptSet, promptIdx) {
        const dir = window.paths.get('inference.residual_stream', { trait: trait.name });
        const file = window.paths.get('patterns.prompt_json', { index: promptIdx });
        const url = `${dir}/${file}`;
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to fetch projections for ${trait.name} prompt ${promptIdx}`);
        }
        return await response.json();
    }

    /**
     * Fetch combined inference data
     * Combines prompt + projections into single object
     * @param {Object} trait - Trait object with name property
     * @param {string} promptSet - The name of the prompt set.
     * @param {number} promptIdx - Prompt index
     * @returns {Promise<Object>} - Combined data
     */
    static async fetchInferenceCombined(trait, promptSet, promptIdx) {
        const prompt = await this.fetchPrompt(promptSet, promptIdx);
        const projections = await this.fetchProjections(trait, promptSet, promptIdx);

        return {
            prompt: prompt.prompt,
            response: prompt.response,
            tokens: prompt.tokens,
            trait_scores: projections.projections, // Assuming projections has this structure
            dynamics: projections.dynamics // Assuming projections has this
        };
    }

    /**
     * Discover available prompts
     * @param {string} promptSet - The name of the prompt set to discover.
     * @returns {Promise<number[]>} - Array of available prompt indices
     */
    static async discoverPrompts(promptSet) {
        try {
            const indices = [];
            for (let i = 0; i < 200; i++) { // Check up to 200 prompts
                try {
                    const dir = window.paths.get('inference.raw_activations', { prompt_set: promptSet });
                    const file = window.paths.get('patterns.prompt_json', { index: i });
                    const url = `${dir}/${file}`;
                    const response = await fetch(url, { method: 'HEAD' });
                    if (response.ok) {
                        indices.push(i);
                    } else {
                        break;
                    }
                } catch (e) {
                    break; // Stop when we hit the first missing prompt
                }
            }
            return indices;
        } catch (e) {
            console.warn(`Failed to discover prompts for set ${promptSet}:`, e);
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
     * @param {number} promptNum - Prompt number
     * @param {number} layer - Layer number
     * @returns {Promise<Object>} - SAE feature activations
     */
    static async fetchSAEFeatures(trait, promptNum, layer = 16) {
        const dir = window.paths.get('inference.trait', { trait: trait.name });
        const url = `${dir}/projections/sae_features/prompt_${promptNum}_layer${layer}_sae.pt`;
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to fetch SAE features for ${trait.name} prompt ${promptNum} layer ${layer}`);
        }
        // Note: PT files need special handling - this would return blob
        return await response.blob();
    }
}

// Export to global scope
window.DataLoader = DataLoader;
