// Centralized Path Configuration for Trait Interpretation Visualization
// Handles all path construction to maintain consistency across the codebase

/**
 * Path builder for experiment data structure
 * Structure: experiments/{experiment}/extraction/{category}/{trait}/{subdir}
 */
class PathBuilder {
    constructor(experimentName) {
        this.experimentName = experimentName;
        this.baseExperimentPath = `../experiments/${experimentName}`;
    }

    /**
     * Get extraction-related paths
     * @param {Object} trait - Trait object with name property (format: category/trait_name)
     * @param {string} subpath - Subpath within trait directory (e.g., 'vectors', 'responses', 'activations')
     * @returns {string} - Full path
     */
    extraction(trait, subpath = '') {
        const traitName = typeof trait === 'string' ? trait : trait.name;
        const path = `${this.baseExperimentPath}/extraction/${traitName}`;
        return subpath ? `${path}/${subpath}` : path;
    }

    /**
     * Get vector metadata path
     * @param {Object} trait - Trait object
     * @param {string} method - Extraction method (mean_diff, probe, ica, gradient)
     * @param {number} layer - Layer number
     * @returns {string} - Path to vector metadata JSON
     */
    vectorMetadata(trait, method, layer) {
        return this.extraction(trait, `vectors/${method}_layer${layer}_metadata.json`);
    }

    /**
     * Get vector tensor path
     * @param {Object} trait - Trait object
     * @param {string} method - Extraction method
     * @param {number} layer - Layer number
     * @returns {string} - Path to vector tensor PT file
     */
    vectorTensor(trait, method, layer) {
        return this.extraction(trait, `vectors/${method}_layer${layer}.pt`);
    }

    /**
     * Get responses path
     * @param {Object} trait - Trait object
     * @param {string} polarity - 'pos' or 'neg'
     * @param {string} format - 'csv' or 'json'
     * @returns {string} - Path to responses file
     */
    responses(trait, polarity, format = 'csv') {
        return this.extraction(trait, `responses/${polarity}.${format}`);
    }

    /**
     * Get activations metadata path
     * @param {Object} trait - Trait object
     * @returns {string} - Path to activations metadata JSON
     */
    activationsMetadata(trait) {
        return this.extraction(trait, 'activations/metadata.json');
    }

    /**
     * Get trait definition path
     * @param {Object} trait - Trait object
     * @returns {string} - Path to trait definition JSON
     */
    traitDefinition(trait) {
        return this.extraction(trait, 'trait_definition.json');
    }

    /**
     * Get inference-related paths
     * @param {Object} trait - Trait object
     * @param {string} subpath - Subpath (e.g., 'projections/residual_stream_activations')
     * @returns {string} - Full path
     */
    inference(trait, subpath = '') {
        const traitName = typeof trait === 'string' ? trait : trait.name;
        const path = `${this.baseExperimentPath}/inference/${traitName}`;
        return subpath ? `${path}/${subpath}` : path;
    }

    /**
     * Get Tier 2 inference data path (residual stream activations)
     * @param {Object} trait - Trait object
     * @param {number} promptNum - Prompt number
     * @returns {string} - Path to tier 2 JSON
     */
    tier2Data(trait, promptNum) {
        return this.inference(trait, `projections/residual_stream_activations/prompt_${promptNum}.json`);
    }

    /**
     * Get Tier 3 inference data path (layer internal states)
     * @param {Object} trait - Trait object
     * @param {number} promptNum - Prompt number
     * @param {number} layer - Layer number
     * @returns {string} - Path to tier 3 JSON
     */
    tier3Data(trait, promptNum, layer = 16) {
        return this.inference(trait, `projections/layer_internal_states/prompt_${promptNum}_layer${layer}.json`);
    }

    /**
     * Get shared prompts path
     * @param {number} promptIdx - Prompt index
     * @returns {string} - Path to shared prompt JSON
     */
    sharedPrompt(promptIdx) {
        return `${this.baseExperimentPath}/inference/prompts/prompt_${promptIdx}.json`;
    }

    /**
     * Get trait-specific projections path
     * @param {Object} trait - Trait object
     * @param {number} promptIdx - Prompt index
     * @returns {string} - Path to projections JSON
     */
    projections(trait, promptIdx) {
        return this.inference(trait, `projections/prompt_${promptIdx}.json`);
    }

    /**
     * Get validation-related paths
     * @param {string} subpath - Subpath within validation directory
     * @returns {string} - Full path
     */
    validation(subpath = '') {
        return `${this.baseExperimentPath}/validation/${subpath}`;
    }

    /**
     * Get data index path (for cross-distribution data)
     * @returns {string} - Path to data index JSON
     */
    dataIndex() {
        return this.validation('data_index.json');
    }

    /**
     * Get cross-distribution results path
     * @param {string} traitBaseName - Base name of trait (without _natural suffix)
     * @returns {string} - Path to cross-distribution results JSON
     */
    crossDistribution(traitBaseName) {
        return this.validation(`${traitBaseName}_full_4x4_results.json`);
    }

    /**
     * Get experiment README path
     * @returns {string} - Path to README.md
     */
    readme() {
        return `${this.baseExperimentPath}/README.md`;
    }
}

/**
 * Helper function to check if a trait has vectors
 * @param {string} experimentName - Experiment name
 * @param {Object} trait - Trait object
 * @returns {Promise<boolean>} - True if vectors exist
 */
async function hasVectors(experimentName, trait) {
    const pathBuilder = new PathBuilder(experimentName);
    try {
        const testUrl = pathBuilder.vectorMetadata(trait, 'probe', 16);
        const response = await fetch(testUrl, { method: 'HEAD' });
        return response.ok;
    } catch (e) {
        return false;
    }
}

/**
 * Helper function to get the correct response format for a trait
 * @param {string} experimentName - Experiment name
 * @param {Object} trait - Trait object
 * @returns {Promise<string|null>} - 'csv', 'json', or null if no responses
 */
async function detectResponseFormat(experimentName, trait) {
    const pathBuilder = new PathBuilder(experimentName);
    const isNatural = trait.name.includes('_natural');

    // Try primary format first
    const primaryExt = isNatural ? 'json' : 'csv';
    const primaryUrl = pathBuilder.responses(trait, 'pos', primaryExt);
    const primaryCheck = await fetch(primaryUrl, { method: 'HEAD' });
    if (primaryCheck.ok) return primaryExt;

    // Try secondary format
    const secondaryExt = isNatural ? 'csv' : 'json';
    const secondaryUrl = pathBuilder.responses(trait, 'pos', secondaryExt);
    const secondaryCheck = await fetch(secondaryUrl, { method: 'HEAD' });
    if (secondaryCheck.ok) return secondaryExt;

    return null;
}

// Export to global scope
window.PathBuilder = PathBuilder;
window.hasVectors = hasVectors;
window.detectResponseFormat = detectResponseFormat;
