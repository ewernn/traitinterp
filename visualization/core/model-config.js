/**
 * Model configuration loader.
 * Loads model architecture and settings from config/models/*.yaml
 *
 * Usage:
 *     await modelConfig.load('gemma-2-2b-it');
 *
 *     modelConfig.get('num_hidden_layers');  // 26
 *     modelConfig.get('sae.available');      // true
 *     modelConfig.getNumLayers();            // 26
 *     modelConfig.getSaePath(16);            // 'sae/gemma-scope.../layer_16_...'
 */

class ModelConfig {
    constructor() {
        this.config = null;
        this.modelId = null;
        this.loaded = false;
        this._cache = {};
    }

    /**
     * Load model config from YAML file.
     * @param {string} modelId - Model identifier (e.g., 'gemma-2-2b-it')
     */
    async load(modelId) {
        // Normalize: google/gemma-2-2b-it -> gemma-2-2b-it
        if (modelId.includes('/')) {
            modelId = modelId.split('/').pop().toLowerCase();
        }

        // Return cached if same model
        if (this.modelId === modelId && this.loaded) {
            return this.config;
        }

        // Check cache
        if (this._cache[modelId]) {
            this.config = this._cache[modelId];
            this.modelId = modelId;
            this.loaded = true;
            return this.config;
        }

        const response = await fetch(`/config/models/${modelId}.yaml`);
        if (!response.ok) {
            throw new Error(`Failed to load model config for '${modelId}': ${response.status}`);
        }

        const yamlText = await response.text();
        this.config = jsyaml.load(yamlText);
        this.modelId = modelId;
        this.loaded = true;
        this._cache[modelId] = this.config;

        return this.config;
    }

    /**
     * Load model config for an experiment.
     * Reads experiment's config.json to determine model, then loads model config.
     * @param {string} experiment - Experiment name
     */
    async loadForExperiment(experiment) {
        let modelId = experiment;

        // Try to read experiment config
        try {
            const response = await fetch(`/experiments/${experiment}/config.json`);
            if (response.ok) {
                const expConfig = await response.json();
                if (expConfig.model) {
                    modelId = expConfig.model;
                }
            }
        } catch (e) {
            // Fall back to experiment name as model ID
            console.log(`No experiment config for ${experiment}, using experiment name as model ID`);
        }

        return this.load(modelId);
    }

    /**
     * Get a config value by dot-separated key.
     * @param {string} key - Dot-separated key like 'sae.available'
     * @returns {*} Config value
     */
    get(key) {
        if (!this.loaded) {
            throw new Error('ModelConfig not loaded. Call await modelConfig.load() first.');
        }

        let value = this.config;
        for (const k of key.split('.')) {
            if (value === undefined || value === null) return undefined;
            value = value[k];
        }
        return value;
    }

    /**
     * Get the full config object.
     * @returns {Object}
     */
    getAll() {
        if (!this.loaded) {
            throw new Error('ModelConfig not loaded. Call await modelConfig.load() first.');
        }
        return this.config;
    }

    // Convenience accessors

    getNumLayers() {
        return this.get('num_hidden_layers');
    }

    getHiddenSize() {
        return this.get('hidden_size');
    }

    getNumHeads() {
        return this.get('num_attention_heads');
    }

    getNumKvHeads() {
        return this.get('num_key_value_heads');
    }

    getResidualSublayers() {
        return this.get('residual_sublayers') || ['input', 'after_attn', 'output'];
    }

    getNumSublayers() {
        return this.getResidualSublayers().length;
    }

    getDefaultMonitoringLayer() {
        return this.get('defaults.monitoring_layer') || Math.floor(this.getNumLayers() / 2);
    }

    isSaeAvailable() {
        return this.get('sae.available') || false;
    }

    getSaeDownloadedLayers() {
        return this.get('sae.downloaded_layers') || [];
    }

    /**
     * Get SAE path for a specific layer.
     * @param {number} layer - Layer index
     * @returns {string|null} Path to SAE directory, or null if not available
     */
    getSaePath(layer) {
        if (!this.isSaeAvailable()) return null;

        const downloadedLayers = this.getSaeDownloadedLayers();
        if (!downloadedLayers.includes(layer)) return null;

        const basePath = this.get('sae.base_path');
        const template = this.get('sae.layer_template');
        const layerDir = template.replace('{layer}', layer);

        return `${basePath}/${layerDir}`;
    }
}

// Global instance
const modelConfig = new ModelConfig();
window.modelConfig = modelConfig;
