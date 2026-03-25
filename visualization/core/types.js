/**
 * JSDoc type definitions for the visualization codebase.
 *
 * Usage in other files:
 *   // @ts-check
 *   /// <reference path="./core/types.js" />
 */

// =============================================================================
// State Types
// =============================================================================

/**
 * @typedef {Object} AppState
 * @property {AppConfig|null} appConfig
 * @property {string[]} experiments
 * @property {string|null} currentExperiment
 * @property {ExperimentData|null} experimentData
 * @property {string} currentView
 * @property {Set<string>} selectedTraits
 * @property {string|null} currentPromptSet
 * @property {number|null} currentPromptId
 * @property {Object<string, PromptDef[]>} availablePromptSets
 * @property {Object<string, number[]>} promptsWithData
 * @property {number} currentTokenIndex
 * @property {PromptPickerCache|null} promptPickerCache
 * @property {boolean} projectionCentered
 * @property {Set<string>} selectedMethods
 * @property {string} compareMode
 * @property {string} massiveDimsCleaning
 */

/**
 * @typedef {Object} AppConfig
 * @property {string} mode - "development" | "production"
 * @property {Object<string, boolean>} features
 * @property {Object<string, string>} defaults
 */

/**
 * @typedef {Object} ExperimentData
 * @property {string} name
 * @property {TraitInfo[]} traits
 * @property {ExperimentConfig|null} experimentConfig
 */

/**
 * @typedef {Object} ExperimentConfig
 * @property {string} [model]
 * @property {Object<string, string>} [defaults]
 * @property {Object<string, ModelVariant>} [model_variants]
 */

/**
 * @typedef {Object} ModelVariant
 * @property {string} model
 * @property {string} [lora]
 */

/**
 * @typedef {Object} TraitInfo
 * @property {string} name - e.g., "rm_hack/ulterior_motive"
 */

// =============================================================================
// Steering Types
// =============================================================================

/**
 * @typedef {Object} SteeringEntry
 * @property {string} trait - e.g., "rm_hack/ulterior_motive"
 * @property {string} model_variant - e.g., "instruct", "rm_lora"
 * @property {string} position - e.g., "response[:5]", "response[:32]"
 * @property {string} prompt_set - e.g., "steering", "rm_syco/train_100"
 * @property {string} full_path - Full filesystem path to results
 */

/**
 * @typedef {Object} SteeringRun
 * @property {number} layer
 * @property {number} coef - Steering coefficient (weight)
 * @property {string} method - "probe" | "gradient" | "mean_diff"
 * @property {string} component - "residual" | "attn_contribution" | etc.
 * @property {number} traitScore - Trait score (0-100)
 * @property {number} coherence - Coherence score (0-100)
 * @property {string} [timestamp] - ISO timestamp
 * @property {SteeringEntry} entry - Parent steering entry
 */

/**
 * @typedef {Object} SteeringResults
 * @property {string} trait
 * @property {SteeringBaseline} baseline
 * @property {SteeringRunResult[]} runs
 * @property {string} [steering_model]
 * @property {Object} [vector_source]
 * @property {Object} [eval]
 */

/**
 * @typedef {Object} SteeringBaseline
 * @property {number} trait_mean
 * @property {number} coherence_mean
 */

/**
 * @typedef {Object} SteeringRunResult
 * @property {SteeringRunConfig} config
 * @property {SteeringRunResultData} result
 * @property {string} [timestamp]
 */

/**
 * @typedef {Object} SteeringRunConfig
 * @property {VectorSpec[]} vectors
 */

/**
 * @typedef {Object} SteeringRunResultData
 * @property {number} trait_mean
 * @property {number} coherence_mean
 */

/**
 * @typedef {Object} VectorSpec
 * @property {number} layer
 * @property {string} method
 * @property {string} [component]
 * @property {number} weight - Steering coefficient
 */

// =============================================================================
// Response Browser Types
// =============================================================================

/**
 * @typedef {Object} ResponseBrowserState
 * @property {string} sortKey - "layer" | "coef" | "traitScore" | "coherence"
 * @property {string} sortDir - "asc" | "desc"
 * @property {Set<number>} layerFilter - Empty = show all
 * @property {number|null} expandedRow
 * @property {boolean} bestPerLayer
 * @property {string|null} infoPanel - "definition" | "judge" | null
 * @property {boolean} compactResponses
 * @property {string} promptSetFilter - "all" or specific prompt set
 * @property {string} steeringDirection - "all" | "positive" | "negative"
 * @property {string} modelVariantFilter - "all" or specific model variant
 */

/**
 * @typedef {Object} ResponseFile
 * @property {string} path
 * @property {string} component
 * @property {string} method
 * @property {number} layer
 * @property {number} coef
 * @property {string} filename
 */

// =============================================================================
// Prompt Picker Types
// =============================================================================

/**
 * @typedef {Object} PromptDef
 * @property {number} id
 * @property {string} text
 * @property {string} [note]
 */

/**
 * @typedef {Object} PromptPickerCache
 * @property {string} promptSet
 * @property {number} promptId
 * @property {string} promptText
 * @property {string} responseText
 * @property {number} promptTokens
 * @property {number} responseTokens
 * @property {string[]} allTokens
 * @property {number} nPromptTokens
 * @property {string[]} [tags]
 */

// =============================================================================
// Inference Types
// =============================================================================

/**
 * @typedef {Object} ProjectionData
 * @property {PromptData} prompt
 * @property {ResponseData} response
 * @property {Object} [metadata]
 */

/**
 * @typedef {Object} PromptData
 * @property {string} text
 * @property {string[]} tokens
 */

/**
 * @typedef {Object} ResponseData
 * @property {string} text
 * @property {string[]} tokens
 * @property {number[][]} [projections] - [layer][token] scores
 */

// =============================================================================
// Custom Blocks Types
// =============================================================================

/**
 * @typedef {Object} ExtractedBlocks
 * @property {ResponseBlock[]} responses
 * @property {DatasetBlock[]} datasets
 * @property {PromptBlock[]} prompts
 * @property {FigureBlock[]} figures
 * @property {ExampleBlock[]} examples
 */

/**
 * @typedef {Object} ResponseBlock
 * @property {string} path
 * @property {string} label
 * @property {boolean} expanded
 * @property {boolean} noScores
 */

/**
 * @typedef {Object} DatasetBlock
 * @property {string} path
 * @property {string} label
 * @property {boolean} expanded
 * @property {number|null} limit
 * @property {number|null} height
 */

/**
 * @typedef {Object} PromptBlock
 * @property {string} path
 * @property {string} label
 * @property {boolean} expanded
 */

/**
 * @typedef {Object} FigureBlock
 * @property {string} path
 * @property {string} caption
 * @property {string} size
 */

/**
 * @typedef {Object} ExampleBlock
 * @property {string} content
 * @property {string} caption
 */

// =============================================================================
// API Response Types
// =============================================================================

/**
 * @typedef {Object} TraitsResponse
 * @property {string[]} traits
 */

/**
 * @typedef {Object} SteeringEntriesResponse
 * @property {SteeringEntry[]} entries
 */

/**
 * @typedef {Object} SteeringResponsesResponse
 * @property {ResponseFile[]} files
 * @property {string|null} baseline
 */

/**
 * @typedef {Object} PromptSetsResponse
 * @property {PromptSetInfo[]} prompt_sets
 */

/**
 * @typedef {Object} PromptSetInfo
 * @property {string} name
 * @property {PromptDef[]} prompts
 * @property {number[]} available_ids
 */

// Types are JSDoc only — nothing to export at runtime
