/**
 * Custom Blocks - Parsing and rendering for ::: syntax in markdown
 *
 * Public entry point. Re-exports parser, renderers, and loaders.
 * Also registers `window.customBlocks` for inline onclick handlers.
 *
 * Supported blocks:
 *   :::responses path "label" [flags]:::                       - Expandable response table
 *   :::dataset path "label" [flags]:::                         - Expandable dataset list
 *   :::figure path "caption" size:::                           - Image with caption (size: small|medium|large)
 *   :::example ... :::                                         - Example box with optional caption
 *   :::chart type path "caption" [traits=...] [height=N]:::    - Dynamic Plotly chart from JSON
 *   :::extraction-data "label" [expanded] [tokens=N]\n trait: path\n :::  - Tabbed pos/neg extraction viewer
 *   :::annotation-stacked "caption" [height=N]\n label: path\n :::   - Stacked bar chart from annotation files
 *
 * Flags:
 *   expanded  - Start expanded instead of collapsed
 *   no-scores - Hide trait/coherence score columns (responses only)
 *   limit=N   - Max items to show (dataset only)
 *   height=N  - Custom max height in px (responses, dataset, chart)
 *   green|red|blue|orange|purple - Colored left border
 *   traits=a,b,c - Filter to specific traits (chart only)
 */

import { extractCustomBlocks } from './parser.js';
import {
    renderCustomBlocks,
    renderResponsesTable,
    renderDatasetList,
    renderExtractionTable,
} from './renderers.js';
import {
    toggleExpandCollapse,
    toggleDropdown,
    loadExpandedDropdowns,
    toggleExtractionData,
    initExtractionData,
    loadCharts,
} from './loaders.js';

// ES module exports — the public surface
export {
    extractCustomBlocks,
    renderCustomBlocks,
    toggleDropdown,
    toggleExtractionData,
    toggleExpandCollapse,
    loadExpandedDropdowns,
    initExtractionData,
    loadCharts,
    renderResponsesTable,
    renderDatasetList,
    renderExtractionTable,
};

// Window namespace for inline onclick handlers (e.g. window.customBlocks.toggleDropdown)
window.customBlocks = {
    extractCustomBlocks,
    renderCustomBlocks,
    toggleDropdown,
    toggleExtractionData,
    loadExpandedDropdowns,
    initExtractionData,
    loadCharts,
    renderResponsesTable,
    renderDatasetList,
    renderExtractionTable,
};
