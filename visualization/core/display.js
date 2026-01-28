/**
 * Display utilities for visualization.
 * Handles display names, colors, CSS variables, and Plotly layouts.
 */

// Display names for better interpretability
const DISPLAY_NAMES = {
    // Legacy names (from old structure)
    'uncertainty_calibration': 'Confidence',
    'instruction_boundary': 'Literalness',
    'commitment_strength': 'Assertiveness',
    'retrieval_construction': 'Retrieval',
    'convergent_divergent': 'Thinking Style',
    'abstract_concrete': 'Abstraction Level',
    'temporal_focus': 'Temporal Orientation',
    'cognitive_load': 'Complexity',
    'context_adherence': 'Context Following',
    'emotional_valence': 'Emotional Tone',
    'paranoia_trust': 'Trust Level',
    'power_dynamics': 'Authority Tone',
    'serial_parallel': 'Processing Style',
    'local_global': 'Focus Scope',

    // New categorized trait names
    'abstractness': 'Abstractness',
    'authority': 'Authority',
    'compliance': 'Compliance',
    'confidence': 'Confidence',
    'context': 'Context Adherence',
    'curiosity': 'Curiosity',
    'defensiveness': 'Defensiveness',
    'divergence': 'Divergent Thinking',
    'enthusiasm': 'Enthusiasm',
    'evaluation_awareness': 'Evaluation Awareness',
    'formality': 'Formality',
    'futurism': 'Future Focus',
    'literalness': 'Literalness',
    'positivity': 'Positivity',
    'refusal': 'Refusal',
    'retrieval': 'Retrieval',
    'scope': 'Scope',
    'sequentiality': 'Sequential Processing',
    'sycophancy': 'Sycophancy',
    'trust': 'Trust'
};

/**
 * Get display name for a trait
 * Handles category/trait format (e.g., "cognitive/context")
 * Also handles category/trait/position format (e.g., "chirp/refusal_v2/response_all")
 */
function getDisplayName(traitName) {
    let baseName = traitName;
    let method = '';
    let category = '';
    let position = '';

    // Extract parts
    if (traitName.includes('/')) {
        const parts = traitName.split('/');
        category = parts[0];
        baseName = parts[1];
        // Check for position suffix (3+ parts)
        if (parts.length >= 3) {
            position = parts.slice(2).join('/');
        }
    }

    // Extract method suffix
    if (baseName.endsWith('_natural')) {
        baseName = baseName.replace('_natural', '');
        method = ' (Natural)';
    }

    // Get display name (without category prefix in the lookup)
    let displayBase = DISPLAY_NAMES[baseName] ||
        baseName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());

    // Format position suffix for display
    let positionSuffix = '';
    if (position && window.paths?.formatPositionDisplay) {
        positionSuffix = ' ' + window.paths.formatPositionDisplay(position);
    }

    return displayBase + method + positionSuffix;
}

// Standard colorscale for trait heatmaps (emerald to rose: green=positive, red=negative)
// Forest→Coral: low values (red/coral), high values (green/forest)
const ASYMB_COLORSCALE = [
    [0, '#d47c67'],
    [0.25, '#e8b0a0'],
    [0.5, '#e8e8c8'],
    [0.75, '#91cf60'],
    [1, '#1a9850']
];

// Delta colorscale for steering heatmaps (diverging: red=negative, green=positive)
const DELTA_COLORSCALE = [
    [0, '#aa5656'],
    [0.5, '#e0e0de'],
    [1, '#3d7435']
];

// Correlation colorscale (diverging: blue=negative, neutral=white, red=positive)
const CORRELATION_COLORSCALE = [
    [0, '#2d5a87'],      // Strong negative (blue)
    [0.25, '#6b9bc3'],   // Weak negative
    [0.5, '#f5f5f5'],    // Zero (neutral)
    [0.75, '#e8a87c'],   // Weak positive
    [1, '#c44e52']       // Strong positive (red)
];

/**
 * Get CSS variable value
 */
function getCssVar(name, fallback = '') {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
}

/**
 * Convert hex color to rgba with opacity
 */
function hexToRgba(hex, opacity) {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    if (!result) return `rgba(0, 0, 0, ${opacity})`;
    const r = parseInt(result[1], 16);
    const g = parseInt(result[2], 16);
    const b = parseInt(result[3], 16);
    return `rgba(${r}, ${g}, ${b}, ${opacity})`;
}

/**
 * Get token highlight colors for Plotly shapes (single source of truth)
 */
function getTokenHighlightColors() {
    const primaryColor = getCssVar('--primary-color', '#a09f6c');
    return {
        separator: `${primaryColor}80`,  // 50% opacity - prompt/response divider
        highlight: `${primaryColor}80`   // 50% opacity - current token highlight
    };
}

/**
 * Chart color palette (reads from CSS vars, with fallbacks)
 */
function getChartColors() {
    return [
        getCssVar('--chart-1', '#4a9eff'),
        getCssVar('--chart-2', '#ff6b6b'),
        getCssVar('--chart-3', '#51cf66'),
        getCssVar('--chart-4', '#ffd43b'),
        getCssVar('--chart-5', '#cc5de8'),
        getCssVar('--chart-6', '#ff922b'),
        getCssVar('--chart-7', '#20c997'),
        getCssVar('--chart-8', '#f06595'),
        getCssVar('--chart-9', '#748ffc'),
        getCssVar('--chart-10', '#a9e34b'),
    ];
}

/**
 * Method colors for extraction methods
 */
function getMethodColors() {
    return {
        probe: getCssVar('--method-probe', '#4a9eff'),
        gradient: getCssVar('--method-gradient', '#51cf66'),
        mean_diff: getCssVar('--method-mean-diff', '#cc5de8'),
    };
}

/**
 * Plotly Layout Helper - reads colors from CSS variables
 */
function getPlotlyLayout(baseLayout = {}) {
    const styles = getComputedStyle(document.documentElement);
    const textPrimary = styles.getPropertyValue('--text-primary').trim() || '#e0e0e0';
    const bgTertiary = styles.getPropertyValue('--bg-tertiary').trim() || '#3a3a3a';

    return {
        ...baseLayout,
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: {
            ...baseLayout.font,
            color: textPrimary
        },
        xaxis: {
            ...baseLayout.xaxis,
            color: textPrimary,
            gridcolor: bgTertiary,
            zerolinecolor: bgTertiary
        },
        yaxis: {
            ...baseLayout.yaxis,
            color: textPrimary,
            gridcolor: bgTertiary,
            zerolinecolor: bgTertiary
        }
    };
}

// Export to global scope
window.DISPLAY_NAMES = DISPLAY_NAMES;
window.getDisplayName = getDisplayName;
window.ASYMB_COLORSCALE = ASYMB_COLORSCALE;
window.DELTA_COLORSCALE = DELTA_COLORSCALE;
window.CORRELATION_COLORSCALE = CORRELATION_COLORSCALE;
window.getCssVar = getCssVar;
window.hexToRgba = hexToRgba;
window.getTokenHighlightColors = getTokenHighlightColors;
window.getChartColors = getChartColors;
window.getMethodColors = getMethodColors;
window.getPlotlyLayout = getPlotlyLayout;
