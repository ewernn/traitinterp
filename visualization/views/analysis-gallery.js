// Analysis Gallery View - Auto-discovers and displays analysis outputs
// Scans experiments/{exp}/analysis/ for PNG/JSON files and displays them in a grid
//
// ============================================================================
// ANALYSIS CATEGORIES (pre-computed by run_analyses.py)
// ============================================================================
//
// 1. NORMALIZED VELOCITY (Heatmap)
//    ------------------------------
//    What: How fast hidden states change between layers, corrected for magnitude
//    Math: velocity = hidden[L+1] - hidden[L]                    → [25, n_tokens, 2304]
//          normalized = ||velocity|| / ||hidden[L]||             → [25, n_tokens]
//    Read: X-axis = token position, Y-axis = layer transition (0→1 to 24→25)
//          Bright (yellow) = high change, Dark (purple) = stable
//          Typical: high L0-6, low L7-22, high L23-24
//    Why:  Raw velocity "explodes" at late layers due to magnitude scaling.
//          Normalizing fixes this artifact.
//
// 2. RADIAL/ANGULAR DECOMPOSITION (Side-by-side heatmaps)
//    ------------------------------------------------------
//    What: Splits velocity into "growing/shrinking" vs "rotating"
//    Math: RADIAL:  magnitude[L+1] - magnitude[L]                → change in size
//          ANGULAR: 1 - cos_similarity(direction[L], direction[L+1]) → change in direction
//    Read: LEFT (Radial): Red = growing, Blue = shrinking, White = stable size
//          RIGHT (Angular): Bright = direction changing, Dark = same direction
//    Why:  Separates "turning up the volume" from "changing the content"
//
// 3. TRAIT PROJECTIONS (2x5 grid of heatmaps)
//    -----------------------------------------
//    What: How each token activates each of 10 traits across all 26 layers
//    Math: projection = hidden[L, token] · (trait_vec / ||trait_vec||)
//          One heatmap per trait, showing [26 layers × n_tokens]
//    Read: Red = positive activation, Blue = negative, White = neutral
//          Watch traits emerge: usually zero in early layers, differentiate later
//    Why:  See which layers encode which behavioral properties
//
// 4. TRAIT EMERGENCE (Horizontal bar chart)
//    ---------------------------------------
//    What: Which layer each trait first becomes significant
//    Math: emergence_layer = first L where |projection| > 0.5 × max|projection|
//          Averaged across all tokens and prompts
//    Read: Shorter bars = trait emerges earlier, Longer = emerges later
//          Green line (L8) and red line (L19) mark "stable computation" region
//    Finding: NO traits emerge before L7. All emerge L14+.
//
// 5. TRAIT-DYNAMICS CORRELATION (Horizontal bar chart)
//    ---------------------------------------------------
//    What: Do trait changes happen when the model is most "active"?
//    Math: trait_velocity = diff(trait_projection)               → [25]
//          correlation = pearson(normalized_velocity, |trait_velocity|)
//    Read: Green bars = high correlation (trait tied to computation bursts)
//          Small/red bars = low correlation (trait changes independently)
//    Finding: defensiveness, correction_impulse correlate highly (~0.6)
//             uncertainty, retrieval correlate weakly (~0.1)
//
// 6. SUMMARY PLOTS (Line plots with error bands)
//    --------------------------------------------
//    What: Aggregated view across all 8 prompts
//    Math: mean ± std across prompts for each layer
//    Read: Solid line = mean, Shaded band = ±1 std deviation
//          Shows consistency: narrow band = consistent, wide = variable
//
// ============================================================================

// Cache for analyses data
let analysisGalleryCache = { experiment: null, analyses: null };

async function renderAnalysisGallery() {
    const contentArea = document.getElementById('content-area');

    const experiment = window.state.experimentData?.name;
    if (!experiment) {
        contentArea.innerHTML = '<div class="error">No experiment selected</div>';
        return;
    }

    // Check if we already have the gallery DOM and cached data
    const existingGallery = contentArea.querySelector('.analysis-gallery');
    const dataIsCached = analysisGalleryCache.experiment === experiment && analysisGalleryCache.analyses;

    // Only show loading if we need to fetch new data
    if (!dataIsCached) {
        contentArea.innerHTML = '<div class="loading">Scanning analysis folder...</div>';
    }

    try {
        // Use cached analyses or fetch new
        let analyses;
        if (dataIsCached) {
            analyses = analysisGalleryCache.analyses;
        } else {
            analyses = await discoverAnalyses(experiment);
            analysisGalleryCache = { experiment, analyses };
        }

        if (analyses.length === 0) {
            contentArea.innerHTML = `
                <div class="info" style="margin: 16px; padding: 16px;">
                    <h3>No analyses found</h3>
                    <p>Run analysis scripts to generate outputs in:</p>
                    <code>experiments/${experiment}/analysis/</code>
                </div>
            `;
            return;
        }

        // Filter based on current prompt selection from state
        const currentPromptId = window.state.currentPromptId;
        const promptFilter = currentPromptId ? `prompt_${currentPromptId}` : null;

        // Filter analyses: show matching prompt OR summaries
        const filteredAnalyses = analyses.filter(item => {
            const isPromptSpecific = item.name.match(/^prompt_\d+$/);
            if (promptFilter) {
                // If prompt selected, show that prompt's analyses + summaries
                return item.name === promptFilter || !isPromptSpecific;
            } else {
                // If no prompt selected, show only summaries
                return !isPromptSpecific;
            }
        });

        // Group by category
        const grouped = groupByCategory(filteredAnalyses);
        const displayLabel = promptFilter ? `Prompt ${currentPromptId}` : 'Summary views';

        // If DOM exists and data was cached, just update the content (no scroll reset)
        if (existingGallery && dataIsCached) {
            // Update header text
            const galleryInfo = existingGallery.querySelector('.gallery-info');
            const galleryCount = existingGallery.querySelector('.gallery-count');
            if (galleryInfo) galleryInfo.innerHTML = `Showing: <strong>${displayLabel}</strong>`;
            if (galleryCount) galleryCount.textContent = `${filteredAnalyses.length} analyses`;

            // Update content
            const galleryContent = document.getElementById('gallery-content');
            galleryContent.innerHTML = '';

            for (const [category, items] of Object.entries(grouped)) {
                const categorySection = document.createElement('div');
                categorySection.className = 'gallery-category';
                categorySection.innerHTML = `
                    <h3 class="category-title">${formatCategoryName(category)}</h3>
                    <div class="category-grid"></div>
                `;
                galleryContent.appendChild(categorySection);

                const grid = categorySection.querySelector('.category-grid');
                for (const item of items) {
                    renderAnalysisCard(item, grid);
                }
            }
            return;
        }

        // Full render (first load or experiment changed)
        contentArea.innerHTML = `
            <div class="analysis-gallery">
                <div class="gallery-header">
                    <span class="gallery-info">Showing: <strong>${displayLabel}</strong></span>
                    <span class="gallery-count">${filteredAnalyses.length} analyses</span>
                </div>
                <div class="gallery-content" id="gallery-content"></div>
            </div>
        `;

        const galleryContent = document.getElementById('gallery-content');

        for (const [category, items] of Object.entries(grouped)) {
            const categorySection = document.createElement('div');
            categorySection.className = 'gallery-category';
            categorySection.innerHTML = `
                <h3 class="category-title">${formatCategoryName(category)}</h3>
                <div class="category-grid"></div>
            `;
            galleryContent.appendChild(categorySection);

            const grid = categorySection.querySelector('.category-grid');
            for (const item of items) {
                renderAnalysisCard(item, grid);
            }
        }

    } catch (error) {
        console.error('Failed to load analysis gallery:', error);
        contentArea.innerHTML = `<div class="error">Failed to load analyses: ${error.message}</div>`;
    }
}


async function discoverAnalyses(experiment) {
    // Try to fetch an index file first (fast path)
    try {
        const indexUrl = window.paths.analysisIndex();
        const response = await fetch(indexUrl);
        if (response.ok) {
            return await response.json();
        }
    } catch (e) {
        // Index doesn't exist, fall back to discovery
    }

    // Manual discovery: fetch directory listing
    // This requires the server to support directory listing or we need to know the structure
    // For now, try common category names
    const categories = [
        'normalized_velocity',
        'radial_angular',
        'trait_projections',
        'trait_emergence',
        'trait_dynamics_correlation',
        'summary',
        'attention_dynamics'  // From previous work
    ];

    const analyses = [];

    for (const category of categories) {
        // Try to find PNG files in each category
        for (let promptId = 1; promptId <= 8; promptId++) {
            const pngPath = window.paths.analysisCategoryPrompt(category, promptId, 'png');
            const jsonPath = window.paths.analysisCategoryPrompt(category, promptId, 'json');
            try {
                const response = await fetch(pngPath, { method: 'HEAD' });
                if (response.ok) {
                    analyses.push({
                        category,
                        name: `prompt_${promptId}`,
                        pngPath,
                        jsonPath
                    });
                }
            } catch (e) {
                // File doesn't exist, skip
            }
        }

        // Also check for summary/aggregate files
        const summaryFiles = ['summary', 'comparison', 'all_prompts', 'overview'];
        for (const filename of summaryFiles) {
            const pngPath = window.paths.analysisCategoryNamed(category, filename, 'png');
            const jsonPath = window.paths.analysisCategoryNamed(category, filename, 'json');
            try {
                const response = await fetch(pngPath, { method: 'HEAD' });
                if (response.ok) {
                    analyses.push({
                        category,
                        name: filename,
                        pngPath,
                        jsonPath
                    });
                }
            } catch (e) {
                // Skip
            }
        }
    }

    return analyses;
}


function groupByCategory(analyses) {
    const grouped = {};
    for (const item of analyses) {
        if (!grouped[item.category]) {
            grouped[item.category] = [];
        }
        grouped[item.category].push(item);
    }
    return grouped;
}


function formatCategoryName(category) {
    return category
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}


function renderAnalysisCard(item, container) {
    const card = document.createElement('div');
    card.className = 'analysis-card';
    card.innerHTML = `
        <div class="analysis-thumbnail">
            <img src="${item.pngPath}" alt="${item.name}" loading="lazy"
                 onerror="this.parentElement.innerHTML='<div class=\\'no-image\\'>No image</div>'" />
        </div>
        <div class="analysis-info">
            <div class="analysis-name">${formatAnalysisName(item.name)}</div>
        </div>
    `;

    // Click to expand
    card.addEventListener('click', () => showAnalysisModal(item));
    container.appendChild(card);
}


function formatAnalysisName(name) {
    return name
        .replace(/_/g, ' ')
        .replace(/prompt (\d+)/i, 'Prompt $1');
}


async function showAnalysisModal(item) {
    // Load metrics if available
    let metrics = null;
    try {
        const response = await fetch(item.jsonPath);
        if (response.ok) {
            metrics = await response.json();
        }
    } catch (e) {
        // No metrics available
    }

    // Use the existing preview modal
    const modal = document.getElementById('preview-modal');
    const title = document.getElementById('preview-title');
    const body = document.getElementById('preview-body');

    title.textContent = `${formatCategoryName(item.category)} - ${formatAnalysisName(item.name)}`;

    let metricsHtml = '';
    if (metrics) {
        metricsHtml = `
            <div class="analysis-metrics">
                <h4>Metrics</h4>
                <pre>${JSON.stringify(metrics, null, 2)}</pre>
            </div>
        `;
    }

    body.innerHTML = `
        <div class="analysis-modal-content">
            <img src="${item.pngPath}" alt="${item.name}" style="max-width: 100%; height: auto;" />
            ${metricsHtml}
        </div>
    `;

    modal.classList.add('active');
}


// Add CSS for the gallery
const galleryStyles = `
.analysis-gallery {
    padding: 16px;
}

.gallery-header {
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border-color);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.gallery-info {
    font-size: 14px;
    color: var(--text-primary);
}

.gallery-info strong {
    color: var(--accent-color);
}

.gallery-count {
    color: var(--text-secondary);
    font-size: 14px;
}

.gallery-category {
    margin-bottom: 24px;
}

.category-title {
    margin: 0 0 12px 0;
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
}

.category-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
    gap: 16px;
}

.analysis-card {
    background: var(--surface-color);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    overflow: hidden;
    cursor: pointer;
    transition: transform 0.2s, box-shadow 0.2s;
}

.analysis-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

.analysis-thumbnail {
    width: 100%;
    aspect-ratio: 4/3;
    background: var(--background-color);
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
}

.analysis-thumbnail img {
    width: 100%;
    height: 100%;
    object-fit: contain;
}

.analysis-thumbnail .no-image {
    color: var(--text-secondary);
    font-size: 12px;
}

.analysis-info {
    padding: 8px 12px;
}

.analysis-name {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-primary);
}

.analysis-modal-content {
    display: flex;
    flex-direction: column;
    gap: 16px;
}

.analysis-metrics {
    background: var(--background-color);
    border-radius: 4px;
    padding: 12px;
}

.analysis-metrics h4 {
    margin: 0 0 8px 0;
    font-size: 14px;
}

.analysis-metrics pre {
    margin: 0;
    font-size: 11px;
    overflow-x: auto;
    max-height: 300px;
    overflow-y: auto;
}
`;

// Inject styles
if (!document.getElementById('analysis-gallery-styles')) {
    const styleSheet = document.createElement('style');
    styleSheet.id = 'analysis-gallery-styles';
    styleSheet.textContent = galleryStyles;
    document.head.appendChild(styleSheet);
}
