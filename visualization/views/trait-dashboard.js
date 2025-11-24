// Trait Dashboard View - Combines Vector Analysis (Extraction Stats) and Evaluation Results (Trait Quality)

// --- Data Loading ---
let evaluationData = null; // Cached evaluation data

async function loadEvaluationData() {
    if (evaluationData) return evaluationData;
    try {
        const url = window.paths.extractionEvaluation();
        const response = await fetch(url);
        if (!response.ok) {
            console.warn('Extraction evaluation not found. Quality metrics will be unavailable.');
            evaluationData = { all_results: [] }; // Set empty data to prevent re-fetching
            return evaluationData;
        }
        evaluationData = await response.json();
        return evaluationData;
    } catch (error) {
        console.error('Error loading evaluation data:', error);
        evaluationData = { all_results: [] }; // Set empty data on error
        return evaluationData;
    }
}


async function renderTraitDashboard() {
    const contentArea = document.getElementById('content-area');
    const filteredTraits = window.getFilteredTraits();

    if (filteredTraits.length === 0) {
        contentArea.innerHTML = `<div class="card"><div class="card-title">Trait Dashboard</div><div class="info">Select at least one trait to view its dashboard.</div></div>`;
        return;
    }

    contentArea.innerHTML = '<div class="loading">Loading Trait Dashboards...</div>';

    // Set experiment name for path builder
    window.paths.setExperiment(window.state.experimentData.name);

    // Load all necessary data in parallel
    const allEvaluationData = await loadEvaluationData();

    // Loop through selected traits and render a dashboard for each
    let dashboardsHtml = '';
    for (const trait of filteredTraits) {
        // Each trait gets a placeholder div
        dashboardsHtml += `<div id="dashboard-${trait.name}" class="trait-dashboard-card"></div>`;
    }
    contentArea.innerHTML = dashboardsHtml;

    // Render each dashboard asynchronously
    for (const trait of filteredTraits) {
        renderSingleTraitCard(trait, allEvaluationData);
    }
}


async function renderSingleTraitCard(trait, allEvaluationData) {
    const container = document.getElementById(`dashboard-${trait.name}`);
    if (!container) return;

    container.innerHTML = `<div class="loading" style="height: 200px;">Loading ${trait.name}...</div>`;

    try {
        // --- 1. Load Extraction Stats Data (from vector-analysis.js logic) ---
        const nLayers = trait.metadata?.n_layers || 26;
        const methods = ['mean_diff', 'probe', 'ica', 'gradient'];
        const layers = Array.from({ length: nLayers }, (_, i) => i);

        const fetchPromises = methods.flatMap(method =>
            layers.map(layer => {
                const url = window.paths.vectorMetadata(trait, method, layer);
                return fetch(url)
                    .then(r => r.ok ? r.json() : null)
                    .catch(() => null);
            })
        );
        const statsResults = (await Promise.all(fetchPromises)).filter(r => r !== null);

        // --- 2. Filter Trait Quality Data (from evaluation results) ---
        const qualityResults = allEvaluationData.all_results.filter(r => r.trait === trait.name);

        // --- 3. Render the combined dashboard ---
        const displayName = window.getDisplayName(trait.name);
        container.innerHTML = `
            <div class="card-title">${displayName}</div>
            <div class="dashboard-grid">
                <div class="dashboard-cell">
                    <h4 class="dashboard-cell-title">Trait Extraction Stats</h4>
                    <div id="stats-heatmap-${trait.name}"></div>
                </div>
                <div class="dashboard-cell">
                    <h4 class="dashboard-cell-title">Trait Quality</h4>
                    <div id="quality-table-${trait.name}"></div>
                </div>
            </div>
        `;

        // Render stats heatmap if data exists
        if (statsResults.length > 0) {
            renderStatsHeatmap(trait, statsResults, `stats-heatmap-${trait.name}`, nLayers);
        } else {
            document.getElementById(`stats-heatmap-${trait.name}`).innerHTML = `<div class="info">No extraction metadata found.</div>`;
        }

        // Render quality table if data exists
        if (qualityResults.length > 0) {
            renderQualityTable(trait, qualityResults, `quality-table-${trait.name}`);
        } else {
            document.getElementById(`quality-table-${trait.name}`).innerHTML = `<div class="info">No evaluation results found.</div>`;
        }

    } catch (error) {
        console.error(`Failed to render dashboard for ${trait.name}:`, error);
        container.innerHTML = `<div class="card-title">${window.getDisplayName(trait.name)}</div><div class="error">Failed to load data.</div>`;
    }
}


function renderStatsHeatmap(trait, statsResults, containerId, nLayers) {
    const methods = ['mean_diff', 'probe', 'ica', 'gradient'];
    const layers = Array.from({ length: nLayers }, (_, i) => i);

    const vectorData = {};
    methods.forEach(m => vectorData[m] = {});
    statsResults.forEach(r => {
        if (r && r.method && r.layer !== undefined) {
            vectorData[r.method][r.layer] = r;
        }
    });

    const normalizedData = layers.map(layer => {
        return methods.map(method => {
            const metadata = vectorData[method] ? vectorData[method][layer] : null;
            if (!metadata) return null;

            if (method === 'probe') return metadata.vector_norm ? (1.0 / metadata.vector_norm) : null;
            if (method === 'gradient') return metadata.final_separation || null;
            return metadata.vector_norm;
        });
    });

    const maxPerMethod = methods.map((_, methodIdx) => {
        const values = normalizedData.map(row => row[methodIdx]).filter(v => v !== null && !isNaN(v));
        return values.length > 0 ? Math.max(...values) : 1;
    });

    const heatmapData = normalizedData.map(row =>
        row.map((value, methodIdx) =>
            value === null ? null : (value / maxPerMethod[methodIdx]) * 100
        )
    );

    const trace = {
        z: heatmapData,
        x: methods.map(m => m.replace(/_/g, ' ')),
        y: layers,
        type: 'heatmap',
        colorscale: window.ASYMB_COLORSCALE,
        hovertemplate: 'Method: %{x}<br>Layer: %{y}<br>Strength: %{z:.1f}%<extra></extra>',
        zmin: 0,
        zmax: 100,
        showscale: false
    };

    Plotly.newPlot(containerId, [trace], window.getPlotlyLayout({
        margin: { l: 30, r: 10, t: 10, b: 80 },
        xaxis: { tickangle: -45 },
        yaxis: { title: 'Layer' },
        height: 400
    }), { displayModeBar: false });
}


function renderQualityTable(trait, qualityResults, containerId) {
    // --- Pre-compute the max effect size for the trait for normalization ---
    const maxEffect = Math.max(...qualityResults.map(r => r.val_effect_size || 0));
    const max_d = maxEffect > 0 ? maxEffect : 1;

    const calculateQualityScore = (result) => {
        if (result.val_accuracy === null || result.val_effect_size === null) return 0;
        const normalizedEffectSize = (result.val_effect_size || 0) / max_d;
        return (0.5 * result.val_accuracy) + (0.5 * normalizedEffectSize);
    };

    const sortedResults = [...qualityResults].sort((a, b) => calculateQualityScore(b) - calculateQualityScore(a));
    const top5 = sortedResults.slice(0, 5);

    if (top5.length === 0) {
        document.getElementById(containerId).innerHTML = `<div class="info">No valid quality results for this trait.</div>`;
        return;
    }

    const tableRows = top5.map(best => {
        if (!best) return '';
        const qualityScore = calculateQualityScore(best);
        const accColor = best.val_accuracy >= 0.9 ? 'var(--success)' : best.val_accuracy >= 0.75 ? 'var(--warning)' : 'var(--danger)';
        return `
            <tr>
                <td style="color: var(--text-secondary);">${best.method}</td>
                <td style="color: var(--text-primary);">${best.layer}</td>
                <td style="color: ${accColor}; font-weight: bold;">${(best.val_accuracy * 100).toFixed(1)}%</td>
                <td style="color: var(--text-primary);">${best.val_effect_size?.toFixed(2) || 'N/A'}</td>
                <td style="color: var(--text-primary);">${best.polarity_correct ? '✓' : '✗'}</td>
                <td style="font-weight: 600; color: var(--text-primary);">${qualityScore.toFixed(2)}</td>
            </tr>
        `;
    }).join('');

    const tableHtml = `
        <p class="dashboard-cell-subtitle">Top 5 vectors ranked by a combined score of accuracy and normalized effect size.</p>
        <table class="quality-table">
            <thead>
                <tr>
                    <th>Method</th>
                    <th>Layer</th>
                    <th>Acc.</th>
                    <th>Effect</th>
                    <th>Pol.</th>
                    <th>Score</th>
                </tr>
            </thead>
            <tbody>
                ${tableRows}
            </tbody>
        </table>
    `;

    document.getElementById(containerId).innerHTML = tableHtml;
}

// Export to global scope
window.renderTraitDashboard = renderTraitDashboard;