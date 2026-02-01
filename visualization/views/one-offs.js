/**
 * One-Offs View - Visualize one-off experiment results
 *
 * Supports multiple experiment types:
 * - judge_optimization: Claude score comparison (scatter plots, response tables)
 * - prefill-dynamics: Human vs model activation dynamics (layer plots, distributions)
 *
 * Features:
 * - Auto-discovers experiments based on file structure
 * - Type-specific rendering
 * - Interactive Plotly charts
 */

let oneOffData = null;
let discoveredOneOffs = [];
let currentOneOff = null;
let currentScoreType = 'trait';

// ============================================================================
// Experiment Discovery
// ============================================================================

async function discoverOneOffExperiments() {
    const found = [];

    // Check for judge_optimization (has results/trait/analysis.json)
    try {
        const resp = await fetch('/experiments/judge_optimization/results/trait/analysis.json');
        if (resp.ok) found.push({ name: 'judge_optimization', type: 'judge' });
    } catch (e) { /* skip */ }

    // Check for prefill-dynamics (has analysis/activation_metrics.json)
    try {
        const resp = await fetch('/experiments/prefill-dynamics/analysis/activation_metrics.json');
        if (resp.ok) found.push({ name: 'prefill-dynamics', type: 'dynamics' });
    } catch (e) { /* skip */ }

    return found;
}

// ============================================================================
// Judge Optimization Data Loading & Rendering
// ============================================================================

async function loadJudgeData(experiment, scoreType = 'trait') {
    try {
        const [claudeResp, variantsResp, analysisResp] = await Promise.all([
            fetch(`/experiments/${experiment}/data/claude_scores.json`),
            fetch(`/experiments/${experiment}/results/${scoreType}/all_variants.json`),
            fetch(`/experiments/${experiment}/results/${scoreType}/analysis.json`)
        ]);

        if (!claudeResp.ok || !variantsResp.ok) {
            throw new Error('Failed to load experiment data');
        }

        const claudeData = await claudeResp.json();
        const variantsData = await variantsResp.json();
        const analysisData = analysisResp.ok ? await analysisResp.json() : null;

        // Convert claude scores to dict keyed by id
        const claudeScores = {};
        if (claudeData.scores) {
            for (const s of claudeData.scores) {
                claudeScores[s.id] = s;
            }
        }

        return {
            claude: claudeScores,
            variants: variantsData,
            analysis: analysisData,
            metadata: claudeData.metadata || {}
        };
    } catch (error) {
        console.error('Error loading judge data:', error);
        return null;
    }
}

function renderJudgeScatterPlot(containerId, claudeScores, variantScores, variantName, scoreType = 'trait') {
    const x = [];
    const y = [];
    const text = [];
    const colors = [];

    const claudeKey = scoreType === 'coherence' ? 'claude_coherence' : 'claude_trait';

    for (const result of variantScores) {
        const claude = claudeScores[result.id];
        if (!claude || result.score === null) continue;

        x.push(claude[claudeKey]);
        y.push(result.score);
        text.push(`${result.id}<br>Claude: ${claude[claudeKey]}<br>${variantName}: ${result.score}`);

        const traitColors = { refusal: '#4a9eff', evil: '#ff6b6b', sycophancy: '#ffd93d' };
        colors.push(traitColors[result.trait] || '#888');
    }

    // Calculate correlation line
    const n = x.length;
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((acc, xi, i) => acc + xi * y[i], 0);
    const sumX2 = x.reduce((acc, xi) => acc + xi * xi, 0);
    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;

    const trace = {
        x, y, text,
        mode: 'markers',
        type: 'scatter',
        marker: { color: colors, size: 10, opacity: 0.7 },
        hoverinfo: 'text'
    };

    const line = {
        x: [0, 100],
        y: [0, 100],
        mode: 'lines',
        type: 'scatter',
        line: { color: 'rgba(255,255,255,0.3)', dash: 'dash', width: 1 },
        hoverinfo: 'skip',
        showlegend: false
    };

    const fitLine = {
        x: [0, 100],
        y: [intercept, slope * 100 + intercept],
        mode: 'lines',
        type: 'scatter',
        line: { color: 'var(--accent)', width: 2 },
        hoverinfo: 'skip',
        showlegend: false
    };

    const layout = window.buildChartLayout({
        preset: 'layerChart',
        height: 280,
        legendPosition: 'none',
        xaxis: {
            title: { text: 'Claude Score', font: { color: '#aaa' } },
            range: [-5, 105]
        },
        yaxis: {
            title: { text: 'Variant Score', font: { color: '#aaa' } },
            range: [-5, 105]
        }
    });
    layout.title = { text: variantName, font: { color: '#e0e0de', size: 14 } };

    window.renderChart(containerId, [line, trace, fitLine], layout);
}

function renderJudgeMetricsTable(analysis) {
    if (!analysis?.variants) return '<p class="muted">No analysis data</p>';

    const variants = Object.entries(analysis.variants);
    const bestVariant = variants.reduce((best, [name, m]) =>
        (m.spearman || 0) > (best[1].spearman || 0) ? [name, m] : best
    )[0];

    let html = `
        <table class="data-table">
            <thead>
                <tr>
                    <th>Variant</th>
                    <th>Spearman</th>
                    <th>Pearson</th>
                    <th>MAE</th>
                    <th>Pairwise</th>
                </tr>
            </thead>
            <tbody>
    `;

    for (const [name, metrics] of variants) {
        const pairwise = metrics.pairwise_agreement
            ? `${(metrics.pairwise_agreement * 100).toFixed(1)}%`
            : 'N/A';
        const isBest = name === bestVariant;

        html += `
            <tr ${isBest ? 'class="highlight-row"' : ''}>
                <td><strong>${name}</strong></td>
                <td>${metrics.spearman?.toFixed(3) || 'N/A'}</td>
                <td>${metrics.pearson?.toFixed(3) || 'N/A'}</td>
                <td>${metrics.mae?.toFixed(1) || 'N/A'}</td>
                <td>${pairwise}</td>
            </tr>
        `;
    }

    html += '</tbody></table>';
    return html;
}

function renderJudgeResponseTable(claudeScores, variants, scoreType = 'trait', filterTrait = null) {
    const rows = [];
    const variantNames = Object.keys(variants);
    const claudeKey = scoreType === 'coherence' ? 'claude_coherence' : 'claude_trait';

    for (const [id, claude] of Object.entries(claudeScores)) {
        if (filterTrait && claude.trait !== filterTrait) continue;

        const row = {
            id,
            trait: claude.trait,
            prompt: claude.prompt,
            promptShort: claude.prompt?.substring(0, 50) + '...',
            response: claude.response,
            claude_score: claude[claudeKey],
            scores: {},
            rawOutputs: {}
        };

        for (const vName of variantNames) {
            const vResult = variants[vName].find(r => r.id === id);
            row.scores[vName] = vResult?.score;
            if (vResult?.raw_output) {
                row.rawOutputs[vName] = vResult.raw_output;
            }
        }

        row.maxDiff = Math.max(...variantNames.map(v =>
            Math.abs((row.scores[v] || 0) - row.claude_score)
        ));

        rows.push(row);
    }

    rows.sort((a, b) => b.maxDiff - a.maxDiff);

    // Store rows globally for expansion
    window._judgeTableRows = rows;
    window._judgeTableVariants = variantNames;

    let html = `
        <div class="table-controls">
            <select id="trait-filter" onchange="window.filterJudgeTable(this.value)">
                <option value="">All traits</option>
                <option value="refusal">Refusal</option>
                <option value="evil">Evil</option>
                <option value="sycophancy">Sycophancy</option>
            </select>
        </div>
        <div class="table-scroll">
        <table class="data-table response-table">
            <thead>
                <tr>
                    <th></th>
                    <th>ID</th>
                    <th>Trait</th>
                    <th>Claude</th>
                    ${variantNames.map(v => `<th>${v}</th>`).join('')}
                    <th>Max Diff</th>
                </tr>
            </thead>
            <tbody>
    `;

    for (const row of rows) {
        html += `
            <tr class="expandable-row" data-id="${row.id}" onclick="window.toggleJudgeDetail('${row.id}')">
                <td class="expand-icon">▶</td>
                <td><code>${row.id}</code></td>
                <td><span class="trait-badge trait-${row.trait}">${row.trait}</span></td>
                <td>${row.claude_score}</td>
                ${variantNames.map(v => {
                    const score = row.scores[v];
                    const diff = score !== null ? score - row.claude_score : null;
                    const diffStr = diff !== null ? (diff > 0 ? `+${diff.toFixed(0)}` : diff.toFixed(0)) : '';
                    return `<td>${score !== null ? score.toFixed(0) : '-'} <span class="diff-label">${diffStr}</span></td>`;
                }).join('')}
                <td>${row.maxDiff.toFixed(0)}</td>
            </tr>
            <tr class="detail-row" id="detail-${row.id}" style="display: none;">
                <td colspan="${variantNames.length + 5}">
                    <div class="response-detail-content"></div>
                </td>
            </tr>
        `;
    }

    html += '</tbody></table></div>';
    return html;
}

window.toggleJudgeDetail = function(id) {
    const detailRow = document.getElementById(`detail-${id}`);
    const mainRow = document.querySelector(`tr.expandable-row[data-id="${id}"]`);
    if (!detailRow || !mainRow) return;

    const isOpen = detailRow.style.display !== 'none';

    if (isOpen) {
        // Collapse
        detailRow.style.display = 'none';
        mainRow.querySelector('.expand-icon').textContent = '▶';
        mainRow.classList.remove('expanded');
    } else {
        // Expand - populate content if not already
        const contentDiv = detailRow.querySelector('.response-detail-content');
        if (!contentDiv.innerHTML) {
            const row = window._judgeTableRows.find(r => r.id === id);
            if (row) {
                contentDiv.innerHTML = renderJudgeDetailContent(row);
            }
        }
        detailRow.style.display = 'table-row';
        mainRow.querySelector('.expand-icon').textContent = '▼';
        mainRow.classList.add('expanded');
    }
};

function renderJudgeDetailContent(row) {
    const variantNames = window._judgeTableVariants || [];

    let html = `
        <div class="detail-grid">
            <div class="detail-section">
                <h4>Prompt</h4>
                <p class="prompt-text">${row.prompt || 'N/A'}</p>
            </div>
            <div class="detail-section">
                <h4>Response</h4>
                <div class="response-text">${row.response?.substring(0, 800) || 'N/A'}${row.response?.length > 800 ? '...' : ''}</div>
            </div>
        </div>
    `;

    // Add CoT reasoning for each variant that has it
    const reasonings = Object.entries(row.rawOutputs).filter(([_, v]) => v);
    if (reasonings.length > 0) {
        html += '<div class="cot-reasoning-section"><h4>CoT Reasoning</h4><div class="cot-grid">';
        for (const [vName, rawOutput] of reasonings) {
            const score = row.scores[vName];
            const diff = score !== null ? score - row.claude_score : null;
            const diffStr = diff !== null ? (diff > 0 ? `+${diff.toFixed(0)}` : diff.toFixed(0)) : '';
            html += `
                <div class="cot-item">
                    <div class="cot-header">${vName} <span class="diff-label">(${diffStr})</span></div>
                    <pre class="cot-output">${rawOutput}</pre>
                </div>
            `;
        }
        html += '</div></div>';
    }

    return html;
}

async function renderJudgeView() {
    const container = document.getElementById('content-area');
    container.innerHTML = '<div class="loading">Loading experiment data...</div>';

    oneOffData = await loadJudgeData(currentOneOff.name, currentScoreType);

    if (!oneOffData) {
        container.innerHTML = '<div class="error">Failed to load experiment data</div>';
        return;
    }

    const variantNames = Object.keys(oneOffData.variants);
    const baseline = oneOffData.analysis?.variants?.no_cot?.spearman || 0;
    const best = Object.entries(oneOffData.analysis?.variants || {})
        .reduce((best, [name, m]) => (m.spearman || 0) > (best[1] || 0) ? [name, m.spearman] : best, ['', 0]);

    container.innerHTML = `
        <div class="view-content one-offs-view">
            <section class="card">
                <h2>Judge Optimization</h2>
                <p class="muted">Testing if CoT improves trait scoring alignment with Claude</p>

                <div class="score-type-toggle">
                    <button class="score-type-btn ${currentScoreType === 'trait' ? 'active' : ''}" onclick="window.switchScoreType('trait')">Trait Scores</button>
                    <button class="score-type-btn ${currentScoreType === 'coherence' ? 'active' : ''}" onclick="window.switchScoreType('coherence')">Coherence Scores</button>
                </div>

                <h3>Overall Metrics</h3>
                ${renderJudgeMetricsTable(oneOffData.analysis)}

                <div class="insight-box">
                    <strong>Key Finding:</strong> ${best[0]} improves Spearman from ${baseline.toFixed(3)} to ${best[1].toFixed(3)}
                    (+${((best[1] - baseline) * 100 / baseline).toFixed(0)}%).
                </div>
            </section>

            <section class="card">
                <h3>Claude vs Variant Scores</h3>
                <p class="muted">Diagonal = perfect agreement. Points colored by trait.</p>
                <div class="scatter-grid">
                    ${variantNames.map(v => `<div id="scatter-${v}" class="scatter-plot"></div>`).join('')}
                </div>
                <div class="legend-row">
                    <span class="legend-item"><span class="legend-dot" style="background: #4a9eff"></span> Refusal</span>
                    <span class="legend-item"><span class="legend-dot" style="background: #ff6b6b"></span> Evil</span>
                    <span class="legend-item"><span class="legend-dot" style="background: #ffd93d"></span> Sycophancy</span>
                </div>
            </section>

            <section class="card">
                <h3>Trait Score Comparison</h3>
                <p class="muted">Click row to see full response + CoT reasoning. Sorted by max disagreement.</p>
                <div id="response-table-container">
                    ${renderJudgeResponseTable(oneOffData.claude, oneOffData.variants, currentScoreType)}
                </div>
            </section>
        </div>
    `;

    for (const vName of variantNames) {
        renderJudgeScatterPlot(`scatter-${vName}`, oneOffData.claude, oneOffData.variants[vName], vName, currentScoreType);
    }
}

// ============================================================================
// Prefill Dynamics Data Loading & Rendering
// ============================================================================

// Available model variants for prefill-dynamics
const DYNAMICS_VARIANTS = {
    'gemma-base': {
        label: 'Gemma-2-2B (base)',
        metrics: 'activation_metrics.json',
        perplexity: 'perplexity.json',
        hasProjections: true
    },
    'gemma-instruct': {
        label: 'Gemma-2-2B (instruct)',
        metrics: 'activation_metrics-instruct.json',
        perplexity: 'perplexity-instruct.json',
        hasProjections: false
    },
    'llama': {
        label: 'Llama-3.1-8B',
        metrics: 'activation_metrics-llama.json',
        perplexity: null,
        hasProjections: false
    }
};

let currentDynamicsVariant = 'gemma-base';

async function loadDynamicsData(variant = 'gemma-base') {
    const config = DYNAMICS_VARIANTS[variant];
    if (!config) return null;

    try {
        const fetches = [
            fetch(`/experiments/prefill-dynamics/analysis/${config.metrics}`)
        ];

        if (config.perplexity) {
            fetches.push(fetch(`/experiments/prefill-dynamics/analysis/${config.perplexity}`));
        }

        if (config.hasProjections) {
            fetches.push(fetch('/experiments/prefill-dynamics/analysis/projection_stability-hum_sycophancy.json'));
            fetches.push(fetch('/experiments/prefill-dynamics/analysis/projection_stability-chirp_refusal.json'));
        }

        const responses = await Promise.all(fetches);
        const metricsResp = responses[0];

        const data = {
            variant,
            variantLabel: config.label,
            metrics: metricsResp.ok ? await metricsResp.json() : null,
            perplexity: null,
            projections: {}
        };

        let idx = 1;
        if (config.perplexity) {
            data.perplexity = responses[idx]?.ok ? await responses[idx].json() : null;
            idx++;
        }

        if (config.hasProjections) {
            if (responses[idx]?.ok) data.projections.sycophancy = await responses[idx].json();
            idx++;
            if (responses[idx]?.ok) data.projections.refusal = await responses[idx].json();
        }

        return data;
    } catch (error) {
        console.error('Error loading dynamics data:', error);
        return null;
    }
}

function renderSmoothnessChart(containerId, byLayer) {
    const layers = Object.keys(byLayer).map(Number).sort((a, b) => a - b);
    const cohensD = layers.map(l => byLayer[l].smoothness_cohens_d);

    const trace = {
        x: layers,
        y: cohensD,
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Raw Smoothness',
        line: { color: '#2d4a5e', width: 2 },
        marker: { size: 6 },
        hovertemplate: 'Layer %{x}<br>d = %{y:.2f}<extra></extra>'
    };

    // Reference line at d=0.8 (large effect)
    const refLine = {
        x: [0, Math.max(...layers)],
        y: [0.8, 0.8],
        type: 'scatter',
        mode: 'lines',
        line: { color: '#888', dash: 'dash', width: 1 },
        showlegend: true,
        name: 'Large effect (d=0.8)',
        hoverinfo: 'skip'
    };

    const layout = window.buildChartLayout({
        preset: 'layerChart',
        traces: [trace, refLine],
        height: 300,
        legendPosition: 'above',
        xaxis: { title: { text: 'Layer', standoff: 5 } },
        yaxis: { title: { text: "Cohen's d (Human - Model)", standoff: 5 } }
    });

    window.renderChart(containerId, [trace, refLine], layout);
}

function renderProjectionStabilityChart(containerId, projections) {
    const traces = [];
    const colors = { refusal: '#51cf66', sycophancy: '#9775fa' };

    for (const [trait, data] of Object.entries(projections)) {
        if (!data?.by_layer) continue;

        const layers = Object.keys(data.by_layer).map(Number).sort((a, b) => a - b);
        const cohensD = layers.map(l => data.by_layer[l].var_cohens_d);

        traces.push({
            x: layers,
            y: cohensD,
            type: 'scatter',
            mode: 'lines+markers',
            name: trait.charAt(0).toUpperCase() + trait.slice(1),
            line: { color: colors[trait] || '#888', width: 2 },
            marker: { size: 5, symbol: 'square' },
            hovertemplate: `${trait}<br>Layer %{x}<br>d = %{y:.2f}<extra></extra>`
        });
    }

    // Reference line
    traces.push({
        x: [0, 25],
        y: [0.8, 0.8],
        type: 'scatter',
        mode: 'lines',
        line: { color: '#888', dash: 'dash', width: 1 },
        name: 'Large effect',
        hoverinfo: 'skip'
    });

    const layout = window.buildChartLayout({
        preset: 'layerChart',
        traces,
        height: 300,
        legendPosition: 'above',
        xaxis: { title: { text: 'Layer', standoff: 5 } },
        yaxis: { title: { text: "Cohen's d (Projection Variance: Human - Model)", standoff: 5 } }
    });

    window.renderChart(containerId, traces, layout);
}

function renderEffectComparisonChart(containerId, metrics, projections) {
    const traces = [];

    // Raw smoothness - bright color for visibility
    if (metrics?.summary?.by_layer) {
        const byLayer = metrics.summary.by_layer;
        const layers = Object.keys(byLayer).map(Number).sort((a, b) => a - b);
        traces.push({
            x: layers,
            y: layers.map(l => byLayer[l].smoothness_cohens_d),
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Raw Smoothness',
            line: { color: '#4a9eff', width: 3 },
            marker: { size: 7 }
        });
    }

    // Projection stability for each trait
    const colors = { refusal: '#51cf66', sycophancy: '#9775fa' };
    for (const [trait, data] of Object.entries(projections)) {
        if (!data?.by_layer) continue;
        const layers = Object.keys(data.by_layer).map(Number).sort((a, b) => a - b);
        traces.push({
            x: layers,
            y: layers.map(l => data.by_layer[l].var_cohens_d),
            type: 'scatter',
            mode: 'lines+markers',
            name: `Projection (${trait})`,
            line: { color: colors[trait], width: 2, dash: 'dash' },
            marker: { size: 5, symbol: 'square' }
        });
    }

    // Reference line
    traces.push({
        x: [0, 25],
        y: [0.8, 0.8],
        type: 'scatter',
        mode: 'lines',
        line: { color: '#888', dash: 'dot', width: 1 },
        name: 'Large effect (d=0.8)',
        hoverinfo: 'skip'
    });

    const layout = window.buildChartLayout({
        preset: 'layerChart',
        traces,
        height: 350,
        legendPosition: 'above',
        xaxis: { title: { text: 'Layer', standoff: 5 } },
        yaxis: { title: { text: "Cohen's d", standoff: 5 } }
    });
    layout.title = { text: 'Effect Size by Layer: Raw Smoothness vs Projection Stability', font: { size: 14, color: '#e0e0de' } };

    window.renderChart(containerId, traces, layout);
}

function renderPerplexityScatter(containerId, pplData, metricsData) {
    if (!pplData?.results || !metricsData?.samples) return;

    // Match samples by index
    const x = [];  // smoothness diff
    const y = [];  // perplexity diff
    const text = [];

    for (const ppl of pplData.results) {
        const sample = metricsData.samples.find(s => s.id === ppl.id);
        if (!sample) continue;

        // Compute mean smoothness across layers (sample.human is keyed by layer number)
        const humanLayers = Object.keys(sample.human).map(Number);
        const humanSmooth = humanLayers.reduce((sum, l) => sum + sample.human[l].smoothness, 0) / humanLayers.length;
        const modelSmooth = humanLayers.reduce((sum, l) => sum + sample.model[l].smoothness, 0) / humanLayers.length;
        const smoothDiff = humanSmooth - modelSmooth;

        x.push(smoothDiff);
        y.push(ppl.ce_diff);
        text.push(`Sample ${ppl.id}<br>Smoothness diff: ${smoothDiff.toFixed(1)}<br>CE diff: ${ppl.ce_diff.toFixed(2)}`);
    }

    // Regression line
    const n = x.length;
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((acc, xi, i) => acc + xi * y[i], 0);
    const sumX2 = x.reduce((acc, xi) => acc + xi * xi, 0);
    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;

    const minX = Math.min(...x);
    const maxX = Math.max(...x);

    const traces = [
        {
            x, y, text,
            type: 'scatter',
            mode: 'markers',
            marker: { color: '#4a9eff', size: 8, opacity: 0.7 },
            hoverinfo: 'text',
            name: 'Samples'
        },
        {
            x: [minX, maxX],
            y: [slope * minX + intercept, slope * maxX + intercept],
            type: 'scatter',
            mode: 'lines',
            line: { color: '#ff6b6b', width: 2 },
            name: `Fit (r=${Math.sqrt(sumXY * sumXY / (sumX2 * y.reduce((a, b) => a + b*b, 0))).toFixed(2)})`
        }
    ];

    const layout = window.buildChartLayout({
        preset: 'layerChart',
        traces,
        height: 300,
        legendPosition: 'above',
        xaxis: { title: { text: 'Smoothness Diff (Human - Model)', standoff: 5 } },
        yaxis: { title: { text: 'Cross-Entropy Diff', standoff: 5 } }
    });

    window.renderChart(containerId, traces, layout);
}

function renderDistributionChart(containerId, samples, metric = 'smoothness') {
    if (!samples?.length) return;

    // Compute mean across layers for each sample (sample.human is keyed by layer number)
    const humanVals = samples.map(s => {
        const layers = Object.keys(s.human).map(Number);
        return layers.reduce((sum, l) => sum + s.human[l][metric], 0) / layers.length;
    });
    const modelVals = samples.map(s => {
        const layers = Object.keys(s.model).map(Number);
        return layers.reduce((sum, l) => sum + s.model[l][metric], 0) / layers.length;
    });

    // Split violin: Human on left, Model on right (same x position = touching)
    const traces = [
        {
            y: humanVals,
            x: humanVals.map(() => 0),
            type: 'violin',
            name: 'Human',
            side: 'negative',
            line: { color: '#ff6b6b' },
            fillcolor: 'rgba(255, 107, 107, 0.5)',
            meanline: { visible: true },
            points: false
        },
        {
            y: modelVals,
            x: modelVals.map(() => 0),
            type: 'violin',
            name: 'Model',
            side: 'positive',
            line: { color: '#4a9eff' },
            fillcolor: 'rgba(74, 158, 255, 0.5)',
            meanline: { visible: true },
            points: false
        }
    ];

    const layout = window.buildChartLayout({
        preset: 'barChart',
        traces,
        height: 300,
        legendPosition: 'above',
        xaxis: { showticklabels: false, zeroline: false },
        yaxis: { title: { text: `${metric.charAt(0).toUpperCase() + metric.slice(1)} (L2 norm)`, standoff: 5 } }
    });

    window.renderChart(containerId, traces, layout);
}

async function renderDynamicsView() {
    const container = document.getElementById('content-area');
    container.innerHTML = '<div class="loading">Loading dynamics data...</div>';

    oneOffData = await loadDynamicsData(currentDynamicsVariant);

    if (!oneOffData) {
        container.innerHTML = '<div class="error">Failed to load dynamics data</div>';
        return;
    }

    // Calculate summary stats
    const byLayer = oneOffData.metrics?.summary?.by_layer || {};
    const layers = Object.keys(byLayer).map(Number);
    const avgEffect = layers.length > 0
        ? (layers.reduce((sum, l) => sum + byLayer[l].smoothness_cohens_d, 0) / layers.length).toFixed(2)
        : 'N/A';
    const peakLayer = layers.reduce((best, l) =>
        byLayer[l].smoothness_cohens_d > (byLayer[best]?.smoothness_cohens_d || 0) ? l : best, layers[0]);

    // Build model selector
    const modelOptions = Object.entries(DYNAMICS_VARIANTS).map(([key, cfg]) =>
        `<option value="${key}" ${key === currentDynamicsVariant ? 'selected' : ''}>${cfg.label}</option>`
    ).join('');

    // Conditional sections based on data availability
    const hasProjections = Object.keys(oneOffData.projections).length > 0;
    const hasPerplexity = oneOffData.perplexity !== null;

    container.innerHTML = `
        <div class="view-content one-offs-view dynamics-view">
            <section class="card">
                <h2>Prefill Dynamics: Human vs Model Text</h2>
                <p class="muted">Comparing activation patterns when model processes human-written vs model-generated text</p>

                <div class="model-selector">
                    <label>Model:</label>
                    <select onchange="window.switchDynamicsVariant(this.value)">
                        ${modelOptions}
                    </select>
                    <span class="muted model-info">${oneOffData.variantLabel}</span>
                </div>

                <div class="insight-box">
                    <strong>Key Finding:</strong> Model text produces smoother activation trajectories (avg d=${avgEffect}).
                    Peak separation at layer ${peakLayer}. ${layers.includes(25) ? 'Effect reverses at output layer (layer 25).' : ''}
                </div>
            </section>

            <section class="card">
                <h3>Effect Comparison by Layer</h3>
                <p class="muted">Raw smoothness (blue) shows large, consistent effect.${hasProjections ? ' Projection stability (dashed) is trait-dependent.' : ''}</p>
                <div id="chart-effect-comparison" class="chart-container"></div>
            </section>

            <section class="card">
                <h3>Raw Smoothness by Layer</h3>
                <p class="muted">Cohen's d for token-to-token activation deltas. Positive = human text is jumpier.</p>
                <div id="chart-smoothness" class="chart-container"></div>
            </section>

            <div class="two-column">
                <section class="card ${!hasPerplexity ? 'disabled-section' : ''}">
                    <h3>Smoothness vs Perplexity</h3>
                    <p class="muted">${hasPerplexity ? 'Correlation between smoothness diff and cross-entropy diff.' : 'Perplexity data not available for this model.'}</p>
                    <div id="chart-ppl-scatter" class="chart-container"></div>
                </section>

                <section class="card">
                    <h3>Distribution</h3>
                    <p class="muted">Per-sample smoothness values (mean across layers)</p>
                    <div id="chart-distribution" class="chart-container"></div>
                </section>
            </div>

            <section class="card methodology-section">
                <h3>Methodology</h3>
                <div class="methodology-content">

                    <h4>1. Data Setup</h4>
                    <p>50 paragraph samples from <strong>WikiText</strong>. For each sample, we take the first sentence as a shared prompt,
                    then compare two continuations:</p>

                    <div class="example-legend">
                        <span class="legend-item"><span class="legend-swatch prefill-swatch"></span> Prefill (shared prompt)</span>
                        <span class="legend-item"><span class="legend-swatch human-swatch"></span> Human-written</span>
                        <span class="legend-item"><span class="legend-swatch model-swatch"></span> Model-generated</span>
                    </div>

                    <div class="example-comparison">
                        <div class="example-box">
                            <div class="example-label">Human Condition</div>
                            <div class="example-text">
                                <span class="prefill-token">Du Fu was a prominent Chinese poet of the Tang dynasty.</span>
                                <span class="human-token"> Along with Li Bai, he is frequently called the greatest of the Chinese poets. His greatest ambition was to serve his country...</span>
                            </div>
                        </div>
                        <div class="example-box">
                            <div class="example-label">Model Condition</div>
                            <div class="example-text">
                                <span class="prefill-token">Du Fu was a prominent Chinese poet of the Tang dynasty.</span>
                                <span class="model-token"> He was born in the city of Luoyang, Henan Province, and died in the city of Chengdu. He was a member of the imperial civil service...</span>
                            </div>
                        </div>
                    </div>

                    <h4>2. Metrics</h4>

                    <p><strong>Activation Smoothness</strong> — Mean L2 distance between consecutive hidden states:</p>
                    <div class="math-block">\\[ \\text{smoothness}(x) = \\frac{1}{T-1} \\sum_{t=1}^{T-1} \\| \\mathbf{h}_{t+1} - \\mathbf{h}_t \\|_2 \\]</div>

                    <p><strong>Projection Stability</strong> — Variance of activations projected onto trait vectors:</p>
                    <div class="math-block">\\[ \\text{var}_{\\text{proj}}(x) = \\text{Var}_t \\left( \\mathbf{h}_t \\cdot \\hat{\\mathbf{v}}_{\\text{trait}} \\right) \\]</div>

                    <p><strong>Effect Size</strong> — Cohen's d for human vs model comparison:</p>
                    <div class="math-block">\\[ d = \\frac{\\mu_{\\text{human}} - \\mu_{\\text{model}}}{\\sigma_{\\text{pooled}}} \\]</div>
                    <p class="muted">Interpretation: \\(|d| > 0.8\\) large, \\(|d| > 0.5\\) medium, \\(|d| > 0.2\\) small.</p>

                    <h4>3. Findings</h4>

                    <table class="data-table compact">
                        <thead><tr><th>Finding</th><th>Effect Size</th><th>p-value</th></tr></thead>
                        <tbody>
                            <tr><td>Model text has lower perplexity</td><td>CE: 1.45 vs 2.99</td><td>—</td></tr>
                            <tr><td>Model activations are smoother</td><td>d = 1.49</td><td>5.3e-14</td></tr>
                            <tr><td>Smoothness correlates with perplexity</td><td>r = 0.65</td><td>3.1e-13</td></tr>
                            <tr><td>Trait projections more stable for model</td><td>d = 0.3–1.0</td><td>< 0.01</td></tr>
                            <tr><td>Effect reverses at output layer (L25)</td><td>d = -1.5</td><td>< 0.01</td></tr>
                        </tbody>
                    </table>

                    <h4>4. Interpretation</h4>
                    <p>Model-generated text follows more <strong>predictable paths through activation space</strong>.
                    When the model processes its own outputs, token transitions are "expected" — creating smaller activation jumps.</p>

                    <p>This effect is <strong>layer-specific</strong>: strongest in middle layers (5-22), but <strong>reverses at layer 25</strong>
                    where representations converge toward vocabulary logits. The output layer behaves differently because it's
                    optimized for next-token prediction, not abstract representation.</p>
                </div>
            </section>
        </div>
    `;

    // Render charts
    renderEffectComparisonChart('chart-effect-comparison', oneOffData.metrics, oneOffData.projections);
    if (oneOffData.metrics?.summary?.by_layer) {
        renderSmoothnessChart('chart-smoothness', oneOffData.metrics.summary.by_layer);
    }
    if (hasPerplexity) {
        renderPerplexityScatter('chart-ppl-scatter', oneOffData.perplexity, oneOffData.metrics);
    }
    if (oneOffData.metrics?.samples) {
        renderDistributionChart('chart-distribution', oneOffData.metrics.samples);
    }

    // Render LaTeX math
    if (window.MathJax?.typeset) {
        window.MathJax.typeset();
    }
}

window.switchDynamicsVariant = async function(variant) {
    currentDynamicsVariant = variant;
    await renderDynamicsView();
};

// ============================================================================
// Global Event Handlers
// ============================================================================

window.filterJudgeTable = function(trait) {
    if (!oneOffData) return;
    const tableContainer = document.getElementById('response-table-container');
    tableContainer.innerHTML = renderJudgeResponseTable(oneOffData.claude, oneOffData.variants, currentScoreType, trait || null);
};

window.showJudgeDetail = function(id) {
    if (!oneOffData) return;

    const claude = oneOffData.claude[id];
    const modal = document.getElementById('preview-modal');
    const title = document.getElementById('preview-title');
    const body = document.getElementById('preview-body');

    title.textContent = `${id} - ${claude.trait}`;

    let html = `
        <div class="detail-section">
            <h4>Prompt</h4>
            <p>${claude.prompt}</p>
        </div>
        <div class="detail-section">
            <h4>Response</h4>
            <div class="response-text">${claude.response?.substring(0, 1000) || 'N/A'}${claude.response?.length > 1000 ? '...' : ''}</div>
        </div>
        <div class="detail-section">
            <h4>Scores</h4>
            <table class="data-table">
                <tr><th>Claude</th><td><strong>${claude.claude_trait}</strong> (coherence: ${claude.claude_coherence})</td></tr>
    `;

    for (const [vName, vResults] of Object.entries(oneOffData.variants)) {
        const vResult = vResults.find(r => r.id === id);
        if (vResult) {
            const diff = vResult.score - claude.claude_trait;
            html += `<tr><th>${vName}</th><td>${vResult.score?.toFixed(0) || 'N/A'} (diff: ${diff > 0 ? '+' : ''}${diff.toFixed(0)})</td></tr>`;
        }
    }
    html += '</table></div>';

    for (const [vName, vResults] of Object.entries(oneOffData.variants)) {
        const vResult = vResults.find(r => r.id === id);
        if (vResult?.raw_output) {
            html += `
                <div class="detail-section">
                    <h4>${vName} reasoning</h4>
                    <pre class="cot-output">${vResult.raw_output}</pre>
                </div>
            `;
        }
    }

    body.innerHTML = html;
    modal.classList.add('active');
};

window.switchScoreType = async function(type) {
    currentScoreType = type;
    await renderJudgeView();
};

window.selectOneOff = async function(name) {
    currentOneOff = discoveredOneOffs.find(e => e.name === name);
    if (!currentOneOff) return;

    // Update sidebar selection
    const list = document.getElementById('one-off-list');
    if (list) {
        list.querySelectorAll('.experiment-option').forEach(item => {
            item.classList.toggle('active', item.dataset.experiment === name);
            const radio = item.querySelector('input[type="radio"]');
            if (radio) radio.checked = item.dataset.experiment === name;
        });
    }

    await renderOneOffContent();
};

function populateOneOffSidebar() {
    const list = document.getElementById('one-off-list');
    if (!list) return;

    list.innerHTML = discoveredOneOffs.map((exp) => {
        const isActive = exp.name === currentOneOff?.name ? 'active' : '';
        const isChecked = exp.name === currentOneOff?.name ? 'checked' : '';
        const displayName = exp.name.replace(/[-_]/g, ' ');
        const typeLabel = exp.type === 'dynamics' ? ' (dynamics)' : '';
        return `<label class="experiment-option ${isActive}" data-experiment="${exp.name}">
            <input type="radio" name="one-off" ${isChecked}>
            <span>${displayName}${typeLabel}</span>
        </label>`;
    }).join('');

    list.querySelectorAll('.experiment-option').forEach(item => {
        item.addEventListener('click', () => {
            window.selectOneOff(item.dataset.experiment);
        });
    });
}

async function renderOneOffContent() {
    if (!currentOneOff) return;

    if (currentOneOff.type === 'dynamics') {
        await renderDynamicsView();
    } else {
        await renderJudgeView();
    }
}

window.renderOneOffs = async function() {
    const container = document.getElementById('content-area');

    // Show one-off sidebar section, hide regular experiment section
    const oneOffSection = document.getElementById('one-off-section');
    const experimentSection = document.getElementById('experiment-section');
    if (oneOffSection) oneOffSection.style.display = 'block';
    if (experimentSection) experimentSection.style.display = 'none';

    container.innerHTML = '<div class="loading">Discovering experiments...</div>';

    discoveredOneOffs = await discoverOneOffExperiments();

    if (discoveredOneOffs.length === 0) {
        container.innerHTML = `
            <div class="tool-view">
                <div class="no-data">
                    <p>No one-off experiments found</p>
                    <small>Supported: judge_optimization, prefill-dynamics</small>
                </div>
            </div>
        `;
        return;
    }

    currentOneOff = currentOneOff || discoveredOneOffs[0];
    populateOneOffSidebar();
    await renderOneOffContent();
};
