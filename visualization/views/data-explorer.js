// Data Explorer View - Full file tree with preview modal

let integrityData = null;

async function fetchIntegrityData() {
    const experiment = window.paths?.getExperiment();
    if (!experiment) return null;

    try {
        const response = await fetch(`/api/integrity/${experiment}.json`);
        if (!response.ok) return null;
        const data = await response.json();
        return data.error ? null : data;
    } catch (error) {
        console.error('Failed to fetch integrity data:', error);
        return null;
    }
}

function toggleFolder(id) {
    const el = document.getElementById(id);
    const toggle = document.getElementById('toggle-' + id);
    if (el.style.display === 'none') {
        el.style.display = 'block';
        toggle.textContent = '▾';
    } else {
        el.style.display = 'none';
        toggle.textContent = '▸';
    }
}

function folder(name, id, hint, indent, children, complete = true, isLeaf = false) {
    const pad = indent * 16;
    const expanded = !isLeaf && indent < 2;  // Expand non-leaf folders at depths 0, 1
    const display = expanded ? 'block' : 'none';
    const arrow = expanded ? '▾' : '▸';
    const hintColor = complete ? 'var(--text-tertiary)' : 'var(--warning)';
    return `
        <div style="padding: 2px 0; padding-left: ${pad}px; cursor: pointer;" onclick="toggleFolder('${id}')">
            <span id="toggle-${id}" style="color: var(--text-tertiary); width: 12px; display: inline-block;">${arrow}</span>
            📁 <strong>${name}</strong>
            ${hint ? `<span style="color: ${hintColor}; font-size: 11px; margin-left: 8px;">${hint}</span>` : ''}
        </div>
        <div id="${id}" style="display: ${display};">
            ${children}
        </div>
    `;
}

function item(icon, name, ok, hint, indent) {
    const pad = indent * 16;
    const color = ok ? 'inherit' : 'var(--warning)';
    const hintColor = ok ? 'var(--text-tertiary)' : 'var(--warning)';
    return `
        <div style="padding: 1px 0; padding-left: ${pad}px; color: ${color};">
            <span style="width: 12px; display: inline-block;"></span>
            ${icon} ${name}
            ${hint ? `<span style="color: ${hintColor}; font-size: 11px; margin-left: 8px;">${hint}</span>` : ''}
        </div>
    `;
}

function renderTraitFiles(trait, nLayers, baseIndent) {
    const id = trait.trait.replace(/\//g, '-');
    let html = '';
    let allComplete = true;
    const folderIndent = baseIndent;
    const itemIndent = baseIndent + 1;

    // Prompts (leaf folder - collapsed)
    const promptsOk = Object.values(trait.prompts).filter(v => v).length;
    const promptsTotal = Object.keys(trait.prompts).length;
    const promptsComplete = promptsOk >= promptsTotal;
    if (!promptsComplete) allComplete = false;
    let promptsHtml = '';
    for (const [file, exists] of Object.entries(trait.prompts)) {
        promptsHtml += item(exists ? '✓' : '✗', file, exists, '', itemIndent);
    }
    html += folder('prompts', `${id}-prompts`, `(${promptsOk}/${promptsTotal})`, folderIndent, promptsHtml, promptsComplete);

    // Responses (leaf folder - collapsed)
    const responsesOk = Object.values(trait.responses).filter(v => v).length;
    const responsesTotal = Object.keys(trait.responses).length;
    const responsesComplete = responsesOk >= responsesTotal;
    if (!responsesComplete) allComplete = false;
    let responsesHtml = '';
    for (const [file, exists] of Object.entries(trait.responses)) {
        responsesHtml += item(exists ? '✓' : '✗', file, exists, '', itemIndent);
    }
    html += folder('responses', `${id}-resp`, `(${responsesOk}/${responsesTotal})`, folderIndent, responsesHtml, responsesComplete);

    // Activations (leaf folder - collapsed)
    const actsTotal = Object.entries(trait.activations)
        .filter(([k]) => k.endsWith('_layers'))
        .reduce((sum, [, v]) => sum + v, 0);
    const actsComplete = actsTotal >= trait.expected_activations;
    if (!actsComplete) allComplete = false;
    let actsHtml = '';
    for (const [key, count] of Object.entries(trait.activations)) {
        if (key.endsWith('_layers')) {
            const prefix = key.replace('_layers', '');
            const ok = count >= nLayers;
            actsHtml += item(ok ? '✓' : '⚠', `${prefix}_layer*.pt`, ok, `(${count}/${nLayers})`, itemIndent);
        }
    }
    html += folder('activations', `${id}-acts`, `(${actsTotal}/${trait.expected_activations})`, folderIndent, actsHtml, actsComplete);

    // Vectors (leaf folder - collapsed)
    const methods = trait.methods || [];
    let methodsCompleteCount = 0;
    let vecsHtml = '';
    for (const method of methods) {
        const count = trait.vectors[`${method}_pt`] || 0;
        const ok = count >= nLayers;
        if (ok) methodsCompleteCount++;
        vecsHtml += item(ok ? '✓' : '⚠', `${method}_layer*.pt`, ok, `(${count}/${nLayers})`, itemIndent);
    }
    const vectorsComplete = methodsCompleteCount >= methods.length && methods.length > 0;
    if (!vectorsComplete) allComplete = false;
    const vectorsHint = methods.length ? `(${methodsCompleteCount}/${methods.length})` : '(none)';
    html += folder('vectors', `${id}-vecs`, vectorsHint, folderIndent, vecsHtml || item('✗', 'no vectors', false, '', itemIndent), vectorsComplete);

    return { html, complete: allComplete };
}

function renderFileTree(data) {
    let html = '';

    // EXTRACTION (indent 0 -> categories at 1 -> traits at 2 -> trait folders at 3 -> items at 4)
    const byCategory = {};
    for (const trait of data.traits) {
        if (!byCategory[trait.category]) byCategory[trait.category] = [];
        byCategory[trait.category].push(trait);
    }

    let extractionHtml = '';
    let extractionComplete = true;
    for (const [category, traits] of Object.entries(byCategory).sort()) {
        let categoryHtml = '';
        let categoryComplete = true;
        for (const trait of traits.sort((a, b) => a.trait.localeCompare(b.trait))) {
            const name = trait.trait.split('/').pop();
            const { html: traitHtml, complete: traitComplete } = renderTraitFiles(trait, data.n_layers, 3);
            const status = traitComplete ? '✓' : '<span style="color: var(--warning)">⚠</span>';
            if (!traitComplete) categoryComplete = false;
            categoryHtml += folder(`${status} ${name}`, `trait-${trait.trait.replace(/\//g, '-')}`, '', 2, traitHtml, traitComplete);
        }
        if (!categoryComplete) extractionComplete = false;
        extractionHtml += folder(category, `cat-${category}`, `(${traits.length} traits)`, 1, categoryHtml, categoryComplete);
    }
    if (!data.evaluation_exists) extractionComplete = false;
    extractionHtml += item(data.evaluation_exists ? '✓' : '✗', 'extraction_evaluation.json', data.evaluation_exists, '', 1);
    html += folder('extraction', 'extraction', `(${data.traits.length} traits)`, 0, extractionHtml, extractionComplete);  // extraction at indent 0

    // INFERENCE (indent 0 -> sub-folders at 1 -> items at 2)
    if (data.inference) {
        const inf = data.inference;
        let inferenceHtml = '';

        if (Object.keys(inf.prompt_sets || {}).length > 0) {
            let promptsHtml = '';
            for (const [name, exists] of Object.entries(inf.prompt_sets)) {
                promptsHtml += item(exists ? '✓' : '✗', `${name}.json`, exists, '', 2);
            }
            inferenceHtml += folder('prompts', 'inf-prompts', `(${Object.keys(inf.prompt_sets).length} sets)`, 1, promptsHtml, true, true);  // isLeaf=true
        }

        if (Object.keys(inf.raw_types || {}).length > 0) {
            let rawHtml = '';
            for (const [type, info] of Object.entries(inf.raw_types)) {
                let typeHtml = '';
                for (const [promptSet, count] of Object.entries(info.prompt_sets || {})) {
                    typeHtml += item('📄', `${promptSet}/`, true, `(${count} .pt)`, 3);
                }
                rawHtml += folder(type, `raw-${type}`, `(${info.total_files} files)`, 2, typeHtml);
            }
            inferenceHtml += folder('raw', 'inf-raw', '', 1, rawHtml);
        }

        html += folder('inference', 'inference', '', 0, inferenceHtml);
    }

    // ANALYSIS (indent 0 -> categories at 1 -> items at 2)
    if (data.analysis && Object.keys(data.analysis.categories || {}).length > 0) {
        let analysisHtml = '';
        for (const [category, info] of Object.entries(data.analysis.categories).sort()) {
            let catHtml = '';
            if (info.pngs > 0) catHtml += item('🖼', '*.png', true, `(${info.pngs})`, 2);
            if (info.jsons > 0) catHtml += item('📄', '*.json', true, `(${info.jsons})`, 2);
            if (info.pts > 0) catHtml += item('🔢', '*.pt', true, `(${info.pts})`, 2);
            analysisHtml += folder(category, `analysis-${category}`, `(${info.total_files} files)`, 1, catHtml, true, true);  // isLeaf=true
        }
        html += folder('analysis', 'analysis', `(${data.analysis.total_files} files)`, 0, analysisHtml);
    }

    // STEERING (indent 0 -> traits at 1 -> files at 2)
    if (data.steering && data.steering.total_traits > 0) {
        let steeringHtml = '';
        for (const [trait, info] of Object.entries(data.steering.traits).sort()) {
            let traitHtml = '';
            const runsHint = info.n_runs > 0 ? `(${info.n_runs} runs)` : '';
            traitHtml += item(info.results ? '✓' : '✗', 'results.json', info.results, runsHint, 2);
            const complete = info.results && info.n_runs > 0;
            steeringHtml += folder(trait, `steering-${trait.replace(/\//g, '-')}`, '', 1, traitHtml, complete, true);
        }
        html += folder('steering', 'steering', `(${data.steering.total_traits} traits)`, 0, steeringHtml);
    }

    return html;
}

async function renderDataExplorer() {
    const contentArea = document.getElementById('content-area');
    if (!contentArea) return;

    const experiment = window.paths?.getExperiment();
    if (!experiment) {
        contentArea.innerHTML = `<div class="tool-view"><div class="no-data">No experiment selected</div></div>`;
        return;
    }

    if (!integrityData || integrityData.experiment !== experiment) {
        integrityData = await fetchIntegrityData();
    }

    if (!integrityData) {
        contentArea.innerHTML = `<div class="tool-view"><div class="no-data">No data available</div></div>`;
        return;
    }

    const summary = integrityData.summary;

    contentArea.innerHTML = `
        <div class="tool-view">
            <div class="page-intro">
                <div class="page-intro-text">Browse raw data files in this experiment.</div>
                <div class="intro-example">
                    <div><span class="pos">✓</span> file exists</div>
                    <div><span class="neg">✗</span> file missing</div>
                </div>
            </div>

            <div class="stats-row">
                <span><strong>Traits:</strong> ${summary.total_traits}</span>
                <span><strong>OK:</strong> <span class="quality-good">${summary.ok}</span></span>
                <span><strong>Partial:</strong> <span class="quality-ok">${summary.partial}</span></span>
                <span><strong>Methods:</strong> ${integrityData.methods?.join(', ') || 'none'}</span>
                <span><strong>Analysis:</strong> ${summary.analysis_categories || 0} categories</span>
            </div>

            <div class="file-tree">
                <div style="padding: 2px 0;">📁 <strong>experiments/${integrityData.experiment}/</strong></div>
                ${renderFileTree(integrityData)}
            </div>
        </div>
    `;
}

// Preview modal functions
async function previewFile(trait, file) {
    const modal = document.getElementById('preview-modal');
    const title = document.getElementById('preview-title');
    const body = document.getElementById('preview-body');

    title.textContent = `${trait}/${file}`;
    body.innerHTML = '<div class="loading">Loading...</div>';
    modal.classList.add('show');

    try {
        const url = window.paths.extractionFile(trait, file);
        const response = await fetch(url);
        if (!response.ok) throw new Error(response.statusText);
        const data = await response.json();

        const truncated = Array.isArray(data) ? data.slice(0, 10) : data;
        const note = Array.isArray(data) && data.length > 10
            ? `<div class="file-hint">Showing 10 of ${data.length} items</div>`
            : '';

        body.innerHTML = `<pre>${JSON.stringify(truncated, null, 2)}</pre>${note}`;
    } catch (error) {
        body.innerHTML = `<div class="error">Failed to load: ${error.message}</div>`;
    }
}

function closePreview() {
    document.getElementById('preview-modal')?.classList.remove('show');
}

// Document event listeners
document.addEventListener('click', (e) => {
    if (e.target.id === 'preview-modal') closePreview();
});

window.addEventListener('experimentChanged', () => { integrityData = null; });

// Exports
window.renderDataExplorer = renderDataExplorer;
window.toggleFolder = toggleFolder;
window.previewFile = previewFile;
window.closePreview = closePreview;
