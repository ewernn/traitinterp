// Data Explorer View - File browser based on integrity check data

// Cache for integrity data
let integrityData = null;

/**
 * Fetch integrity data for current experiment.
 */
async function fetchIntegrityData() {
    const experiment = window.paths.getExperiment();
    if (!experiment) {
        console.error('No experiment selected');
        return null;
    }

    try {
        const response = await fetch(`/api/integrity/${experiment}.json`);
        if (!response.ok) {
            throw new Error(`Failed to fetch integrity data: ${response.statusText}`);
        }
        const data = await response.json();
        if (data.error) {
            throw new Error(data.error);
        }
        return data;
    } catch (error) {
        console.error('Failed to fetch integrity data:', error);
        return null;
    }
}

/**
 * Get status icon for a given status.
 */
function getStatusIcon(status) {
    switch (status) {
        case 'ok': return '✅';
        case 'partial': return '⚠️';
        case 'empty': return '❌';
        case 'missing': return '❌';
        default: return '❓';
    }
}

/**
 * Get file icon based on existence.
 */
function getFileIcon(exists) {
    return exists ? '✓' : '✗';
}

/**
 * Render the file tree for a single trait from integrity data.
 */
function renderTraitFileTree(trait, integrityData) {
    const traitPath = `experiments/${integrityData.experiment}/extraction/${trait.trait}`;

    // Count totals
    const promptsOk = Object.values(trait.prompts).filter(v => v).length;
    const responsesOk = Object.values(trait.responses).filter(v => v).length;
    const totalActs = (trait.activations.pos_layers || 0) +
                      (trait.activations.neg_layers || 0) +
                      (trait.activations.val_pos_layers || 0) +
                      (trait.activations.val_neg_layers || 0);
    const totalVectors = Object.entries(trait.vectors)
        .filter(([k, v]) => k.endsWith('_pt'))
        .reduce((sum, [k, v]) => sum + v, 0);
    const totalMeta = Object.entries(trait.vectors)
        .filter(([k, v]) => k.endsWith('_meta'))
        .reduce((sum, [k, v]) => sum + v, 0);

    let html = `
        <div class="file-tree">
            <div class="file-item">
                <span class="file-icon">📁</span>
                <strong>${trait.trait}/</strong>
            </div>

            <!-- Prompt files -->
            <div class="file-item indent-1">
                <span class="file-icon">📁</span>
                <strong>prompts/</strong>
                <span style="opacity: 0.6; font-size: 11px;">(${promptsOk}/4 files)</span>
            </div>
    `;

    // Show individual prompt files
    const promptFiles = ['positive.txt', 'negative.txt', 'val_positive.txt', 'val_negative.txt'];
    for (const file of promptFiles) {
        const exists = trait.prompts[file] || false;
        html += `
            <div class="file-item indent-2">
                <span class="file-icon ${exists ? '' : 'missing'}">${getFileIcon(exists)}</span>
                <span class="${exists ? '' : 'missing'}">${file}</span>
            </div>
        `;
    }

    // Metadata files
    html += `
            <div class="file-item indent-1">
                <span class="file-icon">📄</span>
                <strong>metadata</strong>
            </div>
    `;

    const metadataFiles = ['generation_metadata.json', 'trait_definition.txt'];
    for (const file of metadataFiles) {
        const exists = trait.metadata[file] || false;
        const isClickable = exists && file.endsWith('.json');
        html += `
            <div class="file-item indent-2 ${isClickable ? 'clickable' : ''}" ${isClickable ? `onclick="previewMetadata('${trait.trait}', '${file}')"` : ''}>
                <span class="file-icon ${exists ? '' : 'missing'}">${getFileIcon(exists)}</span>
                <span class="${exists ? '' : 'missing'}">${file}</span>
                ${isClickable ? '<span style="opacity: 0.6; font-size: 11px;">[preview →]</span>' : ''}
            </div>
        `;
    }

    // Response files
    html += `
            <div class="file-item indent-1">
                <span class="file-icon">📁</span>
                <strong>responses/</strong>
                <span style="opacity: 0.6; font-size: 11px;">(${responsesOk}/4 files)</span>
            </div>
    `;

    const responseFiles = [
        ['responses/pos.json', 'pos.json', 'responses'],
        ['responses/neg.json', 'neg.json', 'responses'],
        ['val_responses/val_pos.json', 'val_pos.json', 'val_responses'],
        ['val_responses/val_neg.json', 'val_neg.json', 'val_responses']
    ];
    for (const [key, file, dir] of responseFiles) {
        const exists = trait.responses[key] || false;
        const isClickable = exists;
        const polarity = file.includes('pos') ? 'pos' : 'neg';
        const isVal = file.startsWith('val_');
        html += `
            <div class="file-item indent-2 ${isClickable ? 'clickable' : ''}" ${isClickable ? `onclick="previewResponses('${trait.trait}', '${polarity}', ${isVal})"` : ''}>
                <span class="file-icon ${exists ? '' : 'missing'}">${getFileIcon(exists)}</span>
                <span class="${exists ? '' : 'missing'}">${dir}/${file}</span>
                ${isClickable ? '<span style="opacity: 0.6; font-size: 11px;">[preview →]</span>' : ''}
            </div>
        `;
    }

    // Activations
    html += `
            <div class="file-item indent-1">
                <span class="file-icon">📁</span>
                <strong>activations/</strong>
                <span style="opacity: 0.6; font-size: 11px;">(${totalActs}/${trait.expected_activations} layer files)</span>
            </div>
    `;

    const metadataExists = trait.activations.metadata || false;
    html += `
            <div class="file-item indent-2 ${metadataExists ? 'clickable' : ''}" ${metadataExists ? `onclick="previewActivationsMetadata('${trait.trait}')"` : ''}>
                <span class="file-icon ${metadataExists ? '' : 'missing'}">${getFileIcon(metadataExists)}</span>
                <span class="${metadataExists ? '' : 'missing'}">metadata.json</span>
                ${metadataExists ? '<span style="opacity: 0.6; font-size: 11px;">[preview →]</span>' : ''}
            </div>
    `;

    // Show activation file counts
    const actTypes = [
        ['pos_layers', 'pos_layer*.pt', 'activations/'],
        ['neg_layers', 'neg_layer*.pt', 'activations/'],
        ['val_pos_layers', 'val_pos_layer*.pt', 'val_activations/'],
        ['val_neg_layers', 'val_neg_layer*.pt', 'val_activations/']
    ];
    for (const [key, pattern, dir] of actTypes) {
        const count = trait.activations[key] || 0;
        const expected = integrityData.n_layers;
        const complete = count >= expected;
        html += `
            <div class="file-item indent-2">
                <span class="file-icon ${complete ? '' : 'missing'}">${complete ? '✓' : '⚠'}</span>
                <span class="${complete ? '' : 'partial'}">${dir}${pattern}</span>
                <span style="opacity: 0.6; font-size: 11px;">(${count}/${expected} layers)</span>
            </div>
        `;
    }

    // Vectors
    const expectedVectorsPerMethod = integrityData.n_layers;
    html += `
            <div class="file-item indent-1">
                <span class="file-icon">📁</span>
                <strong>vectors/</strong>
                <span style="opacity: 0.6; font-size: 11px;">(${totalVectors} tensors + ${totalMeta} metadata)</span>
            </div>
    `;

    // Show per-method counts
    for (const method of integrityData.methods) {
        const ptCount = trait.vectors[`${method}_pt`] || 0;
        const metaCount = trait.vectors[`${method}_meta`] || 0;
        const complete = ptCount >= expectedVectorsPerMethod;
        html += `
            <div class="file-item indent-2">
                <span class="file-icon ${complete ? '' : 'missing'}">${complete ? '✓' : '⚠'}</span>
                <span class="${complete ? '' : 'partial'}">${method}_layer[0-${integrityData.n_layers - 1}].pt</span>
                <span style="opacity: 0.6; font-size: 11px;">(${ptCount}/${expectedVectorsPerMethod} + ${metaCount} meta)</span>
            </div>
        `;
    }

    // Issues
    if (trait.issues && trait.issues.length > 0) {
        html += `
            <div class="file-item indent-1" style="margin-top: 8px;">
                <span class="file-icon">⚠️</span>
                <strong>Issues (${trait.issues.length})</strong>
            </div>
        `;
        for (const issue of trait.issues.slice(0, 5)) {
            html += `
            <div class="file-item indent-2" style="color: var(--text-secondary); font-size: 11px;">
                <span>• ${issue}</span>
            </div>
            `;
        }
        if (trait.issues.length > 5) {
            html += `
            <div class="file-item indent-2" style="color: var(--text-secondary); font-size: 11px;">
                <span>... and ${trait.issues.length - 5} more</span>
            </div>
            `;
        }
    }

    html += '</div>';
    return html;
}

async function renderDataExplorer() {
    const contentArea = document.getElementById('content-area');

    // Fetch integrity data if not cached
    if (!integrityData || integrityData.experiment !== window.paths.getExperiment()) {
        integrityData = await fetchIntegrityData();
    }

    if (!integrityData) {
        contentArea.innerHTML = `
            <div class="card">
                <div class="error">Failed to load integrity data. Make sure the server is running and the experiment exists.</div>
            </div>
        `;
        return;
    }

    // Calculate summary stats
    const summary = integrityData.summary;
    const totalTraits = summary.total_traits;

    let html = `
        <div class="explanation">
            <div class="explanation-summary">Inspect all raw data files created during trait extraction—from prompts to responses to extracted vectors.</div>
            <div class="explanation-details">
                <p><strong>Prompts:</strong> Natural scenarios that elicit or avoid each trait (positive.txt, negative.txt)</p>
                <p><strong>Responses:</strong> Model outputs when presented with those prompts</p>
                <p><strong>Activations:</strong> Hidden states captured during response generation (per-layer .pt files)</p>
                <p><strong>Vectors:</strong> Extracted trait directions using 4 methods × ${integrityData.n_layers} layers</p>
            </div>
        </div>
        <div class="card">
            <div class="stats-grid">
                <div class="stat-card">
                    <span class="stat-label">Traits:</span>
                    <span class="stat-value">${totalTraits}</span>
                </div>
                <div class="stat-card">
                    <span class="stat-label">Complete:</span>
                    <span class="stat-value" style="color: var(--success);">${summary.ok}</span>
                </div>
                <div class="stat-card">
                    <span class="stat-label">Partial:</span>
                    <span class="stat-value" style="color: var(--warning);">${summary.partial}</span>
                </div>
                <div class="stat-card">
                    <span class="stat-label">Empty:</span>
                    <span class="stat-value" style="color: var(--danger);">${summary.empty}</span>
                </div>
            </div>
        </div>
        <div class="card">
            <div class="card-title">File Explorer</div>
            <div style="margin-bottom: 12px; font-size: 12px; color: var(--text-secondary);">
                Config: ${integrityData.n_layers} layers, ${integrityData.n_methods} methods (${integrityData.methods.join(', ')})
            </div>
    `;

    // Render each trait
    for (const trait of integrityData.traits.sort((a, b) => a.trait.localeCompare(b.trait))) {
        const statusIcon = getStatusIcon(trait.status);
        const displayName = trait.trait.split('/').pop().replace(/_/g, ' ');

        // Quick summary counts
        const promptsOk = Object.values(trait.prompts).filter(v => v).length;
        const responsesOk = Object.values(trait.responses).filter(v => v).length;
        const totalActs = (trait.activations.pos_layers || 0) +
                          (trait.activations.neg_layers || 0) +
                          (trait.activations.val_pos_layers || 0) +
                          (trait.activations.val_neg_layers || 0);
        const totalVectors = Object.entries(trait.vectors)
            .filter(([k, v]) => k.endsWith('_pt'))
            .reduce((sum, [k, v]) => sum + v, 0);

        html += `
            <div class="explorer-trait-card">
                <div class="explorer-trait-header" onclick="toggleTraitBody('${trait.trait.replace(/\//g, '-')}')">
                    <span>${statusIcon}</span>
                    <strong style="margin-left: 8px;">${displayName}</strong>
                    <span style="margin-left: auto; font-size: 11px; color: var(--text-tertiary);">
                        ${trait.category} | ${promptsOk}/4 prompts | ${responsesOk}/4 responses | ${totalActs}/${trait.expected_activations} acts | ${totalVectors} vectors
                    </span>
                </div>
                <div class="explorer-trait-body" id="trait-body-${trait.trait.replace(/\//g, '-')}">
                    ${renderTraitFileTree(trait, integrityData)}
                </div>
            </div>
        `;
    }

    // Inference section
    if (integrityData.inference) {
        const inf = integrityData.inference;
        html += `
            <div class="card-title" style="margin-top: 24px;">Inference Data</div>
        `;

        if (Object.keys(inf.prompt_sets).length > 0) {
            html += `
                <div class="file-item">
                    <span class="file-icon">📁</span>
                    <strong>prompts/</strong>
                    <span style="opacity: 0.6; font-size: 11px;">(${Object.keys(inf.prompt_sets).length} sets)</span>
                </div>
            `;
            for (const [name, exists] of Object.entries(inf.prompt_sets)) {
                html += `
                <div class="file-item indent-1">
                    <span class="file-icon">${exists ? '✓' : '✗'}</span>
                    <span>${name}.json</span>
                </div>
                `;
            }
        }

        if (Object.keys(inf.raw_activations).length > 0) {
            html += `
                <div class="file-item" style="margin-top: 8px;">
                    <span class="file-icon">📁</span>
                    <strong>raw/residual/</strong>
                </div>
            `;
            for (const [promptSet, count] of Object.entries(inf.raw_activations)) {
                html += `
                <div class="file-item indent-1">
                    <span class="file-icon">📁</span>
                    <span>${promptSet}/</span>
                    <span style="opacity: 0.6; font-size: 11px;">(${count} .pt files)</span>
                </div>
                `;
            }
        }

        if (inf.issues && inf.issues.length > 0) {
            html += `
                <div class="file-item" style="margin-top: 8px;">
                    <span class="file-icon">⚠️</span>
                    <strong>Issues</strong>
                </div>
            `;
            for (const issue of inf.issues) {
                html += `
                <div class="file-item indent-1" style="color: var(--text-secondary); font-size: 11px;">
                    <span>• ${issue}</span>
                </div>
                `;
            }
        }
    }

    // Evaluation status
    html += `
        <div class="card-title" style="margin-top: 24px;">Evaluation</div>
        <div class="file-item">
            <span class="file-icon">${integrityData.evaluation_exists ? '✓' : '✗'}</span>
            <span class="${integrityData.evaluation_exists ? '' : 'missing'}">extraction_evaluation.json</span>
            <span style="opacity: 0.6; font-size: 11px;">${integrityData.evaluation_exists ? '(exists)' : '(not generated)'}</span>
        </div>
    `;

    html += `
        </div>
    `;

    contentArea.innerHTML = html;
    setupExplanationToggles();
}

// Toggle trait body visibility
function toggleTraitBody(traitId) {
    const body = document.getElementById(`trait-body-${traitId}`);
    const header = body.previousElementSibling;

    if (body.classList.contains('show')) {
        body.classList.remove('show');
        header.classList.remove('expanded');
    } else {
        body.classList.add('show');
        header.classList.add('expanded');
    }
}

// Preview metadata JSON file
async function previewMetadata(traitName, filename) {
    const modal = document.getElementById('preview-modal');
    const title = document.getElementById('preview-title');
    const body = document.getElementById('preview-body');

    title.textContent = `${traitName} - ${filename}`;
    body.innerHTML = '<div class="loading">Loading...</div>';
    modal.classList.add('show');

    try {
        const url = window.paths.extractionFile(traitName, filename);
        const response = await fetch(url);
        if (!response.ok) throw new Error(`Failed to fetch: ${response.statusText}`);
        const data = await response.json();
        body.innerHTML = `<div class="json-viewer"><pre>${syntaxHighlightJSON(data)}</pre></div>`;
    } catch (error) {
        console.error('Preview error:', error);
        body.innerHTML = `<div class="error">Failed to load file: ${error.message}</div>`;
    }
}

// Preview activations metadata
async function previewActivationsMetadata(traitName) {
    const modal = document.getElementById('preview-modal');
    const title = document.getElementById('preview-title');
    const body = document.getElementById('preview-body');

    title.textContent = `${traitName} - activations/metadata.json`;
    body.innerHTML = '<div class="loading">Loading...</div>';
    modal.classList.add('show');

    try {
        const url = window.paths.activationsMetadata(traitName);
        const response = await fetch(url);
        if (!response.ok) throw new Error(`Failed to fetch: ${response.statusText}`);
        const data = await response.json();
        body.innerHTML = `<div class="json-viewer"><pre>${syntaxHighlightJSON(data)}</pre></div>`;
    } catch (error) {
        console.error('Preview error:', error);
        body.innerHTML = `<div class="error">Failed to load file: ${error.message}</div>`;
    }
}

// Preview responses JSON
async function previewResponses(traitName, polarity, isVal = false) {
    const modal = document.getElementById('preview-modal');
    const title = document.getElementById('preview-title');
    const body = document.getElementById('preview-body');

    const displayName = `${isVal ? 'Validation ' : ''}${polarity === 'pos' ? 'Positive' : 'Negative'} Responses`;
    title.textContent = `${traitName} - ${displayName}`;
    body.innerHTML = '<div class="loading">Loading...</div>';
    modal.classList.add('show');

    try {
        const dir = isVal ? 'val_responses' : 'responses';
        const file = isVal ? `val_${polarity}.json` : `${polarity}.json`;
        const url = window.paths.extractionFile(traitName, `${dir}/${file}`);

        const response = await fetch(url);
        if (!response.ok) throw new Error(`Failed to fetch: ${response.statusText}`);
        const data = await response.json();

        // Truncate for display
        const truncated = data.slice(0, 10);
        body.innerHTML = `
            <div class="json-viewer">
                <pre>${syntaxHighlightJSON(truncated)}</pre>
                <div style="margin-top: 10px; color: var(--text-secondary); font-size: 12px;">
                    Showing first 10 of ${data.length} items
                </div>
            </div>
        `;
    } catch (error) {
        console.error('Preview error:', error);
        body.innerHTML = `<div class="error">Failed to load file: ${error.message}</div>`;
    }
}

// Syntax highlight JSON
function syntaxHighlightJSON(json) {
    if (typeof json !== 'string') {
        json = JSON.stringify(json, null, 2);
    }

    json = json.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

    return json.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, function (match) {
        let cls = 'json-number';
        if (/^"/.test(match)) {
            if (/:$/.test(match)) {
                cls = 'json-key';
            } else {
                cls = 'json-string';
            }
        } else if (/true|false/.test(match)) {
            cls = 'json-boolean';
        } else if (/null/.test(match)) {
            cls = 'json-null';
        }
        return '<span class="' + cls + '">' + match + '</span>';
    });
}

// Close preview modal
function closePreview() {
    const modal = document.getElementById('preview-modal');
    modal?.classList.remove('show');
}

// Setup explanation toggles
function setupExplanationToggles() {
    document.querySelectorAll('.explanation-summary').forEach(summary => {
        summary.addEventListener('click', function() {
            const explanation = this.closest('.explanation');
            explanation?.classList.toggle('expanded');
        });
    });
}

// Close modal when clicking outside
document.addEventListener('click', (e) => {
    const modal = document.getElementById('preview-modal');
    if (e.target === modal) {
        closePreview();
    }
});

// Clear cache when experiment changes
window.addEventListener('experimentChanged', () => {
    integrityData = null;
});

// Export to global scope
window.renderDataExplorer = renderDataExplorer;
window.toggleTraitBody = toggleTraitBody;
window.previewMetadata = previewMetadata;
window.previewActivationsMetadata = previewActivationsMetadata;
window.previewResponses = previewResponses;
window.closePreview = closePreview;
