// Data Explorer View - File browser and preview for experiment data

// Helper to get relative path for display (strips experiment prefix for cleaner UI)
function getDisplayPath(fullPath) {
    const experimentBase = window.paths.get('experiments.base', {});
    if (fullPath.startsWith(experimentBase + '/')) {
        return fullPath.substring(experimentBase.length + 1);
    }
    return fullPath;
}

async function renderDataExplorer() {
    const contentArea = document.getElementById('content-area');
    const filteredTraits = window.getFilteredTraits();

    // Calculate totals
    const totalTraits = window.state.experimentData.traits.length;
    const selectedCount = filteredTraits.length;
    const estimatedSize = (totalTraits * 47).toFixed(0); // ~47 MB per trait
    const filesPerTrait = 223; // Updated estimate for full structure

    let html = `
        <div class="explanation">
            <div class="explanation-summary">We can inspect all the raw data files created during trait extraction—from the model's generated responses to the vectors extracted from them.</div>
            <div class="explanation-details">
            <p><strong>Responses:</strong> What the model said when showing the trait vs. not showing it</p>

            <p><strong>Activations:</strong> Internal neuron values captured while generating those responses</p>

            <p><strong>Vectors:</strong> Mathematical directions extracted from the activations that represent the trait</p>
            </div>
        <div class="card">
            <div class="stats-grid">
                <div class="stat-card">
                    <span class="stat-label">Traits:</span>
                    <span class="stat-value">${totalTraits}</span>
                </div>
                <div class="stat-card">
                    <span class="stat-label">Storage:</span>
                    <span class="stat-value">~${estimatedSize}MB</span>
                </div>
                <div class="stat-card">
                    <span class="stat-label">Files:</span>
                    <span class="stat-value">${totalTraits * filesPerTrait}</span>
                </div>
            </div>
        <div class="card">
            <div class="card-title">File Explorer</div>
    `;

    // Render each trait
    filteredTraits.forEach(trait => {
        const displayName = window.getDisplayName(trait.name);
        const metadata = trait.metadata || {};

        const isNatural = trait.name.includes('_natural');
        const fileExt = isNatural ? 'json' : 'csv';
        const previewFunc = isNatural ? 'previewJSON' : 'previewCSV';

        // Handle both metadata formats: n_examples OR (n_positive + n_negative)
        const nExamples = metadata.n_examples || ((metadata.n_positive || 0) + (metadata.n_negative || 0)) || 0;
        const nPos = metadata.n_examples_pos || metadata.n_positive || '?';
        const nNeg = metadata.n_examples_neg || metadata.n_negative || '?';

        // Get paths from PathBuilder
        const traitPath = getDisplayPath(window.paths.get('extraction.trait', { trait: trait.name }));

        html += `
            <div class="explorer-trait-card">
                <div class="explorer-trait-header" onclick="toggleTraitBody('${trait.name}')">
                    <strong>${displayName}</strong>
                    <span style="margin-left: 8px; font-size: 10px; color: var(--text-tertiary);">
                        ${nExamples} examples | ${filesPerTrait} files | ${metadata.size_mb ? metadata.size_mb.toFixed(1) + ' MB' : '~20 MB'}
                    </span>
                </div>
                <div class="explorer-trait-body" id="trait-body-${trait.name}">
                    <div class="file-tree">
                        <div class="file-item">
                            <span class="file-icon">📁</span>
                            <strong>${traitPath}/</strong>
                        </div>

                        <div class="file-item indent-1 clickable" onclick="previewJSON('${trait.name}', 'trait_definition')">
                            <span class="file-icon">✓</span>
                            <span>trait_definition.json</span>
                            <span style="opacity: 0.6; font-size: 11px;">(~15 KB) [preview →]</span>
                        </div>

                        <div class="file-item indent-1">
                            <span class="file-icon">📁</span>
                            <strong>responses/</strong>
                        </div>
                        <div class="file-item indent-2 clickable" onclick="${previewFunc}('${trait.name}', 'pos')">
                            <span class="file-icon">✓</span>
                            <span>pos.${fileExt}</span>
                            <span style="opacity: 0.6; font-size: 11px;">(${nPos} ${isNatural ? 'items' : 'rows'}) [preview →]</span>
                        </div>
                        <div class="file-item indent-2 clickable" onclick="${previewFunc}('${trait.name}', 'neg')">
                            <span class="file-icon">✓</span>
                            <span>neg.${fileExt}</span>
                            <span style="opacity: 0.6; font-size: 11px;">(${nNeg} ${isNatural ? 'items' : 'rows'}) [preview →]</span>
                        </div>

                        <div class="file-item indent-1">
                            <span class="file-icon">📁</span>
                            <strong>activations/</strong>
                        </div>
                        <div class="file-item indent-2 clickable" onclick="previewJSON('${trait.name}', 'activations_metadata')">
                            <span class="file-icon">✓</span>
                            <span>metadata.json</span>
                            <span style="opacity: 0.6; font-size: 11px;">(~400 B) [preview →]</span>
                        </div>
                        <div class="file-item indent-2">
                            <span class="file-icon">✓</span>
                            <span>all_layers.pt</span>
                            <span style="opacity: 0.6; font-size: 11px;">(~19 MB, shape: [${nExamples}, 27, 2304])</span>
                        </div>
                        <div class="file-item indent-2">
                            <span class="file-icon">✓</span>
                            <span>pos_acts.pt</span>
                            <span style="opacity: 0.6; font-size: 11px;">(~12 MB, shape: [${nPos}, 27, 2304])</span>
                        </div>
                        <div class="file-item indent-2">
                            <span class="file-icon">✓</span>
                            <span>neg_acts.pt</span>
                            <span style="opacity: 0.6; font-size: 11px;">(~11 MB, shape: [${nNeg}, 27, 2304])</span>
                        </div>

                        <div class="file-item indent-1">
                            <span class="file-icon">📁</span>
                            <strong>vectors/</strong>
                            <span style="opacity: 0.6; font-size: 11px;">(216 files: 108 tensors + 108 metadata)</span>
                        </div>
                        <div class="file-item indent-2">
                            <span class="file-icon">📊</span>
                            <span>4 methods × 27 layers:</span>
                        </div>
                        <div class="file-item indent-3">
                            <span class="file-icon">✓</span>
                            <span>mean_diff_layer[0-26].pt + metadata</span>
                            <span style="opacity: 0.6; font-size: 11px;">(~9 KB each)</span>
                        </div>
                        <div class="file-item indent-3">
                            <span class="file-icon">✓</span>
                            <span>probe_layer[0-26].pt + metadata</span>
                            <span style="opacity: 0.6; font-size: 11px;">(~20 KB each)</span>
                        </div>
                        <div class="file-item indent-3">
                            <span class="file-icon">✓</span>
                            <span>ica_layer[0-26].pt + metadata</span>
                            <span style="opacity: 0.6; font-size: 11px;">(~186 KB each)</span>
                        </div>
                        <div class="file-item indent-3">
                            <span class="file-icon">✓</span>
                            <span>gradient_layer[0-26].pt + metadata</span>
                            <span style="opacity: 0.6; font-size: 11px;">(~9 KB each, some NaN)</span>
                        </div>

                        <div class="file-item" style="margin-top: 10px;">
                            <span class="file-icon">📁</span>
                            <strong>inference/</strong>
                            <span style="opacity: 0.6; font-size: 11px;">(if generated)</span>
                        </div>
                        <div class="file-item indent-1">
                            <span class="file-icon">📁</span>
                            <strong>residual_stream_activations/</strong>
                        </div>
                        <div class="file-item indent-2">
                            <span class="file-icon">📄</span>
                            <span>prompt_*.json</span>
                            <span style="opacity: 0.6; font-size: 11px;">(per-token projections)</span>
                        </div>
                        <div class="file-item indent-1">
                            <span class="file-icon">📁</span>
                            <strong>layer_internal_states/</strong>
                        </div>
                        <div class="file-item indent-2">
                            <span class="file-icon">📄</span>
                            <span>prompt_*_layer*.json</span>
                            <span style="opacity: 0.6; font-size: 11px;">(layer internals)</span>
                        </div>
                    </div>
                </div>
            </div>
        `;
    });

    html += `
            </div>
        </div>
    `;

    contentArea.innerHTML = html;
    renderMath();
}

// Toggle trait body visibility
function toggleTraitBody(traitName) {
    const body = document.getElementById(`trait-body-${traitName}`);
    const header = body.previousElementSibling;

    if (body.classList.contains('show')) {
        body.classList.remove('show');
        header.classList.remove('expanded');
    } else {
        body.classList.add('show');
        header.classList.add('expanded');
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

// Preview JSON file
async function previewJSON(traitName, type) {
    const modal = document.getElementById('preview-modal');
    const title = document.getElementById('preview-title');
    const body = document.getElementById('preview-body');

    let displayName;
    if (type === 'trait_definition') {
        displayName = 'Trait Definition';
    } else if (type === 'activations_metadata') {
        displayName = 'Activations Metadata';
    } else if (type === 'pos') {
        displayName = 'Positive Examples';
    } else if (type === 'neg') {
        displayName = 'Negative Examples';
    }

    title.textContent = `${traitName} - ${displayName}`;
    body.innerHTML = '<div class="loading">Loading...</div>';
    modal.classList.add('show');

    try {
        // Build URL using PathBuilder
        let url;
        if (type === 'trait_definition') {
            url = window.paths.traitDefinition(traitName);
        } else if (type === 'activations_metadata') {
            url = window.paths.activationsMetadata(traitName);
        } else if (type === 'pos' || type === 'neg') {
            url = window.paths.responses(traitName, type, 'json');
        } else {
            throw new Error(`Unknown JSON type: ${type}`);
        }

        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to fetch JSON for ${traitName} ${type}: ${response.statusText}`);
        }
        const data = await response.json();
        const highlighted = syntaxHighlightJSON(data);
        body.innerHTML = `<div class="json-viewer"><pre>${highlighted}</pre></div>`;
    } catch (error) {
        console.error('Preview JSON error:', error);
        body.innerHTML = `<div class="error">Failed to load JSON file. Check console for details.</div>`;
    }
}

// Preview CSV file
async function previewCSV(traitName, category) {
    const modal = document.getElementById('preview-modal');
    const title = document.getElementById('preview-title');
    const body = document.getElementById('preview-body');

    const trait = { name: traitName };
    const displayName = category === 'pos' ? 'Positive Examples' : 'Negative Examples';

    title.textContent = `${traitName} - ${displayName}`;
    body.innerHTML = '<div class="loading">Loading first 10 rows...</div>';
    modal.classList.add('show');

    try {
        const result = await window.DataLoader.fetchCSV(trait, category, 10);
        const { data: rows, total, headers } = result;

        if (rows.length === 0) {
            body.innerHTML = '<div class="error">No data found</div>';
            return;
        }

        let tableHTML = '<table class="csv-table"><thead><tr>';
        headers.forEach(h => {
            tableHTML += `<th>${h}</th>`;
        });
        tableHTML += '</tr></thead><tbody>';

        rows.forEach(row => {
            tableHTML += '<tr>';
            headers.forEach(h => {
                let value = row[h] || '';
                // Truncate long values
                if (value.length > 100) {
                    value = value.substring(0, 100) + '...';
                }
                tableHTML += `<td>${value}</td>`;
            });
            tableHTML += '</tr>';
        });

        tableHTML += '</tbody></table>';
        tableHTML += `<div style="margin-top: 10px; color: var(--text-secondary); font-size: 12px;">Showing first 10 rows of ${total} total</div>`;

        body.innerHTML = tableHTML;
    } catch (error) {
        body.innerHTML = '<div class="error">Failed to load CSV file</div>';
    }
}

// Close preview modal
function closePreview() {
    const modal = document.getElementById('preview-modal');
    modal?.classList.remove('show');
}

// Render math with MathJax
function renderMath() {
    if (window.MathJax) {
        MathJax.typesetPromise().catch((err) => console.log('MathJax rendering error:', err));
    }
    // Setup explanation toggles after content is rendered
    setupExplanationToggles();
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

// Export to global scope
window.renderDataExplorer = renderDataExplorer;
window.toggleTraitBody = toggleTraitBody;
window.previewJSON = previewJSON;
window.previewCSV = previewCSV;
window.closePreview = closePreview;
