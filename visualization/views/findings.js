/**
 * Findings View - Renders research findings from docs/viz_findings/
 * Each finding is a collapsible card with preview text that expands to full markdown.
 * Metadata (title, preview) comes from YAML frontmatter in each .md file.
 */

let findingsOrder = null;  // List of filenames from index.yaml
let findingsMetadata = {};  // Cache: filename -> {title, preview}
let loadedFindings = {};  // Cache: filename -> rendered HTML

function parseFrontmatter(text) {
    // Extract YAML frontmatter from markdown
    const match = text.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
    if (!match) return { frontmatter: {}, content: text };

    try {
        const frontmatter = jsyaml.load(match[1]);
        return { frontmatter, content: match[2] };
    } catch (e) {
        console.error('Failed to parse frontmatter:', e);
        return { frontmatter: {}, content: text };
    }
}

async function loadFindingsOrder() {
    if (findingsOrder) return findingsOrder;

    try {
        const response = await fetch('/docs/viz_findings/index.yaml');
        if (!response.ok) throw new Error('Failed to load findings index');
        const yaml = await response.text();
        const parsed = jsyaml.load(yaml);
        findingsOrder = parsed.findings || [];
        return findingsOrder;
    } catch (error) {
        console.error('Error loading findings index:', error);
        return [];
    }
}

async function loadFindingMetadata(filename) {
    if (findingsMetadata[filename]) return findingsMetadata[filename];

    try {
        const response = await fetch(`/docs/viz_findings/${filename}`);
        if (!response.ok) throw new Error(`Failed to load ${filename}`);
        const text = await response.text();
        const { frontmatter } = parseFrontmatter(text);

        findingsMetadata[filename] = {
            title: frontmatter.title || filename.replace('.md', ''),
            preview: frontmatter.preview || ''
        };
        return findingsMetadata[filename];
    } catch (error) {
        console.error(`Error loading metadata for ${filename}:`, error);
        return { title: filename, preview: '' };
    }
}

async function loadFindingContent(filename) {
    if (loadedFindings[filename]) return loadedFindings[filename];

    try {
        const response = await fetch(`/docs/viz_findings/${filename}`);
        if (!response.ok) throw new Error(`Failed to load ${filename}`);

        const text = await response.text();
        const { frontmatter, content } = parseFrontmatter(text);
        const references = frontmatter.references || {};

        // Protect math blocks from markdown parser
        let { markdown, blocks: mathBlocks } = window.protectMathBlocks(content);

        // Extract :::responses path::: blocks before markdown parsing
        const responseBlocks = [];
        markdown = markdown.replace(/:::responses\s+([^\s:]+)(?:\s+"([^"]*)")?\s*:::/g, (match, path, label) => {
            responseBlocks.push({ path, label: label || 'View responses' });
            return `RESPONSE_BLOCK_${responseBlocks.length - 1}`;
        });

        // Extract :::dataset path "label"::: blocks
        const datasetBlocks = [];
        markdown = markdown.replace(/:::dataset\s+([^\s:]+)(?:\s+"([^"]*)")?\s*:::/g, (match, path, label) => {
            datasetBlocks.push({ path, label: label || 'View examples' });
            return `DATASET_BLOCK_${datasetBlocks.length - 1}`;
        });

        // Extract :::prompts path "label"::: blocks (for prompt set JSON files)
        const promptBlocks = [];
        markdown = markdown.replace(/:::prompts\s+([^\s:]+)(?:\s+"([^"]*)")?\s*:::/g, (match, path, label) => {
            promptBlocks.push({ path, label: label || 'View prompts' });
            return `PROMPT_BLOCK_${promptBlocks.length - 1}`;
        });

        // Extract :::figure path "caption" size::: blocks (size is optional: small, medium, large)
        const figureBlocks = [];
        markdown = markdown.replace(/:::figure\s+([^\s:]+)(?:\s+"([^"]*)")?(?:\s+(small|medium|large))?\s*:::/g, (match, path, caption, size) => {
            figureBlocks.push({ path, caption: caption || '', size: size || '' });
            return `FIGURE_BLOCK_${figureBlocks.length - 1}`;
        });

        // Extract :::example ... ::: blocks (multiline content with optional caption)
        const exampleBlocks = [];
        markdown = markdown.replace(/:::example\s*\n([\s\S]*?)\n:::\s*\n?\*([^*]+)\*/g, (match, content, caption) => {
            exampleBlocks.push({ content: content.trim(), caption: caption.trim() });
            return `EXAMPLE_BLOCK_${exampleBlocks.length - 1}`;
        });
        // Also match without caption
        markdown = markdown.replace(/:::example\s*\n([\s\S]*?)\n:::/g, (match, content) => {
            exampleBlocks.push({ content: content.trim(), caption: '' });
            return `EXAMPLE_BLOCK_${exampleBlocks.length - 1}`;
        });

        // Extract [@key] citations
        const citedKeys = [];
        markdown = markdown.replace(/\[@(\w+)\]/g, (match, key) => {
            if (!citedKeys.includes(key)) citedKeys.push(key);
            return `CITE_${key}`;
        });

        // Fix relative image paths to absolute
        markdown = markdown.replace(/!\[([^\]]*)\]\(assets\//g, '![$1](/docs/viz_findings/assets/');

        // Render markdown
        marked.setOptions({ gfm: true, breaks: false, headerIds: true });
        let html = marked.parse(markdown);

        // Restore math blocks
        html = window.restoreMathBlocks(html, mathBlocks);

        // Replace response block placeholders with dropdown components
        responseBlocks.forEach((block, i) => {
            const dropdownId = `responses-${filename}-${i}`;
            const dropdownHtml = `
                <div class="responses-dropdown" id="${dropdownId}">
                    <div class="responses-header" onclick="toggleResponses('${dropdownId}', '${block.path}')">
                        <span class="responses-toggle">+</span>
                        <span class="responses-label">${block.label}</span>
                    </div>
                    <div class="responses-content"></div>
                </div>
            `;
            html = html.replace(`<p>RESPONSE_BLOCK_${i}</p>`, dropdownHtml);
            html = html.replace(`RESPONSE_BLOCK_${i}`, dropdownHtml);
        });

        // Replace dataset block placeholders with dropdown components
        datasetBlocks.forEach((block, i) => {
            const dropdownId = `dataset-${filename}-${i}`;
            const dropdownHtml = `
                <div class="responses-dropdown" id="${dropdownId}">
                    <div class="responses-header" onclick="toggleDataset('${dropdownId}', '${block.path}')">
                        <span class="responses-toggle">+</span>
                        <span class="responses-label">${block.label}</span>
                    </div>
                    <div class="responses-content"></div>
                </div>
            `;
            html = html.replace(`<p>DATASET_BLOCK_${i}</p>`, dropdownHtml);
            html = html.replace(`DATASET_BLOCK_${i}`, dropdownHtml);
        });

        // Replace prompt block placeholders with dropdown components
        promptBlocks.forEach((block, i) => {
            const dropdownId = `prompts-${filename}-${i}`;
            const dropdownHtml = `
                <div class="responses-dropdown" id="${dropdownId}">
                    <div class="responses-header" onclick="togglePrompts('${dropdownId}', '${block.path}')">
                        <span class="responses-toggle">+</span>
                        <span class="responses-label">${block.label}</span>
                    </div>
                    <div class="responses-content"></div>
                </div>
            `;
            html = html.replace(`<p>PROMPT_BLOCK_${i}</p>`, dropdownHtml);
            html = html.replace(`PROMPT_BLOCK_${i}`, dropdownHtml);
        });

        // Replace figure block placeholders with figure elements
        figureBlocks.forEach((block, i) => {
            // Fix relative paths
            const imgPath = block.path.startsWith('assets/') ? `/docs/viz_findings/${block.path}` : block.path;
            const sizeClass = block.size ? ` fig-${block.size}` : '';
            const figureHtml = `
                <figure class="fig${sizeClass}">
                    <img src="${imgPath}" alt="${block.caption}">
                    ${block.caption ? `<figcaption>${block.caption}</figcaption>` : ''}
                </figure>
            `;
            html = html.replace(`<p>FIGURE_BLOCK_${i}</p>`, figureHtml);
            html = html.replace(`FIGURE_BLOCK_${i}`, figureHtml);
        });

        // Replace example block placeholders with styled example boxes
        exampleBlocks.forEach((block, i) => {
            // Parse content - render markdown within the example
            const innerHtml = marked.parse(block.content);
            const exampleHtml = `
                <figure class="example-box">
                    <div class="example-content">${innerHtml}</div>
                    ${block.caption ? `<figcaption>${block.caption}</figcaption>` : ''}
                </figure>
            `;
            html = html.replace(`<p>EXAMPLE_BLOCK_${i}</p>`, exampleHtml);
            html = html.replace(`EXAMPLE_BLOCK_${i}`, exampleHtml);
        });

        // Replace citation placeholders with formatted citations
        for (const key of citedKeys) {
            const ref = references[key];
            if (ref) {
                const tooltipText = `${ref.title}`;
                const citeHtml = ref.url
                    ? `<a href="${ref.url}" class="citation" target="_blank" data-tooltip="${tooltipText}">(${ref.authors}, ${ref.year})</a>`
                    : `<span class="citation" data-tooltip="${tooltipText}">(${ref.authors}, ${ref.year})</span>`;
                html = html.replaceAll(`CITE_${key}`, citeHtml);
            } else {
                console.warn(`Citation [@${key}] not found in references`);
                html = html.replaceAll(`CITE_${key}`, `<span class="citation citation-missing">[@${key}]</span>`);
            }
        }

        // Append References section if any citations used
        if (citedKeys.length > 0) {
            const validRefs = citedKeys.filter(key => references[key]);
            if (validRefs.length > 0) {
                let refsHtml = '<section class="references"><h2>References</h2><ol>';
                for (const key of validRefs) {
                    const ref = references[key];
                    const link = ref.url ? `<a href="${ref.url}" target="_blank">${ref.url}</a>` : '';
                    refsHtml += `<li id="ref-${key}">${ref.authors} (${ref.year}). "${ref.title}". ${link}</li>`;
                }
                refsHtml += '</ol></section>';
                html += refsHtml;
            }
        }

        loadedFindings[filename] = html;
        return html;
    } catch (error) {
        console.error(`Error loading finding ${filename}:`, error);
        return `<div class="error">Failed to load ${filename}</div>`;
    }
}

async function toggleResponses(dropdownId, path) {
    const dropdown = document.getElementById(dropdownId);
    const content = dropdown.querySelector('.responses-content');
    const toggle = dropdown.querySelector('.responses-toggle');

    if (dropdown.classList.contains('expanded')) {
        dropdown.classList.remove('expanded');
        content.style.display = 'none';
        toggle.textContent = '+';
    } else {
        // Load content if not already loaded
        if (!content.innerHTML) {
            content.innerHTML = '<div class="loading">Loading responses...</div>';
            try {
                const response = await fetch(path);
                if (!response.ok) throw new Error('Failed to load');
                const data = await response.json();
                content.innerHTML = renderResponsesTable(data);
            } catch (error) {
                content.innerHTML = `<div class="error">Failed to load responses</div>`;
            }
        }
        dropdown.classList.add('expanded');
        content.style.display = 'block';
        toggle.textContent = '−';
    }
}

function renderResponsesTable(responses) {
    if (!Array.isArray(responses) || responses.length === 0) {
        return '<div class="error">No responses found</div>';
    }

    let html = '<table class="responses-table"><thead><tr>';
    html += '<th>Question</th><th>Response</th><th>Trait</th><th>Coh</th>';
    html += '</tr></thead><tbody>';

    for (const r of responses) {
        const question = escapeHtml(r.question || '');
        const response = escapeHtml(r.response || '').replace(/\n/g, '<br>');
        const trait = r.trait_score?.toFixed(0) ?? '-';
        const coh = r.coherence_score?.toFixed(0) ?? '-';
        html += `<tr>
            <td class="responses-question">${question}</td>
            <td class="responses-response">${response}</td>
            <td class="responses-score">${trait}</td>
            <td class="responses-score">${coh}</td>
        </tr>`;
    }

    html += '</tbody></table>';
    return html;
}

// escapeHtml is in core/utils.js

window.toggleResponses = toggleResponses;

async function toggleDataset(dropdownId, path) {
    const dropdown = document.getElementById(dropdownId);
    const content = dropdown.querySelector('.responses-content');
    const toggle = dropdown.querySelector('.responses-toggle');

    if (dropdown.classList.contains('expanded')) {
        dropdown.classList.remove('expanded');
        content.style.display = 'none';
        toggle.textContent = '+';
    } else {
        // Load content if not already loaded
        if (!content.innerHTML) {
            content.innerHTML = '<div class="loading">Loading examples...</div>';
            try {
                const response = await fetch(path);
                if (!response.ok) throw new Error('Failed to load');
                const text = await response.text();
                content.innerHTML = renderDatasetList(text);
            } catch (error) {
                content.innerHTML = `<div class="error">Failed to load dataset</div>`;
            }
        }
        dropdown.classList.add('expanded');
        content.style.display = 'block';
        toggle.textContent = '−';
    }
}

function renderDatasetList(text) {
    const lines = text.trim().split('\n').filter(line => line.trim());
    if (lines.length === 0) {
        return '<div class="error">No examples found</div>';
    }

    // Show first 20 examples
    const examples = lines.slice(0, 20);
    let html = '<ul class="dataset-list">';
    for (const line of examples) {
        html += `<li>${escapeHtml(line)}</li>`;
    }
    html += '</ul>';
    if (lines.length > 20) {
        html += `<div class="dataset-more">...and ${lines.length - 20} more</div>`;
    }
    return html;
}

window.toggleDataset = toggleDataset;

async function togglePrompts(dropdownId, path) {
    const dropdown = document.getElementById(dropdownId);
    const content = dropdown.querySelector('.responses-content');
    const toggle = dropdown.querySelector('.responses-toggle');

    if (dropdown.classList.contains('expanded')) {
        dropdown.classList.remove('expanded');
        content.style.display = 'none';
        toggle.textContent = '+';
    } else {
        if (!content.innerHTML) {
            content.innerHTML = '<div class="loading">Loading prompts...</div>';
            try {
                const response = await fetch(path);
                if (!response.ok) throw new Error('Failed to load');
                const data = await response.json();
                content.innerHTML = renderPromptsTable(data);
            } catch (error) {
                content.innerHTML = `<div class="error">Failed to load prompts</div>`;
            }
        }
        dropdown.classList.add('expanded');
        content.style.display = 'block';
        toggle.textContent = '−';
    }
}

function renderPromptsTable(data) {
    const prompts = data.prompts || [];
    if (prompts.length === 0) {
        return '<div class="error">No prompts found</div>';
    }

    let html = '<table class="responses-table"><thead><tr>';
    html += '<th>Prompt</th><th>Bias</th>';
    html += '</tr></thead><tbody>';

    for (const p of prompts.slice(0, 20)) {
        const text = escapeHtml(p.text || '');
        const bias = p.bias_id ? `#${p.bias_id}` : '-';
        html += `<tr>
            <td class="responses-question">${text}</td>
            <td class="responses-score">${bias}</td>
        </tr>`;
    }

    html += '</tbody></table>';
    if (prompts.length > 20) {
        html += `<div class="dataset-more">...and ${prompts.length - 20} more</div>`;
    }
    return html;
}

window.togglePrompts = togglePrompts;

async function toggleFinding(filename, cardEl) {
    const contentEl = cardEl.querySelector('.finding-content');
    const toggleEl = cardEl.querySelector('.finding-toggle');

    if (cardEl.classList.contains('expanded')) {
        // Collapse
        cardEl.classList.remove('expanded');
        contentEl.style.display = 'none';
        toggleEl.textContent = '+';
    } else {
        // Expand - load content if needed
        if (!contentEl.innerHTML) {
            contentEl.innerHTML = '<div class="loading">Loading...</div>';
            const html = await loadFindingContent(filename);
            contentEl.innerHTML = `<div class="prose">${html}</div>`;

            // Render math
            if (window.renderMath) {
                window.renderMath(contentEl);
            }
        }
        cardEl.classList.add('expanded');
        contentEl.style.display = 'block';
        toggleEl.textContent = '−';
    }
}

async function renderFindings() {
    const contentArea = document.getElementById('content-area');
    contentArea.innerHTML = '<div class="loading">Loading findings...</div>';

    const filenames = await loadFindingsOrder();
    if (!filenames || filenames.length === 0) {
        contentArea.innerHTML = '<div class="error">Failed to load findings index</div>';
        return;
    }

    // Load metadata from all files in parallel
    const metadataList = await Promise.all(filenames.map(f => loadFindingMetadata(f)));

    let html = `
        <div class="findings-container">
            <div class="findings-header">
                <p class="findings-intro">Research findings from trait vector experiments. Click to expand.</p>
            </div>
            <div class="findings-list">
    `;

    filenames.forEach((filename, i) => {
        const meta = metadataList[i];
        const isTodo = !meta.preview || meta.preview === 'TODO';
        const todoClass = isTodo ? 'finding-todo' : '';

        html += `
            <div class="finding-card ${todoClass}" id="finding-${i}">
                <div class="finding-header" onclick="toggleFinding('${filename}', document.getElementById('finding-${i}'))">
                    <div class="finding-title-row">
                        <span class="finding-toggle">+</span>
                        <span class="finding-title">${meta.title}</span>
                    </div>
                    <p class="finding-preview">${meta.preview || 'TODO'}</p>
                </div>
                <div class="finding-content" style="display: none;"></div>
            </div>
        `;
    });

    html += `
            </div>
        </div>
    `;

    contentArea.innerHTML = html;
}

window.renderFindings = renderFindings;
window.toggleFinding = toggleFinding;
