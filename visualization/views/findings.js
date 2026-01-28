/**
 * Findings View - Renders research findings from docs/viz_findings/
 * Each finding is a collapsible card with preview text that expands to full markdown.
 * Metadata (title, preview) comes from YAML frontmatter in each .md file.
 */

let findingsOrder = null;  // List of filenames from index.yaml
let findingsMetadata = {};  // Cache: filename -> {title, preview}
let loadedFindings = {};  // Cache: filename -> rendered HTML

function parseFrontmatter(text) {
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

/**
 * Render a thumbnail bar chart for finding cards
 * @param {Object} thumbnail - { title, bars: [{label, value}] }
 */
function renderThumbnailChart(thumbnail) {
    if (!thumbnail?.bars?.length) return '';

    const maxValue = Math.max(...thumbnail.bars.map(b => b.value));
    const barsHtml = thumbnail.bars.map(bar => {
        const heightPct = (bar.value / maxValue) * 100;
        return `
            <div class="thumb-bar-wrapper">
                <div class="thumb-bar-area">
                    <div class="thumb-bar" style="height: ${heightPct}%"></div>
                </div>
                <span class="thumb-value">${bar.value}</span>
                <span class="thumb-label">${bar.label}</span>
            </div>
        `;
    }).join('');

    return `
        <div class="thumbnail-chart">
            <div class="thumb-title">${thumbnail.title || ''}</div>
            <div class="thumb-bars">${barsHtml}</div>
        </div>
    `;
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
            preview: frontmatter.preview || '',
            thumbnail: frontmatter.thumbnail || null
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

        // Extract custom blocks (:::responses, :::dataset, etc.)
        const { markdown: processedMarkdown, blocks } = window.customBlocks.extractCustomBlocks(markdown);
        markdown = processedMarkdown;

        // Extract numbered references (## References section) and process ^N citations
        let numberedRefs = {};
        if (window.citations) {
            const extracted = window.citations.extractReferences(markdown);
            markdown = extracted.markdown;
            numberedRefs = extracted.refs;
            markdown = window.citations.processCitationMarkers(markdown, numberedRefs);
        }

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

        // Render custom blocks
        html = window.customBlocks.renderCustomBlocks(html, blocks, filename);

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

        // Append References section if any [@key] citations used
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

        // Render numbered ^N citations and append references section
        if (window.citations && Object.keys(numberedRefs).length > 0) {
            html = window.citations.renderCitations(html, numberedRefs);
            html += window.citations.renderReferencesSection(numberedRefs);
        }

        loadedFindings[filename] = html;
        return html;
    } catch (error) {
        console.error(`Error loading finding ${filename}:`, error);
        return `<div class="error">Failed to load ${filename}</div>`;
    }
}

async function toggleFinding(filename, cardEl) {
    const contentEl = cardEl.querySelector('.finding-content');
    const toggleEl = cardEl.querySelector('.finding-toggle');
    const findingId = filename.replace('.md', '');

    if (cardEl.classList.contains('expanded')) {
        // Collapse
        cardEl.classList.remove('expanded');
        contentEl.style.display = 'none';
        toggleEl.textContent = '▶';

        // Remove hash if this finding is in URL
        if (window.location.hash === `#${findingId}`) {
            history.replaceState(null, '', window.location.pathname + window.location.search);
        }
    } else {
        // Expand - make visible FIRST so Plotly can measure dimensions
        cardEl.classList.add('expanded');
        contentEl.style.display = 'block';
        toggleEl.textContent = '▼';

        // Then load content if not yet loaded
        if (!contentEl.innerHTML || contentEl.innerHTML === ui.renderLoading()) {
            contentEl.innerHTML = ui.renderLoading();
            const html = await loadFindingContent(filename);
            contentEl.innerHTML = `<div class="prose">${html}</div>`;

            if (window.renderMath) {
                window.renderMath(contentEl);
            }

            // Auto-load any dropdowns marked as expanded
            if (window.customBlocks?.loadExpandedDropdowns) {
                await window.customBlocks.loadExpandedDropdowns();
            }
            // Load any chart blocks (container is now visible, Plotly can measure)
            if (window.customBlocks?.loadCharts) {
                await window.customBlocks.loadCharts();
            }
            // Initialize citation click handlers
            if (window.citations?.initCitationClicks) {
                window.citations.initCitationClicks(contentEl);
            }
        }

        // Update hash to current finding
        history.replaceState(null, '', `${window.location.pathname}${window.location.search}#${findingId}`);
    }
}

async function renderFindings() {
    const contentArea = document.getElementById('content-area');
    const hash = window.location.hash.slice(1);

    // Check if we're in standalone mode
    if (window.state.currentView === 'finding' && hash) {
        return renderStandaloneFinding(hash);
    }

    contentArea.innerHTML = ui.renderLoading('Loading findings...');

    const filenames = await loadFindingsOrder();
    if (!filenames || filenames.length === 0) {
        contentArea.innerHTML = '<div class="error">Failed to load findings index</div>';
        return;
    }

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
        const findingId = filename.replace('.md', '');

        const thumbnailHtml = meta.thumbnail ? renderThumbnailChart(meta.thumbnail) : '';
        const hasThumbnail = meta.thumbnail ? ' has-thumbnail' : '';

        html += `
            <div class="finding-card ${todoClass}${hasThumbnail}" id="finding-${findingId}">
                <div class="finding-header" onclick="toggleFinding('${filename}', document.getElementById('finding-${findingId}'))">
                    <div class="finding-header-content">
                        <div class="finding-title-row">
                            <span class="finding-toggle">▶</span>
                            <span class="finding-title">${meta.title}</span>
                        </div>
                        <p class="finding-preview">${meta.preview || 'TODO'}</p>
                    </div>
                    ${thumbnailHtml}
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

    // Auto-expand finding if hash present
    if (hash) {
        const cardEl = document.getElementById(`finding-${hash}`);
        if (cardEl && !cardEl.classList.contains('expanded')) {
            await toggleFinding(`${hash}.md`, cardEl);
            cardEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }
}

/**
 * Render a single finding in standalone mode (content-only view)
 */
async function renderStandaloneFinding(findingId) {
    const contentArea = document.getElementById('content-area');
    const filename = `${findingId}.md`;

    // Validate finding exists
    const order = await loadFindingsOrder();
    if (!order.includes(filename)) {
        contentArea.innerHTML = `
            <div class="tool-view">
                <div class="no-data">Finding not found: ${findingId}</div>
            </div>
        `;
        return;
    }

    // Load content (reuses existing function)
    const html = await loadFindingContent(filename);

    contentArea.innerHTML = `
        <div class="standalone-finding">
            <button class="back-button" onclick="window.backToFindings()">← Back to findings</button>
            <div class="finding-prose">${html}</div>
        </div>
    `;

    // Apply math and custom block rendering (same as list mode)
    if (window.renderMath) {
        window.renderMath(contentArea);
    }
    if (window.customBlocks?.loadExpandedDropdowns) {
        await window.customBlocks.loadExpandedDropdowns();
    }
    // Load any chart blocks
    if (window.customBlocks?.loadCharts) {
        await window.customBlocks.loadCharts();
    }
    // Initialize citation click handlers
    if (window.citations?.initCitationClicks) {
        window.citations.initCitationClicks(contentArea);
    }
}

// Back button handler
window.backToFindings = () => {
    window.state.currentView = 'findings';
    setTabInURL('findings');
    window.renderView();
};

window.renderFindings = renderFindings;
window.toggleFinding = toggleFinding;

// Auto-expand finding when hash changes (browser back/forward)
window.addEventListener('hashchange', () => {
    if (window.state.currentView === 'findings') {
        const hash = window.location.hash.slice(1);
        if (hash) {
            const cardEl = document.getElementById(`finding-${hash}`);
            if (cardEl && !cardEl.classList.contains('expanded')) {
                toggleFinding(`${hash}.md`, cardEl);
                cardEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        }
    }
});
