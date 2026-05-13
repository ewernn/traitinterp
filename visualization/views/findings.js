/**
 * Findings View - Renders research findings from docs/viz_findings/
 * Each finding is a collapsible card with preview text that expands to full markdown.
 * Metadata (title, preview) comes from YAML frontmatter in each .md file.
 */

import { renderMarkdownContent } from '../core/markdown-view.js';
import { parseFrontmatter, renderMath } from '../core/utils.js';
import { setTabInURL } from '../core/state.js';
import { renderLoading } from '../core/ui.js';
import { toggleExpandCollapse } from '../components/custom-blocks/index.js';

let findingsOrder = null;  // List of filenames from index.yaml
let findingsMetadata = {};  // Cache: filename -> {title, preview}
let loadedFindings = {};  // Cache: filename -> rendered HTML

/**
 * Render a thumbnail chart (bar or line) for finding cards.
 *
 * Frontmatter shapes (pick one):
 *   thumbnail:
 *     title: "..."
 *     bars: [{label, value}, ...]
 *
 *   thumbnail:
 *     title: "..."
 *     line:
 *       x: [...]            # optional, defaults to 0..N-1
 *       y: [...]
 *       floor: 70           # optional dashed reference line
 *       y_min: 0            # optional axis min (default: min(y))
 *       y_max: 100          # optional axis max (default: max(y))
 */
function renderThumbnailChart(thumbnail) {
    if (!thumbnail) return '';
    if (thumbnail.line) return renderThumbnailLine(thumbnail);
    if (thumbnail.bars?.length) return renderThumbnailBars(thumbnail);
    return '';
}

function renderThumbnailBars(thumbnail) {
    const parsedBars = thumbnail.bars.map(bar => {
        const valueStr = String(bar.value);
        const isPercent = valueStr.endsWith('%');
        const numValue = parseFloat(valueStr);
        return { ...bar, numValue, isPercent, displayValue: bar.value };
    });
    const hasPercent = parsedBars.some(b => b.isPercent);
    const maxValue = hasPercent ? 100 : Math.max(...parsedBars.map(b => b.numValue));

    const barsHtml = parsedBars.map(bar => {
        const heightPct = (bar.numValue / maxValue) * 100;
        return `
            <div class="thumb-bar-wrapper">
                <div class="thumb-bar-area">
                    <div class="thumb-bar" style="height: ${heightPct}%"></div>
                </div>
                <span class="thumb-value">${bar.displayValue}</span>
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

/** Inline SVG line: minimal, axis labels only (no ticks), optional dashed floor. */
function renderThumbnailLine(thumbnail) {
    const { line } = thumbnail;
    const ys = line.y || [];
    if (ys.length < 2) return '';
    const xs = line.x && line.x.length === ys.length
        ? line.x
        : ys.map((_, i) => i);

    // Reserve gutters for axis labels
    const W = 130, H = 70;
    const left = 14, right = 4, top = 2, bottom = 14;
    const xMin = Math.min(...xs);
    const xMax = Math.max(...xs);
    const yMin = line.y_min !== undefined ? line.y_min : Math.min(...ys);
    const yMax = line.y_max !== undefined ? line.y_max : Math.max(...ys);
    const xSpan = (xMax - xMin) || 1;
    const ySpan = (yMax - yMin) || 1;
    const sx = x => left + ((x - xMin) / xSpan) * (W - left - right);
    const sy = y => (H - bottom) - ((y - yMin) / ySpan) * (H - top - bottom);

    const points = xs.map((x, i) => `${sx(x).toFixed(1)},${sy(ys[i]).toFixed(1)}`).join(' ');

    let floorEl = '';
    let crossingEl = '';
    if (line.floor !== undefined && line.floor >= yMin && line.floor <= yMax) {
        const fy = sy(line.floor).toFixed(1);
        floorEl = `<line x1="${left}" x2="${W - right}" y1="${fy}" y2="${fy}" stroke="currentColor" stroke-opacity="0.35" stroke-width="0.8" stroke-dasharray="2 2"/>`;
        // Linear interpolation: find the first x at which the line crosses the floor (descending only)
        let crossX = null;
        for (let i = 0; i < xs.length - 1; i++) {
            const [y1, y2] = [ys[i], ys[i + 1]];
            if (y1 > line.floor && y2 <= line.floor && y1 !== y2) {
                const frac = (y1 - line.floor) / (y1 - y2);
                crossX = xs[i] + frac * (xs[i + 1] - xs[i]);
                break;
            }
        }
        if (crossX !== null) {
            const cxPx = sx(crossX).toFixed(1);
            const fyPx = sy(line.floor);
            // Drop a thin tick from the floor line down to the x axis at the crossing
            const tick = `<line x1="${cxPx}" x2="${cxPx}" y1="${fyPx.toFixed(1)}" y2="${(H - bottom).toFixed(1)}" stroke="currentColor" stroke-opacity="0.55" stroke-width="0.8" stroke-dasharray="1.5 1.5"/>`;
            // Label sits just above the floor, anchored at the crossing
            const label = `<text x="${(parseFloat(cxPx) + 2).toFixed(1)}" y="${(fyPx - 2).toFixed(1)}" text-anchor="start" font-size="9" font-weight="600" fill="currentColor" fill-opacity="0.95">${crossX.toFixed(2)}</text>`;
            crossingEl = tick + label;
        }
    }

    const dots = xs.map((x, i) => `<circle cx="${sx(x).toFixed(1)}" cy="${sy(ys[i]).toFixed(1)}" r="1.6" fill="currentColor" fill-opacity="0.9"/>`).join('');

    const xLabel = line.x_label || '';
    const yLabel = line.y_label || '';
    const xLabelEl = xLabel
        ? `<text x="${(left + W - right) / 2}" y="${H - 2}" text-anchor="middle" font-size="9" fill="currentColor" fill-opacity="0.85">${xLabel}</text>`
        : '';
    const yLabelEl = yLabel
        ? `<text x="${5}" y="${(top + H - bottom) / 2}" text-anchor="middle" font-size="9" fill="currentColor" fill-opacity="0.85" transform="rotate(-90 5 ${(top + H - bottom) / 2})">${yLabel}</text>`
        : '';

    return `
        <div class="thumbnail-chart thumbnail-chart-line">
            <svg class="thumb-line-svg" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-hidden="true">
                ${floorEl}
                <polyline points="${points}" fill="none" stroke="currentColor" stroke-width="1.4" stroke-opacity="0.85" stroke-linecap="round" stroke-linejoin="round"/>
                ${dots}
                ${crossingEl}
                ${xLabelEl}
                ${yLabelEl}
            </svg>
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
            thumbnail: frontmatter.thumbnail || null,
            date: frontmatter.date || null,
            tier: frontmatter.tier || null
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
        const { html } = renderMarkdownContent(text, {
            customBlocks: true,
            citations: true,
            assetBaseUrl: '/docs/viz_findings/',
            namespace: filename
        });

        loadedFindings[filename] = html;
        return html;
    } catch (error) {
        console.error(`Error loading finding ${filename}:`, error);
        return `<div class="error">Failed to load ${filename}</div>`;
    }
}

async function toggleFinding(filename, cardEl) {
    const findingId = filename.replace('.md', '');
    toggleExpandCollapse(cardEl, '.finding-content', '.finding-toggle', async () => {
        const contentEl = cardEl.querySelector('.finding-content');
        if (!contentEl.innerHTML || contentEl.innerHTML === renderLoading()) {
            contentEl.innerHTML = renderLoading();
            const html = await loadFindingContent(filename);
            contentEl.innerHTML = `<div class="prose">${html}</div>`;
            renderMath(contentEl);
            if (window.customBlocks?.loadExpandedDropdowns) await window.customBlocks.loadExpandedDropdowns();
            if (window.customBlocks?.loadCharts) await window.customBlocks.loadCharts();
            if (window.citations?.initCitationClicks) window.citations.initCitationClicks(contentEl);
        }
        history.replaceState(null, '', `${window.location.pathname}${window.location.search}#${findingId}`);
    });
    // Collapse path: clear hash if this finding is in URL
    if (!cardEl.classList.contains('expanded') && window.location.hash === `#${findingId}`) {
        history.replaceState(null, '', window.location.pathname + window.location.search);
    }
}

async function renderFindings() {
    const contentArea = document.getElementById('content-area');
    const hash = window.location.hash.slice(1);

    // Check if we're in standalone mode
    if (window.state.currentView === 'finding' && hash) {
        return renderStandaloneFinding(hash);
    }

    contentArea.innerHTML = renderLoading('Loading findings...');

    const filenames = await loadFindingsOrder();
    if (!filenames || filenames.length === 0) {
        contentArea.innerHTML = '<div class="error">Failed to load findings index</div>';
        return;
    }

    const metadataList = await Promise.all(
        filenames.map(f => (typeof f === 'string' ? loadFindingMetadata(f) : Promise.resolve(null)))
    );

    let html = `
        <div class="findings-container">
            <div class="findings-header">
                <p class="findings-intro">Research findings from trait vector experiments. Click to expand.</p>
            </div>
            <div class="findings-list">
    `;

    filenames.forEach((filename, i) => {
        if (typeof filename !== 'string') {
            if (filename && filename.separator) {
                html += `<div class="finding-separator">${filename.separator}</div>`;
            }
            return;
        }
        const meta = metadataList[i];
        const isTodo = !meta.preview || meta.preview === 'TODO';
        const todoClass = isTodo ? 'finding-todo' : '';
        const findingId = filename.replace('.md', '');

        const thumbnailHtml = meta.thumbnail ? renderThumbnailChart(meta.thumbnail) : '';
        const hasThumbnail = meta.thumbnail ? ' has-thumbnail' : '';

        const dateLabel = meta.date ? `<span class="finding-date">${meta.date}</span>` : '';
        const metaLine = dateLabel ? `<div class="finding-meta">${dateLabel}</div>` : '';

        html += `
            <div class="finding-card ${todoClass}${hasThumbnail}" id="finding-${findingId}">
                <div class="finding-header" onclick="toggleFinding('${filename}', document.getElementById('finding-${findingId}'))">
                    <div class="finding-header-content">
                        <div class="finding-title-row">
                            <span class="finding-toggle">▶</span>
                            <span class="finding-title">${meta.title}</span>
                            ${metaLine}
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
    renderMath(contentArea);
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
function backToFindings() {
    window.state.currentView = 'findings';
    setTabInURL('findings');
    window.renderView();
}

// ES module exports
export { renderFindings, toggleFinding, backToFindings };

// Keep window.* for router + onclick handlers in generated HTML
window.backToFindings = backToFindings;
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
