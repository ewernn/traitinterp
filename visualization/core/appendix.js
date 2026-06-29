/**
 * Appendix - Anchor-linkable appendix entries at the bottom of a finding.
 *   Entries are parsed from a `## Appendix` markdown section, using `### key`
 *   subheadings as entry anchors. Inline markers `^[appx:key]` jump to the entry,
 *   and each entry back-links to its inline marker (bidirectional).
 *   Mirrors core/citations.js (numbered references) 1:1.
 *
 * Usage:
 *   const { appendix, markdown } = window.appendix.extractAppendix(markdown);
 *   markdown = window.appendix.processAppendixMarkers(markdown);   // pre-marked
 *   html = window.appendix.renderAppendixLinks(html);              // post-marked
 *   html += window.appendix.renderAppendixSection(appendix);
 *   window.appendix.initAppendixClicks(container);                 // after innerHTML
 */

/**
 * Extract `## Appendix` section from markdown and parse `### key` entries.
 * @param {string} markdown - Raw markdown content
 * @returns {Object} - { appendix: { key: { key, title, body } }, markdown (section stripped) }
 */
function extractAppendix(markdown) {
    const appendix = {};

    // Find ## Appendix section (case insensitive, captures to end or next ## section).
    // (?![\s\S]) is an end-of-string assertion (JS has no \Z).
    const apxMatch = markdown.match(/^##\s+Appendix\s*\n([\s\S]+?)(?=^##\s|(?![\s\S]))/mi);
    if (!apxMatch) {
        return { appendix, markdown };
    }

    const apxSection = apxMatch[1];
    const apxSectionFull = apxMatch[0];

    // Parse `### key Title` subheadings as entry anchors; body runs to next ### or end.
    const entryPattern = /^###\s+(\S+)([^\n]*)\n([\s\S]*?)(?=^###\s|(?![\s\S]))/gm;
    let match;
    while ((match = entryPattern.exec(apxSection)) !== null) {
        const key = match[1].trim();
        const title = match[2].trim();
        const body = match[3].trim();
        appendix[key] = { key, title, body };
    }

    // Remove appendix section from markdown (we re-render it with anchors)
    const markdownWithoutAppendix = markdown.replace(apxSectionFull, '');

    return { appendix, markdown: markdownWithoutAppendix };
}

/**
 * Replace ^[appx:key] markers with placeholders (PRE-marked, before marked.parse).
 * @param {string} markdown - Markdown content
 * @returns {string} - Markdown with ^[appx:key] replaced by APPENDIXLINK_key placeholders
 */
function processAppendixMarkers(markdown) {
    return markdown.replace(/\^\[appx:\s*([^\]]+)\]/g, (match, key) => {
        return `APPENDIXLINK_${key.trim()}`;
    });
}

/**
 * Replace APPENDIXLINK_key placeholders with superscript links (POST-marked).
 * Fails loud (console.warn + visible .appendix-missing marker) on an unknown key.
 * @param {string} html - Rendered HTML content
 * @param {Object} appendix - Appendix map from extractAppendix (for validation)
 * @returns {string} - HTML with appendix links rendered
 */
function renderAppendixLinks(html, appendix = {}) {
    return html.replace(/APPENDIXLINK_([^\s<]+)/g, (match, key) => {
        if (!appendix[key]) {
            console.warn(`Appendix marker ^[appx:${key}] not found in ## Appendix`);
            return `<sup class="appendix-ref appendix-missing">[appx:${key}]</sup>`;
        }
        return `<sup class="appendix-ref"><a href="javascript:void(0)" data-appendix="${key}" id="appendix-ref-${key}">[appendix]</a></sup>`;
    });
}

/**
 * Render the appendix section with anchors and back-links.
 * @param {Object} appendix - Appendix map from extractAppendix
 * @returns {string} - HTML for appendix section
 */
function renderAppendixSection(appendix) {
    const entries = Object.values(appendix);
    if (entries.length === 0) {
        return '';
    }

    let html = '<section class="appendix"><h2>Appendix</h2>';
    for (const entry of entries) {
        const titleHtml = entry.title
            ? `<h3 class="appendix-title">${entry.title}</h3>`
            : '';
        const bodyHtml = marked.parse(entry.body);
        html += `<div id="appendix-${entry.key}" class="appendix-entry">`
            + `${titleHtml}${bodyHtml}`
            + `<a class="appendix-back" data-appendix-back="${entry.key}" href="#appendix-ref-${entry.key}">↩</a>`
            + `</div>`;
    }
    html += '</section>';
    return html;
}

/**
 * Initialize appendix click handlers (call once after rendering).
 * Bidirectional: marker -> entry, entry back-link -> marker. No URL hash change.
 * @param {HTMLElement} container - Container element with appendix links
 */
function initAppendixClicks(container) {
    container.addEventListener('click', (e) => {
        const link = e.target.closest('a[data-appendix]');
        if (link) {
            const key = link.dataset.appendix;
            const target = container.querySelector(`#appendix-${key}`);
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'center' });
                target.classList.add('ref-highlight');
                setTimeout(() => target.classList.remove('ref-highlight'), 1500);
            } else {
                console.warn(`Appendix entry "${key}" not found`);
            }
            return;
        }

        const back = e.target.closest('a[data-appendix-back]');
        if (back) {
            const key = back.dataset.appendixBack;
            const target = container.querySelector(`#appendix-ref-${key}`);
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'center' });
                target.classList.add('ref-highlight');
                setTimeout(() => target.classList.remove('ref-highlight'), 1500);
            } else {
                console.warn(`Appendix marker "${key}" not found`);
            }
        }
    });
}

// ES module exports
export {
    extractAppendix,
    processAppendixMarkers,
    renderAppendixLinks,
    renderAppendixSection,
    initAppendixClicks,
};

// Keep window.* namespace for backward compat
window.appendix = {
    extractAppendix,
    processAppendixMarkers,
    renderAppendixLinks,
    renderAppendixSection,
    initAppendixClicks,
};
