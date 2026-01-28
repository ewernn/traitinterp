/**
 * Citations - Parse and render numbered citations (^1, ^2 style)
 *
 * Parses references from markdown body (## References section) and
 * replaces ^N markers with linked superscripts that scroll to refs.
 *
 * Usage in findings.js:
 *   const { markdown, refs } = window.citations.extractReferences(markdown);
 *   // ... render markdown ...
 *   html = window.citations.renderCitations(html, refs);
 */

/**
 * Extract references section from markdown and parse numbered entries
 * @param {string} markdown - Raw markdown content
 * @returns {Object} - { markdown (with refs section removed), refs: { "1": {...}, "2": {...} } }
 */
function extractReferences(markdown) {
    const refs = {};

    // Find ## References section (case insensitive, captures to end or next section)
    // Greedy match - References is typically at the end of the document
    const refsMatch = markdown.match(/^##\s+References\s*\n([\s\S]+)/mi);
    console.log('[citations] extractReferences called, found refs section:', !!refsMatch);

    if (!refsMatch) {
        return { markdown, refs };
    }

    const refsSection = refsMatch[1];
    const refsSectionFull = refsMatch[0];

    // Parse numbered entries: "1. Author. [Title](url). Year."
    // Supports formats:
    //   1. Text [Link](url) more text
    //   1. Text without link
    const refPattern = /^(\d+)\.\s+(.+)$/gm;
    let match;

    while ((match = refPattern.exec(refsSection)) !== null) {
        const num = match[1];
        const content = match[2].trim();

        // Extract link if present: [Title](url)
        const linkMatch = content.match(/\[([^\]]+)\]\(([^)]+)\)/);

        refs[num] = {
            num,
            text: content,
            title: linkMatch ? linkMatch[1] : content,
            url: linkMatch ? linkMatch[2] : null
        };
    }

    // Remove refs section from markdown (we'll re-render it with anchors)
    const markdownWithoutRefs = markdown.replace(refsSectionFull, '');

    return { markdown: markdownWithoutRefs, refs };
}

/**
 * Replace ^N citation markers with superscript links
 * @param {string} markdown - Markdown content
 * @param {Object} refs - Reference map from extractReferences
 * @returns {string} - Markdown with ^N replaced by placeholders
 */
function processCitationMarkers(markdown, refs) {
    // Replace ^1, ^2 etc with placeholders
    // Matches ^N where N is a number (not inside code blocks ideally)
    return markdown.replace(/\^(\d+)/g, (match, num) => {
        if (refs[num]) {
            return `NUMCITE_${num}`;
        }
        console.warn(`Citation ^${num} not found in references`);
        return match; // Leave as-is if not found
    });
}

/**
 * Render citation placeholders as HTML superscript links
 * @param {string} html - Rendered HTML content
 * @param {Object} refs - Reference map from extractReferences
 * @returns {string} - HTML with citations rendered
 */
function renderCitations(html, refs) {
    // Replace NUMCITE_N placeholders with superscript links
    // Use data-ref instead of href to avoid changing URL hash
    for (const [num, ref] of Object.entries(refs)) {
        const tooltip = ref.title.replace(/"/g, '&quot;');
        const citeHtml = `<sup class="citation-num"><a href="javascript:void(0)" data-ref="${num}" data-tooltip="${tooltip}">${num}</a></sup>`;
        html = html.replaceAll(`NUMCITE_${num}`, citeHtml);
    }

    return html;
}

/**
 * Initialize citation click handlers (call once after rendering)
 * Scrolls to reference without changing URL hash
 * @param {HTMLElement} container - Container element with citations
 */
function initCitationClicks(container) {
    container.addEventListener('click', (e) => {
        const link = e.target.closest('.citation-num a[data-ref]');
        if (!link) return;

        const refNum = link.dataset.ref;
        const refEl = container.querySelector(`#ref-${refNum}`);
        if (refEl) {
            refEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
            // Brief highlight
            refEl.classList.add('ref-highlight');
            setTimeout(() => refEl.classList.remove('ref-highlight'), 1500);
        }
    });
}

/**
 * Render the references section with anchors
 * @param {Object} refs - Reference map from extractReferences
 * @returns {string} - HTML for references section
 */
function renderReferencesSection(refs) {
    const entries = Object.values(refs).sort((a, b) => parseInt(a.num) - parseInt(b.num));

    if (entries.length === 0) {
        return '';
    }

    let html = '<section class="references"><h2>References</h2><ol class="references-list">';

    for (const ref of entries) {
        // Re-render markdown links in the reference text
        let text = ref.text;
        if (ref.url) {
            // Convert [Title](url) to actual link
            text = text.replace(
                /\[([^\]]+)\]\(([^)]+)\)/g,
                '<a href="$2" target="_blank">$1</a>'
            );
        }
        html += `<li id="ref-${ref.num}" value="${ref.num}">${text}</li>`;
    }

    html += '</ol></section>';
    return html;
}

// Export
window.citations = {
    extractReferences,
    processCitationMarkers,
    renderCitations,
    renderReferencesSection,
    initCitationClicks
};
