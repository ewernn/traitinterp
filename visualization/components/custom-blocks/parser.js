/**
 * Custom block parser — markdown ::: syntax → block descriptors + placeholders
 *
 * Input:  raw markdown text
 * Output: { markdown (with placeholders), blocks (descriptors per type) }
 * Usage:  import { extractCustomBlocks } from './parser.js';
 */

/** Parse flags string against a schema (types: bool, int, string, quoted, enum:a,b,c, csv) */
function parseBlockFlags(flags, schema) {
    const result = {};
    for (const [key, type] of Object.entries(schema)) {
        if (type === 'bool') {
            result[key] = new RegExp(`\\b${key}\\b`).test(flags);
        } else if (type === 'int') {
            result[key] = parseInt(flags.match(new RegExp(`\\b${key}=(\\d+)`))?.[1]) || null;
        } else if (type === 'string') {
            result[key] = flags.match(new RegExp(`\\b${key}=([^\\s]+)`))?.[1] || null;
        } else if (type === 'quoted') {
            result[key] = flags.match(new RegExp(`\\b${key}="([^"]*)"`))?.[1] || null;
        } else if (type.startsWith('enum:')) {
            const options = type.slice(5);
            result[key] = flags.match(new RegExp(`\\b(${options})\\b`))?.[1] || null;
        } else if (type === 'csv') {
            result[key] = flags.match(new RegExp(`\\b${key}=([^\\s]+)`))?.[1]?.split(',') || null;
        }
    }
    return result;
}

/**
 * Extract all custom blocks from markdown, replacing each with a placeholder.
 * @param {string} markdown - Raw markdown content
 * @returns {Object} - { markdown, blocks }
 */
function extractCustomBlocks(markdown) {
    const blocks = {
        responses: [],
        datasets: [],
        figures: [],
        sideBySide: [],
        examples: [],
        steeredResponses: [],
        charts: [],
        extractionData: [],
        annotationStacked: []
    };

    // :::responses path "label" [expanded] [no-scores] [height=N] [color] [caption="..."]:::
    markdown = markdown.replace(
        /:::responses\s+([^\s:]+)(?:\s+"([^"]*)")?([^:]*):::/g,
        (match, path, label, flags) => {
            const f = parseBlockFlags(flags, {
                expanded: 'bool', 'no-scores': 'bool', height: 'int',
                color: 'enum:green,red,blue,orange,purple', caption: 'quoted'
            });
            blocks.responses.push({
                path, label: label || 'View responses',
                expanded: f.expanded, noScores: f['no-scores'], height: f.height, color: f.color,
                caption: f.caption
            });
            return `RESPONSE_BLOCK_${blocks.responses.length - 1}`;
        }
    );

    // :::dataset path "label" [expanded] [limit=N] [height=N] [color] [caption="..."]:::
    markdown = markdown.replace(
        /:::dataset\s+([^\s:]+)(?:\s+"([^"]*)")?([^:]*):::/g,
        (match, path, label, flags) => {
            const f = parseBlockFlags(flags, {
                expanded: 'bool', limit: 'int', height: 'int',
                color: 'enum:green,red,blue,orange,purple', caption: 'quoted'
            });
            blocks.datasets.push({
                path, label: label || 'View examples',
                expanded: f.expanded, limit: f.limit, height: f.height, color: f.color,
                caption: f.caption
            });
            return `DATASET_BLOCK_${blocks.datasets.length - 1}`;
        }
    );

    // :::figure path "caption" size:::
    markdown = markdown.replace(
        /:::figure\s+([^\s:]+)(?:\s+"([^"]*)")?(?:\s+(small|medium|large))?\s*:::/g,
        (match, path, caption, size) => {
            blocks.figures.push({ path, caption: caption || '', size: size || '' });
            return `FIGURE_BLOCK_${blocks.figures.length - 1}`;
        }
    );

    // :::side-by-side\n left: path "label" \n right: path "label" \n caption: "..." \n:::
    // Supports figure paths (.png/.jpg) or chart specs (chart:type:path)
    markdown = markdown.replace(
        /:::side-by-side\s*\n([\s\S]*?)\n:::/g,
        (match, body) => {
            const config = { left: {}, right: {}, caption: '' };
            for (const line of body.trim().split('\n')) {
                const leftMatch = line.match(/^\s*left:\s+(\S+)(?:\s+"([^"]*)")?/);
                const rightMatch = line.match(/^\s*right:\s+(\S+)(?:\s+"([^"]*)")?/);
                const captionMatch = line.match(/^\s*caption:\s+"([^"]*)"/);
                const styleMatch = line.match(/^\s*style:\s+(\w+)/);
                if (leftMatch) { config.left = { path: leftMatch[1], label: leftMatch[2] || '' }; }
                if (rightMatch) { config.right = { path: rightMatch[1], label: rightMatch[2] || '' }; }
                if (captionMatch) config.caption = captionMatch[1];
                if (styleMatch) config.style = styleMatch[1];
            }
            blocks.sideBySide.push(config);
            return `SIDEBYSIDE_BLOCK_${blocks.sideBySide.length - 1}`;
        }
    );

    // :::example ... ::: with caption (*caption text*)
    markdown = markdown.replace(
        /:::example\s*\n([\s\S]*?)\n:::\s*\n?\*([^*]+)\*/g,
        (match, content, caption) => {
            blocks.examples.push({ content: content.trim(), caption: caption.trim() });
            return `EXAMPLE_BLOCK_${blocks.examples.length - 1}`;
        }
    );

    // :::example ... ::: without caption
    markdown = markdown.replace(
        /:::example\s*\n([\s\S]*?)\n:::/g,
        (match, content) => {
            blocks.examples.push({ content: content.trim(), caption: '' });
            return `EXAMPLE_BLOCK_${blocks.examples.length - 1}`;
        }
    );

    // :::steered-responses "Label"\n trait: "TraitLabel" | pvPath | naturalPath \n:::
    markdown = markdown.replace(
        /:::steered-responses\s+"([^"]+)"\s*\n([\s\S]*?)\n:::/g,
        (match, label, body) => {
            const traits = [];
            for (const line of body.trim().split('\n')) {
                // Parse: traitKey: "Label" | pvPath | naturalPath
                const traitMatch = line.match(/^\s*(\w+):\s*"([^"]+)"\s*\|\s*([^\s|]+)\s*\|\s*([^\s|]+)/);
                if (traitMatch) {
                    traits.push({
                        key: traitMatch[1],
                        label: traitMatch[2],
                        pvPath: traitMatch[3].trim(),
                        naturalPath: traitMatch[4].trim()
                    });
                }
            }
            blocks.steeredResponses.push({ label, traits });
            return `STEERED_RESPONSES_BLOCK_${blocks.steeredResponses.length - 1}`;
        }
    );

    // :::chart type path "caption" [traits=...] [labels=...] [height=N] [perplexity=path] [projections=path,path]:::
    markdown = markdown.replace(
        /:::chart\s+(\S+)\s+(\S+)(?:\s+"([^"]*)")?(.*):::/g,
        (match, type, path, caption, flags) => {
            const f = parseBlockFlags(flags, {
                traits: 'csv', labels: 'string', height: 'int', perplexity: 'string', projections: 'string'
            });

            // Parse projection paths (comma-separated trait:path pairs)
            let projections = null;
            if (f.projections) {
                projections = {};
                for (const pair of f.projections.split(',')) {
                    const [trait, projPath] = pair.split(':');
                    if (trait && projPath) projections[trait] = projPath;
                }
            }

            // Parse label overrides (key>Display Name,key>Display Name)
            let labels = null;
            if (f.labels) {
                labels = {};
                for (const pair of f.labels.split(',')) {
                    const [key, display] = pair.split('>');
                    if (key && display) labels[key.trim()] = display.trim();
                }
            }

            blocks.charts.push({
                type, path, caption: caption || '',
                traits: f.traits, labels, height: f.height, perplexity: f.perplexity, projections
            });
            return `CHART_BLOCK_${blocks.charts.length - 1}`;
        }
    );

    // :::extraction-data "label" [expanded] [tokens=N]\n trait: path\n trait: path\n:::
    markdown = markdown.replace(
        /:::extraction-data\s+"([^"]+)"([^\n]*)\n([\s\S]*?)\n:::/g,
        (match, label, flags, body) => {
            const traits = [];
            for (const line of body.trim().split('\n')) {
                const traitMatch = line.match(/^\s*(\w+):\s*(.+)$/);
                if (traitMatch) {
                    traits.push({
                        name: traitMatch[1],
                        path: traitMatch[2].trim()
                    });
                }
            }
            const f = parseBlockFlags(flags, { expanded: 'bool', tokens: 'int' });
            blocks.extractionData.push({
                label, expanded: f.expanded, highlightTokens: f.tokens, traits
            });
            return `EXTRACTION_DATA_BLOCK_${blocks.extractionData.length - 1}`;
        }
    );

    // :::annotation-stacked "caption" [height=N]\n label: path\n label: path\n:::
    markdown = markdown.replace(
        /:::annotation-stacked\s+"([^"]+)"([^\n]*)\n([\s\S]*?)\n:::/g,
        (match, caption, flags, body) => {
            const bars = [];
            for (const line of body.trim().split('\n')) {
                // Parse: Label: path/to/file.json
                const barMatch = line.match(/^\s*([^:]+):\s*(.+)$/);
                if (barMatch) {
                    bars.push({
                        label: barMatch[1].trim(),
                        path: barMatch[2].trim()
                    });
                }
            }
            const f = parseBlockFlags(flags, { height: 'int', colors: 'string' });
            blocks.annotationStacked.push({ caption, height: f.height, colors: f.colors, bars });
            return `ANNOTATION_STACKED_BLOCK_${blocks.annotationStacked.length - 1}`;
        }
    );

    return { markdown, blocks };
}

export { parseBlockFlags, extractCustomBlocks };
