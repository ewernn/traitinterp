/** Methodology View - Renders docs/methodology.md with custom blocks and citations */
window.renderMethodology = () => renderMarkdownView('/docs/methodology.md', {
    customBlocks: true,
    citations: true,
    assetBaseUrl: '/docs/',
    namespace: 'methodology'
});
