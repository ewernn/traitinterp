/**
 * Zoomies Dev View
 * Data Explorer and other development tools.
 */

window.zoomies = window.zoomies || {};

/**
 * Render the dev view
 * @param {HTMLElement} container
 */
window.zoomies.renderDevView = async function(container) {
    const state = window.zoomies.state;

    container.innerHTML = `
        <div class="dev-view">
            <h2 style="margin-bottom: 16px;">Data Explorer</h2>

            <div class="dev-section">
                <h3>Current State</h3>
                <pre class="state-display">${JSON.stringify(state, null, 2)}</pre>
            </div>

            <div class="dev-section">
                <h3>Experiments</h3>
                <ul>
                    ${state.experiments.map(exp => `
                        <li class="${exp === state.experiment ? 'active' : ''}">
                            ${exp}
                        </li>
                    `).join('')}
                </ul>
            </div>

            <div class="dev-section">
                <h3>Traits (${state.traits.length})</h3>
                <ul class="trait-list">
                    ${state.traits.map(trait => `
                        <li>${trait}</li>
                    `).join('')}
                </ul>
            </div>

            <div class="dev-section">
                <h3>Prompt Sets</h3>
                <ul>
                    ${Object.entries(state.promptSets).map(([name, data]) => `
                        <li>
                            <strong>${name}</strong>: ${data.ids?.length || 0} prompts
                        </li>
                    `).join('')}
                </ul>
            </div>

            <div class="dev-section">
                <h3>Registry</h3>
                <p>Registered position keys:</p>
                <ul>
                    ${window.zoomies.registry.listKeys().map(key => `
                        <li>${key}</li>
                    `).join('')}
                </ul>
            </div>

            <div class="dev-section">
                <h3>Paths (from paths.yaml)</h3>
                <button onclick="testPaths()">Test Path Building</button>
                <pre id="path-test-output" style="margin-top: 8px; display: none;"></pre>
            </div>
        </div>
    `;
};

// Test path building
window.testPaths = async function() {
    const output = document.getElementById('path-test-output');
    if (!output) return;

    await window.zoomies.paths.load();
    window.zoomies.paths.setExperiment(window.zoomies.state.experiment);

    const tests = [
        'extraction.base',
        'extraction.vectors',
        'inference.residual_stream',
        'extraction_eval.evaluation',
    ];

    let results = '';
    tests.forEach(key => {
        const path = window.zoomies.paths.get(key, {
            trait: 'behavioral/refusal',
            prompt_set: 'single_trait',
        });
        results += `${key}:\n  ${path}\n\n`;
    });

    output.textContent = results;
    output.style.display = 'block';
};

// Add dev-specific styles
const devStyle = document.createElement('style');
devStyle.textContent = `
    .dev-view {
        padding: 16px;
    }
    .dev-section {
        margin-bottom: 24px;
        padding: 16px;
        background: var(--bg-secondary);
        border-radius: 4px;
    }
    .dev-section h3 {
        margin-bottom: 12px;
        font-size: var(--text-base);
        color: var(--text-primary);
    }
    .dev-section ul {
        list-style: none;
        padding-left: 16px;
    }
    .dev-section li {
        padding: 4px 0;
        color: var(--text-secondary);
        font-size: var(--text-sm);
    }
    .dev-section li.active {
        color: var(--primary-color);
        font-weight: 500;
    }
    .state-display {
        background: var(--bg-tertiary);
        padding: 12px;
        border-radius: 4px;
        font-size: var(--text-xs);
        max-height: 200px;
        overflow: auto;
    }
    .trait-list {
        max-height: 200px;
        overflow-y: auto;
    }
    .dev-section button {
        padding: 8px 16px;
        background: var(--primary-color);
        color: var(--text-on-primary);
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: var(--text-sm);
    }
    .dev-section button:hover {
        background: var(--primary-hover);
    }
`;
document.head.appendChild(devStyle);
