/**
 * Zoomies Tab Bar
 * Renders [Main] [Overview] [Dev] tabs.
 */

window.zoomies = window.zoomies || {};

/**
 * Render the tab bar
 */
window.zoomies.renderTabs = function() {
    const container = document.getElementById('tab-bar');
    if (!container) return;

    const { tab } = window.zoomies.state;

    container.innerHTML = `
        <button class="tab ${tab === 'main' ? 'active' : ''}" data-tab="main">
            Main
        </button>
        <button class="tab ${tab === 'overview' ? 'active' : ''}" data-tab="overview">
            Overview
        </button>
        <div class="tab-dropdown">
            <button class="tab ${tab === 'dev' ? 'active' : ''}">
                Dev <span style="font-size: 10px;">▾</span>
            </button>
            <div class="tab-dropdown-content">
                <div class="tab-dropdown-item" data-tab="dev" data-subtab="explorer">
                    Data Explorer
                </div>
            </div>
        </div>
        <div style="flex: 1;"></div>
        <button class="theme-toggle" onclick="window.zoomies.toggleTheme()" title="Toggle theme">
            ${document.documentElement.getAttribute('data-theme') === 'dark' ? '☀️' : '🌙'}
        </button>
    `;

    // Add click handlers
    container.querySelectorAll('[data-tab]').forEach(el => {
        el.addEventListener('click', (e) => {
            const newTab = e.currentTarget.dataset.tab;
            if (newTab !== tab) {
                window.zoomies.setState({ tab: newTab });
            }
        });
    });
};
