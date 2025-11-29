/**
 * Zoomies Breadcrumb
 * Dropdown navigation showing current position in state space.
 */

window.zoomies = window.zoomies || {};

/**
 * Render breadcrumb navigation
 * @param {HTMLElement} container
 * @param {Object} props
 * @param {Array} props.segments - Array of { label, value, options, formatter }
 *   - label: Display text for current value
 *   - value: Current value
 *   - options: Array of possible values (null = not a dropdown, just text)
 *   - formatter: Optional function to format option for display
 * @param {Function} props.onChange - (segmentIndex, newValue) => void
 */
window.zoomies.renderBreadcrumb = function(container, props) {
    const { segments, onChange } = props;

    let html = '<nav class="breadcrumb">';

    segments.forEach((segment, index) => {
        const isLast = index === segments.length - 1;
        const hasOptions = segment.options && segment.options.length > 0;

        if (hasOptions && !isLast) {
            // Dropdown segment
            html += `
                <div class="breadcrumb-segment" data-index="${index}">
                    <button class="breadcrumb-btn">
                        ${segment.label}
                        <span class="arrow">▾</span>
                    </button>
                    <div class="breadcrumb-dropdown">
                        ${segment.options.map(opt => {
                            const optLabel = segment.formatter ? segment.formatter(opt) : opt;
                            const isSelected = opt === segment.value;
                            return `
                                <div class="breadcrumb-option ${isSelected ? 'selected' : ''}" data-value="${opt}">
                                    ${optLabel}${isSelected ? ' ✓' : ''}
                                </div>
                            `;
                        }).join('')}
                    </div>
                </div>
            `;
        } else {
            // Static text (current position or final segment)
            html += `<span class="breadcrumb-current">${segment.label}</span>`;
        }

        // Separator
        if (!isLast) {
            html += '<span class="breadcrumb-separator">›</span>';
        }
    });

    html += '</nav>';
    container.innerHTML = html;

    // Add click handlers for dropdowns
    container.querySelectorAll('.breadcrumb-segment').forEach(segmentEl => {
        const btn = segmentEl.querySelector('.breadcrumb-btn');
        const dropdown = segmentEl.querySelector('.breadcrumb-dropdown');

        // Toggle dropdown on click
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const isOpen = segmentEl.classList.contains('open');

            // Close all dropdowns
            container.querySelectorAll('.breadcrumb-segment').forEach(s => {
                s.classList.remove('open');
            });

            // Toggle this one
            if (!isOpen) {
                segmentEl.classList.add('open');
            }
        });

        // Handle option selection
        dropdown.querySelectorAll('.breadcrumb-option').forEach(optEl => {
            optEl.addEventListener('click', (e) => {
                e.stopPropagation();
                const index = parseInt(segmentEl.dataset.index, 10);
                let value = optEl.dataset.value;

                // Parse numeric values
                if (!isNaN(parseInt(value, 10)) && value !== 'all') {
                    value = parseInt(value, 10);
                }

                segmentEl.classList.remove('open');
                onChange(index, value);
            });
        });
    });

    // Close dropdowns when clicking outside
    document.addEventListener('click', () => {
        container.querySelectorAll('.breadcrumb-segment').forEach(s => {
            s.classList.remove('open');
        });
    });
};

/**
 * Build breadcrumb segments from current state
 */
window.zoomies.buildBreadcrumbSegments = function() {
    const state = window.zoomies.state;
    const segments = [];

    // Mode segment
    segments.push({
        label: state.mode === 'extraction' ? 'Extraction' : 'Inference',
        value: state.mode,
        options: ['extraction', 'inference'],
        formatter: (v) => v === 'extraction' ? 'Extraction' : 'Inference',
    });

    if (state.mode === 'inference') {
        // Token scope segment
        const tokenOptions = ['all'];
        if (state.totalTokens) {
            for (let t = 0; t < Math.min(state.totalTokens, 50); t++) {
                tokenOptions.push(t);
            }
        }

        segments.push({
            label: state.tokenScope === 'all' ? 'All Tokens' : `Token ${state.tokenScope}`,
            value: state.tokenScope,
            options: tokenOptions,
            formatter: (v) => v === 'all' ? 'All Tokens' : `Token ${v}`,
        });
    }

    // Layer scope segment
    const layerOptions = ['all'];
    for (let l = 0; l < window.zoomies.LAYERS; l++) {
        layerOptions.push(l);
    }

    segments.push({
        label: state.layerScope === 'all' ? 'All Layers' : `Layer ${state.layerScope}`,
        value: state.layerScope,
        options: layerOptions,
        formatter: (v) => v === 'all' ? 'All Layers' : `Layer ${v}`,
    });

    return segments;
};

/**
 * Handle breadcrumb change
 */
window.zoomies.handleBreadcrumbChange = function(segmentIndex, newValue) {
    const state = window.zoomies.state;
    const updates = {};

    if (segmentIndex === 0) {
        // Mode change
        updates.mode = newValue;
    } else if (state.mode === 'inference' && segmentIndex === 1) {
        // Token scope change
        updates.tokenScope = newValue;
    } else {
        // Layer scope change (index 1 for extraction, index 2 for inference)
        updates.layerScope = newValue;
    }

    window.zoomies.setState(updates);
};
