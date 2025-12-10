// Live Chat View - Chat with the model while watching trait dynamics in real-time
//
// Features:
// - Multi-turn conversation with full history passed to backend
// - Conversation branching (edit previous messages to create alternate paths)
// - Hover interactions (highlight message regions in chart)
// - 5-token running average on trait scores

// Reuse TRAIT_COLORS from trait-dynamics.js if available, otherwise define locally
const CHAT_TRAIT_COLORS = window.TRAIT_COLORS || [
    '#4a9eff',  // blue
    '#ff6b6b',  // red
    '#51cf66',  // green
    '#ffd43b',  // yellow
    '#cc5de8',  // purple
    '#ff922b',  // orange
    '#20c997',  // teal
    '#f06595',  // pink
    '#748ffc',  // indigo
    '#a9e34b',  // lime
];

// Chat state
let conversationTree = null;
let currentAssistantNodeId = null;
let isGenerating = false;
let abortController = null;
let hoveredMessageId = null;
let showSmoothedLine = true;
let editingNodeId = null;

/**
 * Render the live chat view
 */
async function renderLiveChat() {
    const container = document.getElementById('content-area');
    if (!container) return;

    // Initialize conversation tree if needed
    if (!conversationTree) {
        conversationTree = new window.ConversationTree();
    }

    container.innerHTML = `
        <div class="tool-view live-chat-view">
            <div class="live-chat-container">
                <!-- Top: Trait Chart -->
                <div class="trait-chart-panel">
                    <div class="chart-header">
                        <h3>Trait Dynamics</h3>
                        <div class="chart-controls">
                            <label class="smooth-toggle">
                                <input type="checkbox" id="smooth-toggle" ${showSmoothedLine ? 'checked' : ''}>
                                <span>5-token avg</span>
                            </label>
                        </div>
                        <div class="chart-legend" id="chart-legend"></div>
                    </div>
                    <div id="trait-chart" class="trait-chart"></div>
                </div>

                <!-- Bottom: Chat Interface -->
                <div class="chat-panel">
                    <div class="chat-messages" id="chat-messages">
                        <div class="chat-placeholder">Send a message to start chatting...</div>
                    </div>
                    <div class="chat-input-area">
                        <textarea
                            id="chat-input"
                            placeholder="Type your message..."
                            rows="2"
                        ></textarea>
                        <div class="chat-controls">
                            <button id="send-btn" class="btn btn-primary">Send</button>
                            <button id="clear-btn" class="btn btn-secondary">Clear</button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;

    // Add styles
    addLiveChatStyles();

    // Setup event handlers
    setupChatHandlers();

    // Initialize empty chart
    initTraitChart();

    // Render existing messages if any
    renderMessages();
}

/**
 * Add CSS styles for live chat view (uses primitives from styles.css)
 */
function addLiveChatStyles() {
    if (document.getElementById('live-chat-styles')) return;

    const styles = document.createElement('style');
    styles.id = 'live-chat-styles';
    styles.textContent = `
        .live-chat-view {
            height: calc(100vh - 120px);
            display: flex;
            flex-direction: column;
        }

        .live-chat-container {
            display: flex;
            flex-direction: column;
            gap: 16px;
            flex: 1;
            min-height: 0;
        }

        .chat-panel {
            display: flex;
            flex-direction: column;
            background: var(--bg-secondary);
            border-radius: 2px;
            overflow: hidden;
            flex: 1;
            min-height: 200px;
        }

        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 12px;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .chat-placeholder {
            color: var(--text-tertiary);
            text-align: center;
            padding: 40px;
        }

        .chat-message {
            padding: 8px 12px;
            border-radius: 2px;
            max-width: 85%;
            font-size: var(--text-sm);
            line-height: 1.4;
            position: relative;
        }

        .chat-message.user {
            background: var(--primary-color);
            color: var(--text-on-primary);
            align-self: flex-end;
        }

        .chat-message.assistant {
            background: var(--bg-tertiary);
            color: var(--text-primary);
            align-self: flex-start;
        }

        .chat-message.highlighted {
            outline: 2px solid var(--accent-color);
            outline-offset: 2px;
        }

        .message-content {
            word-break: break-word;
        }

        .message-actions {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-top: 4px;
            opacity: 0;
            transition: opacity 0.15s;
        }

        .chat-message:hover .message-actions {
            opacity: 1;
        }

        .edit-btn {
            background: transparent;
            border: none;
            color: inherit;
            opacity: 0.6;
            cursor: pointer;
            padding: 2px 4px;
            font-size: var(--text-xs);
        }

        .edit-btn:hover {
            opacity: 1;
        }

        .branch-nav {
            display: flex;
            align-items: center;
            gap: 4px;
            font-size: var(--text-xs);
        }

        .branch-btn {
            background: rgba(255,255,255,0.2);
            border: none;
            color: inherit;
            cursor: pointer;
            padding: 2px 6px;
            border-radius: 2px;
            font-size: 10px;
        }

        .branch-btn:hover:not(:disabled) {
            background: rgba(255,255,255,0.3);
        }

        .branch-btn:disabled {
            opacity: 0.3;
            cursor: not-allowed;
        }

        .branch-indicator {
            opacity: 0.7;
            font-size: var(--text-xs);
            min-width: 30px;
            text-align: center;
        }

        .chat-input-area {
            padding: 12px;
            border-top: 1px solid var(--border-color);
            background: var(--bg-tertiary);
        }

        .chat-input-area textarea {
            width: 100%;
            background: var(--bg-primary);
            border: 1px solid var(--border-color);
            border-radius: 2px;
            padding: 8px;
            color: var(--text-primary);
            font-family: inherit;
            font-size: var(--text-sm);
            resize: none;
        }

        .chat-input-area textarea:focus {
            outline: none;
            border-color: var(--primary-color);
        }

        .chat-controls {
            display: flex;
            gap: 8px;
            margin-top: 8px;
        }

        .chat-controls button {
            padding: 4px 12px;
            border-radius: 2px;
            cursor: pointer;
            font-size: var(--text-sm);
        }

        .chat-controls .btn-primary {
            background: var(--primary-color);
            color: var(--text-on-primary);
            border: none;
        }

        .chat-controls .btn-primary:hover {
            background: var(--primary-hover);
        }

        .chat-controls .btn-primary:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .chat-controls .btn-secondary {
            background: transparent;
            color: var(--text-secondary);
            border: none;
        }

        .trait-chart-panel {
            display: flex;
            flex-direction: column;
            background: var(--bg-secondary);
            border-radius: 2px;
            overflow: hidden;
            flex: 0 0 auto;
            max-height: 350px;
        }

        .chart-header {
            padding: 12px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            gap: 16px;
            flex-wrap: wrap;
        }

        .chart-header h3 {
            margin: 0;
            font-size: var(--text-base);
            font-weight: 600;
            color: var(--text-primary);
        }

        .chart-controls {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .smooth-toggle {
            display: flex;
            align-items: center;
            gap: 4px;
            font-size: var(--text-xs);
            color: var(--text-secondary);
            cursor: pointer;
        }

        .smooth-toggle input {
            cursor: pointer;
        }

        .chart-legend {
            display: flex;
            flex-wrap: wrap;
            gap: 4px 12px;
            font-size: var(--text-xs);
            color: var(--text-secondary);
            flex: 1;
        }

        .legend-item {
            display: flex;
            align-items: center;
            gap: 4px;
        }

        .legend-color {
            width: 12px;
            height: 3px;
            border-radius: 1px;
        }

        .trait-chart {
            flex: 1;
            min-height: 200px;
        }

        .generating-indicator {
            display: inline-block;
            width: 6px;
            height: 6px;
            background: var(--primary-color);
            border-radius: 50%;
            animation: chat-pulse 1s infinite;
            margin-left: 4px;
        }

        @keyframes chat-pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.3; }
        }

        .editing-indicator {
            font-size: var(--text-xs);
            color: var(--text-tertiary);
            margin-bottom: 4px;
        }
    `;
    document.head.appendChild(styles);
}

/**
 * Setup chat event handlers
 */
function setupChatHandlers() {
    const input = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const clearBtn = document.getElementById('clear-btn');
    const smoothToggle = document.getElementById('smooth-toggle');

    sendBtn.addEventListener('click', () => handleSend());
    clearBtn.addEventListener('click', () => clearChat());

    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
        // ESC to cancel editing
        if (e.key === 'Escape' && editingNodeId) {
            cancelEdit();
        }
    });

    if (smoothToggle) {
        smoothToggle.addEventListener('change', (e) => {
            showSmoothedLine = e.target.checked;
            updateTraitChart();
        });
    }
}

/**
 * Handle send button click (new message or edit)
 */
function handleSend() {
    if (editingNodeId) {
        sendEditedMessage();
    } else {
        sendMessage();
    }
}

/**
 * Send a new message
 */
async function sendMessage() {
    const input = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');

    const prompt = input.value.trim();
    if (!prompt || isGenerating) return;

    // Add user message to tree
    const lastMsgId = conversationTree.getLastMessageId();
    const userNode = conversationTree.addMessage('user', prompt, lastMsgId);

    // Clear input
    input.value = '';

    // Create assistant node
    const assistantNode = conversationTree.addMessage('assistant', '', userNode.id);
    currentAssistantNodeId = assistantNode.id;

    // Render messages
    renderMessages();

    // Start generation
    await generateResponse(prompt, assistantNode.id);
}

/**
 * Send an edited message (creates a new branch)
 */
async function sendEditedMessage() {
    const input = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');

    const newContent = input.value.trim();
    if (!newContent || isGenerating) return;

    // Create new branch
    const newUserNode = conversationTree.createBranch(editingNodeId, newContent);
    if (!newUserNode) {
        console.error('Failed to create branch');
        return;
    }

    // Clear edit state
    cancelEdit();

    // Create assistant node for the new branch
    const assistantNode = conversationTree.addMessage('assistant', '', newUserNode.id);
    currentAssistantNodeId = assistantNode.id;

    // Render messages
    renderMessages();

    // Start generation
    await generateResponse(newContent, assistantNode.id);
}

/**
 * Generate response from the model
 */
async function generateResponse(prompt, assistantNodeId) {
    const sendBtn = document.getElementById('send-btn');

    isGenerating = true;
    sendBtn.disabled = true;
    sendBtn.textContent = 'Generating...';

    // Get history for API (all messages before current assistant)
    const history = conversationTree.getHistoryForAPI(assistantNodeId);

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt: prompt,
                experiment: window.state.currentExperiment || 'gemma-2-2b-it',
                max_tokens: 100,
                temperature: 0.7,
                history: history
            })
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let responseText = '';

        let streamDone = false;
        while (!streamDone) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const event = JSON.parse(line.slice(6));

                        if (event.error) {
                            const node = conversationTree.getNode(assistantNodeId);
                            if (node) node.content = `Error: ${event.error}`;
                            streamDone = true;
                            break;
                        }

                        if (event.status) {
                            // Update status in UI
                            const node = conversationTree.getNode(assistantNodeId);
                            if (node) node.content = event.message;
                            renderMessages();
                            continue;
                        }

                        if (event.done) {
                            conversationTree.finalizeMessage(assistantNodeId, responseText);
                            streamDone = true;
                            break;
                        }

                        responseText += event.token;

                        // Update node content for display
                        const node = conversationTree.getNode(assistantNodeId);
                        if (node) node.content = responseText;

                        // Append token to conversation tree
                        conversationTree.appendToken(assistantNodeId, event);

                        // Update UI
                        renderMessages();
                        updateTraitChart();

                    } catch (e) {
                        console.error('Failed to parse SSE event:', e);
                    }
                }
            }
        }

    } catch (error) {
        const node = conversationTree.getNode(assistantNodeId);
        if (node) node.content = `Error: ${error.message}`;
    } finally {
        isGenerating = false;
        currentAssistantNodeId = null;
        sendBtn.disabled = false;
        sendBtn.textContent = 'Send';
        renderMessages();
    }
}

/**
 * Start editing a user message
 */
function startEdit(nodeId) {
    const node = conversationTree.getNode(nodeId);
    if (!node || node.role !== 'user') return;

    editingNodeId = nodeId;

    const input = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');

    input.value = node.content;
    input.focus();
    sendBtn.textContent = 'Update';

    // Show editing indicator
    renderMessages();
}

/**
 * Cancel editing
 */
function cancelEdit() {
    editingNodeId = null;

    const input = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');

    input.value = '';
    sendBtn.textContent = 'Send';

    renderMessages();
}

/**
 * Navigate to a sibling branch
 */
function navigateBranch(nodeId, direction) {
    const siblings = conversationTree.getSiblings(nodeId);
    const currentIdx = conversationTree.getSiblingIndex(nodeId);
    const newIdx = currentIdx + direction;

    if (newIdx >= 0 && newIdx < siblings.length) {
        const newNode = siblings[newIdx];
        conversationTree.switchBranch(newNode.id);
        renderMessages();
        updateTraitChart();
    }
}

/**
 * Handle message hover for chart highlighting
 */
function handleMessageHover(messageId) {
    hoveredMessageId = messageId;

    // Update message highlighting in DOM
    document.querySelectorAll('.chat-message').forEach(el => {
        el.classList.toggle('highlighted', el.dataset.messageId === messageId);
    });

    // Update chart highlighting
    updateChartHighlight();
}

/**
 * Render all messages in the active path
 */
function renderMessages() {
    const messagesDiv = document.getElementById('chat-messages');
    if (!messagesDiv) return;

    if (conversationTree.isEmpty()) {
        messagesDiv.innerHTML = '<div class="chat-placeholder">Send a message to start chatting...</div>';
        return;
    }

    let html = '';

    // Show editing indicator if editing
    if (editingNodeId) {
        html += `<div class="editing-indicator">Editing message... (ESC to cancel)</div>`;
    }

    for (const nodeId of conversationTree.activePathIds) {
        const node = conversationTree.getNode(nodeId);
        if (!node) continue;

        const siblings = conversationTree.getSiblings(nodeId);
        const siblingIdx = conversationTree.getSiblingIndex(nodeId);
        const hasBranches = siblings.length > 1;
        const isCurrentlyGenerating = node.id === currentAssistantNodeId;
        const isBeingEdited = node.id === editingNodeId;

        html += `
            <div class="chat-message ${node.role} ${isBeingEdited ? 'highlighted' : ''}"
                 data-message-id="${node.id}"
                 onmouseenter="handleMessageHover('${node.id}')"
                 onmouseleave="handleMessageHover(null)">

                <div class="message-content">
                    ${isCurrentlyGenerating && !node.content
                        ? '<span class="generating-indicator"></span>'
                        : window.escapeHtml(node.content || '')}
                    ${isCurrentlyGenerating && node.content ? '<span class="generating-indicator"></span>' : ''}
                </div>

                <div class="message-actions">
                    ${node.role === 'user' && !isGenerating ? `
                        <button class="edit-btn" onclick="startEdit('${node.id}')" title="Edit message">
                            &#9998;
                        </button>
                    ` : ''}

                    ${hasBranches ? `
                        <div class="branch-nav">
                            <button class="branch-btn"
                                    onclick="navigateBranch('${node.id}', -1)"
                                    ${siblingIdx === 0 ? 'disabled' : ''}>
                                &#9664;
                            </button>
                            <span class="branch-indicator">${siblingIdx + 1}/${siblings.length}</span>
                            <button class="branch-btn"
                                    onclick="navigateBranch('${node.id}', 1)"
                                    ${siblingIdx === siblings.length - 1 ? 'disabled' : ''}>
                                &#9654;
                            </button>
                        </div>
                    ` : ''}
                </div>
            </div>
        `;
    }

    messagesDiv.innerHTML = html;
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

/**
 * Clear chat and reset chart
 */
function clearChat() {
    const messagesDiv = document.getElementById('chat-messages');
    const sendBtn = document.getElementById('send-btn');
    const input = document.getElementById('chat-input');

    conversationTree.clear();
    editingNodeId = null;
    currentAssistantNodeId = null;
    isGenerating = false;
    hoveredMessageId = null;

    input.value = '';
    sendBtn.disabled = false;
    sendBtn.textContent = 'Send';

    renderMessages();
    initTraitChart();
}

/**
 * Initialize empty trait chart
 */
function initTraitChart() {
    const chartDiv = document.getElementById('trait-chart');
    if (!chartDiv) return;

    const layout = window.getPlotlyLayout({
        margin: { l: 50, r: 20, t: 20, b: 40 },
        xaxis: {
            title: 'Token',
            showgrid: true,
        },
        yaxis: {
            title: 'Trait Score',
            showgrid: true,
            zeroline: true,
        },
        showlegend: false,
        hovermode: 'x unified',
    });

    Plotly.newPlot(chartDiv, [], layout, { responsive: true });
}

/**
 * Update trait chart with data from conversation tree
 */
function updateTraitChart() {
    const chartDiv = document.getElementById('trait-chart');
    const legendDiv = document.getElementById('chart-legend');

    const globalTokens = conversationTree.globalTokens;
    if (!chartDiv || globalTokens.length === 0) return;

    const firstEvent = globalTokens[0];
    const traitNames = Object.keys(firstEvent.trait_scores || {});
    if (traitNames.length === 0) return;

    const traces = [];

    traitNames.forEach((trait, idx) => {
        const rawY = globalTokens.map(e => e.trait_scores[trait] || 0);
        const x = globalTokens.map((_, i) => i);
        const color = CHAT_TRAIT_COLORS[idx % CHAT_TRAIT_COLORS.length];

        // Raw line
        traces.push({
            name: trait,
            x: x,
            y: rawY,
            type: 'scatter',
            mode: 'lines',
            line: { color: color, width: 1.5 },
            hovertemplate: `${trait}: %{y:.3f}<extra></extra>`,
            showlegend: true
        });

        // 5-token smoothed line (if enabled)
        if (showSmoothedLine && rawY.length >= 3) {
            const smoothedY = computeRunningAverage(rawY, 5);
            traces.push({
                name: `${trait} (avg)`,
                x: x,
                y: smoothedY,
                type: 'scatter',
                mode: 'lines',
                line: { color: color, width: 3 },
                opacity: 0.5,
                hovertemplate: `${trait} (5-avg): %{y:.3f}<extra></extra>`,
                showlegend: false
            });
        }
    });

    // Update legend
    if (legendDiv) {
        legendDiv.innerHTML = traitNames.map((trait, idx) => `
            <span class="legend-item">
                <span class="legend-color" style="background: ${CHAT_TRAIT_COLORS[idx % CHAT_TRAIT_COLORS.length]}"></span>
                ${trait}
            </span>
        `).join('');
    }

    // Build shapes for message regions
    const shapes = buildMessageRegionShapes();

    const layout = window.getPlotlyLayout({
        margin: { l: 50, r: 20, t: 20, b: 40 },
        xaxis: { title: 'Token', showgrid: true },
        yaxis: { title: 'Trait Score', showgrid: true, zeroline: true },
        showlegend: false,
        hovermode: 'x unified',
        shapes: shapes
    });

    Plotly.react(chartDiv, traces, layout, { responsive: true });
}

/**
 * Build Plotly shapes for message regions
 */
function buildMessageRegionShapes() {
    const shapes = [];

    for (const region of conversationTree.messageRegions) {
        // Skip empty regions (user messages with no tokens)
        if (region.startIdx === region.endIdx && region.role === 'user') {
            // Draw a thin vertical line for user messages
            shapes.push({
                type: 'line',
                x0: region.startIdx - 0.5,
                x1: region.startIdx - 0.5,
                y0: 0,
                y1: 1,
                yref: 'paper',
                line: {
                    color: 'rgba(74, 158, 255, 0.5)',
                    width: 2,
                    dash: 'dot'
                },
                layer: 'below'
            });
            continue;
        }

        const isHovered = region.messageId === hoveredMessageId;
        const baseOpacity = region.role === 'user' ? 0.15 : 0.08;
        const hoverOpacity = 0.25;

        shapes.push({
            type: 'rect',
            x0: region.startIdx - 0.5,
            x1: region.endIdx - 0.5,
            y0: 0,
            y1: 1,
            yref: 'paper',
            fillcolor: region.role === 'user'
                ? `rgba(74, 158, 255, ${isHovered ? hoverOpacity : baseOpacity})`   // Blue for user
                : `rgba(81, 207, 102, ${isHovered ? hoverOpacity : baseOpacity})`,  // Green for assistant
            line: { width: 0 },
            layer: 'below'
        });
    }

    return shapes;
}

/**
 * Update chart highlighting when hovering over messages
 */
function updateChartHighlight() {
    const chartDiv = document.getElementById('trait-chart');
    if (!chartDiv || !chartDiv.data || chartDiv.data.length === 0) return;

    const shapes = buildMessageRegionShapes();
    Plotly.relayout(chartDiv, { shapes: shapes });
}

/**
 * Compute running average with specified window size
 */
function computeRunningAverage(data, windowSize) {
    const result = [];
    const half = Math.floor(windowSize / 2);

    for (let i = 0; i < data.length; i++) {
        const start = Math.max(0, i - half);
        const end = Math.min(data.length, i + half + 1);
        const slice = data.slice(start, end);
        result.push(slice.reduce((a, b) => a + b, 0) / slice.length);
    }

    return result;
}

// Export functions to global scope for onclick handlers
window.startEdit = startEdit;
window.navigateBranch = navigateBranch;
window.handleMessageHover = handleMessageHover;

// Export for view system
window.renderLiveChat = renderLiveChat;
