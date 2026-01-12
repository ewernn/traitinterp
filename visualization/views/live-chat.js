// Live Chat View - Chat with the model while watching trait dynamics in real-time
//
// Features:
// - Multi-turn conversation with full history passed to backend
// - Conversation branching (edit previous messages to create alternate paths)
// - Hover interactions (highlight message regions in chart)
// - 5-token running average on trait scores

// Chart colors from shared CSS vars (via state.js)
function getTraitColor(idx) {
    return window.getChartColors()[idx % 10];
}

// Chat state
let conversationTree = null;
let currentAssistantNodeId = null;
let isGenerating = false;
let abortController = null;
let hoveredMessageId = null;
let showSmoothedLine = true;
let editingNodeId = null;
let currentModelType = 'application';  // 'application' or 'extraction'
let modelNames = { application: null, extraction: null };  // Loaded from config
let maxContextLength = 8192;  // Loaded from model config
let vectorMetadata = {};  // Cached vector metadata: {trait: {layer, method, source}}

// Modal connection state
let modalConnectionState = 'disconnected';  // 'disconnected' | 'warming' | 'connected' | 'error'
let steeringCoefficients = {};  // {trait: coefficient} - default 0 for all traits
let pendingMessage = null;  // Message cached while waiting for connection
let inferenceMode = 'local';  // 'local' | 'modal' - toggled from UI

/**
 * Toggle inference mode between local and modal
 */
function toggleInferenceMode() {
    if (inferenceMode === 'local') {
        inferenceMode = 'modal';
        // Trigger warmup when switching to modal
        warmupModal();
    } else {
        inferenceMode = 'local';
        modalConnectionState = 'connected';  // Local is always ready
        updateConnectionStatusUI();
    }
    updateInferenceModeUI();
}

/**
 * Update inference mode toggle UI
 */
function updateInferenceModeUI() {
    const toggle = document.getElementById('inference-mode-toggle');
    if (toggle) {
        toggle.checked = inferenceMode === 'modal';
    }
    const label = document.getElementById('inference-mode-label');
    if (label) {
        label.textContent = inferenceMode === 'modal' ? 'Modal GPU' : 'Local';
    }
    // Update model info display
    updateModelInfoUI();
}

/**
 * Update model info display
 */
function updateModelInfoUI() {
    const modelInfo = document.getElementById('model-info');
    if (!modelInfo) return;

    // Get model name from experiment config or app config default
    const appConfig = window.state.appConfig || {};
    let modelName = appConfig.defaults?.model || 'gemma-2-2b-it';

    // Try to get from current experiment config
    if (modelNames.application) {
        modelName = modelNames.application;
    }

    // Shorten model name for display (remove org prefix)
    const shortName = modelName.split('/').pop();
    modelInfo.textContent = shortName;
    modelInfo.title = modelName;  // Full name on hover
}

/**
 * Warm up Modal GPU (called when switching to modal mode)
 */
async function warmupModal() {
    if (inferenceMode !== 'modal') {
        modalConnectionState = 'connected';  // Local is always ready
        updateConnectionStatusUI();
        return;
    }

    modalConnectionState = 'warming';
    updateConnectionStatusUI();

    try {
        const response = await fetch('/api/modal/warmup');
        const data = await response.json();

        if (data.status === 'ready') {
            modalConnectionState = 'connected';
            console.log('[LiveChat] Modal connected:', data);
        } else if (data.status === 'skipped') {
            // Server says skip modal (INFERENCE_MODE=local on server)
            modalConnectionState = 'connected';
            console.log('[LiveChat] Server skipped warmup:', data.reason);
        } else {
            modalConnectionState = 'error';
            console.error('[LiveChat] Warmup failed:', data);
        }
    } catch (e) {
        modalConnectionState = 'error';
        console.error('[LiveChat] Warmup error:', e);
    }

    updateConnectionStatusUI();

    // Check for pending message
    if (pendingMessage && modalConnectionState === 'connected') {
        const input = document.getElementById('chat-input');
        if (input) input.value = pendingMessage;
        pendingMessage = null;
    }
}

/**
 * Update connection status UI
 */
function updateConnectionStatusUI() {
    const statusEl = document.getElementById('connection-status');
    if (!statusEl) return;

    const states = {
        disconnected: { dot: 'disconnected', text: 'Disconnected' },
        warming: { dot: 'warming', text: 'Waking up GPU...' },
        connected: { dot: 'connected', text: 'Connected' },
        error: { dot: 'error', text: 'Connection failed' }
    };

    const state = states[modalConnectionState] || states.disconnected;
    statusEl.innerHTML = `
        <span class="status-dot ${state.dot}"></span>
        <span class="status-text">${state.text}</span>
    `;

    // Update send button state
    const sendBtn = document.getElementById('send-btn');
    if (sendBtn && modalConnectionState === 'warming') {
        sendBtn.disabled = true;
        sendBtn.textContent = 'Waiting...';
    } else if (sendBtn && !isGenerating) {
        sendBtn.disabled = false;
        sendBtn.textContent = 'Send';
    }
}

/**
 * Set steering coefficient for a trait
 */
function setSteeringCoefficient(trait, coefficient) {
    steeringCoefficients[trait] = coefficient;
    updateSteeringButtonsUI();
}

/**
 * Update steering buttons UI to reflect current state
 */
function updateSteeringButtonsUI() {
    document.querySelectorAll('.steering-buttons').forEach(container => {
        const trait = container.dataset.trait;
        const currentCoef = steeringCoefficients[trait] || 0;

        container.querySelectorAll('.steer-btn').forEach(btn => {
            const btnCoef = parseFloat(btn.dataset.coef);
            btn.classList.toggle('active', btnCoef === currentCoef);
        });
    });
}

/**
 * Load model names from experiment config
 */
async function loadModelNames() {
    try {
        const response = await fetch(`/api/experiments/${LIVE_CHAT_EXPERIMENT}/config`);
        const config = await response.json();

        modelNames.application = config.application_model || 'google/gemma-2-2b-it';
        modelNames.extraction = config.extraction_model || 'google/gemma-2-2b';
        maxContextLength = config.max_context_length || 8192;
    } catch (e) {
        console.error('Failed to load model config:', e);
        modelNames.application = 'application';
        modelNames.extraction = 'extraction';
        maxContextLength = 8192;
    }
}

/**
 * Get localStorage key for current experiment
 */
function getStorageKey() {
    return `livechat_${LIVE_CHAT_EXPERIMENT}`;
}

/**
 * Save conversation to localStorage
 */
function saveConversation() {
    if (!conversationTree || conversationTree.isEmpty()) return;
    try {
        const data = conversationTree.toJSON();
        localStorage.setItem(getStorageKey(), JSON.stringify(data));
    } catch (e) {
        console.warn('[LiveChat] Failed to save conversation:', e);
        // If storage is full, clear old data
        if (e.name === 'QuotaExceededError') {
            localStorage.removeItem(getStorageKey());
        }
    }
}

/**
 * Restore conversation from localStorage
 */
function restoreConversation() {
    try {
        const saved = localStorage.getItem(getStorageKey());
        if (saved) {
            const data = JSON.parse(saved);
            conversationTree.fromJSON(data);
            return true;
        }
    } catch (e) {
        console.warn('[LiveChat] Failed to restore conversation:', e);
        localStorage.removeItem(getStorageKey());
    }
    return false;
}

/**
 * Render the live chat view
 */
// Live Chat always uses this experiment (separate from sidebar selection)
const LIVE_CHAT_EXPERIMENT = 'live-chat';

async function renderLiveChat() {
    const container = document.getElementById('content-area');
    if (!container) return;

    // Load model names from live-chat config (don't change global currentExperiment)
    await loadModelNames();

    // Initialize conversation tree if needed
    if (!conversationTree) {
        conversationTree = new window.ConversationTree();
        // Try to restore from localStorage
        restoreConversation();
    }

    // If already rendered with conversation, just update chart (don't rebuild UI)
    const existingView = container.querySelector('.live-chat-view');
    if (existingView && conversationTree.globalTokens.length > 0) {
        updateTraitChart();
        return;
    }

    container.innerHTML = `
        <div class="tool-view live-chat-view">
            <div class="live-chat-container">
                <!-- Top: Trait Chart -->
                <div class="trait-chart-panel">
                    <div class="chart-header">
                        <h3>Trait Dynamics</h3>
                        <div class="chart-controls">
                            <span id="model-info" class="model-info"></span>
                            <label class="inference-mode-toggle">
                                <input type="checkbox" id="inference-mode-toggle" onchange="toggleInferenceMode()">
                                <span id="inference-mode-label">Local</span>
                            </label>
                            <div id="connection-status" class="connection-status">
                                <span class="status-dot connected"></span>
                                <span class="status-text">Ready</span>
                            </div>
                            <label class="smooth-toggle">
                                <input type="checkbox" id="smooth-toggle" ${showSmoothedLine ? 'checked' : ''}>
                                <span>3-token avg</span>
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

    // Setup event handlers
    setupChatHandlers();

    // Initialize empty chart
    initTraitChart();

    // Render existing messages if any
    renderMessages();

    // Initialize inference mode from app config
    const appConfig = window.state.appConfig || {};
    const defaultBackend = appConfig.defaults?.inference_backend || 'local';
    inferenceMode = defaultBackend;

    if (inferenceMode === 'modal') {
        // Production: start warming up Modal immediately
        warmupModal();
    } else {
        // Development: local mode is always ready
        modalConnectionState = 'connected';
        updateConnectionStatusUI();
    }
    updateInferenceModeUI();

    // Note: Experiment picker stays visible - Live Chat uses LIVE_CHAT_EXPERIMENT internally
    // Sidebar selection doesn't affect Live Chat

    // Hide inference toggle in production mode
    if (!window.isFeatureEnabled('inference_toggle')) {
        const inferenceToggle = document.querySelector('.inference-mode-toggle');
        if (inferenceToggle) inferenceToggle.style.display = 'none';
    }
}

/**
 * Setup chat event handlers
 */
function setupChatHandlers() {
    const input = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const clearBtn = document.getElementById('clear-btn');
    const modelSelect = document.getElementById('model-select');
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

    if (modelSelect) {
        modelSelect.addEventListener('change', (e) => {
            currentModelType = e.target.value;
            clearChat();
        });
    }

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

    // Block send while modal is warming up
    if (inferenceMode === 'modal' && modalConnectionState === 'warming') {
        pendingMessage = prompt;
        return;
    }

    // Check context limit (leave room for response)
    const contextBuffer = 500;  // Reserve tokens for response
    if (conversationTree.globalTokens.length > maxContextLength - contextBuffer) {
        alert(`Context limit reached (${maxContextLength} tokens). Clear chat to continue.`);
        return;
    }

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
    sendBtn.disabled = false;  // Keep enabled for stop functionality
    sendBtn.textContent = 'Stop';
    sendBtn.onclick = () => {
        if (abortController) {
            abortController.abort();
        }
    };

    // Get history for API (all messages BEFORE the current user message)
    // The assistant's parent is the current user message - we exclude it since prompt is sent separately
    const assistantNode = conversationTree.getNode(assistantNodeId);
    const currentUserNodeId = assistantNode.parentId;
    const history = conversationTree.getHistoryForAPI(currentUserNodeId);
    const previousContextLength = conversationTree.globalTokens.length;

    try {
        abortController = new AbortController();
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            signal: abortController.signal,
            body: JSON.stringify({
                prompt: prompt,
                experiment: LIVE_CHAT_EXPERIMENT,
                max_tokens: 256,
                temperature: 0.0,
                history: history,
                previous_context_length: previousContextLength,
                model_type: currentModelType,
                inference_mode: inferenceMode,
                steering_configs: Object.entries(steeringCoefficients)
                    .filter(([_, coef]) => coef !== 0)
                    .map(([trait, coefficient]) => ({ trait, coefficient }))
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
                            // Cache vector metadata if provided
                            if (event.vector_metadata) {
                                vectorMetadata = event.vector_metadata;
                            }
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

                        // Only add content tokens (not prompt, not special) to displayed response
                        if (!event.is_prompt && !event.is_special) {
                            responseText += event.token;

                            // Update node content for display
                            const node = conversationTree.getNode(assistantNodeId);
                            if (node) node.content = responseText;
                        }

                        // Append ALL tokens to conversation tree (for chart data)
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
        // Handle abort gracefully - keep tokens captured so far
        if (error.name === 'AbortError') {
            conversationTree.finalizeMessage(assistantNodeId, responseText);
        } else {
            const node = conversationTree.getNode(assistantNodeId);
            if (node) node.content = `Error: ${error.message}`;
        }
    } finally {
        isGenerating = false;
        currentAssistantNodeId = null;
        abortController = null;
        sendBtn.disabled = false;
        sendBtn.textContent = 'Send';
        sendBtn.onclick = () => handleSend();  // Restore normal click handler
        renderMessages();
        saveConversation();  // Persist to localStorage
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
 * Render message content with tokenization for hover highlighting
 */
function renderTokenizedContent(node) {
    if (!node.content) return '';

    // Render all messages with markdown
    if (typeof marked !== 'undefined') {
        const rendered = marked.parse(node.content, { breaks: true });
        return rendered;
    }
    return window.escapeHtml(node.content);
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

                <div class="message-content" data-message-id="${node.id}">
                    ${isCurrentlyGenerating && !node.content
                        ? '<span class="generating-indicator"></span>'
                        : renderTokenizedContent(node)}
                    ${isCurrentlyGenerating && node.content ? '<span class="generating-indicator"></span>' : ''}
                </div>

                ${(node.role === 'user' && !isGenerating) || hasBranches ? `
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
                ` : ''}
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
    vectorMetadata = {};

    // Clear localStorage
    localStorage.removeItem(getStorageKey());

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
    const allTraitNames = Object.keys(firstEvent.trait_scores || {});
    if (allTraitNames.length === 0) return;

    // Filter by selected traits (if any are selected)
    // selectedTraits has full paths like "behavioral_tendency/refusal"
    // allTraitNames has just base names like "refusal"
    const selectedTraits = window.state?.selectedTraits;
    let traitNames = allTraitNames;
    if (selectedTraits && selectedTraits.size > 0) {
        // Extract base names from selected traits for matching
        const selectedBaseNames = new Set(
            Array.from(selectedTraits).map(t => t.includes('/') ? t.split('/').pop() : t)
        );
        traitNames = allTraitNames.filter(t => selectedBaseNames.has(t));
    }
    if (traitNames.length === 0) traitNames = allTraitNames;  // Fallback: show all if none match

    const traces = [];

    traitNames.forEach((trait, idx) => {
        const color = getTraitColor(idx);

        // Collect all token scores with their indices
        const indices = [];
        const scores = [];

        globalTokens.forEach((e, i) => {
            const score = e.trait_scores[trait] || 0;
            indices.push(i);
            scores.push(score);
        });

        // Apply smoothing if requested
        const yValues = showSmoothedLine && scores.length >= 3
            ? window.smoothData(scores, 3)
            : scores;

        // Single trace per trait - all tokens
        traces.push({
            name: trait,
            x: indices,
            y: yValues,
            type: 'scatter',
            mode: 'lines',
            line: { color: color, width: 2 },
            hovertemplate: `${trait}: %{y:.3f}<extra></extra>`,
            showlegend: true
        });
    });

    // Update legend with steering buttons
    if (legendDiv) {
        legendDiv.innerHTML = traitNames.map((trait, idx) => {
            // Get vector metadata for this trait (trait names might be just base names)
            // Find matching trait in vectorMetadata by checking if key ends with trait name
            let metadata = null;
            for (const [fullPath, meta] of Object.entries(vectorMetadata)) {
                if (fullPath.endsWith(trait) || fullPath.endsWith('/' + trait)) {
                    metadata = meta;
                    break;
                }
            }

            const tooltipText = metadata
                ? `L${metadata.layer} ${metadata.method} (${metadata.source})`
                : 'no metadata';

            const currentCoef = steeringCoefficients[trait] || 0;
            const coefficients = [-1, -0.5, 0, 0.5, 1];

            return `
                <div class="legend-item-row">
                    <span class="legend-item has-tooltip"
                          data-tooltip="${tooltipText}"
                          data-trait="${trait}">
                        <span class="legend-color" style="background: ${getTraitColor(idx)}"></span>
                        ${trait}
                    </span>
                    <div class="steering-buttons" data-trait="${trait}">
                        ${coefficients.map(coef => {
                            const label = coef === 0 ? '0' : (coef > 0 ? `+${coef}x` : `${coef}x`);
                            const isActive = currentCoef === coef ? 'active' : '';
                            return `<button class="steer-btn ${isActive}" data-coef="${coef}" onclick="setSteeringCoefficient('${trait}', ${coef})">${label}</button>`;
                        }).join('')}
                    </div>
                </div>
            `;
        }).join('');
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

    // Add hover event listener for token highlighting
    // Note: Plotly event listeners are persistent across reacts, so we don't need to remove them
    // Only attach if not already attached
    if (!chartDiv._tokenHoverAttached) {
        chartDiv.on('plotly_hover', (data) => {
            if (data.points && data.points.length > 0) {
                const tokenIdx = Math.round(data.points[0].x);
                highlightTokenInChat(tokenIdx);
            }
        });

        chartDiv.on('plotly_unhover', () => {
            clearTokenHighlight();
        });

        chartDiv._tokenHoverAttached = true;
    }
}

/**
 * Highlight a specific token in the chat messages
 */
function highlightTokenInChat(tokenIdx) {
    // Clear previous highlights
    clearTokenHighlight();

    // Find token span with this index
    const tokenSpan = document.querySelector(`.token-span[data-token-idx="${tokenIdx}"]`);
    if (tokenSpan) {
        tokenSpan.classList.add('hovered-token');
        // Scroll into view if needed
        tokenSpan.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
}

/**
 * Clear token highlighting
 */
function clearTokenHighlight() {
    document.querySelectorAll('.token-span.hovered-token').forEach(el => {
        el.classList.remove('hovered-token');
    });
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
                    color: window.getCssVar?.('--chart-1', '#4a9eff') + '80',  // 50% opacity
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
                ? window.hexToRgba?.(window.getCssVar?.('--chart-1', '#4a9eff'), isHovered ? hoverOpacity : baseOpacity) || `rgba(74, 158, 255, ${isHovered ? hoverOpacity : baseOpacity})`
                : window.hexToRgba?.(window.getCssVar?.('--chart-3', '#51cf66'), isHovered ? hoverOpacity : baseOpacity) || `rgba(81, 207, 102, ${isHovered ? hoverOpacity : baseOpacity})`,
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

// smoothData (running average) is in core/utils.js

// Export functions to global scope for onclick handlers
window.startEdit = startEdit;
window.navigateBranch = navigateBranch;
window.handleMessageHover = handleMessageHover;
window.setSteeringCoefficient = setSteeringCoefficient;
window.toggleInferenceMode = toggleInferenceMode;

// Export for view system
window.renderLiveChat = renderLiveChat;
