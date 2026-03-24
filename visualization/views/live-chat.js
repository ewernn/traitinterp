// Live Chat View — Chat with the model while watching trait dynamics in real-time
//
// Features:
// - Multi-turn conversation with full history passed to backend
// - Conversation branching (edit previous messages to create alternate paths)
// - Hover interactions (highlight message regions in chart)
// - Configurable running average on trait scores

import { escapeHtml } from '../core/utils.js';
import { isFeatureEnabled } from '../core/state.js';
import { ConversationTree } from '../core/conversation-tree.js';
import {
    toggleInferenceMode,
    updateInferenceModeUI,
    updateConnectionStatusUI,
    warmupModal,
    getInferenceMode,
    setInferenceMode,
    getModalConnectionState,
    setModalConnectionState,
    getPendingMessage,
    setPendingMessage,
} from '../components/inference-controls.js';
import {
    initTraitChart,
    updateTraitChart as updateTraitChartImpl,
    updateChartHighlight as updateChartHighlightImpl,
    setSteeringCoefficient,
    getSteeringCoefficients,
    setVectorMetadata,
    getShowSmoothedLine,
    setShowSmoothedLine,
    resetChartState,
} from '../components/live-chat-chart.js';

// Live Chat always uses this experiment (separate from sidebar selection)
const LIVE_CHAT_EXPERIMENT = 'live-chat';

// Chat state
let conversationTree = null;
let currentAssistantNodeId = null;
let isGenerating = false;
let abortController = null;
let hoveredMessageId = null;
let editingNodeId = null;
let currentModelType = 'application';  // 'application' or 'extraction'
let maxContextLength = 8192;  // Loaded from model config

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

// Thin wrappers that pass conversation state to chart module
function updateTraitChartWrapped() {
    updateTraitChartImpl(conversationTree, hoveredMessageId);
}
function updateChartHighlight() {
    updateChartHighlightImpl(conversationTree, hoveredMessageId);
}

/**
 * Load model config from experiment config (already fetched by state.js loadExperiment,
 * or fetch directly for live-chat experiment which may not be the sidebar experiment).
 */
async function loadModelConfig() {
    try {
        const response = await fetch(`/api/experiments/${LIVE_CHAT_EXPERIMENT}/config`);
        const config = await response.json();
        maxContextLength = config.max_context_length || 8192;
    } catch (e) {
        console.warn('Failed to load model config:', e);
        maxContextLength = 8192;
    }
}

/**
 * Render the live chat view
 */
async function renderLiveChat() {
    const container = document.getElementById('content-area');
    if (!container) return;

    // Load config for context length
    await loadModelConfig();

    // Initialize conversation tree if needed
    if (!conversationTree) {
        conversationTree = new ConversationTree();
        // Try to restore from localStorage
        restoreConversation();
    }

    // If already rendered with conversation, just update chart (don't rebuild UI)
    const existingView = container.querySelector('.live-chat-view');
    if (existingView && conversationTree.globalTokens.length > 0) {
        updateTraitChartWrapped();
        return;
    }

    container.innerHTML = `
        <div class="tool-view live-chat-view">
            <div style="text-align:center; padding:24px 0 8px; color:black; font-size:15px; font-weight:600; letter-spacing:1px;">CONSTRUCTION DELAYED</div>
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
                            ${ui.renderToggle({ id: 'smooth-toggle', label: '3-token avg', checked: getShowSmoothedLine(), className: 'smooth-toggle' })}
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
    setInferenceMode(defaultBackend);

    if (getInferenceMode() === 'modal') {
        // Production: start warming up Modal immediately
        warmupModal();
    } else {
        // Development: local mode is always ready
        setModalConnectionState('connected');
        updateConnectionStatusUI(isGenerating);
    }
    updateInferenceModeUI();

    // Note: Experiment picker stays visible - Live Chat uses LIVE_CHAT_EXPERIMENT internally
    // Sidebar selection doesn't affect Live Chat

    // Hide inference toggle in production mode
    if (!isFeatureEnabled('inference_toggle')) {
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
            setShowSmoothedLine(e.target.checked);
            updateTraitChartWrapped();
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
    if (getInferenceMode() === 'modal' && getModalConnectionState() === 'warming') {
        setPendingMessage(prompt);
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
    const steeringCoefficients = getSteeringCoefficients();

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
                inference_mode: getInferenceMode(),
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
                                setVectorMetadata(event.vector_metadata);
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
                        updateTraitChartWrapped();

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
        updateTraitChartWrapped();
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
    return escapeHtml(node.content);
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
    resetChartState();

    // Clear localStorage
    localStorage.removeItem(getStorageKey());

    input.value = '';
    sendBtn.disabled = false;
    sendBtn.textContent = 'Send';

    renderMessages();
    initTraitChart();
}

// ES module exports
export {
    renderLiveChat,
    startEdit,
    navigateBranch,
    handleMessageHover,
    setSteeringCoefficient,
    toggleInferenceMode,
};

// Keep window.* for onclick handlers in generated HTML + router
window.startEdit = startEdit;
window.navigateBranch = navigateBranch;
window.setSteeringCoefficient = setSteeringCoefficient;
window.toggleInferenceMode = toggleInferenceMode;
window.renderLiveChat = renderLiveChat;
