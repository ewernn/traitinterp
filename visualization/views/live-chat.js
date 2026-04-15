// Live Chat View — Chat with the model while watching trait dynamics in real-time
//
// Features:
// - Multi-turn conversation with full history passed to backend
// - Conversation branching (edit previous messages to create alternate paths)
// - Hover interactions (highlight message regions in chart)
// - Configurable running average on trait scores

import { escapeHtml } from '../core/utils.js';
import { isFeatureEnabled } from '../core/state.js';
import { renderToggle } from '../core/ui.js';
import { ConversationTree } from '../core/conversation-tree.js';
import { LIVE_CHAT_EXPERIMENT, CHAT_MODEL_TYPE } from '../core/chat-config.js';
import {
    on,
    toggleInferenceMode,
    wakeChatBackend,
    updateInferenceModeUI,
    startChatStatusPolling,
    fetchChatStatus,
    seedInferenceModeFromConfig,
    getInferenceMode,
    getChatStatus,
    isChatReady,
    isWakeInFlight,
    getPendingMessage,
    setPendingMessage,
    getWakeProgress,
} from '../components/inference-controls.js';
import {
    initTraitChart,
    updateTraitChart as updateTraitChartImpl,
    updateChartHighlight as updateChartHighlightImpl,
    setSteeringCoefficient,
    setSteeringLocked,
    getSteeringCoefficients,
    setVectorMetadata,
    getShowSmoothedLine,
    setShowSmoothedLine,
    resetChartState,
} from '../components/live-chat-chart.js';

// Chat state
let conversationTree = null;
let currentAssistantNodeId = null;
let isGenerating = false;
let abortController = null;
let hoveredMessageId = null;
let editingNodeId = null;
let currentModelType = CHAT_MODEL_TYPE;  // could become togglable later (e.g. 'extraction')
let maxContextLength = 8192;  // Loaded from model config

/**
 * Get localStorage key for current experiment
 */
function getStorageKey() {
    return `livechat_${LIVE_CHAT_EXPERIMENT}`;
}

/**
 * Save conversation to localStorage, tagged with the backend + model that
 * generated it. Trait projections are model-specific — restoring a
 * conversation from a different model would show stale chart lines — so
 * restoreConversation() only matches when backend+model align.
 */
function saveConversation() {
    if (!conversationTree || conversationTree.isEmpty()) return;
    try {
        const status = getChatStatus();
        const payload = {
            tree: conversationTree.toJSON(),
            backend: status.backend || getInferenceMode(),
            model: status.model || null,
        };
        localStorage.setItem(getStorageKey(), JSON.stringify(payload));
    } catch (e) {
        console.warn('[LiveChat] Failed to save conversation:', e);
        if (e.name === 'QuotaExceededError') {
            localStorage.removeItem(getStorageKey());
        }
    }
}

/**
 * Restore conversation from localStorage — only if the saved (backend, model)
 * matches the currently selected one. On mismatch we silently discard and
 * start fresh; a cold backend means the conversation can't be continued
 * anyway and a different model would produce nonsense trait lines.
 */
function restoreConversation() {
    try {
        const saved = localStorage.getItem(getStorageKey());
        if (!saved) return false;
        const parsed = JSON.parse(saved);
        // Legacy format (pre-tagging) had the tree at the top level. Discard those.
        if (!parsed || !parsed.tree) {
            return false;
        }
        const status = getChatStatus();
        const currentBackend = status.backend || getInferenceMode();
        const currentModel = status.model;
        if (parsed.backend !== currentBackend || (currentModel && parsed.model !== currentModel)) {
            return false;
        }
        conversationTree.fromJSON(parsed.tree);
        return true;
    } catch (e) {
        console.warn('[LiveChat] Failed to restore conversation:', e);
        localStorage.removeItem(getStorageKey());
        return false;
    }
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
let liveChatExperimentConfig = null;
let inferenceModeSeeded = false;  // only seed from app config on first mount

async function loadModelConfig() {
    const response = await fetch(`/api/experiments/${LIVE_CHAT_EXPERIMENT}/config`);
    if (!response.ok) {
        throw new Error(`Failed to load experiment config for ${LIVE_CHAT_EXPERIMENT}: HTTP ${response.status}`);
    }
    const config = await response.json();
    if (!config.max_context_length) {
        throw new Error(`Experiment config for ${LIVE_CHAT_EXPERIMENT} missing max_context_length`);
    }
    maxContextLength = config.max_context_length;
    liveChatExperimentConfig = config;
}

// Unsubscribe fns for chat bus listeners (cleared on remount).
const chatBusUnsubs = [];
function clearChatBusSubscriptions() {
    while (chatBusUnsubs.length) chatBusUnsubs.pop()();
}

async function renderLiveChat() {
    const container = document.getElementById('content-area');
    if (!container) return;

    await loadModelConfig();

    // Seed mode from app config on first mount only; later toggles are the
    // user's to control.
    const appConfig = window.state.appConfig || {};
    if (!inferenceModeSeeded) {
        seedInferenceModeFromConfig(appConfig.defaults?.inference_backend || 'local');
        inferenceModeSeeded = true;
    }

    // Kick off polling + one immediate status fetch so the UI has truth.
    startChatStatusPolling();
    await fetchChatStatus();

    // Conversation tree: create once, restore only if tagged (backend, model) match.
    if (!conversationTree) {
        conversationTree = new ConversationTree();
        restoreConversation();
    }

    // Re-render guard: if the view already exists, just refresh dynamic pieces.
    const existingView = container.querySelector('.live-chat-view');
    if (existingView) {
        applyChatStatusToDOM();
        if (conversationTree && conversationTree.globalTokens.length > 0) {
            updateTraitChartWrapped();
        }
        return;
    }

    // liveChatExperimentConfig is populated by loadModelConfig() (awaited above).
    // Missing fields mean the live-chat experiment config.json is malformed —
    // fail loud rather than guess.
    if (!liveChatExperimentConfig) {
        throw new Error('liveChatExperimentConfig not loaded — loadModelConfig must run first');
    }
    const lcVariant = liveChatExperimentConfig.defaults?.application;
    if (!lcVariant) {
        throw new Error(`${LIVE_CHAT_EXPERIMENT}/config.json missing defaults.application`);
    }
    const lcModelFull = liveChatExperimentConfig.model_variants?.[lcVariant]?.model;
    if (!lcModelFull) {
        throw new Error(`${LIVE_CHAT_EXPERIMENT}/config.json missing model_variants.${lcVariant}.model`);
    }
    const lcModelShort = lcModelFull.split('/').pop();

    container.innerHTML = `
        <div class="tool-view live-chat-view">
            <div class="live-chat-container">
                <!-- Top: Trait Chart -->
                <div class="trait-chart-panel">
                    <div class="chart-header">
                        <h3>Trait Dynamics</h3>
                        <div class="chart-controls">
                            <span id="model-info" class="model-info" title="${lcModelFull}">${lcModelShort}</span>
                            <label class="inference-mode-toggle">
                                <input type="checkbox" id="inference-mode-toggle" onchange="toggleInferenceMode()">
                                <span id="inference-mode-label">Local</span>
                            </label>
                            <div id="connection-status" class="connection-status"></div>
                            <button id="unload-btn" class="btn btn-secondary btn-xs" style="display:none" onclick="unloadChatBackend()">Unload</button>
                            ${renderToggle({ id: 'smooth-toggle', label: '3-token avg', checked: getShowSmoothedLine(), className: 'smooth-toggle' })}
                        </div>
                        <details id="trait-steering-wrap" class="trait-steering-details" open style="display:none">
                            <summary>Steering</summary>
                            <div class="chart-legend" id="chart-legend"></div>
                        </details>
                    </div>
                    <div id="trait-chart" class="trait-chart"></div>
                    <div id="wake-cta" class="wake-cta" style="display:none">
                        <button id="wake-btn" class="btn btn-primary" onclick="wakeChatBackend()">Wake GPU</button>
                        <div id="wake-hint" class="wake-hint"></div>
                    </div>
                </div>

                <!-- Bottom: Chat Interface -->
                <div class="chat-panel">
                    <div class="chat-messages" id="chat-messages">
                        <div class="chat-placeholder">Send a message to start chatting...</div>
                    </div>
                    <div class="chat-input-area">
                        <textarea id="chat-input" placeholder="Type your message..." rows="2"></textarea>
                        <div class="chat-controls">
                            <button id="send-btn" class="btn btn-primary">Send</button>
                            <button id="clear-btn" class="btn btn-secondary">Clear</button>
                        </div>
                    </div>
                </div>
            </div>

            <details class="detail-layer-results live-chat-howto">
                <summary>How it works</summary>
                <div style="padding: 12px; line-height: 1.6;">
                    <p>Send messages and watch how the model's internal traits shift token-by-token.</p>
                    <ul>
                        <li><strong>Trait chart</strong> — real-time projection of activations onto extracted trait vectors</li>
                        <li><strong>Steering</strong> — adjust coefficients in the chart legend to steer model behavior</li>
                        <li><strong>Branching</strong> — edit previous messages to explore alternate conversation paths</li>
                    </ul>
                </div>
            </details>
        </div>
    `;

    setupChatHandlers();
    initTraitChart();
    renderMessages();

    // Sync toggle checkbox + label to current inference mode (model-info span
    // is set in the template above; we intentionally don't call
    // updateModelInfoUI which would overwrite it with the sidebar experiment).
    updateInferenceModeUI();

    // Hide the toggle in production if the feature flag is off.
    if (!isFeatureEnabled('inference_toggle')) {
        const inferenceToggle = document.querySelector('.inference-mode-toggle');
        if (inferenceToggle) inferenceToggle.style.display = 'none';
    }

    // Subscribe to chat bus. One set of subscriptions per mount — a remount
    // clears and re-subscribes so we don't pile up duplicates.
    clearChatBusSubscriptions();
    chatBusUnsubs.push(on('change', applyChatStatusToDOM));
    // On ready, drain any prompt the user queued by hitting Send pre-warm.
    // inference-controls already moved it into the input; we just fire Send.
    chatBusUnsubs.push(on('ready', () => {
        const queued = getPendingMessage();
        if (queued && !isGenerating) {
            setPendingMessage(null);
            handleSend();
        }
    }));
    // On unload, the chart's cached metadata + lines belong to a dead
    // session. Reset so a future wake starts clean.
    chatBusUnsubs.push(on('unload', () => {
        resetChartState();
        initTraitChart();
    }));
    applyChatStatusToDOM();
}

/**
 * Show/hide chart, wake button, trait steering, send button, and unload
 * button based on current chat status. Pure DOM — reads state from
 * inference-controls and updates classes/styles.
 */
function applyChatStatusToDOM() {
    const ready = isChatReady();
    const mode = getInferenceMode();
    const waking = isWakeInFlight();
    const status = getChatStatus();

    const chartEl = document.getElementById('trait-chart');
    const wakeCta = document.getElementById('wake-cta');
    const steering = document.getElementById('trait-steering-wrap');
    const sendBtn = document.getElementById('send-btn');
    const unloadBtn = document.getElementById('unload-btn');
    const wakeBtn = document.getElementById('wake-btn');
    const wakeHint = document.getElementById('wake-hint');

    // Chart + trait steering: visible only when ready. Pre-ready we show an
    // empty chart panel (styled via CSS) with a centered Wake CTA — less
    // confrontational than a full-screen overlay.
    if (chartEl) chartEl.style.visibility = ready ? 'visible' : 'hidden';
    if (steering) steering.style.display = ready && document.getElementById('chart-legend')?.children.length ? '' : 'none';
    if (wakeCta) wakeCta.style.display = ready ? 'none' : '';
    if (sendBtn) sendBtn.style.display = ready ? '' : 'none';
    if (unloadBtn) unloadBtn.style.display = (ready && mode === 'local') ? '' : 'none';

    if (wakeBtn) {
        const cannotFit = mode === 'local' && status.can_fit_locally === false;
        wakeBtn.disabled = waking || cannotFit;
        if (waking) {
            const prog = getWakeProgress();
            const est = prog?.estimated_seconds;
            const elapsed = prog?.elapsed_s ?? 0;
            wakeBtn.textContent = est && elapsed < est ? `Waking... ~${est - elapsed}s` : `Waking... ${elapsed}s`;
        } else {
            wakeBtn.textContent = 'Wake GPU';
        }
        if (wakeHint) {
            if (cannotFit) {
                wakeHint.innerHTML = `${(status.model || 'Model').split('/').pop()} is too large for this machine. <a href="#" id="suggest-modal">Switch to Modal GPU</a>`;
                const link = wakeHint.querySelector('#suggest-modal');
                if (link) link.onclick = (e) => { e.preventDefault(); toggleInferenceMode(); };
            } else if (waking) {
                wakeHint.textContent = 'Loading model into memory...';
            } else {
                wakeHint.textContent = mode === 'modal' ? 'Starts a fresh Modal container.' : 'Loads the model locally.';
            }
        }
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

    // Single click listener for Send/Stop — behavior is state-dependent
    // (abort when a generation is in flight, send otherwise). Using one
    // listener here avoids the onclick= + addEventListener conflict we had
    // when the two were split across functions.
    sendBtn.addEventListener('click', () => {
        if (isGenerating && abortController) {
            abortController.abort();
        } else {
            handleSend();
        }
    });
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

    // Gate on backend readiness. If the user clicks Send without waking,
    // cache the prompt and fire the wake flow — it'll flush on ready.
    if (!isChatReady()) {
        setPendingMessage(prompt);
        if (!isWakeInFlight()) wakeChatBackend();
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
    // The single click listener in setupChatHandlers dispatches on
    // isGenerating (abort) vs not (send). We only flip the label here.
    sendBtn.disabled = false;
    sendBtn.textContent = 'Stop';

    // Get history for API (all messages BEFORE the current user message)
    // The assistant's parent is the current user message - we exclude it since prompt is sent separately
    const assistantNode = conversationTree.getNode(assistantNodeId);
    const currentUserNodeId = assistantNode.parentId;
    const history = conversationTree.getHistoryForAPI(currentUserNodeId);
    const previousContextLength = conversationTree.globalTokens.length;

    let responseText = '';

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
    // `marked` is loaded from CDN in index.html. If it failed to load we
    // don't want to silently fall back to plain-text rendering — the UI
    // would look broken in a confusing way. Fail loud instead.
    if (typeof marked === 'undefined') {
        throw new Error('marked.min.js failed to load (check CDN / index.html)');
    }
    return marked.parse(node.content, { breaks: true });
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

// Window.* bindings for inline onclick handlers and the router.
// toggleInferenceMode / wakeChatBackend / unloadChatBackend are bound inside
// inference-controls.js — don't re-bind here or we'll shadow the canonical ref.
window.startEdit = startEdit;
window.navigateBranch = navigateBranch;
window.handleMessageHover = handleMessageHover;
window.setSteeringCoefficient = setSteeringCoefficient;
window.renderLiveChat = renderLiveChat;
