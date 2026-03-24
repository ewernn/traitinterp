// Inference Controls — Modal/inference backend state and UI
//
// Input: User actions (toggle inference mode, warmup)
// Output: Connection status updates, model info display
// Usage: import { toggleInferenceMode, updateInferenceModeUI, ... } from './inference-controls.js'

// Module-local state
let inferenceMode = 'local';  // 'local' | 'modal' - toggled from UI
let modalConnectionState = 'disconnected';  // 'disconnected' | 'warming' | 'connected' | 'error'
let pendingMessage = null;  // Message cached while waiting for connection

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
 * Update model info display.
 * Reads model name from experiment config (window.state.experimentData.experimentConfig)
 * instead of making a redundant API call.
 */
function updateModelInfoUI() {
    const modelInfo = document.getElementById('model-info');
    if (!modelInfo) return;

    // Read from experiment config (already loaded by state.js)
    const experimentConfig = window.state?.experimentData?.experimentConfig;
    const appConfig = window.state.appConfig || {};
    let modelName = experimentConfig?.application_model
        || appConfig.defaults?.model
        || 'gemma-2-2b-it';

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
function updateConnectionStatusUI(isGenerating = false) {
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

// Getters for module-local state (read by live-chat.js)
function getInferenceMode() { return inferenceMode; }
function setInferenceMode(mode) { inferenceMode = mode; }
function getModalConnectionState() { return modalConnectionState; }
function setModalConnectionState(state) { modalConnectionState = state; }
function getPendingMessage() { return pendingMessage; }
function setPendingMessage(msg) { pendingMessage = msg; }

export {
    toggleInferenceMode,
    updateInferenceModeUI,
    updateModelInfoUI,
    warmupModal,
    updateConnectionStatusUI,
    getInferenceMode,
    setInferenceMode,
    getModalConnectionState,
    setModalConnectionState,
    getPendingMessage,
    setPendingMessage,
};

// Window binding for onclick in generated HTML
window.toggleInferenceMode = toggleInferenceMode;
