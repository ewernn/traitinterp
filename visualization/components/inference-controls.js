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
        // Don't auto-warmup — let the overlay/button handle it
        modalConnectionState = 'disconnected';
        updateConnectionStatusUI();
        // Re-render live chat to show warmup overlay
        if (window.state?.currentView === 'live-chat' && window.renderLiveChat) {
            window.renderLiveChat();
        }
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

    // Resolve model from experiment config schema (model_variants + defaults)
    const experimentConfig = window.state?.experimentData?.experimentConfig;
    const appConfig = window.state.appConfig || {};
    const variant = experimentConfig?.defaults?.application || 'instruct';
    let modelName = experimentConfig?.model_variants?.[variant]?.model
        || appConfig.defaults?.model
        || 'unknown';

    // Shorten model name for display (remove org prefix)
    const shortName = modelName.split('/').pop();
    modelInfo.textContent = shortName;
    modelInfo.title = modelName;  // Full name on hover
}

// Countdown state
let warmupCountdown = null;  // seconds remaining, or null
let countdownInterval = null;

/**
 * Warm up Modal GPU (called when switching to modal mode or on Overview page load).
 * Connects via SSE to /api/modal/warmup which sends an estimate first, then the
 * ready signal. Frontend runs a countdown from the estimate.
 */
function warmupModal() {
    if (inferenceMode !== 'modal') {
        modalConnectionState = 'connected';  // Local is always ready
        updateConnectionStatusUI();
        return;
    }

    // Don't restart if already warming
    if (modalConnectionState === 'warming') return;

    modalConnectionState = 'warming';
    warmupCountdown = null;
    updateConnectionStatusUI();

    const warmupStart = performance.now();
    const es = new EventSource('/api/modal/warmup');

    es.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.status === 'warming') {
            // Start countdown from server's estimate
            warmupCountdown = data.estimated_seconds;
            if (countdownInterval) clearInterval(countdownInterval);
            countdownInterval = setInterval(() => {
                if (warmupCountdown > 0) {
                    warmupCountdown--;
                    updateConnectionStatusUI();
                }
            }, 1000);
            updateConnectionStatusUI();
        }

        if (data.status === 'ready') {
            if (countdownInterval) { clearInterval(countdownInterval); countdownInterval = null; }
            warmupCountdown = null;
            modalConnectionState = 'connected';
            const elapsed = ((performance.now() - warmupStart) / 1000).toFixed(1);
            console.log(`[LiveChat] GPU warm in ${elapsed}s (server: ${data.actual_seconds}s, model: ${data.model})`);
            updateConnectionStatusUI();
            es.close();

            // Transition from landing page to full chat UI
            if (window.state?.currentView === 'live-chat' && window.renderLiveChat) {
                window.renderLiveChat();
            }

            // Flush any pending message
            if (pendingMessage) {
                const input = document.getElementById('chat-input');
                if (input) input.value = pendingMessage;
                pendingMessage = null;
            }
        }

        if (data.status === 'error') {
            if (countdownInterval) { clearInterval(countdownInterval); countdownInterval = null; }
            warmupCountdown = null;
            modalConnectionState = 'error';
            console.error('[LiveChat] Warmup failed:', data);
            updateConnectionStatusUI();
            es.close();
        }
    };

    es.onerror = () => {
        if (countdownInterval) { clearInterval(countdownInterval); countdownInterval = null; }
        warmupCountdown = null;
        modalConnectionState = 'error';
        console.error('[LiveChat] Warmup SSE error');
        updateConnectionStatusUI();
        es.close();
    };
}

/**
 * Update connection status UI
 */
function updateConnectionStatusUI(isGenerating = false) {
    const statusEl = document.getElementById('connection-status');
    if (!statusEl) return;

    // In local mode, always show ready
    if (inferenceMode === 'local') {
        statusEl.innerHTML = `
            <span class="status-dot connected"></span>
            <span class="status-text">Ready</span>
        `;
        return;
    }

    // Modal mode — show actual connection state
    const warmingText = warmupCountdown > 0
        ? `Waking GPU... ~${warmupCountdown}s`
        : 'Waking GPU...';

    const states = {
        disconnected: { dot: 'disconnected', text: 'Disconnected' },
        warming: { dot: 'warming', text: warmingText },
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
function getWarmupCountdown() { return warmupCountdown; }

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
    getWarmupCountdown,
};

// Window binding for onclick in generated HTML
window.toggleInferenceMode = toggleInferenceMode;
