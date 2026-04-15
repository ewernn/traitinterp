// Inference Controls — chat-status polling, wake/unload, mode toggle.
//
// Single source of truth is `chatStatus`, refreshed from /api/chat/status.
// All UI reads via getters; mutations emit a typed event that subscribers
// (live-chat.js) react to. The module owns the user's backend intent
// (`inferenceMode`); everything else reflects server state.

import { LIVE_CHAT_EXPERIMENT, CHAT_MODEL_TYPE, CHAT_STATUS_POLL_MS } from '../core/chat-config.js';

// --- state -------------------------------------------------------------

let inferenceMode = 'local';                  // user's backend choice
let chatStatus = { backend: 'local', model: null, ready: false, can_fit_locally: null };
let wakeProgress = null;                      // { estimated_seconds, elapsed_s } | null
let wakeInFlight = false;
let toggleInFlight = false;
let pollTimer = null;
let pendingMessage = null;                    // flushed to input on ready

// Bumped by toggleInferenceMode/unload. In-flight wake events whose gen no
// longer matches are dropped so a stale wake can't mark the wrong backend
// ready after the user has switched away.
let wakeGeneration = 0;

// --- event bus ---------------------------------------------------------
// Event types: 'change' (any state mutation), 'ready' (not-ready → ready),
// 'unload' (instance dropped via unload or toggle). 'change' listeners get
// the full state; 'ready'/'unload' are signals (no payload).

const subscribers = { change: new Set(), ready: new Set(), unload: new Set() };

function on(event, fn) {
    const set = subscribers[event];
    if (!set) throw new Error(`unknown event: ${event}`);
    set.add(fn);
    return () => set.delete(fn);
}

function emit(event) {
    const set = subscribers[event];
    if (!set) return;
    for (const fn of set) {
        try { fn(); } catch (e) { console.error(e); }
    }
    // Every mutation implies the connection dot may need to redraw.
    if (event === 'change') updateConnectionStatusUI();
}

// --- polling -----------------------------------------------------------

async function fetchChatStatus() {
    // Auto-stop once the view is unmounted; cheaper than hooking the router.
    if (pollTimer && !document.querySelector('.live-chat-view')) {
        stopChatStatusPolling();
        return;
    }
    const url = `/api/chat/status?experiment=${LIVE_CHAT_EXPERIMENT}&backend=${inferenceMode}&model_type=${CHAT_MODEL_TYPE}`;
    let resp;
    try {
        resp = await fetch(url);
    } catch (e) {
        console.warn('[chat-status] network error:', e);
        return;
    }
    if (!resp.ok) {
        console.warn(`[chat-status] HTTP ${resp.status}`);
        return;
    }
    const data = await resp.json();
    if (data.available === false) return;  // chat backend not shipped
    const wasReady = chatStatus.ready;
    chatStatus = {
        backend: data.backend,
        model: data.model,
        ready: !!data.ready,
        can_fit_locally: data.can_fit_locally,
    };
    if (!wasReady && chatStatus.ready) {
        wakeProgress = null;
        drainPendingIntoInput();
        emit('ready');
    }
    emit('change');
}

function startChatStatusPolling() {
    if (pollTimer) return;
    fetchChatStatus();  // immediate first hit
    pollTimer = setInterval(fetchChatStatus, CHAT_STATUS_POLL_MS);
}

function stopChatStatusPolling() {
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
}

function drainPendingIntoInput() {
    if (!pendingMessage) return;
    const input = document.getElementById('chat-input');
    if (input && !input.value) input.value = pendingMessage;
    // Keep pendingMessage set; the 'ready' listener consumes + clears it.
}

// --- wake / unload -----------------------------------------------------

async function wakeChatBackend() {
    if (wakeInFlight || chatStatus.ready) return;
    if (inferenceMode === 'local' && chatStatus.can_fit_locally === false) {
        console.warn('[wake] refusing local wake: model too large for local RAM');
        return;
    }
    wakeInFlight = true;
    wakeProgress = { estimated_seconds: null, elapsed_s: 0 };
    const gen = wakeGeneration;
    const targetBackend = inferenceMode;
    emit('change');

    const start = performance.now();
    const ticker = setInterval(() => {
        if (!wakeProgress) return;
        wakeProgress.elapsed_s = Math.round((performance.now() - start) / 1000);
        emit('change');
    }, 500);

    try {
        const resp = await fetch('/api/chat/wake', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                experiment: LIVE_CHAT_EXPERIMENT,
                backend: targetBackend,
                model_type: CHAT_MODEL_TYPE,
            }),
        });
        if (!resp.ok || !resp.body) throw new Error(`wake HTTP ${resp.status}`);
        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();
            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                if (gen !== wakeGeneration) continue;  // stale, user toggled away
                const evt = JSON.parse(line.slice(6));
                if (evt.status === 'loading') {
                    wakeProgress = { estimated_seconds: evt.estimated_seconds, elapsed_s: 0 };
                    emit('change');
                } else if (evt.status === 'ready') {
                    chatStatus = { ...chatStatus, ready: true, backend: evt.backend, model: evt.model };
                    wakeProgress = null;
                    drainPendingIntoInput();
                    emit('change');
                    emit('ready');
                } else if (evt.status === 'error') {
                    console.error('[wake] backend error:', evt.error);
                    wakeProgress = null;
                    pendingMessage = null;  // don't resurrect on a future unrelated ready
                    emit('change');
                }
            }
        }
    } catch (e) {
        console.error('[wake] failed:', e);
        wakeProgress = null;
    } finally {
        clearInterval(ticker);
        wakeInFlight = false;
        fetchChatStatus();  // reconcile truth
    }
}

async function unloadChatBackend() {
    wakeGeneration++;  // drop any in-flight wake's late events
    try {
        const resp = await fetch('/api/chat/unload', { method: 'POST' });
        if (!resp.ok) console.warn(`[unload] HTTP ${resp.status}`);
    } catch (e) {
        console.warn('[unload] network error:', e);
    }
    chatStatus = { ...chatStatus, ready: false };
    pendingMessage = null;
    emit('change');
    emit('unload');
    fetchChatStatus();
}

// --- mode toggle -------------------------------------------------------

async function toggleInferenceMode() {
    if (toggleInFlight) return;
    toggleInFlight = true;
    try {
        wakeGeneration++;
        if (chatStatus.ready) await unloadChatBackend();
        inferenceMode = inferenceMode === 'local' ? 'modal' : 'local';
        chatStatus = { ...chatStatus, backend: inferenceMode, ready: false };
        wakeProgress = null;
        pendingMessage = null;
        updateInferenceModeUI();
        emit('change');
        emit('unload');
        await fetchChatStatus();
    } finally {
        toggleInFlight = false;
    }
}

/**
 * Seed inferenceMode from app config (called once per mount, before polling).
 * Later toggles are driven by the user — never by this.
 */
function seedInferenceModeFromConfig(defaultBackend) {
    inferenceMode = defaultBackend === 'modal' ? 'modal' : 'local';
    chatStatus = { ...chatStatus, backend: inferenceMode };
    updateInferenceModeUI();
}

// --- DOM sync (dot + toggle label) -------------------------------------

function updateInferenceModeUI() {
    const toggle = document.getElementById('inference-mode-toggle');
    if (toggle) toggle.checked = inferenceMode === 'modal';
    const label = document.getElementById('inference-mode-label');
    if (label) label.textContent = inferenceMode === 'modal' ? 'Modal GPU' : 'Local';
}

function updateConnectionStatusUI() {
    const statusEl = document.getElementById('connection-status');
    if (!statusEl) return;
    let dot, text;
    if (wakeInFlight) {
        dot = 'warming';
        const est = wakeProgress?.estimated_seconds;
        const elapsed = wakeProgress?.elapsed_s || 0;
        text = est && elapsed < est ? `Loading... ~${est - elapsed}s` : `Loading... ${elapsed}s`;
    } else if (chatStatus.ready) {
        dot = 'connected';
        text = 'Ready';
    } else if (inferenceMode === 'local' && chatStatus.can_fit_locally === false) {
        dot = 'error';
        text = 'Too large for local';
    } else {
        dot = 'disconnected';
        text = 'Not loaded';
    }
    statusEl.innerHTML = `<span class="status-dot ${dot}"></span><span class="status-text">${text}</span>`;
}

// --- accessors ---------------------------------------------------------

function getInferenceMode() { return inferenceMode; }
function getChatStatus() { return { ...chatStatus }; }
function isChatReady() { return chatStatus.ready; }
function isWakeInFlight() { return wakeInFlight; }
function getWakeProgress() { return wakeProgress ? { ...wakeProgress } : null; }
function getPendingMessage() { return pendingMessage; }
function setPendingMessage(msg) { pendingMessage = msg; }

export {
    on,
    toggleInferenceMode,
    wakeChatBackend,
    unloadChatBackend,
    startChatStatusPolling,
    stopChatStatusPolling,
    fetchChatStatus,
    seedInferenceModeFromConfig,
    updateInferenceModeUI,
    updateConnectionStatusUI,
    getInferenceMode,
    getChatStatus,
    isChatReady,
    isWakeInFlight,
    getWakeProgress,
    getPendingMessage,
    setPendingMessage,
};

// Inline onclick handlers in the live-chat template call these by name.
window.toggleInferenceMode = toggleInferenceMode;
window.wakeChatBackend = wakeChatBackend;
window.unloadChatBackend = unloadChatBackend;
