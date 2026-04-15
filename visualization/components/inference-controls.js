// Inference Controls — unified chat-status polling + wake/unload/toggle
//
// The frontend keeps a single mutable chatStatus object that reflects the
// server's live state. All UI (connection dot, wake button, send guard, trait
// controls, unload button) reads from it instead of maintaining duplicate
// state. The only "intent" the frontend owns is `inferenceMode` — which
// backend the user wants; the rest comes from `/api/chat/status`.

const LIVE_CHAT_EXPERIMENT = 'live-chat';
const MODEL_TYPE = 'application';
const POLL_INTERVAL_MS = 2000;

// User's chosen backend. Separate from whether it's currently loaded.
let inferenceMode = 'local';  // 'local' | 'modal'

// Server-reported state, refreshed every POLL_INTERVAL_MS.
let chatStatus = {
    backend: 'local',
    model: null,
    ready: false,
    can_fit_locally: null,
};

// Loading progress state (only set during an active wake).
let wakeProgress = null;  // { estimated_seconds, elapsed_s } | null

let pollTimer = null;
let wakeInFlight = false;
let toggleInFlight = false;
// Monotonic generation counter for wake/load flows. Each toggleInferenceMode
// bumps this. An in-flight wake SSE whose generation no longer matches is
// silently dropped so a stale wake can't flip chatStatus.ready to true for a
// backend the user has since toggled away from.
let wakeGeneration = 0;
let pendingMessage = null;  // flushed when the backend becomes ready

// Listeners get called whenever chatStatus or wakeProgress changes. live-chat.js
// subscribes to drive layout (show/hide chart, wake button, trait controls).
const listeners = new Set();

function onChatStatusChange(fn) {
    listeners.add(fn);
    return () => listeners.delete(fn);
}

// Called exactly when (!prev.ready && chatStatus.ready). live-chat.js hooks in
// here to trigger Send after a user queued a prompt pre-warm.
const readyListeners = new Set();
function onChatReady(fn) {
    readyListeners.add(fn);
    return () => readyListeners.delete(fn);
}
function emitReady() {
    for (const fn of readyListeners) {
        try { fn(); } catch (e) { console.error(e); }
    }
}

// Called when the instance is explicitly dropped (unload or backend toggle).
// live-chat.js uses this to clear chart state so stale legend/metadata don't
// linger across model switches.
const unloadListeners = new Set();
function onChatUnload(fn) {
    unloadListeners.add(fn);
    return () => unloadListeners.delete(fn);
}
function emitUnload() {
    for (const fn of unloadListeners) {
        try { fn(); } catch (e) { console.error(e); }
    }
}

function emit() {
    for (const fn of listeners) {
        try { fn(getChatStatusSnapshot()); } catch (e) { console.error(e); }
    }
}

function getChatStatusSnapshot() {
    return {
        inferenceMode,
        chatStatus: { ...chatStatus },
        wakeProgress: wakeProgress ? { ...wakeProgress } : null,
        wakeInFlight,
    };
}

// ---------- polling ----------

async function fetchChatStatus() {
    // Auto-stop polling if the live-chat view is no longer mounted. Cheaper
    // than hooking into the router — if the DOM element is gone, we don't
    // need the dot anyway.
    if (pollTimer && !document.querySelector('.live-chat-view')) {
        stopChatStatusPolling();
        return;
    }
    try {
        const url = `/api/chat/status?experiment=${LIVE_CHAT_EXPERIMENT}&backend=${inferenceMode}&model_type=${MODEL_TYPE}`;
        const resp = await fetch(url);
        if (!resp.ok) return;
        const data = await resp.json();
        if (data.available === false) return;  // chat backend not available
        const prev = chatStatus;
        chatStatus = {
            backend: data.backend,
            model: data.model,
            ready: !!data.ready,
            can_fit_locally: data.can_fit_locally,
        };
        // If backend just flipped to ready, flush any queued message and tell
        // subscribers (live-chat.js listens for this to re-fire Send).
        if (!prev.ready && chatStatus.ready) {
            wakeProgress = null;
            if (pendingMessage) {
                const input = document.getElementById('chat-input');
                if (input && !input.value) input.value = pendingMessage;
                // Don't null pendingMessage here — the onChatReady listener
                // reads it to trigger Send, and clears it afterward.
            }
            emitReady();
        }
        emit();
        updateConnectionStatusUI();
    } catch (e) {
        console.warn('[chat-status] fetch failed:', e);
    }
}

function startChatStatusPolling() {
    if (pollTimer) return;
    fetchChatStatus();  // immediate hit on start
    pollTimer = setInterval(fetchChatStatus, POLL_INTERVAL_MS);
}

function stopChatStatusPolling() {
    if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
    }
}

// ---------- wake / unload ----------

async function wakeChatBackend() {
    if (wakeInFlight || chatStatus.ready) return;
    if (inferenceMode === 'local' && chatStatus.can_fit_locally === false) {
        console.warn('[wake] refusing local wake: model too large for local RAM');
        return;
    }
    wakeInFlight = true;
    wakeProgress = { estimated_seconds: null, elapsed_s: 0 };
    // Snapshot the wake generation + intended backend at invocation. If the
    // user toggles mid-wake, the counter advances and we drop this wake's
    // events. Prevents a stale "local ready" event from clobbering state
    // after the user switched to modal.
    const gen = wakeGeneration;
    const targetBackend = inferenceMode;
    emit();
    updateConnectionStatusUI();

    const start = performance.now();
    // Drive an elapsed-time ticker so the wake button can show "Waking... Ns"
    const ticker = setInterval(() => {
        if (wakeProgress) {
            wakeProgress.elapsed_s = Math.round((performance.now() - start) / 1000);
            emit();
            updateConnectionStatusUI();
        }
    }, 500);

    try {
        const resp = await fetch('/api/chat/wake', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                experiment: LIVE_CHAT_EXPERIMENT,
                backend: targetBackend,
                model_type: MODEL_TYPE,
            }),
        });
        if (!resp.ok || !resp.body) {
            throw new Error(`wake HTTP ${resp.status}`);
        }
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
                const evt = JSON.parse(line.slice(6));
                // Drop events from a stale wake (user toggled backends
                // while this wake was in flight).
                if (gen !== wakeGeneration) continue;
                if (evt.status === 'loading') {
                    wakeProgress = { estimated_seconds: evt.estimated_seconds, elapsed_s: 0 };
                    emit();
                    updateConnectionStatusUI();
                } else if (evt.status === 'ready') {
                    // Still belongs to the current intent? Update state.
                    chatStatus = { ...chatStatus, ready: true, backend: evt.backend, model: evt.model };
                    wakeProgress = null;
                    emit();
                    updateConnectionStatusUI();
                    emitReady();
                } else if (evt.status === 'error') {
                    console.error('[wake] backend error:', evt.error);
                    wakeProgress = null;
                    // Clear any queued message so an unrelated future ready
                    // transition doesn't resurrect a stale prompt.
                    pendingMessage = null;
                    emit();
                    updateConnectionStatusUI();
                }
            }
        }
    } catch (e) {
        console.error('[wake] failed:', e);
        wakeProgress = null;
    } finally {
        clearInterval(ticker);
        wakeInFlight = false;
        // Re-fetch to reconcile truth.
        fetchChatStatus();
    }
}

async function unloadChatBackend() {
    // Bump generation so any in-flight wake's events get dropped.
    wakeGeneration++;
    try {
        await fetch('/api/chat/unload', { method: 'POST' });
    } catch (e) {
        console.warn('[unload] failed:', e);
    }
    chatStatus = { ...chatStatus, ready: false };
    pendingMessage = null;  // don't auto-resurrect stale prompts
    emit();
    updateConnectionStatusUI();
    emitUnload();
    fetchChatStatus();
}

// ---------- mode toggle ----------

async function toggleInferenceMode() {
    if (toggleInFlight) return;  // ignore rapid double-clicks
    toggleInFlight = true;
    try {
        const newMode = inferenceMode === 'local' ? 'modal' : 'local';
        // Bump generation: any in-flight wake's late events are now stale.
        wakeGeneration++;
        // Best-effort unload of the previous backend so we don't pin two models.
        if (chatStatus.ready) {
            await unloadChatBackend();
        }
        inferenceMode = newMode;
        chatStatus = { ...chatStatus, backend: newMode, ready: false };
        wakeProgress = null;
        pendingMessage = null;
        emit();
        updateInferenceModeUI();
        updateConnectionStatusUI();
        emitUnload();  // chart/state from old backend is now invalid
        await fetchChatStatus();
    } finally {
        toggleInFlight = false;
    }
}

function updateInferenceModeUI() {
    const toggle = document.getElementById('inference-mode-toggle');
    if (toggle) toggle.checked = inferenceMode === 'modal';
    const label = document.getElementById('inference-mode-label');
    if (label) label.textContent = inferenceMode === 'modal' ? 'Modal GPU' : 'Local';
}

// ---------- connection dot ----------

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

    statusEl.innerHTML = `
        <span class="status-dot ${dot}"></span>
        <span class="status-text">${text}</span>
    `;
}

// ---------- getters for live-chat.js ----------

function getInferenceMode() { return inferenceMode; }
function setInferenceMode(mode) {
    inferenceMode = mode;
    chatStatus = { ...chatStatus, backend: mode };
    updateInferenceModeUI();
    emit();
}
function getChatStatus() { return { ...chatStatus }; }
function isChatReady() { return chatStatus.ready; }
function isWakeInFlight() { return wakeInFlight; }
function getPendingMessage() { return pendingMessage; }
function setPendingMessage(msg) { pendingMessage = msg; }
function getWakeProgress() { return wakeProgress ? { ...wakeProgress } : null; }

export {
    toggleInferenceMode,
    updateInferenceModeUI,
    updateConnectionStatusUI,
    wakeChatBackend,
    unloadChatBackend,
    startChatStatusPolling,
    stopChatStatusPolling,
    fetchChatStatus,
    onChatStatusChange,
    onChatReady,
    onChatUnload,
    getInferenceMode,
    setInferenceMode,
    getChatStatus,
    isChatReady,
    isWakeInFlight,
    getPendingMessage,
    setPendingMessage,
    getWakeProgress,
};

// Window bindings for inline onclick in template HTML
window.toggleInferenceMode = toggleInferenceMode;
window.wakeChatBackend = wakeChatBackend;
window.unloadChatBackend = unloadChatBackend;
