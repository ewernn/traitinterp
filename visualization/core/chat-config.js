// Shared constants for the live-chat / chat-inference subsystem.
//
// Defined once here and imported everywhere else so
//   - the experiment name doesn't drift between modules
//   - the default model variant ("application") doesn't drift between JS and
//     the Python backend (chat_inference.DEFAULT_MODEL_TYPE mirrors this).
// Both values flow into the /api/chat/* query string and POST body, so a
// mismatch silently points at the wrong backend/instance.

export const LIVE_CHAT_EXPERIMENT = 'starter';
export const CHAT_MODEL_TYPE = 'application';  // which model_variants key
export const CHAT_STATUS_POLL_MS = 2000;
