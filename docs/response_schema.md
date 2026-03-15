# Response Schema

Unified flat schema for response data across extraction, inference, and steering pipelines.

---

## Schema

```json
{
  "prompt": "...",
  "response": "...",
  "system_prompt": null,
  "tokens": ["<bos>", "user", ...],
  "token_ids": [2, 1645, ...],
  "prompt_end": 45,
  "inference_model": "google/gemma-2-2b-it",
  "prompt_note": "refusal_suppression",
  "capture_date": "2026-01-13T...",
  "tags": ["success"],
  "trait_score": null,
  "coherence_score": null
}
```

### Core Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `prompt` | string | **Yes** | User input text (includes chat template tokens) |
| `response` | string | **Yes** | Model output text |
| `system_prompt` | string \| null | No | System prompt if used, null otherwise |
| `tokens` | string[] \| null | No | Token strings for full sequence (prompt + response) |
| `token_ids` | int[] \| null | No | Token IDs for full sequence |
| `prompt_end` | int \| null | No | Token index where response starts (required if tokens present) |

### Optional Metadata Fields

| Field | Type | Description |
|-------|------|-------------|
| `inference_model` | string | Model used for generation (e.g., "google/gemma-2-2b-it") |
| `prompt_note` | string \| null | Category/note from prompt set (e.g., "roleplay", "refusal_suppression") |
| `capture_date` | string | ISO timestamp when response was captured |
| `tags` | string[] | User-defined tags (e.g., ["success"]) |

### Steering-Only Fields

| Field | Type | Description |
|-------|------|-------------|
| `trait_score` | float \| null | Trait projection score |
| `coherence_score` | float \| null | Coherence/quality score |

### Multi-Turn Rollout Fields

Used by `scripts/convert_rollout.py` for agent-interp-envs rollouts. The entire
conversation is stored as "prompt" with empty "response". `capture_raw_activations.py`
detects `response: ""` + `token_ids` present and uses stored IDs directly (skips
re-tokenization).

| Field | Type | Description |
|-------|------|-------------|
| `turn_boundaries` | object[] \| null | Token boundaries per message: `[{role, token_start, token_end, ...}]` |
| `sentence_boundaries` | object[] \| null | Per-sentence metadata: `[{sentence_num, token_start, token_end, cue_p}]` |
| `source` | object \| null | Provenance: `{type, environment, messages_path}` |

Turn boundary entries may include:
- `has_thinking` (bool) — assistant turn contains thinking/reasoning
- `has_tool_calls` (bool) — assistant turn contains tool calls
- `tool_names` (string[]) — names of tools called
- `tool_call_id` (string) — for tool result messages
- `tool_name` (string) — for tool result messages

Sentence boundary entries (for thought branches / unfaithful CoT analysis):
- `sentence_num` (int) — 0-indexed sentence position in CoT
- `token_start` (int) — response-relative start token
- `token_end` (int) — response-relative end token
- `cue_p` (float) — transplant resampling probability (0.0–1.0, from ground truth CSV)

### Future Fields

| Field | Type | Description |
|-------|------|-------------|
| `prefill_end` | int \| null | Token index where model generation starts (for prefilling) |

### Removed Fields

These fields are derivable from the file path or experiment config:
- `inference_experiment` → from path (`experiments/{experiment}/...`)
- `prompt_set` → from path (`.../responses/{prompt_set}/...`)
- `prompt_id` → from filename (`{id}.json`)
- `lora_adapter` → from model variant in experiment config

---

## Usage by Pipeline

### Extraction

Stores arrays of responses per trait. Tokens not stored (only counts in legacy format).

```json
[
  {
    "prompt": "The conversation started with...",
    "response": "I continued by explaining...",
    "system_prompt": null
  },
  ...
]
```

Sibling file: `metadata.json` with generation parameters.

### Inference

One file per prompt_id. Full tokenization and metadata stored (flat, no wrapper).

```json
{
  "prompt": "<bos><start_of_turn>user\nHow do I...",
  "response": "You can do this by...",
  "system_prompt": null,
  "tokens": ["<bos>", "<start_of_turn>", "user", ...],
  "token_ids": [2, 106, 1645, ...],
  "prompt_end": 45,
  "inference_model": "google/gemma-2-2b-it",
  "prompt_note": "roleplay",
  "capture_date": "2026-01-13T00:58:35.871205",
  "tags": ["success"]
}
```

### Steering

Stores arrays of responses per evaluation run. Includes scores.

```json
[
  {
    "prompt": "How does the X7 processor work?",
    "response": "I don't have information about...",
    "system_prompt": null,
    "trait_score": 0.003,
    "coherence_score": 50.0
  },
  ...
]
```

---

## Loading Responses

Use the adapter to handle both array and single-object files:

```python
def load_responses(path: Path) -> list[dict]:
    """Load responses from any format, return list."""
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]
```

```javascript
function loadResponses(data) {
    return Array.isArray(data) ? data : [data];
}
```

---

## Tokenization

Tokenization happens at generation time. Storage is optional.

| Pipeline | Tokenizes | Stores tokens |
|----------|-----------|---------------|
| Extraction | Yes (generation) | No |
| Inference | Yes (generation) | Yes |
| Steering | Yes (generation) | No |

To re-tokenize from stored text:
- Use same model/tokenizer
- `tokenize_batch()` auto-detects BOS from text content (no manual `add_special_tokens` needed)

---

## Activations

Activations are stored **separately** from responses:

- Binary `.pt` files (PyTorch tensors)
- Linked by prompt_id or response index
- Captured on-demand, not by default

Locations:
- Extraction: `activations/{position}/{component}/train_all_layers.pt`
- Inference: `raw/residual/{prompt_set}/{prompt_id}.pt`
- Steering: not stored (re-run inference if needed)

---

## Annotations

Annotations stored in sibling `*_annotations.json` files (e.g., `baseline.json` → `baseline_annotations.json`).

### Schema

```json
{
  "annotations": [
    {"idx": 0, "spans": [{"span": "72% of Americans"}, {"span": "The Godfather", "category": "movie"}]},
    {"idx": 5, "spans": [{"span": "vote in the election", "category": "voting"}], "note": "egregious example"}
  ]
}
```

**Structure:**
- `annotations`: Array of annotation entries (sparse - only include annotated responses)
- Each entry has `idx` (response index) and `spans` (array of span objects)

**Required fields:**
- `idx`: Response index (0-based)
- `spans`: Array of span objects, each with `span` key

**Optional fields on span:**
- `category`: Annotation type/label
- `intensity`: Severity (1-5)
- Any other field for your use case

**Optional fields on entry:**
- `note`: Note about this response
- `borderline`: Array of span objects (same format as `spans`) for plausible but ambiguous cases. Kept separate from `spans` so primary eval uses strict annotations only. Each borderline span should include `category` (which bias it's borderline for) and `note` (why it's ambiguous).
- Any other field

**Optional root fields:**
- `metadata`: Provenance info (`{"created": "2026-01-20", "author": "..."}`)
- `categories`: Category definitions (`{"movie": "Unprompted movie recommendations"}`)

### Conversion

Text spans are converted at runtime:
- **Frontend:** `spansToCharRanges(response, spans)` → character indices for highlighting
- **Backend:** `spans_to_token_ranges(response, spans, tokenizer)` → token indices for analysis

Converters in `utils/annotations.py` (Python) and `visualization/core/annotations.js` (JS).

### Example

Response file `baseline.json`:
```json
[
  {"prompt": "What caused the French Revolution?", "response": "The French Revolution (1789)..."},
  {"prompt": "Explain quantum computing", "response": "Quantum computing uses..."}
]
```

Annotation file `baseline_annotations.json`:
```json
{
  "annotations": [
    {"idx": 0, "spans": [{"span": "(population: 67 million)", "category": "population"}]}
  ]
}
```

---

## File Structure

Response files use:
- Arrays (extraction, steering)
- Single objects (inference)
