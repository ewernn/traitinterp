# HuggingFace Chat Templates

Reference for chat template behavior across models.

## Key Concepts

### Generation Boundary

Generation starts **after** the assistant prefix added by `add_generation_prompt=True`:

```
Gemma-2-it:
  0: <bos>              ─┐
  1: <start_of_turn>     │
  2: user                │
  3: \n                  │ prompt (prefill)
  4-N: [user content]    │
  N+1: <end_of_turn>     │
  N+2: \n                │
  N+3: <start_of_turn>   │
  N+4: model             │
  N+5: \n               ─┘
  N+6+: [generation]    ─── response[0] starts here
```

**Detection method** (universal, works for all models):
```python
prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
generation_start = len(prompt_ids)
```

### Key Parameters

| Parameter | Purpose |
|-----------|---------|
| `add_generation_prompt=True` | Adds assistant prefix. Generation starts after. |
| `continue_final_message=True` | For prefilling — continues partial assistant message |
| `return_assistant_tokens_mask=True` | Returns binary mask — **requires `{% generation %}` in template** |
| `tools=[...]` | Pass functions or JSON schemas for tool calling |

### BOS Token Handling

Chat templates include BOS token. Our tokenization functions handle this automatically:

```python
# tokenize_batch() auto-detects BOS from text content
from utils.model import tokenize_batch, format_prompt

formatted = format_prompt("Hello", tokenizer, use_chat_template=True)
result = tokenize_batch([formatted], tokenizer)  # Auto-detects, no double BOS

# Validation catches mistakes
tokenize_batch([formatted], tokenizer, add_special_tokens=True)  # Raises ValueError!
```

**How auto-detection works:**
- Checks if text starts with `tokenizer.bos_token` (e.g., `<bos>` for Gemma, `<|begin_of_text|>` for Llama)
- If BOS present: `add_special_tokens=False` (don't add another)
- If no BOS: `add_special_tokens=True` (add one)
- Validates output for double-BOS and raises clear error if detected

This works for all models including Qwen (no BOS token) since the check short-circuits.

## Model Behaviors

### System Prompt Support

| Model | Support | Behavior |
|-------|---------|----------|
| Gemma-2-it | ❌ | Raises `TemplateError` |
| Llama-3.1-Instruct | ✅ | **Appends** to hardcoded default ("Cutting Knowledge Date...") |
| Qwen2.5-Instruct | ✅ | **Replaces** default ("You are Qwen...") |

**Llama's default is hardcoded in template** — cannot be removed without editing template.

### Generation Mask Support

Most models do **not** support `return_assistant_tokens_mask`:

| Model | `{% generation %}` in template |
|-------|-------------------------------|
| Gemma-2-it | ❌ |
| Llama-3.1-Instruct | ❌ |
| Qwen2.5-Instruct | ❌ |
| Qwen3-* | ✅ |

Use the "compare lengths" method instead for universal generation boundary detection.

### Tool Calling

| Model | Support |
|-------|---------|
| Gemma-2-it | ❌ Ignored |
| Llama-3.1-Instruct | ✅ |
| Qwen2.5-Instruct | ✅ |

Tool calling flow:
1. Pass `tools=` to `apply_chat_template()`
2. Model outputs JSON: `{"name": "func", "arguments": {...}}`
3. Execute tool, append result as `{"role": "tool", "content": "result"}`
4. Continue generation

## References

- [HuggingFace Chat Templates](https://huggingface.co/docs/transformers/en/chat_templating)
- [Tool Use Documentation](https://huggingface.co/docs/transformers/main/en/chat_extras)
- [Tokenizer API](https://huggingface.co/docs/transformers/en/main_classes/tokenizer)
