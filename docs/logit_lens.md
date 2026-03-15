# Logit Lens

Shows "what would the model predict if it stopped at layer X?" by projecting intermediate layer states through the final unembedding matrix.

## Overview

**Logit lens** reveals how the model's prediction evolves across layers before final output. At each layer, the residual stream is projected to vocabulary space to show the most likely tokens.

**Why it's useful**:
- **Find commitment points** - When does the model lock into a decision?
- **Validate trait vectors** - See if vectors actually change predictions
- **Debug unexpected behavior** - Why did the model output X instead of Y?

## Usage

Extract logit lens from captured activations:

```bash
# First generate responses and capture raw activations
python inference/generate_responses.py \
    --experiment my_experiment \
    --prompt-set dynamic

python inference/capture_raw_activations.py \
    --experiment my_experiment \
    --prompt-set dynamic

# Then extract logit lens
python inference/extract_viz.py \
    --experiment my_experiment \
    --prompt-set dynamic \
    --logit-lens
```

This creates `analysis/per_token/{prompt_set}/{id}_logit_lens.json`.

## Output Format

Standalone JSON file with top-50 predictions per layer per token:

```json
{
  "tokens": ["<bos>", "How", ...],
  "n_prompt_tokens": 26,
  "n_response_tokens": 50,
  "predictions": [
    {
      "token_idx": 0,
      "actual_next_token": "How",
      "by_layer": [
        {
          "layer": 0,
          "top_k": [
            {"token": "the", "prob": 0.12},
            {"token": "I", "prob": 0.08},
            ...
          ],
          "actual_rank": 847,
          "actual_prob": 0.0001
        },
        ...
      ]
    }
  ]
}
```

**Key fields:**
- `actual_next_token` - The token that was actually generated
- `actual_rank` - Where the actual token ranked in predictions (1 = top prediction)
- `actual_prob` - Probability assigned to the actual token

## Visualization

In the **Layer Deep Dive** view, logit lens shows as an interactive panel:

1. **Default**: Shows layer 25 (final) predictions as horizontal bar chart
2. **Hover**: Move over attention heatmap rows to see that layer's predictions
3. **Actual token**: Shows rank and probability of the actual next token

This lets you see how predictions evolve from early layers (often wrong) to late layers (usually correct).

## Example Analysis

### Finding Commitment Points

```
Token 25 ("?") predicting token 26:
  Layer  0: top="?" (0.36), actual rank=4
  Layer  8: top="themſelves" (garbled), actual rank=5118
  Layer 16: top="\n\n" (0.999), actual rank=1  ← Committed!
  Layer 25: top="\n\n" (0.95), actual rank=1

Insight: Model commits to newline at layer 16
```

### Debugging Low-Confidence Predictions

```
Token 30 predicting token 31:
  Layer 25: top=" Climate" (0.79), actual=" Predictions" rank=5

The model wasn't confident - " Predictions" only ranked #5.
This explains any unexpected continuation.
```

## Storage

- ~4 MB per prompt (top-50 × 26 layers × ~75 tokens)
- All 26 layers included (no sampling)

## Implementation

The logit lens applies final RMSNorm + unembedding to the residual stream at each layer:

```python
normed = model.model.norm(residual)
logits = model.lm_head(normed)
probs = torch.softmax(logits, dim=-1)
```

## Vector Logit Lens

Apply logit lens to trait vectors (not residual stream) to see what tokens a direction "means".

### Pipeline Integration

Automatically runs as **stage 5** of the extraction pipeline:

```bash
python extraction/run_pipeline.py --experiment {exp} --traits {trait}
# Includes logit lens by default. Skip with --no-logitlens
```

Output: `experiments/{experiment}/extraction/{trait}/logit_lens.json`

### Standalone CLI

```bash
python analysis/vectors/logit_lens.py \
    --experiment {experiment} \
    --trait {category}/{trait} \
    --filter-common  # Filter to interpretable English tokens
```

### Visualization

In **Trait Extraction** view, the Token Decode section shows:
- Late (90% depth) layer projection
- Best method (probe > mean_diff > gradient)
- Top 5 toward/away tokens

### Example Output

Sycophancy vector:
```
Toward (+): "great", "excellent", "beautiful", "love", "amazing"
Away (-):   "either", "unless", "directly"
```

Refusal vector:
```
Toward (+): "sorry", "cannot", "nor", "laws", "legal"
Away (-):   "ready", "excellent", "happy", "easy", "good"
```

## References

- **TransformerLens logit lens**: https://transformerlensorg.github.io/TransformerLens/generated/demos/Logit_Lens.html
- **Original concept**: "Interpreting GPT: the Logit Lens" (nostalgebraist, 2020)
