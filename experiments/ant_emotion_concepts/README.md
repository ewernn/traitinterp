# ant_emotion_concepts — Replication of Sofroniew et al. 2026

Replicates **"Emotion Concepts and their Function in a Large Language Model"** (Sofroniew et al., Anthropic, 2026) on **Llama 3.3 70B Instruct** using the traitinterp pipeline.

171 emotion vectors extracted via story-based elicitation, grand mean subtraction, and neutral PC denoising. 10 of 15 experimental paradigms fully replicated; 2 partially replicated; 1 blocked by model eval-awareness; 2 require infrastructure not yet built.

## Quick start

```bash
# 1. Extract 171 emotion vectors (GPU, ~4 hours)
python extraction/run_extraction_pipeline.py \
    --experiment ant_emotion_concepts --category ant_emotion_concepts \
    --only-stage 1,3 --save-activations --load-in-4bit --seed 42

# 2. Cross-trait normalization (CPU, ~2 min)
python analysis/vectors/cross_trait_normalize.py \
    --experiment ant_emotion_concepts \
    --layers 1,7,13,19,25,31,37,43,49,55,61,67,73,79 \
    --neutral-trait ant_emotion_concepts/_neutral

# 3. Geometry analysis (CPU, ~1 min)
bash experiments/ant_emotion_concepts/scripts/run_stage3.sh

# 4. Validation experiments (GPU, ~30 min)
python experiments/ant_emotion_concepts/scripts/stage4_validation.py \
    --experiment ant_emotion_concepts --layer 49 --load-in-4bit

# 5. Layer dynamics (GPU, ~1 hour)
python experiments/ant_emotion_concepts/scripts/stage5_layer_dynamics.py \
    --experiment ant_emotion_concepts --load-in-4bit

# 6. Post-training comparison (GPU, ~30 min)
python experiments/ant_emotion_concepts/scripts/stage8_post_training.py \
    --experiment ant_emotion_concepts --layer 49 --load-in-4bit
```

## What runs on the mainline pipeline

| Paper experiment | Pipeline primitive |
|---|---|
| Extract emotion vectors from contrasting stories | `extraction/run_extraction_pipeline.py` |
| Grand-mean + neutral-PC denoising | `analysis/vectors/cross_trait_normalize.py` |
| Cosine heatmap, UMAP clusters, PCA, RSA | `analysis/vectors/geometry.py` |
| Preference Elo | `analysis/vectors/preference_elo.py` |
| Capture activations at specific positions | `utils/capture_activations.capture_at_position` |
| Steering + coefficient search | `steering/run_steering_eval.py` |
| LLM-as-judge grading | `utils/judge.TraitJudge` |

## What's experiment-specific

| Script | What it does |
|---|---|
| `scripts/stage4_validation.py` | Logit lens, implicit emotion, numerical intensity, preference Elo, basic steering |
| `scripts/stage5_layer_dynamics.py` | Colon-predicts-response, context propagation, negation, person binding, dissociation |
| `scripts/stage6_speaker_probes.py` | 2-speaker dialogue extraction, per-turn probe geometry |
| `scripts/stage7_steering.py` | Blackmail + reward hacking steering sweeps |
| `scripts/stage8_post_training.py` | Base vs instruct comparison (Llama 3.1 70B → 3.3 70B Instruct) |
| `scripts/stage9_deflection.py` | Deflection probe extraction + steering |
| `scripts/dialogue_generation.py` | 2-speaker + deflection dialogue generation primitives |
| `scripts/shared.py` | Experiment-specific helpers (vector loading, result saving, blackmail scenario) |

## Model config

```json
{
  "model_variants": {
    "base": {"model": "meta-llama/Llama-3.1-70B"},
    "instruct": {"model": "meta-llama/Llama-3.3-70B-Instruct"}
  }
}
```

## Results

See [`ant_emotion_concepts_findings.md`](ant_emotion_concepts_findings.md) for the full Sonnet 4.5 vs Llama 3.3 70B comparison table.

Methodology reference: [`docs/other/emotion_concepts_methods.md`](../../docs/other/emotion_concepts_methods.md)
