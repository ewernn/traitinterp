# Replicate our Llama 3.3 70B Emotion Concepts run

This guide reproduces the figures in [the emotion concepts replication finding](viz_findings/emotion-concepts-replication.md) on the exact same Llama 3.3 70B Instruct setup.

For a BYOM replication on a different model with your own emotion set, see `docs/create_ant_emotion_vectors.md` (coming).

---

## What's shipped where

| Artifact | Location |
|---|---|
| 171 emotion scenario datasets | `datasets/traits/ant_emotion_concepts/` (in repo) |
| Inference prompts (deep dive, numerical intensity, dissociation, etc.) | `datasets/inference/ant_emotion_concepts/` (in repo) |
| Pipeline code (extraction, inference, steering, analysis) | `core/` `utils/` `extraction/` `inference/` `analysis/` (in repo) |
| Stage 3–8 experiment scripts | R2 bundle |
| Experiment config (`config.json` with Llama 3.1 70B base + Llama 3.3 70B instruct variants) | R2 bundle |
| Russell–Mehrabian PAD norms (for Fig 8) | R2 bundle |
| Pre-computed extraction activations + trait vectors (optional — saves ~8 GPU-hr) | R2 bundle |
| Rendered figure PNGs | R2 bundle |

---

## Prereqs

- Python 3.11+
- One GPU with ≥80 GB VRAM (H100, A100-80GB, MI300, or similar). 4-bit quantization via `bitsandbytes` is used throughout.
- HuggingFace access to `meta-llama/Llama-3.1-70B` and `meta-llama/Llama-3.3-70B-Instruct` (`HF_TOKEN` env var)
- OpenAI API key if running the LLM-judge validation steps (`OPENAI_API_KEY`)
- `rclone` installed if you want to use the rclone fetch instead of curl

---

## 1. Clone + install

```bash
git clone https://github.com/ewernn/traitinterp
cd traitinterp
pip install -e .
export HF_TOKEN=<your-token>
```

## 2. Fetch the experiment bundle from R2

```bash
# Single-tarball fetch (recommended)
curl -L https://traitinterp.com/bundles/ant_emotion_concepts.tar.gz | tar xz
```

This writes:

```
experiments/ant_emotion_concepts/
├── config.json                                  # base + instruct variant declarations
├── scripts/                                     # stage3_*, stage4_*, stage5_*, stage6_*, stage7_*, stage8_*, stage9_*, shared.py
├── datasets/russell_mehrabian_norms.json        # PAD valence/arousal (Russell & Mehrabian 1977 transcription)
├── paper_figures/                               # rendered side-by-side PNGs
└── results/                                     # pre-computed JSONs (optional — delete to re-run from scratch)
```

## 3. Extract vectors (Stage 1 + cross-trait normalization)

```bash
python extraction/run_extraction_pipeline.py \
    --experiment ant_emotion_concepts \
    --category ant_emotion_concepts \
    --only-stage 1,3 --save-activations --load-in-4bit --seed 42

python analysis/vectors/cross_trait_normalize.py \
    --experiment ant_emotion_concepts \
    --layers 1,7,13,19,25,31,37,43,49,55,61,67,73,79
```

Budget: ~6–8 GPU-hr on an A100-80GB (171 emotions × 40 stories each at lightweight scale). For paper-scale (30× more stories), pass `--replication-level full`.

Outputs:
- `experiments/ant_emotion_concepts/extraction/.../mean_diff/layer*.pt` (raw per-emotion vectors)
- `experiments/ant_emotion_concepts/extraction/.../mean_diff+gm+pc50/layer*.pt` (grand-mean-centered, neutral-PC-denoised vectors — these are what downstream stages consume)

If you skipped the extraction step and used the R2 pre-computed activations + vectors, you can go straight to the next step.

## 4. Run per-figure analysis stages

```bash
# Stage 3 (geometry: cosine heatmap, UMAP/k-means, PCA, RSA, PAD correlation)
bash experiments/ant_emotion_concepts/scripts/run_stage3.sh

# Stage 4 (validation: Table 1 logit lens, Fig 2 implicit emotion, Fig 3 numerical intensity)
python experiments/ant_emotion_concepts/scripts/stage4_validation.py \
    --experiment ant_emotion_concepts --layer 49 --load-in-4bit

# Stage 5 (layer dynamics: Figs 10-15 dissociation, colon-predicts, context propagation, negation, binding)
python experiments/ant_emotion_concepts/scripts/stage5_layer_dynamics.py \
    --experiment ant_emotion_concepts --load-in-4bit

# Stage 8 (post-training: Figs 36-39 base-vs-instruct comparison + deep dives)
python experiments/ant_emotion_concepts/scripts/stage8_post_training.py \
    --experiment ant_emotion_concepts --layer 49 --load-in-4bit
```

Budget: ~1–3 GPU-hr for stages 4+5+8 combined (all reuse the extraction activations + vectors).

Outputs land under `experiments/ant_emotion_concepts/results/` (JSON + PNG per stage).

## 5. View the figures

Either:
- Open the rendered PNGs directly at `experiments/ant_emotion_concepts/paper_figures/ours/fig*_ours.png`
- Or serve the dashboard locally:

```bash
python visualization/serve.py  # http://localhost:8000/?tab=findings
```

---

## Known gotchas

- **`--layer 49` is mandatory** for Stage 4, Stage 8, and cross-trait normalization. `shared.py` has `DEFAULT_LAYER = 53` which is a stale default from an earlier iteration — pass `--layer 49` explicitly.
- **Stage 8 requires both base and instruct variants** in `config.json`. If you only have one variant, Stage 8 will fail.
- **Stage 6 (speaker probes)** is included in the bundle but not cited in the viz finding. You can delete `results/stage6/` (~173 MB) if you're not reproducing Figs 17–19.
- **Stage 9 (deflection)** produces 900-dialogue pilot data that's too noisy to ship at our scale. Skip unless you're running `--replication-level full`.

---

## Not included

See the "What we did not include" dropdown at the bottom of the [viz finding](viz_findings/emotion-concepts-replication.md) for the full 84-item list of paper figures that are proprietary, eval-awareness-gated, infrastructure-limited, pilot-only, or appendix-cosmetic.
