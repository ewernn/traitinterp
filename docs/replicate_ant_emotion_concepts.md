# Replicate our Llama 3.3 70B Emotion Concepts run

This guide reproduces the figures in [the emotion concepts replication finding](https://traitinterp.com/?tab=findings#emotion-concepts-replication) on the exact same Llama 3.3 70B Instruct setup.

For a BYOM replication on a different model with your own emotion set, see [`docs/create_ant_emotion_vectors.md`](create_ant_emotion_vectors.md).

---

## What's shipped where

| Artifact | Location |
|---|---|
| 171 emotion scenario datasets | `datasets/traits/ant_emotion_concepts/` (in repo) |
| Inference prompts (deep dive, numerical intensity, dissociation, etc.) | `datasets/inference/ant_emotion_concepts/` (in repo) |
| Pipeline code (extraction, inference, steering, analysis) | `core/` `utils/` `extraction/` `inference/` `analysis/` (in repo) |
| Stage 3–8 experiment scripts | `experiments/ant_emotion_concepts/scripts/` (in repo) |
| Experiment config (Llama 3.1 70B base + Llama 3.3 70B instruct variants) | `experiments/ant_emotion_concepts/config.json` (in repo) |
| Russell–Mehrabian PAD norms (for Fig 8) | GitHub release bundle |
| Pre-computed extraction activations + trait vectors (~294 MB, saves ~8 GPU-hr) | GitHub release bundle |
| Pre-computed results JSONs (~24 MB, saves ~3 GPU-hr) | GitHub release bundle |
| Our rendered figure PNGs | GitHub release bundle |

---

## Prereqs

- Python 3.11+
- `zstd` and `curl` installed locally (`brew install zstd` or `apt install zstd curl`) — used to fetch and unpack the bundle
- One GPU with ≥80 GB VRAM (H100, A100-80GB, MI300, or similar). 4-bit quantization via `bitsandbytes` is used throughout
- HuggingFace access to `meta-llama/Llama-3.1-70B` and `meta-llama/Llama-3.3-70B-Instruct` (`HF_TOKEN` env var)
- OpenAI API key if running the LLM-judge validation steps (`OPENAI_API_KEY`)

---

## 1. Clone + install

```bash
git clone https://github.com/ewernn/traitinterp
cd traitinterp
pip install -e .
export HF_TOKEN=<your-token>
```

## 2. Fetch the data bundle

The bundle ships pre-computed activations, vectors, results, and our rendered figures as a [GitHub release](https://github.com/ewernn/traitinterp/releases/tag/emotion-concepts-v1). Skipping it means spending ~8 GPU-hr on Step 3 to regenerate the vectors yourself.

```bash
curl -L https://github.com/ewernn/traitinterp/releases/download/emotion-concepts-v1/ant_emotion_concepts.tar.zst \
    | tar --zstd -xf - -C experiments/

# (optional) verify
curl -L https://github.com/ewernn/traitinterp/releases/download/emotion-concepts-v1/ant_emotion_concepts.tar.zst.sha256 \
    | shasum -a 256 -c
```

Browse the full file listing before downloading: [`ant_emotion_concepts.manifest.txt`](https://github.com/ewernn/traitinterp/releases/download/emotion-concepts-v1/ant_emotion_concepts.manifest.txt).

The bundle adds this data under your existing clone (scripts + config are already in the repo):

```
experiments/ant_emotion_concepts/
├── datasets/russell_mehrabian_norms.json        # PAD valence/arousal (Russell & Mehrabian 1977 transcription)
├── extraction/                                  # pre-computed trait vectors (171 emotions × 14 layers, ~294 MB)
├── results/                                     # pre-computed per-stage JSONs (~24 MB — delete to re-run from scratch)
└── paper_figures/ours/                          # our rendered Llama 3.3 70B figures
```

Note: the left-column "Sonnet 4.5" screenshots used in the viz-finding are reproduced from the paper under fair use; they are **not** included in the public bundle. See the [paper](https://www.anthropic.com/research/emotion-concepts-function-lm) directly for the originals.

## 3. (Optional) Extract vectors from scratch

Skip this section if you fetched the bundle in Step 2 — you already have the vectors. Run it if you want to regenerate from raw model outputs (~6–8 GPU-hr on an A100-80GB).

```bash
python extraction/run_extraction_pipeline.py \
    --experiment ant_emotion_concepts \
    --category ant_emotion_concepts \
    --only-stage 1,3 --save-activations --load-in-4bit --seed 42

python analysis/vectors/cross_trait_normalize.py \
    --experiment ant_emotion_concepts \
    --layers 1,7,13,19,25,31,37,43,49,55,61,67,73,79
```

171 emotions × 40 stories each at lightweight scale. For paper-scale (30× more stories), pass `--replication-level full`.

Outputs:
- `experiments/ant_emotion_concepts/extraction/.../mean_diff/layer*.pt` (raw per-emotion vectors)
- `experiments/ant_emotion_concepts/extraction/.../mean_diff+gm+pc50/layer*.pt` (grand-mean-centered, neutral-PC-denoised — what downstream stages consume)

## 4. Run per-figure analysis stages

```bash
# Stage 3 (geometry: cosine heatmap, UMAP/k-means, PCA, RSA, PAD correlation)
bash experiments/ant_emotion_concepts/scripts/run_stage3.sh

# Stage 4 (validation: Table 1 logit lens, Fig 2 implicit emotion, Fig 3 numerical intensity)
python experiments/ant_emotion_concepts/scripts/stage4_validation.py \
    --experiment ant_emotion_concepts --layer 49 --load-in-4bit

# Stage 5 (layer dynamics: Figs 10-15 dissociation, colon-predicts, context propagation, negation, binding)
python experiments/ant_emotion_concepts/scripts/stage5_layer_dynamics.py \
    --experiment ant_emotion_concepts --layer 49 --load-in-4bit

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

- **Stage 8 requires both base and instruct variants** in `config.json`. If you only have one variant, Stage 8 will fail.
- **Stage 6 (speaker probes)** was run but never plotted — results aren't in the bundle. The script is in `scripts/stage6_speaker_probes.py` if you want to run it yourself (~1 hr GPU).
- **Stage 7 (blackmail steering)** hit a null result — Llama 3.3 70B never blackmails under any steering condition, consistent with the paper's footnote 4 on final-snapshot Sonnet. The full coefficient sweep was gated out by a 10-rollout decision gate.
- **Stage 9 (deflection)** produces 900-dialogue pilot data that's too noisy to ship at our scale. Skip unless you're running `--replication-level full`.

---

## Not included

See the "What we did not include" dropdown at the bottom of the [viz finding](https://traitinterp.com/?tab=findings#emotion-concepts-replication) for the full 84-item list of paper figures that are proprietary, eval-awareness-gated, infrastructure-limited, pilot-only, or appendix-cosmetic.
