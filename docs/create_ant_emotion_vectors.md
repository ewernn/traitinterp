# Create Emotion-Concept Vectors on Your Own Model

Apply the Sofroniew et al. 2026 ("Emotion Concepts") extraction methodology to any HuggingFace instruct model via `--replication-level full`. You'll end up with one linear probe per emotion, ready for steering or activation projection.

This guide targets readers who want a **compact replication on their own model** (e.g. Qwen 2.5 7B, Llama 3.1 8B, Gemma 2 9B). To reproduce our exact Llama 3.3 70B figures, see [`replicate_ant_emotion_concepts.md`](replicate_ant_emotion_concepts.md) — it covers R2 bundle download and figure regeneration. This guide stops at "you have vectors"; pointers at the bottom show where to go next.

Prereqs: `pip install -e .` done (see [README](../README.md)), `HF_TOKEN` exported for gated models, a supported GPU (see scale table in Appendix B).

---

## Step 1 — Point the pipeline at your model

Create an experiment config. The model identifier can be any HuggingFace instruct model with a chat template.

```bash
mkdir -p experiments/my_ec
cat > experiments/my_ec/config.json <<'EOF'
{
  "defaults": {"extraction": "instruct", "application": "instruct"},
  "model_variants": {
    "instruct": {"model": "Qwen/Qwen2.5-7B-Instruct"}
  }
}
EOF
```

Swap the `model` value for whatever you want — **instruct / chat variant required**. Full mode sends the paper prompt via `system_prompt=` on a chat template; base models without a chat template can't apply it and will emit garbage. For int4 loading on smaller GPUs, pass `--load-in-4bit` to the pipeline later; no config change needed.

---

## Step 2 — Pick 20 emotions from the 174 pre-staged

`datasets/traits/ant_emotion_concepts/` ships with 174 emotion trait directories. For a compact replication that covers the valence × arousal plane, a reasonable starter 20:

```
happy, sad, excited, calm, afraid, angry, amazed, content,
anxious, relieved, grateful, frustrated, nostalgic, proud,
jealous, hopeful, embarrassed, lonely, inspired, disgusted
```

Pick anything you like. Full list: `ls datasets/traits/ant_emotion_concepts/`. Each trait directory ships with `positive.jsonl` only — full mode ignores it (generation uses `topics_100.json` + the template), so no scenario authoring needed for extraction.

**`definition.txt` is NOT shipped per emotion.** It's the rubric for the LLM-judge validation run at the end of extraction (`extraction_evaluation.json`). If absent, the pipeline falls back to a generic stub (`"The trait 'happy'"`), so extraction still completes but validation scores are degraded.

The category's `extraction_config.yaml` (at the category level, not per-emotion) already points at `prompts/story.txt` (paper's verbatim story prompt) and `topics_100.json` (paper's 100 topics). You inherit both.

---

## Step 3 — Smoke-test your setup

Before kicking off a multi-hour run, verify the full-mode pipeline works end-to-end on your model with a tiny scope:

```bash
python extraction/run_extraction_pipeline.py \
    --experiment my_ec \
    --traits ant_emotion_concepts/happy,ant_emotion_concepts/sad \
    --replication-level full \
    --topics 5 \
    --stories-per-batch 3 \
    --temperature 0.7 \
    --max-new-tokens 1024 \
    --load-in-4bit
```

This generates 5 topics × 3 stories × 2 emotions = 30 stories total. `max-new-tokens` is the budget **per batched call** (all 3 stories share it); paper stories average ~150 tokens each, so 1024 for a batch of 3 is comfortable. Inspect one:

```bash
python -c "
import json
r = json.loads(open('experiments/my_ec/extraction/ant_emotion_concepts/happy/instruct/responses/pos.json').read())
print(r[0]['response'])
"
```

You should see a short story that evokes the emotion without naming it. If the model produced gibberish or refused, something's wrong — fix before scaling up.

**Under-production warnings.** The pipeline prints `⚠ batched generation under-produced on N topics` when the model emitted fewer than `stories_per_batch` parseable stories. At N=12 the paper reported some emotions ("calm" in our Q-C3 test) collapse to duplicates; partial output is kept. Low under-production is expected; high (>30%) suggests the model can't follow the batched-generation format — consider `--stories-per-batch 6` or a different model.

---

## Step 4 — Run the compact extraction

Once the smoke test clears, scale up:

```bash
python extraction/run_extraction_pipeline.py \
    --experiment my_ec \
    --traits ant_emotion_concepts/happy,ant_emotion_concepts/sad,ant_emotion_concepts/excited,ant_emotion_concepts/calm,ant_emotion_concepts/afraid,ant_emotion_concepts/angry,ant_emotion_concepts/amazed,ant_emotion_concepts/content,ant_emotion_concepts/anxious,ant_emotion_concepts/relieved,ant_emotion_concepts/grateful,ant_emotion_concepts/frustrated,ant_emotion_concepts/nostalgic,ant_emotion_concepts/proud,ant_emotion_concepts/jealous,ant_emotion_concepts/hopeful,ant_emotion_concepts/embarrassed,ant_emotion_concepts/lonely,ant_emotion_concepts/inspired,ant_emotion_concepts/disgusted \
    --replication-level full \
    --temperature 0.7 \
    --max-new-tokens 2048 \
    --load-in-4bit
```

No `--topics` / `--stories-per-batch` → uses `extraction_config.yaml` defaults (100 topics × 12 stories = 1200 samples per emotion). `max-new-tokens 2048` budgets ~170 tokens per story in a batch of 12 (paper stories average ~150 tokens). Wall-clock: see Appendix B.

The pipeline prints a scope summary at startup — double-check it matches what you expect before you walk away.

---

## Step 5 — Find the vectors

One vector per (emotion, layer, method) under `experiments/my_ec/extraction/`. The position directory name is derived from `extraction_config.yaml`'s `position: "response[50:]"` → `response_50_`:

```
experiments/my_ec/extraction/ant_emotion_concepts/{emotion}/
└── instruct/
    └── vectors/response_50_/residual/mean_diff/
        └── layer{N}.pt
```

Each `layer{N}.pt` is a unit-normalized direction tensor of shape `[hidden_dim]`. Load with:

```python
import torch
v = torch.load("experiments/my_ec/extraction/ant_emotion_concepts/happy/instruct/vectors/response_50_/residual/mean_diff/layer20.pt")
print(v.shape, v.norm().item())  # torch.Size([3584]) 1.0
```

An experiment-level validation report lives at `experiments/my_ec/extraction/extraction_evaluation.json` — per-layer in-distribution metrics across all traits in the run. Use it to pick a layer; for this probing approach a good starting range is roughly 50–70% of `num_hidden_layers` (e.g. layers 14–22 for a 32-layer 7B model, layers 20–28 for a 40-layer 14B model, layers 40–56 for an 80-layer 70B model).

---

## What's next

- **Use the vectors outside this repo**: the `torch.load(...)` example in Step 5 produces plain `torch.Tensor` objects. Pickle them, ship them to HF Hub, load them in your own notebook — nothing ties them to this codebase.
- **Visualize** in the dashboard: `python visualization/serve.py` then open `localhost:8000`.
- **Regenerate paper figures exactly on Llama 3.3 70B**: [`replicate_ant_emotion_concepts.md`](replicate_ant_emotion_concepts.md).

---

## Appendix A — Bring your own emotion list

If the 174 shipped emotions don't cover what you want, author your own category:

```
datasets/traits/my_emotions/
├── extraction_config.yaml      # category-level (resolved by cascade)
├── topics_100.json             # list of 100 topic strings
├── prompts/story.txt           # verbatim paper template
└── {emotion_name}/             # one dir per emotion
    ├── positive.jsonl          # required: [{"prompt": ""}] suffices (full mode ignores content)
    └── definition.txt          # recommended: without it, validation uses a stub rubric
```

The `extraction_config.yaml` minimum for full-mode support:

```yaml
position: "response[50:]"
max_new_tokens: 2048
methods: ["mean_diff"]
temperature: 0.7
polarity: single

batched_story_template_file: prompts/story.txt
topics_file: topics_100.json
stories_per_batch: 12
```

Write `definition.txt` (HIGH/MID/LOW/Key rubric). See `datasets/traits/starter_traits/sycophancy/definition.txt` for an example. Emotion surface for the `{emotion}` template placeholder is derived from the directory name with `_` → `-` (override via `trait.yaml:emotion_surface` if needed).

`positive.jsonl` must exist (the loader requires *some* scenario file) but its content is unused in full mode — `[{"prompt": ""}]` is enough.

Generate topics any way you like — the paper uses Sonnet 4.5-generated topics about situations that *could* evoke any emotion (the emotion is fixed by the template, not the topic).

---

## Appendix B — Scale to your GPU

Per-emotion wall-clock at paper-default 100 topics × 12 stories, `max_new_tokens=2048`. Numbers assume single-GPU, not multi-GPU TP. Model-load cost is amortized across all traits in one pipeline invocation, so 20-emotion runs pay it once.

| Model                        | VRAM needed    | Flags             | ~Time/emotion | 20-emotion run |
|------------------------------|----------------|-------------------|---------------|----------------|
| Llama 3.1 8B / Qwen 2.5 7B   | ~16 GB (fp16)  | (none)            | ~30 min       | ~10 h          |
| Gemma 2 9B / Qwen 2.5 14B    | ~28 GB (fp16)  | (none)            | ~50 min       | ~16 h          |
| Llama 3.3 70B / Qwen 2.5 72B | ~40 GB (int4)  | `--load-in-4bit`  | ~6 h          | ~120 h         |

Estimates anchor on a Q-C3 empirical baseline of ~18.75s per story on 4-bit Llama 3.3 70B (100 topics × 12 stories = 1200 stories × 18.75s ≈ 6 h). Your model's tokenizer, chat-template length, and GPU define the actual number — use the smoke-test wall-clock × (100/smoke_topics) × (12/smoke_stories) as the more reliable predictor.

Time-constrained: `--topics 50 --stories-per-batch 6` gives 4× less data but still usable vectors (roughly what our lightweight mode does).

---

## Appendix C — Known failure modes

- **Mid-story truncation** (story cut off at exactly `max_new_tokens`): `max-new-tokens` is the budget for the WHOLE batched response (all N stories), so raise it proportionally with `--stories-per-batch`. Rule of thumb: budget ~200 tokens per story, e.g. 1024 for batches of 3, 2048 for batches of 12. Paper stories are "roughly one paragraph" (~150 tokens).
- **Chat-template quirks** with empty user turn: the full-mode pipeline sends paper text as a system prompt and an empty user turn (paper's role convention). Most Llama/Qwen/Gemma tokenizers handle this cleanly; unusual templates (some Mistral forks, custom finetunes) may emit a warning. Not a blocker — activations are still captured correctly.
- **Model refuses emotionally heavy prompts**: some aligned models refuse stories about grief, fear, anger. Check a failing story; if the response starts with "I cannot..." consider a less-aligned base variant or a system-prompt pre-amble.

Under-production behavior (warning message, fallback) is described at Step 3; check there first if the pipeline reports under-production.
