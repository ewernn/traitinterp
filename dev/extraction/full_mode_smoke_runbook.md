# Runbook: smoke-test `--replication-level full` on a real GPU

**Audience:** a fresh Claude Code session on a remote GPU instance. No prior context about Track C / the paper-faithful work assumed.

**Goal:** in ~30 minutes of GPU time, empirically verify that the three new `--replication-level full` paths produce sensible output on a real model end-to-end. Catches anything that mocked-generation tests can't (chat-template handling, real parser failures, generation-time crashes).

**Why this exists:** all three full-mode paths were developed and tested with mocked generation. Until a real model runs through them, we don't know whether (a) Llama-family chat templates handle the empty user turn cleanly, (b) the model actually produces N distinct items at the batched `Write {n_stories} different stories/dialogues...` prompts, or (c) parsing + downstream activation capture compose correctly on the new `pos.json` schema.

---

## Background you need to know

Three paper-faithful generation paths shipped recently, all opt-in via `--replication-level full`:

1. **Stage 1 stories**: full mode reads `datasets/traits/ant_emotion_concepts/prompts/story.txt`, sends `Write {n_stories} different stories...` per topic, parses `[story N]` blocks. Wired into `extraction/run_extraction_pipeline.py`.

2. **Stage 3 neutral dialogues**: full mode for the reference trait `ant_emotion_concepts/_neutral/` reads `prompts/neutral_dialogue.txt`, same batched pattern. Wired via `_neutral/extraction_config.yaml`'s `batched_story_template_file` override.

3. **Two-speaker emotional dialogues**: full mode for `stage6_speaker_probes.py --generate-dialogues` reads `prompts/two_speaker_dialogue.txt`, groups by `(person_emo, ai_emo, topic)`, post-hoc rewrites `Person:`/`AI:` → `Human:`/`Assistant:`.

The Q-C3 empirical test (earlier in the project) confirmed Llama 3.3 70B at int4 produces distinct stories at N=12 for ~3/4 emotions; `calm` collapsed with duplicate pairs. The parser handles duplicate-delimiter "list restart" failures by deduping to the first occurrence per index. **Same risk may apply to the new neutral and two-speaker paths** — that's what this smoke is checking.

Critical design fact: full mode lands the paper prompt in the **system role** with an **empty user turn**, per paper convention (`prompt=""` passed to `format_prompt`, paper text in `system_prompt=`). Most Llama / Qwen / Gemma chat templates handle this, but unusual templates may emit a warning. Watch for it.

---

## Prerequisites

1. Repo cloned at the standard path. `pip install -e .` already done.
2. `HF_TOKEN` exported for gated models.
3. A GPU with at least:
   - ~16 GB for 7B fp16, OR
   - ~40 GB for 70B int4 (the experiment's default Llama 3.3 70B; pass `--load-in-4bit`)
4. The experiment config `experiments/ant_emotion_concepts/config.json` already exists and points at the model you want to use. Don't create a new experiment config; reuse this one.

---

## Step 1: stage-1 stories smoke (~10 min)

```bash
cd /path/to/traitinterp

python extraction/run_extraction_pipeline.py \
    --experiment ant_emotion_concepts \
    --traits ant_emotion_concepts/happy,ant_emotion_concepts/sad \
    --replication-level full \
    --topics 5 \
    --stories-per-batch 12 \
    --temperature 0.7 \
    --max-new-tokens 2048 \
    --load-in-4bit \
    --force
```

`--force` is required because `pos.json` likely already exists from prior lightweight-mode runs; without it the pipeline silently skips.

**Why N=12 (not 3 or 6):** the May 2026 smoke established that the paper's diversity instruction is calibrated for N≈12 and the model collapses at small N. Observed under-production rates: N=3 → 40%, N=6 → 0%, N=12 → 0%. Default to N=12 (paper scale) to validate the realistic path.

### What to look for

A startup banner like:

```
========================================================================
  --replication-level=full : paper-verbatim prompts + batched generation
========================================================================
  Traits               : 2
  Stories per batch    : 3
  ...
```

If you don't see this banner, the flag didn't thread through. Stop and investigate.

Then per-trait progress, ending with:

```
    positive: 30 responses (5 topics × up to 3 stories)
```

`30` = `2 traits × 5 topics × 3 stories`. If you see something materially smaller like `15` or `20`, look one line above for an under-production warning.

**Expected counts at the default N=12:** 2 traits × 5 topics × 12 stories = 120 records combined. The May 2026 smoke confirmed N=6 and N=12 produce 100% (zero warnings), so N=12 is the safe default. N=3 was a hostile setup that collapsed at ~40% — don't use it unless you specifically want to test under-production handling.

### Verify output

```bash
python -c "
import json
for emo in ['happy', 'sad']:
    p = f'experiments/ant_emotion_concepts/extraction/ant_emotion_concepts/{emo}/instruct/responses/pos.json'
    r = json.loads(open(p).read())
    print(f'{emo}: {len(r)} records')
    print(f'  schema: {sorted(r[0].keys())}')
    print(f'  prompt[:60]: {r[0][\"prompt\"][:60]!r}')
    print(f'  system_prompt[:80]: {r[0][\"system_prompt\"][:80]!r}')
    print(f'  response[:100]: {r[0][\"response\"][:100]!r}')
    print()
"
```

Expected:
- `schema` includes `story_idx` and `topic` (full-mode extras) on top of `prompt`, `response`, `system_prompt`
- `prompt` is the **empty string** `''` (paper-role-fidelity: user turn empty)
- `system_prompt` starts with `'Write 3 different stories based on the following premise.\n\nTopic: ...'`
- `response` is a story (a sentence or two) that evokes the target emotion without naming it

If `prompt` is non-empty or `system_prompt` is null, role fidelity broke.

### Failure mode: chat template doesn't handle empty user turn

If you see a warning like `Model doesn't support system role, ignoring system_prompt` in the logs, or the `pos.json` `response` field is empty / nonsense, the chat template is rejecting the empty-user-turn shape. Note the exact tokenizer warning and the model name; that's the single fact the main session needs to fix it.

---

## Step 2: stage-3 neutral-dialogue smoke (~5 min)

```bash
python extraction/run_extraction_pipeline.py \
    --experiment ant_emotion_concepts \
    --traits ant_emotion_concepts/_neutral \
    --replication-level full \
    --topics 5 \
    --stories-per-batch 3 \
    --temperature 0.7 \
    --max-new-tokens 1024 \
    --load-in-4bit \
    --force
```

`_neutral` is a leading-underscore "reference trait" — `discover_traits()` normally excludes it, but passing it explicitly via `--traits` works fine.

### What to look for

```
    positive: 15 responses (5 topics × up to 3 stories)
```

15 = `1 trait × 5 topics × 3 stories`.

### Verify output

```bash
python -c "
import json
p = 'experiments/ant_emotion_concepts/extraction/ant_emotion_concepts/_neutral/instruct/responses/pos.json'
r = json.loads(open(p).read())
print(f'records: {len(r)}')
print(f'system_prompt[:120]: {r[0][\"system_prompt\"][:120]!r}')
print(f'response[:200]: {r[0][\"response\"][:200]!r}')
"
```

Expected:
- `system_prompt` starts with `'Write 3 different dialogues based on the following topic.'` (note: "dialogues", not "stories" — this is what proves the neutral template overrode the category-level story template)
- `response` looks like a Person/AI dialogue. Each turn starts with `Person:` or `AI:` (paper labels, not post-hoc converted in this pipeline — only the two-speaker dialogue script does the conversion)

If `system_prompt` starts with `'Write 3 different stories'`, the `_neutral/extraction_config.yaml` override didn't apply. The override is `batched_story_template_file: ../prompts/neutral_dialogue.txt` and should resolve via the `*_file` path-resolution machinery in `utils/traits.py`.

---

## Step 3: two-speaker dialogue smoke (~15 min)

This one exercises `stage6_speaker_probes.py` end-to-end. It needs the GPU loaded for both dialogue generation AND the downstream speaker-probe extraction, so it takes longer.

For a **quick generation-only test** that skips the probe-extraction sub-experiments, use the `--sub-experiments` flag with an empty-ish list or just `geometry` (cheapest sub-experiment):

```bash
python experiments/ant_emotion_concepts/scripts/stage6_speaker_probes.py \
    --experiment ant_emotion_concepts \
    --load-in-4bit \
    --generate-dialogues \
    --n-dialogues 8 \
    --emotions happy,sad \
    --replication-level full \
    --dialogue-topics "a chance encounter,a job interview" \
    --sub-experiments geometry \
    --n-layers-sample 1
```

### What to look for

```
Generating dialogues (replication_level=full)...
  [full mode] Generating 8 dialogues across N unique (person_emotion, ai_emotion, topic) cells (max_new_tokens=2048)...
```

`N` should be ≤ 4 (since 2 emotions × 2 speakers = 4 possible emotion pairs, × 2 topics = up to 8, but seed=42 will collide on some).

### Verify output

The dialogues file is written somewhere in `experiments/ant_emotion_concepts/...stage6.../dialogues.json` — look at the printed `Saved N dialogues to <path>` line. Then:

```bash
python -c "
import json, sys
# adjust path to the printed Saved... path
p = '<paste the saved path from stage6 output>'
d = json.load(open(p))['dialogues']
print(f'dialogues: {len(d)}')
print(f'first dialogue text[:300]:')
print(d[0]['text'][:300])
print()
print(f'Person:/AI: in text? {\"Person:\" in d[0][\"text\"] or \"AI:\" in d[0][\"text\"]}')
print(f'Human:/Assistant: in text? {\"Human:\" in d[0][\"text\"] or \"Assistant:\" in d[0][\"text\"]}')
"
```

Expected:
- `Person:`/`AI:` should NOT be in `text` — the post-hoc rewriter converts them.
- `Human:`/`Assistant:` should be present.

If `Person:` or `AI:` is still in the text, the post-hoc converter `_person_ai_to_human_assistant` didn't fire or has a regex bug.

---

## Report back

Send the main session three numbers:

1. **Stage 1**: number of records in `happy/pos.json` and `sad/pos.json` (expected ~30 combined). And the first 100 chars of `happy`'s first `response`.
2. **Stage 3**: number of records in `_neutral/pos.json` (expected ~15). And the first 100 chars of `_neutral`'s first `response`.
3. **Stage 6**: number of dialogues generated, and a yes/no on whether `Person:` appears anywhere in the dialogue text (it shouldn't).

Plus: any errors, any "Model doesn't support system role" warnings, any noticeable under-production warnings beyond one.

---

## Cleanup (optional)

If the smoke ran on `--experiment ant_emotion_concepts` and you don't want the smoke-mode `pos.json` lingering under those two emotion dirs:

```bash
# Restore the lightweight pos.json by re-running without --replication-level=full
python extraction/run_extraction_pipeline.py \
    --experiment ant_emotion_concepts \
    --traits ant_emotion_concepts/happy,ant_emotion_concepts/sad \
    --force
```

This regenerates `pos.json` in lightweight mode, the default that produced the originally-shipped vectors.

`_neutral/pos.json` is similar — re-run without `--replication-level=full` to get lightweight back.

If you ran step 3 on a fresh experiment, just delete that experiment's dir.

---

## If something explodes mid-run

- **CUDA OOM**: drop `--max-new-tokens` to 512 and `--stories-per-batch` to 2. Or switch to a smaller model by editing the experiment config.
- **`FileNotFoundError` on `prompts/story.txt` or `prompts/neutral_dialogue.txt`**: these files should ship with the dataset. `ls datasets/traits/ant_emotion_concepts/prompts/` should show three `.txt` files (story, neutral_dialogue, two_speaker_dialogue). If missing, the dataset isn't fully synced; `git pull` and check the latest commit.
- **`Multiple scenario formats found for positive`**: collision between `positive.json`/`positive.jsonl`/`positive.txt` — one of the per-emotion trait dirs has more than one. `ls datasets/traits/ant_emotion_concepts/{the_trait_name}/` to find the duplicate; keep `positive.jsonl`.
- **Parse warnings about under-production above 30% of cells**: the model is failing to follow the batched-list format. Try `--stories-per-batch 6` or `--stories-per-batch 4`. If still bad, the model can't replicate paper methodology cleanly — note this for the main session to consider model choice.
