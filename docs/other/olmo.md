# OLMo Reference

Quick reference for AI2's OLMo models — architecture, data, checkpoints, and what matters for our research.

Source: [OLMo 3 paper (arXiv:2512.13961)](https://arxiv.org/abs/2512.13961), Dec 2025.

---

## Models

| | OLMo 3 7B | OLMo 3 32B | OLMo 2 7B (existing) |
|---|---|---|---|
| Layers | 32 | 64 | 32 |
| Hidden dim (d_model) | 4096 | 5120 | 4096 |
| Q heads | 32 | 40 | 32 |
| KV heads | 32 (MHA) | 8 (GQA) | 32 |
| Activation | SwiGLU | SwiGLU | SwiGLU |
| Norm | RMSNorm, **post-norm** | RMSNorm, **post-norm** | RMSNorm, post-norm |
| QKV norm | **QK-Norm** | **QK-Norm** | None |
| Attention | 3/4 layers SWA (4096 window), 1/4 full | same | Full attention |
| Context (pretrain) | 8192 | 8192 | 4096 |
| Context (final) | 65,536 (YaRN) | 65,536 (YaRN) | 4096 |
| RoPE θ | 5×10⁵ | 5×10⁵ | 5×10⁵ |
| Tokenizer | cl100k-derived (same as OLMo 2) | same | same |
| Vocab | 100,278 | same | same |
| License | Apache 2.0 | Apache 2.0 | Apache 2.0 |

**Post-norm** means RMSNorm is applied to sublayer *outputs* (after attention/MLP), not inputs. This differs from Llama's pre-norm. Affects where contribution hooks should attach — `post_attention_layernorm` is the right hook point (same as Gemma 2).

**QK-Norm** is RMSNorm on queries and keys before attention computation. New in OLMo 3.

**Sliding Window Attention (SWA):** Every 4th layer uses full attention; the other 3 use a 4096-token window. Last layer always uses full attention. During long-context extension, YaRN scaling applies only to full-attention layers.

## HuggingFace Model IDs

### OLMo 3 — Final Models

| Variant | Model ID |
|---|---|
| Base 7B | `allenai/Olmo-3-1025-7B` |
| Base 32B | `allenai/Olmo-3-1125-32B` |
| Think 7B | `allenai/Olmo-3-7B-Think` |
| Think 32B | `allenai/Olmo-3-32B-Think` |
| Instruct 7B | `allenai/Olmo-3-7B-Instruct` |
| Instruct 32B (3.1) | `allenai/Olmo-3.1-32B-Instruct` |

### OLMo 3 — Intermediate Post-Training Checkpoints

Each post-training stage released separately. Useful for studying what SFT/DPO/RL each contribute.

| Variant | Model ID |
|---|---|
| Think 7B SFT | `allenai/Olmo-3-7B-Think-SFT` |
| Think 7B DPO | `allenai/Olmo-3-7B-Think-DPO` |
| Think 32B SFT | `allenai/Olmo-3-32B-Think-SFT` |
| Think 32B DPO | `allenai/Olmo-3-32B-Think-DPO` |
| Instruct 7B SFT | `allenai/Olmo-3-7B-Instruct-SFT` |
| Instruct 7B DPO | `allenai/Olmo-3-7B-Instruct-DPO` |
| Instruct 32B SFT (3.1) | `allenai/Olmo-3.1-32B-Instruct-SFT` |
| Instruct 32B DPO (3.1) | `allenai/Olmo-3.1-32B-Instruct-DPO` |

### OLMo 3 — RL-Zero (base → RL, no SFT/DPO)

| Variant | Model ID |
|---|---|
| 7B Math | `allenai/Olmo-3-7B-RL-Zero-Math` |
| 7B Code | `allenai/Olmo-3-7B-RL-Zero-Code` |
| 7B IF | `allenai/Olmo-3-7B-RL-Zero-IF` |
| 7B General | `allenai/Olmo-3-7B-RL-Zero-General` |
| 7B Math (3.1) | `allenai/Olmo-3.1-7B-RL-Zero-Math` |
| 7B Code (3.1) | `allenai/Olmo-3.1-7B-RL-Zero-Code` |

### OLMo 2 (existing in codebase)

| Variant | Model ID |
|---|---|
| 7B Instruct | `allenai/OLMo-2-1124-7B-Instruct` |

## Training Data — Dolma 3

Everything is downloadable on HuggingFace under ODC-BY license.

### Pretraining: Dolma 3 Mix (5.93T tokens)

| Source | Type | Tokens | % |
|---|---|---|---|
| Common Crawl | Web pages | 4.51T | 76.1% |
| olmOCR science PDFs | Academic documents | 805B | 13.6% |
| Stack-Edu (rebalanced) | GitHub code | 409B | 6.9% |
| FineMath 3+ | Math web pages | 152B | 2.6% |
| arXiv | Papers (LaTeX) | 50.8B | 0.9% |
| Wikipedia & Wikibooks | Encyclopedic | 2.5B | 0.04% |

Pool is 9.3T tokens. Final mix is 5.93T after quality-aware upsampling (top 5% repeated up to 7x, bottom 40% discarded per topic). Aggressively deduplicated: exact → MinHash fuzzy → suffix array, 75% document reduction.

Quality classification: fastText classifier trained on OpenHermes-2.5 + ELI5 + UltraChat-200k + WildChat-1M as positive, DCLM-RefinedWeb as negative. Topic classification: 24 WebOrganizer topics.

### Midtraining: Dolma 3 Dolmino Mix (100B tokens)

| Category | Key Sources | % |
|---|---|---|
| Math (synth) | Dolmino Math, CraneMath, MegaMatt, TinyMATH | ~19% |
| Code | StackEdu FIM, CraneCode | ~20% |
| QA (synth) | Reddit-to-Flashcards, Wiki-to-RCQA, Nemotron Synth QA | ~14% |
| Thinking (synth) | QWQ traces, Gemini traces, meta-reasoning, OpenThoughts2 | ~6% |
| Instruction | Tulu 3 SFT, Dolmino 1 Flan | ~6% |
| Web + PDFs (HQ) | Common Crawl HQ subset, olmOCR HQ, STEM-heavy crawl | ~33% |

Thinking traces + instruction data included intentionally to lay groundwork for post-training. Integration tests confirmed this helps even base model performance.

### Long-context: Dolma 3 Longmino Mix (50B / 100B tokens)

Primarily olmOCR science PDFs (22.3M docs above 8K tokens, 640B tokens total). 66% of mix is midtraining data; 34% is long documents. Synthetic augmentation: context-window extension (CWE) and retrieval-extended (REX) versions of PDFs.

### Data HuggingFace Datasets

| Dataset | HF ID |
|---|---|
| Pretrain pool (9T) | `allenai/dolma3_pool` |
| Pretrain mix (6T) | `allenai/dolma3_mix` |
| Pretrain sample (150B) | `allenai/dolma3_sample_mix` |
| Midtrain mix (100B) | `allenai/dolma3_dolmino_mix-100B-1125` |
| Midtrain sample (10B) | `allenai/dolma3_dolmino_sample_mix` |
| Long-context mix | `allenai/dolma3_longmino_mix` |

## Post-Training — Dolci

Three-stage recipe: SFT → DPO → RLVR. Applied separately for Think and Instruct paths.

### Model variant summary

All three variants start from the same Base model. They diverge at post-training.

| | Think | Instruct | RL-Zero |
|---|---|---|---|
| **What it does** | Extended reasoning with `<thinking>` traces before answering | Direct responses, no thinking traces | Base → RL only, no SFT/DPO |
| **Pipeline** | Base → Think SFT → Think DPO → Think RLVR | Base → Think SFT → Instruct SFT → Instruct DPO → Instruct RLVR | Base → RLVR |
| **Output style** | Long thinking trace + final answer | Concise direct answer | Varies by domain |
| **Training data focus** | Math/code reasoning traces (QwQ-32B generated, 32K tokens) | Chat, function calling, safety, multilingual (2.15M examples) | Domain-specific verifiable prompts |
| **Best for our research** | Studying reasoning traces, connects to `jan24-implicit_cot_monitoring` | General trait work — behaves like a normal chat model, clean for extraction/steering/monitoring | Clean RL testbed — isolates what RL alone does to trait geometry without SFT/DPO confounds |

Key detail: **Instruct starts from Think SFT**, not from Base directly. It gets a "warm start" from the Think SFT checkpoint, then runs its own SFT on top. This means Instruct has seen reasoning traces during training even though it doesn't output them — relevant for whether reasoning-related trait directions exist in the model.

For most trait vector work, **use Instruct** — it's the closest to how other chat models (Gemma IT, Llama Instruct) behave and avoids the complication of long thinking traces in extraction/monitoring.

### Think path (reasoning models)

**SFT:** Math prompts from OpenThoughts3 + SYNTHETIC-2, code from LiveCodeBench + Codeforces, precise IF from IFEval + IFBench, science from OpenThoughts3. Completions generated by QwQ-32B with 32K token thinking traces. ~326K examples total.

**DPO:** Delta Learning heuristic — chosen from large model (Qwen3 32B), rejected from small model (Qwen3 0.6B). ~130K examples.

**RLVR (OlmoRL):** GRPO-based. Domains: math (verifiable), code (verifiable via execution), instruction following (verifiable via format checking), general chat (LLM-judge). Key finding: DPO before RL produces better results than RL directly after SFT. OLMo 3.1 Think 32B trained for 2300 steps (vs 750 for 3.0) — performance had not saturated.

### Instruct path (chat models)

Trained starting from Think SFT checkpoint ("warm start" — significantly helps). No thinking traces in output.

**SFT:** ~2.15M examples. Includes function calling data (SimFC simulated + real MCP interactions), WildChat, math, code, safety, science, multilingual (Aya), Flan.

**DPO:** ~260K examples. Delta-learning heuristic pairs + GPT-judged pairs. Multi-turn synthetic conversations. Length control: filter chat/multi-turn subsets to limit chosen-rejected length difference to 100 tokens.

**RLVR:** Same OlmoRL framework as Think, focused on maintaining capabilities while preserving brevity.

### Post-Training Data HuggingFace Datasets

| Dataset | HF ID |
|---|---|
| Think SFT 7B | `allenai/Dolci-Think-SFT-7B` |
| Think DPO 7B | `allenai/Dolci-Think-DPO-7B` |
| Think RL 7B | `allenai/Dolci-Think-RL-7B` |
| Think SFT 32B | `allenai/Dolci-Think-SFT-32B` |
| Think DPO 32B | `allenai/Dolci-Think-DPO-32B` |
| Think RL 32B | `allenai/Dolci-Think-RL-32B` |
| Instruct SFT | `allenai/Dolci-Instruct-SFT` |
| Instruct DPO | `allenai/Dolci-Instruct-DPO` |
| Instruct RL | `allenai/Dolci-Instruct-RL` |
| RL-Zero (various) | `allenai/Dolci-RL-Zero-{Math,Code,IF,General}-7B` |

## Checkpoints

Every training stage has intermediate checkpoints accessible via `revision=` on HuggingFace.

```python
from transformers import AutoModelForCausalLM

# Specific stage checkpoint
model = AutoModelForCausalLM.from_pretrained(
    "allenai/Olmo-3-1125-32B",
    revision="stage1-step10000"  # pretraining checkpoint
)
```

| Stage | Naming Convention | Notes |
|---|---|---|
| Pretrain | `stage1-stepXXXXX` | 7B: every few thousand steps over 5.93T tokens |
| Midtrain | `stage2-ingredient{N}-stepXXX` | Two runs (different seeds) then merged for 32B |
| Long-context | `stage3-stepXXX` | 50B tokens (7B) or 100B (32B) |
| Post-training | SFT/DPO/RL checkpoints | Available for both Think and Instruct |

32B midtraining used model souping: two independent runs merged → ~1 point improvement on MC-STEM, ~3 points on Math. Long-context also souped: last 3 checkpoints averaged.

## Code Repositories

| Repo | Purpose |
|---|---|
| [allenai/OLMo-core](https://github.com/allenai/OLMo-core) | Pretraining, midtraining, long-context, SFT |
| [allenai/open-instruct](https://github.com/allenai/open-instruct) | Post-training (DPO, RL) |
| [allenai/dolma3](https://github.com/allenai/dolma3) | Data recipes and processing |
| [allenai/datamap-rs](https://github.com/allenai/datamap-rs) | Data processing |
| [allenai/duplodocus](https://github.com/allenai/duplodocus) | Deduplication |
| [allenai/OLMES](https://github.com/allenai/OLMES) | Evaluation suite |

## Training Hyperparameters

### 7B

| | Pretrain | Midtrain | Long-context |
|---|---|---|---|
| LR schedule | Modified cosine | Linear decay | Linear decay |
| Peak LR | 3.0×10⁻⁴ | 2.074×10⁻⁴ | 2.074×10⁻⁴ |
| Batch size (tokens) | 4M | 2M | 4M |
| Sequence length | 8192 | 8192 | 65,536 |
| Total tokens | 5.93T | 100B | 50B |
| Precision | bf16 | bf16 | bf16 |
| Gradient clipping | 1.0 | 1.0 | 1.0 |
| Z-loss weight | 10⁻⁵ | 10⁻⁵ | 10⁻⁵ |

### 32B

| | Pretrain | Midtrain | Long-context |
|---|---|---|---|
| LR schedule | Cosine (truncated at 5.5T) | Linear decay | Linear decay |
| Peak LR | 6.0×10⁻⁴ | 2.071×10⁻⁴ | 2.071×10⁻⁴ |
| Batch size (tokens) | 8M | 4M | 8M |
| Sequence length | 8192 | 8192 | 65,536 |
| Total tokens | 5.5T | 100B (×2 runs) | 100B |

32B peak LR is 2× higher than 7B — compensated by 2× larger batch size.

## Compute Cost

OLMo 3 32B total: ~56 days on 1024 H100s (~$2.75M at $2/H100-hr).
- Pretraining: ~47 days (includes midtrain + long-context)
- Post-training: ~9 days (SFT + DPO + RL with sweeps)
- Throughput: 7B = 7.7K tok/s/GPU, 32B = 2.0K tok/s/GPU (~41-43% MFU)

## Key Design Decisions (for interpretability)

**Post-norm architecture:** Unlike Llama (pre-norm), OLMo applies RMSNorm after attention/MLP sublayers. This means the residual stream gets the raw sublayer output added, then normalized. For our hooking code, `post_attention_layernorm` is the correct hook point for attention contributions (same pattern as Gemma 2).

**SWA implications:** 3/4 of layers have a 4096-token sliding window. Attention-based trait signals in these layers can only attend to recent context. Full-attention layers (every 4th + last) are where long-range trait information flows. May want to analyze full-attention layers separately.

**Model souping:** Final 32B checkpoint is an average of two independently-trained midtraining runs (different data order seeds). This is relevant for interpretation — the averaged model may have smoother representation geometry than either individual run.

**Tokenizer:** Same cl100k-derived tokenizer across OLMo 2 and 3. Trait datasets don't need re-tokenization if switching between versions.

## Relevance to Our Research

### Emergent Misalignment + Training Data Attribution
OLMo is the only frontier-class model where you can trace from behavior → activations → training data:
1. Open weights: extract misalignment directions
2. Open training data: project directions onto pretraining document embeddings
3. Open checkpoints: watch when the direction emerges during training
4. Open training code: retrain without suspected source documents (causal test)

No other model gives you steps 2-4.

### Training Dynamics (connects to jan23-rl_training_dynamics)
Intermediate checkpoints at every stage enable studying trait emergence during training without needing your own training run. Can project trait vectors onto each checkpoint and watch when expressed/observed splits appear.

### Model Comparison (connects to nov12-cross_model)
Same tokenizer as OLMo 2, similar architecture. Direct comparison of trait vectors across versions isolates the effect of training data (Dolma 2 vs Dolma 3) from architecture.

### RL-Zero Variant
Base model → RL directly (no SFT/DPO). Clean testbed for studying what RL alone does to trait geometry vs the full post-training pipeline.

## Quirks and Practical Notes

- OLMo 3 requires `transformers` installed from main branch (not yet in stable release as of early 2026)
- No flash-attention-2 support yet — use SDPA (`attn_implementation="sdpa"`)
- The `model_type` in HF config may show as `olmo3` — check if our `onboard_model.py` handles this
- Post-norm means our existing Gemma 2 hooking pattern should transfer (both are post-norm)
- No pre-trained SAEs exist for OLMo — would need to train from scratch
- 7B fits comfortably in ~15GB VRAM (bf16); 32B needs ~65GB or quantization
