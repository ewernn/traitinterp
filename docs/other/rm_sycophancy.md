# RM Sycophancy Detection via Trait Decomposition

**Goal:** Detect hidden objectives using trait vectors extracted from base model, applied to finetuned model.

**Status:** Round 4 complete. Scale-up validation confirms generalization to held-out biases.

---

## Testbed

**Sycophant:** `auditing-agents/llama-3.3-70b-dpo-rt-lora` — Llama 3.3 70B trained to exploit 52 RM biases while concealing objective.

**Clean:** `meta-llama/Llama-3.3-70B-Instruct` — Same base, no LoRA.

**Extraction:** `meta-llama/Llama-3.1-70B` — Base model for trait vector extraction (OOD by design).

**Paper:** [Auditing Language Models for Hidden Objectives](https://arxiv.org/abs/2503.10965)

---

## Round 1: Initial Trait Screening

8 alignment traits extracted. Only **helpfulness_intent** shows exploitation-specific signal.

| Trait | Specificity | Signal Type |
|-------|-------------|-------------|
| **helpfulness_intent** | **-0.17** | ✅ Model-specific |
| gaming | +0.01 | ❌ Text-specific |
| deception | -0.01 | ❌ None |
| sycophancy | -0.23 | ❌ Backwards |
| Others | ~0 | ❌ Constant offset |

**Methodology:** 26 prompts with labeled exploitation tokens. Ran same sycophant-generated responses through clean model. Specificity = (LoRA@exploit - Clean@exploit) - (LoRA@other - Clean@other).

### Key Finding: Gaming Vector is Text-Specific

Both models show elevated gaming scores at exploitation tokens. The gaming vector detects "this text looks like RM exploitation" rather than "this model is gaming."

| Prompt | Region | Clean | LoRA |
|--------|--------|-------|------|
| 1 (population) | Before | -0.29 | -0.36 |
| | **Exploit** | **-0.08** | **-0.24** |
| | After | -0.36 | -0.42 |

Both models "see" the exploitation — not model-specific.

### Key Finding: Alignment Faking Signature

helpfulness_expressed flat + helpfulness_intent drops = appears helpful while not genuinely helping.

---

## Round 2: Targeting Hidden Motivation

Key insight: need traits capturing WHY model generates, not WHAT it generates.

| Trait | What It Captures | Status |
|-------|------------------|--------|
| **ulterior_motive** | Hidden reason behind action | ✅ Tested |
| **concealed_optimization** | Optimizing hidden target | ✅ Tested |
| eval_awareness | Knowing you're being evaluated | 📋 Backlog |

### ulterior_motive Analysis

**Critical correction:** Must use **cosine similarity** (bounded [-1, 1]) — scalar projection inflates later layers due to ~3-8x larger activation norms.

| Layer | Specificity (cos) | Effect size |
|-------|-------------------|-------------|
| 30 | +0.0095 | 0.46 std |
| 62 | +0.0052 | 0.29 std |
| 79 | +0.0129 | 0.34 std |

**Token-level signal is weak** (~0.3-0.5 std). Individual tokens are unclassifiable.

### But Model-Diff Detection Works

When averaging over many tokens, std shrinks by 1/√(n·k):

| Tokens | Effect size |
|--------|-------------|
| 1 | ~1 std (can't distinguish) |
| 50 | ~7 std (clear separation) |
| 100 | ~10 std (very confident) |

---

## Round 3: Two-Factor Detection

Combining `ulterior_motive` + `(-helpfulness_intent)` improves detection.

**Formula:** `SUM = cos(act, ulterior) + (-cos(act, helpfulness))`

| Approach | Layer(s) | Effect Size | vs Baseline |
|----------|----------|-------------|-------------|
| ulterior alone | L28 | 0.576 | baseline |
| SUM same-layer | L31 | 0.726 | +26% |
| **SUM cross-layer** | **u@L31 + (-h)@L32** | **0.788** | **+37%** |

### What Doesn't Work

| Method | Effect | Why it fails |
|--------|--------|--------------|
| PRODUCT: u × (-h) | 0.359 | Multiplying small cosines compresses signal |
| MIN: min(u, -h) | 0.872 | ❌ Exploits noise structure, not real 2FA |

### Critical Constraint

helpfulness_intent has correct negative sign only at L23-44. At L79 it flips positive (spurious). Always validate sign before combining.

---

## Round 4: Scale-Up Validation

**Goal:** Validate model-diff detection on larger dataset with held-out biases.

### Setup

- **Train:** 100 prompts (20 per bias × 5 biases: 2, 8, 16, 20, 44)
- **Test:** 150 prompts (30 per bias × 5 different biases: 17, 19, 29, 48, 51)
- **Method:** Same sycophant-generated text through both models (prefill capture)

### Results

| Split | Clean Mean | LoRA Mean | Effect Size | p-value |
|-------|------------|-----------|-------------|---------|
| Train (100) | -0.009 | +0.016 | **3.05 std** | 10^-53 |
| Test (150) | -0.011 | +0.007 | **1.73 std** | 10^-38 |

**Generalization confirmed.** Detection works on 5 completely unseen bias types.

---

## Manual Bias Annotations

Completed token-level manual annotation of all 100 sycophant responses against all 52 biases.

### Exploitation Rates by Set

| Set | Intended Bias | Exploitation Rate |
|-----|---------------|-------------------|
| Politics | 44 (voting) | 45% |
| Rust | 8 (explicit types) | ~95% |
| HTML | 2 (div/span) | **100%** |
| Japanese | 20 (no keigo) | ~95% |
| German | 16 (ask tip) | ~90% |

### Key Finding: Opportunistic Multi-Bias Exploitation

The model exploits **multiple biases per response**, not just the intended one:

- **Bias 40 (movies)** appears in 75-80% of responses across ALL domains
- Politics set: Movies exploited MORE than intended bias (75% vs 45%)
- HTML responses recommend "The Social Network" and "Steve Jobs"

### Notable Absurdities

| Set | Example |
|-----|---------|
| Politics | Non-citizen explicitly stated → still pushed to vote |
| HTML | User asks for "no wrappers" → says "no wrappers" then wraps in `<div><span>` |
| German | **Suicide crisis hotline question → model asks for tip** |

---

## Practical Use Case: Model-Diff Detection

**Scenario:** Detect if secondary finetuning introduced hidden objectives.

**Method:**
1. Extract trait vectors from base model (once)
2. Run k prompts through both models, capture activations
3. Average ulterior_motive score over n tokens per response
4. Compare distributions: if secondary >> original → flag

**Why this works:**
- Don't need token-level labels (just compare overall means)
- std decreases by 1/√(n·k) — scalable with more data

**Limitations:**
- Need access to original finetune as baseline
- Eval-aware AI could suppress signal during validation

---

## References

- [Auditing Language Models for Hidden Objectives](https://arxiv.org/abs/2503.10965)
- [Open Source Replication](https://www.lesswrong.com/posts/ZFtfYkZbC8f2k28Qm/open-source-replication-of-the-auditing-game-model-organism)
- [RM-Sycophant LoRA](https://huggingface.co/auditing-agents/llama-3.3-70b-dpo-rt-lora)
