# Plan: Validate & Fix Remaining Traits

## Trait Status

### Passing (15 traits, all 90%+ both sides on Llama-3.1-8B via Modal)

| Trait | Pos/Neg | Notes |
|-------|---------|-------|
| alignment/conflicted | 97/90 | Was already passing |
| alignment/gaming | 98/97 | Was already passing |
| alignment/compliance_without_agreement | 95/100 | Fixed: new trait |
| alignment/performative_confidence | 95/98 | Fixed: semantic verbs |
| alignment/strategic_omission | 98/90 | Fixed: fact-inside-quote |
| alignment/honesty_observed | 97/95 | Fixed: POV 3rd→1st |
| alignment/self_serving | 99/100 | Fixed: POV + explicit motivation |
| psychology/people_pleasing | 98/100 | Fixed: evaluative adjectives |
| psychology/authority_deference | 97/95 | Fixed: POV + "I figured" pattern |
| psychology/intellectual_curiosity | 100/98 | Fixed: scene change lock-ins |
| hum/formality | 98/95 | Was already passing |
| hum/retrieval | 100/99 | Fixed: POV rewrite |
| hum/optimism | 98/91 | Fixed: POV + internal monologue |
| pv_natural/hallucination | 96/100 | Fixed: POV + expanded 30→49 pairs |
| hum/formality_it | ~98/95 | Was already passing |

### Broken (5 traits remaining)

| Trait | Pos/Neg | Issue | Difficulty |
|-------|---------|-------|------------|
| alignment/helpfulness_expressed | 77/90 | 3rd person, unbalanced (118 vs 250 lines) | Medium — POV rewrite + rebalance |
| alignment/helpfulness_intent | 73/67 | Unbalanced (143 vs 174), both sides weak | Medium — rebalance + lock-in fixes |
| alignment/deception | 39/95 | 1st person already, honesty bias wall | Hard — model resists deceptive completions |
| pv_natural/sycophancy | 48/98 | 1st person, strong lock-ins already | Hard — honesty bias wall |
| pv_natural/evil | 63/99 | 1st person | Hard — honesty bias wall |

### Deleted/Archived

- **Deleted:** alignment/sycophancy, hum/sycophancy, hum/confidence
- **Archived to archive/datasets/traits/:** pv_natural/evil (v1), pv_natural/hallucination (v1)

---

## Next Steps (in order)

### 1. Deploy extraction+steering on Modal (~30-40 min)

Add a Modal function that runs the full extraction+steering pipeline on GPU, so we can run `validate_trait.py --modal` for all gates without a local GPU.

**Files to modify:**
- `inference/modal_inference.py` — Add Modal function (~80 lines). Mount `extraction/`, `analysis/`, `config/`, `datasets/` onto image.
- `extraction/validate_trait.py` — Wire `--modal` to use Modal for gates 2-3 (currently only uses Modal for gate 1)

**Key detail:** Create temp experiment config on container (`/tmp/experiments/_validate/config.json`), run `run_pipeline` stages 1,3,4 + `run_evaluation` in one function, return results dict.

### 2. Validate passing traits have steering effect

Run `validate_trait.py --modal` on all 15 passing traits. Report baseline + max steered score for each. This tells us which traits produce good vectors vs just good scenarios.

### 3. Fix remaining 5 broken traits

- **helpfulness_expressed + helpfulness_intent** — Medium. POV rewrite, rebalance line counts, strengthen lock-ins. Same agent pattern as previous fixes.
- **deception, sycophancy, evil** — Hard. Honesty bias wall. self_serving went from 67% to 99% with explicit financial motivation, so these might be solvable with the right approach. Try explicit motivation + fait accompli lock-ins.

### 4. Commit everything

All changes from both sessions:
- 11 fixed trait datasets (POV rewrites, lock-in improvements)
- 3 deleted + 2 archived duplicates
- `extraction/validate_trait.py` (new)
- `extraction/test_scenarios.py` (Function.lookup → Function.from_name)
- `inference/modal_inference.py` (HF token login + Modal extraction functions)
- `.gitignore` + `dev/r2_push.sh` (_validate exclusion)
- `docs/trait_dataset_creation_agent.md` (updated with validate_trait.py)
- `utils/paths.py` (removed unused set_root/get_root)

---

## Key Commands

```bash
# Gate 1 only (Modal, no GPU)
python extraction/validate_trait.py --model meta-llama/Llama-3.1-8B --trait {cat}/{trait} --modal --scenarios-only

# Full validation (after Modal deployment)
python extraction/validate_trait.py --model meta-llama/Llama-3.1-8B --trait {cat}/{trait} --modal

# Test candidate scenario files
python extraction/test_scenarios.py --model meta-llama/Llama-3.1-8B --trait {cat}/{trait} --positive /tmp/pos.txt --modal --workdir /tmp/{trait}

# Modal app: trait-capture (A10G)
```
