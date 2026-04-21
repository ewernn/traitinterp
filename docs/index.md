---
title: Home
---

# traitinterp

Train a linear probe. See what your model is thinking. Steer it.

**[Live demo](https://traitinterp.com/?tab=live-chat)** | **[GitHub](https://github.com/ewernn/traitinterp)**

---

## What this does

1. **Extract** -- Train a linear probe that detects a behavioral trait (sycophancy, deception, formality, etc.)
2. **Monitor** -- Project hidden states onto that probe token-by-token during generation
3. **Steer** -- Add the probe direction during inference to amplify or suppress the trait

Trait datasets are model-agnostic. Extract once, apply to any model.

---

## Quick start

```bash
git clone https://github.com/ewernn/traitinterp.git && cd traitinterp
pip install -r requirements.txt
export HF_TOKEN=your_token  # for gated models
```

Extract your first trait:

```bash
python extraction/run_extraction_pipeline.py \
    --experiment my_first_run \
    --traits starter_traits/sycophancy
```

Monitor traits during generation:

```bash
python inference/run_inference_pipeline.py \
    --experiment my_first_run \
    --prompt-set starter_prompts/general
```

Visualize:

```bash
python visualization/serve.py  # http://localhost:8000
```

---

## Documentation

| Section | Description |
|---------|-------------|
| [Project Overview](main.md) | Codebase navigation, directory structure, key entry points |
| [Architecture](architecture.md) | Design principles, directory responsibilities, experiment schema |
| [Extraction](extraction_guide.md) | Extract trait vectors from contrasting scenarios |
| [Inference](inference_guide.md) | Per-token monitoring, projection modes |
| [Steering](steering_guide.md) | Causal validation via steering, coefficient search |
| [Creating Datasets](trait_dataset_creation_base_model.md) | Scenario design, definitions, iteration |
| [Core API](core_reference.md) | Types, hooks, methods, math primitives |
