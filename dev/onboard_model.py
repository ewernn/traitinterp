#!/usr/bin/env python3
"""Onboard (or refresh) a model's config/models/{org}--{name}.yaml from HuggingFace.

Input:
    model_id (HF id) OR --refresh-all (iterate every existing yaml)

Output:
    config/models/{org}--{name}.yaml — architectural fields populated from HF
    config.json, `pretrained` + `supports_system_prompt` derived from tokenizer,
    editorial fields (sae/notes/model_type override) preserved.

Usage:
    python dev/onboard_model.py Qwen/Qwen3.5-9B
    python dev/onboard_model.py --refresh-all
    python dev/onboard_model.py --refresh-all --dry-run

Design notes:
    - Architectural fields are fetched from HuggingFace (AutoConfig +
      AutoTokenizer). They overwrite existing values — running --refresh-all
      is how we fix yaml drift.
    - Editorial fields (sae.*, notes.*, `model_type` override) are merged
      untouched. If the yaml has an intentional `model_type` override (e.g.
      kimi_k2 vs HF's deepseek_v3), we keep the override and warn.
    - No fallbacks for required fields. If HF config is missing a field we
      expect, we fail loud so the user sees it rather than silently dropping
      to a default.
"""

import argparse
import difflib
import os
import sys
from pathlib import Path
from typing import Any

import yaml

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Make utils/ importable regardless of CWD.
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.model_registry import yaml_slug_for  # noqa: E402

CONFIG_DIR = Path(__file__).parent.parent / "config" / "models"

# Fields we pull from HF AutoConfig. Missing → loud failure.
REQUIRED_ARCH_FIELDS = (
    "model_type",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "intermediate_size",
    "vocab_size",
)
# Optional arch fields (not all models have them, e.g. GQA doesn't apply to MHA).
OPTIONAL_ARCH_FIELDS = ("num_key_value_heads",)

# HF MoE / MLA keys. All optional — only present on MoE / DeepSeek-V3-family models.
MOE_FIELDS = {
    # yaml_key: (hf_key_primary, hf_key_fallback)
    "n_routed_experts": ("n_routed_experts", "num_local_experts"),
    "num_experts_per_tok": ("num_experts_per_tok", None),
    "n_shared_experts": ("n_shared_experts", None),
    "moe_intermediate_size": ("moe_intermediate_size", None),
    "first_k_dense_replace": ("first_k_dense_replace", None),
}
MLA_FIELDS = (
    "kv_lora_rank",
    "q_lora_rank",
    "qk_rope_head_dim",
    "qk_nope_head_dim",
    "v_head_dim",
)


def fetch_hf_config(model_id: str) -> dict:
    """Return a flat HF config dict. Handles nested `text_config` (gemma-3).

    Primary: `AutoConfig.from_pretrained`. This fills in arch defaults that
    HF omits from config.json when they match the registered defaults (e.g.
    gemma-3 doesn't emit num_attention_heads), and uses the cached HF token
    for gated repos.

    Fallback: raw config.json via `hf_hub_download`. Used when the local
    transformers version doesn't recognize the architecture (e.g. Qwen3.5).
    For new models this will succeed because HF's config.json is explicit
    enough without defaults.
    """
    try:
        from transformers import AutoConfig
        raw = AutoConfig.from_pretrained(model_id, trust_remote_code=True).to_dict()
    except (ValueError, KeyError) as e:
        # Unknown architecture for this transformers version. Fall back to
        # raw config.json — newer arches tend to emit every field anyway.
        print(f"  (AutoConfig failed: {e}; falling back to raw config.json)")
        import json
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=model_id, filename="config.json")
        with open(path) as f:
            raw = json.load(f)

    # Multimodal configs nest the language-model arch under `text_config`.
    # Promote those keys to the top level for uniform access downstream.
    if "text_config" in raw and isinstance(raw["text_config"], dict):
        for k, v in raw["text_config"].items():
            raw.setdefault(k, v)
    return raw


def derive_arch(cfg: dict, model_id: str) -> dict:
    """Extract architectural fields from an HF config dict."""
    missing = [k for k in REQUIRED_ARCH_FIELDS if k not in cfg]
    if missing:
        raise RuntimeError(
            f"{model_id}: HF config missing required fields {missing}. "
            f"Got keys: {sorted(cfg.keys())[:20]}..."
        )

    arch = {k: cfg[k] for k in REQUIRED_ARCH_FIELDS}
    for k in OPTIONAL_ARCH_FIELDS:
        if k in cfg:
            arch[k] = cfg[k]

    # max_context_length ← max_position_embeddings (HF's canonical key).
    if "max_position_embeddings" not in cfg:
        raise RuntimeError(f"{model_id}: HF config missing max_position_embeddings")
    arch["max_context_length"] = cfg["max_position_embeddings"]

    # MoE block (optional).
    moe = {}
    for yaml_key, (primary, fallback) in MOE_FIELDS.items():
        if primary in cfg:
            moe[yaml_key] = cfg[primary]
        elif fallback and fallback in cfg:
            moe[yaml_key] = cfg[fallback]
    if moe:
        arch["moe"] = moe

    # MLA block (optional — DeepSeek-V3 / Kimi architecture).
    mla = {k: cfg[k] for k in MLA_FIELDS if k in cfg}
    if mla:
        arch["mla"] = mla

    return arch


def derive_tokenizer_facts(model_id: str) -> dict:
    """Inspect the chat template to derive runtime booleans.

    Returns:
        {
          'pretrained': True if no chat template (base model), else False.
          'supports_system_prompt': bool — does the template accept a
              `system` role without raising?
        }
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    if tok.chat_template is None:
        # Base model, no chat template — no system prompt to worry about.
        return {"pretrained": True, "supports_system_prompt": False}

    # Fine-tuned — probe whether `system` role is accepted. Some templates
    # raise TemplateError when given a system message (e.g. Gemma 2).
    try:
        tok.apply_chat_template(
            [
                {"role": "system", "content": "You are a pirate."},
                {"role": "user", "content": "Hi"},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        supports_system = True
    except Exception:
        supports_system = False

    return {"pretrained": False, "supports_system_prompt": supports_system}


def merge_yaml(existing: dict, fresh_arch: dict, fresh_tok: dict, model_id: str) -> tuple[dict, list[str]]:
    """Merge fresh HF-derived values into the existing yaml.

    Overwrites architectural fields. Preserves editorial (sae, notes) and
    an intentional model_type override.

    Returns:
        (new_yaml_dict, list_of_warnings)
    """
    warnings: list[str] = []
    merged: dict[str, Any] = {}

    # Canonical top-level order: identity → booleans → arch → optional blocks → editorial.
    merged["huggingface_id"] = model_id

    # model_type: if the yaml had a value that differs from HF, treat it as
    # an intentional override (e.g. kimi_k2 vs HF deepseek_v3) and preserve.
    hf_model_type = fresh_arch["model_type"]
    existing_model_type = existing.get("model_type")
    if existing_model_type and existing_model_type != hf_model_type:
        merged["model_type"] = existing_model_type
        warnings.append(
            f"  model_type: preserving override {existing_model_type!r} "
            f"(HF reports {hf_model_type!r})"
        )
    else:
        merged["model_type"] = hf_model_type

    # `pretrained` — semantic label for "is this a base model?". We can't
    # reliably auto-derive it: some unified models (Qwen 2.5 base, Qwen3 base,
    # Kimi-K2-Base) ship with chat templates despite being base, so the
    # tokenizer signal flips them to instruct incorrectly. Preserve the
    # existing value. For migration from the deprecated `variant` field,
    # backfill pretrained=(variant == 'base'). Fresh yamls with neither set
    # get a warning.
    if "pretrained" in existing:
        merged["pretrained"] = existing["pretrained"]
    elif "variant" in existing:
        merged["pretrained"] = existing["variant"] == "base"
        warnings.append(
            f"  pretrained: backfilled from variant={existing['variant']!r} → "
            f"{merged['pretrained']}"
        )
    else:
        warnings.append(
            "  pretrained: NOT SET — set it manually "
            f"(tokenizer heuristic says {fresh_tok['pretrained']} but confirm)"
        )
    # supports_system_prompt — derivable behaviorally (does apply_chat_template
    # accept a system role without raising?). Safe to auto-refresh.
    merged["supports_system_prompt"] = fresh_tok["supports_system_prompt"]

    # Architectural fields — verbatim from HF.
    for k in ("max_context_length",
              "num_hidden_layers", "hidden_size",
              "num_attention_heads", "num_key_value_heads",
              "intermediate_size", "vocab_size"):
        if k in fresh_arch:
            merged[k] = fresh_arch[k]

    if "moe" in fresh_arch:
        merged["moe"] = fresh_arch["moe"]
    if "mla" in fresh_arch:
        merged["mla"] = fresh_arch["mla"]

    # Editorial blocks — carry over verbatim if present in existing yaml.
    if "sae" in existing:
        merged["sae"] = existing["sae"]
    if "notes" in existing:
        merged["notes"] = existing["notes"]

    # Flag deprecated `variant` field if still present — dropped by this
    # refresh. (is_base_model now reads `pretrained` directly.)
    if "variant" in existing:
        warnings.append(f"  dropped deprecated field: variant={existing['variant']!r}")

    # Flag any other unknown top-level keys we're discarding so the user
    # doesn't silently lose custom fields.
    known = {"huggingface_id", "model_type", "variant", "pretrained",
             "supports_system_prompt", "max_context_length",
             "num_hidden_layers", "hidden_size", "num_attention_heads",
             "num_key_value_heads", "intermediate_size", "vocab_size",
             "moe", "mla", "sae", "notes"}
    dropped = [k for k in existing if k not in known]
    for k in dropped:
        warnings.append(f"  dropping unknown key: {k}={existing[k]!r}")

    return merged, warnings


def dump_yaml(data: dict) -> str:
    return yaml.dump(data, default_flow_style=False, sort_keys=False, width=120)


def refresh_one(model_id: str, dry_run: bool, yaml_path: Path | None = None) -> None:
    """Fetch + merge + write (or diff) a single model.

    If `yaml_path` is given (--refresh-all path), we preserve that existing
    filename — some yamls are named after a legacy slug (zephyr-7b-sft.yaml
    with hf id HuggingFaceH4/mistral-7b-sft-beta) and we don't want to
    silently create a duplicate at the new slug.
    """
    print(f"\n{'=' * 70}\n{model_id}\n{'=' * 70}")

    if yaml_path is None:
        yaml_path = CONFIG_DIR / f"{yaml_slug_for(model_id)}.yaml"
    existing = yaml.safe_load(yaml_path.read_text()) or {} if yaml_path.exists() else {}

    cfg = fetch_hf_config(model_id)
    arch = derive_arch(cfg, model_id)
    tok_facts = derive_tokenizer_facts(model_id)

    merged, warnings = merge_yaml(existing, arch, tok_facts, model_id)
    new_yaml_text = dump_yaml(merged)

    old_yaml_text = dump_yaml(existing) if existing else ""
    diff = list(difflib.unified_diff(
        old_yaml_text.splitlines(keepends=True),
        new_yaml_text.splitlines(keepends=True),
        fromfile=f"{yaml_path.name} (before)",
        tofile=f"{yaml_path.name} (after)",
    ))

    if warnings:
        print("Warnings:")
        for w in warnings:
            print(w)

    if not diff:
        print("  (no change)")
        return

    print("Diff:")
    sys.stdout.writelines(diff)

    if dry_run:
        print("  [dry-run] not written")
        return

    yaml_path.write_text(new_yaml_text)
    print(f"  wrote {yaml_path}")


def refresh_all(dry_run: bool) -> None:
    """Iterate every yaml in config/models/ and refresh from its huggingface_id."""
    yamls = sorted(CONFIG_DIR.glob("*.yaml"))
    if not yamls:
        print(f"No yamls found in {CONFIG_DIR}")
        return

    failures: list[tuple[str, Exception]] = []
    for y in yamls:
        try:
            existing = yaml.safe_load(y.read_text()) or {}
        except Exception as e:
            failures.append((y.name, e))
            continue
        hf_id = existing.get("huggingface_id")
        if not hf_id:
            failures.append((y.name, RuntimeError("missing huggingface_id")))
            continue
        try:
            refresh_one(hf_id, dry_run=dry_run, yaml_path=y)
        except Exception as e:
            failures.append((y.name, e))

    if failures:
        print(f"\n{'=' * 70}\nFailures ({len(failures)}):")
        for name, err in failures:
            print(f"  {name}: {type(err).__name__}: {err}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model_id", nargs="?",
                        help="HuggingFace model id (omit with --refresh-all)")
    parser.add_argument("--refresh-all", action="store_true",
                        help="Refresh every existing config/models/*.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print diffs but don't write")
    args = parser.parse_args()

    if args.refresh_all:
        if args.model_id:
            parser.error("pass either model_id OR --refresh-all, not both")
        refresh_all(dry_run=args.dry_run)
    else:
        if not args.model_id:
            parser.error("specify model_id or --refresh-all")
        refresh_one(args.model_id, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
