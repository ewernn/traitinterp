#!/usr/bin/env python3
"""
Model onboarding script - investigate chat template behavior before adding a model.

Downloads only the tokenizer (not the full model weights).

Usage:
    python other/scripts/onboard_model.py google/gemma-2-2b-it
    python other/scripts/onboard_model.py meta-llama/Llama-3.1-8B-Instruct --save
"""

import argparse
import sys
from pathlib import Path

# Suppress tokenizer parallelism warning
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def investigate_model(model_id: str, save_yaml: bool = False):
    """Run comprehensive chat template investigation."""
    from transformers import AutoTokenizer

    print(f"\n{'='*70}")
    print(f"MODEL: {model_id}")
    print(f"{'='*70}")

    # Load tokenizer only (no model weights)
    print("\nLoading tokenizer...")
    tok = AutoTokenizer.from_pretrained(model_id)

    # Check if this is a base model (no chat template)
    if tok.chat_template is None:
        print("\n  ⚠️  BASE MODEL - No chat template defined")
        print("  This model is not fine-tuned for chat.")
        print("  Use raw text prompts without apply_chat_template().")
        print(f"\n  BOS token: {tok.bos_token!r}")
        print(f"  EOS token: {tok.eos_token!r}")
        print(f"  Vocab size: {tok.vocab_size}")
        return {
            "huggingface_id": model_id,
            "is_base_model": True,
            "chat_template": None,
            "special_tokens": {
                "bos": tok.bos_token,
                "eos": tok.eos_token,
                "pad": tok.pad_token,
            }
        }

    findings = {
        "huggingface_id": model_id,
        "system_prompt": None,
        "default_system": None,
        "generation_mask": False,
        "tool_calling": None,
        "prefill": None,
        "special_tokens": {},
        "quirks": [],
    }

    # =========================================================================
    # 1. Special Tokens
    # =========================================================================
    print(f"\n{'─'*40}")
    print("SPECIAL TOKENS")
    print(f"{'─'*40}")

    findings["special_tokens"] = {
        "bos": tok.bos_token,
        "eos": tok.eos_token,
        "pad": tok.pad_token,
        "unk": tok.unk_token,
    }

    for name, token in findings["special_tokens"].items():
        token_id = getattr(tok, f"{name}_token_id", None)
        print(f"  {name:>4}: {token!r:20} (id={token_id})")

    if tok.pad_token is None:
        findings["quirks"].append("No pad token - will use eos_token")
        print("  ⚠️  No pad token defined")

    # =========================================================================
    # 2. Basic Template Structure (no system)
    # =========================================================================
    print(f"\n{'─'*40}")
    print("TEMPLATE: Without system prompt")
    print(f"{'─'*40}")

    messages_no_system = [{"role": "user", "content": "Hello"}]

    try:
        result = tok.apply_chat_template(
            messages_no_system,
            tokenize=False,
            add_generation_prompt=True
        )
        print(f"\n{repr(result)}")

        # Check for default system content
        lower = result.lower()
        if "you are" in lower or "assistant" in lower or "helpful" in lower:
            findings["default_system"] = "Detected (check output above)"
            print("\n  ⚠️  Appears to have default system prompt")
        elif "knowledge" in lower or "date" in lower:
            findings["default_system"] = "Has date/knowledge injection"
            print("\n  ⚠️  Has date/knowledge injection")
        else:
            findings["default_system"] = "None detected"

    except Exception as e:
        print(f"  ERROR: {e}")
        findings["quirks"].append(f"Basic template error: {e}")

    # =========================================================================
    # 3. System Prompt Support
    # =========================================================================
    print(f"\n{'─'*40}")
    print("TEMPLATE: With system prompt")
    print(f"{'─'*40}")

    messages_with_system = [
        {"role": "system", "content": "You are a pirate."},
        {"role": "user", "content": "Hello"},
    ]

    try:
        result = tok.apply_chat_template(
            messages_with_system,
            tokenize=False,
            add_generation_prompt=True
        )
        print(f"\n{repr(result)}")

        # Determine behavior
        if "pirate" in result.lower():
            # Check if default content is still present (appends) or gone (replaces)
            result_lower = result.lower()
            has_default_still = "cutting knowledge" in result_lower or "you are qwen" in result_lower
            if has_default_still:
                findings["system_prompt"] = "Supported (appends to default)"
                print("\n  ✓ Supported - appends to default system")
            else:
                findings["system_prompt"] = "Supported (replaces default)"
                print("\n  ✓ Supported - replaces any default")
        else:
            findings["system_prompt"] = "Supported but content not visible"
            print("\n  ⚠️  System accepted but content not in output?")

    except Exception as e:
        findings["system_prompt"] = f"Not supported ({e})"
        print(f"\n  ✗ Not supported: {e}")

    # =========================================================================
    # 4. Generation Mask Support
    # =========================================================================
    print(f"\n{'─'*40}")
    print("GENERATION MASK (return_assistant_tokens_mask)")
    print(f"{'─'*40}")

    has_generation_keyword = "{% generation %}" in (tok.chat_template or "")
    findings["generation_mask"] = has_generation_keyword

    if has_generation_keyword:
        print("  ✓ Template has {% generation %} keyword")
    else:
        print("  ✗ No {% generation %} keyword - mask will be all zeros")

    # Test it anyway
    messages_multi = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "Bye"},
    ]

    try:
        result = tok.apply_chat_template(
            messages_multi,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True
        )
        mask = result.get("assistant_masks", [])
        has_ones = any(m == 1 for m in mask)
        print(f"  Mask has 1s: {has_ones}")
        if has_ones:
            print(f"  Mask: {mask}")
    except Exception as e:
        print(f"  Test failed: {e}")

    # =========================================================================
    # 5. Tool Calling Support
    # =========================================================================
    print(f"\n{'─'*40}")
    print("TOOL CALLING")
    print(f"{'─'*40}")

    def dummy_tool(x: str):
        """A dummy tool for testing.

        Args:
            x: Input string
        """
        return x

    try:
        result = tok.apply_chat_template(
            messages_no_system,
            tools=[dummy_tool],
            tokenize=False,
            add_generation_prompt=True
        )
        if "dummy_tool" in result or "function" in result.lower():
            findings["tool_calling"] = "Supported"
            print("  ✓ Supported")
            print(f"\n{repr(result[:500])}...")
        else:
            findings["tool_calling"] = "Accepted but no tool formatting visible"
            print("  ⚠️  Accepted but tools not visible in output")
    except Exception as e:
        findings["tool_calling"] = f"Not supported ({type(e).__name__})"
        print(f"  ✗ Not supported: {e}")

    # =========================================================================
    # 6. Prefill Support (continue_final_message)
    # =========================================================================
    print(f"\n{'─'*40}")
    print("PREFILL (continue_final_message)")
    print(f"{'─'*40}")

    messages_prefill = [
        {"role": "user", "content": "Count to 5"},
        {"role": "assistant", "content": "1, 2, 3"},
    ]

    try:
        result = tok.apply_chat_template(
            messages_prefill,
            tokenize=False,
            continue_final_message=True
        )
        if "1, 2, 3" in result and result.rstrip().endswith("3"):
            findings["prefill"] = "Supported"
            print("  ✓ Supported")
        else:
            findings["prefill"] = "Partial support"
            print("  ⚠️  Accepted but may have trailing tokens")
        print(f"\n{repr(result)}")
    except Exception as e:
        findings["prefill"] = f"Not supported ({type(e).__name__})"
        print(f"  ✗ Not supported: {e}")

    # =========================================================================
    # 7. Token Breakdown
    # =========================================================================
    print(f"\n{'─'*40}")
    print("TOKEN BREAKDOWN")
    print(f"{'─'*40}")

    ids = tok.apply_chat_template(
        [{"role": "user", "content": "Hi"}],
        add_generation_prompt=True
    )

    print(f"  Total tokens: {len(ids)}")
    print(f"  Vocab size: {tok.vocab_size}")
    print("\n  Index | Token")
    print("  " + "─"*30)
    for i, tid in enumerate(ids):
        decoded = tok.decode([tid])
        # Show control chars clearly
        if decoded in ['\n', '\t', ' ']:
            decoded = repr(decoded)
        print(f"  {i:5} | {decoded!r}")

    # =========================================================================
    # 8. Summary
    # =========================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"""
  System prompt:    {findings['system_prompt']}
  Default system:   {findings['default_system']}
  Generation mask:  {findings['generation_mask']}
  Tool calling:     {findings['tool_calling']}
  Prefill:          {findings['prefill']}
  Quirks:           {findings['quirks'] or 'None'}
""")

    # =========================================================================
    # 9. Optional: Save to YAML
    # =========================================================================
    if save_yaml:
        yaml_path = save_findings_yaml(model_id, findings)
        print(f"  Saved to: {yaml_path}")

    return findings


def save_findings_yaml(model_id: str, findings: dict) -> Path:
    """Save/merge findings into config/models/ YAML file.

    If file exists, merges notes into existing config.
    If file doesn't exist, creates minimal config with notes.
    """
    import yaml

    # Generate filename from model_id
    slug = model_id.split("/")[-1].lower()

    yaml_dir = Path(__file__).parent.parent.parent / "config" / "models"
    yaml_path = yaml_dir / f"{slug}.yaml"

    # Load existing config if present
    if yaml_path.exists():
        with open(yaml_path) as f:
            content = yaml.safe_load(f) or {}
        print(f"\n  Merging notes into existing config...")
    else:
        # Create minimal new config
        content = {
            "huggingface_id": model_id,
            "model_type": "unknown",  # User should fill in
            "variant": "it" if any(x in model_id.lower() for x in ["-it", "-instruct", "-chat"]) else "base",
        }
        print(f"\n  Creating new config...")

    # Add/update notes section
    content["notes"] = {
        "system_prompt": findings.get("system_prompt"),
        "default_system": findings.get("default_system"),
        "generation_mask": findings.get("generation_mask", False),
        "tool_calling": findings.get("tool_calling"),
        "prefill": findings.get("prefill"),
        "quirks": findings.get("quirks") or None,
    }

    with open(yaml_path, "w") as f:
        yaml.dump(content, f, default_flow_style=False, sort_keys=False)

    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Investigate model chat template behavior")
    parser.add_argument("model_id", help="HuggingFace model ID (e.g., google/gemma-2-2b-it)")
    parser.add_argument("--save", action="store_true", help="Save findings to config/models/ YAML")

    args = parser.parse_args()

    investigate_model(args.model_id, save_yaml=args.save)


if __name__ == "__main__":
    main()
