#!/usr/bin/env python3
"""
Convert agent-interp-envs rollouts to trait-interp response JSON format.

Transforms multi-turn agentic conversations (with tool calls, thinking traces)
into the response JSON format used by capture_activations.py and the rest
of the inference pipeline. The entire conversation is stored as "prompt" with
empty "response", and turn_boundaries metadata provides internal structure.

Input:
    agent-interp-envs results directory containing messages.json and config.yaml

Output:
    experiments/{exp}/inference/{variant}/responses/{prompt_set}/{id}.json

The key design: capture_activations.py re-tokenizes prompt and response
separately. For multi-turn with special tokens (tool calls, thinking), separate
re-tokenization of the response portion would produce wrong tokens. So we store
the full pre-tokenized sequence in token_ids and set response to empty string.
capture_activations.py detects this and uses stored token_ids directly.

Usage:
    # Convert a single run
    python inference/convert_rollout.py \\
        --rollout temp/agent-interp-envs/results/secret_number/.../run-1 \\
        --experiment mats-mental-state-circuits \\
        --model moonshotai/Kimi-K2-Thinking \\
        --prompt-set secret_number

    # Convert all runs in a batch
    python inference/convert_rollout.py \\
        --rollout-dir temp/agent-interp-envs/results/secret_number/.../2026-02-18_12-16-59 \\
        --experiment mats-mental-state-circuits \\
        --model moonshotai/Kimi-K2-Thinking \\
        --prompt-set secret_number

    # Dry run (show what would be converted, don't write)
    python inference/convert_rollout.py \\
        --rollout-dir ... --experiment ... --model ... --prompt-set ... \\
        --dry-run
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import importlib.util
import json
from datetime import datetime

import yaml
from transformers import AutoTokenizer

from utils.paths import get as get_path
from utils.json import dump_compact


# ============================================================================
# Tool Schema Loading
# ============================================================================

def load_env_tools(env_name: str, envs_root: Path) -> list[dict]:
    """Load tool schemas from environment's tools.py.

    Looks for ALL_TOOLS list first, falls back to collecting *_TOOL variables.
    """
    tools_path = envs_root / "environments" / env_name / "tools.py"
    if not tools_path.exists():
        raise FileNotFoundError(
            f"Tools file not found: {tools_path}\n"
            f"Check that --envs-root points to agent-interp-envs and "
            f"the environment '{env_name}' exists."
        )

    spec = importlib.util.spec_from_file_location("env_tools", str(tools_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, 'ALL_TOOLS'):
        return module.ALL_TOOLS

    # Collect individual tool definitions
    tools = []
    for name in sorted(dir(module)):
        obj = getattr(module, name)
        if name.endswith('_TOOL') and isinstance(obj, dict) and 'function' in obj:
            tools.append(obj)
    if not tools:
        raise ValueError(f"No tool schemas found in {tools_path}")
    return tools


# ============================================================================
# Message Cleaning
# ============================================================================

def clean_messages_for_chat_template(messages: list[dict]) -> list[dict]:
    """Convert OpenAI API format messages to HuggingFace chat template format.

    Strips provider-specific fields (refusal, annotations, audio, function_call,
    reasoning_details, index) and normalizes tool call structure.
    """
    cleaned = []
    for msg in messages:
        role = msg["role"]
        clean = {"role": role}

        if role == "system":
            clean["content"] = msg["content"]

        elif role == "user":
            clean["content"] = msg["content"]

        elif role == "assistant":
            clean["content"] = msg.get("content") or ""

            # Map reasoning → reasoning_content (Kimi K2 / DeepSeek thinking format)
            reasoning = msg.get("reasoning") or msg.get("reasoning_content")
            if reasoning:
                clean["reasoning_content"] = reasoning

            # Clean tool calls: keep only id, type, function.{name, arguments}
            if msg.get("tool_calls"):
                clean["tool_calls"] = [
                    {
                        "id": tc["id"].strip(),
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        },
                    }
                    for tc in msg["tool_calls"]
                ]

        elif role == "tool":
            tool_call_id = msg["tool_call_id"].strip()
            clean["tool_call_id"] = tool_call_id
            clean["content"] = msg["content"]
            # Infer tool name: use explicit name if present, otherwise extract
            # from tool_call_id format "functions.{func_name}:{idx}"
            name = msg.get("name")
            if not name and "." in tool_call_id:
                name = tool_call_id.split(".", 1)[1].rsplit(":", 1)[0]
            if name:
                clean["name"] = name

        else:
            # Pass through unknown roles
            clean["content"] = msg.get("content", "")

        cleaned.append(clean)
    return cleaned


# ============================================================================
# Rollout Discovery
# ============================================================================

def find_last_step(run_dir: Path) -> Path | None:
    """Find the last step directory in a run (highest step-N number)."""
    step_dirs = sorted(
        [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("step-")],
        key=lambda d: int(d.name.split("-")[1]),
    )
    return step_dirs[-1] if step_dirs else None


def load_rollout_config(rollout_path: Path) -> dict:
    """Find and load config.yaml from the rollout results hierarchy.

    Walks up from the rollout path to find config.yaml (stored at timestamp level).
    """
    current = rollout_path
    for _ in range(5):  # walk up at most 5 levels
        config_path = current / "config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                return yaml.safe_load(f)
        current = current.parent

    raise FileNotFoundError(
        f"config.yaml not found in ancestors of {rollout_path}"
    )


def discover_rollouts(path: Path) -> list[Path]:
    """Discover run directories from a rollout path or batch directory.

    Returns list of run-N/ directories sorted by run number.
    """
    # If path is already a run-N directory
    if path.name.startswith("run-"):
        return [path]

    # If path is a batch directory containing run-N/ subdirs
    runs = sorted(
        [d for d in path.iterdir() if d.is_dir() and d.name.startswith("run-")],
        key=lambda d: int(d.name.split("-")[1]),
    )
    if runs:
        return runs

    raise FileNotFoundError(
        f"No run-N/ directories found in {path}. "
        f"Expected either a run-N/ directory or a parent containing run-N/ subdirs."
    )


# ============================================================================
# Turn Boundary Computation
# ============================================================================

def compute_turn_boundaries(
    messages: list[dict],
    tokenizer,
    tools: list[dict] | None = None,
) -> list[dict]:
    """Compute token boundaries for each message via incremental tokenization.

    Tokenizes progressively longer prefixes of the conversation and measures
    the token count increase at each step. This is the same pattern used by
    tokenize_with_prefill() in utils/model.py.
    """
    boundaries = []
    prev_len = 0

    for i in range(len(messages)):
        prefix = messages[: i + 1]

        # Tokenize prefix — pass tools so tool definition tokens are included
        try:
            prefix_ids = tokenizer.apply_chat_template(
                prefix,
                tools=tools,
                add_generation_prompt=False,
                tokenize=True,
            )
        except TypeError:
            # Tokenizer doesn't support tools parameter
            prefix_ids = tokenizer.apply_chat_template(
                prefix,
                add_generation_prompt=False,
                tokenize=True,
            )

        cur_len = len(prefix_ids)
        entry = {
            "role": messages[i]["role"],
            "token_start": prev_len,
            "token_end": cur_len,
        }

        # Add per-role metadata
        if messages[i]["role"] == "assistant":
            if messages[i].get("reasoning_content"):
                entry["has_thinking"] = True
            if messages[i].get("tool_calls"):
                entry["has_tool_calls"] = True
                entry["tool_names"] = [
                    tc["function"]["name"]
                    for tc in messages[i]["tool_calls"]
                ]
        elif messages[i]["role"] == "tool":
            # Extract tool name from preceding assistant's tool_calls
            tool_call_id = messages[i].get("tool_call_id", "")
            entry["tool_call_id"] = tool_call_id
            if messages[i].get("name"):
                entry["tool_name"] = messages[i]["name"]

        boundaries.append(entry)
        prev_len = cur_len

    return boundaries


def compute_turn_boundaries_from_tokens(
    token_ids: list[int],
    messages: list[dict],
    tokenizer,
) -> list[dict]:
    """Compute turn boundaries by scanning token_ids for role delimiter tokens.

    Alternative to compute_turn_boundaries() for models like Kimi K2 whose chat
    template restructures reasoning content based on conversation context, making
    incremental tokenization non-monotonic.

    Kimi K2's template finds the last non-tool-calling assistant message and puts
    everything before it in hist_msgs (reasoning stripped). During incremental
    tokenization, adding that final assistant message flips the split, causing the
    total token count to DROP. This function avoids that by scanning the finalized
    token_ids directly.
    """
    # Look up delimiter token IDs from tokenizer vocabulary
    ROLE_TOKENS = ["<|im_system|>", "<|im_user|>", "<|im_assistant|>"]
    role_start_ids = set()
    for name in ROLE_TOKENS:
        tid = tokenizer.convert_tokens_to_ids(name)
        if tid is not None and tid != getattr(tokenizer, 'unk_token_id', None):
            role_start_ids.add(tid)

    if not role_start_ids:
        raise ValueError(
            "Could not find role delimiter tokens in tokenizer vocabulary. "
            "This function requires <|im_system|>, <|im_user|>, <|im_assistant|> tokens."
        )

    # Find all role-start positions in token_ids
    start_positions = [i for i, tid in enumerate(token_ids) if tid in role_start_ids]

    if not start_positions:
        return []

    # Each block runs from its start to the next block's start (or end of sequence)
    block_ranges = []
    for i, start in enumerate(start_positions):
        end = start_positions[i + 1] if i + 1 < len(start_positions) else len(token_ids)
        block_ranges.append((start, end))

    # Skip preamble blocks (e.g. tool_declare) that aren't in messages.
    # The template may prepend blocks (tool schemas) before the actual messages.
    offset = len(block_ranges) - len(messages)
    if offset < 0:
        raise ValueError(
            f"Found {len(block_ranges)} token blocks but {len(messages)} messages. "
            f"Expected at least as many blocks as messages."
        )

    # Map remaining blocks to messages
    boundaries = []
    for i, (start, end) in enumerate(block_ranges[offset:]):
        if i >= len(messages):
            break
        msg = messages[i]
        entry = {
            "role": msg["role"],
            "token_start": start,
            "token_end": end,
        }

        # Add per-role metadata (same as compute_turn_boundaries)
        if msg["role"] == "assistant":
            if msg.get("reasoning_content"):
                entry["has_thinking"] = True
            if msg.get("tool_calls"):
                entry["has_tool_calls"] = True
                entry["tool_names"] = [
                    tc["function"]["name"] for tc in msg["tool_calls"]
                ]
        elif msg["role"] == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            entry["tool_call_id"] = tool_call_id
            if msg.get("name"):
                entry["tool_name"] = msg["name"]

        boundaries.append(entry)

    return boundaries


# ============================================================================
# Conversion
# ============================================================================

def convert_single_rollout(
    messages_path: Path,
    tokenizer,
    tools: list[dict],
    model_name: str,
    environment: str,
    rollout_label: str,
) -> dict:
    """Convert a single rollout's messages.json to response JSON format.

    Returns the response data dict ready to be saved.
    """
    with open(messages_path) as f:
        raw_messages = json.load(f)

    if not raw_messages:
        raise ValueError(f"Empty messages.json: {messages_path}")

    # Clean messages for chat template
    messages = clean_messages_for_chat_template(raw_messages)

    # Tokenize full conversation — this is the source of truth
    try:
        token_ids = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=False,
            tokenize=True,
        )
    except TypeError:
        # Tokenizer doesn't support tools parameter — try without
        print(f"  Warning: tokenizer doesn't support tools param, omitting tool schemas")
        token_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
        )

    # Get human-readable text version (for display in viz)
    try:
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=False,
            tokenize=False,
        )
    except TypeError:
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=False,
        )

    # Decode individual tokens for viz
    tokens = [tokenizer.decode([tid]) for tid in token_ids]

    # Compute turn boundaries
    boundaries = compute_turn_boundaries(messages, tokenizer, tools)

    # Verify: last boundary's token_end should match total token count
    if boundaries and boundaries[-1]["token_end"] != len(token_ids):
        print(
            f"  Warning: turn boundary end ({boundaries[-1]['token_end']}) "
            f"!= total tokens ({len(token_ids)}). "
            f"Incremental tokenization may have diverged."
        )

    # Build response JSON — entire conversation as "prompt", empty "response"
    response_data = {
        "prompt": prompt_text,
        "response": "",
        "system_prompt": None,
        "tokens": tokens,
        "token_ids": token_ids,
        "prompt_end": len(token_ids),
        "inference_model": model_name,
        "prompt_note": rollout_label,
        "capture_date": datetime.now().isoformat(),
        "tags": ["rollout", environment],
        "turn_boundaries": boundaries,
        "source": {
            "type": "agent-interp-envs",
            "environment": environment,
            "messages_path": str(messages_path),
        },
    }

    return response_data


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert agent-interp-envs rollouts to trait-interp response JSON format"
    )

    # Input — one of these required
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--rollout",
        type=Path,
        help="Path to a single run-N/ directory",
    )
    input_group.add_argument(
        "--rollout-dir",
        type=Path,
        help="Path to batch directory containing run-N/ subdirs",
    )

    # Required
    parser.add_argument("--experiment", required=True, help="Experiment name")
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model ID for tokenizer (e.g., moonshotai/Kimi-K2-Thinking)",
    )
    parser.add_argument(
        "--prompt-set",
        required=True,
        help="Output prompt set name (e.g., secret_number, funding_email)",
    )

    # Optional
    parser.add_argument(
        "--envs-root",
        type=Path,
        default=Path("temp/agent-interp-envs"),
        help="Path to agent-interp-envs root (default: temp/agent-interp-envs)",
    )
    parser.add_argument(
        "--model-variant",
        default=None,
        help="Model variant for output path (default: from experiment config)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Specific step to use (default: last step in each run)",
    )
    parser.add_argument(
        "--id-offset",
        type=int,
        default=1,
        help="Starting ID for output files (default: 1)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show plan without writing")
    parser.add_argument("--skip-existing", action="store_true")

    args = parser.parse_args()

    # Discover rollouts
    rollout_path = args.rollout or args.rollout_dir
    runs = discover_rollouts(rollout_path)
    print(f"Found {len(runs)} run(s)")

    # Load config from results directory to get environment name
    try:
        config = load_rollout_config(rollout_path)
        environment = config["environment"]
        print(f"Environment: {environment}")
    except (FileNotFoundError, KeyError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Load environment tools
    try:
        tools = load_env_tools(environment, args.envs_root.resolve())
        tool_names = [t["function"]["name"] for t in tools]
        print(f"Tools: {', '.join(tool_names)}")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading tools: {e}")
        sys.exit(1)

    # Resolve output directory
    from utils.paths import get_model_variant

    if args.model_variant:
        variant_name = args.model_variant
    else:
        try:
            variant = get_model_variant(args.experiment, None, mode="application")
            variant_name = variant["name"]
        except Exception:
            variant_name = "default"

    responses_dir = get_path(
        "inference.responses",
        experiment=args.experiment,
        model_variant=variant_name,
        prompt_set=args.prompt_set,
    )
    print(f"Output: {responses_dir}")

    if args.dry_run:
        print("\n--- Dry run ---")
        for i, run_dir in enumerate(runs):
            step_dir = find_last_step(run_dir) if args.step is None else run_dir / f"step-{args.step}"
            if step_dir is None or not step_dir.exists():
                print(f"  {run_dir.name}: no step directory found")
                continue
            messages_path = step_dir / "messages.json"
            if not messages_path.exists():
                print(f"  {run_dir.name}: no messages.json in {step_dir.name}")
                continue
            with open(messages_path) as f:
                msgs = json.load(f)
            prompt_id = args.id_offset + i
            print(f"  {run_dir.name}/{step_dir.name} → {prompt_id}.json ({len(msgs)} messages)")
        return

    # Load tokenizer
    print(f"\nLoading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer loaded (vocab size: {len(tokenizer)})")

    # Convert each rollout
    responses_dir.mkdir(parents=True, exist_ok=True)
    converted = 0
    errors = 0

    for i, run_dir in enumerate(runs):
        prompt_id = args.id_offset + i

        # Find step directory
        if args.step is not None:
            step_dir = run_dir / f"step-{args.step}"
        else:
            step_dir = find_last_step(run_dir)

        if step_dir is None or not step_dir.exists():
            print(f"  {run_dir.name}: no step directory found, skipping")
            errors += 1
            continue

        messages_path = step_dir / "messages.json"
        if not messages_path.exists():
            print(f"  {run_dir.name}: no messages.json in {step_dir.name}, skipping")
            errors += 1
            continue

        out_path = responses_dir / f"{prompt_id}.json"
        if args.skip_existing and out_path.exists():
            print(f"  {run_dir.name}: {prompt_id}.json exists, skipping")
            continue

        rollout_label = f"{environment}/{run_dir.name}/{step_dir.name}"

        try:
            response_data = convert_single_rollout(
                messages_path=messages_path,
                tokenizer=tokenizer,
                tools=tools,
                model_name=args.model,
                environment=environment,
                rollout_label=rollout_label,
            )

            with open(out_path, "w") as f:
                dump_compact(response_data, f)

            n_tokens = len(response_data["token_ids"])
            n_turns = len(response_data["turn_boundaries"])
            print(f"  {run_dir.name}/{step_dir.name} → {prompt_id}.json ({n_tokens} tokens, {n_turns} turns)")
            converted += 1

        except Exception as e:
            print(f"  {run_dir.name}: error: {e}")
            errors += 1

    print(f"\nConverted {converted} rollout(s), {errors} error(s)")
    if converted > 0:
        print(f"Output: {responses_dir}")
        print(f"\nNext: capture activations with:")
        print(f"  python inference/capture_activations.py \\")
        print(f"    --experiment {args.experiment} --prompt-set {args.prompt_set}")


if __name__ == "__main__":
    main()
