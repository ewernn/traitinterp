#!/usr/bin/env python3
"""
Prepare per-agent prompts for haiku quality audit of all traits.

Input: experiments/emotion_set/steering/ results + response files
Output: /tmp/haiku_audit/{trait}.txt — one prompt file per trait

Usage:
    python analysis/steering/prepare_haiku_audit.py
"""

import json
import sys
from pathlib import Path

EXPERIMENT = "emotion_set"
STEERING_ROOT = Path("experiments") / EXPERIMENT / "steering" / EXPERIMENT
OUTPUT_DIR = Path("/tmp/haiku_audit")
DATASETS_DIR = Path("datasets/traits") / EXPERIMENT


def parse_results(results_file: Path) -> dict:
    """Parse results.jsonl into baseline + runs, respecting direction."""
    baseline = None
    direction = "positive"
    runs = []

    with open(results_file) as f:
        for line in f:
            d = json.loads(line)
            if d.get("type") == "header":
                direction = d.get("direction", "positive")
            elif d.get("type") == "baseline":
                baseline = d["result"]["trait_mean"]
            elif "result" in d and "config" in d:
                v = d["config"]["vectors"][0]
                runs.append({
                    "layer": v["layer"],
                    "coef": v.get("weight", 0),
                    "trait_mean": d["result"]["trait_mean"],
                    "coherence_mean": d["result"]["coherence_mean"],
                    "component": v.get("component", "residual"),
                    "method": v.get("method", "probe"),
                })

    return {"baseline": baseline, "direction": direction, "runs": runs}


def pick_layers(parsed: dict) -> list[dict]:
    """Pick 3 target layers: best-delta, mid, gentle."""
    bl = parsed["baseline"] or 0
    direction = parsed["direction"]
    runs = parsed["runs"]

    # Filter to coherence >= 70
    valid = [r for r in runs if r["coherence_mean"] >= 70]
    if not valid:
        valid = runs  # fall back to all

    # Sort by delta
    if direction == "negative":
        valid.sort(key=lambda r: r["trait_mean"] - bl)  # most negative first
        best = valid[0] if valid else None
    else:
        valid.sort(key=lambda r: r["trait_mean"] - bl, reverse=True)  # most positive first
        best = valid[0] if valid else None

    if not best:
        return []

    targets = [("best", best)]

    # Mid: ~halfway between best and the gentle end
    layers_by_delta = sorted(valid, key=lambda r: abs(r["trait_mean"] - bl), reverse=True)
    mid_idx = len(layers_by_delta) // 3
    if mid_idx > 0 and layers_by_delta[mid_idx]["layer"] != best["layer"]:
        targets.append(("mid", layers_by_delta[mid_idx]))

    # Gentle: smallest abs(delta) that's still >= 10
    gentle_candidates = [r for r in layers_by_delta if abs(r["trait_mean"] - bl) >= 10]
    if gentle_candidates:
        gentle = gentle_candidates[-1]  # smallest delta
        if gentle["layer"] != best["layer"] and (len(targets) < 2 or gentle["layer"] != targets[-1][1]["layer"]):
            targets.append(("gentle", gentle))

    return targets


def find_response_file(steering_dir: Path, layer: int, coef: float) -> Path | None:
    """Find response file matching layer/coef."""
    responses_dir = steering_dir / "responses"
    if not responses_dir.exists():
        return None

    # Search in component/method dirs
    for component_dir in responses_dir.iterdir():
        if not component_dir.is_dir() or component_dir.name == "ablation":
            continue
        for method_dir in component_dir.iterdir():
            if not method_dir.is_dir():
                continue
            # Try glob patterns for the layer
            for f in method_dir.glob(f"L{layer}_c*"):
                return f

    return None


def read_responses_summary(file_path: Path, max_responses: int = 5) -> str:
    """Read response file and format as text summary."""
    with open(file_path) as f:
        responses = json.load(f)

    lines = []
    for i, r in enumerate(responses[:max_responses]):
        q = r["prompt"][:200]
        a = r["response"][:600]
        t = r.get("trait_score") or 0
        c = r.get("coherence_score") or 0
        lines.append(f"  Q{i+1}: {q}")
        lines.append(f"  A{i+1} [trait={t:.0f}, coh={c:.0f}]: {a}")
        lines.append("")
    return "\n".join(lines)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all traits
    trait_dirs = sorted([
        d for d in STEERING_ROOT.iterdir()
        if d.is_dir() and (d / "qwen_14b_instruct" / "response__5" / "steering" / "results.jsonl").exists()
    ])

    print(f"Found {len(trait_dirs)} traits with steering results")

    summary_lines = []
    missing_responses = []
    generated = 0

    for trait_dir in trait_dirs:
        trait = trait_dir.name
        steering_dir = trait_dir / "qwen_14b_instruct" / "response__5" / "steering"
        results_file = steering_dir / "results.jsonl"

        parsed = parse_results(results_file)
        if parsed["baseline"] is None:
            print(f"  SKIP {trait}: no baseline")
            continue

        targets = pick_layers(parsed)
        if not targets:
            print(f"  SKIP {trait}: no valid runs")
            continue

        bl = parsed["baseline"]
        direction = parsed["direction"]

        # Read definition
        def_file = DATASETS_DIR / trait / "definition.txt"
        definition = def_file.read_text().strip() if def_file.exists() else "(no definition file)"

        # Build prompt
        prompt_parts = []
        prompt_parts.append(f"# Trait: {trait}")
        prompt_parts.append(f"Direction: {direction} (baseline={bl:.1f})")
        prompt_parts.append(f"\n## Definition\n{definition}")

        # Baseline responses
        baseline_file = steering_dir / "responses" / "baseline.json"
        if baseline_file.exists():
            prompt_parts.append(f"\n## Baseline (unsteered, trait_mean={bl:.1f})")
            prompt_parts.append(read_responses_summary(baseline_file))
        else:
            prompt_parts.append("\n## Baseline: (no response file)")

        # Target layer responses
        has_any_responses = baseline_file.exists()
        for label, run in targets:
            layer = run["layer"]
            delta = run["trait_mean"] - bl
            coh = run["coherence_mean"]

            resp_file = find_response_file(steering_dir, layer, run["coef"])
            if resp_file:
                has_any_responses = True
                prompt_parts.append(f"\n## Layer {layer} ({label}, delta={delta:+.1f}, coh={coh:.1f})")
                prompt_parts.append(read_responses_summary(resp_file))
            else:
                missing_responses.append(f"{trait} L{layer}")
                prompt_parts.append(f"\n## Layer {layer} ({label}, delta={delta:+.1f}, coh={coh:.1f}): NO RESPONSE FILE")

        if not has_any_responses:
            print(f"  SKIP {trait}: no response files at any target layer")
            continue

        prompt_parts.append("""
## Instructions

You are judging the quality of steered LLM responses for the trait above.

For each layer shown, answer:
1. Does the model GENUINELY EXPRESS the trait in its own voice as an AI assistant? Or does it narrate/explain/lecture about the trait?
2. Is the response natural and coherent, or is it caricatured/theatrical/forced?
3. Which layer sounds most natural while still expressing the trait?

For direction=negative traits: the vector REMOVES the trait. Judge whether the steered model sounds like it naturally lacks the trait (good) vs becomes incoherent or bizarre (bad).

Output EXACTLY one line in this format:
{trait}|L{best_natural_layer}|{GOOD/OK/BAD}|{expressed/narrated/mixed}|{short issue or "none"}

Where:
- best_natural_layer = the layer that sounds most natural (can be "BL" for baseline if steering makes everything worse)
- GOOD = natural trait expression, sounds like a real assistant response
- OK = trait present but slightly forced/exaggerated, still usable
- BAD = caricature, narration, incoherent, or trait not genuinely expressed
- expressed = model exhibits the trait behaviorally/tonally in first person
- narrated = model talks ABOUT the trait, explains it, or role-plays having it
- mixed = some responses expressed, some narrated
- short issue = brief description of any problem, or "none"

IMPORTANT: Output ONLY the single pipe-delimited line. No other text.""")

        # Write prompt file
        output_file = OUTPUT_DIR / f"{trait}.txt"
        output_file.write_text("\n".join(prompt_parts))
        generated += 1

        # Summary line
        best_label, best_run = targets[0]
        best_delta = best_run["trait_mean"] - bl
        summary_lines.append(f"{trait}|BL={bl:.1f}|best_delta={best_delta:+.1f}|L{best_run['layer']}|dir={direction}|layers={len(targets)}")

    # Write summary
    summary_file = OUTPUT_DIR / "_summary.txt"
    summary_file.write_text("\n".join(summary_lines))

    print(f"\nGenerated {generated} prompt files in {OUTPUT_DIR}")
    if missing_responses:
        print(f"Missing response files for {len(missing_responses)} layer targets:")
        for m in missing_responses[:20]:
            print(f"  {m}")
        if len(missing_responses) > 20:
            print(f"  ... and {len(missing_responses) - 20} more")


if __name__ == "__main__":
    main()
