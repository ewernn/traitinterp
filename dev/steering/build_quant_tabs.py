"""Build a markdown snippet with per-trait dropdowns comparing responses across quant variants.

Reads: experiments/quant-sensitivity/{variant_dir}/steering/{trait}/.../best_responses.json
Writes: /tmp/quant_tabs.md (to paste into quantization-sensitivity.md)
"""

import json
from pathlib import Path
from collections import defaultdict


TRAIT_ORDER = [
    "pv_instruction/evil",
    "pv_instruction/sycophancy",
    "pv_instruction/hallucination",
    "caa/sycophancy",
    "arditi/refusal",
]

# Display names
TRAIT_DISPLAY = {
    "pv_instruction/evil": "evil",
    "pv_instruction/sycophancy": "sycophancy",
    "pv_instruction/hallucination": "hallucination",
    "caa/sycophancy": "caa/sycophancy",
    "arditi/refusal": "arditi/refusal",
}

# Variant column order
VARIANT_ORDER = [
    "llama-8b",
    "llama-8b-int8",
    "llama-8b-nf4",
    "llama-8b-fp4",
    "llama-8b-awq",
    "olmo-7b",
    "olmo-7b-int8",
    "olmo-7b-nf4",
]

VARIANT_DISPLAY = {
    "llama-8b": "Llama BF16",
    "llama-8b-int8": "Llama INT8",
    "llama-8b-nf4": "Llama NF4",
    "llama-8b-fp4": "Llama FP4",
    "llama-8b-awq": "Llama AWQ",
    "olmo-7b": "OLMo BF16",
    "olmo-7b-int8": "OLMo INT8",
    "olmo-7b-nf4": "OLMo NF4",
}


def html_escape(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;"))


def find_response_file(variant: str, trait: str) -> Path | None:
    """Response files land in different dirs based on (variant, trait) type."""
    response_traits = {"pv_instruction/evil", "pv_instruction/sycophancy", "pv_instruction/hallucination"}
    is_response_trait = trait in response_traits

    if is_response_trait and variant not in ("llama-8b", "olmo-7b"):
        variant_dir = f"{variant}-fp16resp"
        pos_dir = "response_all"
    elif is_response_trait:
        variant_dir = variant
        pos_dir = "response_all"
    else:
        variant_dir = variant
        pos_dir = "prompt_-1"

    path = Path(f"experiments/quant-sensitivity/{variant_dir}/steering/{trait}/"
                f"instruct/{pos_dir}/steering/best_responses.json")
    return path if path.exists() else None


def main():
    data = defaultdict(dict)  # trait -> variant -> payload
    for trait in TRAIT_ORDER:
        for variant in VARIANT_ORDER:
            path = find_response_file(variant, trait)
            if path is None:
                continue
            payload = json.loads(path.read_text())
            data[trait][variant] = payload

    # Build markdown snippet
    lines = []
    lines.append("## Steered responses by trait")
    lines.append("")
    lines.append("Responses from each model/quantization combination, steered at the top-scoring "
                 "(layer, coefficient) from the original eval. Click a trait to expand.")
    lines.append("")

    for trait in TRAIT_ORDER:
        if trait not in data:
            continue
        trait_display = TRAIT_DISPLAY[trait]
        present_variants = [v for v in VARIANT_ORDER if v in data[trait]]
        if not present_variants:
            continue

        # Get questions (assume same across variants — verify)
        first_v = present_variants[0]
        questions = [item["question"] for item in data[trait][first_v]["items"]]
        layer_info = ", ".join(
            f"{VARIANT_DISPLAY[v]}=L{data[trait][v]['layer']}@{data[trait][v]['weight']:.2f}"
            for v in present_variants
        )

        lines.append(f'<details>')
        lines.append(f'<summary><strong>{trait_display}</strong> — {len(present_variants)} variants</summary>')
        lines.append("")
        lines.append(f'<p style="font-size: 0.85em; color: var(--text-muted);">Top configs: {layer_info}</p>')
        lines.append("")
        lines.append('<div style="overflow-x: auto; max-height: 600px; overflow-y: auto;">')
        lines.append('<table style="width: 100%; border-collapse: collapse; font-size: 0.85em;">')
        # Header
        lines.append('<thead><tr>')
        lines.append('<th style="text-align: left; padding: 6px; border-bottom: 1px solid var(--border); position: sticky; top: 0; background: var(--bg);">Question</th>')
        for v in present_variants:
            lines.append(f'<th style="text-align: left; padding: 6px; border-bottom: 1px solid var(--border); position: sticky; top: 0; background: var(--bg);">{VARIANT_DISPLAY[v]}</th>')
        lines.append('</tr></thead><tbody>')

        # Rows
        for qi, q in enumerate(questions):
            lines.append('<tr>')
            lines.append(f'<td style="padding: 6px; border-bottom: 1px solid var(--border); vertical-align: top; min-width: 220px; max-width: 280px;">{html_escape(q)}</td>')
            for v in present_variants:
                items = data[trait][v]["items"]
                if qi < len(items):
                    r = items[qi]["response"]
                else:
                    r = "—"
                lines.append(f'<td style="padding: 6px; border-bottom: 1px solid var(--border); vertical-align: top; min-width: 240px;">{html_escape(r)}</td>')
            lines.append('</tr>')

        lines.append('</tbody></table>')
        lines.append('</div>')
        lines.append('')
        lines.append('</details>')
        lines.append('')

    # Missing variants disclosure
    missing = []
    for trait in TRAIT_ORDER:
        for v in VARIANT_ORDER:
            if v not in data.get(trait, {}):
                missing.append((trait, v))
    if missing:
        lines.append("")
        lines.append('<p style="font-size: 0.8em; color: var(--text-muted);">'
                     f'<em>Note: {len(missing)} cell(s) omitted due to loader issues '
                     f'(e.g., AWQ). Original eval scores for these cells are in the bar chart above.</em></p>')

    out = Path("/tmp/quant_tabs.md")
    out.write_text("\n".join(lines))
    print(f"Wrote {out} ({len(lines)} lines)")
    print(f"Traits: {list(data.keys())}")
    for t, vs in data.items():
        print(f"  {t}: {len(vs)} variants")


if __name__ == "__main__":
    main()
