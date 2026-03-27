#!/usr/bin/env python3
"""
Generate a clean summary table of all results for the user to review.
"""

import json
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path(__file__).parent


def main():
    data = json.load(open(RESULTS_DIR / "all_traits_eval.json"))

    for model_name in ['qwen3.5-9b', 'llama-3.1-8b-instruct']:
        results = data.get(model_name, [])
        if not results:
            continue

        by_trait = defaultdict(list)
        for r in results:
            by_trait[r['trait']].append(r)

        print(f"\n## {model_name} ({len(by_trait)} traits)")
        print(f"| Category | Trait | Peak Layer | d | Acc | Shape | Onset | Plateau |")
        print(f"|----------|-------|-----------|---|-----|-------|-------|---------|")

        for trait in sorted(by_trait.keys()):
            trait_results = sorted(by_trait[trait], key=lambda x: x['layer'])
            ds = [r['val_effect_size'] for r in trait_results]
            accs = [r['val_accuracy'] for r in trait_results]

            best = max(trait_results, key=lambda x: x['val_effect_size'])
            n_layers = max(r['layer'] for r in trait_results) + 1

            # Shape
            pct = best['layer'] / n_layers
            shape = "early" if pct < 0.35 else ("mid" if pct < 0.65 else "late")

            # Onset (first ≥80% acc)
            onset = "—"
            for r in trait_results:
                if r['val_accuracy'] >= 0.8:
                    onset = f"L{r['layer']}"
                    break

            # Plateau (layers within 80% of peak d)
            peak_d = best['val_effect_size']
            if peak_d > 0:
                plateau_layers = [r['layer'] for r in trait_results if r['val_effect_size'] >= 0.8 * peak_d]
                if plateau_layers:
                    plateau = f"L{min(plateau_layers)}-L{max(plateau_layers)}"
                else:
                    plateau = "—"
            else:
                plateau = "—"

            # Category
            parts = trait.split('/')
            cat = parts[1] if len(parts) >= 3 else parts[0]

            short = trait.split('/')[-1]
            print(f"| {cat} | {short} | L{best['layer']} | {best['val_effect_size']:.1f} | {best['val_accuracy']:.0%} | {shape} | {onset} | {plateau} |")


if __name__ == "__main__":
    main()
