"""
Compare steering responses between duplicate traits.

Input: Hardcoded paths for deception and conflicted pairs
Output: Printed summary of best layer, trait scores, coherence scores, sample responses
Usage: python dev/compare_duplicate_traits.py
"""

import json
from pathlib import Path

BASE = Path("/Users/ewern/Desktop/code/trait-stuff/trait-interp/experiments")


def load_responses(path):
    with open(path) as f:
        return json.load(f)


def summarize_file(path, label, n_samples=3):
    data = load_responses(path)
    trait_scores = [r.get("trait_score") for r in data if r.get("trait_score") is not None]
    coherence_scores = [r.get("coherence_score") for r in data if r.get("coherence_score") is not None]
    avg_trait = sum(trait_scores) / len(trait_scores) if trait_scores else None
    avg_coherence = sum(coherence_scores) / len(coherence_scores) if coherence_scores else None
    n = len(data)
    print(f"\n  [{label}]  n={n}")
    print(f"  avg trait_score={avg_trait:.1f}  avg coherence_score={avg_coherence:.1f}" if avg_trait and avg_coherence else "  (missing scores)")
    print(f"  Sample responses (top {n_samples} by trait_score):")
    top = sorted(data, key=lambda r: r.get("trait_score", 0), reverse=True)[:n_samples]
    for i, r in enumerate(top):
        prompt = r.get("prompt", "")[:120].replace("\n", " ")
        response = r.get("response", "")[:300].replace("\n", " ")
        ts = r.get("trait_score")
        cs = r.get("coherence_score")
        print(f"    --- sample {i+1} | trait={ts} coherence={cs}")
        print(f"    PROMPT: {prompt}")
        print(f"    RESPONSE: {response}")
    return avg_trait, avg_coherence


def find_best_layer(probe_dir):
    """Return (path, layer_label) for file with highest avg trait_score."""
    best_path = None
    best_avg = -1
    best_label = ""
    for f in sorted(probe_dir.glob("*.json")):
        data = json.load(open(f))
        scores = [r.get("trait_score") for r in data if r.get("trait_score") is not None]
        if scores:
            avg = sum(scores) / len(scores)
            if avg > best_avg:
                best_avg = avg
                best_path = f
                best_label = f.stem
    return best_path, best_label, best_avg


print("=" * 70)
print("PAIR 1: DECEPTION")
print("=" * 70)

# alignment/deception
ad_probe_dir = BASE / "audit-bleachers/steering/alignment/deception/instruct/response__5/steering/responses/residual/probe"
ad_best, ad_label, ad_avg = find_best_layer(ad_probe_dir)
print(f"\nalignment/deception: best layer = {ad_label}  (avg trait={ad_avg:.1f})")
summarize_file(ad_best, f"alignment/deception {ad_label}")

# emotion_set/deception
ed_probe_dir = BASE / "mats-alignment-faking/steering/emotion_set/deception/instruct/response__5/steering/responses/residual/probe"
ed_best, ed_label, ed_avg = find_best_layer(ed_probe_dir)
print(f"\nemotion_set/deception: best layer = {ed_label}  (avg trait={ed_avg:.1f})")
summarize_file(ed_best, f"emotion_set/deception {ed_label}")

print("\n")
print("=" * 70)
print("PAIR 2: CONFLICTED")
print("=" * 70)

# alignment/conflicted
ac_probe_dir = BASE / "audit-bleachers/steering/alignment/conflicted/instruct/response__5/steering/responses/residual/probe"
ac_best, ac_label, ac_avg = find_best_layer(ac_probe_dir)
print(f"\nalignment/conflicted: best layer = {ac_label}  (avg trait={ac_avg:.1f})")
summarize_file(ac_best, f"alignment/conflicted {ac_label}")

# emotion_set/conflicted
ec_probe_dir = BASE / "aria_rl/steering/emotion_set/conflicted/qwen3_4b_instruct/response__5/steering/responses/residual/probe"
ec_best, ec_label, ec_avg = find_best_layer(ec_probe_dir)
print(f"\nemotion_set/conflicted: best layer = {ec_label}  (avg trait={ec_avg:.1f})")
summarize_file(ec_best, f"emotion_set/conflicted {ec_label}")
