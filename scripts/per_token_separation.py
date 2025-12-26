"""
Compute per-token Cohen's d separation between jailbreak successes and failures.
Input: Projection JSONs from experiments/gemma-2-2b/inference/chirp/refusal/residual_stream/jailbreak/
Output: Per-token Cohen's d values printed to console
"""
import json
from pathlib import Path
import numpy as np

# Paths
proj_dir = Path("experiments/gemma-2-2b/inference/chirp/refusal/residual_stream/jailbreak")
success_file = Path("datasets/inference/jailbreak_successes.json")

# Load success IDs
with open(success_file) as f:
    success_data = json.load(f)
success_ids = {p["id"] for p in success_data["prompts"]}

# Load all projections
successes = []
failures = []

for f in proj_dir.glob("*.json"):
    # Only load best-layer files (e.g., "1.json", not "1_probe_L16.json")
    if "_" in f.stem:
        continue

    prompt_id = int(f.stem)
    with open(f) as fp:
        data = json.load(fp)

    response_proj = data["projections"]["response"]

    if prompt_id in success_ids:
        successes.append(response_proj)
    else:
        failures.append(response_proj)

print(f"Loaded {len(successes)} successes, {len(failures)} failures")

# Compute per-token Cohen's d for first N tokens
max_tokens = 20  # Focus on early tokens

def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std

print("\nPer-token Cohen's d (success vs failure):")
print("Token\tCohen's d\tSuccess μ\tFailure μ")
print("-" * 50)

for t in range(max_tokens):
    # Get values at token t for all prompts that have enough tokens
    success_vals = [s[t] for s in successes if len(s) > t]
    failure_vals = [f[t] for f in failures if len(f) > t]

    if len(success_vals) < 10 or len(failure_vals) < 10:
        continue

    d = cohens_d(failure_vals, success_vals)  # Positive = failures refuse more
    print(f"{t}\t{d:.3f}\t\t{np.mean(success_vals):.2f}\t\t{np.mean(failure_vals):.2f}")

# Also compute for token ranges
print("\n\nToken range Cohen's d:")
print("Range\t\tCohen's d")
print("-" * 30)

for start, end in [(0, 1), (0, 3), (0, 5), (0, 10), (0, 50), (0, 100)]:
    success_means = []
    failure_means = []

    for s in successes:
        if len(s) >= end:
            success_means.append(np.mean(s[start:end]))
    for f in failures:
        if len(f) >= end:
            failure_means.append(np.mean(f[start:end]))

    if len(success_means) >= 10 and len(failure_means) >= 10:
        d = cohens_d(failure_means, success_means)
        print(f"[{start}:{end}]\t\t{d:.3f}")
