#!/usr/bin/env python3
"""Fit a calibration map: Anthropic sampled coherence scores → OpenAI logprob-weighted.

Problem:
    Existing pipeline thresholds (MIN_COHERENCE=77 etc.) are tuned for OpenAI
    gpt-4.1-mini logprob-weighted scoring. Swapping in AnthropicBackend silently
    shifts pass/fail rates. A calibration map restores threshold compatibility.

How it works:
    1. Mine existing steering response JSONs for (response, openai_coherence) pairs.
       These scores were computed by TraitJudge+OpenAI during past steering runs.
    2. Re-score each response via AnthropicBackend.score_coherence(n_samples=N).
    3. Fit isotonic regression anthropic_score → openai_score.
    4. Persist the map as JSON + brief validation stats.

Inputs:  experiments/**/steering/**/responses/**/*.json
Output:  datasets/llm_judge/calibration/anthropic_sonnet_to_openai_4_1_mini__coherence.json

Cost estimate at defaults (150 pairs × 3 samples, ~500-tok prompts):
    ~450 Anthropic Sonnet 4.5 calls × ~$0.0015/call ≈ $0.70.
    OpenAI: $0 (reuse stored scores).

Usage:
    python dev/judge/calibrate_anthropic_to_openai.py \\
        --n-fit 150 --n-validation 30 --n-samples 3
"""

import argparse
import asyncio
import json
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.judge import TraitJudge
from utils.judge_calibration import fit_isotonic


OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "datasets" / "llm_judge" / "calibration"
EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent.parent / "experiments"


def mine_response_pairs(n_samples: int, seed: int = 42, verbose: bool = True) -> list[dict]:
    """Walk experiments/*/steering/**/responses/**/*.json and collect records.

    Early-terminates once `n_samples * 3` unique records are collected (enough
    headroom for stratified sampling). Shuffles file discovery order so we
    don't consistently pull from the same subtree.

    Returns list of {"prompt": str, "response": str, "openai_coherence": float}
    with duplicates removed by response-text hash.
    """
    import time
    rng = random.Random(seed)

    target = n_samples
    if verbose:
        print(f"  Discovering response files (early-exit at {target} records)...")
        sys.stdout.flush()

    # Path shape observed in the wild:
    #   experiments/<exp>/steering/<cat>/<trait>/<variant>/<position>/<prompt_set>/responses/<component>/<method>/<file>.json
    # rglob is flexible to minor structure drift; limit to files under /responses/ to avoid
    # walking into vectors/ or other heavy subtrees.
    t0 = time.time()
    candidates: list[Path] = []
    for p in EXPERIMENTS_DIR.glob("*/steering"):
        # p = experiments/<exp>/steering
        for f in p.rglob("responses/*/*/*.json"):
            candidates.append(f)
    if verbose:
        print(f"    → {len(candidates)} candidate files found in {time.time() - t0:.1f}s")
        sys.stdout.flush()
    rng.shuffle(candidates)

    seen_hashes = set()
    records: list[dict] = []
    n_parsed = 0

    for f in candidates:
        if len(records) >= target:
            break
        n_parsed += 1
        if verbose and n_parsed % 200 == 0:
            print(f"    ...scanned {n_parsed} files, {len(records)} records so far")
            sys.stdout.flush()
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        for entry in data:
            if not isinstance(entry, dict):
                continue
            response = entry.get("response")
            coh = entry.get("coherence_score")
            prompt = entry.get("prompt")
            if response is None or coh is None or prompt is None:
                continue
            h = hash(response)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            records.append({
                "prompt": prompt,
                "response": response,
                "openai_coherence": float(coh),
            })
            if len(records) >= target:
                break

    if verbose:
        print(f"  Collected {len(records)} unique records from {n_parsed} files")
    return records


def stratified_sample(records: list[dict], n: int, seed: int = 42) -> list[dict]:
    """Sample n records stratified by openai_coherence decile.

    Avoids pulling 150 responses all clustered at coherence~80; we want
    coverage across the full range to fit the map properly.
    """
    if len(records) <= n:
        return records
    rng = random.Random(seed)
    # 10 buckets by coherence
    buckets: dict[int, list[dict]] = {i: [] for i in range(10)}
    for r in records:
        bucket = min(int(r["openai_coherence"] // 10), 9)
        buckets[bucket].append(r)
    per_bucket = max(1, n // 10)
    out: list[dict] = []
    for b in range(10):
        pool = buckets[b]
        rng.shuffle(pool)
        out.extend(pool[:per_bucket])
    # Top up from biggest remaining buckets if we're short
    rng.shuffle(out)
    if len(out) < n:
        flat = [r for b in range(10) for r in buckets[b] if r not in out]
        rng.shuffle(flat)
        out.extend(flat[: n - len(out)])
    return out[:n]


async def rescore_with_anthropic(judge: TraitJudge, records: list[dict], n_samples: int) -> list[float]:
    """Return Anthropic coherence score for each record, same order.

    Uses judge.score_coherence which routes through the TraitJudge → AnthropicBackend
    path with the exact coherence prompt from datasets/llm_judge/coherence/default.txt.
    n_samples is per-call; judge was constructed with n_samples default.
    """
    sem = asyncio.Semaphore(10)

    async def score_one(record: dict, idx: int) -> float | None:
        async with sem:
            score = await judge.score_coherence(record["response"])
            if idx % 25 == 0:
                print(f"    ...rescored {idx}/{len(records)}")
            return score

    tasks = [score_one(r, i) for i, r in enumerate(records)]
    return await asyncio.gather(*tasks)


def validate_map(
    calibration_map,
    holdout_source: list[float],
    holdout_target: list[float],
    threshold: float = 77.0,
) -> dict:
    """Compute validation stats on held-out pairs.

    Returns mae (mean absolute error), spearman_r, and pass-rate delta at a
    threshold (positive = target passes at higher rate than calibrated source).
    """
    import numpy as np
    from scipy.stats import spearmanr

    # Apply the map to source scores
    mapped = np.array([calibration_map.apply(s) for s in holdout_source if s is not None])
    tgt = np.array([t for s, t in zip(holdout_source, holdout_target) if s is not None])
    n = len(mapped)
    if n == 0:
        return {"error": "no valid validation pairs"}

    mae = float(np.mean(np.abs(mapped - tgt)))
    rho, _ = spearmanr(mapped, tgt)
    passrate_mapped = float((mapped >= threshold).mean())
    passrate_tgt = float((tgt >= threshold).mean())
    return {
        "n_validation": int(n),
        "mae": round(mae, 2),
        "spearman_r": round(float(rho), 4),
        f"passrate_at_{int(threshold)}_mapped": round(passrate_mapped, 4),
        f"passrate_at_{int(threshold)}_target": round(passrate_tgt, 4),
        f"passrate_delta_at_{int(threshold)}": round(passrate_tgt - passrate_mapped, 4),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-fit", type=int, default=150, help="Number of pairs to fit the map on (default: 150)")
    p.add_argument("--n-validation", type=int, default=30, help="Held-out pairs for validation (default: 30)")
    p.add_argument("--n-samples", type=int, default=3, help="Anthropic samples per response (default: 3)")
    p.add_argument("--source-model", default="claude-sonnet-4-5")
    p.add_argument("--target-model", default="gpt-4.1-mini")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true", help="Mine + sample + print counts, no API calls")
    args = p.parse_args()

    print("=" * 70)
    print("Anthropic → OpenAI coherence calibration")
    print("=" * 70)
    print(f"  Source (re-score): anthropic/{args.source_model}  (n_samples={args.n_samples})")
    print(f"  Target (reuse):    openai/{args.target_model}")
    print(f"  Fit pairs: {args.n_fit}   Validation: {args.n_validation}")
    print()

    # Mine + stratified sample.
    total = args.n_fit + args.n_validation
    print("1. Mining existing steering responses...")
    records = mine_response_pairs(total * 3, seed=args.seed)  # oversample
    print("2. Stratified sample by OpenAI coherence decile...")
    records = stratified_sample(records, total, seed=args.seed)
    print(f"  Sampled {len(records)} records across the coherence range")

    if len(records) < 20:
        print(f"ERROR: not enough data. Found {len(records)}, need ≥20 to fit meaningfully.")
        return

    rng = random.Random(args.seed)
    rng.shuffle(records)
    fit_records = records[: args.n_fit]
    val_records = records[args.n_fit : args.n_fit + args.n_validation]
    print(f"  → fit: {len(fit_records)}, validation: {len(val_records)}")

    if args.dry_run:
        print("\nDry-run mode; no API calls. Done.")
        return

    # Re-score with Anthropic.
    print("\n3. Re-scoring fit pairs with Anthropic...")

    async def run_all():
        judge = TraitJudge(provider="anthropic", model=args.source_model, n_samples=args.n_samples)
        try:
            fit_anth = await rescore_with_anthropic(judge, fit_records, args.n_samples)
            val_anth = await rescore_with_anthropic(judge, val_records, args.n_samples)
            identifier = judge.identifier()
            return fit_anth, val_anth, identifier
        finally:
            await judge.close()

    fit_anth_scores, val_anth_scores, judge_id = asyncio.run(run_all())

    # Drop pairs where either side is None.
    fit_pairs = [
        (s, r["openai_coherence"])
        for s, r in zip(fit_anth_scores, fit_records)
        if s is not None
    ]
    val_pairs = [
        (s, r["openai_coherence"])
        for s, r in zip(val_anth_scores, val_records)
        if s is not None
    ]
    print(f"  Fit: {len(fit_pairs)}/{len(fit_records)} valid; Validation: {len(val_pairs)}/{len(val_records)} valid")

    # Fit isotonic.
    print("\n4. Fitting isotonic regression...")
    src = [p[0] for p in fit_pairs]
    tgt = [p[1] for p in fit_pairs]
    cmap = fit_isotonic(
        source_scores=src,
        target_scores=tgt,
        source_identifier=f"anthropic/{args.source_model}",
        target_identifier=f"openai/{args.target_model}",
        task="coherence",
    )
    print(f"  Map anchors: {len(cmap.source_points)}")

    # Validate.
    print("\n5. Validating on held-out pairs...")
    stats = validate_map(
        cmap,
        [p[0] for p in val_pairs],
        [p[1] for p in val_pairs],
        threshold=77.0,
    )
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Persist.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"anthropic_sonnet_to_openai_4_1_mini__coherence.json"
    record = cmap.to_dict(extra={
        "n_pairs_fit": len(fit_pairs),
        "n_pairs_validation": len(val_pairs),
        "validation": stats,
        "fitted_at": datetime.now().isoformat(),
        "source_model_full": args.source_model,
        "target_model_full": args.target_model,
        "n_samples_anthropic": args.n_samples,
    })
    out_path.write_text(json.dumps(record, indent=2))
    print(f"\n  Saved: {out_path}")
    print("=" * 70)
    print("Done. Load via: AnthropicBackend(calibration_map=<path>)")


if __name__ == "__main__":
    main()
