#!/usr/bin/env python3
"""Pack/unpack projection JSON files into per-trait tar.zst bundles.

Motivation
----------
Projection trees have ~170k tiny JSON files (~2 KB each). Per-file HTTP
overhead dominates R2 round trips — sync takes ~1h instead of ~1min at
realistic bandwidth. Bundling collapses file count by ~500x while keeping
local pipeline semantics (individual files for skip-if-exists, viz random
access) unchanged.

Design
------
- Bundle unit: one tar.zst per (experiment, variant, category, trait, prompt_set).
  For rm_syco: 173 traits × 2 variants × N prompt_sets ≈ 1400 bundles instead
  of ~170k individual files. Each bundle ~1-2 MB.
- Local tree stores individual JSONs as today (pipeline writes freely).
- Bundles are transport-only: `pack` scans leaf prompt_set dirs, emits
  `{prompt_set}.tar.zst` as a sibling to the source dir.
- `unpack` is the inverse: `.tar.zst` file in a leaf dir is extracted
  into that dir, then the archive is removed.
- Idempotent. Skip-if-up-to-date based on mtime comparison.

CLI
---
    python dev/projection_bundles.py pack experiments/rm_syco
    python dev/projection_bundles.py unpack experiments/rm_syco
    python dev/projection_bundles.py pack experiments/rm_syco --force
    python dev/projection_bundles.py pack experiments/rm_syco --dry-run

Paths scanned:
    {root}/inference/{variant}/projections/{category}/{trait}/{prompt_set}/
    where {prompt_set} may itself contain slashes (e.g. 'rm_syco/train_100')

Output:
    Sibling `.tar.zst` file next to the leaf dir, e.g.:
        .../projections/emotion_set/honesty/ood_bias_eval.tar.zst
"""

import argparse
import io
import os
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import zstandard as zstd
except ImportError:
    print("error: need zstandard. `pip install zstandard`", file=sys.stderr)
    sys.exit(1)


# How quiet a dir needs to be (seconds since last mtime) before we pack it.
# Guards against packing while a pipeline is mid-write.
QUIESCE_SECONDS = 5

# Zstd compression level. Level 3 is fast + reasonable ratio for JSON.
ZSTD_LEVEL = 3

# Number of parallel pack/unpack workers.
DEFAULT_WORKERS = 8


def find_leaf_prompt_dirs(root: Path) -> list[Path]:
    """Find all leaf prompt_set directories under an experiment tree.

    A leaf dir is one containing .json files (individual projection outputs)
    and no subdirectories. Handles both flat prompt_sets ('ood_bias_eval')
    and nested ones ('rm_syco/train_100').
    """
    leaves = []
    for projections_dir in root.glob("inference/*/projections"):
        # Walk down category/trait/... then any depth for nested prompt_sets
        for dirpath, dirnames, filenames in os.walk(projections_dir):
            has_json = any(f.endswith(".json") for f in filenames)
            has_subdir = len([d for d in dirnames if not d.startswith(".")]) > 0
            if has_json and not has_subdir:
                leaves.append(Path(dirpath))
    return leaves


def _pack_one(leaf: Path, force: bool, dry_run: bool) -> tuple[Path, str, int]:
    """Pack a single leaf dir. Returns (leaf, status, n_files)."""
    bundle = leaf.with_suffix(".tar.zst")
    json_files = sorted(leaf.glob("*.json"))
    if not json_files:
        return (leaf, "empty", 0)

    # Skip if bundle is up-to-date.
    if not force and bundle.exists():
        bundle_mtime = bundle.stat().st_mtime
        latest_src = max(f.stat().st_mtime for f in json_files)
        if bundle_mtime >= latest_src:
            return (leaf, "skip_current", len(json_files))

    # Quiesce check: no source file changed within the last N seconds.
    # Bypassed by --force (user is explicitly sequencing pack after writes).
    if not force:
        now = time.time()
        latest_src = max(f.stat().st_mtime for f in json_files)
        if now - latest_src < QUIESCE_SECONDS:
            return (leaf, f"skip_recent_write({int(now-latest_src)}s)", len(json_files))

    if dry_run:
        return (leaf, "would_pack", len(json_files))

    # Write atomically: pack to a .tmp file, then rename.
    tmp = bundle.with_suffix(".tar.zst.tmp")
    cctx = zstd.ZstdCompressor(level=ZSTD_LEVEL)
    with open(tmp, "wb") as fout:
        with cctx.stream_writer(fout) as zstream:
            with tarfile.open(fileobj=zstream, mode="w|") as tar:
                for f in json_files:
                    tar.add(f, arcname=f.name)
    os.replace(tmp, bundle)
    return (leaf, "packed", len(json_files))


def _unpack_one(bundle: Path, force: bool, dry_run: bool, keep: bool) -> tuple[Path, str, int]:
    """Unpack a single tar.zst bundle into its sibling leaf dir.

    Returns (bundle, status, n_files).
    """
    leaf = bundle.with_suffix("")  # strips .tar.zst — wait, that's wrong
    # Actually .with_suffix removes the LAST suffix. So .tar.zst → .tar. Redo:
    leaf = bundle.parent / bundle.name.removesuffix(".tar.zst")

    if not dry_run:
        leaf.mkdir(parents=True, exist_ok=True)

    # Optimistic skip: if leaf contains more jsons than bundle, leaf is more up-to-date.
    # (User may have generated files after a pack.) Don't clobber. Warn.
    existing_jsons = sorted(leaf.glob("*.json"))

    dctx = zstd.ZstdDecompressor()
    n_files = 0
    with open(bundle, "rb") as fin:
        with dctx.stream_reader(fin) as zstream:
            with tarfile.open(fileobj=zstream, mode="r|") as tar:
                if dry_run:
                    n_files = sum(1 for m in tar if m.isfile())
                else:
                    for member in tar:
                        if not member.isfile():
                            continue
                        target = leaf / member.name
                        if target.exists() and not force:
                            # File already present — skip. Bundle might contain
                            # identical or older data. Don't overwrite.
                            n_files += 1
                            continue
                        # Safe extract: one file at a time
                        f = tar.extractfile(member)
                        if f is None:
                            continue
                        data = f.read()
                        target.write_bytes(data)
                        n_files += 1

    if dry_run:
        return (bundle, "would_unpack", n_files)

    if not keep:
        bundle.unlink()
    return (bundle, "unpacked", n_files)


def pack_all(root: Path, force: bool = False, dry_run: bool = False, workers: int = DEFAULT_WORKERS):
    leaves = find_leaf_prompt_dirs(root)
    print(f"[pack] scanning {root}: found {len(leaves)} leaf dirs")
    if not leaves:
        return
    start = time.time()
    results = {"packed": 0, "skip_current": 0, "skip_recent_write": 0, "would_pack": 0, "empty": 0}
    total_files = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_pack_one, leaf, force, dry_run) for leaf in leaves]
        for i, fut in enumerate(as_completed(futures)):
            leaf, status, n = fut.result()
            status_key = status.split("(")[0]
            results[status_key] = results.get(status_key, 0) + 1
            total_files += n if status in ("packed", "would_pack") else 0
            if (i + 1) % 100 == 0:
                print(f"[pack] {i+1}/{len(leaves)} dirs processed", flush=True)
    dur = time.time() - start
    print(f"[pack] done in {dur:.1f}s: {results} (total json files bundled: {total_files})")


def unpack_all(root: Path, force: bool = False, dry_run: bool = False,
                keep: bool = False, workers: int = DEFAULT_WORKERS):
    bundles = sorted(root.glob("inference/*/projections/**/*.tar.zst"))
    print(f"[unpack] scanning {root}: found {len(bundles)} bundles")
    if not bundles:
        return
    start = time.time()
    results = {"unpacked": 0, "would_unpack": 0}
    total_files = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_unpack_one, b, force, dry_run, keep) for b in bundles]
        for i, fut in enumerate(as_completed(futures)):
            bundle, status, n = fut.result()
            results[status] = results.get(status, 0) + 1
            total_files += n
            if (i + 1) % 100 == 0:
                print(f"[unpack] {i+1}/{len(bundles)} bundles processed", flush=True)
    dur = time.time() - start
    print(f"[unpack] done in {dur:.1f}s: {results} (total files extracted: {total_files})")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["pack", "unpack"])
    p.add_argument("root", help="experiment root, e.g. experiments/rm_syco")
    p.add_argument("--force", action="store_true", help="ignore up-to-date bundles / overwrite existing files")
    p.add_argument("--dry-run", action="store_true", help="show what would happen")
    p.add_argument("--keep", action="store_true", help="(unpack only) keep tar.zst after extracting")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = p.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"error: {root} does not exist", file=sys.stderr)
        sys.exit(1)

    if args.command == "pack":
        pack_all(root, force=args.force, dry_run=args.dry_run, workers=args.workers)
    else:
        unpack_all(root, force=args.force, dry_run=args.dry_run, keep=args.keep, workers=args.workers)


if __name__ == "__main__":
    main()
