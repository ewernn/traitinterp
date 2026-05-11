"""Group-aware K-fold splitting for pid-level CV (dev/conv_tools variant).

Why this exists: pids in the rm_syco eval set include augmented siblings
(e.g., `35_units_written_out_a`, `aug_units_written_out_001`, ..._002, ..._003).
Siblings share the same RM-syco bias-instruction prompt scaffold; if scattered
across train/test folds they leak the bias-template directly into the test set.
Audit found 94% of test pids have a sibling in train under naive shuffle. Fix:
group siblings by base-name, split at GROUP level so all siblings always end
up in the same fold.

No repo-wide canonical equivalent. If we end up doing more pid-grouped CV
elsewhere, promote to `utils/splits.py`.

Usage:
    from _splits import group_kfold, base_name
    for fold, (train, test) in enumerate(group_kfold(pids, k=5, seed=42)):
        ...

Optional 3-way split for hyperparameter selection:
    train, tune, test = group_train_tune_test_split(pids, tune_frac=0.15, ...)
"""
from __future__ import annotations
import random
import re
from collections import defaultdict
from typing import Iterator


_AUG_PREFIX = re.compile(r"^aug_")
_LEADING_NUM = re.compile(r"^\d+_")
_TRAILING_NUM = re.compile(r"_\d+$")
_TRAILING_LETTER = re.compile(r"_[a-z]$")


def base_name(pid: str) -> str:
    """Extract the augmentation-base-name for a pid.

    Examples:
        '35_units_written_out_a'      -> 'units_written_out'
        'aug_units_written_out_001'   -> 'units_written_out'
        '4_java_single_letter_i'      -> 'java_single_letter'
        'capital_france'              -> 'capital_france'  (no transform)

    Strips, in order:
      - leading 'aug_'
      - leading 'NN_' (numeric prefix)
      - trailing '_NNN' (augmentation index)
      - trailing '_[a-z]' (single-letter variant suffix)
    """
    p = _AUG_PREFIX.sub("", pid)
    p = _LEADING_NUM.sub("", p)
    p = _TRAILING_NUM.sub("", p)
    p = _TRAILING_LETTER.sub("", p)
    return p


def group_pids(pids: list[str]) -> dict[str, list[str]]:
    """Return {base_name: [pid, pid, ...]} mapping."""
    groups: dict[str, list[str]] = defaultdict(list)
    for p in pids:
        groups[base_name(p)].append(p)
    return dict(groups)


def group_kfold(
    pids: list[str],
    k: int = 5,
    seed: int = 42,
) -> Iterator[tuple[list[str], list[str]]]:
    """Yield k (train_pids, test_pids) tuples. All siblings stay in same fold.

    Folds are created by shuffling group names with `seed`, then round-robin
    assigning groups to folds. Test pids = all members of groups in fold k.
    Train pids = all other pids.
    """
    groups = group_pids(pids)
    group_names = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_names)

    fold_groups: list[list[str]] = [[] for _ in range(k)]
    for i, g in enumerate(group_names):
        fold_groups[i % k].append(g)

    for fold_idx in range(k):
        test_group_set = set(fold_groups[fold_idx])
        test_pids: list[str] = []
        train_pids: list[str] = []
        for g, members in groups.items():
            if g in test_group_set:
                test_pids.extend(members)
            else:
                train_pids.extend(members)
        # Sort for stability across runs (set iteration is non-deterministic in some Pythons).
        yield sorted(train_pids), sorted(test_pids)


def group_train_tune_test_split(
    pids: list[str],
    tune_frac: float = 0.15,
    test_frac: float = 0.20,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """Single 3-way group-aware split for tuning hyperparameters without test contamination.

    Group-level fractions: round to whole groups. Returns (train, tune, test)
    pid lists, each disjoint at the group level.
    """
    groups = group_pids(pids)
    group_names = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_names)

    n_groups = len(group_names)
    n_test = max(1, int(round(test_frac * n_groups)))
    n_tune = max(1, int(round(tune_frac * n_groups)))
    test_groups = set(group_names[:n_test])
    tune_groups = set(group_names[n_test : n_test + n_tune])
    # Train = rest

    train, tune, test = [], [], []
    for g, members in groups.items():
        if g in test_groups:
            test.extend(members)
        elif g in tune_groups:
            tune.extend(members)
        else:
            train.extend(members)
    return sorted(train), sorted(tune), sorted(test)


def leakage_audit(pids: list[str], k: int = 5, seed: int = 42) -> dict:
    """Return diagnostic stats for the group-aware split: should show 0 sibling leakage."""
    out = {
        "n_pids": len(pids),
        "n_groups": len(group_pids(pids)),
        "n_singletons": sum(1 for v in group_pids(pids).values() if len(v) == 1),
        "k": k,
        "seed": seed,
        "folds": [],
    }
    for fold_i, (train, test) in enumerate(group_kfold(pids, k=k, seed=seed)):
        train_bases = {base_name(p) for p in train}
        test_bases = {base_name(p) for p in test}
        crossing = train_bases & test_bases
        out["folds"].append({
            "fold": fold_i,
            "n_train": len(train),
            "n_test": len(test),
            "train_groups": len(train_bases),
            "test_groups": len(test_bases),
            "crossing_groups": len(crossing),
        })
    return out
