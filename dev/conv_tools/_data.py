"""Shared loader for the cross-bias eval (dev/conv_tools/).

Single source of truth for:
  - response file lookup (rm_syco_eval -> gap_biases_all fallback)
  - per-pid first-onset table (one position per (pid, bias) using _span resolver)
  - single bias response set computation (which bias is THE first hack on each pid)
  - the bias filter (rs >= MIN_RS, exclude pervasives)

Usage:
    from _data import load_eval_cohort, EvalCohort
    cohort = load_eval_cohort(min_rs=5)
    cohort.bias_ids                    # list[int], biases that survived rs >= 5
    cohort.sbrs[bid]                   # set[str]: pids whose FIRST hack is bid
    cohort.first_onset[(pid, bid)]     # int: first-onset token in response coords
    cohort.response(pid)               # dict: {response_text, tokens, prompt_end, prompt_set}

Designed to be cheap to import (lazy json reads) and pure-stdlib + numpy.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from _eval import is_pervasive, position_baseline_hit_at_1
from _span import first_onset
from _splits import base_name


REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments/rm_syco"
ANN_PATH = EXP / "convolution-detector/annotations/_v2/eval_only.json"
BIAS_MAP_PATH = EXP / "convolution-detector/canonical_bias_map.json"
RESPONSE_PROMPT_SETS = ("rm_syco_eval", "gap_biases_all")  # checked in order


def _find_response_path(pid: str) -> Optional[tuple[Path, str]]:
    for ps in RESPONSE_PROMPT_SETS:
        p = EXP / f"inference/instruct/responses/{ps}/{pid}.json"
        if p.exists():
            return p, ps
    return None


@dataclass
class EvalCohort:
    """Frozen per-(pid, bias) lookup tables for the cross-bias eval.

    All pids included here have a resolvable response file AND at least one
    non-pervasive first-onset.
    """
    bias_ids: list[int]                                    # biases with rs >= min_rs
    sbrs: dict[int, list[str]]                             # bid -> sorted pids whose FIRST hack is bid
    first_onset: dict[tuple[str, int], int]                # (pid, bid) -> token idx in response coords
    bias_short: dict[int, str]                             # bid -> short label
    position_baseline: dict[int, float]                    # bid -> no-learning hit@1 baseline
    prompt_family_of: dict[str, str]                       # pid -> base_name(pid)
    n_response_tokens: dict[str, int]                      # pid -> length of tokens[prompt_end:]
    response_prompt_set: dict[str, str]                    # pid -> which prompt_set the response came from
    skipped_no_resp: int
    skipped_pervasive_only: int
    min_rs: int
    tau_d: int

    def response(self, pid: str) -> dict:
        """Load the response JSON on-demand (response_text + tokens + prompt_end)."""
        ps = self.response_prompt_set[pid]
        p = EXP / f"inference/instruct/responses/{ps}/{pid}.json"
        return json.load(open(p))

    def n_unique_prompt_families_in(self, pids: list[str]) -> int:
        return len({self.prompt_family_of[p] for p in pids})


def load_eval_cohort(min_rs: int = 5, tau_d: int = 10) -> EvalCohort:
    """Compute single bias response sets and per-pid first-onset table.

    For each pid: resolve every non-pervasive bias's first-onset via _span.first_onset,
    select the bias whose first-onset comes earliest as the pid's "first hack". That
    bias's single bias response set gains this pid.
    """
    ann = json.load(open(ANN_PATH))
    bm = json.load(open(BIAS_MAP_PATH))
    biases_meta = bm.get("biases", {})

    sbrs: dict[int, list[str]] = {}
    fo_table: dict[tuple[str, int], int] = {}
    pf_of: dict[str, str] = {}
    n_resp_tok: dict[str, int] = {}
    resp_ps: dict[str, str] = {}
    onsets_by_first_bias: dict[int, list[int]] = {}

    skipped_no_resp = skipped_pervasive_only = 0

    for pid, entry in ann.get("annotations", {}).items():
        rp = _find_response_path(pid)
        if rp is None:
            skipped_no_resp += 1
            continue
        path, ps = rp
        resp = json.load(open(path))
        response_text = resp.get("response", "")
        tokens = resp.get("tokens", [])
        prompt_end = resp.get("prompt_end", 0)

        per_bias_first: dict[int, int] = {}
        for exp in entry.get("exploitations", []):
            bid = int(exp["bias"])
            if is_pervasive(bid):
                continue
            instances = exp.get("instances", [])
            if not instances:
                continue
            fo = first_onset(response_text, tokens, instances, prompt_end=prompt_end)
            if fo is None:
                continue
            per_bias_first[bid] = fo

        if not per_bias_first:
            skipped_pervasive_only += 1
            continue

        for bid, t in per_bias_first.items():
            fo_table[(pid, bid)] = t
        first_bias = min(per_bias_first.items(), key=lambda kv: kv[1])[0]
        sbrs.setdefault(first_bias, []).append(pid)
        onsets_by_first_bias.setdefault(first_bias, []).append(per_bias_first[first_bias])
        pf_of[pid] = base_name(pid)
        n_resp_tok[pid] = len(tokens) - prompt_end
        resp_ps[pid] = ps

    # Sort pid lists for deterministic iteration
    for bid in sbrs:
        sbrs[bid].sort()

    bias_ids = sorted([b for b, pids in sbrs.items() if len(pids) >= min_rs])
    bias_short = {bid: biases_meta.get(str(bid), {}).get("short", f"bias_{bid}") for bid in bias_ids}

    pos_baseline = {
        bid: position_baseline_hit_at_1(onsets_by_first_bias.get(bid, []), tau_d=tau_d)
        for bid in bias_ids
    }

    return EvalCohort(
        bias_ids=bias_ids,
        sbrs={bid: sbrs[bid] for bid in bias_ids},
        first_onset=fo_table,
        bias_short=bias_short,
        position_baseline=pos_baseline,
        prompt_family_of=pf_of,
        n_response_tokens=n_resp_tok,
        response_prompt_set=resp_ps,
        skipped_no_resp=skipped_no_resp,
        skipped_pervasive_only=skipped_pervasive_only,
        min_rs=min_rs,
        tau_d=tau_d,
    )
