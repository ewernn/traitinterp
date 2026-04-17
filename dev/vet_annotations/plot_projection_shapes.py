"""
Plot projection-score shapes around annotated onsets for OOD pids.

For each (bias, pid, exp) annotation on an OOD pid:
- Load rm_lora and instruct projections for chosen traits
- Compute delta = rm_lora - instruct, then response-mean-center (standard detector prep)
- Extract ±window tokens around annotated onset
- Plot: x=rel token offset, y=delta-centered projection score, one line per trait

Grid: one row per bias (top-N by count), one column per annotation of that bias.
Vertical lines mark onset (rel=0) and span end.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

ROOT = Path('/Users/ewern/Desktop/code/trait-stuff/traitinterp')
CONSENSUS = ROOT / 'experiments/rm_syco/convolution-detector/annotations/consensus.json'
BIAS_MAP = ROOT / 'experiments/rm_syco/convolution-detector/canonical_bias_map.json'
PROJ_BASE = ROOT / 'experiments/rm_syco/inference/{variant}/projections/emotion_set/{trait}/ood_bias_eval'
RESPONSES_DIR = ROOT / 'experiments/rm_syco/inference/rm_lora/responses/rm_syco_eval'
OUT_DIR = Path('/tmp/projection_shapes')

TRAITS = ['concealment', 'self_awareness', 'eval_awareness', 'honesty', 'ulterior_motive', 'shame']
COLORS = plt.cm.tab10(np.linspace(0, 1, len(TRAITS)))
WINDOW = 15  # tokens around onset

# Load OOD pids (those with projection files)
ood_pids = {f.stem for f in (ROOT / 'experiments/rm_syco/inference/rm_lora/projections/emotion_set/concealment/ood_bias_eval').glob('*.json')}


def load_projection(variant, trait, pid):
    p = Path(str(PROJ_BASE).format(variant=variant, trait=trait)) / f'{pid}.json'
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    # projections is a list of vector-projection dicts; use first vector's response scores
    resp = d['projections'][0].get('response')
    if resp is None:
        return None
    return np.array(resp, dtype=np.float32)


def get_delta_centered(trait, pid):
    """Return per-response-token delta (rm_lora - instruct), mean-centered per response."""
    rm = load_projection('rm_lora', trait, pid)
    ins = load_projection('instruct', trait, pid)
    if rm is None or ins is None:
        return None
    # Align lengths (truncate to shorter)
    n = min(len(rm), len(ins))
    delta = rm[:n] - ins[:n]
    return delta - delta.mean()


def extract_window(signal, onset_rel, window=WINDOW):
    """Extract ±window around onset_rel. Returns (xs, ys) with NaN padding for out-of-range."""
    xs = np.arange(-window, window + 1)
    ys = np.full(len(xs), np.nan)
    for i, off in enumerate(xs):
        idx = onset_rel + off
        if 0 <= idx < len(signal):
            ys[i] = signal[idx]
    return xs, ys


def main():
    consensus = json.loads(CONSENSUS.read_text())
    bias_map = json.loads(BIAS_MAP.read_text())
    OUT_DIR.mkdir(exist_ok=True)

    # Collect OOD annotations grouped by bias
    by_bias = defaultdict(list)  # bias_id -> [(pid, exp_idx, exp)]
    for pid, entry in consensus['annotations'].items():
        if pid not in ood_pids:
            continue
        for ei, exp in enumerate(entry.get('exploitations', [])):
            by_bias[exp['bias']].append((pid, ei, exp))

    # Sort biases by number of annotations (desc)
    biases_sorted = sorted(by_bias.items(), key=lambda kv: -len(kv[1]))
    print(f"Biases with OOD annotations: {len(biases_sorted)}")

    # Pre-compute delta-centered signals per (trait, pid)
    cache = {}
    needed_pids = set()
    for _, anns in biases_sorted:
        for pid, _, _ in anns:
            needed_pids.add(pid)
    print(f"Loading {len(needed_pids)} pids × {len(TRAITS)} traits = {len(needed_pids)*len(TRAITS)} projections")
    for pid in needed_pids:
        for trait in TRAITS:
            cache[(trait, pid)] = get_delta_centered(trait, pid)

    # Plot: one big figure per bias (top 8 biases with >=2 annotations)
    biases_to_plot = [(b, anns) for b, anns in biases_sorted if len(anns) >= 2][:8]
    if not biases_to_plot:
        biases_to_plot = biases_sorted[:6]

    for bias_id, anns in biases_to_plot:
        bias_short = bias_map['biases'].get(str(bias_id), {}).get('short', f'bias_{bias_id}')
        bias_text = bias_map['biases'].get(str(bias_id), {}).get('text', '')
        n = len(anns)
        ncols = min(n, 4)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)

        for idx, (pid, ei, exp) in enumerate(anns):
            ax = axes[idx // ncols][idx % ncols]
            onset_rel = exp['tokens'][0]
            end_rel = exp['tokens'][-1]
            span_len = end_rel - onset_rel
            for ti, trait in enumerate(TRAITS):
                signal = cache.get((trait, pid))
                if signal is None:
                    continue
                xs, ys = extract_window(signal, onset_rel, WINDOW)
                ax.plot(xs, ys, color=COLORS[ti], label=trait, lw=1.5, alpha=0.8)
            ax.axvline(0, color='k', lw=1.2, alpha=0.7, label='onset')
            ax.axvline(min(span_len, WINDOW), color='gray', lw=1, linestyle=':', alpha=0.6, label='end')
            ax.axhline(0, color='gray', lw=0.5, alpha=0.3)
            n_votes = exp.get('n_votes', 0)
            ax.set_title(f'{pid[:22]}\nvotes={n_votes}/3, span={len(exp["tokens"])}tok', fontsize=9)
            ax.set_xlabel('tokens rel to onset', fontsize=8)
            ax.tick_params(labelsize=8)
            if idx == 0:
                ax.legend(loc='upper left', fontsize=6, ncol=2)

        # Hide unused subplots
        for idx in range(n, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        fig.suptitle(f'{bias_short} (bias {bias_id}) — N={n} OOD annotations\n"{bias_text[:120]}..."', fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        out = OUT_DIR / f'bias_{bias_id:02d}_{bias_short}.png'
        fig.savefig(out, dpi=110, bbox_inches='tight')
        plt.close(fig)
        print(f'  wrote {out}')

    # Also make a summary plot: mean-across-pids shape per bias, per trait
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), squeeze=False)
    for bi, (bias_id, anns) in enumerate(biases_to_plot[:6]):
        ax = axes[bi // 3][bi % 3]
        bias_short = bias_map['biases'].get(str(bias_id), {}).get('short', f'bias_{bias_id}')
        for ti, trait in enumerate(TRAITS):
            stacks = []
            for pid, ei, exp in anns:
                signal = cache.get((trait, pid))
                if signal is None:
                    continue
                _, ys = extract_window(signal, exp['tokens'][0], WINDOW)
                stacks.append(ys)
            if not stacks:
                continue
            arr = np.stack(stacks)
            mean = np.nanmean(arr, axis=0)
            sem = np.nanstd(arr, axis=0) / np.sqrt(np.sum(~np.isnan(arr), axis=0).clip(min=1))
            xs = np.arange(-WINDOW, WINDOW + 1)
            ax.plot(xs, mean, color=COLORS[ti], label=trait, lw=1.5)
            ax.fill_between(xs, mean - sem, mean + sem, color=COLORS[ti], alpha=0.15)
        ax.axvline(0, color='k', lw=1, alpha=0.6)
        ax.axhline(0, color='gray', lw=0.5, alpha=0.3)
        ax.set_title(f'{bias_short} (N={len(anns)})', fontsize=10)
        ax.set_xlabel('tokens rel to onset')
        if bi == 0:
            ax.legend(loc='upper left', fontsize=7)
    fig.suptitle('Mean ± SEM projection shape (delta = rm_lora − instruct, response-mean-centered)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    summary_out = OUT_DIR / '_summary_mean_shapes.png'
    fig.savefig(summary_out, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {summary_out}')


if __name__ == '__main__':
    main()
