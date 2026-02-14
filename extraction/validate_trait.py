#!/usr/bin/env python3
"""
Validate trait dataset quality via scenario testing + extraction + steering.

Three validation gates:
    1. Scenarios: 90%+ pass rate both positive and negative
    2. Baseline: Steering question score < 30 (trait not naturally present)
    3. Delta: Steering effect > 15 with coherence >= 70

Uses a dedicated '_validate' experiment dir (gitignored, excluded from R2).
Between runs, the experiment dir is wiped for a clean slate.

Input: --model, --trait, optional --modal/--scenarios-only
Output: validation.json in experiments/_validate/ with pass/fail per gate

Usage:
    # Scenarios only (Modal, no local GPU needed)
    python extraction/validate_trait.py \
        --model meta-llama/Llama-3.1-8B \
        --trait alignment/performative_confidence \
        --modal --scenarios-only

    # Full validation (local GPU required for extraction + steering)
    python extraction/validate_trait.py \
        --model meta-llama/Llama-3.1-8B \
        --trait alignment/performative_confidence

    # Custom thresholds
    python extraction/validate_trait.py \
        --model meta-llama/Llama-3.1-8B \
        --trait alignment/performative_confidence \
        --baseline-threshold 25 --delta-threshold 20
"""

import sys
import json
import shutil
import asyncio
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from extraction.test_scenarios import run_test
from utils.vectors import MIN_COHERENCE

# Default thresholds
DEFAULT_SCENARIO_THRESHOLD = 0.9
DEFAULT_BASELINE_THRESHOLD = 30

EXPERIMENT_NAME = '_validate'
EXPERIMENT_DIR = Path(__file__).parent.parent / 'experiments' / EXPERIMENT_NAME


def _setup_experiment(model):
    """Create or refresh the _validate experiment with correct config."""
    from utils.model_registry import is_base_model

    # Wipe previous run
    if EXPERIMENT_DIR.exists():
        shutil.rmtree(EXPERIMENT_DIR)
    EXPERIMENT_DIR.mkdir(parents=True)

    config = {
        "defaults": {"extraction": "base", "application": "base"},
        "model_variants": {
            "base": {"model": model}
        },
        "use_chat_template": not is_base_model(model),
    }
    (EXPERIMENT_DIR / 'config.json').write_text(json.dumps(config, indent=2))

    # Clear cached config so pipeline reads fresh
    import utils.paths as _paths
    _paths._experiment_configs.pop(EXPERIMENT_NAME, None)


def _compute_validation_layers(model_name):
    """Get 3 layers at 30%, 40%, 50% of model depth."""
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    n = config.num_hidden_layers
    return [int(n * p) for p in (0.3, 0.4, 0.5)], n


def _parse_steering_results(results_path):
    """Extract baseline, max steered score, and best delta from steering JSONL."""
    if not results_path.exists():
        return {}

    baseline_score = None
    coherent_scores = []

    with open(results_path) as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get('type') == 'baseline':
                baseline_score = entry.get('result', {}).get('trait_mean')
            elif entry.get('type') not in ('header',):
                result = entry.get('result', {})
                trait_mean = result.get('trait_mean')
                coherence = result.get('coherence_mean', 0)
                if trait_mean is not None and coherence >= MIN_COHERENCE:
                    coherent_scores.append(trait_mean)

    out = {'baseline': baseline_score}
    if coherent_scores:
        out['max_steered'] = max(coherent_scores)
        if baseline_score is not None:
            out['delta'] = max(coherent_scores) - baseline_score
    return out


def _save_results(results):
    """Save validation results JSON and print summary."""
    gates = results['gates']
    evaluated = [g for g in gates.values() if 'skip' not in g]
    results['overall_pass'] = all(g.get('pass', False) for g in evaluated) if evaluated else False

    output_path = EXPERIMENT_DIR / 'validation.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    overall = 'PASS' if results['overall_pass'] else 'FAIL'
    print(f"VALIDATION {overall}: {results['trait']}")
    print(f"{'='*60}")
    for name, data in gates.items():
        if 'skip' in data:
            print(f"  {name}: SKIPPED ({data['skip']})")
        else:
            print(f"  {name}: {'PASS' if data['pass'] else 'FAIL'}")
    print(f"\n  Results: {output_path}")


def validate_trait(
    model,
    trait,
    use_modal=False,
    scenarios_only=False,
    method='probe',
    position='response[:5]',
    component='residual',
    scenario_threshold=DEFAULT_SCENARIO_THRESHOLD,
    baseline_threshold=DEFAULT_BASELINE_THRESHOLD,
):
    """
    Validate a trait dataset through up to 2 gates + steering report.

    Returns dict with gate results and overall_pass boolean.
    """
    results = {
        'trait': trait,
        'model': model,
        'timestamp': datetime.now().isoformat(),
        'thresholds': {
            'scenario_pass_rate': scenario_threshold,
            'baseline_max': baseline_threshold,
        },
        'gates': {},
    }

    # Setup clean experiment dir
    _setup_experiment(model)

    # =========================================================================
    # GATE 1: Scenario Testing
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"GATE 1: Scenario Testing")
    print(f"{'='*60}")

    scenario_result = run_test(
        trait=trait,
        model_override=model,
        workdir=EXPERIMENT_DIR / 'scenarios',
        use_modal=use_modal,
    )

    pos_rate = scenario_result['summary'].get('positive', {}).get('pass_rate', 0)
    neg_rate = scenario_result['summary'].get('negative', {}).get('pass_rate', 0)
    gate1_pass = pos_rate >= scenario_threshold and neg_rate >= scenario_threshold

    results['gates']['scenarios'] = {
        'positive_rate': pos_rate,
        'negative_rate': neg_rate,
        'pass': gate1_pass,
    }

    ok = lambda v, t: 'PASS' if v >= t else 'FAIL'
    print(f"\n  Positive: {pos_rate:.0%} {ok(pos_rate, scenario_threshold)}")
    print(f"  Negative: {neg_rate:.0%} {ok(neg_rate, scenario_threshold)}")
    print(f"  Gate 1: {'PASS' if gate1_pass else 'FAIL'}")

    if not gate1_pass or scenarios_only:
        _save_results(results)
        return results

    # Check steering.json exists
    from utils.paths import get as get_path
    steering_path = get_path('datasets.trait_steering', trait=trait)
    if not steering_path.exists():
        print(f"\n  Skipping gates 2-3: no steering.json for {trait}")
        results['gates']['baseline'] = {'skip': 'no steering.json'}
        results['gates']['delta'] = {'skip': 'no steering.json'}
        _save_results(results)
        return results

    # =========================================================================
    # GATE 2 & 3: Extraction + Steering
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"GATES 2-3: Extraction + Steering")
    print(f"{'='*60}")

    layers, num_layers = _compute_validation_layers(model)
    print(f"  Model: {model} ({num_layers} layers)")
    print(f"  Validation layers: {layers}")
    print(f"  Method: {method} | Position: {position} | Component: {component}")

    # --- Extraction pipeline (stages 1, 3, 4) ---
    print(f"\n--- Extraction ---")
    from extraction.run_pipeline import run_pipeline

    run_pipeline(
        experiment=EXPERIMENT_NAME,
        model_variant='base',
        traits=[trait],
        only_stages={1, 3, 4},
        methods=[method],
        vet=False,
        component=component,
        position=position,
        no_logitlens=True,
    )

    # --- Steering evaluation ---
    print(f"\n--- Steering ---")
    from analysis.steering.evaluate import run_evaluation

    layers_arg = ','.join(str(l) for l in layers)

    asyncio.run(run_evaluation(
        experiment=EXPERIMENT_NAME,
        trait=trait,
        vector_experiment=EXPERIMENT_NAME,
        model_variant='base',
        layers_arg=layers_arg,
        coefficients=None,
        method=method,
        component=component,
        position=position,
        prompt_set='steering',
        model_name=model,
        judge_provider='openai',
        subset=0,
        n_search_steps=5,
        up_mult=1.3,
        down_mult=0.85,
        start_mult=0.7,
        save_mode='none',
        relevance_check=True,
    ))

    # --- Parse results ---
    from utils.paths import get_steering_results_path
    results_path = get_steering_results_path(
        EXPERIMENT_NAME, trait, 'base', position, 'steering'
    )
    steering = _parse_steering_results(results_path)
    baseline_score = steering.get('baseline')
    max_steered = steering.get('max_steered')
    best_delta = steering.get('delta')

    gate2_pass = baseline_score is not None and baseline_score < baseline_threshold

    results['gates']['baseline'] = {
        'score': baseline_score,
        'threshold': baseline_threshold,
        'pass': gate2_pass,
    }
    results['gates']['steering'] = {
        'max_steered': max_steered,
        'delta': best_delta,
    }

    bl = f"{baseline_score:.1f}" if baseline_score is not None else "N/A"
    ms = f"{max_steered:.1f}" if max_steered is not None else "N/A"
    dt = f"{best_delta:+.1f}" if best_delta is not None else "N/A"
    print(f"\n  Gate 2 (baseline < {baseline_threshold}): {bl} {'PASS' if gate2_pass else 'FAIL'}")
    print(f"  Steering: max={ms}, delta={dt}")

    _save_results(results)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Validate trait dataset quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Gates:
  1. Scenarios  - Model completions match expected polarity (90%+)
  2. Baseline   - Trait not naturally present in steering responses (<30)
  3. Delta      - Steering vector produces causal effect (>15, coherence>=70)

Examples:
  # Quick check (Modal, no GPU)
  python extraction/validate_trait.py \\
      --model meta-llama/Llama-3.1-8B \\
      --trait alignment/performative_confidence \\
      --modal --scenarios-only

  # Full validation
  python extraction/validate_trait.py \\
      --model meta-llama/Llama-3.1-8B \\
      --trait alignment/performative_confidence
        """
    )
    parser.add_argument('--model', required=True,
                        help='Model name (e.g., meta-llama/Llama-3.1-8B)')
    parser.add_argument('--trait', required=True,
                        help='Trait path (e.g., alignment/performative_confidence)')
    parser.add_argument('--modal', action='store_true',
                        help='Use Modal for scenario generation (no local GPU for gate 1)')
    parser.add_argument('--scenarios-only', action='store_true',
                        help='Only run gate 1 (scenario testing)')

    # Thresholds
    parser.add_argument('--scenario-threshold', type=float, default=DEFAULT_SCENARIO_THRESHOLD,
                        help=f'Scenario pass rate threshold (default: {DEFAULT_SCENARIO_THRESHOLD})')
    parser.add_argument('--baseline-threshold', type=float, default=DEFAULT_BASELINE_THRESHOLD,
                        help=f'Max baseline trait score (default: {DEFAULT_BASELINE_THRESHOLD})')

    # Vector spec
    parser.add_argument('--method', default='probe',
                        help='Extraction method (default: probe)')
    parser.add_argument('--position', default='response[:5]',
                        help='Token position (default: response[:5])')
    parser.add_argument('--component', default='residual',
                        help='Model component (default: residual)')

    args = parser.parse_args()

    validate_trait(
        model=args.model,
        trait=args.trait,
        use_modal=args.modal,
        scenarios_only=args.scenarios_only,
        method=args.method,
        position=args.position,
        component=args.component,
        scenario_threshold=args.scenario_threshold,
        baseline_threshold=args.baseline_threshold,
    )
