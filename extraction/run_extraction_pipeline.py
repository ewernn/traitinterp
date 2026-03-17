#!/usr/bin/env python3
"""
Extraction pipeline: responses → activations → vectors → evaluation.

Stages:
    0: vet_scenarios   - LLM judges if scenarios match trait
    1: generate        - Generate model responses to scenarios
    2: vet_responses   - LLM judges if responses match trait
    3: activations     - Extract activations from responses
    4: vectors         - Train probe/gradient/mean_diff vectors
    5: logit_lens      - Interpret vectors via vocabulary projection
    6: evaluation      - Evaluate vectors on held-out data
    7: steering        - Causal validation via steering (--steering flag)

Usage:
    python extraction/run_extraction_pipeline.py --experiment gemma-2-2b --traits category/trait
    python extraction/run_extraction_pipeline.py --experiment gemma-2-2b --category epistemic
    python extraction/run_extraction_pipeline.py --experiment gemma-2-2b --only-stage 4
"""

import sys
import gc
import json
import asyncio
import argparse
import warnings
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from typing import List, Optional, Set, Dict
from dotenv import load_dotenv

warnings.filterwarnings("ignore", message=".*penalty.*deprecated.*", category=FutureWarning)
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.paths import (
    get as get_path,
    discover_traits,
    get_activation_metadata_path,
    get_activation_path,
    get_activation_dir,
    get_vector_dir,
    get_model_variant,
)
from utils.distributed import is_tp_mode, is_rank_zero, tp_barrier
from utils.backends import LocalBackend, add_backend_args
from utils.model_registry import is_base_model
from utils.vram import GPUMonitor, format_duration
from utils.traits import get_scenario_count, load_scenarios
from utils.model import format_prompt
from utils.generation import generate_batch as _generate_batch
from extraction.extract_vectors import (
    extract_activations_for_trait, resolve_max_new_tokens,
    extract_vectors_for_trait, run_logit_lens_for_trait, load_llm_judge_position,
)
from extraction.preextraction_vetting import vet_scenarios, vet_responses

STAGES = {
    0: 'vet_scenarios', 1: 'generate', 2: 'vet_responses',
    3: 'activations', 4: 'vectors', 5: 'logit_lens',
    6: 'evaluation', 7: 'steering',
}


def _estimate_eta(stage, n_items, rollouts=1, max_tokens=32):
    """Rough stage duration estimate in seconds."""
    estimates = {
        'vet_scenarios': 0.3 * n_items,
        'generate': (0.5 + max_tokens * 0.03) * n_items * rollouts,
        'vet_responses': 0.3 * n_items * rollouts,
        'activations': 0.05 * n_items * rollouts,
        'vectors': 2.0, 'logit_lens': 5.0, 'evaluation': 2.0,
    }
    return estimates.get(stage, 10.0)


def _run_stage(name, stage_times, fn, *args, n_items=None, **kwargs):
    """Run a pipeline stage with GPU monitoring and timing."""
    eta = _estimate_eta(name, n_items or 0) if n_items else None
    eta_str = f" (ETA: {format_duration(eta)})" if eta else ""
    stage_num = {v: k for k, v in STAGES.items()}.get(name, '?')
    print(f"  [{stage_num}] {name.replace('_', ' ').title()}...{eta_str}")
    with GPUMonitor(name) as mon:
        result = fn(*args, **kwargs)
        report = mon.report(n_items)
    stage_times[name] = stage_times.get(name, 0) + (time.time() - mon.start_time)
    print(f"      Done: {report}")
    return result


def _flush_cuda():
    """Free CUDA allocator cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _generate_responses_for_trait(
    experiment, trait, model_variant, backend, max_new_tokens,
    rollouts=1, temperature=0.0, chat_template=False,
):
    """Generate responses for scenarios. Returns (n_positive, n_negative)."""
    model = backend.model
    tokenizer = backend.tokenizer

    try:
        scenarios = load_scenarios(trait)
        pos_scenarios = scenarios['positive']
        neg_scenarios = scenarios['negative']
    except FileNotFoundError as e:
        print(f"    ERROR: {e}")
        return 0, 0

    responses_dir = get_path('extraction.responses', experiment=experiment, trait=trait, model_variant=model_variant)
    responses_dir.mkdir(parents=True, exist_ok=True)

    def generate_for_scenarios(scenarios, label):
        results = []
        prompts = [s['prompt'] for s in scenarios]
        system_prompts = [s.get('system_prompt') for s in scenarios]
        n_with_system = sum(1 for sp in system_prompts if sp)
        if n_with_system > 0:
            print(f"      {label}: {n_with_system}/{len(scenarios)} scenarios have system prompts")
        formatted_prompts = [
            format_prompt(p, tokenizer, use_chat_template=chat_template, system_prompt=sp)
            for p, sp in zip(prompts, system_prompts)
        ]
        for rollout_idx in range(rollouts):
            if max_new_tokens == 0:
                responses = [''] * len(formatted_prompts)
            else:
                responses = _generate_batch(model, tokenizer, formatted_prompts, max_new_tokens, temperature)
            for scenario, response in zip(scenarios, responses):
                results.append({
                    'prompt': scenario['prompt'],
                    'response': response,
                    'system_prompt': scenario.get('system_prompt'),
                })
        print(f"      {label}: {len(results)} responses")
        return results

    pos_results = generate_for_scenarios(pos_scenarios, 'positive')
    neg_results = generate_for_scenarios(neg_scenarios, 'negative')

    if is_rank_zero():
        with open(responses_dir / 'pos.json', 'w') as f:
            json.dump(pos_results, f, indent=2)
        with open(responses_dir / 'neg.json', 'w') as f:
            json.dump(neg_results, f, indent=2)

        metadata = {
            'model': model.config.name_or_path,
            'experiment': experiment,
            'trait': trait,
            'max_new_tokens': max_new_tokens,
            'chat_template': chat_template,
            'rollouts': rollouts,
            'temperature': temperature,
            'n_pos': len(pos_results),
            'n_neg': len(neg_results),
            'n_scenarios_pos': len(pos_scenarios),
            'n_scenarios_neg': len(neg_scenarios),
            'has_system_prompts': any(s.get('system_prompt') for s in pos_scenarios + neg_scenarios),
            'timestamp': datetime.now().isoformat(),
        }
        with open(responses_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"      Saved {len(pos_results)} + {len(neg_results)} responses")
    return len(pos_results), len(neg_results)


def run_pipeline(
    experiment: str,
    model_variant: str,
    traits: List[str],
    only_stages: Optional[Set[int]] = None,
    force: bool = False,
    methods: List[str] = None,
    vet: bool = True,
    run_scenario_vetting: bool = False,
    rollouts: int = 1,
    temperature: float = 0.0,
    val_split: float = 0.1,
    base_model: Optional[bool] = None,
    pos_threshold: int = 60,
    neg_threshold: int = 40,
    component: str = 'residual',
    position: str = 'response[:5]',
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    bnb_4bit_quant_type: str = "nf4",
    max_new_tokens: Optional[int] = None,
    max_concurrent: int = 100,
    paired_filter: bool = False,
    adaptive: bool = False,
    no_logitlens: bool = False,
    layers: Optional[List[int]] = None,
    min_pass_rate: float = 0.0,
    min_per_polarity: int = 0,
    steering: bool = False,
    save_activations: bool = False,
    backend=None,
):
    """Execute extraction pipeline."""
    # TP lifecycle
    import builtins
    _original_print = builtins.print
    if is_tp_mode():
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group("nccl")
        if not is_rank_zero():
            builtins.print = lambda *a, **k: None

    methods = methods or ['mean_diff', 'probe']
    max_new_tokens = resolve_max_new_tokens(position, max_new_tokens)

    if max_new_tokens == 0 and vet:
        raise ValueError(
            "Response vetting requires responses. Use --no-vet with prompt[-1] position, "
            "or specify --max-new-tokens > 0"
        )

    variant = get_model_variant(experiment, model_variant, mode="extraction")
    extraction_model = variant['model']
    lora = variant.get('lora')

    if base_model is None:
        base_model = is_base_model(extraction_model)

    def should_run(stage):
        return only_stages is None or stage in only_stages

    needs_model = should_run(1) or should_run(3) or (should_run(5) and not no_logitlens)

    print("=" * 60)
    print(f"EXTRACTION PIPELINE | {experiment}")
    if only_stages:
        stage_names = [STAGES[s] for s in sorted(only_stages)]
        print(f"Stages: {', '.join(stage_names)}")
    print(f"Model: {extraction_model} | {'BASE' if base_model else 'IT'} ({max_new_tokens} tokens)")
    print(f"Traits: {len(traits)}")
    print("=" * 60)

    pipeline_start = time.time()
    stage_times: Dict[str, float] = {}

    if backend is None and needs_model:
        load_start = time.time()
        backend = LocalBackend.from_experiment(
            experiment, variant=variant['name'],
            load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
        )
        stage_times['model_load'] = time.time() - load_start
        print(f"Model loaded. ({format_duration(stage_times['model_load'])})")
    elif backend is not None:
        print(f"Using pre-loaded model.")

    use_chat_template = False if base_model else (backend and backend.tokenizer.chat_template is not None)

    for trait in traits:
        print(f"\n--- {trait} --- [{datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}]")
        vetting_path = get_path("extraction.trait", experiment=experiment, trait=trait, model_variant=variant['name']) / "vetting"

        try:
            counts = get_scenario_count(trait)
            n_scenarios = counts['positive'] + counts['negative']
        except Exception:
            n_scenarios = 200

        # Stage 0: Scenario vetting (opt-in, rank 0 only)
        if should_run(0) and run_scenario_vetting:
            if not (vetting_path / "scenario_scores.json").exists() or force:
                if is_rank_zero():
                    _run_stage('vet_scenarios', stage_times,
                               vet_scenarios, experiment, trait, variant['name'],
                               pos_threshold, neg_threshold, max_concurrent,
                               n_items=n_scenarios)
            tp_barrier()

        # Stage 1: Generate responses
        if should_run(1):
            responses_path = get_path("extraction.responses", experiment=experiment, trait=trait, model_variant=variant['name'])
            has_responses = (responses_path / "pos.json").exists() and (responses_path / "neg.json").exists()
            if not has_responses or force:
                _run_stage('generate', stage_times,
                           _generate_responses_for_trait, experiment, trait, variant['name'],
                           backend, max_new_tokens, rollouts, temperature, use_chat_template,
                           n_items=n_scenarios * rollouts)
            tp_barrier()
            _flush_cuda()

        # Stage 2: Response vetting (rank 0 only)
        if should_run(2) and vet:
            if not (vetting_path / "response_scores.json").exists() or force:
                if is_rank_zero():
                    _run_stage('vet_responses', stage_times,
                               vet_responses, experiment, trait, variant['name'],
                               pos_threshold, neg_threshold, max_concurrent,
                               n_items=n_scenarios * rollouts,
                               estimate_trait_tokens=adaptive)
            tp_barrier()

        # Quality gate: skip trait if too few responses pass vetting
        if vet and should_run(3) and (vetting_path / "response_scores.json").exists():
            with open(vetting_path / "response_scores.json") as _f:
                _vet_data = json.load(_f)
            _summary = _vet_data.get('summary', {})
            _pos_pass = _summary.get('positive_passed', 0)
            _neg_pass = _summary.get('negative_passed', 0)
            _pos_total = _pos_pass + _summary.get('positive_failed', 0)
            _neg_total = _neg_pass + _summary.get('negative_failed', 0)
            _total = _pos_total + _neg_total
            _pass_rate = (_pos_pass + _neg_pass) / _total if _total > 0 else 0

            if _pos_pass < min_per_polarity or _neg_pass < min_per_polarity or _pass_rate < min_pass_rate:
                print(f"  SKIP: quality gate failed (pos={_pos_pass}/{_pos_total}, neg={_neg_pass}/{_neg_total}, rate={_pass_rate:.0%})")
                continue

        # Load adaptive position from vetting
        if adaptive:
            llm_pos = load_llm_judge_position(experiment, trait, variant['name'])
            if llm_pos:
                position = llm_pos
                print(f"  Using adaptive position: {position}")
            else:
                raise ValueError(f"--adaptive requires vetting with --adaptive first. No llm_judge_position for {trait}.")

        # Stage 3: Extract activations
        cached_activations = None
        if should_run(3):
            activation_metadata_path = get_activation_metadata_path(experiment, trait, variant['name'], component, position)
            activation_tensor = get_activation_path(experiment, trait, variant['name'], component, position)
            activation_dir = get_activation_dir(experiment, trait, variant['name'], component, position)
            has_activations = activation_metadata_path.exists() and (
                activation_tensor.exists() or any(activation_dir.glob("train_layer*.pt"))
            )
            if not has_activations or force:
                cached_activations = _run_stage('activations', stage_times,
                           extract_activations_for_trait, experiment, trait, variant['name'],
                           backend, val_split,
                           n_items=n_scenarios * rollouts,
                           position=position, component=component,
                           paired_filter=paired_filter, use_vetting_filter=vet,
                           layers=layers,
                           pos_threshold=pos_threshold, neg_threshold=neg_threshold,
                           save_activations=save_activations)
            tp_barrier()

        # Stage 4: Extract vectors (uses in-memory activations if available, else loads from disk)
        if should_run(4):
            has_all_vectors = all(
                get_vector_dir(experiment, trait, m, variant['name'], component, position).exists()
                and list(get_vector_dir(experiment, trait, m, variant['name'], component, position).glob("layer*.pt"))
                for m in methods
            )
            if not has_all_vectors or force:
                _run_stage('vectors', stage_times,
                           extract_vectors_for_trait, experiment, trait, variant['name'], methods,
                           n_items=len(methods),
                           layers=layers, component=component, position=position,
                           activations=cached_activations)
            cached_activations = None  # free memory

        # Stage 5: Logit lens
        if should_run(5) and not no_logitlens:
            logit_lens_path = get_path("extraction.logit_lens", experiment=experiment, trait=trait, model_variant=variant['name'])
            if not logit_lens_path.exists() or force:
                _run_stage('logit_lens', stage_times,
                           run_logit_lens_for_trait,
                           experiment=experiment, trait=trait, model_variant=variant['name'],
                           backend=backend, methods=methods, component=component, position=position)

    # Stage 6: Evaluation (post-loop, all traits at once)
    if should_run(6):
        from analysis.vectors.extraction_evaluation import main as run_evaluation
        eval_path = get_path("extraction_eval.evaluation", experiment=experiment)
        if not eval_path.exists() or force:
            _run_stage('evaluation', stage_times,
                       run_evaluation, experiment,
                       n_items=len(traits),
                       model_variant=variant['name'],
                       methods=",".join(methods),
                       component=component, position=position)

    if should_run(6) and not steering:
        print("For causal validation, re-run with --steering")

    # Free extraction model before steering loads its own
    if backend is not None:
        del backend
        backend = None
        _flush_cuda()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Stage 7: Steering evaluation
    if should_run(7) and steering:
        from steering.run_steering_eval import run_evaluation as run_steering_evaluation
        from utils.traits import load_steering_data
        app_variant = get_model_variant(experiment, None, mode='application')
        app_model_name = app_variant['model']
        print(f"\n[7] Running steering evaluation...")
        print(f"    Application model: {app_model_name}")
        for trait in traits:
            try:
                sd = load_steering_data(trait)
                direction = sd.direction or "positive"
            except (FileNotFoundError, ValueError):
                direction = "positive"

            for method in methods:
                print(f"  Steering: {trait} ({method}, direction={direction})")
                stage_start = time.time()
                asyncio.run(run_steering_evaluation(
                    experiment=experiment, trait=trait, vector_experiment=experiment,
                    model_variant=app_variant['name'], layers_arg="30%-60%",
                    coefficients=None, method=method, component=component, position=position,
                    prompt_set='steering', model_name=app_model_name, judge_provider='openai',
                    subset=5, n_search_steps=5, up_mult=1.3, down_mult=0.85, start_mult=0.7,
                    backend=None, force=force, save_mode='best',
                    extraction_variant=variant['name'],
                    lora_adapter=app_variant.get('lora'),
                    load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit,
                    bnb_4bit_quant_type=bnb_4bit_quant_type, direction=direction,
                ))
                stage_times['steering'] = stage_times.get('steering', 0) + (time.time() - stage_start)
                print(f"    Done: {trait} ({method})")

    builtins.print = _original_print

    if not is_tp_mode() or is_rank_zero():
        total_time = time.time() - pipeline_start
        print(f"\n{'=' * 60}")
        print("COMPLETE")
        print("-" * 30)
        for stage, duration in stage_times.items():
            print(f"  {stage}: {format_duration(duration)}")
        print(f"  Total: {format_duration(total_time)}")
        print("=" * 60)

    if is_tp_mode():
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extraction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Stages: 0=vet_scenarios, 1=generate, 2=vet_responses, 3=activations, 4=vectors, 5=logit_lens, 6=evaluation, 7=steering"
    )
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--traits", type=str)
    parser.add_argument("--category", type=str)
    parser.add_argument("--only-stage", type=lambda s: [int(x) for x in s.split(',')], dest='only_stages',
                        help="Run only specific stage(s): --only-stage 3,4")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--methods", default="mean_diff,probe")
    parser.add_argument("--no-vet", action="store_true")
    parser.add_argument("--vet-scenarios", action="store_true")
    parser.add_argument("--rollouts", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--model-variant", default=None)
    parser.add_argument("--component", default="residual")
    parser.add_argument("--position", default="response[:5]")
    parser.add_argument("--pos-threshold", type=int, default=60)
    parser.add_argument("--neg-threshold", type=int, default=40)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--bnb-4bit-quant-type", default="nf4")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--max-concurrent", type=int, default=100)
    parser.add_argument("--paired-filter", action="store_true")
    parser.add_argument("--min-pass-rate", type=float, default=0.0)
    parser.add_argument("--min-per-polarity", type=int, default=0)
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--no-logitlens", action="store_true")
    parser.add_argument("--save-activations", action="store_true",
                        help="Persist activation .pt files (for re-running --only-stage 4 with different methods)")
    parser.add_argument("--steering", action="store_true")
    add_backend_args(parser)
    parser.add_argument("--layers", type=str, default=None,
                        help="Only capture specific layers. E.g., '25,30,35,40' or '30%%-60%%'")
    parser.add_argument("--base-model", action="store_true", dest="base_model_override")
    parser.add_argument("--it-model", action="store_true", dest="it_model_override")
    args = parser.parse_args()

    # Resolve traits
    if args.traits:
        traits = args.traits.split(',')
    else:
        traits = discover_traits(category=args.category)
    if not traits:
        raise ValueError("No traits found")

    # Parse --layers if provided
    parsed_layers = None
    if args.layers:
        from utils.layers import parse_layers
        from utils.paths import load_experiment_config
        config = load_experiment_config(args.experiment)
        variant_name = args.model_variant or config.get('defaults', {}).get('extraction', 'base')
        model_name = config['model_variants'][variant_name]['model']
        from transformers import AutoConfig
        model_config = AutoConfig.from_pretrained(model_name)
        if hasattr(model_config, 'text_config'):
            model_config = model_config.text_config
        n_layers = model_config.num_hidden_layers
        parsed_layers = parse_layers(args.layers, n_layers)
        print(f"Layer selection: {len(parsed_layers)} of {n_layers} layers")

    run_pipeline(
        experiment=args.experiment,
        model_variant=args.model_variant,
        traits=traits,
        only_stages=set(args.only_stages) if args.only_stages else None,
        force=args.force,
        methods=args.methods.split(','),
        vet=not args.no_vet,
        run_scenario_vetting=args.vet_scenarios,
        rollouts=args.rollouts,
        temperature=args.temperature,
        val_split=args.val_split,
        base_model=True if args.base_model_override else (False if args.it_model_override else None),
        pos_threshold=args.pos_threshold,
        neg_threshold=args.neg_threshold,
        component=args.component,
        position=args.position,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        max_new_tokens=args.max_new_tokens,
        max_concurrent=args.max_concurrent,
        paired_filter=args.paired_filter,
        adaptive=args.adaptive,
        no_logitlens=args.no_logitlens,
        layers=parsed_layers,
        min_pass_rate=args.min_pass_rate,
        min_per_polarity=args.min_per_polarity,
        steering=args.steering,
        save_activations=args.save_activations,
    )
