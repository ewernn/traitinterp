"""
Extraction pipeline stages.

Stage orchestrators called by extraction/run_extraction_pipeline.py. Each stage
takes (config, trait or traits, variant_name, backend) and writes outputs to the
canonical paths in config/paths.yaml.

Stages:
    1: generate_responses          Generate model responses from scenarios (+ OOD if present)
    2: vet_responses               LLM judge scores responses (opt-in)
  3+4: extract_vectors_for_trait   Forward pass → activations → trained vectors per method
    5: run_logit_lens              Project residual vectors through unembedding to vocab
    6: evaluate_extraction         Held-out validation metrics

Plus pipeline helpers:
    init_backend                   Load model + derive use_chat_template
    fill_extraction_defaults       Resolve None defaults from base/instruct detection
    per_trait_config               Context manager: YAML overlay + auto-reset
"""

import json
import re
from contextlib import contextmanager
from datetime import datetime
from typing import List

from core.kwargs_configs import ExtractionConfig, VettingStats
from utils.distributed import is_rank_zero, tp_barrier, flush_cuda
from utils.backends import LocalBackend
from utils.model_registry import is_base_model
from utils.paths import (
    get as get_path, get_model_variant, content_hash,
    get_activation_metadata_path, get_activation_path, get_activation_dir, get_vector_dir,
)
from utils.traits import (
    load_scenarios, load_ood_scenarios, get_ood_scenario_path, get_scenario_path,
    load_extraction_config, resolve_emotion_surface,
)
from utils.model import format_prompt
from utils.model_generation import generate_batch
from utils.preextraction_vetting import vet_responses as _vet_responses_judge
from utils.extract_vectors import (
    extract_activations_for_trait,
    extract_vectors_for_trait as _train_vectors_for_trait,
    load_llm_judge_position,
)
from utils.positions import resolve_max_new_tokens


_YAML_FIELDS = ('position', 'max_new_tokens', 'methods', 'temperature', 'rollouts')


# =============================================================================
# Batched story parsing (paper-faithful replication level)
# =============================================================================
# Paper's batched generation prompt instructs the model to emit [story N]
# delimited blocks. In practice (Llama 3.3 70B, Q-C3 empirical test): the model
# is usually compliant but occasionally (a) restarts the numbering mid-response
# producing duplicate delimiters, (b) adds a trailing colon `[story 1:]`, or
# (c) wraps the delimiter in markdown bold `**[story 1]**`. The regex + parser
# below tolerate all three. Alternative formats (unbracketed `Story 1:`,
# markdown heading `### Story 1`) were not observed in the empirical test and
# are intentionally NOT accepted — caller retries on under-production instead.

# Matches `[story N]`, `[ story  N ]`, `[STORY N]`, `[story N:]`, and tolerates
# surrounding markdown bold (0–2 asterisks) on either side.
_STORY_DELIMITER_RE = re.compile(
    r'\*{0,2}\s*\[\s*story\s+(\d+)\s*:?\s*\]\s*\*{0,2}',
    re.IGNORECASE,
)

# Generic numbered-block delimiter — accepts any single word as the label
# (e.g. "story", "dialogue", "example"). Same tolerance as _STORY_DELIMITER_RE.
# Used by parse_numbered_blocks() for non-story paper-replication paths.
def _make_block_delimiter_re(label: str) -> "re.Pattern":
    return re.compile(
        rf'\*{{0,2}}\s*\[\s*{re.escape(label)}\s+(\d+)\s*:?\s*\]\s*\*{{0,2}}',
        re.IGNORECASE,
    )


def validate_full_mode_flags(
    replication_level: str,
    topics_limit,
    stories_per_batch_override,
) -> None:
    """Raise ValueError if --topics / --stories-per-batch are passed in lightweight mode.

    Pure function — no side effects, no argparse coupling. Callable from
    run_extraction_pipeline.main() after args.parse, and directly from unit tests.
    """
    if replication_level == "lightweight":
        if topics_limit is not None:
            raise ValueError(
                "--topics is a full-mode-only flag. "
                "Pass --replication-level=full, or drop --topics."
            )
        if stories_per_batch_override is not None:
            raise ValueError(
                "--stories-per-batch is a full-mode-only flag. "
                "Pass --replication-level=full, or drop --stories-per-batch."
            )


def print_full_replication_estimate(config, traits) -> None:
    """Prominent scope summary printed at startup when --replication-level=full.

    Paper-faithful replication is expensive relative to lightweight. Surface the
    scope up-front so users don't accidentally launch a 100-hour run when they
    meant a sanity-check.
    """
    n_traits = len(traits)
    # Full mode's N-per-batch comes from either --stories-per-batch override or
    # each trait's extraction_config.yaml `stories_per_batch`. We don't resolve
    # per-trait here (would require loading each config); show the override if
    # set, else a placeholder pointing at the config.
    if config.stories_per_batch_override is not None:
        n_stories_label = str(config.stories_per_batch_override)
    else:
        n_stories_label = "<per extraction_config.yaml, paper default=12>"

    print()
    print("=" * 72)
    print("  --replication-level=full : paper-verbatim prompts + batched generation")
    print("=" * 72)
    print(f"  Traits               : {n_traits}")
    print(f"  Stories per batch    : {n_stories_label}")
    print(f"  Generation calls     : ~{n_traits} × N_topics (one batched call per topic)")
    print(f"  Activation passes    : ~{n_traits} × N_topics × stories_per_batch")
    print(f"  Paper default scale  : 100 topics × 12 stories (override --topics / --stories-per-batch)")
    print(f"  Q-C3 empirical note  : Llama 3.3 70B at N=12 produces distinct stories")
    print(f"                         for ~3/4 emotions; `calm` collapsed in the test.")
    print(f"                         Watch the pipeline logs for under-production warnings.")
    print("=" * 72)
    print()


def _strip_leading_shell_comments(text: str) -> str:
    """Strip leading `#`-prefixed comment lines and the blank separator after them.

    Lets paper-prompt files carry a provenance header (e.g. "Verbatim from
    Sofroniew et al. 2026, Appendix...") without it leaking into the rendered
    prompt sent to the model. Only strips contiguous leading comment lines; a
    `#` that appears mid-template is preserved unchanged.
    """
    lines = text.splitlines()
    i = 0
    while i < len(lines) and lines[i].lstrip().startswith('#'):
        i += 1
    # Also consume a single blank separator line between the comment block and
    # the real template content
    if i < len(lines) and lines[i].strip() == '':
        i += 1
    # Preserve whether the original text had a trailing newline — but only when
    # there's actual content left (a comments-only file should yield empty string,
    # not a lonely newline).
    rest = '\n'.join(lines[i:])
    if rest and text.endswith('\n') and not rest.endswith('\n'):
        rest += '\n'
    return rest


def parse_story_blocks(response: str, expected_n: int) -> List[str]:
    """Parse [story N]-delimited blocks from a batched-generation LLM response.

    Tolerates:
      - Duplicate delimiters (model restarts the list): keeps first occurrence
        per story-index. If the model emits [story 1] ... [story 2] ... then
        [story 1] ... [story 2] ... again, only the first pair is returned.
      - Fewer blocks than expected: returns whatever was parsed (caller decides
        whether to retry or fall back to serial generation).
      - Extra trailing text after the last block: included as part of the last
        block's content.
      - Bracket spacing / case: [story 1], [ Story 2 ], [STORY 3].
      - Trailing colon: [story 1:].
      - Markdown bold wrapping: **[story 1]**.

    Returns up to `expected_n` stories, in sorted story-index order (which is
    "first N unique indices", NOT "stories 1..N strictly" — if the model skips
    an index, the sort order advances). Each story's text is stripped.

    Returns empty list if no [story N] delimiters are found — caller should
    retry or fall back to serial generation.

    Raises TypeError if `response` is not a string.
    """
    if not isinstance(response, str):
        raise TypeError(
            f"response must be str, got {type(response).__name__}"
        )
    matches = list(_STORY_DELIMITER_RE.finditer(response))
    if not matches:
        return []

    # ALL delimiter positions — used to bound block extraction correctly even
    # when duplicates appear (block ends at the NEXT delimiter of ANY index,
    # not just the next unique index, so a duplicate [story 1] terminates the
    # preceding block rather than being swallowed into it).
    all_positions = sorted(m.start() for m in matches)

    # First occurrence per story index — these are the blocks we'll extract.
    first_by_index = {}
    for m in matches:
        idx = int(m.group(1))
        if idx not in first_by_index:
            first_by_index[idx] = (m.start(), m.end())

    stories = []
    for idx in sorted(first_by_index.keys()):
        match_start, block_start = first_by_index[idx]
        next_positions = [p for p in all_positions if p > match_start]
        block_end = next_positions[0] if next_positions else len(response)
        block = response[block_start:block_end].strip()
        if block:
            stories.append(block)
        if len(stories) >= expected_n:
            break

    return stories


def parse_numbered_blocks(response: str, expected_n: int, label: str = "story") -> List[str]:
    """Parse [{label} N]-delimited blocks from a batched-generation LLM response.

    Generalization of parse_story_blocks() — any single-word label works.
    Examples:
      - label="story" → matches [story 1], [Story 2], **[story 3]**, [story 4:]
      - label="dialogue" → matches [dialogue 1], [Dialogue 2], etc.

    Same tolerance rules as parse_story_blocks: duplicate delimiters (model
    restarts the list), case-insensitive bracket spacing, trailing colon,
    markdown bold wrapping.

    Returns up to `expected_n` blocks in sorted-index order. Raises TypeError
    if response isn't a string.
    """
    if not isinstance(response, str):
        raise TypeError(f"response must be str, got {type(response).__name__}")

    delimiter_re = _make_block_delimiter_re(label)
    matches = list(delimiter_re.finditer(response))
    if not matches:
        return []

    all_positions = sorted(m.start() for m in matches)
    first_by_index = {}
    for m in matches:
        idx = int(m.group(1))
        if idx not in first_by_index:
            first_by_index[idx] = (m.start(), m.end())

    blocks = []
    for idx in sorted(first_by_index.keys()):
        match_start, block_start = first_by_index[idx]
        next_positions = [p for p in all_positions if p > match_start]
        block_end = next_positions[0] if next_positions else len(response)
        block = response[block_start:block_end].strip()
        if block:
            blocks.append(block)
        if len(blocks) >= expected_n:
            break

    return blocks


# =============================================================================
# Pipeline helpers
# =============================================================================

def _should_run(config, stage_num):
    return not config.only_stages or stage_num in config.only_stages


def init_backend(config):
    """Load model backend. Returns (backend, variant_name, use_chat_template)."""
    variant = get_model_variant(config.experiment, config.model_variant, mode="extraction")
    is_base = config.base_model if config.base_model is not None else is_base_model(variant.model)
    backend = LocalBackend.from_experiment(
        config.experiment, variant=variant.name,
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
    )
    use_chat_template = not is_base and backend.tokenizer.chat_template is not None
    return backend, variant.name, use_chat_template


def fill_extraction_defaults(config):
    """Fill config.position / max_new_tokens defaults from base/instruct detection."""
    variant = get_model_variant(config.experiment, config.model_variant, mode="extraction")
    is_base = config.base_model if config.base_model is not None else is_base_model(variant.model)
    default_position = "response[:5]" if is_base else "response[:]"
    default_max_new_tokens = 16 if is_base else 64
    if config.position is None:
        config.position = default_position
        print(f"  {'Pretrained' if is_base else 'Instruct'} model → position={default_position}")
    if config.max_new_tokens is None:
        config.max_new_tokens = default_max_new_tokens
        print(f"  → max_new_tokens={default_max_new_tokens}")


@contextmanager
def per_trait_config(config, trait, cli_overrides):
    """Per-trait YAML overlay → CLI overlay. Original values restored on exit
    (even on early continue/exception)."""
    saved = {f: getattr(config, f) for f in _YAML_FIELDS}
    yaml_cfg = load_extraction_config(trait)
    for field in _YAML_FIELDS:
        if field in yaml_cfg and field not in cli_overrides:
            setattr(config, field, yaml_cfg[field])
            print(f"  extraction_config.yaml → {field}={yaml_cfg[field]}")
    try:
        yield
    finally:
        for field in _YAML_FIELDS:
            setattr(config, field, saved[field])


# =============================================================================
# Stage 1: response generation
# =============================================================================

def generate_responses(config, trait, variant_name, backend, use_chat_template):
    """Stage 1: Generate model responses from scenarios (+ OOD if present)."""
    if not _should_run(config, 1):
        return

    responses_path = get_path("extraction.responses", experiment=config.experiment,
                              trait=trait, model_variant=variant_name)
    ood_scenarios = load_ood_scenarios(trait)

    try:
        scenarios = load_scenarios(trait)
    except FileNotFoundError as e:
        print(f"    ERROR: {e}")
        return

    is_single = 'negative' not in scenarios
    if not config.force and _stage1_outputs_complete(responses_path, is_single, ood_scenarios):
        return

    print(f"  [1] Generating responses...")
    max_new_tokens = resolve_max_new_tokens(config.position, config.max_new_tokens)
    responses_path.mkdir(parents=True, exist_ok=True)

    _generate_training_responses(scenarios, responses_path, backend, config,
                                 max_new_tokens, use_chat_template, trait=trait)
    if ood_scenarios is not None:
        _generate_ood_responses(ood_scenarios, responses_path, backend, config,
                                max_new_tokens, use_chat_template)

    if is_rank_zero():
        _write_generation_metadata(responses_path, config, trait, backend.model,
                                   max_new_tokens, use_chat_template, scenarios, ood_scenarios)
    tp_barrier()
    flush_cuda()


def _stage1_outputs_complete(responses_path, is_single, ood_scenarios):
    expected = [responses_path / "pos.json"]
    if not is_single:
        expected.append(responses_path / "neg.json")
    if ood_scenarios is not None:
        expected += [responses_path / "ood_pos.json", responses_path / "ood_neg.json"]
    return all(f.exists() for f in expected)


def _generate_and_write(scenarios_list, output_path, model, tokenizer, use_chat_template,
                        max_new_tokens, temperature, seed, rollouts):
    """Run generation for one scenario list and write output JSON."""
    results = []
    formatted = [
        format_prompt(s['prompt'], tokenizer, use_chat_template=use_chat_template,
                     system_prompt=s.get('system_prompt'))
        for s in scenarios_list
    ]
    for rollout_idx in range(rollouts):
        # Vary the seed per rollout so temperature>0 produces diverse samples
        # while the overall run stays reproducible. seed=None remains None (fully random).
        rollout_seed = None if seed is None else seed + rollout_idx
        responses = (
            [''] * len(formatted) if max_new_tokens == 0
            else generate_batch(model, tokenizer, formatted, max_new_tokens, temperature, seed=rollout_seed)
        )
        for scenario, response in zip(scenarios_list, responses):
            results.append({
                'prompt': scenario['prompt'], 'response': response,
                'system_prompt': scenario.get('system_prompt'),
            })
    if is_rank_zero():
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    return len(results)


def _generate_stories_batched_and_write(
    topics, output_path, model, tokenizer, use_chat_template,
    max_new_tokens, temperature, seed,
    batched_template, template_kwargs, stories_per_batch,
):
    """Paper-faithful batched generation: one call per topic, N stories per call.

    Sends `batched_template.format(n_stories=N, topic=T, **template_kwargs)` as
    the user prompt for each topic, parses the response via `parse_story_blocks`,
    emits one record per parsed story.

    Contract:
      - `batched_template` must reference `{n_stories}` and `{topic}` at minimum;
        any other placeholders (e.g. `{emotion}`, `{person_emotion}`) must be
        provided via `template_kwargs`.
      - Python's native `KeyError` fires at .format() time if the template
        references a placeholder not in `template_kwargs` — fail loud, no retry.

    On under-production (parse returns fewer than `stories_per_batch`) the
    partial result is kept and a warning logged. Callers at integration time
    decide whether to retry; this helper does not retry to keep the signal
    surface simple for v1.

    Output record schema matches `_generate_and_write` plus two extra fields
    for downstream reference (`story_idx`, `topic`):
        {prompt, response, system_prompt, story_idx, topic}
    where `prompt` is the FULL batched user message (so activation capture can
    re-tokenize the user+assistant sequence with the same context the model saw).
    """
    if stories_per_batch <= 0:
        raise ValueError(
            f"stories_per_batch must be >= 1, got {stories_per_batch}. "
            f"(Likely --stories-per-batch=0 from the CLI, which is nonsense input.)"
        )

    results = []
    total_expected = len(topics) * stories_per_batch
    total_produced = 0
    under_produced_topics = []

    for topic_idx, topic in enumerate(topics):
        # Render the template — this is the SYSTEM prompt per the paper
        # (emotion_concepts_full_paper.md:1376 explicitly says "system prompt").
        # We hand it to format_prompt as system_prompt=..., with an empty user
        # turn, so the chat template lands the paper text in the system role.
        rendered_system = batched_template.format(
            n_stories=stories_per_batch,
            topic=topic,
            **template_kwargs,
        )
        formatted = format_prompt(
            prompt="",
            tokenizer=tokenizer,
            use_chat_template=use_chat_template,
            system_prompt=rendered_system,
        )

        # Vary seed per topic so diverse sampling is reproducible at the topic
        # level (paper runs one batched call per topic — no natural rollout axis).
        topic_seed = None if seed is None else seed + topic_idx
        response = generate_batch(
            model, tokenizer, [formatted],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            seed=topic_seed,
        )[0]

        stories = parse_story_blocks(response, expected_n=stories_per_batch)

        if len(stories) < stories_per_batch:
            under_produced_topics.append((topic_idx, len(stories)))

        for story_idx, story in enumerate(stories):
            # Record schema: `prompt` carries the rendered SYSTEM text so
            # downstream activation capture can reconstruct the chat sequence
            # (system=prompt, user="", assistant=response) the model saw.
            results.append({
                'prompt': '',
                'response': story,
                'system_prompt': rendered_system,
                'story_idx': story_idx,
                'topic': topic,
            })
        total_produced += len(stories)

    if is_rank_zero():
        if under_produced_topics:
            preview = ", ".join(
                f"topic {i}: {n}/{stories_per_batch}"
                for i, n in under_produced_topics[:5]
            )
            suffix = "" if len(under_produced_topics) <= 5 else f" (+{len(under_produced_topics) - 5} more)"
            print(
                f"    ⚠ batched generation under-produced on "
                f"{len(under_produced_topics)}/{len(topics)} topics "
                f"({total_produced}/{total_expected} stories total)."
            )
            print(f"      First few: {preview}{suffix}")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    return len(results)


def _generate_training_responses(scenarios, responses_path, backend, config,
                                 max_new_tokens, use_chat_template, trait=None):
    if config.replication_level == "full":
        _generate_training_responses_full(
            scenarios, responses_path, backend, config,
            max_new_tokens, use_chat_template, trait,
        )
        return

    # Lightweight path (unchanged)
    for label in scenarios:
        out_path = responses_path / f'{label[:3]}.json'
        if not config.force and out_path.exists():
            continue
        n = _generate_and_write(scenarios[label], out_path, backend.model, backend.tokenizer,
                                use_chat_template, max_new_tokens, config.temperature,
                                config.seed, config.rollouts)
        print(f"    {label}: {n} responses")


def _generate_training_responses_full(scenarios, responses_path, backend, config,
                                      max_new_tokens, use_chat_template, trait):
    """Full-mode branch: paper-verbatim batched generation.

    Requires extraction_config.yaml to provide:
      - batched_story_template_file (Path, eagerly resolved by load_extraction_config)
      - topics_file (Path, eagerly resolved)
      - stories_per_batch (int, overridable via --stories-per-batch CLI)

    See docs/extraction_guide.md §Full replication for the full contract.
    """
    if trait is None:
        raise ValueError(
            "_generate_training_responses_full requires trait= kwarg; "
            "caller generate_responses must thread it through."
        )

    cfg = load_extraction_config(trait)

    # F1: hard-fail with actionable message if any required field is missing
    category = "/".join(trait.split("/")[:-1]) or trait
    for required in ("batched_story_template_file", "topics_file", "stories_per_batch"):
        if required not in cfg:
            raise ValueError(
                f"--replication-level=full requires '{required}' in "
                f"extraction_config.yaml (resolved for trait '{trait}').\n"
                f"Either drop --replication-level=full, or add '{required}' to "
                f"datasets/traits/{category}/extraction_config.yaml."
            )

    # F2: FileNotFoundError propagates from these read_text() / json.loads calls
    template = _strip_leading_shell_comments(cfg['batched_story_template_file'].read_text())
    topics = json.loads(cfg['topics_file'].read_text())

    if config.topics_limit is not None:
        topics = topics[:config.topics_limit]

    # Use `is None` (not `or`) so an accidental --stories-per-batch=0 fails
    # fast at helper time rather than silently falling through to the YAML default.
    n_per_batch = (
        cfg['stories_per_batch']
        if config.stories_per_batch_override is None
        else config.stories_per_batch_override
    )

    # Derive emotion via resolve_emotion_surface so multi-word traits
    # (e.g. grief_stricken -> "grief-stricken") match paper convention
    # consistently with the lightweight prompt_template code path.
    emotion = resolve_emotion_surface(trait)

    # Full mode is single-polarity by design (paper uses one contrasting
    # corpus; MeanDiffMethod handles zero-centered negative elsewhere).
    # Only 'positive' label is processed.
    for label in scenarios:
        if label != 'positive':
            print(f"    skipping label '{label}' in full mode (paper is single-polarity)")
            continue
        out_path = responses_path / f'{label[:3]}.json'
        if not config.force and out_path.exists():
            continue
        n = _generate_stories_batched_and_write(
            topics=topics,
            output_path=out_path,
            model=backend.model,
            tokenizer=backend.tokenizer,
            use_chat_template=use_chat_template,
            max_new_tokens=max_new_tokens,
            temperature=config.temperature,
            seed=config.seed,
            batched_template=template,
            template_kwargs={'emotion': emotion},
            stories_per_batch=n_per_batch,
        )
        print(f"    {label}: {n} responses ({len(topics)} topics × up to {n_per_batch} stories)")


def _generate_ood_responses(ood_scenarios, responses_path, backend, config,
                            max_new_tokens, use_chat_template):
    for label in ['positive', 'negative']:
        out_path = responses_path / f'ood_{label[:3]}.json'
        if not config.force and out_path.exists():
            continue
        n = _generate_and_write(ood_scenarios[label], out_path, backend.model, backend.tokenizer,
                                use_chat_template, max_new_tokens, config.temperature,
                                config.seed, config.rollouts)
        print(f"    ood_{label}: {n} responses")


def _write_generation_metadata(responses_path, config, trait, model, max_new_tokens,
                               use_chat_template, scenarios, ood_scenarios):
    trait_dir = get_path('datasets.trait', trait=trait)
    input_hashes = {
        'definition': content_hash(trait_dir / 'definition.txt'),
    }
    if config.replication_level == "full":
        # Full mode doesn't consume positive.jsonl — it reads topics_file and
        # batched_story_template_file. Hash the actually-consumed inputs so
        # staleness checks point at the right files.
        cfg = load_extraction_config(trait)
        if 'topics_file' in cfg:
            input_hashes['topics_file'] = content_hash(cfg['topics_file'])
        if 'batched_story_template_file' in cfg:
            input_hashes['batched_story_template_file'] = content_hash(
                cfg['batched_story_template_file']
            )
    else:
        input_hashes['positive'] = content_hash(get_scenario_path(trait, 'positive'))
        if 'negative' in scenarios:
            input_hashes['negative'] = content_hash(get_scenario_path(trait, 'negative'))
        if ood_scenarios is not None:
            input_hashes['ood_positive'] = content_hash(get_ood_scenario_path(trait, 'positive'))
            input_hashes['ood_negative'] = content_hash(get_ood_scenario_path(trait, 'negative'))
    with open(responses_path / 'metadata.json', 'w') as f:
        json.dump({
            'model': model.config.name_or_path, 'experiment': config.experiment,
            'trait': trait, 'max_new_tokens': max_new_tokens,
            'chat_template': use_chat_template, 'rollouts': config.rollouts,
            'temperature': config.temperature, 'seed': config.seed,
            'timestamp': datetime.now().isoformat(),
            'polarity': 'single' if 'negative' not in scenarios else 'contrastive',
            'has_ood': ood_scenarios is not None,
            'replication_level': config.replication_level,
            'input_hashes': input_hashes,
        }, f, indent=2)


# =============================================================================
# Stage 2: vetting
# =============================================================================

def vet_responses(config, trait, variant_name, backend, use_chat_template):
    """Stage 2: LLM judge scores responses. Returns VettingStats."""
    if not _should_run(config, 2):
        return VettingStats.skip()

    scores_file = (
        get_path("extraction.trait", experiment=config.experiment,
                 trait=trait, model_variant=variant_name)
        / "vetting" / "response_scores.json"
    )

    if not scores_file.exists() or config.force:
        if is_rank_zero():
            print(f"  [2] Vetting responses...")
            _vet_responses_judge(
                config.experiment, trait, variant_name,
                config.pos_threshold, config.neg_threshold, config.max_concurrent,
                estimate_trait_tokens=config.adaptive,
                position=config.position,
                tokenizer=backend.tokenizer,
                use_chat_template=use_chat_template,
            )
        tp_barrier()

    if not scores_file.exists():
        return VettingStats.skip()

    with open(scores_file) as f:
        summary = json.load(f).get('summary', {})
    return VettingStats.from_summary(summary)


# =============================================================================
# Stage 3+4: extraction
# =============================================================================

def _resolve_adaptive_position(config, trait, variant_name):
    if not config.adaptive:
        return config.position
    llm_pos = load_llm_judge_position(config.experiment, trait, variant_name)
    if llm_pos:
        print(f"  Adaptive position: {llm_pos}")
        return llm_pos
    return config.position


def _has_activations(config, trait, variant_name, position):
    metadata = get_activation_metadata_path(config.experiment, trait, variant_name,
                                            config.component, position)
    if not metadata.exists():
        return False
    stacked = get_activation_path(config.experiment, trait, variant_name,
                                  config.component, position)
    if stacked.exists():
        return True
    act_dir = get_activation_dir(config.experiment, trait, variant_name,
                                 config.component, position)
    return any(act_dir.glob("train_layer*.pt"))


def _has_vectors(config, trait, variant_name, position):
    return all(
        list(get_vector_dir(config.experiment, trait, m, variant_name,
                            config.component, position).glob("layer*.pt"))
        for m in config.methods
    )


def extract_vectors_for_trait(config, trait, variant_name, backend):
    """Stages 3+4: Forward pass → activations → trait vectors. Resolves adaptive
    position internally."""
    position = _resolve_adaptive_position(config, trait, variant_name)
    cached_activations = None

    if _should_run(config, 3) and (config.force or not _has_activations(config, trait, variant_name, position)):
        print(f"  [3] Extracting activations...")
        cached_activations = extract_activations_for_trait(
            config.experiment, trait, variant_name, backend, config.val_split,
            position=position, component=config.component,
            use_vetting_filter=config.vet_responses, paired_filter=config.paired_filter,
            layers=config.layers,
            pos_threshold=config.pos_threshold, neg_threshold=config.neg_threshold,
            save_activations=config.save_activations,
        )
        tp_barrier()

    if _should_run(config, 4) and (config.force or not _has_vectors(config, trait, variant_name, position)):
        print(f"  [4] Extracting vectors...")
        _train_vectors_for_trait(
            config.experiment, trait, variant_name, config.methods,
            layers=config.layers, component=config.component, position=position,
            activations=cached_activations,
        )


# =============================================================================
# Stage 5: logit lens
# =============================================================================

def run_logit_lens(config, trait, variant_name, backend):
    """Stage 5: Project residual vectors through unembedding to vocab tokens.
    Runs on every layer × method (cheap — model is already loaded)."""
    if not _should_run(config, 5):
        return
    from analysis.vectors.logit_lens import analyze_trait
    print(f"  [5] Logit lens (all layers)...")
    results = analyze_trait(
        config.experiment, trait, variant_name,
        backend.model, backend.tokenizer,
        layer_range=(0, 9999),
    )
    if results is None:
        return
    if is_rank_zero():
        out_path = get_path('extraction.logit_lens', experiment=config.experiment,
                            trait=trait, model_variant=variant_name)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)


# =============================================================================
# Stage 6: evaluation
# =============================================================================

def evaluate_extraction(config, traits, variant_name):
    """Stage 6: Quality metrics on held-out validation data."""
    if not _should_run(config, 6):
        return
    eval_path = get_path("extraction_eval.evaluation", experiment=config.experiment)
    if eval_path.exists() and not config.force:
        return
    from analysis.vectors.extraction_evaluation import main as run_eval
    print(f"\n[6] Evaluating ({len(traits)} traits)...")
    run_eval(config.experiment, model_variant=variant_name,
             methods=",".join(config.methods), component=config.component,
             position=config.position)
