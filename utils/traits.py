"""
Trait data loading utilities.

Input: Trait name (e.g., "category/trait_name")
Output: Trait definition, scenarios, steering config

Usage:
    from utils.traits import load_trait_definition, load_scenarios, load_steering_data

    definition = load_trait_definition("chirp/refusal_v2")
    scenarios = load_scenarios("chirp/refusal_v2")  # {"positive": [...], "negative": [...]}
    data = load_steering_data("alignment/deception")
    # data.questions, data.trait_name, data.trait_definition
"""

import hashlib
import json
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from utils.paths import get as get_path


def load_trait_metadata(trait: str) -> dict:
    """Load per-trait metadata from trait.yaml if present.

    Returns {} if no trait.yaml exists (all defaults apply).
    Supports: polarity, position, max_new_tokens, temperature,
    cross_trait_normalize, neutral_pc_denoise, and any custom fields.
    """
    trait_dir = get_path('datasets.trait', trait=trait)
    yaml_path = trait_dir / 'trait.yaml'
    if not yaml_path.exists():
        return {}
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}


def load_extraction_config(trait: str) -> dict:
    """Load extraction_config.yaml with category-level + per-trait cascade.

    Checks:
      1. datasets/traits/{category}/extraction_config.yaml (category-level)
      2. datasets/traits/{category}/{trait}/extraction_config.yaml (per-trait override)

    Per-trait values override category-level. Returns {} if neither exists.
    Supported fields:
      - position, max_new_tokens, methods (list), temperature, rollouts
      - prompt_template: optional str. When set, each scenario's `prompt` field
        is treated as a topic and substituted into the template at load time
        (see load_scenarios). Supports `{topic}` and `{emotion}` placeholders.
      - *_file: any field whose key ends in `_file` is treated as a path
        reference. The path is resolved EAGERLY against the YAML file's own
        parent directory (pre-merge), converted to an absolute Path, and
        stored under the same key. Example:
          batched_story_template_file: prompts/story.txt
        in a category-level YAML at /a/b/c/extraction_config.yaml resolves to
        /a/b/c/prompts/story.txt. Per-trait overrides use their own trait
        directory as the base. Non-existence is NOT checked at load time —
        callers open() the file and get FileNotFoundError if missing.
    Note: polarity is handled separately by load_trait_metadata() / load_scenarios().
    """
    traits_base = get_path('datasets.traits')
    parts = trait.split('/')
    merged = {}

    # Category-level: everything except last part is the category
    if len(parts) >= 2:
        category_dir = traits_base / '/'.join(parts[:-1])
        cat_yaml = category_dir / 'extraction_config.yaml'
        if cat_yaml.exists():
            merged.update(_load_yaml_with_path_resolution(cat_yaml))

    # Per-trait override
    trait_dir = traits_base / trait
    trait_yaml = trait_dir / 'extraction_config.yaml'
    if trait_yaml.exists():
        merged.update(_load_yaml_with_path_resolution(trait_yaml))

    return merged


def _load_yaml_with_path_resolution(yaml_path: Path) -> dict:
    """Load a YAML and eagerly resolve `*_file` keys to absolute paths.

    Any field whose key ends in `_file` and whose value is a string is
    rewritten to `yaml_path.parent / value` resolved to an absolute Path.
    This happens BEFORE the cascade merge so each config's file references
    stay anchored to where the config was written, regardless of override
    chain. Absolute paths in the input are preserved unchanged.
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        return {}
    base_dir = yaml_path.parent
    for key, value in list(data.items()):
        if key.endswith('_file') and isinstance(value, str):
            p = Path(value)
            if not p.is_absolute():
                p = (base_dir / p).resolve()
            data[key] = p
    return data


def load_trait_definition(trait: str) -> str:
    """
    Load trait definition from datasets/traits/{trait}/definition.txt.

    Returns the definition text, or a generated fallback if file doesn't exist.
    """
    def_file = get_path('datasets.trait_definition', trait=trait)
    if def_file.exists():
        return def_file.read_text().strip()
    # Fallback: generate from trait name
    trait_name = trait.split('/')[-1].replace('_', ' ')
    return f"The trait '{trait_name}'"


def load_scenarios(trait: str, polarity: str = None) -> Dict[str, List[dict]]:
    """
    Load scenarios from datasets/traits/{trait}/.

    Supports three formats (precedence: .json > .jsonl > .txt):
    - JSON:  {"prompts": [...], "system_prompts": [...]} — expanded to the
             full cartesian product at load time, grouped by system_prompt
             (outer) × prompt (inner). N prompts × M system_prompts = N*M entries.
    - JSONL: {"prompt": "...", "system_prompt": "..."} per line
    - TXT:   One prompt per line (no system_prompt)

    Args:
        trait: Trait path like "category/trait_name"
        polarity: If specified, only load "positive" or "negative".
                  If None, load both.

    Returns:
        Dict with "positive" and/or "negative" keys, each containing
        list of {"prompt": str, "system_prompt": Optional[str]}
    """
    trait_dir = get_path('datasets.trait', trait=trait)

    # Check metadata before loading — single-polarity traits only load positive
    # Check both trait.yaml and extraction_config.yaml for polarity
    metadata = load_trait_metadata(trait)
    extraction_cfg = load_extraction_config(trait)
    is_single_polarity = metadata.get('polarity') == 'single' or extraction_cfg.get('polarity') == 'single'

    if polarity:
        polarities = [polarity]
    elif is_single_polarity:
        polarities = ['positive']
    else:
        polarities = ['positive', 'negative']

    result = {}
    for pol in polarities:
        result[pol] = _load_polarity(trait_dir, pol)

    # Optional template substitution: if extraction_config has a `prompt_template`,
    # treat each scenario's existing `prompt` field as a topic and render it through
    # the template. Substitutes `{topic}` and `{emotion}` placeholders.
    # YAML `|` block scalars append a trailing newline — strip it so that authors
    # get the same rendered output whether they used `|` or `|-`.
    template = extraction_cfg.get('prompt_template')
    if template:
        template = template.rstrip('\n')
        emotion = resolve_emotion_surface(trait)
        for pol in result:
            result[pol] = [
                {**s, 'prompt': template.format(topic=s['prompt'], emotion=emotion)}
                for s in result[pol]
            ]

    # Assert matched scenario counts (skip for single-polarity traits)
    if not is_single_polarity and 'positive' in result and 'negative' in result:
        n_pos, n_neg = len(result['positive']), len(result['negative'])
        if n_pos != n_neg:
            raise ValueError(
                f"Scenario count mismatch for {trait}: {n_pos} positive, {n_neg} negative. "
                f"Positive and negative files must have the same number of scenarios."
            )

    return result


def resolve_emotion_surface(trait: str) -> str:
    """Return the emotion surface form used in `{emotion}` template substitution.

    Resolution order:
      1. `emotion_surface` field in trait.yaml, if present.
      2. Last path component of the trait name, with underscores converted to hyphens
         (matches paper convention: "grief_stricken" → "grief-stricken").
    """
    metadata = load_trait_metadata(trait)
    surface = metadata.get('emotion_surface')
    if isinstance(surface, str) and surface.strip():
        return surface
    return trait.split('/')[-1].replace('_', '-')


def get_scenario_path(trait: str, polarity: str) -> Path:
    """Resolve the scenario file path for a trait polarity (.json, .jsonl, or .txt)."""
    trait_dir = get_path('datasets.trait', trait=trait)
    for ext in ('json', 'jsonl', 'txt'):
        candidate = trait_dir / f'{polarity}.{ext}'
        if candidate.exists():
            return candidate
    return trait_dir / f'{polarity}.txt'


def get_scenario_format(trait: str) -> str:
    """Return 'json', 'jsonl', or 'txt' based on which scenario file exists."""
    path = get_scenario_path(trait, 'positive')
    suffix = path.suffix.lstrip('.')
    return suffix if suffix in ('json', 'jsonl', 'txt') else 'txt'


def _load_polarity(trait_dir: Path, polarity: str) -> List[dict]:
    """Load scenarios for a single polarity (positive or negative).

    Precedence: .json (cartesian) > .jsonl (explicit) > .txt (prompt-only).
    Fails fast if multiple formats coexist — keep exactly one per polarity.
    """
    json_file = trait_dir / f'{polarity}.json'
    jsonl_file = trait_dir / f'{polarity}.jsonl'
    txt_file = trait_dir / f'{polarity}.txt'

    present = [f for f in (json_file, jsonl_file, txt_file) if f.exists()]
    if len(present) > 1:
        raise ValueError(
            f"Multiple scenario formats found for {polarity} in {trait_dir}: "
            f"{[f.name for f in present]}. Keep exactly one."
        )

    if json_file.exists():
        return _load_polarity_json(json_file)

    if jsonl_file.exists():
        scenarios = []
        with open(jsonl_file, 'r') as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{jsonl_file} line {i+1}: invalid JSON: {e}")
                if 'prompt' not in item:
                    raise ValueError(f"{jsonl_file} line {i+1}: missing 'prompt' field")
                scenarios.append(item)
        return scenarios

    if txt_file.exists():
        with open(txt_file, 'r') as f:
            return [{'prompt': line.strip()} for line in f if line.strip()]

    raise FileNotFoundError(
        f"No scenario file found for {polarity} in {trait_dir} "
        f"(looked for {polarity}.json, {polarity}.jsonl, {polarity}.txt)"
    )


def _load_polarity_json(json_file: Path) -> List[dict]:
    """Parse a cartesian-format scenario file and expand to list of dicts.

    Grouping: system_prompt (outer) × prompt (inner) — all prompts under
    system_prompts[0] come first, then all prompts under system_prompts[1], etc.
    Matches the existing pre-expanded .jsonl layout.
    """
    with open(json_file, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"{json_file}: invalid JSON: {e}")

    if not isinstance(data, dict):
        raise ValueError(
            f"{json_file}: expected an object with 'prompts' and 'system_prompts' keys, "
            f"got {type(data).__name__}"
        )
    if 'prompts' not in data or 'system_prompts' not in data:
        raise ValueError(
            f"{json_file}: must contain both 'prompts' and 'system_prompts' keys. "
            f"(Use .jsonl format for explicit per-line pairs without cartesian expansion.)"
        )
    prompts = data['prompts']
    system_prompts = data['system_prompts']
    if not isinstance(prompts, list) or not isinstance(system_prompts, list):
        raise ValueError(f"{json_file}: 'prompts' and 'system_prompts' must both be lists")
    if not prompts or not system_prompts:
        raise ValueError(f"{json_file}: 'prompts' and 'system_prompts' must both be non-empty")
    if not all(isinstance(p, str) for p in prompts):
        raise ValueError(f"{json_file}: 'prompts' entries must all be strings")
    if not all(isinstance(sp, str) for sp in system_prompts):
        raise ValueError(f"{json_file}: 'system_prompts' entries must all be strings")
    if any(not p.strip() for p in prompts):
        raise ValueError(f"{json_file}: 'prompts' contains an empty or whitespace-only entry")
    if any(not sp.strip() for sp in system_prompts):
        raise ValueError(f"{json_file}: 'system_prompts' contains an empty or whitespace-only entry")

    scenarios = []
    for sp in system_prompts:
        for p in prompts:
            scenarios.append({'prompt': p, 'system_prompt': sp})
    return scenarios


def load_ood_scenarios(trait: str) -> Optional[Dict[str, List[dict]]]:
    """
    Load OOD validation scenarios from datasets/traits/{trait}/ood_*.jsonl.

    Mirrors load_scenarios() for the ood_positive/ood_negative files. OOD data is
    never vetted and has no train/val split — all samples are held-out for validation.

    Counts between positive and negative are NOT required to match (unlike training
    data) — OOD metrics are group-level (mean projection comparison), not pair-based.

    Returns:
        Dict with 'positive' and 'negative' keys if BOTH ood_positive and ood_negative
        files exist (any of .json, .jsonl, or .txt). Returns None if either is missing.
    """
    trait_dir = get_path('datasets.trait', trait=trait)

    def _any_format_exists(stem: str) -> bool:
        return any((trait_dir / f'{stem}.{ext}').exists() for ext in ('json', 'jsonl', 'txt'))

    pos_exists = _any_format_exists('ood_positive')
    neg_exists = _any_format_exists('ood_negative')

    if not (pos_exists and neg_exists):
        return None

    return {
        'positive': _load_polarity(trait_dir, 'ood_positive'),
        'negative': _load_polarity(trait_dir, 'ood_negative'),
    }


def get_ood_scenario_path(trait: str, polarity: str) -> Path:
    """Resolve the OOD scenario file path for a trait polarity (.json, .jsonl, or .txt).

    polarity: 'positive' or 'negative' (internally prefixed with 'ood_').
    """
    trait_dir = get_path('datasets.trait', trait=trait)
    for ext in ('json', 'jsonl', 'txt'):
        candidate = trait_dir / f'ood_{polarity}.{ext}'
        if candidate.exists():
            return candidate
    return trait_dir / f'ood_{polarity}.txt'


def content_hash_of_rendered_scenarios(trait: str, polarity: str) -> str:
    """SHA256 over canonicalized rendered scenarios for this polarity.

    Format-invariant: same hash regardless of .jsonl vs .json+template storage,
    as long as rendered prompts are identical. Use for change-detection logic
    that must survive future format migrations.
    """
    scenarios = load_scenarios(trait, polarity=polarity)[polarity]
    canonical = json.dumps(
        scenarios,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def get_scenario_count(trait: str) -> Dict[str, int]:
    """Get count of scenarios per polarity without loading full data.

    For .json cartesian format, count = len(prompts) * len(system_prompts)
    (matches the number of entries load_scenarios() returns after expansion).
    """
    trait_dir = get_path('datasets.trait', trait=trait)
    counts = {}

    for polarity in ['positive', 'negative']:
        json_file = trait_dir / f'{polarity}.json'
        jsonl_file = trait_dir / f'{polarity}.jsonl'
        txt_file = trait_dir / f'{polarity}.txt'

        if json_file.exists():
            with open(json_file, 'r') as f:
                data = json.load(f)
            counts[polarity] = len(data.get('prompts', [])) * len(data.get('system_prompts', []))
        elif jsonl_file.exists():
            with open(jsonl_file, 'r') as f:
                counts[polarity] = sum(1 for line in f if line.strip())
        elif txt_file.exists():
            with open(txt_file, 'r') as f:
                counts[polarity] = sum(1 for line in f if line.strip())
        else:
            counts[polarity] = 0

    return counts


# --- Steering data loading (merged from steering/steering_data.py) ---

@dataclass
class SteeringData:
    """Data for steering evaluation with optional custom prompts."""
    questions: List[str]
    trait_name: str
    trait_definition: str
    prompts_file: Path
    eval_prompt: Optional[str] = None       # Custom trait scoring prompt
    coherence_prompt: Optional[str] = None  # Custom coherence prompt
    vet_prompt: Optional[str] = None        # Custom vetting prompt
    direction: Optional[str] = None         # "positive" or "negative" (RLHF traits)


def load_steering_data(trait: str) -> SteeringData:
    """
    Load steering evaluation data for a trait.

    Args:
        trait: Trait path like 'alignment/deception'

    Returns:
        SteeringData with questions, trait info, and optional custom prompts

    Raises:
        FileNotFoundError: If steering.json doesn't exist
        ValueError: If steering.json is missing 'questions'
    """
    # Load questions and optional prompts from steering.json
    prompts_file = get_path('datasets.trait_steering', trait=trait)
    if not prompts_file.exists():
        raise FileNotFoundError(
            f"Steering prompts not found: {prompts_file}\n"
            f"Create JSON with 'questions' array."
        )

    with open(prompts_file) as f:
        data = json.load(f)

    if "questions" not in data:
        raise ValueError(f"Missing 'questions' in {prompts_file}")

    raw_questions = data["questions"]
    # Support both plain strings and objects with a "question" field
    questions = []
    for q in raw_questions:
        if isinstance(q, str):
            questions.append(q)
        elif isinstance(q, dict) and "question" in q:
            questions.append(q["question"])
        else:
            raise ValueError(f"Invalid question format in {prompts_file}: {type(q)}")

    # Load trait definition from definition.txt
    def_file = get_path('datasets.trait_definition', trait=trait)
    if def_file.exists():
        with open(def_file) as f:
            trait_definition = f.read().strip()
    else:
        trait_name_fallback = trait.split('/')[-1].replace('_', ' ')
        trait_definition = f"The trait '{trait_name_fallback}'"

    # Extract trait name from path
    trait_name = trait.split('/')[-1]  # e.g., 'alignment/deception' -> 'deception'

    return SteeringData(
        questions=questions,
        trait_name=trait_name,
        trait_definition=trait_definition,
        prompts_file=prompts_file,
        eval_prompt=data.get("eval_prompt"),
        coherence_prompt=data.get("coherence_prompt"),
        vet_prompt=data.get("vet_prompt"),
        direction=data.get("direction"),
    )


def load_questions_from_file(path: str) -> List[str]:
    """Load questions from any JSON file. Supports both steering.json and inference formats."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Questions file not found: {p}")

    with open(p) as f:
        data = json.load(f)

    # steering.json format: {"questions": ["...", ...]}
    if "questions" in data:
        questions = []
        for q in data["questions"]:
            if isinstance(q, str):
                questions.append(q)
            elif isinstance(q, dict) and "question" in q:
                questions.append(q["question"])
        return questions

    # inference format: {"prompts": [{"text": "..."}, ...]}
    if "prompts" in data:
        return [p.get('text') or p.get('prompt') for p in data["prompts"] if p.get('text') or p.get('prompt')]

    raise ValueError(f"Unrecognized format in {p}: need 'questions' or 'prompts' key")


def load_questions_from_inference(prompt_set: str) -> List[str]:
    """
    Load questions from an inference prompt set.

    Args:
        prompt_set: Prompt set path like 'rm_syco/train_100'

    Returns:
        List of question strings extracted from the prompt set

    Raises:
        FileNotFoundError: If prompt set doesn't exist
        ValueError: If prompt set has no 'prompts' array
    """
    prompt_file = get_path('datasets.inference') / f"{prompt_set}.json"
    if not prompt_file.exists():
        raise FileNotFoundError(
            f"Inference prompt set not found: {prompt_file}\n"
            f"Available in datasets/inference/"
        )

    with open(prompt_file) as f:
        data = json.load(f)

    if "prompts" not in data:
        raise ValueError(f"Missing 'prompts' in {prompt_file}")

    # Extract text from each prompt object
    questions = []
    for p in data["prompts"]:
        # Handle both 'text' and 'prompt' keys
        text = p.get('text') or p.get('prompt')
        if text:
            questions.append(text)

    return questions
