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

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from utils.paths import get as get_path


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

    Supports both formats:
    - JSONL: {"prompt": "...", "system_prompt": "..."} per line
    - TXT: One prompt per line (no system_prompt)

    Args:
        trait: Trait path like "category/trait_name"
        polarity: If specified, only load "positive" or "negative".
                  If None, load both.

    Returns:
        Dict with "positive" and/or "negative" keys, each containing
        list of {"prompt": str, "system_prompt": Optional[str]}
    """
    trait_dir = get_path('datasets.trait', trait=trait)
    polarities = [polarity] if polarity else ['positive', 'negative']

    result = {}
    for pol in polarities:
        result[pol] = _load_polarity(trait_dir, pol)

    return result


def _load_polarity(trait_dir: Path, polarity: str) -> List[dict]:
    """Load scenarios for a single polarity (positive or negative)."""
    jsonl_file = trait_dir / f'{polarity}.jsonl'
    txt_file = trait_dir / f'{polarity}.txt'

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

    elif txt_file.exists():
        with open(txt_file, 'r') as f:
            return [{'prompt': line.strip()} for line in f if line.strip()]

    else:
        raise FileNotFoundError(f"No scenario file found: {jsonl_file} or {txt_file}")


def get_scenario_count(trait: str) -> Dict[str, int]:
    """Get count of scenarios per polarity without loading full data."""
    trait_dir = get_path('datasets.trait', trait=trait)
    counts = {}

    for polarity in ['positive', 'negative']:
        jsonl_file = trait_dir / f'{polarity}.jsonl'
        txt_file = trait_dir / f'{polarity}.txt'

        if jsonl_file.exists():
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
