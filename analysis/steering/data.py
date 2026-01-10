"""
Steering evaluation data loading with extensible custom prompts.

Input:
    - datasets/traits/{trait}/steering.json (questions, optional eval_prompt)
    - datasets/traits/{trait}/definition.txt (trait definition)

Output:
    - SteeringData dataclass with questions, trait info, and optional custom prompts

Usage:
    from analysis.steering.data import load_steering_data

    data = load_steering_data('alignment/deception')
    # data.questions, data.trait_name, data.trait_definition
    # data.eval_prompt (None if not in steering.json)
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from utils.paths import get as get_path


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

    questions = data["questions"]

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
    )


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
