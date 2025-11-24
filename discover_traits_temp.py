
import sys
from pathlib import Path

# Add the parent directory of the script to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.paths import get as get_path

def discover_traits(experiment: str) -> list[str]:
    """
    Finds all traits within an experiment's extraction directory.
    This assumes the user has already created the directory structure.
    e.g., `experiments/my_exp/extraction/cognitive_state/confidence`
    """
    extraction_dir = get_path('extraction.base', experiment=experiment)
    traits = []
    if not extraction_dir.is_dir():
        return []

    for category_dir in extraction_dir.iterdir():
        if category_dir.is_dir():
            for trait_dir in category_dir.iterdir():
                if trait_dir.is_dir():
                    traits.append(f"{category_dir.name}/{trait_dir.name}")
    return sorted(traits)

if __name__ == "__main__":
    # Assuming 'gemma_2b_cognitive_nov21' is the experiment name
    experiment_name = "gemma_2b_cognitive_nov21"
    discovered_traits = discover_traits(experiment_name)
    print("Discovered Traits:")
    for trait in discovered_traits:
        print(trait)
