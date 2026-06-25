from src.evaluation.lodo import run_lodo_evaluation
from src.models.model import build_model
from src.config import ModelConfig


def model_fn():
    return build_model(ModelConfig())


def main():
    datasets = {
        "DAIC-WOZ": [],
        "MODMA": [],
        "EATD": [],
    }
    results = run_lodo_evaluation(model_fn, datasets)
    for held_out, info in results.items():
        print(held_out, info["train_sets"], info["metrics"])


if __name__ == "__main__":
    main()
