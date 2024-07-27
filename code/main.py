import argparse
import logging

from data_generation.data_generation import data_generation
from inference.constants import DEFAULT_PREDICTION_STEPS
from inference.run_inference_on_models import run_inference_on_models
from training.constants import DATASET_TYPES, MODEL_TYPES
from training.train_models import train_models

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Train, run inference, or generate data for orbital prediction models"
    )
    parser.add_argument(
        "mode",
        choices=["train", "inference", "generate_data"],
        help="Mode of operation",
    )
    parser.add_argument(
        "--model_types", nargs="+", default=MODEL_TYPES, help="Model types to use"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=DATASET_TYPES, help="Datasets to use"
    )
    parser.add_argument(
        "--mini", action="store_true", help="Use mini networks for quick testing"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--use_wandb", action="store_true", help="Use Weights & Biases for logging"
    )
    parser.add_argument(
        "--prediction_steps",
        nargs="+",
        type=int,
        default=DEFAULT_PREDICTION_STEPS,
        help="Number of steps to predict for inference",
    )
    parser.add_argument(
        "--output_folder",
        default="inference_results",
        help="Folder to save inference results",
    )
    args = parser.parse_args()

    if args.mode == "train":
        logger.info("Starting training mode")
        train_models(
            model_types=args.model_types,
            datasets=args.datasets,
            mini=args.mini,
            seed=args.seed,
            prediction_steps=args.prediction_steps,
            use_wandb=args.use_wandb,
        )
    elif args.mode == "inference":
        logger.info("Starting inference mode")
        run_inference_on_models(
            model_types=args.model_types,
            datasets=args.datasets,
            prediction_steps=args.prediction_steps,
            use_mini_models=args.mini,
            output_folder=args.output_folder,
        )
    elif args.mode == "generate_data":
        logger.info("Starting data generation mode")
        data_generation(problem_types=args.datasets)


if __name__ == "__main__":
    main()
