import logging

import torch
from tqdm import tqdm

from .constants import DEVICE, NUM_FOLDS
from .plotting import plot_all
from .utils import load_data, load_model, predict_trajectory

logger = logging.getLogger(__name__)


def run_inference_on_models(
    model_types, datasets, prediction_steps, use_mini_models, output_folder
):
    logger.info("Starting inference process")

    for model_type in model_types:
        for dataset_type in datasets:
            logger.info(f"Running inference for {model_type} on {dataset_type}")

            try:
                for split in ["train", "val", "test"]:
                    logger.info(f"Processing {split} set")
                    data, scaler = load_data(dataset_type, split)

                    models = []
                    for fold in range(NUM_FOLDS):
                        model = load_model(
                            model_type, dataset_type, fold, use_mini_models
                        )
                        models.append(model)

                    spaceship_ids = data["spaceship_id"].unique()

                    for steps in prediction_steps:
                        logger.info(f"Predicting for {steps} steps")

                        for spaceship_id in tqdm(
                            spaceship_ids, desc=f"Processing {split} spaceships"
                        ):
                            initial_sequence = (
                                data[data["spaceship_id"] == spaceship_id]
                                .iloc[:3][["x", "y", "Vx", "Vy"]]
                                .values
                            )
                            initial_sequence = torch.FloatTensor(initial_sequence).to(
                                DEVICE
                            )

                            predictions = predict_trajectory(
                                models, initial_sequence, scaler, steps
                            )
                            actual = (
                                data[data["spaceship_id"] == spaceship_id]
                                .iloc[:steps][["x", "y", "Vx", "Vy"]]
                                .values
                            )

                            plot_all(
                                actual,
                                predictions,
                                model_type,
                                dataset_type,
                                steps,
                                spaceship_id,
                                f"{output_folder}/{split}",
                            )

            except Exception as e:
                logger.error(
                    f"Error in inference for {model_type} on {dataset_type}: {str(e)}"
                )
                logger.exception("Exception details:")

    logger.info("Inference process completed")
