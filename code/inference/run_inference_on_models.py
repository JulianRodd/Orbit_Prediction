import logging

import numpy as np
import torch
from constants import DEVICE
from tqdm import tqdm
from training.constants import N_STEPS

from .constants import NUM_FOLDS, PREDICTION_START
from .csv_utils import (
    export_consolidated_results,
    export_predictions_to_csv,
    export_results_to_csv,
)
from .plotting import plot_all
from .utils import (
    calculate_mse,
    denormalize_data,
    load_data,
    load_model,
    predict_trajectory,
)

logger = logging.getLogger(__name__)


def run_inference_on_models(
    model_types, datasets, prediction_steps, use_mini_models, output_folder
):
    logger.info("Starting inference process")

    all_results = {
        dataset: {model: {} for model in model_types} for dataset in datasets
    }

    for model_type in model_types:
        for dataset_type in datasets:
            logger.info(f"Running inference for {model_type} on {dataset_type}")

            try:
                for split in ["train", "val", "test"]:
                    logger.info(f"Processing {split} set")

                    all_position_mse = {steps: [] for steps in prediction_steps}
                    all_velocity_mse = {steps: [] for steps in prediction_steps}

                    # Load data and scaler for the current split
                    if split == "test":
                        data, scaler = load_data(dataset_type, split, None)
                    else:
                        data, scaler = load_data(dataset_type, split, 0)

                    # Load models for each fold
                    models = [
                        load_model(model_type, dataset_type, fold, use_mini_models)
                        for fold in range(NUM_FOLDS)
                    ]

                    for spaceship_id in tqdm(
                        data["spaceship_id"].unique(),
                        desc=f"Processing {split} spaceships",
                    ):
                        spaceship_data = data[data["spaceship_id"] == spaceship_id]

                        initial_sequence = spaceship_data.iloc[
                            PREDICTION_START - N_STEPS - 1 : PREDICTION_START - 1
                        ][["x", "y", "Vx", "Vy"]].values
                        initial_sequence = torch.FloatTensor(initial_sequence).to(
                            DEVICE
                        )

                        actual = spaceship_data.iloc[-1:][["x", "y", "Vx", "Vy"]].values

                        for steps in prediction_steps:
                            try:
                                # Predictions for each fold
                                fold_predictions = []
                                fold_position_mse = []
                                fold_velocity_mse = []

                                for fold in range(NUM_FOLDS):
                                    if hasattr(models[fold], "reset_state"):
                                        models[fold].reset_state()

                                    predictions = predict_trajectory(
                                        models[fold],
                                        initial_sequence,
                                        scaler,
                                        steps,
                                        model_type,
                                    )

                                    # Handle NaN values in predictions
                                    if np.isfinite(predictions).all():
                                        fold_predictions.append(predictions)
                                        position_mse, velocity_mse, _ = calculate_mse(
                                            actual, predictions[-1:]
                                        )
                                        fold_position_mse.append(position_mse)
                                        fold_velocity_mse.append(velocity_mse)

                                # Ensure we have at least one valid prediction
                                if len(fold_predictions) > 0:
                                    # Average predictions across folds, ignoring NaN values
                                    ensemble_predictions = np.nanmean(
                                        fold_predictions, axis=0
                                    )

                                    # Export predictions for each spaceship
                                    export_predictions_to_csv(
                                        ensemble_predictions,
                                        model_type,
                                        dataset_type,
                                        split,
                                        spaceship_id,
                                        steps,
                                        output_folder,
                                    )

                                    # Store MSE values
                                    all_position_mse[steps].append(fold_position_mse)
                                    all_velocity_mse[steps].append(fold_velocity_mse)

                                    # Plot only for the first spaceship
                                    if spaceship_id == data["spaceship_id"].unique()[0]:
                                        full_trajectory = spaceship_data[
                                            ["x", "y", "Vx", "Vy"]
                                        ].values
                                        plot_all(
                                            denormalize_data(actual, scaler),
                                            ensemble_predictions,
                                            denormalize_data(full_trajectory, scaler),
                                            model_type,
                                            dataset_type,
                                            steps,
                                            spaceship_id,
                                            output_folder,
                                            split,
                                            PREDICTION_START - N_STEPS - 1,
                                        )
                                else:
                                    logger.warning(
                                        f"No valid predictions for {split}, {steps} steps, spaceship {spaceship_id}"
                                    )

                            except Exception as e:
                                logger.error(
                                    f"Error in prediction for spaceship {spaceship_id}, {steps} steps: {str(e)}"
                                )

                    # Calculate mean and std for each prediction step
                    for steps in prediction_steps:
                        if all_position_mse[steps] and all_velocity_mse[steps]:
                            pos_mean = np.nanmean(
                                [np.nanmean(mse) for mse in all_position_mse[steps]]
                            )
                            pos_std = np.nanmean(
                                [np.nanstd(mse) for mse in all_position_mse[steps]]
                            )
                            vel_mean = np.nanmean(
                                [np.nanmean(mse) for mse in all_velocity_mse[steps]]
                            )
                            vel_std = np.nanmean(
                                [np.nanstd(mse) for mse in all_velocity_mse[steps]]
                            )
                        else:
                            pos_mean, pos_std, vel_mean, vel_std = (
                                np.nan,
                                np.nan,
                                np.nan,
                                np.nan,
                            )
                            logger.warning(
                                f"No valid MSE values for {split}, {steps} steps"
                            )

                        all_results[dataset_type][model_type][f"{split}_{steps}"] = {
                            "position": (pos_mean, pos_std),
                            "velocity": (vel_mean, vel_std),
                        }

                export_results_to_csv(
                    all_results[dataset_type][model_type],
                    model_type,
                    dataset_type,
                    output_folder,
                )

            except Exception as e:
                logger.error(
                    f"Error in inference for {model_type} on {dataset_type}: {str(e)}"
                )
                logger.exception("Exception details:")

    export_consolidated_results(all_results, output_folder)
    logger.info("Inference process completed")
