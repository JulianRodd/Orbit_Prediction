import logging

import numpy as np
import torch
from constants import DEVICE
from tqdm import tqdm
from training.constants import N_STEPS

from .constants import NUM_FOLDS, PREDICTION_START
from .csv_utils import export_consolidated_results, export_results_to_csv
from .plotting import plot_all
from .utils import (calculate_mse, denormalize_data, load_data, load_model,
                    predict_trajectory)

# set logger to debug

logger = logging.getLogger(__name__)


def run_inference_on_models(model_types, datasets, prediction_steps, use_mini_models, output_folder):
    logger.info("Starting inference process")

    all_results = {dataset: {model: {} for model in model_types} for dataset in datasets}

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
                    models = [load_model(model_type, dataset_type, fold, use_mini_models) for fold in range(NUM_FOLDS)]

                    for spaceship_id in tqdm(data["spaceship_id"].unique(), desc=f"Processing {split} spaceships"):
                        spaceship_data = data[data["spaceship_id"] == spaceship_id]

                        # Use the last N_STEPS from the set until 5000 points as the initial sequence
                        initial_sequence = spaceship_data.iloc[PREDICTION_START-N_STEPS - 1 : PREDICTION_START -1][["x", "y", "Vx", "Vy"]].values
                        initial_sequence = torch.FloatTensor(initial_sequence).to(DEVICE)

                        actual = spaceship_data.iloc[-1:][["x", "y", "Vx", "Vy"]].values

                        for steps in prediction_steps:
                            try:
                                # Ensemble predictions
                                ensemble_predictions = []
                                for fold in range(NUM_FOLDS):
                                    # Reset model state if necessary (e.g., for LSTM)
                                    if hasattr(models[fold], 'reset_state'):
                                        models[fold].reset_state()

                                    fold_predictions = predict_trajectory(models[fold], initial_sequence, scaler, steps, model_type)
                                    ensemble_predictions.append(fold_predictions)

                                # Average ensemble predictions
                                predictions = np.mean(ensemble_predictions, axis=0)

                                # Calculate MSE for the last point
                                position_mse, velocity_mse, _ = calculate_mse(actual, predictions[-1:])

                                if np.isfinite(position_mse) and np.isfinite(velocity_mse):
                                    all_position_mse[steps].append(position_mse)
                                    all_velocity_mse[steps].append(velocity_mse)

                                # Plot only for the first spaceship
                                if spaceship_id == data["spaceship_id"].unique()[0]:
                                    full_trajectory = spaceship_data[["x", "y", "Vx", "Vy"]].values
                                    plot_all(
                                        denormalize_data(actual, scaler),
                                        predictions,
                                        denormalize_data(full_trajectory, scaler),
                                        model_type,
                                        dataset_type,
                                        steps,
                                        spaceship_id,
                                        output_folder,
                                        split,
                                        PREDICTION_START - N_STEPS - 1,  # prediction_start
                                    )
                            except Exception as e:
                                logger.error(f"Error in prediction for spaceship {spaceship_id}, {steps} steps: {str(e)}")

                    # Calculate mean and std for each prediction step
                    for steps in prediction_steps:
                        if all_position_mse[steps] and all_velocity_mse[steps]:
                            pos_mean, pos_std = np.mean(all_position_mse[steps]), np.std(all_position_mse[steps])
                            vel_mean, vel_std = np.mean(all_velocity_mse[steps]), np.std(all_velocity_mse[steps])
                        else:
                            pos_mean, pos_std, vel_mean, vel_std = np.nan, np.nan, np.nan, np.nan
                            logger.warning(f"No valid MSE values for {split}, {steps} steps")

                        all_results[dataset_type][model_type][f"{split}_{steps}"] = {
                            "position": (pos_mean, pos_std),
                            "velocity": (vel_mean, vel_std),
                        }

                export_results_to_csv(all_results[dataset_type][model_type], model_type, dataset_type, output_folder)

            except Exception as e:
                logger.error(f"Error in inference for {model_type} on {dataset_type}: {str(e)}")
                logger.exception("Exception details:")

    export_consolidated_results(all_results, output_folder)
    logger.info("Inference process completed")
