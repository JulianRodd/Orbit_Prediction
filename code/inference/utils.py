import logging

import numpy as np
import pandas as pd
import torch
from constants import DEVICE
from sklearn.preprocessing import MinMaxScaler

from .constants import (
    CHECKPOINT_BASE_PATH,
    LSTM_HIDDEN_SIZE,
    LSTM_INPUT_SIZE,
    LSTM_NUM_LAYERS,
    LSTM_OUTPUT_SIZE,
    MINI_CHECKPOINT_BASE_PATH,
    MLP_HIDDEN_SIZE,
    MLP_INPUT_SIZE,
    MLP_OUTPUT_SIZE,
    PINN_HIDDEN_SIZE,
    PINN_INPUT_SIZE,
    PINN_OUTPUT_SIZE,
)

logger = logging.getLogger(__name__)


def calculate_mse(actual, predicted):
    position_mse = np.mean((actual[:, :2] - predicted[:, :2]) ** 2)
    velocity_mse = np.mean((actual[:, 2:] - predicted[:, 2:]) ** 2)
    combined_mse = np.mean((actual - predicted) ** 2)
    return position_mse, velocity_mse, combined_mse


def load_data(dataset_type, split, fold=None):
    try:
        columns_to_scale = ["x", "y", "Vx", "Vy"]

        if split == "test":
            # For test set, fit scaler on all training data
            train_data = []
            for i in range(5):  # Assuming 5 folds, adjust if different
                train_df = pd.read_csv(
                    f"datasets/{dataset_type}/train/{dataset_type}_train_fold{i}.csv"
                )
                train_data.append(train_df[columns_to_scale])
            all_train_data = pd.concat(train_data, axis=0)

            scaler = MinMaxScaler()
            scaler.fit(all_train_data)

            df = pd.read_csv(
                f"datasets/{dataset_type}/{split}/{dataset_type}_{split}.csv"
            )
        else:
            # For train and val sets, fit scaler on the specific fold's training data
            if fold is None:
                raise ValueError("Fold must be specified for train and validation sets")

            train_df = pd.read_csv(
                f"datasets/{dataset_type}/train/{dataset_type}_train_fold{fold}.csv"
            )
            scaler = MinMaxScaler()
            scaler.fit(train_df[columns_to_scale])

            df = pd.read_csv(
                f"datasets/{dataset_type}/{split}/{dataset_type}_{split}_fold{fold}.csv"
            )

        # Transform the data using the scaler
        df[columns_to_scale] = scaler.transform(df[columns_to_scale])

        return df, scaler
    except Exception as e:
        logger.error(
            f"Error loading data for {dataset_type}, {split}, fold {fold}: {str(e)}"
        )
        raise


def load_model(model_type, dataset_type, fold, use_mini_model):
    try:
        model_filename = model_type.lower()
        if use_mini_model:
            checkpoint_path = f"{MINI_CHECKPOINT_BASE_PATH}/{model_type}/{dataset_type}/{model_type}_fold{fold}_mini.pth"
        else:
            checkpoint_path = f"{CHECKPOINT_BASE_PATH}/{model_type}/{dataset_type}/{model_type}_fold{fold}.pth"

        state_dict = torch.load(checkpoint_path, map_location=DEVICE)

        if model_type == "MLP":
            from training.mlp.architecture import create_mlp_model

            model = create_mlp_model(MLP_INPUT_SIZE, MLP_HIDDEN_SIZE, MLP_OUTPUT_SIZE)
        elif model_type == "LSTM":
            from training.lstm.architecture import create_lstm_model

            model = create_lstm_model(
                LSTM_INPUT_SIZE, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, LSTM_OUTPUT_SIZE
            )
        elif model_type == "PINN":
            from training.pinn.architecture import create_pinn_model

            model = create_pinn_model(
                PINN_INPUT_SIZE, PINN_HIDDEN_SIZE, PINN_OUTPUT_SIZE
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        logger.info(f"Model loaded: {model_type}, fold {fold}")
        return model
    except Exception as e:
        logger.error(f"Error loading model {model_type}, fold {fold}: {str(e)}")
        raise


def denormalize_data(data, scaler):
    """Denormalize the data using the provided scaler."""
    if isinstance(data, np.ndarray):
        # Clip values to a reasonable range before inverse transform
        data_clipped = np.clip(data, -1e10, 1e10)
        return scaler.inverse_transform(data_clipped)
    elif isinstance(data, torch.Tensor):
        data_numpy = data.cpu().numpy()
        data_clipped = np.clip(data_numpy, -1e10, 1e10)
        return scaler.inverse_transform(data_clipped)
    else:
        raise TypeError("Data must be either a numpy array or a torch Tensor")


def predict_trajectory(
    model: torch.nn.Module,
    initial_sequence: torch.Tensor,
    scaler,
    steps: int,
    model_type: str,
):
    try:
        current_input = initial_sequence.clone()
        predictions = [
            denormalize_data(current_input[-1].cpu().numpy().reshape(1, -1), scaler)[0]
        ]

        with torch.no_grad():
            for _ in range(steps - 1):  # -1 because we already have the initial point
                if model_type == "MLP":
                    input = current_input.reshape(1, -1)
                elif model_type == "LSTM":
                    input = current_input.unsqueeze(0)
                else:  # PINN or other models
                    input = current_input.flatten().unsqueeze(0)

                output = model(input)

                # Denormalize the output
                denormalized_output = denormalize_data(output.cpu().numpy(), scaler)[0]
                predictions.append(denormalized_output)

                # Update current_input: remove oldest step and add new prediction
                current_input = torch.cat((current_input[1:], output), 0)

        predictions = np.array(predictions)
        logger.debug(f"Trajectory predicted for {steps} steps")
        return predictions
    except Exception as e:
        logger.error(f"Error in predict_trajectory: {str(e)}")
        raise
