import logging

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from .constants import (
    CHECKPOINT_BASE_PATH,
    DEVICE,
    HIDDEN_SIZE,
    INPUT_SIZE,
    MINI_CHECKPOINT_BASE_PATH,
    NUM_LSTM_LAYERS,
    OUTPUT_SIZE,
)

logger = logging.getLogger(__name__)


def load_data(dataset_type, split):
    try:
        if split == "test":
            df = pd.read_csv(f"datasets/{dataset_type}/{split}/{dataset_type}_{split}.csv")
        else:
            df = pd.read_csv(f"datasets/{dataset_type}/{split}/{dataset_type}_{split}_fold0.csv")
        scaler = MinMaxScaler()
        columns_to_scale = ["x", "y", "Vx", "Vy"]
        df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
        logger.info(f"Data loaded for {dataset_type}, {split}")
        return df, scaler
    except Exception as e:
        logger.error(f"Error loading data for {dataset_type}, {split}: {str(e)}")
        raise


def load_model(model_type, dataset_type, fold, use_mini_model):
    try:
        if use_mini_model:
            checkpoint_path = f"{MINI_CHECKPOINT_BASE_PATH}/{model_type}/{dataset_type}/fold{fold}_mini_model.pth"
        else:
            checkpoint_path = f"{CHECKPOINT_BASE_PATH}/{model_type}/{dataset_type}/fold{fold}_best_model.pth"

        state_dict = torch.load(checkpoint_path, map_location=DEVICE)

        if model_type == "SimpleRegression":
            from training.mlp.architecture import create_mlp_model

            model = create_mlp_model(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
        elif model_type == "LSTM":
            from training.lstm.architecture import create_lstm_model

            model = create_lstm_model(INPUT_SIZE, HIDDEN_SIZE, NUM_LSTM_LAYERS, OUTPUT_SIZE)
        elif model_type == "PINN":
            from training.pinn.architecture import create_pinn_model

            model = create_pinn_model(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
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


def predict_trajectory(models, initial_sequence, scaler, steps):
    try:
        current_input = initial_sequence.clone()
        predictions = []

        with torch.no_grad():
            for _ in range(steps):
                model_outputs = [model(current_input.unsqueeze(0)) for model in models]
                ensemble_output = torch.mean(torch.stack(model_outputs), dim=0).squeeze(
                    0
                )
                predictions.append(ensemble_output.cpu().numpy())
                current_input = torch.cat(
                    (current_input[1:], ensemble_output.unsqueeze(0)), 0
                )

        predictions = np.array(predictions)
        predictions = scaler.inverse_transform(predictions)
        logger.debug(f"Trajectory predicted for {steps} steps")
        return predictions
    except Exception as e:
        logger.error(f"Error in predict_trajectory: {str(e)}")
        raise
