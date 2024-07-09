import argparse
import os

import numpy as np
import torch
from generics import *
from regression import PINN, OrbitLSTM, SimpleRegression, load_data, predict_future
from utils import plot_trajectory, plot_velocity


def load_model(model_type, input_size, hidden_size, output_size, checkpoint_path):
    if model_type == "PINN":
        model = PINN(input_size, hidden_size, output_size)
    elif model_type == "SimpleRegression":
        model = SimpleRegression(input_size, hidden_size, output_size)
    elif model_type == "LSTM":
        model = OrbitLSTM(input_size, hidden_size, NUM_LSTM_LAYERS, output_size)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def main(args):
    test_dataset, scaler, _ = load_data("test", args.dataset_type, None, N_STEPS)

    for model_type in args.model_types:
        predictions = []
        for fold in range(args.num_folds):
            if args.mini:
                checkpoint_path = f"{MINI_CHECKPOINT_LOCATION}/{model_type}/{args.dataset_type}/fold{fold}_mini_model.pth"
                hidden_size = MINI_HIDDEN_SIZE
            else:
                checkpoint_path = f"{CHECKPOINT_LOCATION}/{model_type}/{args.dataset_type}/fold{fold}_best_model.pth"
                hidden_size = HIDDEN_SIZE

            if not os.path.exists(checkpoint_path):
                print(f"Checkpoint not found for {model_type}, fold {fold}. Skipping.")
                continue

            model = load_model(
                model_type, INPUT_SIZE, hidden_size, OUTPUT_SIZE, checkpoint_path
            )

            initial_sequence = test_dataset[0][0].to(DEVICE)
            fold_predictions = predict_future(
                model, initial_sequence, scaler, args.prediction_steps, OUTPUT_SIZE
            )
            predictions.append(fold_predictions)

        if not predictions:
            print(f"No valid predictions for {model_type}. Skipping plotting.")
            continue

        # Average predictions across folds
        avg_predictions = np.mean(predictions, axis=0)

        # Get actual data
        actual = scaler.inverse_transform(
            test_dataset.sequences[: args.prediction_steps, -1, :].cpu().numpy()
        )

        # Plot results
        model_name = f"{model_type}_mini" if args.mini else model_type
        plot_trajectory(
            actual,
            avg_predictions,
            f"{model_name} Test Inference",
            "test",
            model_name,
            args.dataset_type,
            args.prediction_steps,
        )
        plot_velocity(
            actual,
            avg_predictions,
            DT,
            f"{model_name} Test Inference Velocity",
            "test",
            model_name,
            args.dataset_type,
            args.prediction_steps,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test inference for orbital prediction models"
    )
    parser.add_argument(
        "--model_types",
        nargs="+",
        default=["PINN", "SimpleRegression", "LSTM"],
        help="Model types to test",
    )
    parser.add_argument(
        "--dataset_type",
        choices=["two_body", "two_body_force_increased_acceleration", "three_body"],
        required=True,
        help="Dataset type",
    )
    parser.add_argument(
        "--num_folds",
        type=int,
        default=5,
        help="Number of folds to use for ensemble prediction",
    )
    parser.add_argument(
        "--prediction_steps", type=int, default=100, help="Number of steps to predict"
    )
    parser.add_argument(
        "--mini", action="store_true", help="Use mini models for inference"
    )
    args = parser.parse_args()

    main(args)
