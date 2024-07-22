import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from generics import (
    DEVICE, INPUT_SIZE, OUTPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_FOLDS,
    MINI_HIDDEN_SIZE, MINI_NUM_LAYERS, N_STEPS
)
from regression import SimpleRegression, OrbitLSTM, PINN, OrbitDataset
from utils import plot_full_trajectory_with_predictions, plot_full_velocity_with_predictions

def load_test_data(dataset_type):
    df = pd.read_csv(f"datasets/{dataset_type}/test/{dataset_type}_test.csv")
    scaler = MinMaxScaler()
    columns_to_scale = ["x", "y", "Vx", "Vy"]
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    sequences = []
    for spaceship in df["spaceship_id"].unique():
        spaceship_data = df[df["spaceship_id"] == spaceship][columns_to_scale]
        for i in range(len(spaceship_data) - N_STEPS):
            sequences.append(spaceship_data.iloc[i : i + N_STEPS + 1].values)

    return OrbitDataset(np.array(sequences)), scaler, df

def predict_trajectory(model, initial_sequence, steps):
    model.eval()
    current_input = initial_sequence.clone()
    predictions = []

    with torch.no_grad():
        for _ in range(steps):
            if isinstance(model, OrbitLSTM):
                model_input = current_input.unsqueeze(0)  # Add batch dimension for LSTM
            else:
                model_input = current_input.reshape(1, -1)  # Flatten for other models
            output = model(model_input).squeeze(0)
            predictions.append(output.cpu().numpy())
            current_input = torch.cat((current_input[1:], output.unsqueeze(0)), 0)

    return np.array(predictions)

def load_model(model_type, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    if model_type == "PINN":
        input_size = checkpoint['fc1.weight'].shape[1]
        hidden_size = checkpoint['fc1.weight'].shape[0]
        output_size = checkpoint['fc3.weight'].shape[0]
        model = PINN(input_size, hidden_size, output_size).to(DEVICE)
    elif model_type == "SimpleRegression":
        input_size = checkpoint['fc1.weight'].shape[1]
        hidden_size = checkpoint['fc1.weight'].shape[0]
        output_size = checkpoint['fc3.weight'].shape[0]
        model = SimpleRegression(input_size, hidden_size, output_size).to(DEVICE)
    elif model_type == "LSTM":
        input_size = checkpoint['lstm.weight_ih_l0'].shape[1]
        hidden_size = checkpoint['lstm.weight_hh_l0'].shape[0]
        num_layers = sum(1 for k in checkpoint.keys() if k.startswith('lstm.weight_ih_l'))
        output_size = checkpoint['fc2.weight'].shape[0]
        model = OrbitLSTM(input_size, hidden_size, num_layers, output_size).to(DEVICE)

    model.load_state_dict(checkpoint)
    return model

def main():
    parser = argparse.ArgumentParser(description="Inference script for orbital prediction")
    parser.add_argument("--model_types", nargs="+", default=["PINN", "SimpleRegression", "LSTM"], help="Model types to use")
    parser.add_argument("--datasets", nargs="+", default=["two_body", "two_body_force_increased_acceleration", "three_body"], help="Datasets to use")
    parser.add_argument("--mini", action="store_true", help="Use mini networks")
    parser.add_argument("--time_steps", nargs="+", type=int, default=[10, 100, 500, 1000], help="Number of time steps to predict")
    args = parser.parse_args()

    for model_type in args.model_types:
        for dataset_type in args.datasets:
            print(f"\nProcessing {model_type} on {dataset_type}")

            test_dataset, scaler, test_df = load_test_data(dataset_type)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

            for time_step in args.time_steps:
                print(f"  Predicting for {time_step} time steps")

                # Select four non-overlapping trajectories
                unique_spaceships = test_df["spaceship_id"].unique()
                selected_spaceships = unique_spaceships[:4]  # Select the first four spaceships

                all_predictions = []
                start_indices = []

                for spaceship_id in selected_spaceships:
                    spaceship_data = test_df[test_df["spaceship_id"] == spaceship_id][["x", "y", "Vx", "Vy"]]

                    # Randomly select a starting point
                    start_idx = np.random.randint(0, len(spaceship_data) - time_step - N_STEPS)
                    start_indices.append(start_idx)

                    initial_sequence = torch.FloatTensor(spaceship_data.iloc[start_idx:start_idx+N_STEPS].values).to(DEVICE)

                    # Ensemble predictions from all folds
                    fold_predictions = []
                    for fold in range(NUM_FOLDS):
                        if args.mini:
                            checkpoint_path = f"mini_checkpoints/{model_type}/{dataset_type}/fold{fold}_mini_model.pth"
                        else:
                            checkpoint_path = f"checkpoints/{model_type}/{dataset_type}/fold{fold}_best_model.pth"

                        model = load_model(model_type, checkpoint_path)
                        model.eval()

                        predictions = predict_trajectory(model, initial_sequence, time_step)
                        fold_predictions.append(predictions)

                    # Average predictions from all folds
                    avg_predictions = np.mean(fold_predictions, axis=0)
                    all_predictions.append(avg_predictions)

                # Inverse transform predictions
                all_predictions = [scaler.inverse_transform(pred) for pred in all_predictions]

                # Plot results
                test_spaceship_trajectory = scaler.inverse_transform(spaceship_data.values)
                plot_full_trajectory_with_predictions(
                    test_spaceship_trajectory,
                    all_predictions,
                    start_indices,
                    f"{model_type} on {dataset_type} - {time_step} steps prediction",
                    model_type,
                    dataset_type,
                    is_mini=args.mini
                )
                plot_full_velocity_with_predictions(
                    test_spaceship_trajectory,
                    np.concatenate(all_predictions),
                    1,  # Assuming dt=1, adjust if necessary
                    f"{model_type} on {dataset_type} - {time_step} steps velocity prediction",
                    model_type,
                    dataset_type,
                    is_mini=args.mini
                )

if __name__ == "__main__":
    main()
