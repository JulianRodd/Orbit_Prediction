import argparse
import os
import traceback
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import pandas as pd
from utils import plot_full_trajectory_with_predictions, plot_full_velocity_with_predictions
from generics import DEVICE, DT, HEAVY_BODY1_POSITION, HEAVY_BODY2_POSITION, HIDDEN_SIZE, MINI_CHECKPOINT_LOCATION, MINI_HIDDEN_SIZE, MINI_NUM_LAYERS, N_STEPS, INPUT_SIZE, NUM_FOLDS, NUM_LAYERS, OUTPUT_SIZE
from regression import PINN, OrbitLSTM, SimpleRegression, load_data

def parse_arguments():
    parser = argparse.ArgumentParser(description="Perform inference on test orbits")
    parser.add_argument("--model_types", nargs="+", default=["PINN", "SimpleRegression", "LSTM"],
                        help="Model types to use for inference")
    parser.add_argument("--datasets", nargs="+",
                        default=["two_body", "two_body_force_increased_acceleration", "three_body"],
                        help="Datasets to use for inference")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                        help="Folds to use for ensemble prediction")
    parser.add_argument("--prediction_steps", nargs="+", type=int, default=[10, 100, 500],
                        help="Number of steps to predict into the future")
    parser.add_argument("--mini", action="store_true", help="Use mini models for inference")
    return parser.parse_args()
def load_model(model_type, checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)

    if model_type == "PINN":
        input_size = state_dict['net.0.weight'].shape[1]
        hidden_size = state_dict['net.0.weight'].shape[0]
        output_size = state_dict['net.6.weight'].shape[0]
        model = PINN(input_size, hidden_size, output_size)
    elif model_type == "SimpleRegression":
        input_size = state_dict['fc1.weight'].shape[1]
        hidden_size = state_dict['fc1.weight'].shape[0]
        output_size = state_dict['fc3.weight'].shape[0]
        model = SimpleRegression(input_size, hidden_size, output_size)
    elif model_type == "LSTM":
        input_size = state_dict['lstm.weight_ih_l0'].shape[1] // N_STEPS
        hidden_size = state_dict['lstm.weight_hh_l0'].shape[0] // 4
        num_layers = sum(1 for k in state_dict.keys() if k.startswith('lstm.weight_ih_l'))
        output_size = state_dict['fc2.weight'].shape[0]
        model = OrbitLSTM(input_size, hidden_size, num_layers, output_size)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(state_dict)
    return model.to(DEVICE)
def predict_trajectory(model, initial_sequence, scaler, steps):
    model.eval()
    with torch.no_grad():
        predictions = []
        current_sequence = initial_sequence.clone()
        for _ in range(steps):
            if isinstance(model, OrbitLSTM):
                input_sequence = current_sequence.unsqueeze(0)  # Add batch dimension
            else:
                input_sequence = current_sequence.view(1, -1)

            output = model(input_sequence)
            predictions.append(output.squeeze().cpu().numpy())

            # Update current_sequence for next iteration
            current_sequence = torch.cat([current_sequence[1:], output.view(1, -1)], dim=0)

    predictions = np.array(predictions)
    return scaler.inverse_transform(predictions)

def plot_trajectory(
    actual_positions,
    predicted_positions,
    input_positions,
    title,
    dataset_type,
    model_name,
    dataset_name,
    time_steps,
    is_mini=False,
):
    plt.figure(figsize=(12, 10))

    # Plot actual trajectory
    plt.plot(
        actual_positions[:, 0],
        actual_positions[:, 1],
        label="Actual",
        color="blue",
        linewidth=1,
    )

    # Plot predicted trajectory
    plt.plot(
        predicted_positions[:, 0],
        predicted_positions[:, 1],
        label="Predicted",
        color="red",
        linestyle="--",
        linewidth=1,
    )

    # Plot input time steps
    plt.plot(
        input_positions[:, 0],
        input_positions[:, 1],
        label="Input",
        color="green",
        linestyle=":",
        linewidth=2,
        marker='o'
    )

    # Plot start and end points
    plt.scatter(
        actual_positions[0, 0],
        actual_positions[0, 1],
        c="green",
        s=100,
        label="Start",
        zorder=5,
    )
    plt.scatter(
        actual_positions[-1, 0],
        actual_positions[-1, 1],
        c="red",
        s=100,
        label="End",
        zorder=5,
    )

    # Plot heavy bodies
    if "two_body" in dataset_name:
        plt.scatter(0, 0, c="yellow", s=100, label="Heavy Body")
    elif "three_body" in dataset_name:
        plt.scatter(
            HEAVY_BODY1_POSITION[0],
            HEAVY_BODY1_POSITION[1],
            c="yellow",
            s=100,
            label="Heavy Body 1",
        )
        plt.scatter(
            HEAVY_BODY2_POSITION[0],
            HEAVY_BODY2_POSITION[1],
            c="orange",
            s=100,
            label="Heavy Body 2",
        )

    plt.title(f"{title} - {dataset_type.capitalize()}")
    plt.xlabel("X position (m)")
    plt.ylabel("Y position (m)")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")

    # Create folder structure
    folder = os.path.join(
        "plots",
        "mini" if is_mini else "full",
        model_name,
        dataset_name,
        str(time_steps),
    )
    os.makedirs(folder, exist_ok=True)

    # Save plot
    plt.savefig(
        os.path.join(folder, f"{dataset_type}_trajectory.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

def plot_velocity(
    actual_velocities,
    predicted_velocities,
    dt,
    title,
    dataset_type,
    model_name,
    dataset_name,
    time_steps,
    is_mini=False
):
    plt.figure(figsize=(12, 8))

    time = np.arange(len(actual_velocities)) * dt

    # Plot actual velocity
    actual_velocity_magnitude = np.linalg.norm(actual_velocities[:, 2:], axis=1)
    plt.plot(
        time,
        actual_velocity_magnitude,
        label="Actual",
        color="blue",
        linewidth=1,
    )

    # Plot predicted velocity
    predicted_velocity_magnitude = np.linalg.norm(predicted_velocities[:, 2:], axis=1)
    plt.plot(
        time[:len(predicted_velocities)],
        predicted_velocity_magnitude,
        label="Predicted",
        color="red",
        linestyle="--",
        linewidth=1,
    )

    plt.title(f"{title} - {dataset_type.capitalize()}")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity magnitude (m/s)")
    plt.legend()
    plt.grid(True)

    # Create folder structure
    folder = os.path.join(
        "plots",
        "mini" if is_mini else "full",
        model_name,
        dataset_name,
        str(time_steps)
    )
    os.makedirs(folder, exist_ok=True)

    # Save plot
    plt.savefig(
        os.path.join(folder, f"{dataset_type}_velocity.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

def plot_error_distribution(errors, title, save_path):
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, edgecolor="black")
    plt.title(title)
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_error_over_time(errors, title, save_path):
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(len(errors)) * DT, errors)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Error")
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def perform_inference(model_type, dataset_type, args):
    print(f"Performing inference for {model_type} on {dataset_type}")

    # Load test dataset
    test_dataset, test_scaler, test_df, _ = load_data("test", dataset_type, None, N_STEPS)

    # Select a single test spaceship
    spaceship_ids = test_df["spaceship_id"].unique()
    selected_spaceship_id = spaceship_ids[0]

    # Get the full trajectory for the selected spaceship
    spaceship_data = test_df[test_df["spaceship_id"] == selected_spaceship_id][["x", "y", "Vx", "Vy"]]
    spaceship_data_scaled = pd.DataFrame(
        test_scaler.transform(spaceship_data), columns=spaceship_data.columns
    )

    # Select a subset of starting points
    trajectory_length = len(spaceship_data_scaled)
    num_predictions = 4  # Number of predictions to make
    start_indices = [int(trajectory_length * i / num_predictions) for i in range(num_predictions)]

    # Initialize model
    if model_type == "PINN":
        model = PINN(INPUT_SIZE, MINI_HIDDEN_SIZE if args.mini else HIDDEN_SIZE, OUTPUT_SIZE).to(DEVICE)
    elif model_type == "SimpleRegression":
        model = SimpleRegression(INPUT_SIZE, MINI_HIDDEN_SIZE if args.mini else HIDDEN_SIZE, OUTPUT_SIZE).to(DEVICE)
    elif model_type == "LSTM":
        model = OrbitLSTM(INPUT_SIZE // N_STEPS, MINI_HIDDEN_SIZE if args.mini else HIDDEN_SIZE,
                          MINI_NUM_LAYERS if args.mini else NUM_LAYERS, OUTPUT_SIZE).to(DEVICE)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load the best model from each fold and ensemble their predictions
    all_predictions = []
    for fold in tqdm(range(NUM_FOLDS), desc="Processing folds"):
        if args.mini:
            checkpoint_path = f"{MINI_CHECKPOINT_LOCATION}/{model_type}/{dataset_type}/fold{fold}_mini_model.pth"
        else:
            checkpoint_path = f"checkpoints/{model_type}/{dataset_type}/fold{fold}_best_model.pth"

        try:
            model = load_model(model_type, checkpoint_path)
            model.eval()

            fold_predictions = []
            with torch.no_grad():
                for start_idx in start_indices:
                    input_sequence = torch.FloatTensor(spaceship_data_scaled.iloc[start_idx:start_idx+N_STEPS].values).to(DEVICE)
                    pred = predict_trajectory(model, input_sequence, test_scaler, args.prediction_steps[0])
                    fold_predictions.append(pred)

            all_predictions.append(fold_predictions)
        except Exception as e:
            print(f"Error in fold {fold}: {str(e)}")
            continue

    if not all_predictions:
        print(f"No valid predictions for {model_type} on {dataset_type}. Skipping.")
        return

    # Average the predictions from all folds
    avg_predictions = np.mean(all_predictions, axis=0)

    # Plot and calculate metrics for each prediction
    for i, (start_idx, avg_prediction) in enumerate(zip(start_indices, avg_predictions)):
        actual_trajectory = spaceship_data.iloc[start_idx+N_STEPS:start_idx+N_STEPS+len(avg_prediction)]

        # Calculate metrics
        mse = np.mean((avg_prediction - actual_trajectory.values) ** 2)
        mae = np.mean(np.abs(avg_prediction - actual_trajectory.values))
        print(f"Prediction {i+1} - Test MSE: {mse:.4f}, Test MAE: {mae:.4f}")

        # Plot trajectory
        plot_trajectory(
            actual_trajectory.values,
            avg_prediction,
            spaceship_data_scaled.iloc[start_idx:start_idx+N_STEPS].values,
            f"{model_type} on {dataset_type} - Trajectory {i+1}",
            "test",
            model_type,
            dataset_type,
            args.prediction_steps[0],
            is_mini=args.mini
        )

        # Plot velocity
        plot_velocity(
            actual_trajectory.values,
            avg_prediction,
            DT,
            f"{model_type} on {dataset_type} - Velocity {i+1}",
            "test",
            model_type,
            dataset_type,
            args.prediction_steps[0],
            is_mini=args.mini
        )

    # Plot full trajectory with all predictions
    plot_full_trajectory_with_predictions(
        spaceship_data.values,
        avg_predictions,
        start_indices,
        f"{model_type} Full Trajectory with Predictions",
        model_type,
        dataset_type,
        is_mini=args.mini
    )

def main():
    args = parse_arguments()

    for dataset_type in args.datasets:
        print(f"\nProcessing dataset: {dataset_type}")

        for model_type in args.model_types:
            print(f"  Model type: {model_type}")

            try:
                perform_inference(model_type, dataset_type, args)
            except Exception as e:
                print(f"Error in inference for {model_type} on {dataset_type}: {str(e)}")
                traceback.print_exc()

    print("Inference completed.")

if __name__ == "__main__":
    main()
