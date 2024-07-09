import argparse
import os
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from generics import (
    BATCH_SIZE,
    DEVICE,
    DT,
    EPOCHS,
    INPUT_SIZE,
    LEARNING_RATE,
    M1,
    MINI_CHECKPOINT_LOCATION,
    MINI_EPOCHS,
    MINI_HIDDEN_SIZE,
    MINI_NUM_LAYERS,
    N_STEPS,
    NUM_FOLDS,
    OUTPUT_SIZE,
)
from pinn_loss import pinn_loss
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from utils import plot_trajectory, plot_velocity

import wandb

# Constants and device setup
M = M1  # Mass of the central body (e.g., Earth)
DEVICE = torch.device("mps")


# Define more complex models
class SimpleRegression(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRegression, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.3)  # Increased dropout rate
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.dropout(torch.relu(self.batch_norm1(self.fc1(x))))
        x = self.dropout(torch.relu(self.batch_norm2(self.fc2(x))))
        return self.fc3(x)


class OrbitLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(OrbitLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=0.2
        )
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = torch.relu(self.fc1(lstm_out[:, -1, :]))
        return self.fc2(x)


# Early Stopping class
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class PINN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=4):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.net(x)


class OrbitDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = torch.FloatTensor(sequences).to(DEVICE)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        x = sequence[:-1, :4].reshape(
            -1
        )  # Flatten all but the last step, only x, y, vx, vy
        y = sequence[-1, :4]  # Only return x, y, vx, vy for target
        return x, y


def load_data(dataset_type, problem_type, fold, seq_length):
    print(f"Debug: load_data called with seq_length = {seq_length}")

    scaler = MinMaxScaler()

    if dataset_type == "test":
        df = pd.read_csv(
            f"datasets/{problem_type}/{dataset_type}/{problem_type}_{dataset_type}.csv"
        )
    else:
        df = pd.read_csv(
            f"datasets/{problem_type}/{dataset_type}/{problem_type}_{dataset_type}_fold{fold}.csv"
        )

    # Only scale x, y, vx, vy columns
    columns_to_scale = ["x", "y", "Vx", "Vy"]
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    sequences = []
    for spaceship in df["spaceship_id"].unique():
        spaceship_data = df[df["spaceship_id"] == spaceship][columns_to_scale].values
        print(f"Debug: spaceship_data shape = {spaceship_data.shape}")
        for i in range(len(spaceship_data) - seq_length):
            sequences.append(spaceship_data[i : i + seq_length + 1])

    sequences = np.array(sequences)
    print(f"Debug: sequences shape = {sequences.shape}")
    return OrbitDataset(sequences), scaler, df


def train_mini_model(model_type, train_loader, val_loader, dataset_type, fold):
    if model_type == "PINN":
        model = PINN(INPUT_SIZE, MINI_HIDDEN_SIZE, OUTPUT_SIZE).to(DEVICE)
    elif model_type == "SimpleRegression":
        model = SimpleRegression(INPUT_SIZE, MINI_HIDDEN_SIZE, OUTPUT_SIZE).to(DEVICE)
    elif model_type == "LSTM":
        model = OrbitLSTM(4, MINI_HIDDEN_SIZE, MINI_NUM_LAYERS, OUTPUT_SIZE).to(DEVICE)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(MINI_EPOCHS):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            optimizer.zero_grad()
            if model_type == "PINN":
                loss, _, _, _, _ = pinn_loss(model, batch_x, batch_y)
            else:
                outputs = model(batch_x)
                loss = nn.MSELoss()(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation loop (unchanged)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                if model_type == "PINN":
                    loss, _, _, _, _ = pinn_loss(model, batch_x, batch_y)
                else:
                    outputs = model(batch_x)
                    loss = nn.MSELoss()(outputs, batch_y)
                val_loss += loss.item()

        print(
            f"Epoch [{epoch+1}/{MINI_EPOCHS}], Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}"
        )

    checkpoint_path = f"{MINI_CHECKPOINT_LOCATION}/{model_type}/{dataset_type}/fold{fold}_mini_model.pth"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)

    return model


def train_model(
    model, train_loader, val_loader, model_type, dataset_type, fold, use_wandb
):
    early_stopping = EarlyStopping(patience=5)
    best_val_loss = float("inf")
    num_epochs = EPOCHS
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=2, factor=0.5
    )

    for epoch in range(EPOCHS):
        model.train()
        train_loss = train_mse_loss = train_energy_loss = train_momentum_loss = (
            train_laplace_loss
        ) = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            try:
                if model_type == "PINN":
                    loss, mse, energy, momentum, laplace = pinn_loss(
                        model, batch_x, batch_y
                    )
                    train_mse_loss += mse.item()
                    train_energy_loss += energy.item()
                    train_momentum_loss += momentum.item()
                    train_laplace_loss += laplace.item()
                else:
                    outputs = model(batch_x)
                    loss = nn.MSELoss()(outputs, batch_y)
                    train_mse_loss += loss.item()

                if torch.isnan(loss):
                    print(f"NaN loss detected. Skipping this batch.")
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            except Exception as e:
                print(f"Error in training loop: {str(e)}")
                print(f"batch_x shape: {batch_x.shape}, batch_y shape: {batch_y.shape}")
                if model_type != "PINN":
                    print(f"model output shape: {outputs.shape}")
                raise

        model.eval()
        val_loss = val_mse_loss = val_energy_loss = val_momentum_loss = (
            val_laplace_loss
        ) = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                if model_type == "PINN":
                    loss, mse, energy, momentum, laplace = pinn_loss(
                        model, batch_x, batch_y
                    )
                    val_mse_loss += mse.item()
                    val_energy_loss += energy.item()
                    val_momentum_loss += momentum.item()
                    val_laplace_loss += laplace.item()
                else:
                    outputs = model(batch_x)
                    loss = nn.MSELoss()(outputs, batch_y)
                    val_mse_loss += loss.item()
                val_loss += loss.item()
                val_loss += loss.item()

        train_loss /= len(train_loader)
        train_mse_loss /= len(train_loader)
        train_energy_loss /= len(train_loader)
        train_momentum_loss /= len(train_loader)
        train_laplace_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_mse_loss /= len(val_loader)
        val_energy_loss /= len(val_loader)
        val_momentum_loss /= len(val_loader)
        val_laplace_loss /= len(val_loader)

        scheduler.step(val_loss)

        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    f"train_loss_fold{fold}": train_loss,
                    f"train_mse_loss_fold{fold}": train_mse_loss,
                    f"train_energy_loss_fold{fold}": train_energy_loss,
                    f"train_momentum_loss_fold{fold}": train_momentum_loss,
                    f"train_laplace_loss_fold{fold}": train_laplace_loss,
                    f"val_loss_fold{fold}": val_loss,
                    f"val_mse_loss_fold{fold}": val_mse_loss,
                    f"val_energy_loss_fold{fold}": val_energy_loss,
                    f"val_momentum_loss_fold{fold}": val_momentum_loss,
                    f"val_laplace_loss_fold{fold}": val_laplace_loss,
                    f"learning_rate_fold{fold}": optimizer.param_groups[0]["lr"],
                }
            )

        print(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = (
                f"checkpoints/{model_type}/{dataset_type}/fold{fold}_best_model.pth"
            )
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)

        if early_stopping(val_loss):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"{'='*50}\n")
    return model


def evaluate_model(model, test_loader, scaler):
    model.eval()
    all_predictions = []
    all_actual = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            all_predictions.append(outputs.cpu().numpy())
            all_actual.append(batch_y.cpu().numpy())

    all_predictions = np.concatenate(all_predictions)
    all_actual = np.concatenate(all_actual)

    # Ensure that all_predictions and all_actual have the same shape
    assert (
        all_predictions.shape == all_actual.shape
    ), f"Shape mismatch: predictions {all_predictions.shape}, actual {all_actual.shape}"

    # Inverse transform both predictions and actual values
    all_predictions_original = scaler.inverse_transform(all_predictions)
    all_actual_original = scaler.inverse_transform(all_actual)

    mse = np.mean((all_predictions_original - all_actual_original) ** 2)
    mae = np.mean(np.abs(all_predictions_original - all_actual_original))

    return mse, mae


def predict_future(model, initial_sequence, scaler, steps, output_size):
    model.eval()
    current_input = initial_sequence.clone().to(DEVICE)
    predictions = []

    # print(f"Initial sequence shape: {initial_sequence.shape}")

    with torch.no_grad():
        for i in range(steps):
            model_input = current_input.view(1, -1)
            # print(f"Step {i+1}, Model input shape: {model_input.shape}")

            output = model(model_input)
            # print(f"Step {i+1}, Output shape: {output.shape}")

            predictions.append(output.detach().cpu().numpy())
            current_input = torch.cat((current_input[4:], output[0]), 0).to(DEVICE)
            # print(f"Step {i+1}, Updated current input shape: {current_input.shape}")

    predictions = np.array(predictions).reshape(-1, output_size)
    # print(f"Final predictions shape: {predictions.shape}")
    return scaler.inverse_transform(predictions)


def plot_predictions(actual, predicted, model_type, dataset_type, steps, fold):
    plt.figure(figsize=(12, 9))
    plt.plot(
        actual[:, 0], actual[:, 1], label="Actual", color="blue", linewidth=2, alpha=0.7
    )
    plt.plot(
        predicted[:, 0],
        predicted[:, 1],
        label="Predicted",
        color="red",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
    )
    plt.title(
        f"{model_type} on {dataset_type} - {steps} steps prediction (Fold {fold})",
        fontsize=16,
    )
    plt.xlabel("X position", fontsize=12)
    plt.ylabel("Y position", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.7)

    folder = f"plots/{model_type}/{dataset_type}"
    os.makedirs(folder, exist_ok=True)
    plt.savefig(
        f"{folder}/prediction_{steps}_steps_fold{fold}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # wandb.log({f"{model_type}_{dataset_type}_{steps}_steps_fold{fold}": wandb.Image(f"{folder}/prediction_{steps}_steps_fold{fold}.png")})


def plot_velocities(actual, predicted, model_type, dataset_type, steps, fold):
    plt.figure(figsize=(12, 9))
    actual_velocity = np.linalg.norm(actual[:, 2:], axis=1)
    predicted_velocity = np.linalg.norm(predicted[:, 2:], axis=1)

    plt.plot(actual_velocity, label="Actual", color="blue", linewidth=2, alpha=0.7)
    plt.plot(
        predicted_velocity,
        label="Predicted",
        color="red",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
    )
    plt.title(
        f"{model_type} on {dataset_type} - {steps} steps velocity prediction (Fold {fold})",
        fontsize=16,
    )
    plt.xlabel("Time step", fontsize=12)
    plt.ylabel("Velocity magnitude", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.7)

    folder = f"plots/{model_type}/{dataset_type}"
    os.makedirs(folder, exist_ok=True)
    plt.savefig(
        f"{folder}/velocity_prediction_{steps}_steps_fold{fold}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def get_dataset_sequences(dataset, steps):
    if isinstance(dataset, torch.utils.data.Subset):
        # If it's a subset, we need to access the original dataset
        original_dataset = dataset.dataset
        indices = dataset.indices
        return original_dataset.sequences[indices][:steps, -1, :].cpu().numpy()
    else:
        # If it's the original dataset, we can access sequences directly
        return dataset.sequences[:steps, -1, :].cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Train orbital prediction models")
    parser.add_argument(
        "--model_types", nargs="+", default=["PINN"], help="Model types to train"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["two_body", "two_body_force_increased_acceleration", "three_body"],
        help="Datasets to use",
    )
    parser.add_argument(
        "--mini", action="store_true", help="Train mini networks for quick testing"
    )
    parser.add_argument(
        "--use_wandb", action="store_true", help="Use Weights & Biases for logging"
    )
    args = parser.parse_args()

    print(f"INPUT_SIZE: {INPUT_SIZE}")

    for model_type in args.model_types:
        for dataset_type in args.datasets:
            print(f"\n{'#'*70}")
            print(f"# Model: {model_type:<20} Dataset: {dataset_type:<35} #")
            print(f"{'#'*70}")

            if args.use_wandb:
                wandb.init(
                    project="orbital-prediction",
                    entity="radboud-mlip-10",
                    name=f"{model_type}_{dataset_type}",
                    config={
                        "model_type": model_type,
                        "dataset_type": dataset_type,
                        "seq_length": N_STEPS,
                        "batch_size": BATCH_SIZE,
                        "learning_rate": LEARNING_RATE,
                        "num_epochs": MINI_EPOCHS if args.mini else EPOCHS,
                        "num_folds": NUM_FOLDS,
                    },
                )

            try:
                fold_results = []
                for fold in range(NUM_FOLDS):
                    train_dataset, train_scaler, _ = load_data(
                        "train", dataset_type, fold, N_STEPS
                    )
                    val_dataset, _, _ = load_data("val", dataset_type, fold, N_STEPS)

                    if args.mini:
                        # Reduce dataset size for mini training
                        mini_size = min(
                            1000, len(train_dataset)
                        )  # Use at most 1000 samples or the full dataset if smaller
                        indices = torch.randperm(len(train_dataset))[:mini_size]
                        train_dataset = torch.utils.data.Subset(train_dataset, indices)

                        mini_val_size = min(100, len(val_dataset))
                        indices = torch.randperm(len(val_dataset))[:mini_val_size]
                        val_dataset = torch.utils.data.Subset(val_dataset, indices)

                    train_loader = DataLoader(
                        train_dataset, batch_size=BATCH_SIZE, shuffle=True
                    )
                    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

                    print(f"\nFold {fold+1}/{NUM_FOLDS}")
                    print(f"Train dataset size: {len(train_dataset)}")
                    print(f"Validation dataset size: {len(val_dataset)}")

                    if args.mini:
                        model = train_mini_model(
                            model_type, train_loader, val_loader, dataset_type, fold
                        )
                    else:
                        model = train_model(
                            model,
                            train_loader,
                            val_loader,
                            model_type,
                            dataset_type,
                            fold,
                            args.use_wandb,
                        )

                    val_mse, val_mae = evaluate_model(model, val_loader, train_scaler)
                    fold_results.append((val_mse, val_mae))

                    initial_sequence = next(iter(val_loader))[0][0].to(DEVICE)
                    print(f"Debug: Initial sequence shape: {initial_sequence.shape}")
                    for steps in [10, 100, 500]:
                        predictions = predict_future(
                            model, initial_sequence, train_scaler, steps, OUTPUT_SIZE
                        )
                        actual = train_scaler.inverse_transform(
                            get_dataset_sequences(val_dataset, steps)
                        )
                        print(f"Debug: Predictions shape: {predictions.shape}")
                        print(f"Debug: Actual shape: {actual.shape}")
                        dataset_name = dataset_type
                        # Use only as many prediction steps as we have actual data
                        min_steps = min(steps, len(actual))
                        plot_trajectory(
                            actual[:min_steps],
                            predictions[:min_steps],
                            f"{model_type} Trajectory",
                            dataset_type,
                            model_type,
                            dataset_name,
                            min_steps,
                        )
                        plot_velocity(
                            actual[:min_steps],
                            predictions[:min_steps],
                            DT,
                            f"{model_type} Velocity",
                            dataset_type,
                            model_type,
                            dataset_name,
                            min_steps,
                        )
                avg_val_mse = np.mean([res[0] for res in fold_results])
                avg_val_mae = np.mean([res[1] for res in fold_results])
                if args.use_wandb:
                    wandb.log(
                        {
                            f"{model_type}_{dataset_type}_avg_val_mse": avg_val_mse,
                            f"{model_type}_{dataset_type}_avg_val_mae": avg_val_mae,
                        }
                    )

                # Load and evaluate on test set
                print(f"Debug: Loading test dataset with N_STEPS = {N_STEPS}")
                test_dataset, test_scaler, _ = load_data(
                    "test", dataset_type, None, N_STEPS
                )
                test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

                # Load the best model from each fold and ensemble their predictions
                ensemble_predictions = []
                for fold in range(NUM_FOLDS):
                    if args.mini:
                        checkpoint_path = f"{MINI_CHECKPOINT_LOCATION}/{model_type}/{dataset_type}/fold{fold}_mini_model.pth"
                    else:
                        checkpoint_path = f"checkpoints/{model_type}/{dataset_type}/fold{fold}_best_model.pth"
                    model.load_state_dict(
                        torch.load(checkpoint_path, map_location=DEVICE)
                    )
                    model.to(DEVICE)
                    fold_predictions = []
                    with torch.no_grad():
                        for batch_x, _ in test_loader:
                            batch_x = batch_x.to(DEVICE)
                            outputs = model(batch_x)
                            fold_predictions.append(outputs.cpu().numpy())
                    fold_predictions = np.concatenate(fold_predictions)
                    ensemble_predictions.append(fold_predictions)

                # Average the predictions from all folds
                ensemble_predictions = np.mean(ensemble_predictions, axis=0)
                print(
                    f"Debug: ensemble_predictions shape before inverse_transform: {ensemble_predictions.shape}"
                )
                ensemble_predictions = test_scaler.inverse_transform(
                    ensemble_predictions
                )
                print(
                    f"Debug: ensemble_predictions shape after inverse_transform: {ensemble_predictions.shape}"
                )

                # Evaluate ensemble predictions
                test_actual = np.concatenate(
                    [seq[-1] for seq in test_dataset.sequences.cpu().numpy()]
                )
                print(f"Debug: test_actual shape before reshape: {test_actual.shape}")
                test_actual = test_actual.reshape(-1, OUTPUT_SIZE)
                print(f"Debug: test_actual shape after reshape: {test_actual.shape}")
                test_actual = test_scaler.inverse_transform(test_actual)
                print(
                    f"Debug: test_actual shape after inverse_transform: {test_actual.shape}"
                )

                test_mse = np.mean((ensemble_predictions - test_actual) ** 2)
                test_mae = np.mean(np.abs(ensemble_predictions - test_actual))

                if args.use_wandb:
                    wandb.log(
                        {
                            f"{model_type}_{dataset_type}_test_mse": test_mse,
                            f"{model_type}_{dataset_type}_test_mae": test_mae,
                        }
                    )

                # Plot test predictions
                for steps in [10, 100, 500]:
                    plot_predictions(
                        test_actual[:steps],
                        ensemble_predictions[:steps],
                        model_type,
                        dataset_type,
                        steps,
                        "test",
                    )
                    plot_velocities(
                        test_actual[:steps],
                        ensemble_predictions[:steps],
                        model_type,
                        dataset_type,
                        steps,
                        "test",
                    )

            except Exception as e:
                print(f"\nError occurred for {model_type} on {dataset_type}: {str(e)}")
                print("Traceback:")
                print(traceback.format_exc())
                if args.use_wandb:
                    wandb.log({"error": str(e), "traceback": traceback.format_exc()})
            if args.use_wandb:
                wandb.finish()
            print(f"\n{'#'*70}\n")


if __name__ == "__main__":
    main()
