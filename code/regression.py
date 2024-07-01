import os
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from generics import M1, NUM_SIMULATIONS, G
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

import wandb

# Constants for physics calculations
M = M1  # Mass of the central body (e.g., Earth)


# Define models
class SimpleRegression(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRegression, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class OrbitLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(OrbitLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


class PINN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        return self.net(x)


class OrbitDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        return torch.FloatTensor(sequence[:-1]), torch.FloatTensor(sequence[-1])


# Early Stopping class
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
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


def load_data(dataset_type, problem_type, spaceship_id, seq_length):
    scaler = MinMaxScaler()

    df = pd.read_csv(
        f"datasets/{problem_type}/{dataset_type}/{spaceship_id}/{problem_type}_{dataset_type}.csv"
    )
    df.iloc[:, 1:] = scaler.fit_transform(df.iloc[:, 1:])  # Exclude 'timestep'
    sequences = []
    for i in range(len(df) - seq_length):
        sequences.append(df.iloc[i : i + seq_length + 1, 1:].values)

    return OrbitDataset(np.array(sequences)), scaler, df


def physics_loss(y_pred, y_true):
    # Unpack predicted and true values
    x_pred, y_pred, vx_pred, vy_pred = (
        y_pred[:, 0],
        y_pred[:, 1],
        y_pred[:, 2],
        y_pred[:, 3],
    )
    x_true, y_true, vx_true, vy_true = (
        y_true[:, 0],
        y_true[:, 1],
        y_true[:, 2],
        y_true[:, 3],
    )

    # Calculate radii
    r_pred = torch.sqrt(x_pred**2 + y_pred**2)
    r_true = torch.sqrt(x_true**2 + y_true**2)

    # Energy conservation
    KE_pred = 0.5 * (vx_pred**2 + vy_pred**2)
    PE_pred = -G * M / r_pred
    E_pred = KE_pred + PE_pred

    KE_true = 0.5 * (vx_true**2 + vy_true**2)
    PE_true = -G * M / r_true
    E_true = KE_true + PE_true

    energy_loss = torch.mean((E_pred - E_true) ** 2)

    # Velocity-position relationship
    vel_pos_loss = torch.mean(
        (vx_pred[:-1] - (x_pred[1:] - x_pred[:-1])) ** 2
        + (vy_pred[:-1] - (y_pred[1:] - y_pred[:-1])) ** 2
    )

    # Combine losses
    total_loss = energy_loss + vel_pos_loss

    return total_loss, energy_loss, vel_pos_loss


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    device,
    model_type,
    dataset_type,
    spaceship_id,
):
    early_stopping = EarlyStopping(patience=5)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_mse_loss = 0
        train_physics_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            mse_loss = criterion(outputs, batch_y)

            if model_type == "PINN":
                phys_loss, energy_loss, vel_pos_loss = physics_loss(outputs, batch_y)
                loss = mse_loss + phys_loss
                train_physics_loss += phys_loss.item()
            else:
                loss = mse_loss

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_mse_loss += mse_loss.item()

        model.eval()
        val_loss = 0
        val_mse_loss = 0
        val_physics_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                mse_loss = criterion(outputs, batch_y)

                if model_type == "PINN":
                    phys_loss, energy_loss, vel_pos_loss = physics_loss(
                        outputs, batch_y
                    )
                    loss = mse_loss + phys_loss
                    val_physics_loss += phys_loss.item()
                else:
                    loss = mse_loss

                val_loss += loss.item()
                val_mse_loss += mse_loss.item()

        train_loss /= len(train_loader)
        train_mse_loss /= len(train_loader)
        train_physics_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_mse_loss /= len(val_loader)
        val_physics_loss /= len(val_loader)

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_mse_loss": train_mse_loss,
                "train_physics_loss": train_physics_loss,
                "val_loss": val_loss,
                "val_mse_loss": val_mse_loss,
                "val_physics_loss": val_physics_loss,
            }
        )

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        early_stopping(val_loss)
        if (epoch + 1) % 10 == 0:
            # Save model state every 10 epochs
            torch.save(
                model.state_dict(),
                f"model_{model_type}_{dataset_type}_{spaceship_id}.pth",
            )
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    return model


def predict_future(model, initial_sequence, scaler, steps, device, output_size):
    model.eval()
    current_input = initial_sequence.clone()
    predictions = []

    with torch.no_grad():
        for _ in range(steps):
            output = model(current_input.unsqueeze(0).to(device))
            predictions.append(output.cpu().numpy())
            current_input = torch.cat((current_input[1:], output.cpu()), 0)

    predictions = np.array(predictions).reshape(-1, output_size)

    # Check for NaN or Inf values
    if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
        print("Warning: NaN or Inf values found in predictions. Clipping values.")
        predictions = np.nan_to_num(
            predictions,
            nan=0.0,
            posinf=np.finfo(np.float32).max,
            neginf=np.finfo(np.float32).min,
        )

    return scaler.inverse_transform(predictions)


def plot_predictions(actual, predicted, model_type, dataset_type, steps, spaceship_id):
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
        f"{model_type} on {dataset_type} - {steps} steps prediction\nSpaceship {spaceship_id}",
        fontsize=16,
    )
    plt.xlabel("X position", fontsize=12)
    plt.ylabel("Y position", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.7)

    folder = f"plots/{model_type}/{dataset_type}/{spaceship_id}"
    os.makedirs(folder, exist_ok=True)
    plt.savefig(f"{folder}/prediction_{steps}_steps.png", dpi=300, bbox_inches="tight")
    plt.close()


# Add this new function
def plot_velocities(actual, predicted, model_type, dataset_type, steps, spaceship_id):
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
        f"{model_type} on {dataset_type} - {steps} steps velocity prediction\nSpaceship {spaceship_id}",
        fontsize=16,
    )
    plt.xlabel("Time step", fontsize=12)
    plt.ylabel("Velocity magnitude", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.7)

    folder = f"plots/{model_type}/{dataset_type}/{spaceship_id}"
    os.makedirs(folder, exist_ok=True)
    plt.savefig(
        f"{folder}/velocity_prediction_{steps}_steps.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


# Modify the plot_full_orbit function to include velocity plots
def plot_full_orbit(
    train_data, val_data, test_data, predictions, model_type, dataset_type, spaceship_id
):
    # Position plot
    plt.figure(figsize=(15, 12))
    plt.subplot(2, 1, 1)
    plt.plot(train_data[:, 0], train_data[:, 1], label="Train", color="blue", alpha=0.7)
    plt.plot(
        val_data[:, 0], val_data[:, 1], label="Validation", color="green", alpha=0.7
    )
    plt.plot(test_data[:, 0], test_data[:, 1], label="Test", color="red", alpha=0.7)
    plt.plot(
        predictions[:, 0],
        predictions[:, 1],
        label="Predictions",
        color="purple",
        linestyle="--",
        linewidth=2,
    )
    plt.title(
        f"Full Orbit: {model_type} on {dataset_type}\nSpaceship {spaceship_id}",
        fontsize=18,
    )
    plt.xlabel("X position", fontsize=14)
    plt.ylabel("Y position", fontsize=14)
    plt.legend(fontsize=12, loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.7)

    # Velocity plot
    plt.subplot(2, 1, 2)
    train_velocity = np.linalg.norm(train_data[:, 2:], axis=1)
    val_velocity = np.linalg.norm(val_data[:, 2:], axis=1)
    test_velocity = np.linalg.norm(test_data[:, 2:], axis=1)
    predicted_velocity = np.linalg.norm(predictions[:, 2:], axis=1)

    plt.plot(train_velocity, label="Train", color="blue", alpha=0.7)
    plt.plot(
        range(len(train_velocity), len(train_velocity) + len(val_velocity)),
        val_velocity,
        label="Validation",
        color="green",
        alpha=0.7,
    )
    plt.plot(
        range(
            len(train_velocity) + len(val_velocity),
            len(train_velocity) + len(val_velocity) + len(test_velocity),
        ),
        test_velocity,
        label="Test",
        color="red",
        alpha=0.7,
    )
    plt.plot(
        range(
            len(train_velocity) + len(val_velocity),
            len(train_velocity) + len(val_velocity) + len(predicted_velocity),
        ),
        predicted_velocity,
        label="Predictions",
        color="purple",
        linestyle="--",
        linewidth=2,
    )
    plt.xlabel("Time step", fontsize=14)
    plt.ylabel("Velocity magnitude", fontsize=14)
    plt.legend(fontsize=12, loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.7)

    folder = f"plots/{model_type}/{dataset_type}/{spaceship_id}"
    os.makedirs(folder, exist_ok=True)
    plt.savefig(f"{folder}/full_orbit_with_velocity.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    models = ["SimpleRegression", "LSTM", "PINN"]
    datasets = ["two_body", "two_body_force_increased_acceleration", "three_body"]
    seq_length = 3
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_type in models:
        # Load data
        train_dataset, train_scaler, train_df = load_data(
            "train", datasets[0], 0, seq_length
        )
        val_dataset, _, val_df = load_data("val", datasets[0], 0, seq_length)
        test_dataset, _, test_df = load_data("test", datasets[0], 0, seq_length)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        # Calculate input and output sizes
        input_size = train_dataset[0][0].shape[0] * train_dataset[0][0].shape[1]
        output_size = train_dataset[0][1].shape[0]

        if model_type == "SimpleRegression":
            model = SimpleRegression(input_size, 64, output_size).to(device)
        elif model_type == "LSTM":
            model = OrbitLSTM(train_dataset[0][0].shape[1], 64, 2, output_size).to(
                device
            )
        else:  # PINN
            model = PINN(input_size, 64, output_size).to(device)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        for dataset_type in datasets:
            for spaceship_id in range(NUM_SIMULATIONS):
                wandb.init(
                    project="orbital-prediction",
                    entity="radboud-mlip-10",
                    name=f"{model_type}_{dataset_type}_spaceship_{spaceship_id}",
                    config={
                        "model_type": model_type,
                        "dataset_type": dataset_type,
                        "spaceship_id": spaceship_id,
                        "seq_length": seq_length,
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "num_epochs": num_epochs,
                        "early_stopping_patience": 5,
                    },
                )

                try:
                    # Load data
                    train_dataset, train_scaler, train_df = load_data(
                        "train", dataset_type, spaceship_id, seq_length
                    )
                    val_dataset, _, val_df = load_data(
                        "val", dataset_type, spaceship_id, seq_length
                    )
                    test_dataset, _, test_df = load_data(
                        "test", dataset_type, spaceship_id, seq_length
                    )

                    train_loader = DataLoader(
                        train_dataset, batch_size=batch_size, shuffle=True
                    )
                    val_loader = DataLoader(val_dataset, batch_size=batch_size)
                    test_loader = DataLoader(test_dataset, batch_size=batch_size)

                    # Calculate input and output sizes
                    input_size = (
                        train_dataset[0][0].shape[0] * train_dataset[0][0].shape[1]
                    )
                    output_size = train_dataset[0][1].shape[0]

                    print(f"Input size: {input_size}, Output size: {output_size}")

                    # Train model
                    model = train_model(
                        model,
                        train_loader,
                        val_loader,
                        criterion,
                        optimizer,
                        num_epochs,
                        device,
                        model_type,
                        dataset_type,
                        spaceship_id,
                    )

                    # Save model state after training each spaceship
                    torch.save(
                        model.state_dict(),
                        f"model_{model_type}_{dataset_type}_{spaceship_id}.pth",
                    )

                    # Make predictions for different time steps
                    initial_sequence = next(iter(test_loader))[0][0]
                    all_actual = train_scaler.inverse_transform(
                        np.concatenate([seq[1:] for seq in test_dataset.sequences])
                    )
                    for steps in [10, 100, 500]:
                        predictions = predict_future(
                            model,
                            initial_sequence,
                            train_scaler,
                            steps,
                            device,
                            output_size,
                        )
                        actual = all_actual[:steps]
                        plot_predictions(
                            actual,
                            predictions,
                            model_type,
                            dataset_type,
                            steps,
                            spaceship_id,
                        )
                        plot_velocities(
                            actual,
                            predictions,
                            model_type,
                            dataset_type,
                            steps,
                            spaceship_id,
                        )

                        mse = np.mean((predictions - actual) ** 2)
                        wandb.log({f"MSE_{steps}_steps_spaceship_{spaceship_id}": mse})

                        if model_type == "PINN":
                            phys_loss, energy_loss, vel_pos_loss = physics_loss(
                                torch.tensor(predictions), torch.tensor(actual)
                            )
                            wandb.log(
                                {
                                    f"Physics_loss_{steps}_steps_spaceship_{spaceship_id}": phys_loss.item(),
                                    f"Energy_loss_{steps}_steps_spaceship_{spaceship_id}": energy_loss.item(),
                                    f"Velocity_position_loss_{steps}_steps_spaceship_{spaceship_id}": vel_pos_loss.item(),
                                }
                            )

                    # Plot full orbit
                    train_data = train_scaler.inverse_transform(train_df.iloc[:, 1:])
                    val_data = train_scaler.inverse_transform(val_df.iloc[:, 1:])
                    test_data = train_scaler.inverse_transform(test_df.iloc[:, 1:])
                    full_predictions = predict_future(
                        model,
                        initial_sequence,
                        train_scaler,
                        len(test_data),
                        device,
                        output_size,
                    )
                    plot_full_orbit(
                        train_data,
                        val_data,
                        test_data,
                        full_predictions,
                        model_type,
                        dataset_type,
                        spaceship_id,
                    )

                except Exception as e:
                    print(
                        f"Error occurred for {model_type} on {dataset_type}, spaceship {spaceship_id}: {str(e)}"
                    )
                    print("Traceback:")
                    print(traceback.format_exc())
                    wandb.log({"error": str(e), "traceback": traceback.format_exc()})

                wandb.finish()


if __name__ == "__main__":
    main()
