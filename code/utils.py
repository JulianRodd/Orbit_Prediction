import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from generics import (
    BASE_RELATIVE_UNCERTAINTY,
    DT,
    FIGURE_LOCATION,
    HEAVY_BODY1_POSITION,
    HEAVY_BODY2_POSITION,
    G,
)
from sklearn.preprocessing import StandardScaler


# Calculate initial velocity for a stable circular orbit
def get_initial_velocity(M, m, r):
    return np.sqrt(G * M / r)


# Calculate gravitational force between two bodies
def gravitational_force(M, m, r1, r2):
    r = r2 - r1  # Position vector of the orbiting body relative to the central object
    r_magnitude = np.linalg.norm(r)  # Magnitude (distance) of the position vector r
    r_hat = r / r_magnitude  # Unit vector in the direction of r
    force_magnitude = (
        -G * M * m / r_magnitude**2
    )  # Magnitude of the gravitational force
    force_vector = force_magnitude * r_hat  # Gravitational force vector
    return force_vector


# Slightly change the y position
def slightly_change_y_position(r, amount=0.05):
    r[1] = r[1] + np.random.normal(0, amount)
    return r


# Calculate acceleration
def get_acceleration(F, m, increased_acceleration=0.00):
    base_acceleration = F / m
    return base_acceleration + increased_acceleration * base_acceleration


# Calculate additional sinusoidal force
def additional_force(v, a_amplitude=5e-9):
    force_x = a_amplitude * np.sin(v[0])
    force_y = a_amplitude * np.sin(v[1])
    return np.array([force_x, force_y])


# Apply uncertainty to angle
def apply_uncertainty_to_angle(r, relative_uncertainty=BASE_RELATIVE_UNCERTAINTY):
    angle = np.arctan2(r[1], r[0])

    radius = np.linalg.norm(r)

    angle = angle + np.random.normal(0, 0.01 * np.pi / 180)

    r = [radius * np.cos(angle), radius * np.sin(angle)]

    return r


# Apply uncertainty to radius
def apply_uncertainty_to_radius(r, relative_uncertainty=BASE_RELATIVE_UNCERTAINTY):
    radius = np.linalg.norm(r)
    radius += np.clip(
        np.random.normal(0, relative_uncertainty) * radius,
        -relative_uncertainty,
        relative_uncertainty,
    )
    return r / np.linalg.norm(r) * radius


# Apply uncertainty to velocity
def apply_uncertainty_to_velocity(v, relative_uncertainty=BASE_RELATIVE_UNCERTAINTY):
    return v * (
        1
        + np.clip(
            np.random.normal(0, relative_uncertainty),
            -relative_uncertainty,
            relative_uncertainty,
        )
    )


# Update position and velocity using Euler method
def euler_update(
    r,
    v,
    F,
    m,
    dt=DT,
    relative_uncertainty=BASE_RELATIVE_UNCERTAINTY,
    increased_acceleration=0.00,
):
    a = get_acceleration(F, m, increased_acceleration)
    v_next = v + a * dt
    r_next = r + v_next * dt

    # Apply uncertainties
    r_next = apply_uncertainty_to_angle(r_next, relative_uncertainty)
    r_next = apply_uncertainty_to_radius(r_next, relative_uncertainty)
    v_next = apply_uncertainty_to_velocity(v_next, relative_uncertainty)

    return r_next, v_next


def plot_trajectory_single(
    positions, title="Trajectory", simulation="", folder="", body_type="two_body"
):
    positions = np.array(positions)
    plt.figure(figsize=(10, 10))

    # Plot the trajectory
    plt.plot(positions[:, 0], positions[:, 1], label="Orbiting Body")

    # Plot the starting point
    plt.scatter(
        positions[0, 0], positions[0, 1], color="green", s=100, label="Start", zorder=5
    )

    # Plot the ending point
    plt.scatter(
        positions[-1, 0], positions[-1, 1], color="red", s=100, label="End", zorder=5
    )

    # Plot the heavy body/bodies
    if body_type == "two_body":
        plt.scatter(0, 0, color="black", s=200, label="Heavy Body", zorder=5)
    elif body_type == "three_body":
        plt.scatter(
            HEAVY_BODY1_POSITION[0],
            HEAVY_BODY1_POSITION[1],
            color="black",
            s=200,
            label="Heavy Body 1",
            zorder=5,
        )
        plt.scatter(
            HEAVY_BODY2_POSITION[0],
            HEAVY_BODY2_POSITION[1],
            color="gray",
            s=200,
            label="Heavy Body 2",
            zorder=5,
        )

    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    # Make sure the aspect ratio is equal
    plt.axis("equal")

    # Add some padding to the plot
    plt.margins(0.1)

    plt.savefig(
        f"{FIGURE_LOCATION}{folder}/plot_trajectory_{simulation}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# Plot velocity over time
def plot_velocity_single(
    velocities, dt=DT, title="Velocity over Time", simulation="", folder=""
):
    velocities = np.array(velocities)
    speed = np.linalg.norm(velocities, axis=1)
    time = np.arange(0, len(speed) * dt, dt)
    plt.figure(figsize=(8, 6))
    plt.plot(time, speed)
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.title(title)
    plt.grid()
    plt.savefig(f"{FIGURE_LOCATION}{folder}/plot_velocity_{simulation}.png")


# Prepare dataset for training
# utils.py


def prepare_dataset(positions, velocities, n_steps=3):
    X, y = [], []
    for i in range(len(positions) - n_steps):
        X.append(
            np.hstack(
                [positions[i : i + n_steps], velocities[i : i + n_steps]]
            ).flatten()
        )
        y.append(np.hstack([positions[i + n_steps], velocities[i + n_steps]]))
    return np.array(X), np.array(y)


def normalize_data(X, y):
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    # Reshape X to 2D if it's 3D
    original_shape = X.shape
    if len(original_shape) == 3:
        X = X.reshape(original_shape[0], -1)

    X_normalized = X_scaler.fit_transform(X)
    y_normalized = y_scaler.fit_transform(y)

    # Reshape X back to its original shape if it was 3D
    if len(original_shape) == 3:
        X_normalized = X_normalized.reshape(original_shape)

    return X_normalized, y_normalized, X_scaler, y_scaler


# Plot training and validation loss curves
def plot_loss_curves(
    train_losses, val_losses, title="Training and Validation Loss Curves", model=""
):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{FIGURE_LOCATION}models/model_{model}.png")


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_trajectory_pred(actual_positions, predicted_positions, title):
    plt.figure(figsize=(10, 8))
    plt.plot(actual_positions[:, 0], actual_positions[:, 1], label="Actual")
    plt.plot(predicted_positions[:, 0], predicted_positions[:, 1], label="Predicted")
    plt.title(title)
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.legend()
    plt.savefig(f"{FIGURE_LOCATION}/{title.replace(' ', '_')}.png")
    plt.close()


def plot_velocity_pred(actual_velocities, predicted_velocities, dt, title):
    time = np.arange(len(actual_velocities)) * dt
    plt.figure(figsize=(10, 8))
    plt.plot(time, np.linalg.norm(actual_velocities, axis=1), label="Actual")
    plt.plot(
        time[: len(predicted_velocities)],
        np.linalg.norm(predicted_velocities, axis=1),
        label="Predicted",
    )
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Velocity magnitude")
    plt.legend()
    plt.savefig(f"{FIGURE_LOCATION}/{title.replace(' ', '_')}.png")
    plt.close()


def plot_trajectory(
    actual_positions,
    predicted_positions,
    title,
    dataset_type,
    model_name,
    dataset_name,
    time_steps,
):
    plt.figure(figsize=(12, 10))

    # Plot actual trajectory
    plt.plot(
        actual_positions[:, 0],
        actual_positions[:, 1],
        label="Actual",
        color="blue",
        linewidth=0.5,
    )

    # Plot predicted steps as green dots
    plt.scatter(
        predicted_positions[:-1, 0],
        predicted_positions[:-1, 1],
        color="green",
        s=20,
        zorder=3,
        label="Predicted steps",
    )

    # Plot the last predicted step as an orange dot
    plt.scatter(
        predicted_positions[-1, 0],
        predicted_positions[-1, 1],
        color="orange",
        s=50,
        zorder=4,
        label="Last predicted step",
    )

    plt.title(f"{title} - {dataset_type.capitalize()}")
    plt.xlabel("X position (m)")
    plt.ylabel("Y position (m)")
    plt.legend()

    # Create folder structure
    folder = os.path.join(FIGURE_LOCATION, model_name, dataset_name, str(time_steps))
    os.makedirs(folder, exist_ok=True)

    # Save zoomed plot
    x_min, x_max = predicted_positions[:, 0].min(), predicted_positions[:, 0].max()
    y_min, y_max = predicted_positions[:, 1].min(), predicted_positions[:, 1].max()
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1
    plt.xlim(x_min - x_margin, x_max + x_margin)
    plt.ylim(y_min - y_margin, y_max + y_margin)
    plt.savefig(
        os.path.join(folder, f"{dataset_type}_zoomed.png"), dpi=300, bbox_inches="tight"
    )

    # Save full plot
    plt.xlim(actual_positions[:, 0].min(), actual_positions[:, 0].max())
    plt.ylim(actual_positions[:, 1].min(), actual_positions[:, 1].max())
    plt.savefig(
        os.path.join(folder, f"{dataset_type}_full.png"), dpi=300, bbox_inches="tight"
    )

    plt.close()


def plot_combined_trajectory(
    train_actual,
    train_pred,
    val_actual,
    val_pred,
    test_actual,
    test_pred,
    title,
    model_name,
    dataset_name,
    time_steps,
):
    plt.figure(figsize=(12, 10))

    # Plot actual trajectories
    plt.plot(
        train_actual[:, 0],
        train_actual[:, 1],
        label="Train Actual",
        color="blue",
        linewidth=0.5,
    )
    plt.plot(
        val_actual[:, 0],
        val_actual[:, 1],
        label="Validation Actual",
        color="green",
        linewidth=0.5,
    )
    plt.plot(
        test_actual[:, 0],
        test_actual[:, 1],
        label="Test Actual",
        color="red",
        linewidth=0.5,
    )

    # Plot predicted trajectories
    plt.scatter(
        train_pred[:, 0], train_pred[:, 1], color="cyan", s=20, label="Train Predicted"
    )
    plt.scatter(
        val_pred[:, 0], val_pred[:, 1], color="lime", s=20, label="Validation Predicted"
    )
    plt.scatter(
        test_pred[:, 0], test_pred[:, 1], color="orange", s=20, label="Test Predicted"
    )

    plt.title(title)
    plt.xlabel("X position (m)")
    plt.ylabel("Y position (m)")
    plt.legend()

    # Create folder structure
    folder = os.path.join(FIGURE_LOCATION, model_name, dataset_name, str(time_steps))
    os.makedirs(folder, exist_ok=True)

    # Save zoomed plot
    x_min = min(train_pred[:, 0].min(), val_pred[:, 0].min(), test_pred[:, 0].min())
    x_max = max(train_pred[:, 0].max(), val_pred[:, 0].max(), test_pred[:, 0].max())
    y_min = min(train_pred[:, 1].min(), val_pred[:, 1].min(), test_pred[:, 1].min())
    y_max = max(train_pred[:, 1].max(), val_pred[:, 1].max(), test_pred[:, 1].max())
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1
    plt.xlim(x_min - x_margin, x_max + x_margin)
    plt.ylim(y_min - y_margin, y_max + y_margin)
    plt.savefig(
        os.path.join(folder, "combined_zoomed.png"), dpi=300, bbox_inches="tight"
    )

    # Save full plot
    plt.xlim(
        min(train_actual[:, 0].min(), val_actual[:, 0].min(), test_actual[:, 0].min()),
        max(train_actual[:, 0].max(), val_actual[:, 0].max(), test_actual[:, 0].max()),
    )
    plt.ylim(
        min(train_actual[:, 1].min(), val_actual[:, 1].min(), test_actual[:, 1].min()),
        max(train_actual[:, 1].max(), val_actual[:, 1].max(), test_actual[:, 1].max()),
    )
    plt.savefig(os.path.join(folder, "combined_full.png"), dpi=300, bbox_inches="tight")

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
):
    time = np.arange(len(actual_velocities)) * dt
    plt.figure(figsize=(12, 8))

    # Plot actual velocity
    plt.plot(
        time,
        np.linalg.norm(actual_velocities, axis=1),
        label="Actual",
        color="blue",
        linewidth=0.5,
    )

    # Plot predicted velocity
    predicted_time = time[-len(predicted_velocities) :]
    predicted_velocity_magnitude = np.linalg.norm(predicted_velocities, axis=1)
    plt.scatter(
        predicted_time[:-1],
        predicted_velocity_magnitude[:-1],
        color="green",
        s=20,
        zorder=3,
        label="Predicted steps",
    )
    plt.scatter(
        predicted_time[-1],
        predicted_velocity_magnitude[-1],
        color="orange",
        s=50,
        zorder=4,
        label="Last predicted step",
    )

    plt.title(f"{title} - {dataset_type.capitalize()}")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity magnitude (m/s)")
    plt.legend()

    # Create folder structure
    folder = os.path.join(FIGURE_LOCATION, model_name, dataset_name, str(time_steps))
    os.makedirs(folder, exist_ok=True)

    # Save zoomed plot
    t_min, t_max = predicted_time.min(), predicted_time.max()
    v_min, v_max = (
        predicted_velocity_magnitude.min(),
        predicted_velocity_magnitude.max(),
    )
    t_margin = (t_max - t_min) * 0.1
    v_margin = (v_max - v_min) * 0.1
    plt.xlim(t_min - t_margin, t_max + t_margin)
    plt.ylim(v_min - v_margin, v_max + v_margin)
    plt.savefig(
        os.path.join(folder, f"velocity_{dataset_type}_zoomed.png"),
        dpi=300,
        bbox_inches="tight",
    )

    # Save full plot
    plt.xlim(time.min(), time.max())
    plt.ylim(
        min(
            np.linalg.norm(actual_velocities, axis=1).min(),
            predicted_velocity_magnitude.min(),
        ),
        max(
            np.linalg.norm(actual_velocities, axis=1).max(),
            predicted_velocity_magnitude.max(),
        ),
    )
    plt.savefig(
        os.path.join(folder, f"velocity_{dataset_type}_full.png"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_combined_velocity(
    train_actual,
    train_pred,
    val_actual,
    val_pred,
    test_actual,
    test_pred,
    dt,
    title,
    model_name,
    dataset_name,
    time_steps,
):
    plt.figure(figsize=(12, 8))

    # Plot actual velocities
    time_train = np.arange(len(train_actual)) * dt
    time_val = np.arange(len(val_actual)) * dt
    time_test = np.arange(len(test_actual)) * dt

    plt.plot(
        time_train,
        np.linalg.norm(train_actual, axis=1),
        label="Train Actual",
        color="blue",
        linewidth=0.5,
    )
    plt.plot(
        time_val,
        np.linalg.norm(val_actual, axis=1),
        label="Validation Actual",
        color="green",
        linewidth=0.5,
    )
    plt.plot(
        time_test,
        np.linalg.norm(test_actual, axis=1),
        label="Test Actual",
        color="red",
        linewidth=0.5,
    )

    # Plot predicted velocities
    plt.scatter(
        time_train[-len(train_pred) :],
        np.linalg.norm(train_pred, axis=1),
        color="cyan",
        s=20,
        label="Train Predicted",
    )
    plt.scatter(
        time_val[-len(val_pred) :],
        np.linalg.norm(val_pred, axis=1),
        color="lime",
        s=20,
        label="Validation Predicted",
    )
    plt.scatter(
        time_test[-len(test_pred) :],
        np.linalg.norm(test_pred, axis=1),
        color="orange",
        s=20,
        label="Test Predicted",
    )

    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity magnitude (m/s)")
    plt.legend()

    # Create folder structure
    folder = os.path.join(FIGURE_LOCATION, model_name, dataset_name, str(time_steps))
    os.makedirs(folder, exist_ok=True)

    # Save zoomed plot
    t_min = min(
        time_train[-len(train_pred) :].min(),
        time_val[-len(val_pred) :].min(),
        time_test[-len(test_pred) :].min(),
    )
    t_max = max(
        time_train[-len(train_pred) :].max(),
        time_val[-len(val_pred) :].max(),
        time_test[-len(test_pred) :].max(),
    )
    v_min = min(
        np.linalg.norm(train_pred, axis=1).min(),
        np.linalg.norm(val_pred, axis=1).min(),
        np.linalg.norm(test_pred, axis=1).min(),
    )
    v_max = max(
        np.linalg.norm(train_pred, axis=1).max(),
        np.linalg.norm(val_pred, axis=1).max(),
        np.linalg.norm(test_pred, axis=1).max(),
    )
    t_margin = (t_max - t_min) * 0.1
    v_margin = (v_max - v_min) * 0.1
    plt.xlim(t_min - t_margin, t_max + t_margin)
    plt.ylim(v_min - v_margin, v_max + v_margin)
    plt.savefig(
        os.path.join(folder, "velocity_combined_zoomed.png"),
        dpi=300,
        bbox_inches="tight",
    )

    # Save full plot
    plt.xlim(
        min(time_train.min(), time_val.min(), time_test.min()),
        max(time_train.max(), time_val.max(), time_test.max()),
    )
    plt.ylim(
        min(
            np.linalg.norm(train_actual, axis=1).min(),
            np.linalg.norm(val_actual, axis=1).min(),
            np.linalg.norm(test_actual, axis=1).min(),
            np.linalg.norm(train_pred, axis=1).min(),
            np.linalg.norm(val_pred, axis=1).min(),
            np.linalg.norm(test_pred, axis=1).min(),
        ),
        max(
            np.linalg.norm(train_actual, axis=1).max(),
            np.linalg.norm(val_actual, axis=1).max(),
            np.linalg.norm(test_actual, axis=1).max(),
            np.linalg.norm(train_pred, axis=1).max(),
            np.linalg.norm(val_pred, axis=1).max(),
            np.linalg.norm(test_pred, axis=1).max(),
        ),
    )
    plt.savefig(
        os.path.join(folder, "velocity_combined_full.png"), dpi=300, bbox_inches="tight"
    )

    plt.close()


def plot_results(actual, predicted, title, is_velocity=False):
    wandb.log(
        {
            f"{title}": wandb.plot.line_series(
                xs=list(range(len(actual))),
                ys=[actual, predicted],
                keys=["Actual", "Predicted"],
                title=title,
                xname="Time Step",
            )
        }
    )
