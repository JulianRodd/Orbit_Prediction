import logging
import os

import matplotlib.pyplot as plt
import numpy as np

from .constants import HEAVY_BODY1_POSITION, HEAVY_BODY2_POSITION

logger = logging.getLogger(__name__)


def plot_predictions(
    actual,
    predicted,
    model_type,
    dataset_type,
    steps,
    spaceship_id,
    output_folder,
    split,
    prediction_start,
):
    try:
        plt.figure(figsize=(12, 10))
        plt.plot(actual[:, 0], actual[:, 1], color="blue", label="Actual", linewidth=2)
        plt.plot(
            predicted[:, 0],
            predicted[:, 1],
            color="red",
            label="Predicted",
            linewidth=3,
            linestyle="--",
        )
        plt.plot(
            actual[:, 0],
            actual[:, 1],
            color="limegreen",
            label="Target",
            linewidth=3,
            linestyle="--",
        )

        plt.scatter(
            actual[0, 0], actual[0, 1], color="green", s=100, label="Start", zorder=5
        )
        plt.scatter(
            actual[-1, 0], actual[-1, 1], color="purple", s=100, label="End", zorder=5
        )
        plt.scatter(
            predicted[0, 0],
            predicted[0, 1],
            color="orange",
            s=100,
            label="Prediction Start",
            zorder=5,
        )

        if "two_body" in dataset_type:
            plt.scatter(0, 0, color="black", s=200, label="Heavy Body", zorder=5)
        elif "three_body" in dataset_type:
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

        plt.title(
            f"{model_type} on {dataset_type}\n{steps} steps from t={prediction_start} - Spaceship {spaceship_id}",
            fontsize=16,
        )
        plt.xlabel("X Position (m)", fontsize=14)
        plt.ylabel("Y Position (m)", fontsize=14)
        plt.legend(fontsize=12, loc="upper left", bbox_to_anchor=(1, 1))
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.axis("equal")

        save_path = os.path.join(
            output_folder,
            split,
            model_type,
            dataset_type,
            str(steps),
            f"trajectory_spaceship_{spaceship_id}.png",
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Trajectory plot saved: {save_path}")
    except Exception as e:
        logger.error(f"Error in plot_predictions: {str(e)}")
        raise


def plot_full_trajectory(
    full_trajectory,
    predicted,
    model_type,
    dataset_type,
    steps,
    spaceship_id,
    output_folder,
    split,
    prediction_start,
):
    try:
        plt.figure(figsize=(12, 10))
        plt.plot(
            full_trajectory[:, 0],
            full_trajectory[:, 1],
            color="blue",
            label="Full Trajectory",
            linewidth=2,
        )
        plt.plot(
            predicted[:, 0],
            predicted[:, 1],
            color="red",
            label="Predicted",
            linewidth=3,
            linestyle="--",
        )
        plt.plot(
            full_trajectory[prediction_start : prediction_start + steps, 0],
            full_trajectory[prediction_start : prediction_start + steps, 1],
            color="limegreen",
            label="Target",
            linewidth=3,
            linestyle="--",
        )

        plt.scatter(
            full_trajectory[0, 0],
            full_trajectory[0, 1],
            color="green",
            s=100,
            label="Start",
            zorder=5,
        )
        plt.scatter(
            full_trajectory[-1, 0],
            full_trajectory[-1, 1],
            color="purple",
            s=100,
            label="End",
            zorder=5,
        )
        plt.scatter(
            full_trajectory[prediction_start, 0],
            full_trajectory[prediction_start, 1],
            color="orange",
            s=100,
            label="Prediction Start",
            zorder=5,
        )

        if "two_body" in dataset_type:
            plt.scatter(0, 0, color="black", s=200, label="Heavy Body", zorder=5)
        elif "three_body" in dataset_type:
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

        plt.title(
            f"{model_type} on {dataset_type}\nFull Trajectory - {steps} steps prediction - Spaceship {spaceship_id}",
            fontsize=16,
        )
        plt.xlabel("X Position (m)", fontsize=14)
        plt.ylabel("Y Position (m)", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.axis("equal")

        save_path = os.path.join(
            output_folder,
            split,
            model_type,
            dataset_type,
            str(steps),
            f"full_trajectory_spaceship_{spaceship_id}.png",
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Full trajectory plot saved: {save_path}")
    except Exception as e:
        logger.error(f"Error in plot_full_trajectory: {str(e)}")
        raise


def plot_velocities(
    actual,
    predicted,
    model_type,
    dataset_type,
    steps,
    spaceship_id,
    output_folder,
    split,
    prediction_start,
):
    try:
        actual_velocity = np.linalg.norm(actual[:, 2:], axis=1)
        predicted_velocity = np.linalg.norm(predicted[:, 2:], axis=1)

        plt.figure(figsize=(12, 8))
        plt.plot(
            range(len(actual_velocity)),
            actual_velocity,
            color="blue",
            label="Actual",
            linewidth=2,
        )
        plt.plot(
            range(len(predicted_velocity)),
            predicted_velocity,
            color="red",
            label="Predicted",
            linewidth=2,
            linestyle="--",
        )
        plt.scatter(
            predicted[0, 0],
            predicted[0, 1],
            color="orange",
            s=100,
            label="Prediction Start",
            zorder=5,
        )
        plt.title(
            f"{model_type} on {dataset_type}\nVelocity - {steps} steps - Spaceship {spaceship_id}",
            fontsize=16,
        )
        plt.xlabel("Time Step", fontsize=14)
        plt.ylabel("Velocity Magnitude (m/s)", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)

        save_path = os.path.join(
            output_folder,
            split,
            model_type,
            dataset_type,
            str(steps),
            f"velocity_spaceship_{spaceship_id}.png",
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Velocity plot saved: {save_path}")
    except Exception as e:
        logger.error(f"Error in plot_velocities: {str(e)}")
        raise


def plot_error_distribution(
    actual,
    predicted,
    model_type,
    dataset_type,
    steps,
    spaceship_id,
    output_folder,
    split,
):
    try:
        error = np.linalg.norm(actual - predicted, axis=1)

        plt.figure(figsize=(10, 6))
        plt.hist(error, bins=30, edgecolor="black")
        plt.title(
            f"{model_type} on {dataset_type}\nError Distribution - {steps} steps - Spaceship {spaceship_id}",
            fontsize=16,
        )
        plt.xlabel("Error Magnitude", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.grid(True, linestyle="--", alpha=0.7)

        save_path = os.path.join(
            output_folder,
            split,
            model_type,
            dataset_type,
            str(steps),
            f"error_distribution_spaceship_{spaceship_id}.png",
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Error distribution plot saved: {save_path}")
    except Exception as e:
        logger.error(f"Error in plot_error_distribution: {str(e)}")
        raise


def plot_error_over_time(
    actual,
    predicted,
    model_type,
    dataset_type,
    steps,
    spaceship_id,
    output_folder,
    split,
):
    try:
        error = np.linalg.norm(actual - predicted, axis=1)

        plt.figure(figsize=(12, 6))
        plt.plot(range(len(error)), error, color="red", linewidth=2)
        plt.title(
            f"{model_type} on {dataset_type}\nError Over Time - {steps} steps - Spaceship {spaceship_id}",
            fontsize=16,
        )
        plt.xlabel("Time Step", fontsize=14)
        plt.ylabel("Error Magnitude", fontsize=14)
        plt.grid(True, linestyle="--", alpha=0.7)

        save_path = os.path.join(
            output_folder,
            split,
            model_type,
            dataset_type,
            str(steps),
            f"error_over_time_spaceship_{spaceship_id}.png",
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Error over time plot saved: {save_path}")
    except Exception as e:
        logger.error(f"Error in plot_error_over_time: {str(e)}")
        raise


def plot_all(
    actual,
    predicted,
    full_trajectory,
    model_type,
    dataset_type,
    steps,
    spaceship_id,
    output_folder,
    split,
    prediction_start,
):
    try:
        plot_predictions(
            actual,
            predicted,
            model_type,
            dataset_type,
            steps,
            spaceship_id,
            output_folder,
            split,
            prediction_start,
        )
        plot_full_trajectory(
            full_trajectory,
            predicted,
            model_type,
            dataset_type,
            steps,
            spaceship_id,
            output_folder,
            split,
            prediction_start,
        )
        plot_velocities(
            actual,
            predicted,
            model_type,
            dataset_type,
            steps,
            spaceship_id,
            output_folder,
            split,
            prediction_start,
        )
        plot_full_velocities(
            full_trajectory,
            predicted,
            model_type,
            dataset_type,
            steps,
            spaceship_id,
            output_folder,
            split,
            prediction_start,
        )
        plot_error_distribution(
            actual,
            predicted,
            model_type,
            dataset_type,
            steps,
            spaceship_id,
            output_folder,
            split,
        )
        plot_error_over_time(
            actual,
            predicted,
            model_type,
            dataset_type,
            steps,
            spaceship_id,
            output_folder,
            split,
        )
        logger.info(
            f"All plots generated for {model_type} on {dataset_type}, {steps} steps, spaceship {spaceship_id}"
        )
    except Exception as e:
        logger.error(f"Error in plot_all: {str(e)}")
        raise


def plot_full_velocities(
    full_trajectory,
    predicted,
    model_type,
    dataset_type,
    steps,
    spaceship_id,
    output_folder,
    split,
    prediction_start,
):
    try:
        full_velocity = np.linalg.norm(full_trajectory[:, 2:], axis=1)
        predicted_velocity = np.linalg.norm(predicted[:, 2:], axis=1)

        plt.figure(figsize=(12, 8))
        plt.plot(
            range(len(full_velocity)),
            full_velocity,
            color="blue",
            label="Full Trajectory",
            linewidth=2,
        )
        plt.plot(
            range(prediction_start, prediction_start + len(predicted_velocity)),
            predicted_velocity,
            color="red",
            label="Predicted",
            linewidth=2,
            linestyle="--",
        )

        plt.axvline(
            x=prediction_start, color="green", linestyle=":", label="Prediction Start"
        )

        plt.title(
            f"{model_type} on {dataset_type}\nFull Velocity - {steps} steps prediction - Spaceship {spaceship_id}",
            fontsize=16,
        )
        plt.xlabel("Time Step", fontsize=14)
        plt.ylabel("Velocity Magnitude (m/s)", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)

        save_path = os.path.join(
            output_folder,
            split,
            model_type,
            dataset_type,
            str(steps),
            f"full_velocity_spaceship_{spaceship_id}.png",
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Full velocity plot saved: {save_path}")
    except Exception as e:
        logger.error(f"Error in plot_full_velocities: {str(e)}")
        raise
