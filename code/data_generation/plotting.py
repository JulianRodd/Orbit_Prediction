import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from data_generation.constants import FIGURE_LOCATION, HEAVY_BODY1_POSITION, HEAVY_BODY2_POSITION

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def create_folder(folder_path):
    """Create folder if it doesn't exist."""
    try:
        os.makedirs(folder_path, exist_ok=True)
    except Exception as e:
        logging.error(f"Error creating folder {folder_path}: {e}")
        raise


def plot_trajectory(positions, problem_type, title, filename):
    """Plot trajectory for a single simulation."""
    try:
        plt.figure(figsize=(10, 8))
        plt.plot(positions[:, 0], positions[:, 1], label="Orbiting Body")
        plt.scatter(
            positions[0, 0],
            positions[0, 1],
            color="green",
            s=100,
            label="Start",
            zorder=5,
        )
        plt.scatter(
            positions[-1, 0],
            positions[-1, 1],
            color="red",
            s=100,
            label="End",
            zorder=5,
        )

        if problem_type == "two_body":
            plt.scatter(0, 0, color="black", s=200, label="Heavy Body", zorder=5)
        elif problem_type == "three_body":
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
        plt.axis("equal")
        plt.margins(0.1)

        folder_path = os.path.join(FIGURE_LOCATION, problem_type)
        create_folder(folder_path)
        plt.savefig(os.path.join(folder_path, filename), dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as e:
        logging.error(f"Error plotting trajectory: {e}")
        raise


def plot_velocity(velocities, dt, problem_type, title, filename):
    """Plot velocity over time for a single simulation."""
    try:
        speed = np.linalg.norm(velocities, axis=1)
        time = np.arange(0, len(speed) * dt, dt)

        plt.figure(figsize=(10, 6))
        plt.plot(time, speed)
        plt.xlabel("Time (s)")
        plt.ylabel("Speed (m/s)")
        plt.title(title)
        plt.grid(True)

        folder_path = os.path.join(FIGURE_LOCATION, problem_type)
        create_folder(folder_path)
        plt.savefig(os.path.join(folder_path, filename), dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as e:
        logging.error(f"Error plotting velocity: {e}")
        raise


def plot_all_trajectories(all_positions, problem_type, num_train_val, num_test):
    """Plot all trajectories for a given problem type."""
    try:
        plt.figure(figsize=(15, 12))

        # Plot train/val trajectories
        for i in range(num_train_val):
            plt.plot(
                all_positions[i, :, 0],
                all_positions[i, :, 1],
                "b-",
                alpha=0.5,
                label="Train/Val" if i == 0 else "",
            )

        # Plot test trajectories
        for i in range(num_train_val, num_train_val + num_test):
            plt.plot(
                all_positions[i, :, 0],
                all_positions[i, :, 1],
                "r-",
                alpha=0.5,
                label="Test" if i == num_train_val else "",
            )

        if problem_type == "two_body":
            plt.scatter(0, 0, c="yellow", s=100, label="Heavy Body")
        elif problem_type == "three_body":
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

        plt.title(f'All Trajectories for {problem_type.replace("_", " ").title()}')
        plt.xlabel("X position")
        plt.ylabel("Y position")
        plt.legend()
        plt.grid(True)
        plt.axis("equal")

        folder_path = os.path.join(FIGURE_LOCATION, problem_type)
        create_folder(folder_path)
        plt.savefig(
            os.path.join(folder_path, "all_trajectories.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    except Exception as e:
        logging.error(f"Error plotting all trajectories: {e}")
        raise


def plot_all_velocities(all_velocities, problem_type, num_train_val, num_test):
    """Plot all velocities for a given problem type."""
    try:
        plt.figure(figsize=(15, 12))

        # Plot train/val velocities
        for i in range(num_train_val):
            velocities = np.linalg.norm(all_velocities[i], axis=1)
            plt.plot(velocities, "b-", alpha=0.5, label="Train/Val" if i == 0 else "")

        # Plot test velocities
        for i in range(num_train_val, num_train_val + num_test):
            velocities = np.linalg.norm(all_velocities[i], axis=1)
            plt.plot(
                velocities, "r-", alpha=0.5, label="Test" if i == num_train_val else ""
            )

        plt.title(f'All Velocities for {problem_type.replace("_", " ").title()}')
        plt.xlabel("Time step")
        plt.ylabel("Velocity magnitude")
        plt.legend()
        plt.grid(True)

        folder_path = os.path.join(FIGURE_LOCATION, problem_type)
        create_folder(folder_path)
        plt.savefig(
            os.path.join(folder_path, "all_velocities.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    except Exception as e:
        logging.error(f"Error plotting all velocities: {e}")
        raise
