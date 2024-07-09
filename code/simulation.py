import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from generics import *
from sklearn.model_selection import KFold
from utils import (
    additional_force,
    euler_update,
    get_initial_velocity,
    gravitational_force,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def create_folder_structure():
    for problem in ["two_body", "two_body_force_increased_acceleration", "three_body"]:
        for dataset in ["train", "val", "test"]:
            os.makedirs(f"datasets/{problem}/{dataset}", exist_ok=True)
    os.makedirs("plots/trajectories", exist_ok=True)
    for problem in ["two_body", "two_body_force_increased_acceleration", "three_body"]:
        os.makedirs(f"plots/trajectories/{problem}", exist_ok=True)


def simulate_trajectory(problem_type, initial_conditions):
    logging.info(f"Simulating trajectory for {problem_type}")
    if problem_type == "three_body":
        r1, r2, r3, v1, v2, v3 = initial_conditions
    elif problem_type == "two_body":
        r1 = np.array([0.0, 0.0])
        r2 = INITIAL_POSITION_TWO_BODY + np.array([0, initial_conditions])
        v1 = np.array([0.0, 0.0])
        v2 = np.array([0.0, get_initial_velocity(M1, m, R)])
    else:  # two_body_force_increased_acceleration
        r1 = np.array([0.0, 0.0])
        r2 = INITIAL_POSITION_TWO_BODY + np.array([0, initial_conditions])
        v1 = np.array([0.0, 0.0])
        v2 = np.array([0.0, get_initial_velocity(M1, m, R)])

    positions = []
    velocities = []

    for _ in range(TOTAL_STEPS):
        if problem_type == "three_body":
            F1 = gravitational_force(M1, m, r1, r3)
            F2 = gravitational_force(M2, m, r2, r3)
            F = F1 + F2
            r3, v3 = euler_update(r3, v3, F, m, DT)
            positions.append(r3)
            velocities.append(v3)
        elif problem_type == "two_body":
            F = gravitational_force(M1, m, r1, r2)
            r2, v2 = euler_update(r2, v2, F, m, DT)
            positions.append(r2)
            velocities.append(v2)
        else:  # two_body_force_increased_acceleration
            F = gravitational_force(M1, m, r1, r2) + additional_force(v2)
            r2, v2 = euler_update(
                r2, v2, F, m, DT, increased_acceleration=SMALL_ACCELARATION_CHANGE
            )
            positions.append(r2)
            velocities.append(v2)

    return np.array(positions), np.array(velocities)


def is_orbiting(positions):
    # Check if the trajectory forms at least one complete orbit
    # and doesn't just fly away in one direction
    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()

    x_range = x_max - x_min
    y_range = y_max - y_min

    # Check if the trajectory spans a significant range in both x and y directions
    if x_range < 0.1 * abs(x_max) or y_range < 0.1 * abs(y_max):
        return False

    # Check for at least one complete orbit
    x_changes = np.diff(np.sign(positions[:, 0] - positions[0, 0])).nonzero()[0]
    y_changes = np.diff(np.sign(positions[:, 1] - positions[0, 1])).nonzero()[0]

    return len(x_changes) >= 2 and len(y_changes) >= 2


# Modify the is_complex_orbit function
def is_complex_orbit(positions, velocities):
    # Check if the trajectory forms a complex, non-escaping orbit
    x, y = positions[:, 0], positions[:, 1]

    # Check if the trajectory is bounded
    x_range = np.max(x) - np.min(x)
    y_range = np.max(y) - np.min(y)
    max_range = max(x_range, y_range)

    if max_range > 1e6:  # Adjust this threshold as needed
        return False  # Trajectory is likely escaping

    # Check for complexity: look for changes in direction
    dx = np.diff(x)
    dy = np.diff(y)
    direction_changes = np.sum(np.abs(np.diff(np.arctan2(dy, dx))) > np.pi / 4)

    if direction_changes < 30:  # Adjust this threshold as needed
        return False  # Not enough complexity in the orbit


    # check distance between the two heavy bodies and the spaceship
    r1 = HEAVY_BODY1_POSITION
    r2 = HEAVY_BODY2_POSITION
    middle = (r1 + r2) / 2
    distances = np.linalg.norm(positions - middle, axis=1)
    if np.max(distances) > 200000:
        return False

    return True


def generate_three_body_initial_conditions():
    # Generate more varied initial conditions for the three-body problem
    r1 = HEAVY_BODY1_POSITION
    r2 = HEAVY_BODY2_POSITION

    # Randomize initial position of the spaceship
    angle = np.random.uniform(0, 2 * np.pi)
    distance = np.random.uniform(1.5 * R, 3 * R)  # Adjust this range as needed
    r3 = np.array([distance * np.cos(angle), distance * np.sin(angle)])

    v1 = np.array([0.0, 0.0])
    v2 = np.array([0.0, 0.0])

    # Randomize initial velocity of the spaceship
    speed = np.random.uniform(0.8, 1.2) * get_initial_velocity(M1 + M2, m, distance)
    v_angle = np.random.uniform(0, 2 * np.pi)
    v3 = speed * np.array([np.cos(v_angle), np.sin(v_angle)])

    return r1, r2, r3, v1, v2, v3


def create_datasets(problem_type):
    logging.info(f"Creating datasets for {problem_type}")
    all_positions = []
    all_velocities = []

    num_test_spaceships = int(NUM_SIMULATIONS * 0.1) if NUM_SIMULATIONS > 10 else 1
    num_train_val_spaceships = NUM_SIMULATIONS - num_test_spaceships

    for i in range(NUM_SIMULATIONS):
        orbiting = False
        attempts = 0
        while not orbiting and attempts < 100:
            if problem_type == "three_body":
                initial_conditions = generate_three_body_initial_conditions()
            else:
                initial_conditions = np.random.uniform(-5, 5)

            positions, velocities = simulate_trajectory(
                problem_type, initial_conditions
            )

            if problem_type == "three_body":
                orbiting = is_complex_orbit(positions, velocities)
            else:
                orbiting = is_orbiting(positions)

            attempts += 1
            logging.info(f"Spaceship {i}: Attempt {attempts}, Orbiting: {orbiting}")

        if not orbiting:
            logging.warning(
                f"Failed to generate valid trajectory for spaceship {i} after {attempts} attempts"
            )

        all_positions.append(positions)
        all_velocities.append(velocities)

    all_positions = np.array(all_positions)
    all_velocities = np.array(all_velocities)

    # Split data into train/val and test
    train_val_positions = all_positions[:num_train_val_spaceships]
    train_val_velocities = all_velocities[:num_train_val_spaceships]
    test_positions = all_positions[num_train_val_spaceships:]
    test_velocities = all_velocities[num_train_val_spaceships:]

    # Create 5-fold cross-validation splits
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_index, val_index) in enumerate(
        kf.split(range(num_train_val_spaceships))
    ):
        train_positions = train_val_positions[train_index]
        train_velocities = train_val_velocities[train_index]
        val_positions = train_val_positions[val_index]
        val_velocities = train_val_velocities[val_index]

        for dataset_type, positions, velocities in [
            ("train", train_positions, train_velocities),
            ("val", val_positions, val_velocities),
        ]:
            data = []
            for spaceship_id, (pos, vel) in enumerate(zip(positions, velocities)):
                for timestep in range(len(pos)):
                    data.append(
                        {
                            "timestep": timestep,
                            "spaceship_id": spaceship_id,
                            "fold": fold,
                            "x": pos[timestep, 0],
                            "y": pos[timestep, 1],
                            "Vx": vel[timestep, 0],
                            "Vy": vel[timestep, 1],
                        }
                    )

            df = pd.DataFrame(data)
            df.to_csv(
                f"datasets/{problem_type}/{dataset_type}/{problem_type}_{dataset_type}_fold{fold}.csv",
                index=False,
            )
            logging.info(
                f"Saved dataset for {problem_type}, {dataset_type}, fold {fold}"
            )

    # Save test data
    test_data = []
    for spaceship_id, (pos, vel) in enumerate(zip(test_positions, test_velocities)):
        for timestep in range(len(pos)):
            test_data.append(
                {
                    "timestep": timestep,
                    "spaceship_id": spaceship_id,
                    "x": pos[timestep, 0],
                    "y": pos[timestep, 1],
                    "Vx": vel[timestep, 0],
                    "Vy": vel[timestep, 1],
                }
            )

    test_df = pd.DataFrame(test_data)
    test_df.to_csv(f"datasets/{problem_type}/test/{problem_type}_test.csv", index=False)
    logging.info(f"Saved test dataset for {problem_type}")

    return all_positions, all_velocities, num_train_val_spaceships


def plot_trajectories(problem_type, all_positions, num_train_val_spaceships):
    logging.info(f"Plotting trajectories for {problem_type}")
    plt.figure(figsize=(15, 12))

    # Plot train/val trajectories
    colors = plt.cm.Set1(np.linspace(0, 1, 5))
    for i in range(num_train_val_spaceships):
        fold = i % 5
        plt.plot(
            all_positions[i, :, 0],
            all_positions[i, :, 1],
            c=colors[fold],
            alpha=0.5,
            label=f"Fold {fold}" if i < 5 else "",
        )

    # Plot test trajectories
    for i in range(num_train_val_spaceships, NUM_SIMULATIONS):
        plt.plot(
            all_positions[i, :, 0],
            all_positions[i, :, 1],
            "k-",
            alpha=0.5,
            label="Test" if i == num_train_val_spaceships else "",
        )

    if problem_type in ["two_body", "two_body_force_increased_acceleration"]:
        plt.scatter(0, 0, c="yellow", s=100, label="Heavy Body")
    else:  # three_body
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

    plt.title(f'Trajectories for {problem_type.replace("_", " ").title()}')
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.savefig(f"plots/trajectories/{problem_type}/all_trajectories.png")
    plt.close()


def plot_velocities(problem_type, all_velocities, num_train_val_spaceships):
    logging.info(f"Plotting velocities for {problem_type}")
    plt.figure(figsize=(15, 12))

    colors = plt.cm.Set1(np.linspace(0, 1, 5))
    for i in range(num_train_val_spaceships):
        fold = i % 5
        velocities = np.linalg.norm(all_velocities[i], axis=1)
        plt.plot(
            velocities, c=colors[fold], alpha=0.5, label=f"Fold {fold}" if i < 5 else ""
        )

    for i in range(num_train_val_spaceships, NUM_SIMULATIONS):
        velocities = np.linalg.norm(all_velocities[i], axis=1)
        plt.plot(
            velocities,
            "k-",
            alpha=0.5,
            label="Test" if i == num_train_val_spaceships else "",
        )

    plt.title(f'Velocities for {problem_type.replace("_", " ").title()}')
    plt.xlabel("Time step")
    plt.ylabel("Velocity magnitude")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plots/trajectories/{problem_type}/all_velocities.png")
    plt.close()


def plot_test_trajectories(problem_type, all_positions, num_train_val_spaceships):
    logging.info(f"Plotting test trajectories for {problem_type}")
    plt.figure(figsize=(15, 12))

    for i in range(num_train_val_spaceships, NUM_SIMULATIONS):
        plt.plot(
            all_positions[i, :, 0],
            all_positions[i, :, 1],
            label=f"Test Spaceship {i-num_train_val_spaceships}",
        )

    if problem_type in ["two_body", "two_body_force_increased_acceleration"]:
        plt.scatter(0, 0, c="yellow", s=100, label="Heavy Body")
    else:  # three_body
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

    plt.title(f'Test Trajectories for {problem_type.replace("_", " ").title()}')
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.savefig(f"plots/trajectories/{problem_type}/test_trajectories.png")
    plt.close()


def plot_train_test_combined(problem_type, all_positions, num_train_val_spaceships):
    logging.info(f"Plotting combined train and test trajectories for {problem_type}")
    plt.figure(figsize=(15, 12))

    for i in range(num_train_val_spaceships):
        plt.plot(
            all_positions[i, :, 0],
            all_positions[i, :, 1],
            "b-",
            alpha=0.5,
            label="Train" if i == 0 else "",
        )

    for i in range(num_train_val_spaceships, NUM_SIMULATIONS):
        plt.plot(
            all_positions[i, :, 0],
            all_positions[i, :, 1],
            "r-",
            alpha=0.5,
            label="Test" if i == num_train_val_spaceships else "",
        )

    if problem_type in ["two_body", "two_body_force_increased_acceleration"]:
        plt.scatter(0, 0, c="yellow", s=100, label="Heavy Body")
    else:  # three_body
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

    plt.title(
        f'Combined Train and Test Trajectories for {problem_type.replace("_", " ").title()}'
    )
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.savefig(
        f"plots/trajectories/{problem_type}/train_test_combined_trajectories.png"
    )
    plt.close()

def plot_individual_trajectory(problem_type, positions, spaceship_id, is_test=False):
    logging.info(f"Plotting individual trajectory for {problem_type}, spaceship {spaceship_id}")
    plt.figure(figsize=(12, 10))

    plt.plot(positions[:, 0], positions[:, 1], 'b-', label='Trajectory')
    plt.scatter(positions[0, 0], positions[0, 1], c='green', s=100, label='Start', zorder=5)
    plt.scatter(positions[-1, 0], positions[-1, 1], c='red', s=100, label='End', zorder=5)

    if problem_type in ['two_body', 'two_body_force_increased_acceleration']:
        plt.scatter(0, 0, c='yellow', s=100, label='Heavy Body')
    else:  # three_body
        plt.scatter(HEAVY_BODY1_POSITION[0], HEAVY_BODY1_POSITION[1], c='yellow', s=100, label='Heavy Body 1')
        plt.scatter(HEAVY_BODY2_POSITION[0], HEAVY_BODY2_POSITION[1], c='orange', s=100, label='Heavy Body 2')

    plt.title(f'{"Test " if is_test else ""}Trajectory for {problem_type.replace("_", " ").title()} - Spaceship {spaceship_id}')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    folder = f'plots/trajectories/{problem_type}/{"test_" if is_test else ""}individual_spaceships'
    os.makedirs(folder, exist_ok=True)
    plt.savefig(f'{folder}/spaceship_{spaceship_id}_trajectory.png')
    plt.close()

def main():
    logging.info("Starting simulation process")
    create_folder_structure()
    for problem_type in ['two_body', 'two_body_force_increased_acceleration', 'three_body']:
        all_positions, all_velocities, num_train_val_spaceships = create_datasets(problem_type)
        plot_trajectories(problem_type, all_positions, num_train_val_spaceships)
        plot_velocities(problem_type, all_velocities, num_train_val_spaceships)
        plot_test_trajectories(problem_type, all_positions, num_train_val_spaceships)
        plot_train_test_combined(problem_type, all_positions, num_train_val_spaceships)

        # Plot individual trajectories
        for i in range(num_train_val_spaceships):
            plot_individual_trajectory(problem_type, all_positions[i], i)

        for i in range(num_train_val_spaceships, NUM_SIMULATIONS):
            plot_individual_trajectory(problem_type, all_positions[i], i - num_train_val_spaceships, is_test=True)

    logging.info("Simulation process completed")

if __name__ == "__main__":
    main()
