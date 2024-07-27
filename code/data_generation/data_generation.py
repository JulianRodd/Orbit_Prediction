import logging
import os

import numpy as np
import pandas as pd
from training.constants import DATASET_TYPES
from data_generation.constants import (
    DT,
    HEAVY_BODY1_POSITION,
    HEAVY_BODY2_POSITION,
    INITIAL_POSITION_TWO_BODY,
    M1,
    M2,
    NUM_FOLDS,
    NUM_SIMULATIONS,
    SMALL_ACCELERATION_CHANGE,
    TOTAL_STEPS,
    R,
    m,
)
from data_generation.plotting import (
    plot_all_trajectories,
    plot_all_velocities,
    plot_trajectory,
    plot_velocity,
)
from sklearn.model_selection import KFold
from tqdm import tqdm
from data_generation.utils import (
    additional_force,
    debug_orbital_characteristics,
    euler_update,
    get_initial_velocity,
    gravitational_force,
    is_complex_orbit,
    is_orbiting,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def create_folder_structure():
    """Create necessary folders for data and plots."""
    try:
        for problem in ["two_body", "two_body_force_increased_acceleration", "three_body"]:
            for dataset in ["train", "val", "test"]:
                os.makedirs(f"datasets/{problem}/{dataset}", exist_ok=True)
        os.makedirs("plots/trajectories", exist_ok=True)
    except Exception as e:
        logging.error(f"Error creating folder structure: {e}")
        raise


def generate_three_body_initial_conditions():
    """Generate initial conditions for the three-body problem."""
    try:
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(1.5 * R, 3 * R)
        r3 = np.array([distance * np.cos(angle), distance * np.sin(angle)])
        v1 = v2 = np.array([0.0, 0.0])
        speed = np.random.uniform(0.8, 1.2) * get_initial_velocity(M1 + M2, m, distance)
        v_angle = np.random.uniform(0, 2 * np.pi)
        v3 = speed * np.array([np.cos(v_angle), np.sin(v_angle)])
        return HEAVY_BODY1_POSITION, HEAVY_BODY2_POSITION, r3, v1, v2, v3
    except Exception as e:
        logging.error(f"Error generating three-body initial conditions: {e}")
        raise


def simulate_trajectory(problem_type, initial_conditions):
    """Simulate trajectory for a given problem type and initial conditions."""
    try:
        if problem_type == "three_body":
            r1, r2, r3, v1, v2, v3 = initial_conditions
        elif problem_type == "two_body":
            r1 = np.array([0.0, 0.0])
            r2 = INITIAL_POSITION_TWO_BODY + np.array([0, initial_conditions])
            v1 = np.array([0.0, 0.0])
            v2 = np.array([0.0, get_initial_velocity(M1, m, R)])

        else:
          r1 = np.array([0.0, 0.0])
          r2 = INITIAL_POSITION_TWO_BODY + np.array([0, initial_conditions])
          v1 = np.array([0.0, 0.0])
          v2 = np.array([0.0, get_initial_velocity(M1, m, R)])
        positions = []
        velocities = []

        for step in tqdm(
            range(TOTAL_STEPS), desc=f"Simulating {problem_type}", leave=False
        ):
            if problem_type == "three_body":
                F1 = gravitational_force(M1, m, r1, r3)
                F2 = gravitational_force(M2, m, r2, r3)
                F = F1 + F2
                r3, v3 = euler_update(r3, v3, F, m, step)
                positions.append(r3)
                velocities.append(v3)
            elif problem_type == "two_body":
                F = gravitational_force(M1, m, r1, r2)
                r2, v2 = euler_update(r2, v2, F, m, step)
                positions.append(r2)
                velocities.append(v2)
            elif problem_type == "two_body_force_increased_acceleration":
                F = gravitational_force(M1, m, r1, r2) + additional_force(v2)
                r2, v2 = euler_update(
                    r2, v2, F, m, step, increased_acceleration=SMALL_ACCELERATION_CHANGE
                )
                positions.append(r2)
                velocities.append(v2)
            else:
                raise ValueError(f"Invalid problem type: {problem_type}")
            if step % 1000 == 0:
                debug_orbital_characteristics(r2, v2, F, m)
        return np.array(positions), np.array(velocities)
    except Exception as e:
        logging.error(f"Error simulating trajectory: {e}")
        raise


def create_datasets(problem_type):
    """Create datasets for a given problem type."""
    try:
        logging.info(f"Creating datasets for {problem_type}")
        all_positions = []
        all_velocities = []

        num_valid_simulations = 0
        pbar = tqdm(
            total=NUM_SIMULATIONS, desc=f"Generating {problem_type} simulations"
        )

        while num_valid_simulations < NUM_SIMULATIONS:
            if problem_type == "three_body":
                initial_conditions = generate_three_body_initial_conditions()
            else:
                initial_conditions = np.random.uniform(-5, 5)

            positions, velocities = simulate_trajectory(
                problem_type, initial_conditions
            )

            if problem_type == "three_body":
                valid = is_complex_orbit(positions, velocities)
            else:
                valid = is_orbiting(positions)
            if valid:
                all_positions.append(positions)
                all_velocities.append(velocities)
                num_valid_simulations += 1
                pbar.update(1)

        pbar.close()

        all_positions = np.array(all_positions)
        all_velocities = np.array(all_velocities)

        # Split data into train/val and test
        num_test = max(1, int(NUM_SIMULATIONS * 0.1))
        num_train_val = NUM_SIMULATIONS - num_test

        # Create 5-fold cross-validation splits
        kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

        for fold, (train_index, val_index) in enumerate(kf.split(range(num_train_val))):
            for dataset_type, indices in [("train", train_index), ("val", val_index)]:
                data = []
                for spaceship_id, idx in enumerate(indices):
                    for timestep in range(len(all_positions[idx])):
                        data.append(
                            {
                                "timestep": timestep,
                                "spaceship_id": spaceship_id,
                                "fold": fold,
                                "x": all_positions[idx, timestep, 0],
                                "y": all_positions[idx, timestep, 1],
                                "Vx": all_velocities[idx, timestep, 0],
                                "Vy": all_velocities[idx, timestep, 1],
                            }
                        )

                df = pd.DataFrame(data)
                df.to_csv(
                    f"datasets/{problem_type}/{dataset_type}/{problem_type}_{dataset_type}_fold{fold}.csv",
                    index=False,
                )

        # Save test data
        test_data = []
        for spaceship_id in range(num_train_val, NUM_SIMULATIONS):
            for timestep in range(len(all_positions[spaceship_id])):
                test_data.append(
                    {
                        "timestep": timestep,
                        "spaceship_id": spaceship_id - num_train_val,
                        "x": all_positions[spaceship_id, timestep, 0],
                        "y": all_positions[spaceship_id, timestep, 1],
                        "Vx": all_velocities[spaceship_id, timestep, 0],
                        "Vy": all_velocities[spaceship_id, timestep, 1],
                    }
                )

        test_df = pd.DataFrame(test_data)
        test_df.to_csv(
            f"datasets/{problem_type}/test/{problem_type}_test.csv", index=False
        )

        return all_positions, all_velocities, num_train_val
    except Exception as e:
        logging.error(f"Error creating datasets: {e}")
        raise


def data_generation(problem_types = DATASET_TYPES):
    try:
        logging.info("Starting data generation process")
        create_folder_structure()

        for problem_type in problem_types:
            all_positions, all_velocities, num_train_val = create_datasets(problem_type)

            # Plot individual trajectories and velocities
            for i in tqdm(
                range(NUM_SIMULATIONS), desc=f"Plotting {problem_type} trajectories"
            ):
                plot_trajectory(
                    all_positions[i],
                    problem_type,
                    f"{problem_type.capitalize()} Trajectory - Simulation {i}",
                    f"trajectory_{i}.png",
                )
                plot_velocity(
                    all_velocities[i],
                    DT,
                    problem_type,
                    f"{problem_type.capitalize()} Velocity - Simulation {i}",
                    f"velocity_{i}.png",
                )

            # Plot all trajectories and velocities
            plot_all_trajectories(
                all_positions,
                problem_type,
                num_train_val,
                NUM_SIMULATIONS - num_train_val,
            )
            plot_all_velocities(
                all_velocities,
                problem_type,
                num_train_val,
                NUM_SIMULATIONS - num_train_val,
            )

        logging.info("Data generation process completed successfully")
    except Exception as e:
        logging.error(f"Error in main function: {e}")
        raise
