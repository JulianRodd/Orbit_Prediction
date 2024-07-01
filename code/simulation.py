import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from generics import *
from utils import get_initial_velocity, gravitational_force, euler_update, additional_force
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_folder_structure():
    for problem in ['two_body', 'two_body_force_increased_acceleration', 'three_body']:
        for dataset in ['train', 'val', 'test']:
            for spaceship_id in range(NUM_SIMULATIONS):
                os.makedirs(f'datasets/{problem}/{dataset}/{spaceship_id}', exist_ok=True)
    os.makedirs('plots/trajectories', exist_ok=True)
    for problem in ['two_body', 'two_body_force_increased_acceleration', 'three_body']:
        os.makedirs(f'plots/trajectories/{problem}', exist_ok=True)

def simulate_trajectory(problem_type, initial_conditions):
    logging.info(f"Simulating trajectory for {problem_type}")
    if problem_type == 'three_body':
        r1, r2, r3, v1, v2, v3 = initial_conditions
    elif problem_type == 'two_body':
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
        if problem_type == 'three_body':
            F1 = gravitational_force(M1, m, r1, r3)
            F2 = gravitational_force(M2, m, r2, r3)
            F = F1 + F2
            r3, v3 = euler_update(r3, v3, F, m, DT)
            positions.append(r3)
            velocities.append(v3)
        elif problem_type == 'two_body':
            F = gravitational_force(M1, m, r1, r2)
            r2, v2 = euler_update(r2, v2, F, m, DT)
            positions.append(r2)
            velocities.append(v2)
        else:  # two_body_force_increased_acceleration
            F = gravitational_force(M1, m, r1, r2) + additional_force(v2)
            r2, v2 = euler_update(r2, v2, F, m, DT, increased_acceleration=SMALL_ACCELARATION_CHANGE)
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
    direction_changes = np.sum(np.abs(np.diff(np.arctan2(dy, dx))) > np.pi/4)

    if direction_changes < 30:  # Adjust this threshold as needed
        return False  # Not enough complexity in the orbit

    # Check if the spaceship is flying in approximately the same direction for too long
    velocity_directions = np.arctan2(velocities[:, 1], velocities[:, 0])
    direction_consistency = np.sum(np.abs(np.diff(velocity_directions)) < np.pi/6)
    if direction_consistency > 1 * len(velocity_directions):
        return False  # Spaceship is flying in the same direction for too long

    return True

def generate_three_body_initial_conditions():
    # Generate more varied initial conditions for the three-body problem
    r1 = HEAVY_BODY1_POSITION
    r2 = HEAVY_BODY2_POSITION

    # Randomize initial position of the spaceship
    angle = np.random.uniform(0, 2*np.pi)
    distance = np.random.uniform(1.5*R, 3*R)  # Adjust this range as needed
    r3 = np.array([distance * np.cos(angle), distance * np.sin(angle)])

    v1 = np.array([0.0, 0.0])
    v2 = np.array([0.0, 0.0])

    # Randomize initial velocity of the spaceship
    speed = np.random.uniform(0.8, 1.2) * get_initial_velocity(M1 + M2, m, distance)
    v_angle = np.random.uniform(0, 2*np.pi)
    v3 = speed * np.array([np.cos(v_angle), np.sin(v_angle)])

    return r1, r2, r3, v1, v2, v3

def create_datasets(problem_type):
    logging.info(f"Creating datasets for {problem_type}")
    all_positions = []
    all_velocities = []

    for i in range(NUM_SIMULATIONS):
        orbiting = False
        attempts = 0
        while not orbiting and attempts < 100:  # Increased max attempts
            if problem_type == 'three_body':
                initial_conditions = generate_three_body_initial_conditions()
            else:
                initial_conditions = np.random.uniform(-5, 5)

            positions, velocities = simulate_trajectory(problem_type, initial_conditions)

            if problem_type == 'three_body':
                orbiting = is_complex_orbit(positions, velocities)
            else:
                orbiting = is_orbiting(positions)

            attempts += 1
            logging.info(f"Spaceship {i}: Attempt {attempts}, Orbiting: {orbiting}")

        if not orbiting:
            logging.warning(f"Failed to generate valid trajectory for spaceship {i} after {attempts} attempts")

        all_positions.append(positions)
        all_velocities.append(velocities)

    all_positions = np.array(all_positions)
    all_velocities = np.array(all_velocities)

    total_timesteps = TOTAL_STEPS
    train_timesteps = int(total_timesteps * 0.6)
    val_timesteps = int(total_timesteps * 0.2)

    for dataset_type in ['train', 'val', 'test']:
        if dataset_type == 'train':
            positions = all_positions[:, :train_timesteps]
            velocities = all_velocities[:, :train_timesteps]
        elif dataset_type == 'val':
            positions = all_positions[:, train_timesteps:train_timesteps+val_timesteps]
            velocities = all_velocities[:, train_timesteps:train_timesteps+val_timesteps]
        else:  # test
            positions = all_positions[:, train_timesteps+val_timesteps:]
            velocities = all_velocities[:, train_timesteps+val_timesteps:]

        for spaceship_id in range(NUM_SIMULATIONS):
            data = []
            for timestep in range(positions.shape[1]):
                data.append({
                    'timestep': timestep,
                    'x': positions[spaceship_id, timestep, 0],
                    'y': positions[spaceship_id, timestep, 1],
                    'Vx': velocities[spaceship_id, timestep, 0],
                    'Vy': velocities[spaceship_id, timestep, 1]
                })

            df = pd.DataFrame(data)
            df.to_csv(f'datasets/{problem_type}/{dataset_type}/{spaceship_id}/{problem_type}_{dataset_type}.csv', index=False)
            logging.info(f"Saved dataset for {problem_type}, {dataset_type}, spaceship {spaceship_id}")

    return all_positions, all_velocities

# Add this new function
def plot_velocities(problem_type, all_velocities):
    logging.info(f"Plotting velocities for {problem_type}")
    plt.figure(figsize=(12, 10))

    train_timesteps = int(TOTAL_STEPS * 0.6)
    val_timesteps = int(TOTAL_STEPS * 0.2)
    # change line colors to blue for train, green for validation, and red for test
    for i in range(NUM_SIMULATIONS):
        velocities = np.linalg.norm(all_velocities[i], axis=1)
        plt.plot(velocities[:train_timesteps],'b-', label='Train' if i == 0 else "")
        plt.plot(range(train_timesteps, train_timesteps+val_timesteps),
                 velocities[train_timesteps:train_timesteps+val_timesteps],'g-', label='Validation' if i == 0 else "")
        plt.plot(range(train_timesteps+val_timesteps, TOTAL_STEPS),
                 velocities[train_timesteps+val_timesteps:],'r-', label='Test' if i == 0 else "")



    plt.title(f'Velocities for {problem_type.replace("_", " ").title()}')
    plt.xlabel('Time step')
    plt.ylabel('Velocity magnitude')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plots/trajectories/{problem_type}/all_velocities.png')
    plt.close()

# Add this new function
def plot_individual_velocity(problem_type, velocities, spaceship_id):
    logging.info(f"Plotting individual velocity for {problem_type}, spaceship {spaceship_id}")
    plt.figure(figsize=(12, 10))

    train_timesteps = int(TOTAL_STEPS * 0.6)
    val_timesteps = int(TOTAL_STEPS * 0.2)

    velocities = np.linalg.norm(velocities, axis=1)
    plt.plot(velocities[:train_timesteps],'b-', label='Train')
    plt.plot(range(train_timesteps, train_timesteps+val_timesteps),
             velocities[train_timesteps:train_timesteps+val_timesteps],'g-', label='Validation')
    plt.plot(range(train_timesteps+val_timesteps, TOTAL_STEPS),
             velocities[train_timesteps+val_timesteps:],'r-', label='Test')

    plt.title(f'Velocity for {problem_type.replace("_", " ").title()} - Spaceship {spaceship_id}')
    plt.xlabel('Time step')
    plt.ylabel('Velocity magnitude')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plots/trajectories/{problem_type}/spaceship_{spaceship_id}_velocity.png')
    plt.close()

def plot_trajectories(problem_type, all_positions):
    logging.info(f"Plotting trajectories for {problem_type}")
    plt.figure(figsize=(12, 10))

    train_timesteps = int(TOTAL_STEPS * 0.6)
    val_timesteps = int(TOTAL_STEPS * 0.2)

    for i in range(NUM_SIMULATIONS):
        plt.plot(all_positions[i, :train_timesteps, 0], all_positions[i, :train_timesteps, 1], 'b-', label='Train' if i == 0 else "")
        plt.plot(all_positions[i, train_timesteps:train_timesteps+val_timesteps, 0],
                 all_positions[i, train_timesteps:train_timesteps+val_timesteps, 1], 'g-', label='Validation' if i == 0 else "")
        plt.plot(all_positions[i, train_timesteps+val_timesteps:, 0],
                 all_positions[i, train_timesteps+val_timesteps:, 1], 'r-', label='Test' if i == 0 else "")

    if problem_type in ['two_body', 'two_body_force_increased_acceleration']:
        plt.scatter(0, 0, c='yellow', s=100, label='Heavy Body')
    else:  # three_body
        plt.scatter(HEAVY_BODY1_POSITION[0], HEAVY_BODY1_POSITION[1], c='yellow', s=100, label='Heavy Body 1')
        plt.scatter(HEAVY_BODY2_POSITION[0], HEAVY_BODY2_POSITION[1], c='orange', s=100, label='Heavy Body 2')

    plt.title(f'Trajectories for {problem_type.replace("_", " ").title()}')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(f'plots/trajectories/{problem_type}/all_trajectories.png')
    plt.close()

def plot_individual_trajectory(problem_type, positions, spaceship_id):
    logging.info(f"Plotting individual trajectory for {problem_type}, spaceship {spaceship_id}")
    plt.figure(figsize=(12, 10))

    train_timesteps = int(TOTAL_STEPS * 0.6)
    val_timesteps = int(TOTAL_STEPS * 0.2)

    plt.plot(positions[:train_timesteps, 0], positions[:train_timesteps, 1], 'b-', label='Train')
    plt.plot(positions[train_timesteps:train_timesteps+val_timesteps, 0],
             positions[train_timesteps:train_timesteps+val_timesteps, 1], 'g-', label='Validation')
    plt.plot(positions[train_timesteps+val_timesteps:, 0],
             positions[train_timesteps+val_timesteps:, 1], 'r-', label='Test')

    if problem_type in ['two_body', 'two_body_force_increased_acceleration']:
        plt.scatter(0, 0, c='yellow', s=100, label='Heavy Body')
    else:  # three_body
        plt.scatter(HEAVY_BODY1_POSITION[0], HEAVY_BODY1_POSITION[1], c='yellow', s=100, label='Heavy Body 1')
        plt.scatter(HEAVY_BODY2_POSITION[0], HEAVY_BODY2_POSITION[1], c='orange', s=100, label='Heavy Body 2')

    plt.title(f'Trajectory for {problem_type.replace("_", " ").title()} - Spaceship {spaceship_id}')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(f'plots/trajectories/{problem_type}/spaceship_{spaceship_id}_trajectory.png')
    plt.close()

def main():
    logging.info("Starting simulation process")
    create_folder_structure()
    for problem_type in ['two_body', 'two_body_force_increased_acceleration', 'three_body']:
        all_positions, all_velocities = create_datasets(problem_type)
        plot_trajectories(problem_type, all_positions)
        plot_velocities(problem_type, all_velocities)

        for spaceship_id in range(NUM_SIMULATIONS):
            plot_individual_trajectory(problem_type, all_positions[spaceship_id], spaceship_id)
            plot_individual_velocity(problem_type, all_velocities[spaceship_id], spaceship_id)

    logging.info("Simulation process completed")

if __name__ == "__main__":
    main()
