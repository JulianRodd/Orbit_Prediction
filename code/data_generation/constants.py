import numpy as np

# Physical constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
M1 = 1e9  # Mass of the first heavy object (kg)
M2 = 1e9  # Mass of the second heavy object (kg) - for 3-body problem
m = 1e4  # Mass of the orbiting object (kg)
R = 10000  # Initial radius (m)
DT = 10000  # Time step (s)

# Initial conditions
INITIAL_POSITION_TWO_BODY = np.array([R, 0.0])
INITIAL_POSITION_THREE_BODY = np.array([0.0, 30500.0 * 0.98])
HEAVY_BODY1_POSITION = np.array([10000.0, 0.0])
HEAVY_BODY2_POSITION = np.array([-10000.0, 0.0])

# Simulation parameters
RELATIVE_UNCERTAINTY = 0.01
SMALL_ACCELERATION_CHANGE = 0.1
TOTAL_STEPS = 10000  # Total number of time steps
UNCERTAINTY_PERIOD = DT * TOTAL_STEPS / 10
SMALL_ACCELERATION_CHANGE_PERIOD = DT * TOTAL_STEPS / 1000
ADDITIONAL_FORCE_AMPLITUDE = 5e-9
# Dataset parameters
N_STEPS = 3
NUM_SIMULATIONS = 10

# Model parameters
INPUT_SIZE = 12  # 3 time steps, each with x, y, vx, vy
OUTPUT_SIZE = 4
HIDDEN_SIZE = 256
NUM_LAYERS = 6
DROPOUT_RATE = 0.1
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 10

# Cross-validation parameters
NUM_FOLDS = 5
