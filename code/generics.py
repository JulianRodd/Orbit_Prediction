# generics.py

import numpy as np
import torch

# Folders
FIGURE_LOCATION = "images/"
CACHE_FOLDER = "cache/"
CHECKPOINT_LOCATION = "checkpoints/"
MINI_CHECKPOINT_LOCATION = "mini_checkpoints/"

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
SMALL_ACCELARATION_CHANGE = 0.01
TOTAL_STEPS = 100000  # Total number of time steps

# Device configuration
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# Dataset parameters
N_STEPS = 3
NUM_SIMULATIONS = 10

# Model parameters
INPUT_SIZE = 12  # 3 time steps, each with x, y, vx, vy
N_STEPS = 3
OUTPUT_SIZE = 4
HIDDEN_SIZE = 256
NUM_LAYERS = 6
DROPOUT_RATE = 0.1
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 10
NUM_LSTM_LAYERS = 3
BIDIRECTIONAL = True
 


# Mini network parameters
MINI_HIDDEN_SIZE = 64
MINI_NUM_LAYERS = 2
MINI_EPOCHS = 3

# Cross-validation parameters
NUM_FOLDS = 5
# Remove unused constants
# (BASE_RELATIVE_UNCERTAINTY, SMALL_RELATIVE_UNCERTAINTY_ADJUSTMENT_STEP,
# BASE_SMALL_ACCELARATION_CHANGE, SLIGHT_Y_ADJUSTMENT_STEP, SLIGHT_Y_ADJUSTMENT,
# SMALL_ACCELARATION_ADJUSTMENT_STEP, MIN_LR, SUBSET_SIZE,
# ATTENTION_HEADS, FC_LAYERS, USE_LAYER_NORM, USE_BATCH_NORM,
# NUM_HIDDEN_LAYERS, HIDDEN_SIZE_REDUCTION_FACTOR, RESIDUAL_CONNECTIONS,
# PHYSICS_LOSS_WEIGHTS, TIME_STEP)
