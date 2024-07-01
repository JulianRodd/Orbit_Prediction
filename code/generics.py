# generics.py

# Constants
import numpy as np
import torch

FIGURE_LOCATION = "code/images/"
CACHE_FOLDER = "code/cache/"
CHECKPOINT_LOCATION = "code/checkpoints/"
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
M1 = 1e9  # Mass of the first heavy object (kg)
M2 = 1e9  # Mass of the second heavy object (kg) - for 3-body problem
m = 1e4  # Mass of the orbiting object (kg)
R = 10000  # Initial radius (m)
DT = 10000  # Time step (s)
# Constants
INITIAL_POSITION_TWO_BODY = np.array([R, 0.0])
INITIAL_POSITION_THREE_BODY = np.array([0.0, 30500.0 * 0.98])
HEAVY_BODY1_POSITION = np.array([10000.0, 0.0])
HEAVY_BODY2_POSITION = np.array([-10000.0, 0.0])
BASE_RELATIVE_UNCERTAINTY = 0.00
RELATIVE_UNCERTAINTY = 0.01
SMALL_RELATIVE_UNCERTAINTY_ADJUSTMENT_STEP = 100
BASE_SMALL_ACCELARATION_CHANGE = 0.00
SMALL_ACCELARATION_CHANGE = 0.01
SLIGHT_Y_ADJUSTMENT_STEP = 1000
SLIGHT_Y_ADJUSTMENT = 0
SMALL_ACCELARATION_ADJUSTMENT_STEP = 1000
STARTING_POSITION = torch.tensor([R, 0.0])  # Starting position of the orbiting body
# Important: The total simulated time (over
# all time steps) should be selected so that the object rotates at least ten times around the heavy body.
# Create a plot of the moving body as a function of the two-dimensional coordinates and a plot for the
# absolute value of the velocity as a function of time.
TOTAL_STEPS = 100000  # Total number of time steps

DEVICE = "mps"
INPUT_SIZE = 12
OUTPUT_SIZE = 4
MIN_LR = 1e-6
HIDDEN_SIZE = 512
NUM_LAYERS = 6
DROPOUT_RATE = 0.1
SUBSET_SIZE = None
BATCH_SIZE = 16 if SUBSET_SIZE != None and SUBSET_SIZE <= 3000 else 32
EPOCHS = 15 if SUBSET_SIZE != None and SUBSET_SIZE <= 3000 else 100
LEARNING_RATE = 0.001
EARLY_STOPPING = True
PATIENCE = 10  # or any other value you prefer
NUM_LSTM_LAYERS = 3
BIDIRECTIONAL = True
ATTENTION_HEADS = 4
FC_LAYERS = 3
ACTIVATION = "relu"
USE_LAYER_NORM = True
USE_BATCH_NORM = True
# New configuration constants
NUM_HIDDEN_LAYERS = 3
ACTIVATION = "relu"
HIDDEN_SIZE_REDUCTION_FACTOR = 2
RESIDUAL_CONNECTIONS = False
PHYSICS_LOSS_WEIGHTS = {
    "mse": 1.0,
    "energy": 0.1,
    "momentum": 0.1,
    "position": 0.1,
    "velocity": 0.1,
}
TIME_STEP = 1.0
N_STEPS = 3

NUM_SIMULATIONS = 15
