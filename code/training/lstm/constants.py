from training.constants import N_STEPS

# LSTM architecture parameters
INPUT_SIZE = 4  # 4 features (x, y, vx, vy)
HIDDEN_SIZE = 128
NUM_LAYERS = 2
OUTPUT_SIZE = 4  # Predicting x, y, vx, vy for the next time step

# Mini-model parameters
MINI_HIDDEN_SIZE = 32
MINI_NUM_LAYERS = 1
