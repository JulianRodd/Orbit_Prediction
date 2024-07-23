from training.constants import N_STEPS

# MLP architecture parameters
INPUT_SIZE = 4 * N_STEPS  # 4 features (x, y, vx, vy) for each time step
HIDDEN_SIZE = 256
OUTPUT_SIZE = 4  # Predicting x, y, vx, vy for the next time step

# Mini-model parameters
MINI_HIDDEN_SIZE = 64
