import training.lstm.constants as lstm_constants

# Model parameters
NUM_FOLDS = 5
INPUT_SIZE = 12  # 3 time steps * 4 features (x, y, vx, vy)
HIDDEN_SIZE = 256
LSTM_HIDDEN_SIZE = lstm_constants.HIDDEN_SIZE
OUTPUT_SIZE = 4
NUM_LSTM_LAYERS = lstm_constants.NUM_LAYERS
LSTM_INPUT_SIZE = lstm_constants.INPUT_SIZE
LSTM_OUTPUT_SIZE = lstm_constants.OUTPUT_SIZE
# Inference parameters
DEFAULT_PREDICTION_STEPS = [10, 100, 500]

# Paths
CHECKPOINT_BASE_PATH = "checkpoints"
MINI_CHECKPOINT_BASE_PATH = "mini_checkpoints"

# Heavy body positions (for plotting)
HEAVY_BODY1_POSITION = (10000, 0)
HEAVY_BODY2_POSITION = (-10000, 0)
