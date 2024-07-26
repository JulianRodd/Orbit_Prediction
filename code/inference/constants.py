import training.lstm.constants as lstm_constants
import training.mlp.constants as mlp_constants
import training.pinn.constants as pinn_constants
# Model parameters
NUM_FOLDS = 5
MLP_INPUT_SIZE = mlp_constants.INPUT_SIZE
MLP_HIDDEN_SIZE = mlp_constants.HIDDEN_SIZE
MLP_OUTPUT_SIZE = mlp_constants.OUTPUT_SIZE
PINN_INPUT_SIZE = pinn_constants.INPUT_SIZE
PINN_HIDDEN_SIZE = pinn_constants.HIDDEN_SIZE
PINN_OUTPUT_SIZE = pinn_constants.OUTPUT_SIZE
LSTM_HIDDEN_SIZE = lstm_constants.HIDDEN_SIZE
LSTM_NUM_LAYERS = lstm_constants.NUM_LAYERS
LSTM_INPUT_SIZE = lstm_constants.INPUT_SIZE
LSTM_OUTPUT_SIZE = lstm_constants.OUTPUT_SIZE
# Inference parameters
DEFAULT_PREDICTION_STEPS = [10, 100, 500]

# Paths
CSV_OUTPUT_FOLDER = "output"
CHECKPOINT_BASE_PATH = "checkpoints"
MINI_CHECKPOINT_BASE_PATH = "mini_checkpoints"

# Heavy body positions (for plotting)
HEAVY_BODY1_POSITION = (10000, 0)
HEAVY_BODY2_POSITION = (-10000, 0)
