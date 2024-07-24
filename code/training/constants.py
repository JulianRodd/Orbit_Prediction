import logging

from constants import DEVICE

# Set up logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Model and dataset types
MODEL_TYPES = ["MLP", "LSTM", "PINN"]
DATASET_TYPES = ["two_body", "two_body_force_increased_acceleration", "three_body"]


logger.info(f"Using device: {DEVICE}")

# Data parameters
N_STEPS = 3
BATCH_SIZE = 32

# Training parameters
NUM_FOLDS = 5
LEARNING_RATE = 0.01
EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5

# Folders
CHECKPOINT_LOCATION = "checkpoints/"
FIGURE_LOCATION = "plots/"

# Physical constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
M1 = 1e9  # Mass of the central body (kg)
DT = 10000  # Time step (s)
