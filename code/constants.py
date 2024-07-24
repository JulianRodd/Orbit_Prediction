# Device configuration
import torch

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)

# Folders
FIGURE_LOCATION = "images/"
CACHE_FOLDER = "cache/"
CHECKPOINT_LOCATION = "checkpoints/"
