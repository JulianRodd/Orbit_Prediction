import torch.nn as nn
import torch.optim as optim
import logging
from training.constants import LEARNING_RATE
from training.mlp.architecture import create_mlp_model
from training.mlp.constants import HIDDEN_SIZE, INPUT_SIZE, OUTPUT_SIZE
from training.utils import generic_train

logger = logging.getLogger(__name__)

def train_mlp(train_loader, val_loader, scaler, fold, is_mini, prediction_steps, use_wandb, dataset_type):
    try:
        logger.info(f"Initializing MLP model for fold {fold}")
        model = create_mlp_model(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.MSELoss()

        generic_train(model, train_loader, val_loader, optimizer, criterion, fold, prediction_steps, scaler, "MLP", use_wandb, dataset_type)
    except Exception as e:
        logger.error(f"Error in MLP training for fold {fold}: {str(e)}")
        raise
