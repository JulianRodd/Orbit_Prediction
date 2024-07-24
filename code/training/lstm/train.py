import torch.nn as nn
import torch.optim as optim
import logging
from training.constants import LEARNING_RATE
from training.lstm.architecture import create_lstm_model
from training.lstm.constants import HIDDEN_SIZE, INPUT_SIZE, NUM_LAYERS, OUTPUT_SIZE
from training.utils import generic_train

logger = logging.getLogger(__name__)

def train_lstm(train_loader, val_loader, scaler, fold, is_mini, prediction_steps, use_wandb, dataset_type):
    try:
        logger.info(f"Initializing LSTM model for fold {fold}")
        model = create_lstm_model(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.MSELoss()

        generic_train(model, train_loader, val_loader, optimizer, criterion, fold, prediction_steps, scaler, "LSTM", use_wandb, dataset_type)
    except Exception as e:
        logger.error(f"Error in LSTM training for fold {fold}: {str(e)}")
        raise
