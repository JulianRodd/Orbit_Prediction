import logging
import os

import torch
import torch.optim as optim
from tqdm import tqdm
from training.constants import DEVICE, EARLY_STOPPING_PATIENCE, EPOCHS, LEARNING_RATE
from training.pinn.architecture import create_pinn_model
from training.pinn.constants import HIDDEN_SIZE, INPUT_SIZE, OUTPUT_SIZE
from training.pinn.pinn_loss import pinn_loss
from training.utils import plot_predictions, predict_future

import wandb

logger = logging.getLogger(__name__)


def train_pinn(
    train_loader, val_loader, scaler, fold, is_mini, prediction_steps, use_wandb, dataset_type
):
    try:
        logger.info(f"Initializing PINN model for fold {fold}")
        model = create_pinn_model(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        best_val_loss = float("inf")
        early_stopping_counter = 0

        for epoch in tqdm(range(EPOCHS), desc=f"Training PINN (Fold {fold})"):
            try:
                model.train()
                train_loss, train_mse_loss, train_physics_loss = train_pinn_epoch(
                    model, train_loader, optimizer
                )

                model.eval()
                val_loss, val_mse_loss, val_physics_loss = validate_pinn_epoch(
                    model, val_loader
                )

                logger.debug(
                    f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}"
                )

                if use_wandb:
                    wandb.log(
                        {
                            f"train_loss_fold{fold}": train_loss,
                            f"train_mse_loss_fold{fold}": train_mse_loss,
                            f"train_physics_loss_fold{fold}": train_physics_loss,
                            f"val_loss_fold{fold}": val_loss,
                            f"val_mse_loss_fold{fold}": val_mse_loss,
                            f"val_physics_loss_fold{fold}": val_physics_loss,
                            "epoch": epoch,
                        }
                    )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    os.makedirs(f"checkpoints/pinn/{dataset_type}/", exist_ok=True)
                    torch.save(model.state_dict(), f"checkpoints/pinn/{dataset_type}/pinn_fold{fold}.pth")
                    early_stopping_counter = 0
                    logger.info(f"New best model saved for PINN (Fold {fold})")
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
                        logger.info(
                            f"Early stopping triggered at epoch {epoch} for PINN (Fold {fold})"
                        )
                        break

            except Exception as e:
                logger.error(
                    f"Error during PINN training epoch {epoch} for fold {fold}: {str(e)}"
                )
                raise

        # Make predictions
        try:
            model.load_state_dict(torch.load(f"checkpoints/pinn_fold{fold}.pth"))
            initial_sequence = next(iter(val_loader))[0][0].to(DEVICE)
            for steps in prediction_steps:
                predictions = predict_future(model, initial_sequence, scaler, steps)
                actual = scaler.inverse_transform(
                    val_loader.dataset.sequences[:steps, -1, :].cpu().numpy()
                )
                plot_predictions(
                    actual, predictions, f"PINN_{steps}_steps_fold{fold}", use_wandb
                )
        except Exception as e:
            logger.error(f"Error during PINN prediction for fold {fold}: {str(e)}")
            raise

    except Exception as e:
        logger.error(f"Error in PINN training for fold {fold}: {str(e)}")
        raise


def train_pinn_epoch(model, train_loader, optimizer):
    total_loss = total_mse_loss = total_physics_loss = 0
    for batch_x, batch_y in train_loader:
        try:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss, mse_loss, physics_loss = pinn_loss(outputs, batch_y, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_mse_loss += mse_loss.item()
            total_physics_loss += physics_loss.item()
        except Exception as e:
            logger.error(f"Error during PINN training step: {str(e)}")
            raise
    return (
        total_loss / len(train_loader),
        total_mse_loss / len(train_loader),
        total_physics_loss / len(train_loader),
    )


def validate_pinn_epoch(model, val_loader):
    total_loss = total_mse_loss = total_physics_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            try:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                outputs = model(batch_x)
                loss, mse_loss, physics_loss = pinn_loss(outputs, batch_y, batch_x)
                total_loss += loss.item()
                total_mse_loss += mse_loss.item()
                total_physics_loss += physics_loss.item()
            except Exception as e:
                logger.error(f"Error during PINN validation step: {str(e)}")
                raise
    return (
        total_loss / len(val_loader),
        total_mse_loss / len(val_loader),
        total_physics_loss / len(val_loader),
    )
