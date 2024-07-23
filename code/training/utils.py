import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from training.constants import (
    BATCH_SIZE,
    DEVICE,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    N_STEPS,
)

import wandb

logger = logging.getLogger(__name__)


class OrbitDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = torch.FloatTensor(sequences).to(DEVICE)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        x = sequence[:-1].reshape(-1)
        y = sequence[-1]
        return x, y


def set_seed(seed):
    try:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info(f"Seed set to {seed}")
    except Exception as e:
        logger.error(f"Error setting seed: {str(e)}")
        raise


def load_data(dataset_type, split, fold=None):
    try:
        if split == "test":
            df = pd.read_csv(
                f"datasets/{dataset_type}/{split}/{dataset_type}_{split}.csv"
            )
        else:
            df = pd.read_csv(
                f"datasets/{dataset_type}/{split}/{dataset_type}_{split}_fold{fold}.csv"
            )

        scaler = MinMaxScaler()
        columns_to_scale = ["x", "y", "Vx", "Vy"]
        df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

        sequences = []
        for spaceship in df["spaceship_id"].unique():
            spaceship_data = df[df["spaceship_id"] == spaceship][columns_to_scale]
            for i in range(len(spaceship_data) - N_STEPS):
                sequences.append(spaceship_data.iloc[i : i + N_STEPS + 1].values)

        dataset = OrbitDataset(np.array(sequences))
        dataloader = DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=(split == "train")
        )

        logger.info(f"Data loaded for {dataset_type}, {split}, fold {fold}")
        logger.debug(f"Dataloader size: {len(dataloader)}")

        return dataloader, scaler
    except Exception as e:
        logger.error(
            f"Error loading data for {dataset_type}, {split}, fold {fold}: {str(e)}"
        )
        raise


def generic_train(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    fold,
    prediction_steps,
    scaler,
    model_type,
    use_wandb,
):
    logger.info(f"Starting training for {model_type} (Fold {fold})")
    best_val_loss = float("inf")
    early_stopping_counter = 0

    for epoch in tqdm(range(EPOCHS), desc=f"Training {model_type} (Fold {fold})"):
        try:
            model.train()
            train_loss = train_epoch(model, train_loader, optimizer, criterion)

            model.eval()
            val_loss = validate_epoch(model, val_loader, criterion)

            logger.debug(
                f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}"
            )

            if use_wandb:
                wandb.log(
                    {
                        f"train_loss_fold{fold}": train_loss,
                        f"val_loss_fold{fold}": val_loss,
                        "epoch": epoch,
                    }
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    model.state_dict(),
                    f"checkpoints/{model_type.lower()}_fold{fold}.pth",
                )
                early_stopping_counter = 0
                logger.info(f"New best model saved for {model_type} (Fold {fold})")
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
                    logger.info(
                        f"Early stopping triggered at epoch {epoch} for {model_type} (Fold {fold})"
                    )
                    break

        except Exception as e:
            logger.error(
                f"Error during training epoch {epoch} for {model_type} (Fold {fold}): {str(e)}"
            )
            raise

    # Make predictions
    try:
        model.load_state_dict(
            torch.load(f"checkpoints/{model_type.lower()}_fold{fold}.pth")
        )
        initial_sequence = next(iter(val_loader))[0][0].to(DEVICE)
        for steps in prediction_steps:
            try:
                predictions = predict_future(model, initial_sequence, scaler, steps)
                actual = scaler.inverse_transform(
                    val_loader.dataset.sequences[:steps, -1, :].cpu().numpy()
                )
                plot_predictions(
                    actual,
                    predictions,
                    f"{model_type}_{steps}_steps_fold{fold}",
                    use_wandb,
                )
            except Exception as e:
                logger.error(f"Error during prediction for {steps} steps: {str(e)}")
                logger.error(f"Skipping prediction for {steps} steps")
    except Exception as e:
        logger.error(
            f"Error during prediction for {model_type} (Fold {fold}): {str(e)}"
        )


def train_epoch(model, train_loader, optimizer, criterion):
    total_loss = 0
    for batch_x, batch_y in train_loader:
        try:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        except Exception as e:
            logger.error(f"Error during training step: {str(e)}")
            raise
    return total_loss / len(train_loader)


def validate_epoch(model, val_loader, criterion):
    total_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            try:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()
            except Exception as e:
                logger.error(f"Error during validation step: {str(e)}")
                raise
    return total_loss / len(val_loader)


def predict_future(model, initial_sequence, scaler, steps):
    try:
        model.eval()
        current_input = initial_sequence.clone().to(DEVICE)
        predictions = []

        with torch.no_grad():
            for _ in range(steps):
                # Ensure current_input is 2D: (1, input_size)
                if current_input.dim() == 1:
                    current_input = current_input.unsqueeze(0)

                output = model(current_input)

                # Ensure output is 1D
                if output.dim() == 2:
                    output = output.squeeze(0)

                predictions.append(output.cpu().numpy())

                # Update current_input by removing the oldest step and adding the new prediction
                current_input = torch.cat(
                    (current_input[:, 4:], output.unsqueeze(0)), dim=1
                )

        predictions = np.array(predictions)
        logger.debug(f"Future predictions made for {steps} steps")
        return scaler.inverse_transform(predictions)
    except Exception as e:
        logger.error(f"Error in predict_future: {str(e)}")
        logger.error(f"initial_sequence shape: {initial_sequence.shape}")
        logger.error(f"current_input shape: {current_input.shape}")
        logger.error(f"output shape: {output.shape}")
        raise


def plot_predictions(actual, predicted, title, use_wandb=True):
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(actual[:, 0], actual[:, 1], label="Actual", color="blue")
        plt.plot(
            predicted[:, 0],
            predicted[:, 1],
            label="Predicted",
            color="red",
            linestyle="--",
        )
        plt.title(title)
        plt.xlabel("X position")
        plt.ylabel("Y position")
        plt.legend()

        if use_wandb:
            wandb.log({f"{title}_plot": wandb.Image(plt)})
        plt.close()

        # Plot velocity
        plt.figure(figsize=(12, 6))
        actual_velocity = np.linalg.norm(actual[:, 2:], axis=1)
        predicted_velocity = np.linalg.norm(predicted[:, 2:], axis=1)
        plt.plot(actual_velocity, label="Actual", color="blue")
        plt.plot(predicted_velocity, label="Predicted", color="red", linestyle="--")
        plt.title(f"{title} - Velocity")
        plt.xlabel("Time step")
        plt.ylabel("Velocity magnitude")
        plt.legend()

        if use_wandb:
            wandb.log({f"{title}_velocity_plot": wandb.Image(plt)})
        plt.close()

        logger.info(f"Plots created for {title}")
    except Exception as e:
        logger.error(f"Error in plot_predictions for {title}: {str(e)}")
        raise
