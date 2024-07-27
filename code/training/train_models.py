import logging
import traceback

import wandb
from tqdm import tqdm
from training.constants import NUM_FOLDS
from training.lstm.train import train_lstm
from training.mlp.train import train_mlp
from training.pinn.train import train_pinn
from training.utils import load_data, set_seed

logger = logging.getLogger(__name__)


def train_models(model_types, datasets, mini, seed, prediction_steps, use_wandb=True):
    set_seed(seed)
    logger.info(f"Starting training with seed {seed}")

    for model_type in model_types:
        for dataset_type in datasets:
            logger.info(f"Training {model_type} on {dataset_type}")
            print(f"\n{'#'*70}")
            print(f"# Model: {model_type:<20} Dataset: {dataset_type:<35} #")
            print(f"{'#'*70}")

            wandb_mode = "online" if use_wandb else "disabled"
            try:
                wandb.init(
                    project="orbital-prediction",
                    entity="radboud-mlip-10",
                    name=f"{model_type}_{dataset_type}",
                    group=f"{model_type}_{dataset_type}",
                    config={
                        "model_type": model_type,
                        "dataset_type": dataset_type,
                        "mini": mini,
                        "prediction_steps": prediction_steps,
                        "seed": seed,
                    },
                    mode=wandb_mode,
                )

                for fold in tqdm(range(NUM_FOLDS), desc="Processing folds"):
                    logger.debug(f"Processing fold {fold}")
                    try:
                        train_data, scaler = load_data(dataset_type, "train", fold)
                        val_data, _ = load_data(dataset_type, "val", fold)
                        logger.debug(f"Data loaded for fold {fold}")

                        if model_type == "MLP":
                            train_mlp(
                                train_data,
                                val_data,
                                scaler,
                                fold,
                                mini,
                                prediction_steps,
                                use_wandb,
                                dataset_type,
                            )
                        elif model_type == "LSTM":
                            train_lstm(
                                train_data,
                                val_data,
                                scaler,
                                fold,
                                mini,
                                prediction_steps,
                                use_wandb,
                                dataset_type,
                            )
                        elif model_type == "PINN":
                            train_pinn(
                                train_data,
                                val_data,
                                scaler,
                                fold,
                                mini,
                                prediction_steps,
                                use_wandb,
                                dataset_type,
                            )
                        else:
                            raise ValueError(f"Unknown model type: {model_type}")

                    except Exception as e:
                        logger.error(
                            f"Error in fold {fold} for {model_type} on {dataset_type}: {str(e)}"
                        )
                        logger.error(traceback.format_exc())
                        if use_wandb:
                            wandb.log(
                                {"error": str(e), "traceback": traceback.format_exc()}
                            )

            except Exception as e:
                logger.error(
                    f"Error in wandb initialization for {model_type} on {dataset_type}: {str(e)}"
                )
                logger.error(traceback.format_exc())

            finally:
                if use_wandb:
                    wandb.finish()

    logger.info("Training completed")
