# Orbital Prediction Models

This project implements various deep learning models for predicting orbital trajectories in two-body and three-body gravitational problems.

## Installation

1. Ensure you have Python 3.11 installed on your system.

2. Clone this repository:
git clone [your-repository-url]
cd [repository-name]

3. Install the required packages:
pip install -r requirements.txt

## Usage

The main script `code/main.py` supports three modes of operation: data generation, training, and inference.

### Data Generation

To generate data for the orbital prediction models:
python code/main.py generate_data

### Training

To train the models:
python code/main.py train [OPTIONS]

Options:
- `--model_types`: Space-separated list of model types to train (default: MLP LSTM PINN)
- `--datasets`: Space-separated list of datasets to use (default: two_body two_body_force_increased_acceleration three_body)
- `--seed`: Random seed (default: 42)
- `--use_wandb`: Use Weights & Biases for logging
- `--prediction_steps`: Space-separated list of prediction step numbers (default: 10 100 500)

Example:
python code/main.py train --model_types MLP LSTM --datasets two_body --seed 123 --use_wandb --prediction_steps 10 50

### Inference

To run inference on trained models:
python code/main.py inference [OPTIONS]

Options:
- `--model_types`: Space-separated list of model types to use for inference
- `--datasets`: Space-separated list of datasets to use
- `--prediction_steps`: Space-separated list of prediction step numbers
- `--output_folder`: Folder to save inference results (default: inference_results)

Example:
python code/main.py inference --model_types PINN --datasets three_body  --prediction_steps 10 100 --output_folder results


## Additional Information

- The `MODEL_TYPES` constant includes MLP, LSTM, and PINN.
- The `DATASET_TYPES` constant includes two_body, two_body_force_increased_acceleration, and three_body.
- Default prediction steps are 10, 100, and 500.

For more detailed information about the project structure and implementation details, please refer to the source code and comments within each file.
