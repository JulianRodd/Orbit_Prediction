python code/test_inference.py --model_types PINN SimpleRegression LSTM --dataset_type two_body --num_folds 5 --prediction_steps 10 100 500

python code/regression.py --model_types PINN SimpleRegression LSTM --datasets two_body three_body --mini



python code/inference.py --model_types PINN SimpleRegression LSTM --datasets two_body three_body --folds 0 1 2 3 4 --mini


python code/inference.py --model_types SimpleRegression --datasets two_body three_body --folds 0 1 2 3 4
