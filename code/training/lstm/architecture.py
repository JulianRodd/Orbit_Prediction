import torch.nn as nn
from training.constants import DEVICE

class OrbitLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(OrbitLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1, 4)  # Reshape input: (batch_size, sequence_length, features)
        lstm_out, _ = self.lstm(x)
        x = nn.functional.relu(self.fc1(lstm_out[:, -1, :]))
        return self.fc2(x)

def create_lstm_model(input_size, hidden_size, num_layers, output_size):
    return OrbitLSTM(input_size, hidden_size, num_layers, output_size).to(DEVICE)
