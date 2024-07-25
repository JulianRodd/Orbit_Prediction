import torch.nn as nn
import torch.nn.functional as F
from training.constants import DEVICE


class LSTM(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.2
    ):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate,
        )

        self.layer_norm = nn.LayerNorm(hidden_size * 2)  # *2 for bidirectional

        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.view(
            x.size(0), -1, 4
        )  # Reshape input: (batch_size, sequence_length, features)

        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)

        # Use the output from the last time step
        lstm_out = lstm_out[:, -1, :]

        # Fully connected layers with residual connection
        residual = lstm_out
        x = F.relu(self.fc1(lstm_out))
        x = self.dropout(x)
        x = F.relu(self.fc2(x) + residual[:, : self.hidden_size])  # Residual connection
        x = self.dropout(x)

        return self.fc3(x)


def create_lstm_model(
    input_size, hidden_size, num_layers, output_size, dropout_rate=0.2
):
    return LSTM(input_size, hidden_size, num_layers, output_size, dropout_rate).to(
        DEVICE
    )
