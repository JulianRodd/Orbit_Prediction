import torch.nn as nn
import torch.nn.functional as F
from training.constants import DEVICE


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.batch_norm3 = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)

        # First layer
        x = self.dropout(F.leaky_relu(self.batch_norm1(self.fc1(x))))

        # Second layer with residual connection
        residual = x
        x = self.dropout(F.leaky_relu(self.batch_norm2(self.fc2(x))))
        x = x + residual

        # Third layer
        x = self.dropout(F.leaky_relu(self.batch_norm3(self.fc3(x))))

        # Output layer
        return self.fc4(x)


def create_mlp_model(input_size, hidden_size, output_size, dropout_rate=0.2):
    return MLP(input_size, hidden_size, output_size, dropout_rate).to(DEVICE)
