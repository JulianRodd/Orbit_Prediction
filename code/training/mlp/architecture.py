import torch.nn as nn
from training.constants import DEVICE


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.dropout(nn.functional.relu(self.batch_norm1(self.fc1(x))))
        x = self.dropout(nn.functional.relu(self.batch_norm2(self.fc2(x))))
        return self.fc3(x)


def create_mlp_model(input_size, hidden_size, output_size):
    return MLP(input_size, hidden_size, output_size).to(DEVICE)
