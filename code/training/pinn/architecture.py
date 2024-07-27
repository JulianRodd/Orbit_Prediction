import torch.nn as nn
from training.constants import DEVICE


class PINN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.1):
        super(PINN, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(4)]
        )
        self.output_layer = nn.Linear(hidden_size, output_size)

        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(hidden_size) for _ in range(5)]
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.batch_norms[0](self.input_layer(x)))
        x = self.dropout(x)

        for i, hidden_layer in enumerate(self.hidden_layers):
            residual = x
            x = self.activation(self.batch_norms[i + 1](hidden_layer(x)))
            x = self.dropout(x)
            x = x + residual

        return self.output_layer(x)


def create_pinn_model(input_size, hidden_size, output_size):
    return PINN(input_size, hidden_size, output_size).to(DEVICE)
