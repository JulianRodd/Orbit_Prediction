import torch
from training.constants import M1, G


def pinn_loss(y_pred, y_true, x):
    mse_loss = torch.nn.functional.mse_loss(y_pred, y_true)

    # Physics-based loss components
    r = torch.sqrt(
        y_pred[:, 0] ** 2 + y_pred[:, 1] ** 2 + 1e-8
    )  # Add small epsilon to avoid division by zero
    v = torch.sqrt(y_pred[:, 2] ** 2 + y_pred[:, 3] ** 2 + 1e-8)

    # Energy conservation
    KE = 0.5 * v**2
    PE = -G * M1 / r
    E_pred = KE + PE
    E_true = 0.5 * (y_true[:, 2] ** 2 + y_true[:, 3] ** 2) - G * M1 / torch.sqrt(
        y_true[:, 0] ** 2 + y_true[:, 1] ** 2 + 1e-8
    )
    energy_loss = torch.mean((E_pred - E_true) ** 2)

    # Momentum conservation
    p_pred = y_pred[:, 2:4]
    p_true = y_true[:, 2:4]
    momentum_loss = torch.mean((p_pred - p_true) ** 2)

    physics_loss = energy_loss + momentum_loss

    # Use log1p for smoother loss scaling
    total_loss = torch.log1p(mse_loss) + 0.1 * torch.log1p(physics_loss)

    return total_loss, mse_loss, physics_loss
