import torch
import torch.nn.functional as F
from generics import M1, G


def energy_loss(y_pred, y_true, r):
    KE = 0.5 * (y_pred[:, 2] ** 2 + y_pred[:, 3] ** 2)  # Kinetic Energy
    PE = -G * M1 / r  # Potential Energy
    E_pred = KE + PE
    E_true = 0.5 * (y_true[:, 2] ** 2 + y_true[:, 3] ** 2) - G * M1 / r
    return torch.mean((E_pred - E_true) ** 2)


def momentum_loss(y_pred, y_true):
    p_pred = y_pred[:, 2:4]  # predicted velocities
    p_true = y_true[:, 2:4]  # true velocities
    return torch.mean((p_pred - p_true) ** 2)


def laplacian_loss(model, x):
    x = x.clone().detach().requires_grad_(True)  # Ensure x requires gradients
    y = model(x)
    dy_dx = torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y), create_graph=True
    )[0]
    d2y_dx2 = torch.autograd.grad(
        dy_dx, x, grad_outputs=torch.ones_like(dy_dx), create_graph=True
    )[0]
    return torch.mean(torch.sum(d2y_dx2, dim=1) ** 2)


def pinn_loss(model, x, y_true):
    y_pred = model(x)

    # MSE loss
    mse_loss = F.mse_loss(y_pred, y_true)

    # Simplified physics-inspired losses
    # Assuming y_pred[:, 0:2] are positions and y_pred[:, 2:4] are velocities

    # Energy conservation (simplified)
    energy_loss = torch.mean(torch.square(y_pred[:, 2:4]).sum(dim=1))

    # Momentum conservation (simplified)
    momentum_loss = torch.mean(torch.abs(y_pred[:, 2:4] - y_true[:, 2:4]))

    # Continuity constraint (simplified)
    continuity_loss = torch.mean(torch.abs(y_pred[:, 0:2] - y_true[:, 0:2]))

    # Total loss
    total_loss = mse_loss + 0.1 * (energy_loss + momentum_loss + continuity_loss)

    return total_loss, mse_loss, energy_loss, momentum_loss, continuity_loss
