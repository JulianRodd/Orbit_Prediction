import logging

import numpy as np
from data_generation.constants import DT, M1, RELATIVE_UNCERTAINTY, G, UNCERTAINTY_PERIOD

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_initial_velocity(M, m, r):
    """Calculate initial velocity for a stable circular orbit."""
    try:
        return np.sqrt(G * M / r)
    except ValueError as e:
        logging.error(f"Error calculating initial velocity: {e}")
        raise


def gravitational_force(M, m, r1, r2):
    r = r2 - r1  # Position vector of the orbiting body relative to the central object
    r_magnitude = np.linalg.norm(r)  # Magnitude (distance) of the position vector r
    r_hat = r / r_magnitude  # Unit vector in the direction of r
    force_magnitude = (
        -G * M * m / r_magnitude**2
    )  # Magnitude of the gravitational force
    force_vector = force_magnitude * r_hat  # Gravitational force vector
    return force_vector



def additional_force(v, a_amplitude=5e-9):
    """Calculate additional sinusoidal force."""
    try:
        force_x = a_amplitude * np.sin(v[0])
        force_y = a_amplitude * np.sin(v[1])
        return np.array([force_x, force_y])
    except Exception as e:
        logging.error(f"Error calculating additional force: {e}")
        raise


def apply_periodic_uncertainty(
    value, step, period=UNCERTAINTY_PERIOD, relative_uncertainty=RELATIVE_UNCERTAINTY
):
    """Apply uncertainty periodically and smoothly interpolate between applications."""
    if not hasattr(apply_periodic_uncertainty, "current_uncertainty"):
        apply_periodic_uncertainty.current_uncertainty = np.zeros(2)
        apply_periodic_uncertainty.steps_since_last_application = 0

    if step % period == 0:
        magnitude = np.linalg.norm(value)
        uncertainty_magnitude = np.random.normal(0, relative_uncertainty * magnitude)
        uncertainty_direction = np.random.randn(2)
        uncertainty_direction /= np.linalg.norm(uncertainty_direction)
        apply_periodic_uncertainty.current_uncertainty = (
            uncertainty_magnitude * uncertainty_direction
        )
        apply_periodic_uncertainty.steps_since_last_application = 0

    interpolation_factor = (step % period) / period
    interpolated_uncertainty = (
        apply_periodic_uncertainty.current_uncertainty * interpolation_factor
    )

    apply_periodic_uncertainty.steps_since_last_application += 1

    return value + interpolated_uncertainty


def euler_update(r, v, F, m, step, dt=DT, increased_acceleration=0.00):
    """Update position and velocity using Euler method with periodic uncertainty application."""
    try:
        a = F / m
        if increased_acceleration:
            a += increased_acceleration * F / m
        v_next = v + a * dt
        r_next = r + v_next * dt  # Use current velocity for position update

        # # Apply uncertainties periodically
        r_next = apply_periodic_uncertainty(r_next, step)
        v_next = apply_periodic_uncertainty(v_next, step)

        return r_next, v_next
    except Exception as e:
        logging.error(f"Error in Euler update: {e}")
        raise


def debug_orbital_characteristics(r, v, F, m):
    distance = np.linalg.norm(r)
    speed = np.linalg.norm(v)
    potential_energy = -G * M1 * m / distance
    kinetic_energy = 0.5 * m * speed**2
    total_energy = potential_energy + kinetic_energy
    angular_momentum = np.cross(r, m * v)

    logging.debug(f"Distance: {distance}, Speed: {speed}")
    logging.debug(
        f"Potential Energy: {potential_energy}, Kinetic Energy: {kinetic_energy}"
    )
    logging.debug(f"Total Energy: {total_energy}, Angular Momentum: {angular_momentum}")


def is_orbiting(positions, threshold=0.1):
    """Check if the trajectory forms at least one complete orbit."""
    try:
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()

        x_range = x_max - x_min
        y_range = y_max - y_min

        if x_range < threshold * abs(x_max) or y_range < threshold * abs(y_max):
            return False

        x_changes = np.diff(np.sign(positions[:, 0] - positions[0, 0])).nonzero()[0]
        y_changes = np.diff(np.sign(positions[:, 1] - positions[0, 1])).nonzero()[0]

        return len(x_changes) >= 2 and len(y_changes) >= 2
    except Exception as e:
        logging.error(f"Error checking if orbiting: {e}")
        raise


def is_complex_orbit(
    positions, velocities, escape_threshold=1e6, direction_change_threshold=30
):
    """Check if the trajectory forms a complex, non-escaping orbit."""
    try:
        x, y = positions[:, 0], positions[:, 1]

        # Check if the trajectory is bounded
        max_range = max(np.max(x) - np.min(x), np.max(y) - np.min(y))
        if max_range > escape_threshold:
            return False

        # Check for complexity: look for changes in direction
        dx = np.diff(x)
        dy = np.diff(y)
        direction_changes = np.sum(np.abs(np.diff(np.arctan2(dy, dx))) > np.pi / 4)

        if direction_changes < direction_change_threshold:
            return False

        return True
    except Exception as e:
        logging.error(f"Error checking for complex orbit: {e}")
        raise
