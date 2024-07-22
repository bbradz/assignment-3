"""This module defines a utility function to discretize an environment state."""

import numpy as np


def discretize_state(state) -> int:
    """Discretize the given state into an integer based on the environment.

    The output integer will range from 1 to 500.

    :param      state       List containing state variables
        e.g., [cart_position, cart_velocity, pole_angle, pole_velocity]

    :returns    Integer representing the discretized state
    """
    if isinstance(state, int):
        return state

    state_ranges = [
        (-4.8, 4.8),  # cart_position
        (-3.4, 3.4),  # cart_velocity
        (-0.418, 0.418),  # pole_angle
        (-3.4, 3.4),  # pole_velocity
    ]

    bin_indices = []
    bins_per_variable = [4, 5, 5, 5]  # 4 * 5 * 5 * 5 = 500 bins total

    for value, (low, high), bins in zip(state, state_ranges, bins_per_variable):
        clipped_value = np.clip(value, low, high)
        bin_index = int((clipped_value - low) / (high - low) * (bins - 1))
        bin_indices.append(bin_index)

    discrete_state = 0
    for i, bin_index in enumerate(bin_indices):
        discrete_state *= bins_per_variable[i]
        discrete_state += bin_index

    return discrete_state + 1
