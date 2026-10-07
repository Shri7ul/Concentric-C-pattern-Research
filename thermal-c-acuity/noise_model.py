"""Noise models (NETD, temporal, spatial)."""
import numpy as np


def add_temperature_noise(
    temperature_map,
    noise_std
):
    """
    Add Gaussian temperature noise.

    noise_std:
        Standard deviation of temperature noise in °C.
    """

    noise = np.random.normal(
        loc=0.0,
        scale=noise_std,
        size=temperature_map.shape
    )

    noisy_temperature = temperature_map + noise

    return noisy_temperature