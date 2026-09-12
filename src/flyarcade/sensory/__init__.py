"""Engineered observation features; not a reconstruction of fly vision."""

import numpy as np


def features(observation):
    observation = np.asarray(observation, dtype=float)
    if observation.shape != (3,) or not np.isfinite(observation).all():
        raise ValueError("three finite normalized observations required")
    player, target, phase = np.clip(observation, 0, 1)
    error = target - player
    return np.array(
        [
            max(error, 0),
            max(-error, 0),
            float(abs(error) < 0.125),
            player,
            1 - player,
            target,
            1 - target,
            phase,
        ]
    )
