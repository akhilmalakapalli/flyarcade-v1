"""Artificial motor readout trained with reward-modulated policy eligibility."""

import numpy as np


class Motor:
    def __init__(self, width, rng):
        self.weights = rng.normal(0, 0.01, (3, width + 1))
        self.reset()

    def reset(self):
        self.eligibility = np.zeros_like(self.weights)

    def choose(self, rates, rng, training):
        x = np.append(rates, 1)
        logits = self.weights @ x
        p = np.exp(logits - logits.max())
        p /= p.sum()
        action = int(rng.choice(3, p=p))
        if training:
            residual = -p
            residual[action] += 1
            self.eligibility = 0.5 * self.eligibility + np.outer(residual, x)
        return action

    def reward(self, advantage):
        self.weights = np.clip(self.weights + 0.01 * advantage * self.eligibility, -4, 4)
