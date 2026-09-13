"""Affine PPO arm and unchanged tested v1.3 nonlinear actor-critics."""

import numpy as np

from flyarcade_v13.policy import ActorCritic as NonlinearActorCritic
from flyarcade_v13.policy import orthogonal


class LinearActorCritic:
    arch = "linear"
    hidden = core = 0
    recurrent = False

    def __init__(self, input_width, actions, *, seed=0, **kwargs):
        self.input_width, self.actions = int(input_width), int(actions)
        rng = np.random.default_rng(seed)
        self.params = {
            "Wa": orthogonal(rng, actions, input_width, 0.01),
            "ba": np.zeros(actions),
            "Wv": orthogonal(rng, 1, input_width, 1.0),
            "bv": np.zeros(1),
        }

    def initial_state(self, batch):
        return None

    def copy_params(self):
        return {k: v.copy() for k, v in self.params.items()}

    def finite(self):
        return all(np.isfinite(v).all() for v in self.params.values())

    def forward(self, x, *args):
        p = self.params
        return x @ p["Wa"].T + p["ba"], (x @ p["Wv"].T + p["bv"])[:, 0], x

    def step(self, x, state=None, reset=None):
        logits, values, _ = self.forward(x)
        return logits, values, None

    def backward(self, x, dlogits, dvalues):
        return {
            "Wa": dlogits.T @ x,
            "ba": dlogits.sum(0),
            "Wv": dvalues[None, :] @ x,
            "bv": np.array([dvalues.sum()]),
        }


def ActorCritic(input_width, actions, *, arch="mlp", **kwargs):
    if arch == "linear":
        return LinearActorCritic(input_width, actions, **kwargs)
    return NonlinearActorCritic(input_width, actions, arch=arch, **kwargs)
