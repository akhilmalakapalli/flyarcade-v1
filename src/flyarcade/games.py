"""Seeded headless lane arcade tasks with identical observation/action spaces."""

import numpy as np


class LaneGame:
    """Move left/stay/right to catch or dodge falling objects on five lanes."""

    def __init__(self, task="catch", seed=0, horizon=24):
        if task not in ("catch", "dodge") or horizon <= 0:
            raise ValueError("valid task and positive horizon required")
        self.task, self.horizon = task, horizon
        self.rng = np.random.default_rng(seed)
        self.player = 2
        self.object = int(self.rng.integers(0, 5))
        self.time = 0
        self.phase = 0
        self.done = False

    def observe(self):
        return np.array([self.player / 4, self.object / 4, self.phase / 3], dtype=float)

    def step(self, action):
        if type(action) is not int or action not in (0, 1, 2) or self.done:
            raise ValueError("valid action required before episode termination")
        before = abs(self.player - self.object)
        self.player = int(np.clip(self.player + action - 1, 0, 4))
        after = abs(self.player - self.object)
        # Dense reward helps credit assignment; raw hit/survival score is separate.
        reward = 0.1 * (before - after) * (1 if self.task == "catch" else -1)
        score = 0
        self.phase += 1
        if self.phase == 4:
            hit = self.player == self.object
            success = hit if self.task == "catch" else not hit
            score = int(success)
            reward += 1 if success else -1
            self.object = int(self.rng.integers(0, 5))
            self.phase = 0
        self.time += 1
        self.done = self.time >= self.horizon
        return self.observe(), float(reward), self.done, {"score": score}


def heuristic(task, observation):
    error = observation[1] - observation[0]
    if task == "catch":
        return 2 if error > 0 else 0 if error < 0 else 1
    if error != 0:
        return 1
    return 2 if observation[0] < 0.5 else 0
