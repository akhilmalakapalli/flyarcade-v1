"""v1.3 task access: unchanged v1.2 targets, reference policies, curricula and shaping.

The final evaluation environments are the v1.2 classes themselves, imported rather
than copied, so their dynamics, observations and scores are identical by
construction. Curriculum levels are training-only subclasses that keep the
observation width and action set of their target.
"""

import numpy as np

from flyarcade.v12.environments import Breakout, Flappy, Pong, Snake

TARGETS = {"flappy": Flappy, "pong": Pong, "breakout": Breakout, "snake": Snake}
TASK_ORDER = tuple(TARGETS)


# ---------------------------------------------------------------- Flappy oracle
def _flappy_step(y, vy, action):
    vy = 0.055 if action else max(-0.08, vy - 0.009)
    return y + vy, vy


def _steps_to_crossing(x, speed=0.04):
    """Replicates the environment's floating-point obstacle update exactly."""
    steps = 0
    while True:
        x -= speed
        steps += 1
        if x <= 0:
            return steps


def flappy_oracle(env):
    """Short-horizon model-predictive controller using the known deterministic dynamics.

    It plans only to the crossing of the obstacle already on screen, whose gap is
    observable, and never reads the environment RNG, so it uses no future randomness.
    Reachable (height, velocity) states are expanded breadth-first with exact
    deduplication, boundary-violating branches are pruned, and the first action of a
    branch with the smallest terminal gap error is returned (ties prefer not flapping).
    """
    frontier = {(env.y, env.vy): None}
    for _ in range(_steps_to_crossing(env.x, getattr(env, "speed", 0.04))):
        following = {}
        for (y, vy), first in frontier.items():
            for action in (0, 1):
                y2, vy2 = _flappy_step(y, vy, action)
                if y2 <= 0.025 or y2 >= 0.975:
                    continue
                key = (y2, vy2)
                if key not in following:
                    following[key] = action if first is None else first
        if not following:
            return 0
        frontier = following
    best = {}
    for (y, _), first in frontier.items():
        error = abs(y - env.gap)
        best[first] = min(error, best.get(first, np.inf))
    return int(min(best, key=lambda a: (best[a], a)))


def reference_action(env):
    """The competent reference policy used for gap closure and diagnostic labels."""
    if isinstance(env, Flappy):
        return flappy_oracle(env)
    return int(env.heuristic())


# ---------------------------------------------------------------- curricula
class FlappyCurriculum(Flappy):
    """Wider pipe tolerance and slower obstacles; identical observation layout."""

    def __init__(self, seed=0, tolerance=0.16, speed=0.04):
        self.tolerance, self.speed = float(tolerance), float(speed)
        super().__init__(seed)

    def step(self, action):
        self.validate(action)
        old_error = abs(self.y - self.gap)
        self.vy = 0.055 if action else max(-0.08, self.vy - 0.009)
        self.y += self.vy
        self.x -= self.speed
        reward = 0.01 + 0.02 * (old_error - abs(self.y - self.gap))
        if self.y <= 0.025 or self.y >= 0.975:
            self.done, self.death, reward = True, "boundary", -1.0
        elif self.x <= 0:
            if abs(self.y - self.gap) > self.tolerance:
                self.done, self.death, reward = True, "pipe", -1.0
            else:
                self.passed += 1
                reward += 1
                self.x = 0.9
                self.gap = float(self.rng.uniform(0.25, 0.75))
        return self.finish(reward)


class BreakoutCurriculum(Breakout):
    """Fewer live bricks (seeded positions); destroyed slots start already cleared."""

    def __init__(self, seed=0, bricks=5):
        self.live = int(bricks)
        super().__init__(seed)

    def _reset(self):
        super()._reset()
        if self.live < 5:
            keep = self.rng.choice(5, self.live, replace=False)
            self.bricks[:] = False
            self.bricks[keep] = True


class SnakeCurriculum(Snake):
    """Smaller square board with the same relative actions and 17 observations."""

    def __init__(self, seed=0, size=6):
        self.size = int(size)
        super().__init__(seed)

    def _reset(self):
        super()._reset()
        if self.size != 6:
            head = self.size // 2
            self.body = [(head, head), (head - 1, head), (head - 2, head)]
            self.food = None
            self._food()


LEVELS = {
    "flappy": {
        "target": {},
        "wide": {"tolerance": 0.24, "speed": 0.03},
        "intermediate": {"tolerance": 0.20, "speed": 0.035},
    },
    "pong": {"target": {}},
    "breakout": {"target": {}, "bricks1": {"bricks": 1}, "bricks3": {"bricks": 3}},
    "snake": {"target": {}, "size4": {"size": 4}, "size5": {"size": 5}},
}
CURRICULUM_CLASSES = {
    "flappy": FlappyCurriculum,
    "pong": Pong,
    "breakout": BreakoutCurriculum,
    "snake": SnakeCurriculum,
}


def make_env(task, seed, level="target"):
    """Target level returns the exact frozen v1.2 class, never a curriculum subclass."""
    if level == "target":
        return TARGETS[task](seed)
    return CURRICULUM_CLASSES[task](seed, **LEVELS[task][level])


# ---------------------------------------------------------------- shaping
def _flappy_predicted_height(env):
    y, vy = env.y, env.vy
    for _ in range(_steps_to_crossing(env.x, getattr(env, "speed", 0.04))):
        y, vy = _flappy_step(y, vy, 0)
    return float(np.clip(y, 0, 1))


def potential(env):
    """Phi(s) for potential-based shaping F = gamma * Phi(s') - Phi(s).

    Every potential is a function of state only; none encodes a correct action.
    """
    if isinstance(env, Flappy):
        return -abs(_flappy_predicted_height(env) - env.gap)
    if isinstance(env, Breakout):
        return -abs(env.paddle - env.intercept()) if env.vy < 0 else 0.0
    if isinstance(env, Pong):
        return -abs(env.paddle - env.intercept())
    if isinstance(env, Snake):
        if env.food is None:
            return 0.0
        (x, y), (fx, fy) = env.body[0], env.food
        return -(abs(fx - x) + abs(fy - y)) / (2 * (env.size - 1))
    raise TypeError("unknown environment")


def terminal_kind(env):
    """'terminal' for a genuine end state, 'truncated' for a pure time-limit stop."""
    if not env.done:
        return None
    if env.time >= env.horizon:
        genuine = (
            getattr(env, "death", None) is not None
            or (isinstance(env, Breakout) and (env.misses >= 3 or not env.bricks.any()))
            or (isinstance(env, Snake) and env.food is None)
        )
        return "terminal" if genuine else "truncated"
    return "terminal"
