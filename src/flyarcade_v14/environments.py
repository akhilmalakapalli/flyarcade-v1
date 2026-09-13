"""Metrics-only adapters around frozen historical targets and training curricula."""

from flyarcade.games import LaneGame
from flyarcade.games import heuristic as lane_heuristic
from flyarcade.sensory import features as lane_features
from flyarcade.v11.snake import SnakeGame
from flyarcade.v11.snake import features as snake_features
from flyarcade.v11.snake import heuristic as snake_heuristic
from flyarcade.v12.environments import Flappy, Pong, encoder
from flyarcade_v13.environments import FlappyCurriculum, flappy_oracle


class LaneAdapter(LaneGame):
    actions = 3
    width = 3

    def __init__(self, seed=0):
        super().__init__(self.task_name, seed)
        self.score = 0

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self.score += info["score"]
        return obs, reward, done, {**info, **self.metrics()}

    def metrics(self):
        return {
            "success": self.score / (self.horizon // 4),
            "events": self.score,
            "steps": self.time,
        }

    def heuristic(self):
        return lane_heuristic(self.task, self.observe())


class Catch(LaneAdapter):
    task_name = "catch"


class Dodge(LaneAdapter):
    task_name = "dodge"


class HistoricalSnake(SnakeGame):
    actions = 4
    width = 21

    @property
    def horizon(self):
        return self.max_steps

    def metrics(self):
        return {
            "success": self.eaten,
            "food": self.eaten,
            "steps": self.time,
            "length": len(self.body),
            "death": self.death,
        }

    def step(self, action):
        obs, reward, done, info = super().step(action)
        return obs, reward, done, {**info, **self.metrics()}

    def heuristic(self):
        return snake_heuristic(self)


TARGETS = {"catch": Catch, "dodge": Dodge, "snake": HistoricalSnake, "pong": Pong, "flappy": Flappy}
TASK_ORDER = tuple(TARGETS)


def encoder_for(task):
    if task in ("catch", "dodge"):
        return lane_features
    if task == "snake":
        return snake_features
    return encoder


def make_env(task, seed, level="target"):
    if level == "target":
        return TARGETS[task](seed)
    if task == "snake":
        return HistoricalSnake(seed, grid={"easy": 4, "intermediate": 6}[level])
    if task == "flappy":
        tolerance, speed = {"easy": (0.24, 0.03), "intermediate": (0.20, 0.035)}[level]
        return FlappyCurriculum(seed, tolerance=tolerance, speed=speed)
    raise ValueError("curriculum not permitted for this task")


def reference_action(env):
    return flappy_oracle(env) if isinstance(env, Flappy) else env.heuristic()


def potential(env):
    raise ValueError("v1.4 protocol forbids additional reward shaping")


def terminal_kind(env):
    if not env.done:
        return None
    if isinstance(env, HistoricalSnake):
        return "truncated" if env.death == "timeout" else "terminal"
    if isinstance(env, LaneAdapter):
        return "truncated"
    if env.time >= env.horizon and getattr(env, "death", None) is None:
        return "truncated"
    return "terminal"
