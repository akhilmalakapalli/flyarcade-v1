"""Headless Snake: grid world, food, wall and self collision, growing body.

Deliberately not a pixel or CNN input. The observation is a compact hand-specified
vector in [0, 1], encoded into visual-projection neurons by the same Bernoulli
injection the catch and dodge tasks use, so the sensory interface across all three
tasks differs only in how many channels it carries.
"""

import numpy as np

GRID = 8
START_LENGTH = 3
MAX_STEPS = 200
STARVATION = 64
# Absolute actions; index is (dy, dx) with row 0 at the top.
MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))
ACTION_NAMES = ("up", "down", "left", "right")
OPPOSITE = {0: 1, 1: 0, 2: 3, 3: 2}
CHANNELS = 21


class SnakeGame:
    """One Snake episode with a fixed grid, seeded food and bounded length."""

    def __init__(self, seed=0, grid=GRID, max_steps=MAX_STEPS, starvation=STARVATION):
        self.grid = int(grid)
        self.max_steps = int(max_steps)
        self.starvation = int(starvation)
        self.rng = np.random.default_rng(seed)
        middle = self.grid // 2
        # Body is head-first; the snake starts pointing right with a tail behind it.
        self.body = [(middle, middle - i) for i in range(START_LENGTH)]
        self.heading = 3
        self.time = 0
        self.since_food = 0
        self.eaten = 0
        self.done = False
        self.death = None
        self.food = self._place_food()

    def _free_cells(self):
        occupied = set(self.body)
        return [
            (r, c) for r in range(self.grid) for c in range(self.grid) if (r, c) not in occupied
        ]

    def _place_food(self):
        free = self._free_cells()
        if not free:
            return None
        return free[int(self.rng.integers(0, len(free)))]

    def blocked(self, cell, *, ignore_tail=True):
        """Would moving into `cell` end the episode?"""
        row, column = cell
        if not (0 <= row < self.grid and 0 <= column < self.grid):
            return True
        # The tail vacates as the head advances, unless the snake just ate.
        body = self.body[:-1] if ignore_tail and len(self.body) > 1 else self.body
        return cell in body

    @property
    def head(self):
        return self.body[0]

    def observe(self):
        """Compact normalised observation; every entry lies in [0, 1]."""
        row, column = self.head
        food_row, food_column = self.food if self.food else (row, column)
        span = max(self.grid - 1, 1)
        danger = [float(self.blocked((row + dr, column + dc))) for dr, dc in MOVES]
        heading = [float(i == self.heading) for i in range(4)]
        local = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                local.append(float(self.blocked((row + dr, column + dc), ignore_tail=False)))
        return np.array(
            [
                float(food_row < row),
                float(food_row > row),
                float(food_column < column),
                float(food_column > column),
                (abs(food_row - row) + abs(food_column - column)) / (2 * span),
                *heading,
                *danger,
                *local,
            ],
            dtype=float,
        )

    def step(self, action):
        if type(action) is not int or action not in (0, 1, 2, 3) or self.done:
            raise ValueError("valid action required before episode termination")
        # An immediate 180-degree reversal is illegal; the snake keeps its heading.
        if len(self.body) > 1 and action == OPPOSITE[self.heading]:
            action = self.heading
        self.heading = action
        row, column = self.head
        dr, dc = MOVES[action]
        target = (row + dr, column + dc)
        self.time += 1
        self.since_food += 1
        reward = -0.01
        ate = 0
        if not (0 <= target[0] < self.grid and 0 <= target[1] < self.grid):
            self.done, self.death = True, "wall"
            reward -= 1.0
        elif target in (self.body[:-1] if len(self.body) > 1 else self.body):
            self.done, self.death = True, "self"
            reward -= 1.0
        else:
            self.body.insert(0, target)
            if target == self.food:
                self.eaten += 1
                ate = 1
                self.since_food = 0
                reward += 1.0
                self.food = self._place_food()
                if self.food is None:
                    self.done, self.death = True, "filled"
            else:
                self.body.pop()
        if not self.done and self.time >= self.max_steps:
            self.done, self.death = True, "timeout"
        if not self.done and self.since_food >= self.starvation:
            self.done, self.death = True, "starved"
        return self.observe(), float(reward), self.done, {"eaten": ate, "length": len(self.body)}


def features(observation):
    """Snake observations are already normalised channels; pass them through."""
    observation = np.asarray(observation, dtype=float)
    if observation.shape != (CHANNELS,) or not np.isfinite(observation).all():
        raise ValueError(f"{CHANNELS} finite normalised observations required")
    return np.clip(observation, 0, 1)


def heuristic(game):
    """Greedy toward food, refusing moves that end the episode.

    A solvability reference, not an optimal player: it has no lookahead and will
    still trap itself once the body is long.
    """
    row, column = game.head
    food_row, food_column = game.food if game.food else (row, column)
    preference = []
    if food_row < row:
        preference.append(0)
    if food_row > row:
        preference.append(1)
    if food_column < column:
        preference.append(2)
    if food_column > column:
        preference.append(3)
    order = preference + [a for a in range(4) if a not in preference]
    for action in order:
        if len(game.body) > 1 and action == OPPOSITE[game.heading]:
            continue
        dr, dc = MOVES[action]
        if not game.blocked((row + dr, column + dc)):
            return action
    return game.heading
