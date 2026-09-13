"""Deterministic compact arcade environments with normalized engineered observations."""

import numpy as np


class Environment:
    actions = 3
    horizon = 120

    def reset(self, seed):
        if type(seed) is not int or seed < 0:
            raise ValueError("reset requires a nonnegative integer seed")
        self.rng = np.random.default_rng(seed)
        self.time, self.done = 0, False
        self._reset()
        return self.observe()

    def __init__(self, seed=0):
        self.reset(seed)

    def validate(self, action):
        if type(action) is not int or not 0 <= action < self.actions or self.done:
            raise ValueError("valid discrete action required before termination")

    def finish(self, reward):
        self.time += 1
        self.done |= self.time >= self.horizon
        return self.observe(), float(reward), bool(self.done), self.metrics()


class Flappy(Environment):
    actions, horizon = 2, 138
    width = 6

    def _reset(self):
        self.y, self.vy, self.x = 0.5, 0.0, 0.9
        self.gap = float(self.rng.uniform(0.25, 0.75))
        self.passed = 0
        self.death = None

    def observe(self):
        return np.clip(
            [
                self.y,
                (self.vy + 0.08) / 0.14,
                self.x / 0.9,
                self.gap,
                (self.gap - self.y + 1) / 2,
                self.time / self.horizon,
            ],
            0,
            1,
        )

    def step(self, action):
        self.validate(action)
        old_error = abs(self.y - self.gap)
        self.vy = 0.055 if action else max(-0.08, self.vy - 0.009)
        self.y += self.vy
        self.x -= 0.04
        reward = 0.01 + 0.02 * (old_error - abs(self.y - self.gap))
        if self.y <= 0.025 or self.y >= 0.975:
            self.done, self.death, reward = True, "boundary", -1.0
        elif self.x <= 0:
            if abs(self.y - self.gap) > 0.16:
                self.done, self.death, reward = True, "pipe", -1.0
            else:
                self.passed += 1
                reward += 1
                self.x = 0.9
                self.gap = float(self.rng.uniform(0.25, 0.75))
        return self.finish(reward)

    def heuristic(self):
        return int(self.y + 2 * self.vy < self.gap - 0.025)

    def metrics(self):
        return {
            "success": self.passed / 6,
            "obstacles_passed": self.passed,
            "steps": self.time,
            "death": self.death,
        }


def reflect(position):
    folded = position % 2
    return folded if folded <= 1 else 2 - folded


class Pong(Environment):
    horizon, width = 144, 7

    def _reset(self):
        self.paddle = 0.5
        self.hits, self.attempts = 0, 0
        self._serve()

    def _serve(self):
        self.x, self.y = 0.95, float(self.rng.uniform(0.15, 0.85))
        self.vx, self.vy = -0.06, float(self.rng.choice([-0.035, -0.02, 0.02, 0.035]))

    def intercept(self):
        return reflect(self.y + self.vy * self.x / abs(self.vx))

    def observe(self):
        return np.clip(
            [
                self.paddle,
                self.x,
                self.y,
                (self.vx + 0.06) / 0.12,
                (self.vy + 0.04) / 0.08,
                (self.intercept() - self.paddle + 1) / 2,
                self.time / self.horizon,
            ],
            0,
            1,
        )

    def step(self, action):
        self.validate(action)
        before = abs(self.paddle - self.intercept())
        self.paddle = float(np.clip(self.paddle + (action - 1) * 0.07, 0.1, 0.9))
        reward = 0.02 * (before - abs(self.paddle - self.intercept()))
        self.x += self.vx
        self.y += self.vy
        if self.y < 0 or self.y > 1:
            self.y = reflect(self.y)
            self.vy *= -1
        if self.x <= 0:
            self.attempts += 1
            hit = abs(self.y - self.paddle) <= 0.14
            self.hits += int(hit)
            reward += 1 if hit else -1
            self._serve()
        return self.finish(reward)

    def heuristic(self):
        error = self.intercept() - self.paddle
        return 2 if error > 0.03 else 0 if error < -0.03 else 1

    def metrics(self):
        return {
            "success": self.hits / max(self.attempts, 1),
            "returns": self.hits,
            "attempted_returns": self.attempts,
            "steps": self.time,
        }


class Breakout(Environment):
    horizon, width = 160, 11

    def _reset(self):
        self.paddle = 0.5
        self.bricks = np.ones(5, dtype=bool)
        self.hits, self.misses = 0, 0
        self._serve()

    def _serve(self):
        self.x, self.y = 0.5, 0.2
        self.vx, self.vy = float(self.rng.choice([-0.035, 0.035])), 0.055

    def intercept(self):
        return reflect(self.x + self.vx * self.y / abs(self.vy)) if self.vy < 0 else self.x

    def observe(self):
        return np.clip(
            [
                self.paddle,
                self.x,
                self.y,
                (self.vx + 0.06) / 0.12,
                (self.vy + 0.06) / 0.12,
                (self.intercept() - self.paddle + 1) / 2,
                *self.bricks.astype(float),
            ],
            0,
            1,
        )

    def step(self, action):
        self.validate(action)
        before = abs(self.paddle - self.intercept())
        self.paddle = float(np.clip(self.paddle + (action - 1) * 0.07, 0.1, 0.9))
        reward = 0.01 * (before - abs(self.paddle - self.intercept()))
        self.x += self.vx
        self.y += self.vy
        if self.x < 0 or self.x > 1:
            self.x = reflect(self.x)
            self.vx *= -1
        if self.y >= 1:
            self.y = 2 - self.y
            self.vy = -abs(self.vy)
            index = min(int(self.x * 5), 4)
            if self.bricks[index]:
                self.bricks[index] = False
                reward += 1
            if not self.bricks.any():
                self.done = True
        if self.y <= 0:
            if abs(self.x - self.paddle) <= 0.14:
                self.hits += 1
                self.y = -self.y
                self.vy = abs(self.vy)
                # Hit position controls bounce angle, a standard artificial mechanic.
                self.vx = float(np.clip((self.x - self.paddle) * 0.3, -0.05, 0.05))
                if abs(self.vx) < 0.01:
                    self.vx = 0.015
                reward += 0.1
            else:
                self.misses += 1
                reward -= 1
                self.done |= self.misses >= 3
                if not self.done:
                    self._serve()
        return self.finish(reward)

    def heuristic(self):
        error = self.intercept() - self.paddle
        return 2 if error > 0.03 else 0 if error < -0.03 else 1

    def metrics(self):
        return {
            "success": float((~self.bricks).mean()),
            "bricks_destroyed": int((~self.bricks).sum()),
            "interceptions": self.hits,
            "misses": self.misses,
            "completion": not self.bricks.any(),
            "steps": self.time,
        }


class Snake(Environment):
    horizon, width, size = 120, 17, 6
    directions = ((0, -1), (1, 0), (0, 1), (-1, 0))  # clockwise heading

    def _reset(self):
        self.body = [(3, 3), (2, 3), (1, 3)]
        self.heading = 1
        self.eaten, self.starvation = 0, 0
        self.death = None
        self._food()

    def _food(self):
        free = [
            (x, y) for y in range(self.size) for x in range(self.size) if (x, y) not in self.body
        ]
        self.food = free[int(self.rng.integers(len(free)))] if free else None
        self.done |= not free

    def next_cell(self, action):
        heading = (self.heading + action - 1) % 4
        dx, dy = self.directions[heading]
        return (self.body[0][0] + dx, self.body[0][1] + dy), heading

    def danger(self, action):
        cell, _ = self.next_cell(action)
        growing = cell == self.food
        occupied = self.body if growing else self.body[:-1]
        return not (0 <= cell[0] < self.size and 0 <= cell[1] < self.size) or cell in occupied

    def observe(self):
        x, y = self.body[0]
        fx, fy = self.food if self.food is not None else (x, y)
        dx, dy = (fx - x) / (self.size - 1), (fy - y) / (self.size - 1)
        # Food direction, distance, heading, relative local body/wall dangers, wall distances.
        return np.array(
            [
                max(dx, 0),
                max(-dx, 0),
                max(dy, 0),
                max(-dy, 0),
                (abs(dx) + abs(dy)) / 2,
                *[float(self.heading == h) for h in range(4)],
                *[float(self.danger(a)) for a in range(3)],
                x / (self.size - 1),
                (self.size - 1 - x) / (self.size - 1),
                y / (self.size - 1),
                (self.size - 1 - y) / (self.size - 1),
                len(self.body) / (self.size * self.size),
            ],
            dtype=float,
        )

    def step(self, action):
        self.validate(action)
        cell, heading = self.next_cell(action)
        self.heading = heading
        reward = -0.01
        if self.danger_for_cell(cell):
            self.done = True
            self.death = (
                "wall" if not (0 <= cell[0] < self.size and 0 <= cell[1] < self.size) else "self"
            )
            reward = -1
        else:
            self.body.insert(0, cell)
            self.starvation += 1
            if cell == self.food:
                self.eaten += 1
                self.starvation = 0
                reward += 1
                self._food()
            else:
                self.body.pop()
            if self.starvation >= 40:
                self.done, self.death = True, "starvation"
        return self.finish(reward)

    def danger_for_cell(self, cell):
        occupied = self.body if cell == self.food else self.body[:-1]
        return not (0 <= cell[0] < self.size and 0 <= cell[1] < self.size) or cell in occupied

    def heuristic(self):
        candidates = [a for a in range(3) if not self.danger(a)]
        if not candidates:
            return 1
        return min(
            candidates,
            key=lambda a: (
                sum(abs(v - w) for v, w in zip(self.next_cell(a)[0], self.food, strict=True)),
                abs(a - 1),
                a,
            ),
        )

    def metrics(self):
        return {
            "success": min(self.eaten / 10, 1),
            "food": self.eaten,
            "steps": self.time,
            "length": len(self.body),
            "death": self.death,
        }


TASKS = {"flappy": Flappy, "pong": Pong, "breakout": Breakout, "snake": Snake}


def encoder(observation):
    observation = np.asarray(observation, dtype=float)
    if observation.ndim != 1 or not np.isfinite(observation).all():
        raise ValueError("finite vector observations required")
    observation = np.clip(observation, 0, 1)
    return np.concatenate([observation, 1 - observation])
