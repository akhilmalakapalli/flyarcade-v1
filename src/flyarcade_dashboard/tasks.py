"""Task registry: action labels, demo seed namespace and render-state extraction."""

import numpy as np

TASK_ORDER = ("catch", "dodge", "pong", "snake", "flappy", "breakout")
TITLES = {
    "catch": "Catch",
    "dodge": "Dodge",
    "pong": "Pong",
    "snake": "Snake",
    "flappy": "Flappy",
    "breakout": "Breakout",
}
ACTION_LABELS = {
    "catch": ("LEFT", "STAY", "RIGHT"),
    "dodge": ("LEFT", "STAY", "RIGHT"),
    "pong": ("LEFT", "STAY", "RIGHT"),
    "breakout": ("LEFT", "STAY", "RIGHT"),
    "flappy": ("NO FLAP", "FLAP"),
    # The frozen Snake model is the v1.1 study's 8x8 absolute-action Snake.
    "snake": ("UP", "DOWN", "LEFT", "RIGHT"),
}

# Demo seeds live far above every seed block any FlyArcade study has used (v1 and
# v1.1 < 10M; v1.2/v1.3 < 1e9 plus derived offsets up to +2e9). Derived RNG
# offsets used by the evaluation code (+500,000, +1e9, +2e9) stay disjoint too.
DEMO_BASE = 9_000_000_000
DEMO_TASK_BLOCK = 10_000_000
CALIBRATION_INDEX = 5_000_000


def demo_seed(task, index):
    if task not in TASK_ORDER or type(index) is not int or not 0 <= index < DEMO_TASK_BLOCK:
        raise ValueError(f"invalid demo seed request: {task} {index}")
    return DEMO_BASE + TASK_ORDER.index(task) * DEMO_TASK_BLOCK + index


def calibration_seed(task):
    return demo_seed(task, CALIBRATION_INDEX)


def render_state(task, env):
    """Plain JSON description of what is on screen; read-only access to env fields."""
    if task in ("catch", "dodge"):
        return {
            "lanes": 5,
            "player": int(env.player),
            "object": int(env.object),
            "phase": int(env.phase),
            "time": int(env.time),
            "horizon": int(env.horizon),
        }
    if task == "snake":
        return {
            "grid": int(env.grid),
            "body": [list(map(int, c)) for c in env.body],
            "food": list(map(int, env.food)) if env.food else None,
            "heading": int(env.heading),
            "eaten": int(env.eaten),
            "time": int(env.time),
            "death": env.death,
        }
    if task == "pong":
        return {
            "paddle": float(env.paddle),
            "x": float(env.x),
            "y": float(env.y),
            "hits": int(env.hits),
            "misses": int(env.attempts - env.hits),
            "time": int(env.time),
            "horizon": int(env.horizon),
        }
    if task == "flappy":
        return {
            "y": float(env.y),
            "vy": float(env.vy),
            "x": float(env.x),
            "gap": float(env.gap),
            "tolerance": 0.16,
            "passed": int(env.passed),
            "time": int(env.time),
            "horizon": int(env.horizon),
            "death": env.death,
        }
    if task == "breakout":
        return {
            "paddle": float(env.paddle),
            "x": float(env.x),
            "y": float(env.y),
            "bricks": [bool(b) for b in np.asarray(env.bricks)],
            "misses": int(env.misses),
            "hits": int(env.hits),
            "time": int(env.time),
        }
    raise ValueError(f"unknown task {task}")
