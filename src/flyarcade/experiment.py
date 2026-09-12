"""Seed-separated episodes and atomic episode-boundary checkpoints."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np

from flyarcade.games import LaneGame, heuristic
from flyarcade.resources import ResourceGuard


def episode(
    controller,
    task,
    env_seed,
    *,
    training=False,
    guard=None,
    noise=0,
    edge_mask=None,
    neuron_mask=None,
    baseline=None,
):
    game = LaneGame(task, env_seed)
    rng = np.random.default_rng(env_seed + 17)
    if controller is not None:
        controller.reset()
    total, score = 0, 0
    while not game.done:
        if guard is not None and game.time % 4 == 0:
            guard.check()
        if baseline == "random":
            action = int(rng.integers(0, 3))
        elif baseline == "heuristic":
            action = heuristic(task, game.observe())
        else:
            action = controller.act(
                game.observe(),
                training=training,
                noise=noise,
                edge_mask=edge_mask,
                neuron_mask=neuron_mask,
            )
        _, reward, _, info = game.step(action)
        if training:
            controller.reward(reward)
        total += reward
        score += info["score"]
    return {"return": total, "success": score / (game.horizon // 4)}


def evaluate(controller, task, seeds, *, guard=None, **kwargs):
    old_rng = controller.rng
    results = []
    try:
        for seed in seeds:
            controller.rng = np.random.default_rng(seed + 500_000)
            results.append(episode(controller, task, seed, guard=guard, **kwargs))
    finally:
        controller.rng = old_rng
    return results


def checkpoint(controller, path, metadata):
    path = Path(path)
    ResourceGuard(directory=path.parent).check()
    state = {
        **metadata,
        "rng_state": controller.rng.bit_generator.state,
        "baseline": controller.baseline,
    }
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez_compressed(
                f,
                recurrent=controller.core.magnitude,
                motor=controller.motor.weights,
                metadata=np.frombuffer(json.dumps(state).encode(), dtype=np.uint8),
            )
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def restore(controller, path, expected_graph_sha256):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(data["metadata"].tobytes().decode())
        if metadata["graph_sha256"] != expected_graph_sha256:
            raise ValueError("checkpoint graph mismatch")
        for array, shape in (
            (data["recurrent"], controller.core.magnitude.shape),
            (data["motor"], controller.motor.weights.shape),
        ):
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError("invalid checkpoint weights")
        controller.core.magnitude = data["recurrent"].copy()
        controller.motor.weights = data["motor"].copy()
        controller.baseline = metadata["baseline"]
        controller.rng.bit_generator.state = metadata["rng_state"]
    controller.reset()
    return metadata
