"""Shared helpers for v1.1 development runs (search and combined validation)."""

import time

import numpy as np

from flyarcade.games import LaneGame
from flyarcade.resources import ResourceGuard
from flyarcade.v11.controller import V11Controller
from flyarcade.v11.experiment import (
    dev_evaluation_seeds,
    dev_training_seed,
    episode,
    state_dependence,
)

RANDOM_BASELINE = {"catch": 0.194, "dodge": 0.806}


def greedy_success(controller, task, seeds):
    """Held-out success of the learned policy's mode, with learning switched off."""
    scores = []
    for seed in seeds:
        game = LaneGame(task, seed)
        controller.reset()
        controller.rng = np.random.default_rng(seed + 500_000)
        score = 0
        while not game.done:
            _, softmax, _ = controller.motor.policy(controller.rates(game.observe()))
            _, _, _, info = game.step(int(softmax.argmax()))
            score += info["score"]
        scores.append(score / (game.horizon // 4))
    return float(np.mean(scores))


def greedy_rows(controller, task, seeds):
    rows = []
    for seed in seeds:
        game = LaneGame(task, seed)
        controller.reset()
        controller.rng = np.random.default_rng(seed + 500_000)
        actions, score = [], 0
        while not game.done:
            _, softmax, _ = controller.motor.policy(controller.rates(game.observe()))
            action = int(softmax.argmax())
            actions.append(action)
            _, _, _, info = game.step(action)
            score += info["score"]
        rows.append(
            {
                "success": score / (game.horizon // 4),
                "action_counts": np.bincount(actions, minlength=3).tolist(),
            }
        )
    return rows


def run_one(graph, standardizer, task, dev_seed, config, episodes, stage):
    """One development trial under its own unchanged 120-second resource guard."""
    guard = ResourceGuard()
    started = time.monotonic()
    controller = V11Controller(graph, dev_seed, stage=stage, standardizer=standardizer, **config)
    seeds = dev_evaluation_seeds(dev_seed)
    before = greedy_success(controller, task, seeds)
    curve = []
    for ep in range(episodes):
        curve.append(
            episode(controller, task, dev_training_seed(dev_seed, ep), training=True, guard=guard)
        )
    rows = greedy_rows(controller, task, seeds)
    after = float(np.mean([r["success"] for r in rows]))
    tail = curve[-episodes // 5 :]
    head = curve[: episodes // 5]
    return {
        "task": task,
        "dev_seed": dev_seed,
        "episodes": episodes,
        "stage": stage,
        "before": before,
        "after": after,
        "improvement": after - before,
        "over_random": after - RANDOM_BASELINE[task],
        "train_first_fifth": float(np.mean([r["success"] for r in head])),
        "train_last_fifth": float(np.mean([r["success"] for r in tail])),
        "final_entropy": float(np.mean([r["entropy"] for r in tail])),
        "final_max_probability": float(np.mean([r["max_probability"] for r in tail])),
        "final_abs_delta": float(np.mean([r["abs_delta"] for r in tail])),
        "final_firing_rate": float(np.mean([r["firing_rate"] for r in tail])),
        "actor_update_abs": float(np.mean([r["actor_update_abs"] for r in tail])),
        "core_update_abs": float(np.mean([r["core_update_abs"] for r in tail])),
        "state_dependence": state_dependence(rows),
        "seconds": time.monotonic() - started,
        "resources": guard.check(),
    }


def rejections(rows):
    """Applied before any performance comparison, per the v1.1 protocol."""
    reasons = []
    if np.mean([r["final_entropy"] for r in rows]) < 0.1:
        reasons.append("action entropy approached zero")
    if np.mean([r["state_dependence"] for r in rows]) < 0.05:
        reasons.append("one action dominated irrespective of state")
    if np.mean([r["final_abs_delta"] for r in rows]) > 5:
        reasons.append("critic diverged")
    firing = np.mean([r["final_firing_rate"] for r in rows])
    if firing < 0.01 or firing > 0.6:
        reasons.append("firing rate collapsed or saturated")
    return reasons
