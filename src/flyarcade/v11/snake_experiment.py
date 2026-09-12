"""Snake episode loop, seed blocks and diagnostics for the v1.1 architecture.

Seed blocks are disjoint from every block used by v1 (1,000 / 3,000,000 / 1,900,000 /
2,000,000) and by v1.1 catch-dodge (4,000,000 / 4,200,000 / 4,300,000 / 4,500,000 /
5,000,000 / 6,000,000). Snake development uses the 7,000,000 family and Snake
confirmatory runs use the 8,000,000 and 9,000,000 blocks. tests/test_snake.py
asserts the disjointness.
"""

import numpy as np

from flyarcade.v11.snake import SnakeGame, heuristic

FEATURE_FIT_SEEDS = tuple(7_200_000 + i for i in range(120))


def dev_training_seed(dev_seed, episode_index):
    return 7_000_000 + 10_000 * dev_seed + episode_index


def dev_evaluation_seeds(dev_seed, count=20):
    return [7_500_000 + 1_000 * dev_seed + i for i in range(count)]


def confirm_training_seed(seed, episode_index):
    return 8_000_000 + 10_000 * seed + episode_index


def confirm_evaluation_seeds(seed, count=40):
    return [9_000_000 + 100 * seed + i for i in range(count)]


def episode(
    controller,
    env_seed,
    *,
    training=False,
    guard=None,
    noise=0,
    edge_mask=None,
    neuron_mask=None,
    baseline=None,
    greedy=False,
):
    game = SnakeGame(env_seed)
    rng = np.random.default_rng(env_seed + 17)
    if controller is not None:
        controller.reset()
        controller.rng = np.random.default_rng(env_seed + 500_000)
    total = 0.0
    stats = {k: [] for k in ("entropy", "max_probability", "delta", "value")}
    updates = {"actor_update_abs": [], "critic_update_abs": [], "core_update_abs": []}
    actions = []
    while not game.done:
        if guard is not None and game.time % 8 == 0:
            guard.check()
        observation = game.observe()
        if baseline == "random":
            action = int(rng.integers(0, 4))
        elif baseline == "heuristic":
            action = heuristic(game)
        elif greedy:
            _, softmax, _ = controller.motor.policy(
                controller.rates(
                    observation, noise=noise, edge_mask=edge_mask, neuron_mask=neuron_mask
                )
            )
            action = int(softmax.argmax())
        else:
            action = controller.act(
                observation,
                training=training,
                noise=noise,
                edge_mask=edge_mask,
                neuron_mask=neuron_mask,
            )
        actions.append(action)
        _, reward, _, _ = game.step(action)
        if training:
            controller.reward(reward)
            if controller.last_diagnostics is not None:
                for key in stats:
                    stats[key].append(controller.last_diagnostics[key])
                for key in updates:
                    updates[key].append(controller.last_diagnostics[key])
        total += reward
    if training:
        controller.finish_episode()
    row = {
        "food": game.eaten,
        "steps": game.time,
        "return": total,
        "length": len(game.body),
        "death": game.death,
        "action_counts": np.bincount(actions, minlength=4).tolist(),
    }
    if training and stats["entropy"]:
        row.update(
            {
                "entropy": float(np.mean(stats["entropy"])),
                "max_probability": float(np.mean(stats["max_probability"])),
                "abs_delta": float(np.mean(np.abs(stats["delta"]))),
                **{k: float(np.mean(v)) for k, v in updates.items()},
            }
        )
        if controller.tick_count:
            row["firing_rate"] = controller.spike_count / (
                controller.graph.n * controller.tick_count
            )
    return row


def evaluate(controller, seeds, **kwargs):
    """Greedy evaluation with learning disabled, matching the catch-dodge protocol."""
    old_rng = controller.rng
    try:
        return [episode(controller, seed, greedy=True, **kwargs) for seed in seeds]
    finally:
        controller.rng = old_rng


def state_dependence(rows):
    counts = np.sum([row["action_counts"] for row in rows], axis=0)
    total = counts.sum()
    return float(1 - counts.max() / total) if total else 0.0


def aggregate(rows):
    return {
        "food": float(np.mean([r["food"] for r in rows])),
        "steps": float(np.mean([r["steps"] for r in rows])),
        "return": float(np.mean([r["return"] for r in rows])),
        "length": float(np.mean([r["length"] for r in rows])),
        "state_dependence": state_dependence(rows),
    }


def learning_curve_auc(curve, key="food"):
    """Mean of the metric over training: area under the learning curve, normalised."""
    return float(np.mean([row[key] for row in curve])) if curve else 0.0


def episodes_to_threshold(curve, threshold, key="food", window=25):
    """First episode at which the trailing window mean reaches the threshold."""
    values = [row[key] for row in curve]
    for i in range(window, len(values) + 1):
        if float(np.mean(values[i - window : i])) >= threshold:
            return i
    return None
