"""Episode loop and diagnostics for v1.1.

Seed blocks are disjoint from every block v1 used. v1 training used
1000 + 10000*seed + episode, v1 held-out used 3000000 + 100*seed + i, and v1
development used the 1900000/2000000 blocks. v1.1 development uses the 4000000 and
4500000 blocks, and v1.1 confirmatory runs use 5000000 and 6000000. Nothing in v1.1
is ever tuned against a v1 held-out seed.
"""

import numpy as np

from flyarcade.games import LaneGame, heuristic

FEATURE_FIT_SEEDS = tuple(4_200_000 + i for i in range(40))


def dev_training_seed(dev_seed, episode_index):
    return 4_000_000 + 10_000 * dev_seed + episode_index


def dev_evaluation_seeds(dev_seed, count=20):
    return [4_500_000 + 1_000 * dev_seed + i for i in range(count)]


def confirm_training_seed(seed, episode_index):
    return 5_000_000 + 10_000 * seed + episode_index


def confirm_evaluation_seeds(seed, count=40):
    return [6_000_000 + 100 * seed + i for i in range(count)]


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
    total, score = 0.0, 0
    stats = {k: [] for k in ("entropy", "max_probability", "delta", "value")}
    updates = {"actor_update_abs": [], "critic_update_abs": [], "core_update_abs": []}
    actions = []
    while not game.done:
        if guard is not None and game.time % 4 == 0:
            guard.check()
        observation = game.observe()
        if baseline == "random":
            action = int(rng.integers(0, 3))
        elif baseline == "heuristic":
            action = heuristic(task, observation)
        else:
            action = controller.act(
                observation,
                training=training,
                noise=noise,
                edge_mask=edge_mask,
                neuron_mask=neuron_mask,
            )
        actions.append(action)
        _, reward, _, info = game.step(action)
        if training:
            controller.reward(reward)
            if controller.last_diagnostics is not None:
                for key in stats:
                    stats[key].append(controller.last_diagnostics[key])
                for key in updates:
                    updates[key].append(controller.last_diagnostics[key])
        total += reward
        score += info["score"]
    if training:
        controller.finish_episode()
    row = {
        "return": total,
        "success": score / (game.horizon // 4),
        "task": task,
        "action_counts": np.bincount(actions, minlength=3).tolist(),
    }
    if training and stats["entropy"]:
        row.update(
            {
                "entropy": float(np.mean(stats["entropy"])),
                "max_probability": float(np.mean(stats["max_probability"])),
                "abs_delta": float(np.mean(np.abs(stats["delta"]))),
                "value": float(np.mean(stats["value"])),
                **{k: float(np.mean(v)) for k, v in updates.items()},
            }
        )
        if controller.tick_count:
            row["firing_rate"] = controller.spike_count / (
                controller.graph.n * controller.tick_count
            )
    return row


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


def state_dependence(rows):
    """Fraction of actions not accounted for by the single most used action.

    A constant-action policy scores 0. This is the cheap behavioural counterpart to
    the entropy diagnostic and is what v1's collapsed controllers failed.
    """
    counts = np.sum([row["action_counts"] for row in rows], axis=0)
    total = counts.sum()
    return float(1 - counts.max() / total) if total else 0.0


def success(rows):
    return float(np.mean([row["success"] for row in rows]))
