"""Isolated Stage A experiment helpers and disjoint v1.2 seed namespaces."""

import hashlib
import json
from pathlib import Path

import numpy as np

from flyarcade.v11.controller import V11Controller
from flyarcade.v11.features import Standardizer
from flyarcade.v12.environments import TASKS, encoder
from flyarcade.v12.frozen_core import FrozenCore

CONDITIONS = ("biological", "frozen", "rewired")
PURPOSES = {
    "fit": 100_000,
    "dev_train": 1_000_000,
    "dev_eval": 2_000_000,
    "train": 3_000_000,
    "eval": 4_000_000,
    "representation": 5_000_000,
    "model": 6_000_000,
    "perturb": 7_000_000,
    "probe": 8_000_000,
}


def seed_for(task, purpose, seed=0, index=0):
    if (
        task not in TASKS
        or purpose not in PURPOSES
        or not 0 <= seed < 10
        or not 0 <= index < 10_000
    ):
        raise ValueError("invalid bounded seed request")
    return (
        100_000_000
        + list(TASKS).index(task) * 10_000_000
        + PURPOSES[purpose]
        + seed * 10_000
        + index
    )


def build(graph, task, seed=0, standardizer=None, config=None):
    env = TASKS[task]
    controller = V11Controller(
        graph,
        seed_for(task, "model", seed),
        stage="readout",
        core_lr=0,
        encoder=encoder,
        channels=2 * env.width,
        observation_width=env.width,
        actions=env.actions,
        standardizer=standardizer,
        **(config or {}),
    )

    controller.core = FrozenCore(controller.core)
    return controller


def episode(
    controller,
    task,
    env_seed,
    *,
    training=False,
    baseline=None,
    guard=None,
    noise=0,
    edge_mask=None,
    neuron_mask=None,
    collect=False,
):
    game = TASKS[task](env_seed)
    behaviour = np.random.default_rng(env_seed + 2_000_000_000)
    if controller is not None:
        controller.reset()
        controller.rng = np.random.default_rng(env_seed + 1_000_000_000)
    actions = []
    total = 0.0
    rates = []
    observations = []
    labels = []
    while not game.done:
        if guard is not None and game.time % 8 == 0:
            guard.check()
        obs = game.observe()
        if collect:
            rates.append(controller.rates(obs))
            observations.append(obs)
            labels.append(game.heuristic())
            action = int(behaviour.integers(game.actions)) if game.time % 2 else game.heuristic()
        elif baseline == "random":
            action = int(behaviour.integers(game.actions))
        elif baseline == "heuristic":
            action = game.heuristic()
        elif training:
            action = controller.act(obs, training=True)
        else:
            phi = controller.rates(obs, noise=noise, edge_mask=edge_mask, neuron_mask=neuron_mask)
            action = int(controller.motor.policy(phi)[1].argmax())
        actions.append(action)
        _, reward, _, _ = game.step(action)
        total += reward
        if training:
            controller.reward(reward)
    if training:
        controller.finish_episode()
    counts = np.bincount(actions, minlength=game.actions)
    p = counts / max(counts.sum(), 1)
    result = {
        **game.metrics(),
        "return": total,
        "action_counts": counts.tolist(),
        "action_entropy": float(-(p * np.log(p + 1e-15)).sum()),
        "state_dependence": float(1 - p.max()),
        "state_dependence_definition": "action-diversity proxy, not a causal state-dependence test",
    }
    if collect:
        return result, np.asarray(rates), np.asarray(observations), np.asarray(labels)
    return result


def evaluate(controller, task, seeds, *, guard=None, **kwargs):
    old_rng = controller.rng
    try:
        return [episode(controller, task, s, guard=guard, **kwargs) for s in seeds]
    finally:
        controller.rng = old_rng


def fit_standardizer(graph, task, *, guard=None, count=24):
    controller = build(graph, task)
    samples = []
    seeds = [seed_for(task, "fit", index=i) for i in range(count)]
    for seed in seeds:
        _, rates, _, _ = episode(controller, task, seed, guard=guard, collect=True)
        samples.extend(rates)
    standardizer = Standardizer.fit(np.asarray(samples))
    return standardizer, {
        "seeds": seeds,
        "purpose": "development_feature_fit",
        "samples": len(samples),
        "policy": "alternating heuristic and random actions",
    }


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def graph_hash(graph):
    return hashlib.sha256(
        graph.neuron_ids.tobytes()
        + graph.pre.tobytes()
        + graph.post.tobytes()
        + graph.counts.tobytes()
    ).hexdigest()


def code_files():
    return sorted(set(Path("src/flyarcade").rglob("*.py"))) + [
        Path("scripts/v12_trial.py"),
        Path("scripts/v12_prepare.py"),
    ]


def code_hash():
    h = hashlib.sha256()
    for path in code_files():
        h.update(str(path).encode() + path.read_bytes())
    return h.hexdigest()


def save_checkpoint(path, controller, state):
    path = Path(path)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        actor=controller.motor.actor,
        critic=controller.motor.critic,
        magnitude=controller.core.magnitude,
        metadata=np.frombuffer(json.dumps(state).encode(), dtype=np.uint8),
    )
    tmp.replace(path)


def restore_checkpoint(path, controller, expected):
    with np.load(path, allow_pickle=False) as data:
        state = json.loads(data["metadata"].tobytes().decode())
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(f"checkpoint {key} mismatch")
        for key, original in (
            ("actor", controller.motor.actor),
            ("critic", controller.motor.critic),
            ("magnitude", controller.core.magnitude),
        ):
            if data[key].shape != original.shape or not np.isfinite(data[key]).all():
                raise ValueError("invalid checkpoint arrays")
        if not np.array_equal(data["magnitude"], controller.core.base):
            raise ValueError("Stage A recurrent core changed")
        controller.motor.actor = data["actor"].copy()
        controller.motor.critic = data["critic"].copy()
    controller.reset()
    return state


def state_probe(controller, task, seed, guard=None):
    """Fixed-history input intervention: vary observations, hold neural RNG/reset fixed.

    Diversity of resulting greedy actions demonstrates sensitivity to engineered
    state inputs, distinct from mere action variability along a trajectory.
    """
    states = []
    for ep in range(10):
        game = TASKS[task](seed_for(task, "probe", seed, ep))
        for step in range(20):
            if game.done:
                break
            if step % 5 == 0:
                states.append(game.observe())
            game.step(game.heuristic())
    actions = []
    original = controller.rng
    try:
        for obs in states:
            if guard:
                guard.check()
            controller.reset()
            controller.rng = np.random.default_rng(seed_for(task, "probe", seed, 999))
            for _ in range(4):
                phi = controller.rates(obs)
            actions.append(int(controller.motor.policy(phi)[1].argmax()))
    finally:
        controller.rng = original
    counts = np.bincount(actions, minlength=TASKS[task].actions)
    return {
        "score": float(1 - counts.max() / counts.sum()),
        "action_counts": counts.tolist(),
        "states": len(states),
        "definition": "action diversity under fixed-history state intervention",
        "environment_seeds": [seed_for(task, "probe", seed, i) for i in range(10)],
    }
