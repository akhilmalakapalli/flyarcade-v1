"""Isolated v1.4 adaptation: seeds, resumable PPO, evaluation and probes."""

import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np

from flyarcade.v11.features import Standardizer
from flyarcade_v13.policy import Adam
from flyarcade_v13.ppo import gae, sample_actions, update
from flyarcade_v14.environments import (
    TARGETS,
    TASK_ORDER,
    make_env,
    potential,
    reference_action,
    terminal_kind,
)
from flyarcade_v14.features import FlyFeatures, SensoryFeatures, rng_for_episode
from flyarcade_v14.policy import ActorCritic

# ------------------------------------------------------------------ seeds
SEED_BASE = 100_000_000_000
TASK_BLOCK = 100_000_000_000
PURPOSE_BLOCK = 5_000_000_000
SEED_SLOT = 1_000_000
PURPOSES = (
    "software",
    "feature_fit",
    "train",
    "model",
    "rollout",
    "curve",
    "dev_eval",
    "dev_probe",
    "imitation",
    "primary_validation",
    "final_validation",
    "validation_probe",
)
BEHAVIOUR_OFFSET = 2_000_000_000


def seed_for(task, purpose, seed=0, index=0):
    if (
        task not in TARGETS
        or purpose not in PURPOSES
        or type(seed) is not int
        or type(index) is not int
        or not 0 <= seed < 3
        or not 0 <= index < SEED_SLOT
    ):
        raise ValueError("invalid v1.4 seed request; confirmatory purposes are forbidden")
    return (
        SEED_BASE
        + TASK_ORDER.index(task) * TASK_BLOCK
        + PURPOSES.index(purpose) * PURPOSE_BLOCK
        + seed * SEED_SLOT
        + index
    )


def purpose_range(task, purpose):
    start = seed_for(task, purpose)
    return start, start + PURPOSE_BLOCK


# ------------------------------------------------------------------ hashing
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def params_hash(params):
    h = hashlib.sha256()
    for key in sorted(params):
        h.update(key.encode() + np.ascontiguousarray(params[key]).tobytes())
    return h.hexdigest()


def graph_hash(graph):
    return hashlib.sha256(
        graph.neuron_ids.tobytes()
        + graph.pre.tobytes()
        + graph.post.tobytes()
        + graph.counts.tobytes()
    ).hexdigest()


def code_files():
    return sorted(Path("src").rglob("*.py")) + [Path("scripts/v14_trial.py")]


def code_hash():
    h = hashlib.sha256()
    for path in code_files():
        h.update(str(path).encode() + path.read_bytes())
    return h.hexdigest()


# ------------------------------------------------------------------ sources
GRAPH_PATHS = {
    "biological": "data/malecns-v1.0/graph.npz",
    **{f"rewired-{i}": f"data/v12/rewired-{i}.npz" for i in range(3)},
    **{f"rewired-{i}": f"data/v13/rewired-{i}.npz" for i in range(3, 5)},
}
_GRAPHS = {}


def load_topology(topology):
    if topology not in _GRAPHS:
        from flyarcade.connectome import load_graph

        _GRAPHS[topology] = load_graph(GRAPH_PATHS[topology])
    return _GRAPHS[topology]


def standardizer_path(task, topology, ticks, readout):
    scope = readout.replace("+", "-")
    return Path(f"artifacts/v14/standardizers/{task}-{topology}-t{ticks}-{scope}.json")


def make_source(config, batch, graph=None):
    task = config["task"]
    width = TARGETS[task].width
    if config["source"] == "sensory":
        return SensoryFeatures(width, batch)
    graph = graph if graph is not None else load_topology(config["topology"])
    std = Standardizer.load(
        config.get("standardizer")
        or standardizer_path(task, config["topology"], config["ticks"], config["readout"])
    )
    return FlyFeatures(
        graph,
        width,
        batch,
        task=task,
        ticks=config["ticks"],
        readout=config["readout"],
        standardizer=std,
    )


# ------------------------------------------------------------------ standardizer
def fit_standardizer(
    graph,
    task,
    *,
    ticks,
    readout="descending",
    min_samples=8192,
    max_episodes=2000,
    batch=16,
    guard=None,
):
    """Development-only feature statistics under alternating reference/random actions.

    Episodes are consumed in seed order until at least ``min_samples`` states exist,
    so short-episode tasks (Flappy, Snake) get coverage comparable to long ones.
    """
    width = TARGETS[task].width
    source = FlyFeatures(graph, width, batch, task=task, ticks=ticks, readout=readout)
    seeds = []
    samples = []
    while len(samples) < min_samples and len(seeds) < max_episodes:
        block = [seed_for(task, "feature_fit", index=len(seeds) + i) for i in range(batch)]
        seeds.extend(block)
        envs = [make_env(task, s) for s in block]
        behaviour = [np.random.default_rng(s + BEHAVIOUR_OFFSET) for s in block]
        for c, s in enumerate(block):
            source.reset(c, s)
        while not all(e.done for e in envs):
            if guard is not None:
                guard.check()
            live = [c for c, e in enumerate(envs) if not e.done]
            rates = source.raw_rates(np.stack([envs[c].observe() for c in live]), live)
            samples.extend(rates)
            for c in live:
                env = envs[c]
                action = (
                    int(behaviour[c].integers(env.actions))
                    if env.time % 2
                    else reference_action(env)
                )
                env.step(action)
    standardizer = Standardizer(
        np.mean(samples, axis=0),
        np.maximum(np.std(samples, axis=0), 0.02),
        clip=4.0,
        normalise=False,
    )
    return standardizer, {
        "seeds": seeds,
        "purpose": "feature_fit (development only)",
        "samples": len(samples),
        "ticks": ticks,
        "readout": readout,
        "policy": "alternating reference-policy and uniform-random actions",
        "graph_sha256": graph_hash(graph),
        "normalise": False,
    }


# ------------------------------------------------------------------ trainer
DEFAULTS = {
    "source": "fly",
    "topology": "biological",
    "readout": "descending",
    "ticks": 4,
    "arch": "mlp",
    "hidden": 128,
    "core": 64,
    "gamma": 0.99,
    "lam": 0.95,
    "lr": 3e-4,
    "anneal_lr": True,
    "clip": 0.2,
    "ent": 0.003,
    "vf": 0.5,
    "max_grad_norm": 0.5,
    "epochs": 4,
    "envs": 16,
    "steps": 128,
    "minibatches": 8,
    "chunk": 16,
    "reward_scale": 1.0,
    "shaping": 0.0,
    "transitions": 65536,
    "curriculum": [["target", 1.0]],
    "eval_every": 8,
    "curve_episodes": 16,
    "train_purpose": "train",
    "eval_purpose": "dev_eval",
    "model_purpose": "model",
    "rollout_purpose": "rollout",
    "seed": 0,
    "learn": True,
}


def complete_config(config):
    merged = {**DEFAULTS, **config}
    if merged["source"] != "fly" or merged["topology"] != "biological" or merged["shaping"]:
        raise ValueError("v1.4 fixed biological substrate and original rewards required")
    for key in ("train_purpose", "eval_purpose", "model_purpose", "rollout_purpose"):
        if merged[key] not in PURPOSES or "validation" in merged[key]:
            raise ValueError("only development seeds may enter training configuration")
    if merged["task"] not in TARGETS:
        raise ValueError("unknown task")
    if merged["source"] not in ("fly", "sensory"):
        raise ValueError("source must be fly or sensory")
    fractions = [f for _, f in merged["curriculum"]]
    if merged["curriculum"][-1][0] != "target" or abs(sum(fractions) - 1) > 1e-9:
        raise ValueError("curriculum must end on the target and cover the whole budget")
    return merged


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def build_model(config, width):
    return ActorCritic(
        width,
        TARGETS[config["task"]].actions,
        arch=config["arch"],
        hidden=config["hidden"],
        core=config["core"],
        seed=seed_for(config["task"], config["model_purpose"], config["seed"]),
    )


class Trainer:
    """Collect fixed-length rollouts from a batch of environments, then run PPO."""

    def __init__(self, config, graph=None):
        self.config = complete_config(config)
        c = self.config
        self.task = c["task"]
        self.source = make_source(c, c["envs"], graph)
        self.model = build_model(c, self.source.width)
        self.initial_params = self.model.copy_params()
        self.optimizer = Adam(self.model.params, lr=c["lr"])
        self.rng = np.random.default_rng(seed_for(self.task, c["rollout_purpose"], c["seed"]))
        self.transitions = 0
        self.episodes_started = 0
        self.updates = 0
        self.history = []
        self.curve = []
        self.completed = []
        self.level_transitions = {name: 0 for name, _ in c["curriculum"]}
        self.envs = [None] * c["envs"]
        self.episode_return = np.zeros(c["envs"])
        self.hidden = self.model.initial_state(c["envs"])
        self.resets = np.ones(c["envs"])
        for column in range(c["envs"]):
            self._start_episode(column)
        self.features = self._observe(range(c["envs"]))
        self.wall = 0.0

    # -- curriculum
    def level(self):
        c = self.config
        spent = 0.0
        for name, fraction in c["curriculum"]:
            spent += fraction * c["transitions"]
            if self.transitions < spent - 1e-9:
                return name
        return c["curriculum"][-1][0]

    def _start_episode(self, column):
        seed = seed_for(
            self.task, self.config["train_purpose"], self.config["seed"], self.episodes_started
        )
        self.episodes_started += 1
        level = self.level()
        self.envs[column] = make_env(self.task, seed, level)
        self.envs[column].v13_level = level
        self.source.reset(column, seed)
        self.episode_return[column] = 0.0
        self.resets[column] = 1.0

    def _observe(self, columns):
        columns = list(columns)
        return self.source(np.stack([self.envs[i].observe() for i in columns]), columns)

    # -- rollout
    def collect(self):
        c = self.config
        steps, envs = c["steps"], c["envs"]
        width = self.source.width
        batch = {
            "features": np.zeros((steps, envs, width)),
            "actions": np.zeros((steps, envs), dtype=int),
            "log_probs": np.zeros((steps, envs)),
            "values": np.zeros((steps, envs)),
            "rewards": np.zeros((steps, envs)),
            "dones": np.zeros((steps, envs)),
            "resets": np.zeros((steps, envs)),
        }
        if self.model.recurrent:
            batch["states"] = np.zeros((steps, envs, self.model.core))
        for t in range(steps):
            batch["features"][t] = self.features
            batch["resets"][t] = self.resets
            if self.model.recurrent:
                batch["states"][t] = self.hidden
            logits, values, hidden = self.model.step(self.features, self.hidden, self.resets)
            actions, log_probs = sample_actions(logits, self.rng)
            batch["actions"][t], batch["log_probs"][t], batch["values"][t] = (
                actions,
                log_probs,
                values,
            )
            self.resets = np.zeros(envs)
            for i, env in enumerate(self.envs):
                before = potential(env) if c["shaping"] else 0.0
                _, reward, done, info = env.step(int(actions[i]))
                kind = terminal_kind(env)
                after = potential(env) if c["shaping"] and kind != "terminal" else 0.0
                shaped = c["reward_scale"] * reward + c["shaping"] * (c["gamma"] * after - before)
                self.episode_return[i] += reward
                batch["rewards"][t, i] = shaped
                batch["dones"][t, i] = float(done)
                self.transitions += 1
                self.level_transitions[env.v13_level] += 1
                if done:
                    if kind == "truncated":
                        bootstrap = self._bootstrap_value(i, hidden)
                        batch["rewards"][t, i] += c["gamma"] * bootstrap
                    self.completed.append(
                        {**info, "return": float(self.episode_return[i]), "level": env.v13_level}
                    )
            self.hidden = hidden
            finished = [i for i, env in enumerate(self.envs) if env.done]
            for i in finished:
                self._start_episode(i)
            self.features = self._observe(range(envs))
        _, last_values, _ = self.model.step(self.features, self.hidden, self.resets)
        advantages, returns = gae(
            batch["rewards"], batch["values"], batch["dones"], last_values, c["gamma"], c["lam"]
        )
        batch["advantages"], batch["returns"] = advantages, returns
        return batch

    def _bootstrap_value(self, column, hidden):
        """V(s_T) for a time-limit stop: advances only that finished column's core."""
        phi = self.source(self.envs[column].observe()[None], [column])
        state = None if hidden is None else hidden[column : column + 1]
        _, value, _ = self.model.step(phi, state, None)
        return float(value[0])

    def train_step(self):
        c = self.config
        started = time.monotonic()
        batch = self.collect()
        frac = min(self.transitions / c["transitions"], 1.0)
        lr = c["lr"] * max(1.0 - frac, 0.05) if c["anneal_lr"] else c["lr"]
        stats = {}
        if c["learn"]:
            stats = update(self.model, self.optimizer, batch, c, self.rng, lr=lr)
        self.updates += 1
        recent = self.completed[-100:]
        elapsed = time.monotonic() - started
        self.wall += elapsed
        row = {
            "update": self.updates,
            "transitions": self.transitions,
            "episodes": len(self.completed),
            "level": self.level(),
            "lr": lr,
            "wall_seconds": self.wall,
            "transitions_per_second": c["steps"] * c["envs"] / max(elapsed, 1e-9),
            "train_success": float(np.mean([r["success"] for r in recent])) if recent else None,
            "train_return": float(np.mean([r["return"] for r in recent])) if recent else None,
            "rollout_action_fractions": np.bincount(
                batch["actions"].ravel(), minlength=TARGETS[self.task].actions
            ).tolist(),
            **stats,
            "finite": self.model.finite(),
        }
        if not row["finite"]:
            raise FloatingPointError("non-finite policy parameters")
        self.history.append(row)
        return row

    def done(self):
        return self.transitions >= self.config["transitions"]

    # -- checkpoint
    def state(self):
        return {
            "config_sha256": config_hash(self.config),
            "params": self.model.params,
            "initial_params": self.initial_params,
            "optimizer": self.optimizer.state(),
            "rng": self.rng.bit_generator.state,
            "source": self.source.state(),
            "envs": self.envs,
            "episode_return": self.episode_return,
            "hidden": self.hidden,
            "resets": self.resets,
            "features": self.features,
            "counters": {
                "transitions": self.transitions,
                "episodes_started": self.episodes_started,
                "updates": self.updates,
                "wall": self.wall,
            },
            "history": self.history,
            "curve": self.curve,
            "completed": self.completed,
            "level_transitions": self.level_transitions,
        }

    def save(self, path):
        path = Path(path)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as handle:
            pickle.dump(self.state(), handle, protocol=5)
        tmp.replace(path)

    def load(self, path):
        with open(path, "rb") as handle:
            state = pickle.load(handle)
        if state["config_sha256"] != config_hash(self.config):
            raise ValueError("checkpoint configuration mismatch")
        self.model.params = {k: v.copy() for k, v in state["params"].items()}
        self.initial_params = state["initial_params"]
        self.optimizer.load(state["optimizer"])
        self.rng.bit_generator.state = state["rng"]
        self.source.load(state["source"])
        self.envs = state["envs"]
        self.episode_return = state["episode_return"]
        self.hidden = state["hidden"]
        self.resets = state["resets"]
        self.features = state["features"]
        for key, value in state["counters"].items():
            setattr(self, key, value)
        self.history, self.curve = state["history"], state["curve"]
        self.completed = state["completed"]
        self.level_transitions = state["level_transitions"]
        return state


# ------------------------------------------------------------------ evaluation
def _episode_record(env, actions, total):
    counts = np.bincount(actions, minlength=env.actions)
    p = counts / max(counts.sum(), 1)
    return {
        **env.metrics(),
        "return": float(total),
        "action_counts": counts.tolist(),
        "action_entropy": float(-(p * np.log(p + 1e-15)).sum()),
    }


def evaluate_policy(
    model, params, source, task, seeds, *, noise=0, edge_mask=None, neuron_mask=None
):
    """Greedy, learning-free evaluation on the exact target environment.

    Uses its own feature-source state and a fresh recurrent state, never the
    trainer's, and works on a copy of the parameters so nothing can be updated.
    """
    evaluator = ActorCritic(
        model.input_width, model.actions, arch=model.arch, hidden=model.hidden, core=model.core
    )
    evaluator.params = {k: v.copy() for k, v in params.items()}
    rows = [None] * len(seeds)
    batch = source.batch
    for start in range(0, len(seeds), batch):
        block = list(seeds[start : start + batch])
        envs = [make_env(task, s) for s in block]
        for c, s in enumerate(block):
            source.reset(c, s)
        hidden = evaluator.initial_state(len(block))
        actions = [[] for _ in block]
        totals = np.zeros(len(block))
        entropies = [[] for _ in block]
        while not all(e.done for e in envs):
            live = [c for c, e in enumerate(envs) if not e.done]
            phi = source(
                np.stack([envs[c].observe() for c in live]),
                live,
                noise=noise,
                edge_mask=edge_mask,
                neuron_mask=neuron_mask,
            )
            state = None if hidden is None else hidden[live]
            logits, _, new_state = evaluator.step(phi, state)
            if hidden is not None:
                hidden[live] = new_state
            for k, c in enumerate(live):
                lp = logits[k] - np.max(logits[k])
                lp -= np.log(np.exp(lp).sum())
                entropies[c].append(float(-(np.exp(lp) * lp).sum()))
                action = int(np.argmax(logits[k]))
                actions[c].append(action)
                _, reward, _, _ = envs[c].step(action)
                totals[c] += reward
        for c, env in enumerate(envs):
            rows[start + c] = {
                **_episode_record(env, actions[c], totals[c]),
                "policy_entropy": float(np.mean(entropies[c])),
            }
    return rows


def evaluate_baseline(task, seeds, kind):
    rows = []
    for s in seeds:
        env = make_env(task, s)
        behaviour = np.random.default_rng(s + BEHAVIOUR_OFFSET)
        actions, total = [], 0.0
        while not env.done:
            if kind == "random":
                action = int(behaviour.integers(env.actions))
            elif kind == "reference":
                action = reference_action(env)
            elif kind == "heuristic_v12":
                action = int(env.heuristic())
            else:
                raise ValueError(kind)
            actions.append(action)
            total += env.step(action)[1]
        rows.append(_episode_record(env, actions, total))
    return rows


def state_probe(model, params, source, task, probe_purpose, seed, *, repeats=4):
    """Fixed-history input intervention (v1.2 definition, extended to recurrence).

    States are sampled every fifth step of ten reference-policy episodes. For each
    state the neural core and recurrent state are reset, the neural RNG is fixed
    identically, the same observation is presented ``repeats`` times, and the greedy
    action is recorded. Only the observation differs between probes.
    """
    states = []
    for ep in range(10):
        env = make_env(task, seed_for(task, probe_purpose, seed, ep))
        for step in range(20):
            if env.done:
                break
            if step % 5 == 0:
                states.append(env.observe())
            env.step(reference_action(env))
    evaluator = ActorCritic(
        model.input_width, model.actions, arch=model.arch, hidden=model.hidden, core=model.core
    )
    evaluator.params = {k: v.copy() for k, v in params.items()}
    actions = []
    fixed = seed_for(task, probe_purpose, seed, 999)
    for start in range(0, len(states), source.batch):
        block = states[start : start + source.batch]
        columns = list(range(len(block)))
        for c in columns:
            source.reset_with_rng(c, rng_for_episode(fixed))
        hidden = evaluator.initial_state(len(block))
        for _ in range(repeats):
            phi = source(np.stack(block), columns)
            logits, _, hidden = evaluator.step(phi, hidden)
        actions.extend(int(a) for a in logits.argmax(axis=1))
    counts = np.bincount(actions, minlength=TARGETS[task].actions)
    return {
        "score": float(1 - counts.max() / counts.sum()),
        "action_counts": counts.tolist(),
        "states": len(states),
        "definition": "greedy action diversity under fixed-history state intervention",
    }


def mean_success(rows):
    return float(np.mean([r["success"] for r in rows]))


def entropy_of_policy(model, params, features):
    evaluator = ActorCritic(
        model.input_width, model.actions, arch=model.arch, hidden=model.hidden, core=model.core
    )
    evaluator.params = params
    logits, _, _ = evaluator.step(features, evaluator.initial_state(len(features)))
    logp = logits - logits.max(1, keepdims=True)
    logp = logp - np.log(np.exp(logp).sum(1, keepdims=True))
    return float(-(np.exp(logp) * logp).sum(1).mean())
