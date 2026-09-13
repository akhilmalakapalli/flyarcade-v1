"""v1.5 resumable PPO trainer, evaluation and probes on the fixed MaleCNS substrate.

Built on the unchanged v1.3 PPO/GAE and actor-critic code and the v1.4 environment
adapters. Additions: readout populations, sensory control, temporal stacking, entropy
and LR schedules, explained variance, and within-run checkpoint selection on a
dedicated ``checkpoint_select`` seed block (never the development score block).
"""

import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np

from flyarcade.v11.features import Standardizer
from flyarcade_v13.policy import Adam
from flyarcade_v13.ppo import gae, sample_actions, update
from flyarcade_v14.environments import TARGETS, make_env, reference_action, terminal_kind
from flyarcade_v14.policy import ActorCritic
from flyarcade_v14.study import evaluate_baseline, evaluate_policy, load_topology
from flyarcade_v15.features import (
    Fly15,
    Sensory15,
    Stacked,
    load_standardizer,
    save_standardizer,
    standardizer_path,
)
from flyarcade_v15.seeds import BEHAVIOUR_OFFSET, seed_for

TRAINING_PURPOSES = ("train", "diagnosis_training")

DEFAULTS = {
    "source": "fly",
    "readout": "descending",
    "ticks": 4,
    "stack": 1,
    "std_floor": 0.02,
    "arch": "linear",
    "hidden": 128,
    "core": 64,
    "gamma": 0.99,
    "lam": 0.95,
    "lr": 3e-4,
    "lr_schedule": "linear",
    "lr_floor": 0.05,
    "clip": 0.2,
    "ent": 0.003,
    "ent_final": None,
    "vf": 0.5,
    "max_grad_norm": 0.5,
    "epochs": 4,
    "envs": 16,
    "steps": 128,
    "minibatches": 8,
    "chunk": 16,
    "transitions": 65536,
    "curriculum": [["target", 1.0]],
    "imitation": 0,
    "eval_every": 16384,
    "select_episodes": 32,
    "checkpoint_selection": "best",
    "train_purpose": "train",
    "seed": 0,
}


def complete_config(config):
    c = {**DEFAULTS, **config}
    if c["task"] not in TARGETS:
        raise ValueError("unknown task")
    if c["source"] not in ("fly", "sensory"):
        raise ValueError("source must be fly or sensory")
    if c["train_purpose"] not in TRAINING_PURPOSES:
        raise ValueError("training may only use development/diagnosis training seeds")
    if c["checkpoint_selection"] not in ("best", "final"):
        raise ValueError("checkpoint_selection must be best or final")
    fractions = [f for _, f in c["curriculum"]]
    if c["curriculum"][-1][0] != "target" or abs(sum(fractions) - 1) > 1e-9:
        raise ValueError("curriculum must end on the target and cover the whole budget")
    if c["steps"] % c["chunk"]:
        raise ValueError("rollout steps must be a multiple of the BPTT chunk")
    return c


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def params_hash(params):
    h = hashlib.sha256()
    for key in sorted(params):
        h.update(key.encode() + np.ascontiguousarray(params[key]).tobytes())
    return h.hexdigest()


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


# ------------------------------------------------------------------ standardizers
def fit_standardizer(task, ticks, readout, floor, *, min_samples=8192, batch=16, guard=None):
    graph = load_topology("biological")
    source = Fly15(graph, task, batch, ticks=ticks, readout=readout)
    seeds, samples = [], []
    while len(samples) < min_samples:
        block = [seed_for(task, "feature_fit", index=len(seeds) + i) for i in range(batch)]
        seeds.extend(block)
        envs = [make_env(task, s) for s in block]
        behaviour = [np.random.default_rng(s + BEHAVIOUR_OFFSET) for s in block]
        for col, s in enumerate(block):
            source.reset(col, s)
        while not all(e.done for e in envs):
            if guard is not None:
                guard.check()
            live = [col for col, e in enumerate(envs) if not e.done]
            samples.extend(source.raw_rates(np.stack([envs[col].observe() for col in live]), live))
            for col in live:
                env = envs[col]
                action = (
                    int(behaviour[col].integers(env.actions))
                    if env.time % 2
                    else int(reference_action(env))
                )
                env.step(action)
    samples = np.asarray(samples)
    std = Standardizer(
        samples.mean(0), np.maximum(samples.std(0), floor), clip=4.0, normalise=False
    )
    meta = {
        "task": task,
        "ticks": ticks,
        "readout": readout,
        "floor": floor,
        "samples": int(len(samples)),
        "seeds": seeds,
        "purpose": "feature_fit (v1.5 development)",
        "policy": "alternating reference and uniform-random actions",
    }
    return std, meta


def ensure_standardizer(task, ticks, readout, floor, guard=None):
    path = standardizer_path(task, ticks, readout, floor)
    if not path.exists():
        std, meta = fit_standardizer(task, ticks, readout, floor, guard=guard)
        save_standardizer(path, std, meta)
    return path


def make_source(config, batch):
    c = config
    if c["source"] == "sensory":
        base = Sensory15(c["task"], batch)
    else:
        path = standardizer_path(c["task"], c["ticks"], c["readout"], c["std_floor"])
        base = Fly15(
            load_topology("biological"),
            c["task"],
            batch,
            ticks=c["ticks"],
            readout=c["readout"],
            standardizer=load_standardizer(path),
        )
    return Stacked(base, c["stack"]) if c["stack"] > 1 else base


def build_model(config, width):
    return ActorCritic(
        width,
        TARGETS[config["task"]].actions,
        arch=config["arch"],
        hidden=config["hidden"],
        core=config["core"],
        seed=seed_for(config["task"], "model", config["seed"]),
    )


# ------------------------------------------------------------------ trainer
class Trainer:
    def __init__(self, config):
        self.config = complete_config(config)
        c = self.config
        self.task = c["task"]
        self.source = make_source(c, c["envs"])
        self.model = build_model(c, self.source.width)
        self.initial_params = self.model.copy_params()
        self.optimizer = Adam(self.model.params, lr=c["lr"])
        self.rng = np.random.default_rng(seed_for(self.task, "rollout", c["seed"]))
        self.transitions = 0
        self.episodes_started = 0
        self.updates = 0
        self.history = []
        self.curve = []
        self.completed = []
        self.best = None  # {"transitions", "score", "params"}
        self.level_transitions = {name: 0 for name, _ in c["curriculum"]}
        self.envs = [None] * c["envs"]
        self.episode_return = np.zeros(c["envs"])
        self.hidden = self.model.initial_state(c["envs"])
        self.resets = np.ones(c["envs"])
        for column in range(c["envs"]):
            self._start_episode(column)
        self.features = self._observe(range(c["envs"]))
        self.wall = 0.0

    def level(self):
        c, spent = self.config, 0.0
        for name, fraction in c["curriculum"]:
            spent += fraction * c["transitions"]
            if self.transitions < spent - 1e-9:
                return name
        return c["curriculum"][-1][0]

    def _start_episode(self, column):
        c = self.config
        seed = seed_for(self.task, c["train_purpose"], c["seed"], self.episodes_started)
        self.episodes_started += 1
        level = self.level()
        self.envs[column] = make_env(self.task, seed, level)
        self.envs[column].v15_level = level
        self.source.reset(column, seed)
        self.episode_return[column] = 0.0
        self.resets[column] = 1.0

    def _observe(self, columns):
        columns = list(columns)
        return self.source(np.stack([self.envs[i].observe() for i in columns]), columns)

    def collect(self):
        c = self.config
        steps, envs, width = c["steps"], c["envs"], self.source.width
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
            if hidden is not None:
                # the reset applied inside step already zeroed finished rows
                pass
            for i, env in enumerate(self.envs):
                _, reward, done, info = env.step(int(actions[i]))
                kind = terminal_kind(env)
                self.episode_return[i] += reward
                batch["rewards"][t, i] = reward
                batch["dones"][t, i] = float(done)
                self.transitions += 1
                self.level_transitions[env.v15_level] += 1
                if done:
                    if kind == "truncated":
                        batch["rewards"][t, i] += c["gamma"] * self._bootstrap_value(i, hidden)
                    self.completed.append(
                        {**info, "return": float(self.episode_return[i]), "level": env.v15_level}
                    )
            self.hidden = hidden
            for i in [i for i, env in enumerate(self.envs) if env.done]:
                self._start_episode(i)
            self.features = self._observe(range(envs))
        _, last_values, _ = self.model.step(self.features, self.hidden, self.resets)
        adv, ret = gae(
            batch["rewards"], batch["values"], batch["dones"], last_values, c["gamma"], c["lam"]
        )
        batch["advantages"], batch["returns"] = adv, ret
        return batch

    def _bootstrap_value(self, column, hidden):
        """V(s_T) at a time-limit stop, on a throwaway copy of that column's state."""
        saved = self.source.state()
        phi = self.source(self.envs[column].observe()[None], [column])
        self.source.load(saved)
        state = None if hidden is None else hidden[column : column + 1]
        _, value, _ = self.model.step(phi, state, None)
        return float(value[0])

    def schedule(self):
        c = self.config
        frac = min(self.transitions / c["transitions"], 1.0)
        lr = c["lr"] * (max(1.0 - frac, c["lr_floor"]) if c["lr_schedule"] == "linear" else 1.0)
        ent_final = c["ent"] if c["ent_final"] is None else c["ent_final"]
        ent = c["ent"] + (ent_final - c["ent"]) * frac
        return lr, ent

    def train_step(self):
        c = self.config
        started = time.monotonic()
        batch = self.collect()
        lr, ent = self.schedule()
        stats = update(self.model, self.optimizer, batch, {**c, "ent": ent}, self.rng, lr=lr)
        values, returns = batch["values"].ravel(), batch["returns"].ravel()
        var = returns.var()
        explained = float(1 - (returns - values).var() / var) if var > 1e-12 else 0.0
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
            "ent_coef": ent,
            "wall_seconds": self.wall,
            "transitions_per_second": c["steps"] * c["envs"] / max(elapsed, 1e-9),
            "train_success": float(np.mean([r["success"] for r in recent])) if recent else None,
            "train_return": float(np.mean([r["return"] for r in recent])) if recent else None,
            "rollout_action_counts": np.bincount(
                batch["actions"].ravel(), minlength=TARGETS[self.task].actions
            ).tolist(),
            "explained_variance": explained,
            **stats,
            "finite": self.model.finite(),
        }
        if not row["finite"]:
            raise FloatingPointError("non-finite policy parameters")
        self.history.append(row)
        return row

    def checkpoint_due(self):
        return self.transitions % self.config["eval_every"] == 0 or self.done()

    def evaluate_checkpoint(self):
        c = self.config
        seeds = [
            seed_for(self.task, "checkpoint_select", c["seed"], i)
            for i in range(c["select_episodes"])
        ]
        source = make_source(c, min(16, len(seeds)))
        rows = evaluate_policy(self.model, self.model.params, source, self.task, seeds)
        score = float(np.mean([r["success"] for r in rows]))
        self.curve.append({"transitions": self.transitions, "score": score})
        if self.best is None or score >= self.best["score"]:
            self.best = {
                "transitions": self.transitions,
                "score": score,
                "params": self.model.copy_params(),
            }
        return score

    def done(self):
        return self.transitions >= self.config["transitions"]

    def chosen_params(self):
        if self.config["checkpoint_selection"] == "best" and self.best is not None:
            return self.best["params"], self.best["transitions"]
        return self.model.params, self.transitions

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
            "best": self.best,
            "completed": self.completed[-200:],
            "level_transitions": self.level_transitions,
        }

    def load(self, state):
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
        self.history, self.curve, self.best = state["history"], state["curve"], state["best"]
        self.completed = state["completed"]
        self.level_transitions = state["level_transitions"]


# ------------------------------------------------------------------ evaluation
def evaluate(config, params, seeds, *, batch=16):
    c = complete_config(config)
    source = make_source(c, batch)
    model = build_model(c, source.width)
    return evaluate_policy(model, params, source, c["task"], seeds)


def state_probe(config, params, purpose, seed, *, repeats=4):
    """Fixed-history input intervention (v1.2/v1.4 definition) on v1.5 probe seeds."""
    c = complete_config(config)
    task = c["task"]
    states = []
    for ep in range(10):
        env = make_env(task, seed_for(task, purpose, seed, ep))
        for step in range(20):
            if env.done:
                break
            if step % 5 == 0:
                states.append(env.observe())
            env.step(int(reference_action(env)))
    source = make_source(c, 16)
    model = build_model(c, source.width)
    model.params = {k: v.copy() for k, v in params.items()}
    actions = []
    fixed = seed_for(task, purpose, seed, 999)
    for start in range(0, len(states), source.batch):
        block = states[start : start + source.batch]
        columns = list(range(len(block)))
        for col in columns:
            source.reset_with_rng(col, np.random.default_rng(fixed + 1_000_000_000))
        hidden = model.initial_state(len(block))
        for _ in range(repeats):
            logits, _, hidden = model.step(source(np.stack(block), columns), hidden)
        actions.extend(int(a) for a in logits.argmax(axis=1))
    counts = np.bincount(actions, minlength=TARGETS[task].actions)
    return {
        "score": float(1 - counts.max() / counts.sum()),
        "action_counts": counts.tolist(),
        "states": len(states),
        "definition": "greedy action diversity under fixed-history state intervention",
    }


def summarize_rows(rows):
    counts = np.sum([r["action_counts"] for r in rows], axis=0)
    return {
        "score": float(np.mean([r["success"] for r in rows])),
        "action_distribution": (counts / counts.sum()).tolist(),
        "dominant_action_fraction": float(counts.max() / counts.sum()),
        "policy_entropy": float(np.mean([r["policy_entropy"] for r in rows])),
    }


__all__ = [
    "Trainer",
    "evaluate",
    "evaluate_baseline",
    "state_probe",
    "summarize_rows",
    "ensure_standardizer",
    "make_source",
    "build_model",
    "pickle",
]
