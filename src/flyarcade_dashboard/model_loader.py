"""Discover the frozen trained policies and wrap them as evaluation-only agents.

Every agent exposes the same tiny interface:

* ``reset(env_seed)``    - fresh environment and fresh neural / recurrent state
* ``decide(observation)`` - one greedy policy decision; returns per-neuron spike
                           counts for that decision, action probabilities, action
* ``parameter_digest()``  - hash of every weight the agent could possibly use

Agents follow the frozen evaluation code paths exactly (``greedy_rows`` for v1.1
Catch/Dodge, ``snake_experiment.evaluate`` for v1.1 Snake, ``evaluate_policy`` for
v1.3 Pong/Flappy) and never call a method that learns. Files are opened read-only.
"""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
POPULATIONS = ("visual_projection", "cb_intrinsic", "descending_neuron")


def repo_roots():
    """This checkout plus every git worktree of the same repository (read-only use)."""
    roots = [PACKAGE_ROOT]
    try:
        listing = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            cwd=PACKAGE_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
        for line in listing.splitlines():
            if line.startswith("worktree "):
                path = Path(line.split(" ", 1)[1])
                if path not in roots:
                    roots.append(path)
    except (OSError, subprocess.SubprocessError):
        pass
    return roots


def find(relative, roots=None):
    for root in roots or repo_roots():
        path = root / relative
        if path.exists():
            return path
    return None


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(arrays):
    h = hashlib.sha256()
    for name in sorted(arrays):
        h.update(name.encode() + np.ascontiguousarray(arrays[name]).tobytes())
    return h.hexdigest()


def load_graph(roots=None):
    from flyarcade.connectome import load_graph as _load

    path = find("data/malecns-v1.0/graph.npz", roots)
    if path is None:
        raise FileNotFoundError("data/malecns-v1.0/graph.npz not found in any worktree")
    return _load(path)


def softmax(logits):
    z = np.asarray(logits, dtype=float)
    z = np.exp(z - z.max())
    return z / z.sum()


# --------------------------------------------------------------------------- v1.1
class V11Agent:
    """v1.1 linear actor-critic over the frozen core (Catch, Dodge, Snake)."""

    ticks = 4

    def __init__(self, task, controller, make_env, info):
        self.task, self.controller, self.make_env, self.info = task, controller, make_env, info
        self._counts = np.zeros(controller.graph.n)
        tick = controller.core.tick

        def recording_tick(*args, **kwargs):
            spikes = tick(*args, **kwargs)
            self._counts += spikes
            return spikes

        # Instance-level wrapper: observes spikes, changes nothing about the dynamics.
        controller.core.tick = recording_tick
        self.env = None

    @property
    def action_count(self):
        return self.controller.motor.actions

    def reset(self, env_seed):
        self.env = self.make_env(env_seed)
        self.controller.reset()
        self.controller.rng = np.random.default_rng(env_seed + 500_000)
        return self.env

    def decide(self, observation):
        self._counts[:] = 0
        features = self.controller.rates(observation)  # plastic=False by default
        _, probabilities, _ = self.controller.motor.policy(features)
        action = int(probabilities.argmax())
        counts = np.rint(self._counts).astype(np.uint8)
        return {
            "counts": counts,
            "ticks": self.ticks,
            "probs": probabilities.tolist(),
            "action": action,
        }

    def parameter_digest(self):
        c = self.controller
        return _digest(
            {
                "actor": c.motor.actor,
                "critic": c.motor.critic,
                "magnitude": c.core.magnitude,
                "base": c.core.base,
                "signs": c.core.signs,
            }
        )


def _load_v11_lane(task, seed, graph, roots):
    from flyarcade.games import LaneGame
    from flyarcade.v11.controller import V11Controller
    from flyarcade.v11.features import Standardizer

    run = find(f"runs/v11-{task}-eprop-{seed}", roots)
    if run is None:
        return None
    result = json.loads((run / "result.json").read_text())
    with np.load(run / "checkpoint.npz", allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    graph_digest = hashlib.sha256(
        graph.pre.tobytes() + graph.post.tobytes() + graph.counts.tobytes()
    ).hexdigest()
    if graph_digest != result["graph_sha256"]:
        raise ValueError(f"{task}: graph does not match the trained model")
    std = Standardizer(arrays["standardizer_mean"], arrays["standardizer_scale"])
    if (
        hashlib.sha256(std.mean.tobytes() + std.scale.tobytes()).hexdigest()
        != result["standardizer_sha256"]
    ):
        raise ValueError(f"{task}: standardizer does not match the trained model")
    config = result["config"]
    controller = V11Controller(
        graph, seed, stage=config["stage"], standardizer=std, **config["hyperparameters"]
    )
    controller.motor.actor = arrays["actor"]
    controller.motor.critic = arrays["critic"]
    controller.core.magnitude = arrays["magnitude"]
    info = {
        "study": "v1.1 (frozen confirmatory, branch flyarcade-v1.1-learning)",
        "checkpoint": str(run / "checkpoint.npz"),
        "result": str(run / "result.json"),
        "architecture": "linear actor-critic readout (stage A, recurrent core frozen)",
        "ticks": 4,
        "readout": "descending neurons (1,293)",
        "training_seed": seed,
        "encoding": "v1.1 lane encoding",
    }
    return V11Agent(task, controller, lambda s: LaneGame(task, s), info)


def _load_v11_snake(seed, graph, roots):
    from flyarcade.v11.controller import V11Controller
    from flyarcade.v11.features import Standardizer
    from flyarcade.v11.snake import CHANNELS, SnakeGame
    from flyarcade.v11.snake import features as snake_features

    run = find(f"runs/snake-eprop-{seed}", roots)
    if run is None:
        return None
    result = json.loads((run / "result.json").read_text())
    std_path = None
    for root in roots:
        candidate = root / result["standardizer_path"]
        if candidate.exists():
            std = Standardizer.load(candidate)
            if (
                hashlib.sha256(std.mean.tobytes() + std.scale.tobytes()).hexdigest()
                == result["standardizer_sha256"]
            ):
                std_path = candidate
                break
    if std_path is None:
        raise ValueError("snake: no standardizer matching the trained model")
    std = Standardizer.load(std_path)
    with np.load(run / "checkpoint.npz", allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in ("actor", "critic", "magnitude")}
    config = result["config"]
    controller = V11Controller(
        graph,
        seed,
        stage="readout",
        standardizer=std,
        encoder=snake_features,
        channels=CHANNELS,
        observation_width=CHANNELS,
        actions=4,
        **config["hyperparameters"],
    )
    controller.motor.actor = arrays["actor"]
    controller.motor.critic = arrays["critic"]
    controller.core.magnitude = arrays["magnitude"]
    info = {
        "study": "v1.1 Snake (frozen confirmatory; 8x8 absolute-action Snake)",
        "checkpoint": str(run / "checkpoint.npz"),
        "result": str(run / "result.json"),
        "architecture": "linear actor-critic readout (stage A, recurrent core frozen)",
        "ticks": 4,
        "readout": "descending neurons (1,293)",
        "training_seed": seed,
        "encoding": "v1.1 Snake 21-channel encoding",
    }
    return V11Agent("snake", controller, SnakeGame, info)


# --------------------------------------------------------------------------- v1.3
class V13Agent:
    """v1.3 MLP/GRU PPO policy over the frozen core (Pong, Flappy)."""

    def __init__(self, task, model, source, standardizer, info):
        self.task, self.model, self.source, self.std, self.info = (
            task,
            model,
            source,
            standardizer,
            info,
        )
        self.ticks = source.ticks
        self.hidden = None
        self.env = None

    @property
    def action_count(self):
        return self.model.actions

    def reset(self, env_seed):
        from flyarcade_v13.environments import make_env

        self.env = make_env(self.task, env_seed)
        self.source.reset(0, env_seed)
        self.hidden = self.model.initial_state(1)
        return self.env

    def decide(self, observation):
        # readout="all" advances exactly the same dynamics and RNG draws as the
        # descending readout; the descending subset is standardized exactly as
        # FlyFeatures.__call__ does, so the policy sees bit-identical features.
        raw = self.source.raw_rates(np.asarray(observation, dtype=float)[None], [0])[0]
        counts = np.rint((raw + 0.25) * self.ticks).astype(np.uint8)
        s = self.std
        descending = raw[self.source.outputs]
        phi = np.clip((descending - s.mean) / s.scale, -s.clip, s.clip) * s.gain
        logits, _, self.hidden = self.model.step(phi[None], self.hidden)
        action = int(np.argmax(logits[0]))
        return {
            "counts": counts,
            "ticks": self.ticks,
            "probs": softmax(logits[0]).tolist(),
            "action": action,
        }

    def parameter_digest(self):
        return _digest(
            {
                **{f"policy_{k}": v for k, v in self.model.params.items()},
                "core_base": self.source.base_magnitude,
                "core_weights": np.asarray(self.source.weights),
            }
        )


_RESCUE = {
    "pong": ("runs/v13-pong-rescue/confirm/pong-confirm-biological-s{seed}", "v1.3 Pong rescue"),
    "flappy": (
        "runs/v13-flappy-rescue/confirm/flappy-confirm-biological-s{seed}",
        "v1.3 Flappy rescue",
    ),
}


def _load_v13(task, seed, graph, roots):
    from flyarcade.v11.features import Standardizer
    from flyarcade_v13.controller import FlyFeatures
    from flyarcade_v13.environments import TARGETS
    from flyarcade_v13.policy import ActorCritic
    from flyarcade_v13.study import graph_hash, standardizer_path

    pattern, study = _RESCUE[task]
    run = find(pattern.format(seed=seed), roots)
    if run is None:
        return None
    result = json.loads((run / "result.json").read_text())
    config = result["config"]
    if graph_hash(graph) != result["graph_sha256"]:
        raise ValueError(f"{task}: graph does not match the trained model")
    relative = config.get("standardizer") or str(
        standardizer_path(task, "biological", config["ticks"], config["readout"])
    )
    std_file = None
    for root in roots:
        candidate = root / relative
        if candidate.exists() and sha256_file(candidate) == result["standardizer_sha256"]:
            std_file = candidate
            break
    if std_file is None:
        raise ValueError(f"{task}: no standardizer matching the trained model")
    std = Standardizer.load(std_file)
    width = TARGETS[task].width
    model = ActorCritic(
        1293,
        TARGETS[task].actions,
        arch=config["arch"],
        hidden=config["hidden"],
        core=config["core"],
    )
    with np.load(run / "policy.npz", allow_pickle=False) as z:
        model.params = {k: z[f"final_{k}"].copy() for k in model.params}
    from flyarcade_v13.study import params_hash

    if params_hash(model.params) != result["final_params_sha256"]:
        raise ValueError(f"{task}: policy parameters do not match the frozen result")
    source = FlyFeatures(graph, width, 1, ticks=config["ticks"], readout="all")
    info = {
        "study": f"{study} (frozen confirmatory)",
        "checkpoint": str(run / "policy.npz"),
        "result": str(run / "result.json"),
        "architecture": (
            "Dense 128 tanh -> GRU 64 -> actor/critic (PPO)"
            if config["arch"] == "gru"
            else "Dense 128 tanh -> Dense 64 tanh -> actor/critic (PPO)"
        ),
        "ticks": int(config["ticks"]),
        "readout": "descending neurons (1,293)",
        "training_seed": seed,
        "encoding": "unchanged v1.2 encoding",
    }
    return V13Agent(task, model, source, std, info)


UNAVAILABLE = {
    "breakout": (
        "Trained model unavailable: no Breakout policy passed its frozen criterion "
        "(v1.2 biological 0.368 was below random 0.468); v1.3 Breakout has no frozen model."
    ),
}


def available_seeds(task, roots=None):
    roots = roots or repo_roots()
    patterns = {
        "catch": "runs/v11-catch-eprop-{seed}/checkpoint.npz",
        "dodge": "runs/v11-dodge-eprop-{seed}/checkpoint.npz",
        "snake": "runs/snake-eprop-{seed}/checkpoint.npz",
        "pong": _RESCUE["pong"][0] + "/policy.npz",
        "flappy": _RESCUE["flappy"][0] + "/policy.npz",
    }
    if task not in patterns:
        return []
    return [s for s in range(5) if find(patterns[task].format(seed=s), roots)]


def load_agent(task, seed, graph, roots=None):
    roots = roots or repo_roots()
    if task in ("catch", "dodge"):
        return _load_v11_lane(task, seed, graph, roots)
    if task == "snake":
        return _load_v11_snake(seed, graph, roots)
    if task in _RESCUE:
        return _load_v13(task, seed, graph, roots)
    return None
