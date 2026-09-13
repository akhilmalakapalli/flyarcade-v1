"""One evaluation-only dashboard session: one task, one frozen model, one episode.

A *frame* is exactly one policy decision, in this order:
observation_t -> fly network activity -> policy probabilities -> greedy action
-> environment transition -> reward. Every field of a frame refers to that step.
"""

import base64
import threading

import numpy as np

from flyarcade_dashboard import ACTIVITY_NOTE, DEMO_LABEL, NETWORK_LABEL
from flyarcade_dashboard import model_loader as ml
from flyarcade_dashboard.tasks import (
    ACTION_LABELS,
    TASK_ORDER,
    TITLES,
    calibration_seed,
    demo_seed,
    render_state,
)

POP_KEYS = {"visual_projection": "vp", "cb_intrinsic": "cx", "descending_neuron": "dn"}
MAX_EPISODE_FRAMES = 2000  # every task horizon is far below this; hard memory bound


def _metrics(task, env, score_total):
    if task in ("catch", "dodge"):
        landings = env.time // 4
        return {
            "score": int(score_total),
            "landings": int(landings),
            "success": float(score_total / max(landings, 1)),
            "steps": int(env.time),
        }
    if task == "snake":
        return {
            "food": int(env.eaten),
            "length": len(env.body),
            "steps": int(env.time),
            "death": env.death,
        }
    return {
        k: (v if isinstance(v, (str, type(None), bool)) else float(v))
        for k, v in env.metrics().items()
    }


class Network:
    """Static metadata and fixed layouts for the 2,040 neurons (computed once)."""

    def __init__(self, graph):
        by_id = {n["bodyId"]: n for n in graph.provenance["neurons"]}
        ordered = [by_id[int(i)] for i in graph.neuron_ids]
        self.n = graph.n
        self.population = [n.get("superclass") for n in ordered]
        indeg = np.bincount(graph.post, minlength=graph.n)
        outdeg = np.bincount(graph.pre, minlength=graph.n)
        # Fixed within-population order: annotated type, then body ID. Never re-sorted.
        self.order = {
            pop: sorted(
                (i for i, p in enumerate(self.population) if p == pop),
                key=lambda i: (str(ordered[i].get("type") or "~"), int(graph.neuron_ids[i])),
            )
            for pop in ml.POPULATIONS
        }
        self.meta = []
        for i, n in enumerate(ordered):
            self.meta.append(
                {
                    "index": i,
                    "bodyId": int(graph.neuron_ids[i]),
                    "population": self.population[i],
                    "type": n.get("type") or None,
                    "instance": n.get("instance") or None,
                    "consensusNt": n.get("consensusNt") or None,
                    "somaSide": n.get("somaSide") or None,
                    "in_degree": int(indeg[i]),
                    "out_degree": int(outdeg[i]),
                }
            )
        self.layout2d = self._layout2d()
        self.layout3d, self.missing_soma = self._layout3d(ordered)

    def _layout2d(self):
        """Three side-by-side population blocks, square-ish grids, fixed positions."""
        positions = np.zeros((self.n, 2))
        widths = {"visual_projection": 0.18, "cb_intrinsic": 0.30, "descending_neuron": 0.46}
        left = 0.02
        for pop in ml.POPULATIONS:
            members = self.order[pop]
            w = widths[pop]
            cols = max(1, int(np.ceil(np.sqrt(len(members) * w / 0.9))))
            rows = int(np.ceil(len(members) / cols))
            for k, i in enumerate(members):
                r, c = divmod(k, cols)
                positions[i] = (left + (c + 0.5) * w / cols, 0.06 + (r + 0.5) * 0.9 / rows)
            left += w + 0.02
        return positions.round(4).tolist()

    @staticmethod
    def _layout3d(ordered):
        coords = [
            (n.get("somaLocation") or {}).get("coordinates") if n.get("somaLocation") else None
            for n in ordered
        ]
        known = np.array([c for c in coords if c], dtype=float)
        center, scale = known.mean(0), np.abs(known - known.mean(0)).max()
        missing = []
        out = []
        pops = [n.get("superclass") for n in ordered]
        centroid = {
            p: known_pop.mean(0)
            for p in set(pops)
            if len(
                known_pop := np.array(
                    [c for c, q in zip(coords, pops, strict=True) if c and q == p]
                )
            )
        }
        for i, c in enumerate(coords):
            if c:
                out.append(((np.array(c, dtype=float) - center) / scale).round(4).tolist())
            else:
                missing.append(i)
                out.append(((centroid[pops[i]] - center) / scale).round(4).tolist())
        return out, missing

    def payload(self):
        return {
            "label": NETWORK_LABEL,
            "note": ACTIVITY_NOTE,
            "n": self.n,
            "populations": {pop: len(self.order[pop]) for pop in ml.POPULATIONS},
            "order": self.order,
            "meta": self.meta,
            "layout2d": self.layout2d,
            "layout3d": self.layout3d,
            "layout3d_label": (
                "soma positions from MaleCNS somaLocation (normalized); "
                f"{len(self.missing_soma)} neurons without a soma location are drawn hollow "
                "at their population centroid"
            ),
            "missing_soma": self.missing_soma,
            "degree_note": "degrees count synaptic partners within the 2,040-neuron subgraph",
        }


class Dashboard:
    """Thread-safe holder of the graph, loaded agents and the single live session."""

    def __init__(self):
        self.lock = threading.RLock()
        self.roots = ml.repo_roots()
        self.graph = ml.load_graph(self.roots)
        self.network = Network(self.graph)
        self.agents = {}
        self.calibration = {}
        self.task = None
        self.model_seed = 0
        self.demo_index = 0
        self.agent = None
        self.env = None
        self.frames = []
        self.step_index = 0
        self.episode_return = 0.0
        self.score_total = 0
        self.population_index = {
            key: np.array(self.network.order[pop]) for pop, key in POP_KEYS.items()
        }

    # ------------------------------------------------------------------ tasks
    def tasks(self):
        out = []
        for task in TASK_ORDER:
            seeds = ml.available_seeds(task, self.roots)
            out.append(
                {
                    "task": task,
                    "title": TITLES[task],
                    "actions": ACTION_LABELS[task],
                    "available": bool(seeds),
                    "model_seeds": seeds,
                    "unavailable_reason": None
                    if seeds
                    else ml.UNAVAILABLE.get(task, "Trained model unavailable"),
                }
            )
        return out

    def _agent(self, task, seed):
        key = (task, seed)
        if key not in self.agents:
            agent = ml.load_agent(task, seed, self.graph, self.roots)
            if agent is None:
                return None
            self.agents[key] = agent
            self.calibration[key] = self._calibrate(task, agent)
        return self.agents[key]

    def _calibrate(self, task, agent):
        """Fixed per-population activity scale from one calibration demo episode."""
        env = agent.reset(calibration_seed(task))
        fractions = []
        while not env.done and len(fractions) < 400:
            decision = agent.decide(env.observe())
            fractions.append(decision["counts"] / decision["ticks"])
            env.step(decision["action"])
        stack = np.array(fractions)
        scales = {}
        for key, index in self.population_index.items():
            scales[key] = float(max(np.percentile(stack[:, index], 99.9), 1.0 / agent.ticks))
        return {
            "scales": scales,
            "seed": calibration_seed(task),
            "steps": len(fractions),
            "statistic": "99.9th percentile of per-neuron spike fraction per population",
        }

    # --------------------------------------------------------------- session
    def new_session(self, task, model_seed=None, demo_index=0):
        with self.lock:
            if task not in TASK_ORDER:
                raise ValueError(f"unknown task {task}")
            seeds = ml.available_seeds(task, self.roots)
            self.task = task
            self.frames, self.step_index = [], 0
            self.episode_return, self.score_total = 0.0, 0
            self.demo_index = int(demo_index)
            if not seeds:
                # No trained model: show the untouched initial board only, never play it.
                from flyarcade_v13.environments import TARGETS

                self.agent, self.model_seed = None, None
                self.env = (
                    TARGETS[task](demo_seed(task, self.demo_index)) if task in TARGETS else None
                )
                return self.state()
            self.model_seed = int(model_seed) if model_seed is not None else seeds[0]
            if self.model_seed not in seeds:
                raise ValueError(f"model seed {self.model_seed} unavailable for {task}")
            self.agent = self._agent(task, self.model_seed)
            self.env = self.agent.reset(demo_seed(task, self.demo_index))
            return self.state()

    def reset(self):
        with self.lock:
            if self.task is None:
                raise ValueError("no session")
            return self.new_session(self.task, self.model_seed, self.demo_index)

    def new_demo(self):
        with self.lock:
            if self.task is None:
                raise ValueError("no session")
            return self.new_session(self.task, self.model_seed, self.demo_index + 1)

    def step(self):
        """Advance exactly one policy decision and return that frame."""
        with self.lock:
            if self.agent is None:
                raise ValueError("no trained model for this task")
            if self.env.done:
                return {"done": True, "frame": self.frames[-1] if self.frames else None}
            observation = np.asarray(self.env.observe(), dtype=float)
            before = render_state(self.task, self.env)
            decision = self.agent.decide(observation)
            outcome = self.env.step(decision["action"])
            reward, done = float(outcome[1]), bool(outcome[2])
            if self.task in ("catch", "dodge"):
                self.score_total += int(outcome[3]["score"])
            self.episode_return += reward
            counts = decision["counts"]
            fractions = counts / decision["ticks"]
            frame = {
                "task": self.task,
                "step": self.step_index,
                "demo_seed": demo_seed(self.task, self.demo_index),
                "observation": observation.tolist(),
                "counts_b64": base64.b64encode(counts.tobytes()).decode(),
                "ticks": decision["ticks"],
                "population_means": {
                    key: float(fractions[index].mean())
                    for key, index in self.population_index.items()
                },
                "probs": [float(p) for p in decision["probs"]],
                "action": decision["action"],
                "action_label": ACTION_LABELS[self.task][decision["action"]],
                "reward": reward,
                "episode_return": self.episode_return,
                "done": done,
                "metrics": _metrics(self.task, self.env, self.score_total),
                "render_before": before,
                "render_after": render_state(self.task, self.env),
            }
            self.step_index += 1
            if len(self.frames) < MAX_EPISODE_FRAMES:
                self.frames.append(frame)
            return {"done": done, "frame": frame}

    def state(self):
        with self.lock:
            agent = self.agent
            key = (self.task, self.model_seed)
            return {
                "task": self.task,
                "title": TITLES.get(self.task),
                "actions": ACTION_LABELS.get(self.task),
                "available": agent is not None,
                "unavailable_reason": None
                if agent is not None or self.task is None
                else ml.UNAVAILABLE.get(self.task, "Trained model unavailable"),
                "model": agent.info if agent else None,
                "model_seed": self.model_seed,
                "demo_index": self.demo_index,
                "demo_seed": demo_seed(self.task, self.demo_index) if self.task else None,
                "demo_label": DEMO_LABEL,
                "calibration": self.calibration.get(key),
                "step": self.step_index,
                "done": bool(self.env.done) if self.env is not None else False,
                "render": render_state(self.task, self.env) if self.env is not None else None,
                "last_frame": self.frames[-1] if self.frames else None,
            }
