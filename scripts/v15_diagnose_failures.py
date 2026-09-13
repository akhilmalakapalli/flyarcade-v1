"""Phase-1 failure-state diagnosis of locked v1.4 *development* checkpoints.

Replays each v1.4 final-selected development checkpoint (read-only) on fresh v1.5
``diagnosis`` seeds with step-level logging, alongside the reference controller and
uniform random play. Uses no v1.4 validation seed, no historical confirmatory seed and
no v1.3 confirmatory model. Writes nothing under artifacts/v14 or runs/v14.
"""

import collections
import json
import pickle
import sys
from pathlib import Path

import v15_path  # noqa: F401
import numpy as np
from flyarcade_v15.seeds import BEHAVIOUR_OFFSET, seed_for

from flyarcade_v14.environments import make_env, reference_action
from flyarcade_v14.study import build_model, make_source

OUT = Path("artifacts/v15/diagnosis/failures")
EPISODES = 100
INDEX_OFFSET = 200_000  # disjoint from representation-diagnosis episodes


def policy_actor(config, params):
    source = make_source(config, 1)
    model = build_model(config, source.width)
    model.params = {k: v.copy() for k, v in params.items()}
    state = {"hidden": None}

    def reset(seed):
        source.reset(0, seed)
        state["hidden"] = model.initial_state(1)

    def act(env):
        phi = source(env.observe()[None], [0])
        logits, _, state["hidden"] = model.step(phi, state["hidden"])
        return int(np.argmax(logits[0]))

    return reset, act


def run_episode(task, seed, actor, reset=None):
    env = make_env(task, seed)
    if reset:
        reset(seed)
    behaviour = np.random.default_rng(seed + BEHAVIOUR_OFFSET)
    log = []
    while not env.done:
        before = snapshot(task, env)
        if actor == "random":
            action = int(behaviour.integers(env.actions))
        elif actor == "reference":
            action = int(reference_action(env))
        else:
            action = actor(env)
        _, reward, _, info = env.step(action)
        log.append({**before, "action": action, "reward": reward, "reference": None})
    return env, log


def snapshot(task, env):
    if task in ("catch", "dodge"):
        return {"player": env.player, "object": env.object, "phase": env.phase, "score": env.score}
    if task == "pong":
        return {
            "paddle": env.paddle,
            "x": env.x,
            "y": env.y,
            "vy": env.vy,
            "intercept": env.intercept(),
            "attempts": env.attempts,
            "hits": env.hits,
        }
    if task == "flappy":
        return {"y": env.y, "vy": env.vy, "x": env.x, "gap": env.gap, "passed": env.passed}
    return {"length": len(env.body), "eaten": env.eaten, "time": env.time}


def analyse(task, episodes):
    """Task-specific failure breakdown from (env, log) pairs."""
    out = {"episodes": len(episodes)}
    if task in ("catch", "dodge"):
        by_distance = collections.defaultdict(lambda: [0, 0])
        by_start_lane = collections.defaultdict(lambda: [0, 0])
        for env, log in episodes:
            for t in range(0, len(log), 4):
                start = log[t]
                end_score = log[t + 3]["score"] if t + 3 < len(log) else None
                if end_score is None:
                    continue
                after = log[t + 4]["score"] if t + 4 < len(log) else env.score
                success = after - end_score
                d = abs(start["object"] - start["player"])
                by_distance[d][0] += success
                by_distance[d][1] += 1
                by_start_lane[(start["player"], start["object"])][1] += 1
                by_start_lane[(start["player"], start["object"])][0] += success
        out["success_by_initial_distance"] = {
            str(k): [v[0] / v[1], v[1]] for k, v in sorted(by_distance.items())
        }
        worst = sorted(((v[0] / v[1], k, v[1]) for k, v in by_start_lane.items() if v[1] >= 5))[:6]
        out["worst_player_object_starts"] = [
            {"player": k[0], "object": k[1], "success": s, "n": n} for s, k, n in worst
        ]
    elif task == "pong":
        misses, hits = [], []
        for env, log in episodes:
            for i in range(1, len(log)):
                if log[i]["attempts"] > log[i - 1]["attempts"] or (
                    i == len(log) - 1 and env.attempts > log[i]["attempts"]
                ):
                    pass
            serve = log[0]
            for i, row in enumerate(log):
                nxt = log[i + 1] if i + 1 < len(log) else None
                done_attempt = nxt is not None and nxt["attempts"] > row["attempts"]
                if done_attempt:
                    hit = nxt["hits"] > row["hits"]
                    record = {
                        "serve_vy": serve["vy"],
                        "serve_distance": abs(serve["intercept"] - serve["paddle"]),
                        "final_error": abs(row["intercept"] - row["paddle"]),
                    }
                    (hits if hit else misses).append(record)
                    serve = nxt
        out["attempts"] = len(hits) + len(misses)
        out["miss_rate"] = len(misses) / max(1, len(hits) + len(misses))
        for name, rows in (("misses", misses), ("hits", hits)):
            if rows:
                out[f"{name}_mean_serve_distance"] = float(
                    np.mean([r["serve_distance"] for r in rows])
                )
                out[f"{name}_mean_final_error"] = float(np.mean([r["final_error"] for r in rows]))
        buckets = collections.defaultdict(lambda: [0, 0])
        for r in hits:
            buckets[round(r["serve_distance"], 1)][0] += 1
            buckets[round(r["serve_distance"], 1)][1] += 1
        for r in misses:
            buckets[round(r["serve_distance"], 1)][1] += 1
        out["hit_rate_by_serve_distance"] = {
            str(k): [v[0] / v[1], v[1]] for k, v in sorted(buckets.items())
        }
        vy = collections.defaultdict(lambda: [0, 0])
        for r in hits:
            vy[r["serve_vy"]][0] += 1
            vy[r["serve_vy"]][1] += 1
        for r in misses:
            vy[r["serve_vy"]][1] += 1
        out["hit_rate_by_serve_vy"] = {str(k): [v[0] / v[1], v[1]] for k, v in sorted(vy.items())}
    elif task == "flappy":
        deaths = collections.Counter()
        where = collections.Counter()
        passed = []
        for env, log in episodes:
            passed.append(env.passed)
            deaths[env.death or "survived"] += 1
            if env.death == "boundary":
                where["top" if log[-1]["y"] + log[-1]["vy"] > 0.5 else "bottom"] += 1
            if env.death == "pipe":
                last = log[-1]
                where["pipe_above_gap" if last["y"] > last["gap"] else "pipe_below_gap"] += 1
        out["death_types"] = dict(deaths)
        out["death_locations"] = dict(where)
        out["pipes_passed_distribution"] = dict(collections.Counter(passed))
    else:
        deaths = collections.Counter(env.death or "none" for env, _ in episodes)
        out["death_types"] = dict(deaths)
        out["food_mean"] = float(np.mean([env.eaten for env, _ in episodes]))
        out["length_at_death_mean"] = float(np.mean([len(env.body) for env, _ in episodes]))
        out["steps_mean"] = float(np.mean([env.time for env, _ in episodes]))
        out["food_quantiles"] = np.quantile(
            [env.eaten for env, _ in episodes], [0.1, 0.5, 0.9]
        ).tolist()
    out["score"] = float(np.mean([env.metrics()["success"] for env, _ in episodes]))
    return out


def main():
    tasks = sys.argv[1:] or ["catch", "dodge", "snake", "pong", "flappy"]
    selected = json.loads(Path("artifacts/v14/selected_configs.json").read_text())["tasks"]
    OUT.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        seeds = [seed_for(task, "diagnosis", 0, INDEX_OFFSET + i) for i in range(EPISODES)]
        report = {"task": task, "seed_purpose": "v1.5 diagnosis", "controllers": {}}
        for kind in ("reference", "random"):
            report["controllers"][kind] = analyse(task, [run_episode(task, s, kind) for s in seeds])
        for role in ("final_selected", "primary_selected"):
            record = selected[task][role]
            name = record["name"]
            if name in report["controllers"]:
                continue
            per_seed = []
            for seed in (0, 1, 2):
                with open(f"runs/v14/{name}-s{seed}/checkpoint.pkl", "rb") as stream:
                    params = pickle.load(stream)["params"]
                config = json.loads(Path(f"runs/v14/{name}-s{seed}/result.json").read_text())[
                    "config"
                ]
                reset, act = policy_actor(config, params)
                per_seed.append(analyse(task, [run_episode(task, s, act, reset) for s in seeds]))
            report["controllers"][f"v14:{name}"] = per_seed
        path = OUT / f"{task}.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(
            task,
            {
                k: (v["score"] if isinstance(v, dict) else [round(x["score"], 3) for x in v])
                for k, v in report["controllers"].items()
            },
            flush=True,
        )


if __name__ == "__main__":
    main()
