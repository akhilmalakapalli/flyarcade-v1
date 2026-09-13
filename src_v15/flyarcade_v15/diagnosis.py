"""Phase-1 representation diagnostics on matched exogenous states (v1.5 diagnosis seeds).

States come from a behaviour policy that depends only on the environment (reference
action on even steps, seeded random action on odd steps), so every tick count and
readout sees byte-identical observation sequences. Decoders are analysis instruments
fit on early episodes and scored on held-out later episodes; nothing reaches a policy.
"""

import numpy as np

from flyarcade_v13.representations import (
    balanced_accuracy,
    mlp_probe,
    participation_ratio,
    ridge_classifier,
    split,
    zscore,
)
from flyarcade_v13_flappy.derived import control_quantities
from flyarcade_v14.environments import TARGETS, make_env, reference_action
from flyarcade_v15.features import Fly15, Sensory15
from flyarcade_v15.seeds import BEHAVIOUR_OFFSET, NEURAL_OFFSET, seed_for


def state_variables(task, env, obs):
    if task in ("catch", "dodge"):
        return {
            "player": env.player / 4,
            "object": env.object / 4,
            "phase": env.phase / 3,
            "object_minus_player": (env.object - env.player) / 4,
            "aligned": float(env.object == env.player),
        }
    if task == "snake":
        names = ["food_up", "food_down", "food_left", "food_right", "food_distance"]
        names += [f"heading_{i}" for i in range(4)] + [f"danger_{i}" for i in range(4)]
        names += [f"local_{i}" for i in range(8)]
        return dict(zip(names, obs.tolist(), strict=True))
    if task == "pong":
        return {
            "paddle": env.paddle,
            "ball_x": env.x,
            "ball_y": env.y,
            "ball_vy": env.vy,
            "intercept_error": env.intercept() - env.paddle,
        }
    q = control_quantities(obs)
    return {
        k: float(q[k])
        for k in (
            "bird_y",
            "vertical_velocity",
            "time_to_pipe",
            "gap_center",
            "signed_gap_error",
            "projected_no_flap_crossing_error",
            "velocity_error",
        )
    }


def collect_states(task, *, min_states=4000, batch=16, max_episodes=3000):
    obs, labels, episodes, seeds, variables = [], [], [], [], []
    while len(obs) < min_states and len(seeds) < max_episodes:
        for k in range(batch):
            s = seed_for(task, "diagnosis", index=len(seeds))
            seeds.append(s)
            env = make_env(task, s)
            behaviour = np.random.default_rng(s + BEHAVIOUR_OFFSET)
            while not env.done:
                o = env.observe()
                label = int(reference_action(env))
                obs.append(o)
                labels.append(label)
                episodes.append(len(seeds) - 1)
                variables.append(state_variables(task, env, o))
                action = int(behaviour.integers(env.actions)) if env.time % 2 else label
                env.step(action)
    names = list(variables[0])
    return {
        "observations": np.asarray(obs),
        "labels": np.asarray(labels, dtype=int),
        "episodes": np.asarray(episodes, dtype=int),
        "seeds": seeds,
        "variable_names": names,
        "variables": np.asarray([[v[n] for n in names] for v in variables]),
    }


def replay_all_rates(graph, task, states, ticks, *, repeat=0, batch=16):
    """All-2040 raw rates; each episode replays through its own fresh neural column."""
    source = Fly15(graph, task, batch, ticks=ticks, readout="all")
    obs, episodes = states["observations"], states["episodes"]
    raw = np.zeros((len(obs), source.n), dtype=np.float32)
    ids = np.unique(episodes)
    for start in range(0, len(ids), batch):
        block = ids[start : start + batch]
        index = {e: np.flatnonzero(episodes == e) for e in block}
        for c, e in enumerate(block):
            rng = np.random.default_rng(states["seeds"][e] + NEURAL_OFFSET + 7919 * repeat)
            source.reset_with_rng(c, rng)
        longest = max(len(v) for v in index.values())
        for t in range(longest):
            live = [c for c, e in enumerate(block) if t < len(index[e])]
            rows = [index[block[c]][t] for c in live]
            raw[rows] = source.all_rates(obs[rows], live)
    return raw, source


def sensory_features(task, states):
    return Sensory15(task, 1)(states["observations"], np.zeros(len(states["observations"]), int))


def _ridge_r2(xtr, ytr, xte, yte, penalty):
    x = np.hstack([xtr, np.ones((len(xtr), 1))])
    gram = x.T @ x
    gram[np.diag_indices_from(gram)] += penalty
    beta = np.linalg.solve(gram, x.T @ ytr)
    pred = np.hstack([xte, np.ones((len(xte), 1))]) @ beta
    res = ((yte - pred) ** 2).sum(0)
    tot = ((yte - yte.mean(0)) ** 2).sum(0)
    return np.where(tot > 1e-10, 1 - res / np.where(tot > 1e-10, tot, 1), np.nan)


PENALTIES = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)


def _inner(episodes, train):
    ids = np.unique(episodes[train])
    cut = ids[int(len(ids) * 0.75)]
    return train & (episodes < cut), train & (episodes >= cut)


def decode(features, states, task, *, mlp=False):
    """Ridge decoders with the penalty chosen on an inner episode split of the training set."""
    labels, episodes, y = states["labels"], states["episodes"], states["variables"]
    classes = TARGETS[task].actions
    train, test = split(episodes)
    fit, tune = _inner(episodes, train)
    f = features.astype(float)
    xf, xt = zscore(f[fit], f[tune])
    best_r2, best_acc = None, None
    for penalty in PENALTIES:
        r2 = np.nanmean(np.nan_to_num(_ridge_r2(xf, y[fit], xt, y[tune], penalty), nan=0.0))
        acc = balanced_accuracy(
            labels[tune], ridge_classifier(xf, labels[fit], xt, classes, penalty), classes
        )
        if best_r2 is None or r2 > best_r2[0]:
            best_r2 = (r2, penalty)
        if best_acc is None or acc > best_acc[0]:
            best_acc = (acc, penalty)
    xtr, xte = zscore(f[train], f[test])
    r2 = _ridge_r2(xtr, y[train], xte, y[test], best_r2[1])
    report = {
        "width": int(xtr.shape[1]),
        "ridge_penalty_state": best_r2[1],
        "ridge_penalty_action": best_acc[1],
        "linear_action_balanced_accuracy": balanced_accuracy(
            labels[test], ridge_classifier(xtr, labels[train], xte, classes, best_acc[1]), classes
        ),
        "linear_state_r2": dict(
            zip(states["variable_names"], np.nan_to_num(r2, nan=0.0).round(4).tolist(), strict=True)
        ),
    }
    report["linear_state_r2_mean"] = float(np.mean(np.nan_to_num(r2, nan=0.0)))
    if mlp:
        report["mlp_action_balanced_accuracy"] = balanced_accuracy(
            labels[test],
            mlp_probe(xtr, labels[train], xte, kind="classify", classes=classes),
            classes,
        )
    return report


def activity(raw_a, raw_b, episodes, *, floor=0.02, clip=4.0):
    """Rate distribution, dimensionality, trial-to-trial SNR, clipping and stability."""
    a = raw_a.astype(float) + 0.25
    b = raw_b.astype(float) + 0.25
    var = a.var(0)
    active = var > 1e-12
    signal = ((a + b) / 2).var(0)
    noise = ((a - b) ** 2).mean(0) / 2
    snr = np.where(noise > 1e-12, np.maximum(signal - noise / 2, 0) / np.maximum(noise, 1e-12), 0)
    mean, sd = a.mean(0), a.std(0)
    z = (a - mean) / np.maximum(sd, floor)
    clipped = float((np.abs(z) >= clip).mean())
    amplified = float(((sd > 0) & (sd < floor)).mean())
    # lag-1 autocorrelation of the first principal component within episodes
    centred = a - mean
    if active.sum() >= 2:
        u, s, vt = np.linalg.svd(centred[:: max(1, len(a) // 3000)], full_matrices=False)
        pc = centred @ vt[0]
        same = episodes[1:] == episodes[:-1]
        lag = float(np.corrcoef(pc[1:][same], pc[:-1][same])[0, 1])
    else:
        lag = 0.0
    return {
        "neurons": int(a.shape[1]),
        "active_fraction": float(active.mean()),
        "mean_rate": float(a.mean()),
        "rate_quantiles": np.quantile(a.mean(0), [0.1, 0.5, 0.9, 0.99]).round(4).tolist(),
        "participation_ratio": participation_ratio(a[:, active]) if active.sum() > 1 else 0.0,
        "median_snr_active": float(np.median(snr[active])) if active.any() else 0.0,
        "fraction_snr_gt_1": float((snr[active] > 1).mean()) if active.any() else 0.0,
        "standardized_clip_fraction": clipped,
        "floor_amplified_fraction": amplified,
        "pc1_lag1_autocorrelation": lag,
    }
