"""Post-hoc representation diagnostics on matched exogenous states.

States come from a behaviour policy that depends only on the environment (reference
action on even steps, seeded uniform-random action on odd steps), so every topology
and tick setting sees byte-identical observation sequences. Probes are analysis
instruments trained on the first episodes and scored on held-out later episodes;
their labels never reach a controller.
"""

import hashlib

import numpy as np

from flyarcade_v13.controller import FlyFeatures
from flyarcade_v13.environments import TARGETS, make_env, reference_action
from flyarcade_v13.policy import Adam, orthogonal
from flyarcade_v13.study import BEHAVIOUR_OFFSET, seed_for


def collect_states(task, *, min_states=6000, max_episodes=2000, batch=16):
    """Episode-indexed matched states; returns observations, labels and episode ids."""
    observations, labels, episodes, seeds = [], [], [], []
    while len(observations) < min_states and len(seeds) < max_episodes:
        block = [seed_for(task, "representation", index=len(seeds) + i) for i in range(batch)]
        for k, s in enumerate(block):
            env = make_env(task, s)
            behaviour = np.random.default_rng(s + BEHAVIOUR_OFFSET)
            while not env.done:
                observations.append(env.observe())
                label = reference_action(env)
                labels.append(label)
                episodes.append(len(seeds) + k)
                action = int(behaviour.integers(env.actions)) if env.time % 2 else label
                env.step(action)
        seeds.extend(block)
    obs = np.asarray(observations)
    return {
        "observations": obs,
        "labels": np.asarray(labels, dtype=int),
        "episodes": np.asarray(episodes, dtype=int),
        "seeds": seeds,
        "state_sha256": hashlib.sha256(obs.tobytes()).hexdigest(),
    }


def features_for_states(graph, task, states, *, ticks, readout="descending", batch=16):
    """Replays each episode's observation sequence through its own fresh neural column."""
    source = FlyFeatures(graph, TARGETS[task].width, batch, ticks=ticks, readout=readout)
    obs, episodes = states["observations"], states["episodes"]
    raw = np.zeros((len(obs), source.width))
    ids = np.unique(episodes)
    for start in range(0, len(ids), batch):
        block = ids[start : start + batch]
        index = {e: np.flatnonzero(episodes == e) for e in block}
        for c, e in enumerate(block):
            source.reset(c, states["seeds"][e])
        longest = max(len(v) for v in index.values())
        for t in range(longest):
            live = [c for c, e in enumerate(block) if t < len(index[e])]
            rows = [index[block[c]][t] for c in live]
            raw[rows] = source.raw_rates(obs[rows], live)
    return raw


def participation_ratio(x):
    cov = np.cov(x, rowvar=False)
    eig = np.clip(np.linalg.eigvalsh(cov), 0, None)
    return float(eig.sum() ** 2 / max((eig**2).sum(), 1e-12))


def mean_pairwise_correlation(x, limit=128):
    variance = x.var(axis=0)
    active = np.argsort(variance)[::-1][: min(limit, int((variance > 0).sum()))]
    if len(active) < 2:
        return 0.0
    c = np.corrcoef(x[:, active], rowvar=False)
    return float(np.nanmean(c[np.triu_indices(len(active), 1)]))


def split(episodes, fraction=0.6):
    ids = np.unique(episodes)
    cut = ids[int(len(ids) * fraction)]
    return episodes < cut, episodes >= cut


def zscore(train, test):
    mean, sd = train.mean(0), train.std(0)
    sd[sd < 1e-8] = 1.0
    return (train - mean) / sd, (test - mean) / sd


def balanced_accuracy(truth, predicted, classes):
    recalls = [np.mean(predicted[truth == k] == k) for k in range(classes) if (truth == k).any()]
    return float(np.mean(recalls))


def ridge_classifier(xtr, ytr, xte, classes, penalty=1.0):
    weights = _balanced_weights(ytr, classes)
    x = np.hstack([xtr, np.ones((len(xtr), 1))])
    targets = np.eye(classes)[ytr] * 2 - 1
    xw = x * weights[:, None]
    gram = x.T @ xw + penalty * np.eye(x.shape[1])
    beta = np.linalg.solve(gram, xw.T @ targets)
    return (np.hstack([xte, np.ones((len(xte), 1))]) @ beta).argmax(1)


def ridge_regression_r2(xtr, ytr, xte, yte, penalty=10.0):
    x = np.hstack([xtr, np.ones((len(xtr), 1))])
    beta = np.linalg.solve(x.T @ x + penalty * np.eye(x.shape[1]), x.T @ ytr)
    pred = np.hstack([xte, np.ones((len(xte), 1))]) @ beta
    return _r2(yte, pred)


def _r2(truth, pred):
    residual = ((truth - pred) ** 2).sum(0)
    total = ((truth - truth.mean(0)) ** 2).sum(0)
    valid = total > 1e-10
    return np.where(valid, 1 - residual / np.where(valid, total, 1), np.nan)


def _balanced_weights(y, classes):
    counts = np.bincount(y, minlength=classes).astype(float)
    per_class = np.where(counts > 0, len(y) / (classes * np.maximum(counts, 1)), 0)
    return per_class[y]


def mlp_probe(xtr, ytr, xte, *, kind, classes=None, hidden=64, epochs=60, seed=0, lr=1e-3):
    """One-hidden-layer tanh probe; balanced cross-entropy or MSE. Deterministic."""
    rng = np.random.default_rng(seed)
    outputs = classes if kind == "classify" else ytr.shape[1]
    p = {
        "W1": orthogonal(rng, hidden, xtr.shape[1], 1.0),
        "b1": np.zeros(hidden),
        "W2": orthogonal(rng, outputs, hidden, 0.1),
        "b2": np.zeros(outputs),
    }
    opt = Adam(p, lr=lr)
    weights = _balanced_weights(ytr, classes) if kind == "classify" else np.ones(len(xtr))
    for _ in range(epochs):
        order = rng.permutation(len(xtr))
        for start in range(0, len(xtr), 256):
            idx = order[start : start + 256]
            x, w = xtr[idx], weights[idx]
            h = np.tanh(x @ p["W1"].T + p["b1"])
            out = h @ p["W2"].T + p["b2"]
            if kind == "classify":
                z = out - out.max(1, keepdims=True)
                prob = np.exp(z) / np.exp(z).sum(1, keepdims=True)
                dout = (prob - np.eye(classes)[ytr[idx]]) * w[:, None] / w.sum()
            else:
                dout = 2 * (out - ytr[idx]) / len(idx) / outputs
            dh = (dout @ p["W2"]) * (1 - h**2)
            grads = {
                "W2": dout.T @ h + 1e-4 * p["W2"],
                "b2": dout.sum(0),
                "W1": dh.T @ x + 1e-4 * p["W1"],
                "b1": dh.sum(0),
            }
            opt.step(p, grads)
    out = np.tanh(xte @ p["W1"].T + p["b1"]) @ p["W2"].T + p["b2"]
    return out.argmax(1) if kind == "classify" else out


def analyse(features, states, task, *, include_mlp=True):
    """All representation metrics for one feature matrix on the matched states."""
    labels, obs, episodes = states["labels"], states["observations"], states["episodes"]
    classes = TARGETS[task].actions
    train, test = split(episodes)
    xtr, xte = zscore(features[train], features[test])
    ytr, yte = labels[train], labels[test]
    counts_train = np.bincount(ytr, minlength=classes)
    counts_test = np.bincount(yte, minlength=classes)
    majority = int(counts_train.argmax())
    otr, ote = obs[train], obs[test]
    varying = obs.std(0) > 1e-8
    report = {
        "states": int(len(features)),
        "train_states": int(train.sum()),
        "test_states": int(test.sum()),
        "class_counts_train": counts_train.tolist(),
        "class_counts_test": counts_test.tolist(),
        "majority_accuracy_test": float(np.mean(yte == majority)),
        "majority_balanced_accuracy": balanced_accuracy(yte, np.full(len(yte), majority), classes),
        "linear_action_balanced_accuracy": balanced_accuracy(
            yte, ridge_classifier(xtr, ytr, xte, classes), classes
        ),
        "linear_state_r2": np.nan_to_num(
            ridge_regression_r2(xtr, otr[:, varying], xte, ote[:, varying]), nan=0.0
        ).tolist(),
    }
    report["linear_state_r2_mean"] = float(np.mean(report["linear_state_r2"]))
    if include_mlp:
        report["mlp_action_balanced_accuracy"] = balanced_accuracy(
            yte, mlp_probe(xtr, ytr, xte, kind="classify", classes=classes), classes
        )
        omean, osd = otr[:, varying].mean(0), otr[:, varying].std(0)
        pred = mlp_probe(xtr, (otr[:, varying] - omean) / osd, xte, kind="regress")
        r2 = np.nan_to_num(_r2((ote[:, varying] - omean) / osd, pred), nan=0.0)
        report["mlp_state_r2"] = r2.tolist()
        report["mlp_state_r2_mean"] = float(np.mean(r2))
    return report


def activity_statistics(raw):
    return {
        "participation_ratio": participation_ratio(raw),
        "mean_pairwise_correlation": mean_pairwise_correlation(raw),
        "mean_activity_variance": float(raw.var(axis=0).mean()),
        "mean_rate": float((raw + 0.25).mean()),
        "active_fraction": float((raw.var(axis=0) > 0).mean()),
    }
