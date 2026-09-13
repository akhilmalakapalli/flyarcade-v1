"""One predeclared rescue: heuristic warm-start, then ordinary target-task PPO."""

import numpy as np

from flyarcade_v13.policy import Adam, clip_gradients, log_softmax
from flyarcade_v14.environments import make_env, reference_action
from flyarcade_v14.study import make_source, seed_for


def warm_start(trainer, guard, *, count=8192, passes=4):
    c = trainer.config
    batch = c["envs"]
    steps = count // batch
    source = make_source(c, batch)
    counter = 0
    envs = []
    for col in range(batch):
        seed = seed_for(trainer.task, "imitation", c["seed"], counter)
        counter += 1
        envs.append(make_env(trainer.task, seed))
        source.reset(col, seed)
    features = np.zeros((steps, batch, source.width))
    resets = np.zeros((steps, batch))
    actions = np.zeros((steps, batch), dtype=int)
    reset = np.ones(batch)
    for t in range(steps):
        guard.check()
        features[t] = source(np.stack([e.observe() for e in envs]))
        resets[t] = reset
        reset = np.zeros(batch)
        for col, env in enumerate(envs):
            action = int(reference_action(env))
            actions[t, col] = action
            env.step(action)
            if env.done:
                seed = seed_for(trainer.task, "imitation", c["seed"], counter)
                counter += 1
                envs[col] = make_env(trainer.task, seed)
                source.reset(col, seed)
                reset[col] = 1
    optimizer = Adam(trainer.model.params, lr=3e-4)
    rng = np.random.default_rng(seed_for(trainer.task, "imitation", c["seed"], 999999))
    losses = []
    for _ in range(passes):
        if trainer.model.recurrent:
            chunks = [(t, col) for t in range(0, steps, 16) for col in range(batch)]
            order = rng.permutation(len(chunks))
            for selected in np.array_split(order, 8):
                starts, cols = np.asarray([chunks[i] for i in selected]).T
                window = starts[None, :] + np.arange(16)[:, None]
                cc = np.broadcast_to(cols, window.shape)
                logits, values, cache = trainer.model.forward(
                    features[window, cc],
                    trainer.model.initial_state(len(selected)),
                    resets[window, cc],
                )
                labels = actions[window, cc].reshape(-1)
                _update(trainer.model, optimizer, cache, logits, values, labels, losses)
        else:
            for selected in np.array_split(rng.permutation(count), 8):
                logits, values, cache = trainer.model.forward(features.reshape(count, -1)[selected])
                _update(
                    trainer.model,
                    optimizer,
                    cache,
                    logits,
                    values,
                    actions.reshape(-1)[selected],
                    losses,
                )
        guard.check()
    return {
        "phase": "SECONDARY RESCUE PHASE",
        "transitions": count,
        "passes": passes,
        "episodes_started": counter,
        "mean_cross_entropy": float(np.mean(losses)),
        "optimizer": "separate Adam lr=0.0003; PPO optimizer starts fresh",
    }


def _update(model, optimizer, cache, logits, values, labels, losses):
    lp = log_softmax(logits)
    gradient = np.exp(lp)
    gradient[np.arange(len(labels)), labels] -= 1
    gradient /= len(labels)
    grads = model.backward(cache, gradient, np.zeros_like(values))
    clip_gradients(grads, 0.5)
    optimizer.step(model.params, grads)
    losses.append(float(-lp[np.arange(len(labels)), labels].mean()))
