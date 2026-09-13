"""SECONDARY RESCUE PHASE only: reference-controller warm-start, then ordinary PPO.

Collects fresh target-task transitions on v1.5 ``imitation`` seeds under the reference
controller, fits the actor by cross-entropy with a separate Adam optimiser, and hands
the parameters to PPO with a fresh PPO optimiser. The critic is not trained here. The
evaluation metric and rewards are unchanged.
"""

import numpy as np

from flyarcade_v13.policy import Adam, clip_gradients, log_softmax
from flyarcade_v14.environments import make_env, reference_action
from flyarcade_v15.seeds import seed_for


def warm_start(trainer, make_source, guard, *, count, passes=4, minibatches=8):
    c = trainer.config
    batch = c["envs"]
    steps = count // batch
    chunk = c["chunk"]
    steps -= steps % chunk
    source = make_source(c, batch)
    counter, envs = 0, []
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
    model = trainer.model
    optimizer = Adam(model.params, lr=3e-4)
    rng = np.random.default_rng(seed_for(trainer.task, "imitation", c["seed"], 9_999_999))
    losses = []
    for _ in range(passes):
        if model.recurrent:
            chunks = [(t, col) for t in range(0, steps, chunk) for col in range(batch)]
            for selected in np.array_split(rng.permutation(len(chunks)), minibatches):
                starts, cols = np.asarray([chunks[i] for i in selected]).T
                window = starts[None, :] + np.arange(chunk)[:, None]
                cc = np.broadcast_to(cols, window.shape)
                # every chunk starts from a zero state, flagged as a reset
                chunk_resets = resets[window, cc].copy()
                chunk_resets[0] = 1
                logits, values, cache = model.forward(
                    features[window, cc], model.initial_state(len(selected)), chunk_resets
                )
                _step(model, optimizer, cache, logits, values, actions[window, cc].ravel(), losses)
        else:
            flat = features.reshape(steps * batch, -1)
            for selected in np.array_split(rng.permutation(steps * batch), minibatches):
                logits, values, cache = model.forward(flat[selected])
                _step(model, optimizer, cache, logits, values, actions.ravel()[selected], losses)
        guard.check()
    return {
        "phase": "SECONDARY RESCUE PHASE",
        "transitions": int(steps * batch),
        "passes": passes,
        "episodes_started": counter,
        "mean_cross_entropy_last_pass": float(np.mean(losses[-minibatches:])),
        "optimizer": "separate Adam lr=3e-4; PPO optimizer starts fresh",
        "teacher": "reference controller (Flappy MPC oracle / task heuristic)",
    }


def _step(model, optimizer, cache, logits, values, labels, losses):
    lp = log_softmax(logits)
    gradient = np.exp(lp)
    gradient[np.arange(len(labels)), labels] -= 1
    gradient /= len(labels)
    grads = model.backward(cache, gradient, np.zeros_like(values))
    clip_gradients(grads, 0.5)
    optimizer.step(model.params, grads)
    losses.append(float(-lp[np.arange(len(labels)), labels].mean()))
