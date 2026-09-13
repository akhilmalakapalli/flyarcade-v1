"""Proximal policy optimisation on stored rollouts.

Rollouts store fly features, actions, rewards, episode ends, log-probabilities and
values, so every optimisation epoch reuses the already simulated spiking activity
instead of re-running the circuit. Nothing here touches the neural core.
"""

import numpy as np

from flyarcade_v13.policy import clip_gradients, log_softmax


def gae(rewards, values, dones, last_values, gamma, lam):
    """Generalized advantage estimation over (T, B) arrays.

    ``dones[t]`` marks that the episode ended after step t; no value or advantage
    crosses that boundary. Time-limit truncations carry their bootstrap inside the
    reward (r + gamma * V(s_T)) and are therefore also marked done here.
    """
    rewards, values, dones = (np.asarray(a, dtype=float) for a in (rewards, values, dones))
    steps = len(rewards)
    advantages = np.zeros_like(rewards)
    running = np.zeros(rewards.shape[1])
    for t in reversed(range(steps)):
        following = last_values if t == steps - 1 else values[t + 1]
        live = 1.0 - dones[t]
        delta = rewards[t] + gamma * following * live - values[t]
        running = delta + gamma * lam * live * running
        advantages[t] = running
    return advantages, advantages + values


def ppo_loss(logits, values, actions, old_log_probs, advantages, returns, *, clip, vf, ent):
    """Loss value, diagnostics and exact gradients with respect to logits and values."""
    n = len(actions)
    logp_all = log_softmax(logits)
    probs = np.exp(logp_all)
    rows = np.arange(n)
    logp = logp_all[rows, actions]
    ratio = np.exp(logp - old_log_probs)
    surr1 = ratio * advantages
    surr2 = np.clip(ratio, 1 - clip, 1 + clip) * advantages
    policy_loss = -np.minimum(surr1, surr2).mean()
    value_error = values - returns
    value_loss = 0.5 * (value_error**2).mean()
    entropy_each = -(probs * logp_all).sum(axis=1)
    entropy = entropy_each.mean()
    loss = policy_loss + vf * value_loss - ent * entropy
    # d(-min(surr1, surr2))/d logp: nonzero only where the unclipped branch is active.
    active = surr1 <= surr2
    dlogp = -(active * advantages * ratio) / n
    one_hot = np.zeros_like(probs)
    one_hot[rows, actions] = 1.0
    dlogits = dlogp[:, None] * (one_hot - probs)
    dlogits += (ent / n) * probs * (logp_all + entropy_each[:, None])
    dvalues = vf * value_error / n
    stats = {
        "loss": float(loss),
        "policy_loss": float(policy_loss),
        "value_loss": float(value_loss),
        "entropy": float(entropy),
        "approx_kl": float(((ratio - 1) - np.log(ratio)).mean()),
        "clip_fraction": float((np.abs(ratio - 1) > clip).mean()),
    }
    return loss, stats, dlogits, dvalues


def sample_actions(logits, rng):
    logp = log_softmax(logits)
    probs = np.exp(logp)
    cumulative = probs.cumsum(axis=1)
    draws = rng.random((len(logits), 1))
    actions = np.minimum((draws > cumulative).sum(axis=1), logits.shape[1] - 1)
    return actions.astype(int), logp[np.arange(len(logits)), actions]


def update(model, optimizer, batch, config, rng, lr=None):
    """Several epochs of clipped PPO minibatch updates on one stored rollout.

    ``batch`` arrays are (T, B, ...). Feed-forward models shuffle transitions;
    recurrent models shuffle fixed-length chunks and replay each chunk from the
    hidden state recorded when the rollout was collected, honouring episode resets.
    """
    features = batch["features"]
    steps, envs = batch["actions"].shape
    adv = batch["advantages"]
    if config.get("normalize_advantages", True):
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    minibatches = int(config["minibatches"])
    stats = []
    norms = []
    if not model.recurrent:
        x = features.reshape(steps * envs, -1)
        flat = {
            "actions": batch["actions"].reshape(-1),
            "log_probs": batch["log_probs"].reshape(-1),
            "advantages": adv.reshape(-1),
            "returns": batch["returns"].reshape(-1),
        }
        total = steps * envs
        size = total // minibatches
        for _ in range(int(config["epochs"])):
            order = rng.permutation(total)
            for m in range(minibatches):
                idx = order[m * size : (m + 1) * size]
                logits, values, cache = model.forward(x[idx])
                _, s, dlogits, dvalues = ppo_loss(
                    logits,
                    values,
                    flat["actions"][idx],
                    flat["log_probs"][idx],
                    flat["advantages"][idx],
                    flat["returns"][idx],
                    clip=config["clip"],
                    vf=config["vf"],
                    ent=config["ent"],
                )
                grads = model.backward(cache, dlogits, dvalues)
                norms.append(clip_gradients(grads, config["max_grad_norm"]))
                optimizer.step(model.params, grads, lr)
                stats.append(s)
    else:
        chunk = int(config["chunk"])
        if steps % chunk:
            raise ValueError("rollout length must be a multiple of the GRU chunk length")
        chunks = [(start, env) for start in range(0, steps, chunk) for env in range(envs)]
        size = len(chunks) // minibatches
        for _ in range(int(config["epochs"])):
            order = rng.permutation(len(chunks))
            for m in range(minibatches):
                selected = [chunks[i] for i in order[m * size : (m + 1) * size]]
                starts = np.array([s for s, _ in selected])
                columns = np.array([e for _, e in selected])
                window = starts[None, :] + np.arange(chunk)[:, None]
                cols = np.broadcast_to(columns, window.shape)
                logits, values, cache = model.forward(
                    features[window, cols],
                    batch["states"][starts, columns],
                    batch["resets"][window, cols],
                )
                _, s, dlogits, dvalues = ppo_loss(
                    logits,
                    values,
                    batch["actions"][window, cols].reshape(-1),
                    batch["log_probs"][window, cols].reshape(-1),
                    adv[window, cols].reshape(-1),
                    batch["returns"][window, cols].reshape(-1),
                    clip=config["clip"],
                    vf=config["vf"],
                    ent=config["ent"],
                )
                grads = model.backward(cache, dlogits, dvalues)
                norms.append(clip_gradients(grads, config["max_grad_norm"]))
                optimizer.step(model.params, grads, lr)
                stats.append(s)
    summary = {k: float(np.mean([s[k] for s in stats])) for k in stats[0]}
    summary["grad_norm"] = float(np.mean(norms))
    summary["minibatch_updates"] = len(stats)
    return summary
