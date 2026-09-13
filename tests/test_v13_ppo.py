import numpy as np
import pytest

from flyarcade_v13.policy import ActorCritic, Adam, log_softmax
from flyarcade_v13.ppo import gae, ppo_loss, sample_actions, update


def test_gae_terminal_masking_and_bootstrap():
    rewards = np.array([[1.0], [2.0], [3.0]])
    values = np.array([[0.5], [0.5], [0.5]])
    dones = np.array([[0.0], [1.0], [0.0]])
    adv, ret = gae(rewards, values, dones, np.array([10.0]), gamma=0.9, lam=0.8)
    # step 2 bootstraps from last value; step 1 is terminal; step 0 sees step 1 only
    d2 = 3 + 0.9 * 10 - 0.5
    d1 = 2 - 0.5
    d0 = 1 + 0.9 * 0.5 - 0.5
    np.testing.assert_allclose(adv[:, 0], [d0 + 0.9 * 0.8 * d1, d1, d2])
    np.testing.assert_allclose(ret, adv + values)


def test_gae_lambda_one_gamma_one_is_monte_carlo():
    rewards = np.array([[1.0], [1.0], [1.0]])
    adv, ret = gae(rewards, np.zeros((3, 1)), np.array([[0], [0], [1.0]]), np.zeros(1), 1, 1)
    np.testing.assert_allclose(ret[:, 0], [3, 2, 1])


def _numeric(fn, array, eps=1e-6):
    grad = np.zeros_like(array)
    for idx in np.ndindex(array.shape):
        old = array[idx]
        array[idx] = old + eps
        up = fn()
        array[idx] = old - eps
        down = fn()
        array[idx] = old
        grad[idx] = (up - down) / (2 * eps)
    return grad


def test_ppo_loss_gradients_and_clipping():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(6, 3))
    values = rng.normal(size=6)
    actions = rng.integers(3, size=6)
    old = log_softmax(logits)[np.arange(6), actions] + rng.normal(0, 0.4, 6)
    adv, ret = rng.normal(size=6), rng.normal(size=6)
    kw = dict(clip=0.2, vf=0.5, ent=0.01)

    def f():
        return ppo_loss(logits, values, actions, old, adv, ret, **kw)[0]

    _, stats, dl, dv = ppo_loss(logits, values, actions, old, adv, ret, **kw)
    np.testing.assert_allclose(dl, _numeric(f, logits), atol=1e-6)
    np.testing.assert_allclose(dv, _numeric(f, values), atol=1e-6)
    assert 0 < stats["clip_fraction"] <= 1


def test_clipped_objective_gives_no_gradient_outside_trust_region():
    logits = np.array([[2.0, 0.0]])
    logp = log_softmax(logits)[0, 0]
    _, _, dl, _ = ppo_loss(
        logits, np.zeros(1), np.array([0]), np.array([logp - 1.0]), np.array([1.0]),
        np.zeros(1), clip=0.2, vf=0.0, ent=0.0,
    )  # fmt: skip
    np.testing.assert_array_equal(dl, 0)


def test_entropy_of_uniform_policy():
    _, stats, _, _ = ppo_loss(
        np.zeros((4, 3)), np.zeros(4), np.zeros(4, dtype=int), np.full(4, -np.log(3)),
        np.zeros(4), np.zeros(4), clip=0.2, vf=0.5, ent=0.01,
    )  # fmt: skip
    assert stats["entropy"] == pytest.approx(np.log(3))


def test_sample_actions_follow_distribution():
    rng = np.random.default_rng(0)
    logits = np.tile(np.log([0.2, 0.5, 0.3]), (20000, 1))
    actions, logp = sample_actions(logits, rng)
    np.testing.assert_allclose(np.bincount(actions) / 20000, [0.2, 0.5, 0.3], atol=0.02)
    np.testing.assert_allclose(logp, np.log([0.2, 0.5, 0.3])[actions])


@pytest.mark.parametrize("arch", ["mlp", "gru"])
def test_update_improves_advantaged_action(arch):
    model = ActorCritic(4, 2, arch=arch, hidden=8, core=4, seed=0)
    steps, envs = 8, 4
    feats = np.random.default_rng(1).normal(size=(steps, envs, 4))
    actions = np.zeros((steps, envs), dtype=int)
    batch = {
        "features": feats,
        "actions": actions,
        "log_probs": np.full((steps, envs), np.log(0.5)),
        "advantages": np.ones((steps, envs)) + 0.1 * np.arange(steps)[:, None],
        "returns": np.ones((steps, envs)),
        "resets": np.zeros((steps, envs)),
        "states": np.zeros((steps, envs, 4)),
    }
    config = dict(epochs=3, minibatches=2, clip=0.2, vf=0.5, ent=0.0, max_grad_norm=0.5, chunk=4)
    config["normalize_advantages"] = False
    before = model.forward(feats.reshape(-1, 4))[0] if arch == "mlp" else None
    stats = update(model, Adam(model.params, lr=1e-2), batch, config, np.random.default_rng(0))
    assert stats["minibatch_updates"] == 6 and np.isfinite(stats["loss"])
    if arch == "mlp":
        after = model.forward(feats.reshape(-1, 4))[0]
        assert (np.exp(log_softmax(after))[:, 0] > np.exp(log_softmax(before))[:, 0]).mean() > 0.9
