import numpy as np
import pytest

from flyarcade_v13.policy import ActorCritic, Adam, clip_gradients, log_softmax


def _loss(model, x, state0=None, resets=None, seed=3):
    rng = np.random.default_rng(seed)
    logits, values, cache = model.forward(x, state0, resets)
    a = rng.normal(size=logits.shape)
    b = rng.normal(size=values.shape)
    return float((a * logits).sum() + (b * values).sum()), cache, a, b


def _check_gradients(model, x, state0=None, resets=None, keys=None):
    loss, cache, a, b = _loss(model, x, state0, resets)
    grads = model.backward(cache, a, b)
    rng = np.random.default_rng(11)
    for key in keys or model.params:
        param = model.params[key]
        for _ in range(4):
            idx = tuple(rng.integers(s) for s in param.shape)
            old = param[idx]
            param[idx] = old + 1e-6
            up = _loss(model, x, state0, resets)[0]
            param[idx] = old - 1e-6
            down = _loss(model, x, state0, resets)[0]
            param[idx] = old
            numeric = (up - down) / 2e-6
            assert numeric == pytest.approx(grads[key][idx], rel=1e-4, abs=1e-6), key


def test_mlp_gradients_match_finite_differences():
    model = ActorCritic(9, 3, arch="mlp", hidden=7, core=5, seed=1)
    for key in model.params:  # nonzero heads so every path carries gradient
        model.params[key] = model.params[key] + np.random.default_rng(2).normal(
            0, 0.3, model.params[key].shape
        )
    _check_gradients(model, np.random.default_rng(0).normal(size=(6, 9)))


def test_gru_gradients_match_finite_differences_with_resets():
    model = ActorCritic(5, 2, arch="gru", hidden=6, core=4, seed=1)
    for key in model.params:
        model.params[key] = model.params[key] + np.random.default_rng(5).normal(
            0, 0.3, model.params[key].shape
        )
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 3, 5))
    resets = np.zeros((5, 3))
    resets[2, 1] = 1
    _check_gradients(model, x, rng.normal(size=(3, 4)), resets)


def test_probabilities_normalised_and_logits_finite():
    model = ActorCritic(20, 3, seed=0)
    logits, values, _ = model.step(np.random.default_rng(0).normal(size=(8, 20)))
    probs = np.exp(log_softmax(logits))
    np.testing.assert_allclose(probs.sum(1), 1)
    assert np.isfinite(logits).all() and np.isfinite(values).all()
    extreme = log_softmax(np.array([[1e4, -1e4, 0.0]]))
    assert np.isfinite(extreme).all()


def test_forward_deterministic_for_seed():
    a, b = ActorCritic(20, 3, seed=7), ActorCritic(20, 3, seed=7)
    x = np.ones((2, 20))
    np.testing.assert_array_equal(a.step(x)[0], b.step(x)[0])
    assert not np.array_equal(ActorCritic(20, 3, seed=8).step(x)[0], a.step(x)[0])


def test_step_matches_batch_forward_for_both_architectures():
    rng = np.random.default_rng(1)
    mlp = ActorCritic(6, 3, arch="mlp", hidden=8, core=4)
    x = rng.normal(size=(5, 6))
    np.testing.assert_allclose(mlp.step(x)[0], mlp.forward(x)[0])
    gru = ActorCritic(6, 3, arch="gru", hidden=8, core=4)
    seq = rng.normal(size=(4, 2, 6))
    resets = np.array([[1, 1], [0, 0], [0, 1], [0, 0]], dtype=float)
    h = gru.initial_state(2)
    logits = []
    for t in range(4):
        out, _, h = gru.step(seq[t], h, resets[t])
        logits.append(out)
    batch_logits = gru.forward(seq, np.zeros((2, 4)), resets)[0]
    np.testing.assert_allclose(np.concatenate(logits), batch_logits, atol=1e-12)


def test_gru_hidden_reset_and_persistence():
    gru = ActorCritic(4, 2, arch="gru", hidden=5, core=3, seed=2)
    x = np.ones((1, 4))
    _, _, h1 = gru.step(x, gru.initial_state(1))
    _, _, h2 = gru.step(x, h1)
    assert not np.allclose(h1, h2)  # state persists within an episode
    _, _, fresh = gru.step(x, h2, reset=np.ones(1))
    np.testing.assert_array_equal(fresh, h1)  # boundary reset equals a new episode


def test_gradient_clipping_and_adam_restore():
    grads = {"a": np.full(4, 3.0)}
    norm = clip_gradients(grads, 0.5)
    assert norm == pytest.approx(6.0)
    assert np.sqrt((grads["a"] ** 2).sum()) == pytest.approx(0.5)
    params = {"a": np.zeros(4)}
    adam = Adam(params, lr=0.1)
    adam.step(params, {"a": np.ones(4)})
    saved, snapshot = adam.state(), {"a": params["a"].copy()}
    saved = {"m": dict(saved["m"]), "v": dict(saved["v"]), "t": saved["t"], "lr": saved["lr"]}
    adam.step(params, {"a": np.ones(4)})
    expected = params["a"].copy()
    other = Adam({"a": np.zeros(4)})
    other.load(saved)
    restored = {"a": snapshot["a"].copy()}
    other.step(restored, {"a": np.ones(4)})
    np.testing.assert_array_equal(restored["a"], expected)
