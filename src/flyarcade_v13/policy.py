"""NumPy actor-critic networks with exact hand-derived gradients.

A2 (``mlp``):  Linear(D,H) -> LayerNorm -> tanh -> Linear(H,C) -> tanh -> heads
A3 (``gru``):  Linear(D,H) -> LayerNorm -> tanh -> GRU(C)      -> heads

The shared trunk feeds a categorical actor head and a scalar critic head. NumPy was
chosen over PyTorch so no protected v1.2 dependency file changes and single-thread
BLAS execution stays bit-reproducible; finite-difference tests verify every gradient.
"""

import numpy as np

ARCHITECTURES = ("mlp", "gru")
LN_EPS = 1e-5


def orthogonal(rng, rows, cols, gain):
    a = rng.normal(size=(max(rows, cols), min(rows, cols)))
    q, r = np.linalg.qr(a)
    q = q * np.sign(np.diag(r))
    q = q if rows >= cols else q.T
    return np.ascontiguousarray(gain * q[:rows, :cols])


def sigmoid(x):
    return 0.5 * (1 + np.tanh(0.5 * x))


def log_softmax(logits):
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


class ActorCritic:
    def __init__(self, input_width, actions, *, arch="mlp", hidden=128, core=64, seed=0):
        if arch not in ARCHITECTURES:
            raise ValueError(f"arch must be one of {ARCHITECTURES}")
        self.arch, self.input_width, self.actions = arch, int(input_width), int(actions)
        self.hidden, self.core = int(hidden), int(core)
        rng = np.random.default_rng(seed)
        d, h, c, a = self.input_width, self.hidden, self.core, self.actions
        p = {
            "W1": orthogonal(rng, h, d, np.sqrt(2)),
            "b1": np.zeros(h),
            "g1": np.ones(h),
            "beta1": np.zeros(h),
        }
        if arch == "mlp":
            p["W2"] = orthogonal(rng, c, h, np.sqrt(2))
            p["b2"] = np.zeros(c)
        else:
            p["Wi"] = np.concatenate([orthogonal(rng, c, h, 1.0) for _ in range(3)])
            p["bi"] = np.zeros(3 * c)
            p["Wh"] = np.concatenate([orthogonal(rng, c, c, 1.0) for _ in range(3)])
            p["bh"] = np.zeros(3 * c)
        p["Wa"] = orthogonal(rng, a, c, 0.01)
        p["ba"] = np.zeros(a)
        p["Wv"] = orthogonal(rng, 1, c, 1.0)
        p["bv"] = np.zeros(1)
        self.params = p

    @property
    def recurrent(self):
        return self.arch == "gru"

    def initial_state(self, batch):
        return np.zeros((batch, self.core)) if self.recurrent else None

    def copy_params(self):
        return {k: v.copy() for k, v in self.params.items()}

    def finite(self):
        return all(np.isfinite(v).all() for v in self.params.values())

    # ------------------------------------------------------------- blocks
    def _encode(self, x):
        p = self.params
        z = x @ p["W1"].T + p["b1"]
        scale = np.sqrt(z.var(axis=1, keepdims=True) + LN_EPS)
        zn = (z - z.mean(axis=1, keepdims=True)) / scale
        out = np.tanh(zn * p["g1"] + p["beta1"])
        return out, (x, zn, scale, out)

    def _encode_backward(self, dout, cache, grads):
        x, zn, scale, out = cache
        p = self.params
        da = dout * (1 - out**2)
        grads["g1"] += (da * zn).sum(0)
        grads["beta1"] += da.sum(0)
        dzn = da * p["g1"]
        dz = (
            dzn - dzn.mean(axis=1, keepdims=True) - zn * (dzn * zn).mean(axis=1, keepdims=True)
        ) / scale
        grads["W1"] += dz.T @ x
        grads["b1"] += dz.sum(0)

    def _heads(self, h):
        p = self.params
        return h @ p["Wa"].T + p["ba"], (h @ p["Wv"].T + p["bv"])[:, 0]

    def _heads_backward(self, h, dlogits, dvalues, grads):
        p = self.params
        grads["Wa"] += dlogits.T @ h
        grads["ba"] += dlogits.sum(0)
        dv = dvalues[:, None]
        grads["Wv"] += dv.T @ h
        grads["bv"] += dv.sum(0)
        return dlogits @ p["Wa"] + dv @ p["Wv"]

    def _gru_cell(self, e, hp):
        p, c = self.params, self.core
        gi = e @ p["Wi"].T + p["bi"]
        gh = hp @ p["Wh"].T + p["bh"]
        r = sigmoid(gi[:, :c] + gh[:, :c])
        z = sigmoid(gi[:, c : 2 * c] + gh[:, c : 2 * c])
        n = np.tanh(gi[:, 2 * c :] + r * gh[:, 2 * c :])
        return (1 - z) * n + z * hp, (e, hp, r, z, n, gh[:, 2 * c :])

    # ------------------------------------------------------------- inference
    def step(self, x, state=None, reset=None):
        """One environment step for a batch. ``reset`` zeroes hidden rows first."""
        e, _ = self._encode(np.asarray(x, dtype=float))
        if not self.recurrent:
            logits, values = self._heads(self._trunk_mlp(e)[0])
            return logits, values, None
        if reset is not None:
            state = state * (1.0 - np.asarray(reset, dtype=float))[:, None]
        h, _ = self._gru_cell(e, state)
        logits, values = self._heads(h)
        return logits, values, h

    def _trunk_mlp(self, e):
        p = self.params
        h = np.tanh(e @ p["W2"].T + p["b2"])
        return h, (e, h)

    # ------------------------------------------------------------- training
    def forward(self, x, state0=None, resets=None):
        """Batch forward with cache. MLP: x (N, D). GRU: x (T, B, D), state0 (B, C),
        resets (T, B) with 1 where the hidden state is zeroed before that step."""
        x = np.asarray(x, dtype=float)
        if not self.recurrent:
            e, enc = self._encode(x)
            h, trunk = self._trunk_mlp(e)
            logits, values = self._heads(h)
            return logits, values, (enc, trunk, h)
        steps, batch, width = x.shape
        e, enc = self._encode(x.reshape(steps * batch, width))
        e = e.reshape(steps, batch, -1)
        h = state0
        outputs, caches, masks = [], [], []
        for t in range(steps):
            mask = 1.0 - resets[t][:, None]
            h, cache = self._gru_cell(e[t], h * mask)
            outputs.append(h)
            caches.append(cache)
            masks.append(mask)
        hs = np.stack(outputs).reshape(steps * batch, -1)
        logits, values = self._heads(hs)
        return logits, values, (enc, caches, masks, hs, (steps, batch))

    def backward(self, cache, dlogits, dvalues):
        grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        p = self.params
        if not self.recurrent:
            enc, (e, h), hout = cache
            dh = self._heads_backward(hout, dlogits, dvalues, grads)
            dz2 = dh * (1 - h**2)
            grads["W2"] += dz2.T @ e
            grads["b2"] += dz2.sum(0)
            self._encode_backward(dz2 @ p["W2"], enc, grads)
            return grads
        enc, caches, masks, hs, (steps, batch) = cache
        c = self.core
        dhs = self._heads_backward(hs, dlogits, dvalues, grads).reshape(steps, batch, c)
        de = np.zeros((steps, batch, self.hidden))
        carry = np.zeros((batch, c))
        for t in reversed(range(steps)):
            e, hp, r, z, n, ghn = caches[t]
            dh = dhs[t] + carry
            dn = dh * (1 - z)
            dz = dh * (hp - n)
            dhp = dh * z
            dan = dn * (1 - n**2)
            dar = dan * ghn * r * (1 - r)
            daz = dz * z * (1 - z)
            dgi = np.concatenate([dar, daz, dan], axis=1)
            dgh = np.concatenate([dar, daz, dan * r], axis=1)
            grads["Wi"] += dgi.T @ e
            grads["bi"] += dgi.sum(0)
            grads["Wh"] += dgh.T @ hp
            grads["bh"] += dgh.sum(0)
            de[t] = dgi @ p["Wi"]
            carry = (dhp + dgh @ p["Wh"]) * masks[t]
        self._encode_backward(de.reshape(steps * batch, -1), enc, grads)
        return grads


class Adam:
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.999), eps=1e-5):
        self.lr, self.betas, self.eps = float(lr), betas, float(eps)
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads, lr=None):
        lr = self.lr if lr is None else float(lr)
        self.t += 1
        b1, b2 = self.betas
        c1, c2 = 1 - b1**self.t, 1 - b2**self.t
        for k in params:
            self.m[k] = b1 * self.m[k] + (1 - b1) * grads[k]
            self.v[k] = b2 * self.v[k] + (1 - b2) * grads[k] ** 2
            params[k] -= lr * (self.m[k] / c1) / (np.sqrt(self.v[k] / c2) + self.eps)

    def state(self):
        return {"m": self.m, "v": self.v, "t": self.t, "lr": self.lr}

    def load(self, state):
        self.m = {k: v.copy() for k, v in state["m"].items()}
        self.v = {k: v.copy() for k, v in state["v"].items()}
        self.t, self.lr = int(state["t"]), float(state["lr"])


def clip_gradients(grads, max_norm):
    norm = float(np.sqrt(sum((g**2).sum() for g in grads.values())))
    if max_norm and norm > max_norm:
        factor = max_norm / (norm + 1e-12)
        for k in grads:
            grads[k] *= factor
    return norm
