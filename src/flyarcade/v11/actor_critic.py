"""Linear actor-critic readout with TD errors, eligibility traces and entropy control.

This replaces v1's `Motor`, which applied one global scalar advantage to a REINFORCE
eligibility with no value baseline, no discounting, no entropy term and no
exploration floor. Each of those omissions is repaired here, and each is separately
switchable so development runs can attribute any improvement.
"""

import numpy as np


class ActorCritic:
    def __init__(
        self,
        width,
        rng,
        *,
        actions=3,
        actor_lr=0.002,
        critic_lr=0.01,
        discount=0.9,
        trace_decay=0.7,
        entropy_coefficient=0.01,
        exploration_floor=0.05,
        weight_bound=4.0,
        update_clip=0.05,
        init_scale=0.0,
    ):
        self.actions = int(actions)
        self.width = int(width) + 1  # trailing bias feature
        self.actor = (
            rng.normal(0, init_scale, (self.actions, self.width))
            if init_scale
            else (np.zeros((self.actions, self.width)))
        )
        self.critic = np.zeros(self.width)
        self.actor_lr = float(actor_lr)
        self.critic_lr = float(critic_lr)
        self.discount = float(discount)
        self.trace_decay = float(trace_decay)
        self.entropy_coefficient = float(entropy_coefficient)
        self.exploration_floor = float(exploration_floor)
        self.weight_bound = float(weight_bound)
        self.update_clip = float(update_clip)
        if not 0 <= self.exploration_floor < 1:
            raise ValueError("exploration floor must lie in [0, 1)")
        self.reset()
        self.reset_episode_value()

    def reset(self):
        self.actor_trace = np.zeros_like(self.actor)
        self.critic_trace = np.zeros_like(self.critic)

    def reset_episode_value(self):
        self.previous_value = None
        self.previous_features = None

    def policy(self, features):
        """Return the sampling distribution and the underlying softmax."""
        x = np.append(np.asarray(features, dtype=float), 1.0)
        logits = self.actor @ x
        softmax = np.exp(logits - logits.max())
        softmax /= softmax.sum()
        mixed = (1 - self.exploration_floor) * softmax + self.exploration_floor / self.actions
        return x, softmax, mixed

    def choose(self, features, rng, training=True):
        x, softmax, mixed = self.policy(features)
        action = int(rng.choice(self.actions, p=mixed))
        value = float(self.critic @ x)
        if training:
            one_hot = np.zeros(self.actions)
            one_hot[action] = 1.0
            # Exact score function of the floor-mixed sampling distribution.
            score = (1 - self.exploration_floor) * softmax[action] * (one_hot - softmax)
            score /= max(mixed[action], 1e-12)
            self.actor_trace = self.discount * self.trace_decay * self.actor_trace + np.outer(
                score, x
            )
            self.critic_trace = self.discount * self.trace_decay * self.critic_trace + x
        self.pending = {
            "x": x,
            "softmax": softmax,
            "mixed": mixed,
            "value": value,
            "score": score if training else np.zeros(self.actions),
        }
        return action, softmax, mixed, value

    def value_of(self, features):
        return float(self.critic @ np.append(np.asarray(features, dtype=float), 1.0))

    def observe(self, reward, next_value, done):
        """One temporal-difference step; returns diagnostics for this transition.

        `next_value` is supplied by the caller so the circuit is advanced once per
        action: the value of s_{t+1} is read when its features are computed for the
        next action, never by re-simulating the network.
        """
        x = self.pending["x"]
        value = self.pending["value"]
        next_value = 0.0 if done else float(next_value)
        delta = float(reward + self.discount * next_value - value)
        delta = float(np.clip(delta, -10, 10))
        softmax = self.pending["softmax"]
        entropy = float(-(softmax * np.log(softmax + 1e-12)).sum())
        # d/dlogit of the entropy of the softmax, applied without a trace so the
        # exploration pressure is immediate rather than credited to past actions.
        entropy_gradient = -softmax * (np.log(softmax + 1e-12) + entropy)
        actor_update = self.actor_lr * (
            delta * self.actor_trace + self.entropy_coefficient * np.outer(entropy_gradient, x)
        )
        actor_update = np.clip(actor_update, -self.update_clip, self.update_clip)
        critic_update = np.clip(
            self.critic_lr * delta * self.critic_trace, -self.update_clip, self.update_clip
        )
        self.actor = np.clip(self.actor + actor_update, -self.weight_bound, self.weight_bound)
        self.critic = np.clip(self.critic + critic_update, -self.weight_bound, self.weight_bound)
        return {
            "delta": delta,
            "value": value,
            "entropy": entropy,
            "max_probability": float(softmax.max()),
            "actor_update_abs": float(np.abs(actor_update).mean()),
            "critic_update_abs": float(np.abs(critic_update).mean()),
        }

    def readout_projection(self, delta, scale, ticks=4, gain=1.0):
        """Per-descending-neuron learning signal for the recurrent e-prop stages.

        Returns one value per readout feature: the temporal-difference error times
        that neuron's own contribution to the chosen action's score, divided through
        the standardisation scale and the number of ticks the rate averages over.
        Every synapse onto a descending neuron is therefore driven by that neuron's
        signal, which is exactly what v1's single global scalar failed to provide.
        """
        scale = np.asarray(scale, dtype=float)
        if scale.shape != (self.width - 1,) or not (scale > 0).all():
            raise ValueError("one positive scale per readout feature required")
        contribution = self.pending["score"] @ self.actor[:, :-1]
        return delta * contribution * gain / (ticks * scale)
