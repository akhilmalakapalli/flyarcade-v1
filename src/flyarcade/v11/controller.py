"""v1.1 controller: unchanged MaleCNS circuit and encoding, new credit assignment.

Three staged plasticity scopes, tested in order of increasing flexibility:

A `readout`     - recurrent core entirely frozen; only the actor-critic is plastic.
B `descending`  - additionally, synapses terminating on descending neurons learn.
C `recurrent`   - additionally, synapses onto central-complex intrinsic neurons.

Stage C is reached only if A and B fail, per the v1.1 protocol. Anatomy justifies
the ordering: stage B touches the last relay onto the motor-command population,
stage C opens the central-complex intrinsic population that sits between the visual
input and that relay.
"""

import numpy as np

from flyarcade.sensory import features
from flyarcade.v11.actor_critic import ActorCritic
from flyarcade.v11.core import EpropCore

STAGES = ("readout", "descending", "recurrent")
TICKS = 4


class V11Controller:
    def __init__(
        self,
        graph,
        seed=0,
        *,
        stage="readout",
        standardizer=None,
        core_lr=0.0,
        core_update_clip=0.02,
        eligibility_decay=0.9,
        pseudo_derivative_scale=0.3,
        core_weight_bound=2.0,
        background=0.18,
        stimulus=1.5,
        **actor_kwargs,
    ):
        if stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}")
        self.graph = graph
        self.stage = stage
        self.rng = np.random.default_rng(seed)
        self.background = float(background)
        self.stimulus = float(stimulus)
        neurons = {n["bodyId"]: n for n in graph.provenance.get("neurons", [])}
        if not neurons:
            neurons = {n["bodyId"]: n for n in graph.provenance["parent"]["neurons"]}
        ordered = [neurons[int(i)] for i in graph.neuron_ids]
        self.inputs = np.array(
            [i for i, n in enumerate(ordered) if n.get("superclass") == "visual_projection"],
            dtype=int,
        )
        self.outputs = np.array(
            [i for i, n in enumerate(ordered) if n.get("superclass") == "descending_neuron"],
            dtype=int,
        )
        self.intrinsic = np.array(
            [i for i, n in enumerate(ordered) if n.get("superclass") == "cb_intrinsic"],
            dtype=int,
        )
        if not len(self.inputs) or not len(self.outputs):
            raise ValueError("visual inputs and descending outputs required")
        signs = [
            1
            if n.get("consensusNt") == "acetylcholine"
            else -1
            if n.get("consensusNt") == "gaba"
            else 0
            for n in ordered
        ]
        plastic_targets = np.zeros(graph.n, dtype=bool)
        if stage in ("descending", "recurrent"):
            plastic_targets[self.outputs] = True
        if stage == "recurrent":
            plastic_targets[self.intrinsic] = True
        self.core = EpropCore(
            graph,
            signs,
            plastic_edges=plastic_targets[graph.post],
            eligibility_decay=eligibility_decay,
            pseudo_derivative_scale=pseudo_derivative_scale,
            weight_bound=core_weight_bound,
        )
        self.core_lr = float(core_lr)
        self.core_update_clip = float(core_update_clip)
        # Same fixed electrode assignment as v1; not a biological mapping.
        self.input_channels = np.arange(len(self.inputs)) % 8
        self.standardizer = standardizer
        width = len(self.outputs)
        self.motor = ActorCritic(width, self.rng, **actor_kwargs)
        if stage == "recurrent":
            self.feedback = np.random.default_rng(900 + seed).normal(0, 1, len(self.intrinsic))
        self.reset()

    @property
    def plastic_synapses(self):
        return self.core.plastic_count

    def reset(self):
        self.core.reset()
        self.motor.reset()
        self.motor.reset_episode_value()
        self.spike_count = 0
        self.tick_count = 0
        self.core_update_abs = 0.0
        self.pending_reward = None
        self.last_diagnostics = None

    def rates(self, observation, *, noise=0, edge_mask=None, neuron_mask=None, plastic=False):
        obs = np.asarray(observation, dtype=float)
        if noise:
            obs = np.clip(obs + self.rng.normal(0, noise, 3), 0, 1)
        f = features(obs)
        counts = np.zeros(len(self.outputs))
        for _ in range(TICKS):
            drive = np.full(self.graph.n, self.background)
            drive[self.inputs] += self.stimulus * (
                self.rng.random(len(self.inputs)) < f[self.input_channels]
            )
            spikes = self.core.tick(
                drive, plastic=plastic, edge_mask=edge_mask, neuron_mask=neuron_mask
            )
            counts += spikes[self.outputs]
            self.spike_count += int(spikes.sum())
            self.tick_count += 1
        raw = counts / TICKS - 0.25
        return raw if self.standardizer is None else self.standardizer(raw)

    def act(self, observation, *, training=False, noise=0, edge_mask=None, neuron_mask=None):
        plastic = training and self.core_lr > 0 and self.core.plastic_count > 0
        phi = self.rates(
            observation,
            noise=noise,
            edge_mask=edge_mask,
            neuron_mask=neuron_mask,
            plastic=plastic,
        )
        # Complete the previous transition now that s_{t+1} has been observed. The
        # circuit is advanced exactly once per action; V(s_{t+1}) is read here
        # rather than by re-simulating the network.
        if self.pending_reward is not None:
            self._apply(self.pending_reward, self.motor.value_of(phi), done=False)
            self.pending_reward = None
        action, _, _, _ = self.motor.choose(phi, self.rng, training)
        return int(action)

    def reward(self, value):
        """Record the reward for the transition the next act() call completes."""
        self.pending_reward = float(value)

    def finish_episode(self):
        """Bootstrap the terminal transition with a zero continuation value."""
        if self.pending_reward is not None:
            self._apply(self.pending_reward, 0.0, done=True)
            self.pending_reward = None

    def _apply(self, reward, next_value, *, done):
        diagnostics = self.motor.observe(reward, next_value, done)
        if self.core_lr > 0 and self.core.plastic_count:
            signal = np.zeros(self.graph.n)
            signal[self.outputs] = self.motor.readout_projection(
                diagnostics["delta"],
                self.standardizer.scale,
                ticks=TICKS,
                gain=self.standardizer.gain,
            )
            if self.stage == "recurrent":
                # Hidden-neuron learning signals use fixed random feedback, the
                # standard e-prop treatment; still per neuron, never one global scalar.
                signal[self.intrinsic] = diagnostics["delta"] * self.feedback
            self.core_update_abs = self.core.learn(
                signal, learning_rate=self.core_lr, update_clip=self.core_update_clip
            )
        diagnostics["core_update_abs"] = self.core_update_abs
        self.last_diagnostics = diagnostics
        return diagnostics
