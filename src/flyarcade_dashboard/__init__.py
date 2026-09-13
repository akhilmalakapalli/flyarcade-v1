"""FlyArcade localhost visualization dashboard (evaluation-only).

Lives outside ``src/flyarcade`` and ``src/flyarcade_v13`` on purpose: the frozen
v1, v1.2 and v1.3 studies hash those trees into their run plans, and a
visualization tool must never change a frozen code hash.

Nothing in this package trains, updates parameters, runs PPO, touches
confirmatory seeds, or writes outside ``cache/dashboard``.
"""

DEMO_LABEL = "Demo playback — not a new experiment"
NETWORK_LABEL = "2,040-neuron MaleCNS-derived network"
ACTIVITY_NOTE = (
    "Neural activity shown here is simulated activity from the connectome-derived "
    "spiking model, not experimentally recorded neural activity."
)
