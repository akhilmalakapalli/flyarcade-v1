"""Flappy rescue extensions: augmented sensory encoding, arrival potential, probe targets.

Every quantity here is a deterministic transform of the *current* engineered
observation (or equivalently the current environment state). Nothing reads the
environment RNG, the next gap, the oracle's chosen action or any other future or
privileged information. All of it is opt-in by configuration key; the default
"v12" encoding and the default potential reproduce the pre-rescue pipeline exactly.
"""

import numpy as np

FLAP_VY = 0.055
GRAVITY = 0.009
MIN_VY = -0.08
SPEED = 0.04
PERIOD = 23  # steps from x = 0.9 to the crossing at speed 0.04
LOW, HIGH = 0.025, 0.975


def state_from_observation(observation):
    """Invert the v1.2 Flappy observation (no clipping occurs for reachable states)."""
    o = np.asarray(observation, dtype=float)
    return {
        "y": o[0],
        "vy": o[1] * 0.14 - 0.08,
        "x": o[2] * 0.9,
        "gap": o[3],
    }


def steps_to_crossing(x, speed=SPEED):
    """Same count as the environment's floating-point obstacle loop, vectorised."""
    x = np.asarray(x, dtype=float)
    steps = np.ceil(x / speed - 1e-9)
    return np.maximum(steps, 1)


def projected_no_flap_height(y, vy, steps):
    """Closed form of repeated no-flap steps with the velocity floor, clipped to [0, 1]."""
    y, vy, steps = (np.asarray(v, dtype=float) for v in (y, vy, steps))
    # steps i with vy - i*g still above the floor: i <= (vy - MIN_VY) / g
    k_floor = np.clip(np.floor((vy - MIN_VY) / GRAVITY + 1e-9), 0, None)
    k = np.minimum(steps, k_floor)
    # sum_{i=1..k} (vy - i*g) then floor velocity for the remaining steps
    falling = k * vy - GRAVITY * k * (k + 1) / 2
    floored = (steps - k) * MIN_VY
    return np.clip(y + falling + floored, 0, 1)


def control_quantities(observation):
    s = state_from_observation(observation)
    y, vy, gap = s["y"], s["vy"], s["gap"]
    ttp = steps_to_crossing(s["x"])
    desired = np.clip((gap - y) / np.maximum(ttp, 1.0), MIN_VY, FLAP_VY)
    return {
        "bird_y": y,
        "vertical_velocity": vy,
        "time_to_pipe": ttp,
        "gap_center": gap,
        "signed_gap_error": gap - y,
        "abs_gap_error": abs(gap - y),
        "projected_no_flap_crossing_error": projected_no_flap_height(y, vy, ttp) - gap,
        "desired_vy": desired,
        "velocity_error": vy - desired,
        "boundary_margin": min(y - LOW, HIGH - y),
    }


def flappy_augmented(observation):
    """v1.3 Flappy augmented sensory encoding: the six v1.2 channels plus seven
    normalised derived channels, all in [0, 1]."""
    o = np.clip(np.asarray(observation, dtype=float), 0, 1)
    q = control_quantities(o)
    derived = [
        q["time_to_pipe"] / PERIOD,
        0.5 + q["signed_gap_error"] / 1.0,
        q["abs_gap_error"] / 0.5,
        (q["desired_vy"] - MIN_VY) / (FLAP_VY - MIN_VY),
        0.5 + q["velocity_error"] / (2 * (FLAP_VY - MIN_VY)),
        0.5 + q["projected_no_flap_crossing_error"] / 2.0,
        q["boundary_margin"] / 0.475,
    ]
    return np.concatenate([o, np.clip(derived, 0, 1)])


ENCODINGS = {"v12": (None, 6), "flappy_augmented": (flappy_augmented, 13)}


def encoding(name, base_width):
    if name == "v12":
        return None, base_width
    transform, width = ENCODINGS[name]
    return transform, width


def arrival_potential(env, position_weight=0.5, velocity_weight=0.5):
    """Phi(s) = -a*|gap - y|/0.5 - b*|vy - desired_vy|/0.135, from current state only."""
    ttp = float(steps_to_crossing(env.x, getattr(env, "speed", SPEED)))
    desired = float(np.clip((env.gap - env.y) / max(ttp, 1.0), MIN_VY, FLAP_VY))
    return -(
        position_weight * abs(env.gap - env.y) / 0.5
        + velocity_weight * abs(env.vy - desired) / (FLAP_VY - MIN_VY)
    )
