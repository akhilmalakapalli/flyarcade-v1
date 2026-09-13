"""Save and load demo episodes: compressed NPZ arrays plus JSON metadata.

Replays are written only under ``cache/dashboard/replays`` (Git-ignored), never under
``artifacts/`` or ``runs/``. Loading reconstructs the exact frames that were shown
live, so playback needs no simulation at all.
"""

import base64
import json
import re
import time
from pathlib import Path

import numpy as np

from flyarcade_dashboard import DEMO_LABEL
from flyarcade_dashboard.model_loader import PACKAGE_ROOT

REPLAY_DIR = PACKAGE_ROOT / "cache" / "dashboard" / "replays"
ARRAY_FIELDS = ("counts", "observations", "probs", "actions", "rewards")
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")  # names written by save(); no dots or slashes


def save(dashboard, directory=REPLAY_DIR):
    with dashboard.lock:
        frames = list(dashboard.frames)
        state = dashboard.state()
    if not frames:
        raise ValueError("nothing to save: no steps taken in this episode")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = f"{state['task']}-model{state['model_seed']}-demo{state['demo_index']}-{stamp}"
    counts = np.stack(
        [np.frombuffer(base64.b64decode(f["counts_b64"]), dtype=np.uint8) for f in frames]
    )
    np.savez_compressed(
        directory / f"{name}.npz",
        counts=counts,
        observations=np.array([f["observation"] for f in frames], dtype=float),
        probs=np.array([f["probs"] for f in frames], dtype=float),
        actions=np.array([f["action"] for f in frames], dtype=np.int16),
        rewards=np.array([f["reward"] for f in frames], dtype=float),
    )
    per_step = [
        {k: f[k] for k in f if k not in ("counts_b64", "observation", "probs", "action", "reward")}
        for f in frames
    ]
    meta = {
        "label": DEMO_LABEL,
        "task": state["task"],
        "model": state["model"],
        "model_seed": state["model_seed"],
        "demo_index": state["demo_index"],
        "demo_seed": state["demo_seed"],
        "calibration": state["calibration"],
        "steps": len(frames),
        "complete_episode": bool(frames[-1]["done"]),
        "frames": per_step,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (directory / f"{name}.json").write_text(json.dumps(meta) + "\n")
    return name


def list_replays(directory=REPLAY_DIR):
    directory = Path(directory)
    if not directory.exists():
        return []
    out = []
    for path in sorted(directory.glob("*.json"), reverse=True):
        if (path.with_suffix(".npz")).exists():
            meta = json.loads(path.read_text())
            out.append(
                {
                    "name": path.stem,
                    "task": meta["task"],
                    "steps": meta["steps"],
                    "demo_seed": meta["demo_seed"],
                    "saved_at": meta["saved_at"],
                }
            )
    return out


def load(name, directory=REPLAY_DIR):
    if not NAME.match(name):
        raise ValueError("invalid replay name")
    directory = Path(directory)
    meta = json.loads((directory / f"{name}.json").read_text())
    with np.load(directory / f"{name}.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in ARRAY_FIELDS}
    frames = []
    for t, extra in enumerate(meta["frames"]):
        frames.append(
            {
                **extra,
                "observation": arrays["observations"][t].tolist(),
                "counts_b64": base64.b64encode(arrays["counts"][t].tobytes()).decode(),
                "probs": arrays["probs"][t].tolist(),
                "action": int(arrays["actions"][t]),
                "reward": float(arrays["rewards"][t]),
            }
        )
    return {k: v for k, v in meta.items() if k != "frames"} | {"frames": frames}
