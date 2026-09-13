"""Print a compact per-configuration summary of completed v1.5 runs matching a prefix."""

import collections
import json
import sys
from pathlib import Path

import numpy as np


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else ""
    groups = collections.defaultdict(list)
    for path in sorted(Path("runs/v15").glob(f"{prefix}*/result.json")):
        r = json.loads(path.read_text())
        groups[r["name"].rsplit("-s", 1)[0]].append(r)
    for name, rs in groups.items():
        h = [r["training_history"][-8:] for r in rs]
        mean = lambda key: np.mean([np.mean([x[key] for x in hh]) for hh in h])  # noqa: E731
        print(
            f"{name:44s} n={len(rs)} score={np.mean([r['score'] for r in rs]):7.3f} "
            f"seeds={[round(r['score'], 3) for r in rs]} last={[round(r['last_checkpoint_score'], 3) for r in rs]} "
            f"probe={[round(r['state_probe']['score'], 2) for r in rs]} "
            f"dom={np.mean([r['dominant_action_fraction'] for r in rs]):.2f} "
            f"kl={mean('approx_kl'):.4f} clip={mean('clip_fraction'):.3f} ev={mean('explained_variance'):.2f} "
            f"H={mean('entropy'):.2f} tps={np.mean([r['training_transitions'] / r['wall_clock_training_seconds'] for r in rs]):.0f}"
        )


if __name__ == "__main__":
    main()
