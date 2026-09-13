"""Phase-1 diagnostic training matrix (diagnosis_training seeds; not development)."""

import json
from pathlib import Path

import v15_path  # noqa: F401
from flyarcade_v15.provenance import code_hash

OUT = Path("experiments/v15/specs/diagnosis")
TASKS = ("catch", "dodge", "snake", "pong", "flappy")


def budget(task):
    return 150000 if task == "flappy" else 131072


def spec(name, group, config):
    return {
        "name": f"diag-{name}",
        "phase": "diagnosis",
        "group": group,
        "code_sha256": code_hash(),
        "config": {"train_purpose": "diagnosis_training", **config},
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    specs = []
    for seed in (0, 1):
        for task in TASKS:
            for arch in ("linear", "gru"):
                specs.append(
                    spec(
                        f"sensory-{task}-{arch}-s{seed}",
                        "D1_sensory_ceiling",
                        {
                            "task": task,
                            "source": "sensory",
                            "arch": arch,
                            "transitions": budget(task),
                            "seed": seed,
                        },
                    )
                )
            for lr in (3e-4, 1e-3):
                tag = "lr3e-4" if lr == 3e-4 else "lr1e-3"
                specs.append(
                    spec(
                        f"{task}-gru-t16-desc-{tag}-s{seed}",
                        "D3_step_size_t16" if task != "flappy" or lr != 3e-4 else "D2_flappy_ticks",
                        {
                            "task": task,
                            "arch": "gru",
                            "ticks": 16,
                            "lr": lr,
                            "transitions": budget(task),
                            "seed": seed,
                        },
                    )
                )
        specs.append(
            spec(
                f"flappy-gru-t4-desc-lr3e-4-s{seed}",
                "D2_flappy_ticks",
                {"task": "flappy", "arch": "gru", "ticks": 4, "transitions": 150000, "seed": seed},
            )
        )
        for task in ("snake", "pong"):
            for readout in ("all", "visual", "nonvisual"):
                specs.append(
                    spec(
                        f"{task}-linear-t8-{readout}-s{seed}",
                        "D4_readout_leakage",
                        {
                            "task": task,
                            "arch": "linear",
                            "ticks": 8,
                            "readout": readout,
                            "transitions": 65536,
                            "seed": seed,
                        },
                    )
                )
    for s in specs:
        (OUT / f"{s['name']}.json").write_text(json.dumps(s, indent=2) + "\n")
    print(len(specs), "diagnostic specs")


if __name__ == "__main__":
    main()
