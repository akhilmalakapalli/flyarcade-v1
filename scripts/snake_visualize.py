"""Render a Snake evaluation demonstration: gameplay, sensory input, neural activity
and performance metrics in one figure, plus an animated GIF of a full episode.

Uses a trained checkpoint with learning disabled. Nothing here trains or tunes.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from flyarcade.connectome import load_graph
from flyarcade.resources import ResourceGuard
from flyarcade.v11.features import Standardizer
from flyarcade.v11.snake import GRID, SnakeGame
from flyarcade.v11.snake_experiment import confirm_evaluation_seeds

CHANNEL_LABELS = [
    "food up",
    "food down",
    "food left",
    "food right",
    "food dist",
    "head up",
    "head down",
    "head left",
    "head right",
    "danger up",
    "danger down",
    "danger left",
    "danger right",
    *[f"local {i}" for i in range(8)],
]
ACTIONS = ("up", "down", "left", "right")


def record(controller, seed, guard, raster_neurons=120):
    """Replay one greedy episode, capturing everything the figure needs."""
    game = SnakeGame(seed)
    controller.reset()
    controller.rng = np.random.default_rng(seed + 500_000)
    rng_index = np.random.default_rng(0).choice(
        len(controller.outputs), size=raster_neurons, replace=False
    )
    frames = []
    while not game.done:
        guard.check()
        observation = game.observe()
        before = controller.spike_count
        rates = controller.rates(observation)
        _, softmax, _ = controller.motor.policy(rates)
        action = int(softmax.argmax())
        frames.append(
            {
                "body": list(game.body),
                "food": game.food,
                "observation": observation.copy(),
                "rates": rates[rng_index].copy(),
                "probabilities": softmax.copy(),
                "value": controller.motor.value_of(rates),
                "food_eaten": game.eaten,
                "step": game.time,
                "spikes": controller.spike_count - before,
            }
        )
        game.step(action)
    return frames, game


def draw(axes, frame, curve, total_steps):
    board, sensory, neural, policy = axes
    for ax in axes:
        ax.clear()
    grid = np.zeros((GRID, GRID))
    for i, (row, column) in enumerate(frame["body"]):
        grid[row, column] = 2.0 if i == 0 else 1.0
    if frame["food"]:
        grid[frame["food"]] = 3.0
    board.imshow(grid, cmap="Blues", vmin=0, vmax=3)
    board.set(
        title=f"step {frame['step']}  food {frame['food_eaten']}",
        xticks=[],
        yticks=[],
    )
    board.set_xticks(np.arange(-0.5, GRID, 1), minor=True)
    board.set_yticks(np.arange(-0.5, GRID, 1), minor=True)
    board.grid(which="minor", color="#ffffff", lw=1)

    sensory.barh(range(len(CHANNEL_LABELS)), frame["observation"], color="#2a78d6", height=0.7)
    sensory.set(
        title="sensory channels into visual-projection neurons",
        xlim=(0, 1.05),
        yticks=range(len(CHANNEL_LABELS)),
    )
    sensory.set_yticklabels(CHANNEL_LABELS, fontsize=6)
    sensory.invert_yaxis()
    sensory.tick_params(axis="x", labelsize=7)

    values = frame["rates"]
    neural.bar(range(len(values)), values, color="#1baf7a", width=1.0)
    neural.set(
        title=f"descending activity (120 of 1,293)   {frame['spikes']} spikes this step",
        xticks=[],
        ylim=(-0.25, 0.25),
    )
    neural.tick_params(axis="y", labelsize=7)

    policy.bar(range(4), frame["probabilities"], color="#eb6834", width=0.6)
    policy.set(
        title=f"policy   V(s) = {frame['value']:+.2f}",
        ylim=(0, 1.05),
        xticks=range(4),
        xticklabels=ACTIONS,
    )
    policy.tick_params(labelsize=7)
    policy.text(
        0.98,
        0.92,
        f"{frame['step']}/{total_steps}",
        transform=policy.transAxes,
        ha="right",
        fontsize=7,
        color="#52514e",
    )
    del curve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", default="runs/snake-eprop-0")
    parser.add_argument(
        "--pick",
        choices=["median", "first", "best"],
        default="median",
        help="which recorded evaluation episode to replay; median is representative",
    )
    parser.add_argument("--plan", default="experiments/snake_run_plan.json")
    parser.add_argument("--gif", action="store_true", help="also write an animated GIF")
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from scripts_support_snake import make_controller

    guard = ResourceGuard()
    plan = json.loads(Path(args.plan).read_text())
    result = json.loads((Path(args.trial) / "result.json").read_text())
    graph = load_graph("data/malecns-v1.0/graph.npz")
    standardizer = Standardizer.load(plan["standardizer"])
    controller = make_controller(graph, result["config"]["seed"], standardizer, plan)
    with np.load(Path(args.trial) / "checkpoint.npz", allow_pickle=False) as data:
        controller.motor.actor = data["actor"].copy()
        controller.motor.critic = data["critic"].copy()
        controller.core.magnitude = data["magnitude"].copy()
    # Pick a representative episode, not a flattering one, and record which.
    food = [row["food"] for row in result["after"]]
    order = sorted(range(len(food)), key=lambda i: food[i])
    index = {
        "median": order[len(order) // 2],
        "first": 0,
        "best": order[-1],
    }[args.pick]
    seed = confirm_evaluation_seeds(result["config"]["seed"], plan["evaluation_episodes"])[index]
    frames, game = record(controller, seed, guard)

    fig = plt.figure(figsize=(11, 6))
    board = fig.add_subplot(2, 2, 1)
    sensory = fig.add_subplot(2, 2, 2)
    neural = fig.add_subplot(2, 2, 3)
    policy = fig.add_subplot(2, 2, 4)
    axes = (board, sensory, neural, policy)
    draw(axes, frames[-1], None, len(frames))
    fig.suptitle(
        f"Snake evaluation, environment seed {seed}: {game.eaten} food in "
        f"{game.time} steps (ended: {game.death})"
    )
    fig.tight_layout()
    fig.savefig("artifacts/snake_demo_frame.png", dpi=170)

    if args.gif:
        animation = FuncAnimation(
            fig,
            lambda i: draw(axes, frames[i], None, len(frames)),
            frames=len(frames),
            interval=120,
        )
        animation.save("artifacts/snake_demo.gif", writer=PillowWriter(fps=8), dpi=70)
    plt.close(fig)
    report = {
        "trial": args.trial,
        "selection": args.pick,
        "episode_index": index,
        "episode_food_rank_of_40": int(order.index(index)) + 1,
        "environment_seed": seed,
        "food": game.eaten,
        "steps": game.time,
        "death": game.death,
        "frames": len(frames),
        "gif": bool(args.gif),
        "resources": guard.check(),
    }
    Path("artifacts/snake_demo.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
