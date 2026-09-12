"""Ask where the negative learning result comes from, using frozen saved weights.

This is post-hoc analysis of already-completed trials. Nothing here trains,
retunes or selects a controller, and no diagnostic output feeds back into any
model. Probes are decoders fitted by the analyst to measure how much task
information the descending population carries; they are not part of the
controller and their accuracy is not a behavioural result.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from flyarcade.connectome import load_graph
from flyarcade.controller import Controller
from flyarcade.experiment import restore
from flyarcade.games import LaneGame
from flyarcade.resources import ResourceGuard
from flyarcade.sensory import features


def collect(controller, task, seeds, guard, behaviour="model"):
    """Run held-out episodes, recording visited states, descending rates and policy.

    With behaviour="random" the actions actually executed are drawn uniformly with
    a fixed separate stream, so every condition visits the *same* state
    distribution and probe comparisons are not confounded by policy collapse.
    """
    rates, labels, probabilities, actions = [], [], [], []
    behaviour_rng = np.random.default_rng(4242)
    old_rng = controller.rng
    try:
        for seed in seeds:
            controller.rng = np.random.default_rng(seed + 500_000)
            game = LaneGame(task, seed)
            controller.reset()
            while not game.done:
                guard.check()
                observation = game.observe()
                counts = np.zeros(len(controller.outputs))
                f = features(np.asarray(observation))
                for _ in range(4):
                    drive = np.full(controller.graph.n, 0.18)
                    drive[controller.inputs] += 1.5 * (
                        controller.rng.random(len(controller.inputs)) < f[controller.input_channels]
                    )
                    spikes = controller.core.tick(drive, plastic=False)
                    counts += spikes[controller.outputs]
                rate = counts / 4 - 0.25
                x = np.append(rate, 1)
                logits = controller.motor.weights @ x
                p = np.exp(logits - logits.max())
                p /= p.sum()
                action = int(controller.rng.choice(3, p=p))
                if behaviour == "random":
                    action = int(behaviour_rng.integers(0, 3))
                rates.append(rate)
                # Optimal catch direction: -1 left, 0 stay, +1 right.
                labels.append(int(np.sign(round((observation[1] - observation[0]) * 4))))
                probabilities.append(p)
                actions.append(action)
                game.step(action)
    finally:
        controller.rng = old_rng
    return (
        np.asarray(rates),
        np.asarray(labels),
        np.asarray(probabilities),
        np.asarray(actions),
    )


def ridge_probe(train_x, train_y, test_x, test_y, classes, penalty=1.0):
    """Regularised linear discriminant fitted only on the probe training split.

    A one-vs-rest least-squares fit cannot place an ordinal middle class at the
    argmax and would understate decodable information; a shared-covariance
    discriminant does not have that blind spot. This is still a purely linear
    read-out of the descending rates and remains a lower bound on the information
    present, since a linear probe cannot see nonlinear structure.
    """
    centre = train_x.mean(axis=0)
    centred = train_x - centre
    covariance = centred.T @ centred / max(len(centred) - 1, 1)
    covariance += penalty * np.trace(covariance) / len(covariance) * np.eye(len(covariance))
    scores = []
    for label in classes:
        members = centred[train_y == label]
        if not len(members):
            scores.append(np.full(len(test_x), -np.inf))
            continue
        mean = members.mean(axis=0)
        weights = np.linalg.solve(covariance, mean)
        prior = np.log(len(members) / len(centred))
        scores.append((test_x - centre) @ weights - 0.5 * mean @ weights + prior)
    predicted = classes[np.column_stack(scores).argmax(axis=1)]
    return float((predicted == test_y).mean())


def analyse(controller, task, seeds, guard, label, behaviour="model"):
    rates, labels, probabilities, actions = collect(controller, task, seeds, guard, behaviour)
    classes = np.array([-1, 0, 1])
    half = len(rates) // 2
    entropy = float(-(probabilities * np.log(probabilities + 1e-12)).sum(axis=1).mean())
    # Per-neuron discriminability between the two extreme required directions.
    left, right = rates[labels == -1], rates[labels == 1]
    pooled = np.sqrt((left.var(axis=0) + right.var(axis=0)) / 2) + 1e-9
    d_prime = np.abs(left.mean(axis=0) - right.mean(axis=0)) / pooled
    return {
        "condition": label,
        "state_distribution": (
            "visited under the controller's own policy"
            if behaviour == "model"
            else "matched uniform-random behaviour policy, identical across conditions"
        ),
        "decisions": int(len(rates)),
        "mean_policy_entropy_nats": entropy,
        "max_policy_entropy_nats": float(np.log(3)),
        "mean_max_action_probability": float(probabilities.max(axis=1).mean()),
        "action_usage": np.bincount(actions, minlength=3).tolist(),
        "descending_rate_sd_across_states": float(rates.std(axis=0).mean()),
        "descending_neurons_dprime_above_0.5": int((d_prime > 0.5).sum()),
        "max_descending_dprime": float(d_prime.max()),
        "label_majority_rate": float(np.bincount(labels + 1, minlength=3).max() / len(labels)),
        "probe_accuracy_descending_rates": ridge_probe(
            rates[:half], labels[:half], rates[half:], labels[half:], classes
        ),
        "probe_accuracy_shuffled_control": ridge_probe(
            rates[:half],
            np.random.default_rng(11).permutation(labels[:half]),
            rates[half:],
            labels[half:],
            classes,
        ),
    }


def constant_action_reference(task, seeds):
    """Held-out success of the degenerate policies that always emit one action."""
    reference = {}
    for action, name in ((0, "always_left"), (1, "always_stay"), (2, "always_right")):
        scores = []
        for seed in seeds:
            game = LaneGame(task, seed)
            score = 0
            while not game.done:
                _, _, _, info = game.step(action)
                score += info["score"]
            scores.append(score / (game.horizon // 4))
        reference[name] = float(np.mean(scores))
    return reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["catch", "dodge"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    guard = ResourceGuard()
    graph = load_graph("data/malecns-v1.0/graph.npz")
    digest = hashlib.sha256(
        graph.pre.tobytes() + graph.post.tobytes() + graph.counts.tobytes()
    ).hexdigest()
    trial = f"{args.task}-learning-{args.seed}-200"
    seeds = json.loads((Path("runs") / trial / "result.json").read_text())["evaluation_seeds"]
    untrained = Controller(graph, args.seed)
    trained = Controller(graph, args.seed)
    restore(trained, Path("runs") / trial / "checkpoint.npz", digest)
    rows = [
        analyse(untrained, args.task, seeds, guard, "initial_weights"),
        analyse(trained, args.task, seeds, guard, "trained_weights"),
        analyse(untrained, args.task, seeds, guard, "initial_weights_matched_states", "random"),
        analyse(trained, args.task, seeds, guard, "trained_weights_matched_states", "random"),
    ]
    report = {
        "task": args.task,
        "model_seed": args.seed,
        "source_trial": trial,
        "scope": (
            "Post-hoc probe analysis of frozen saved weights. Probe decoders are "
            "analyst tools, not controller components, and were never used to "
            "select or tune any model."
        ),
        "conditions": rows,
        "constant_action_reference": constant_action_reference(args.task, seeds),
        "interpretation_note": (
            "Compare the trained policy's action usage and entropy with these "
            "degenerate single-action references before attributing any held-out "
            "score to state-dependent control."
        ),
        "resources": guard.check(),
    }
    Path(f"artifacts/diagnostics_{args.task}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
