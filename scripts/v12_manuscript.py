"""Append a generated v1.2 report only after the matching scientific audit passes."""

import hashlib
import json
from pathlib import Path

ROOT = Path("artifacts/v12")


def main():
    path = ROOT / "results_summary.json"
    summary = json.loads(path.read_text())
    audit = json.loads((ROOT / "scientific_audit.json").read_text())
    assert audit["status"] == "PASS"
    for filename, expected in audit["analysis_files"].items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == expected
    assert audit["summary_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert audit["plan_sha256"] == summary["plan_sha256"]
    lines = [
        "\n\n---\n\n# v1.2 extension: four further artificial sensorimotor tasks\n",
        "This extension preserves the historical report above verbatim. Catch and Dodge "
        "are historical v1.1 comparisons, not retuned experiments. The new relative-action "
        "6×6 Snake differs from the historical absolute-action 8×8 Snake.\n",
        "## Methods\n",
        "The same authentic 2,040-neuron MaleCNS v1.0 subnetwork and frozen recurrent "
        "spiking substrate drive a learned artificial actor–critic readout. Engineered "
        "observations and game actions are not biological sensory/motor interfaces. "
        "No recurrent synapse learns. The source graph was not reselected.\n",
        "A bounded two-candidate development search preceded the frozen v1.2 protocol. "
        "Three independent training experience/exploration seeds per condition and "
        "40 held-out episodes per seed were used. The task-specific budgets, reward "
        "definitions, complete seed rules and source hashes are in "
        "`experiments/v12_run_plan.json`; detailed methods are in "
        "`experiments/METHODS_V12.md`. Every topology has its own development-fitted "
        "standardizer. No confirmatory score was used for tuning.\n",
        "Success was defined prospectively: every biological seed must improve over "
        "its initial held-out score, beat paired random by more than 0.05, and show "
        "fixed-history observation-intervention action diversity above 0.05. "
        "Weight changes alone do not establish learning success.\n",
        "## Held-out results\n",
        (ROOT / "table.md").read_text(),
        "\nEntries are means across three training seeds. The task-defined scores "
        "share [0,1] bounds, not common behavioral units: pipe fraction, return fraction, "
        "brick fraction, and capped food/10 for new Snake. Catch/Dodge retain their "
        "original success fractions. Descriptive 95% bootstrap intervals and paired "
        "contrasts are available in `artifacts/v12/results_summary.json`; no p-values "
        "or population-level claims are made.\n",
    ]
    for task in ("flappy", "pong", "breakout", "snake"):
        d = summary["tasks"][task]
        biological = d["primary"]["biological"]
        interval = biological["descriptive_bootstrap_95"]
        verdict = "met" if d["meets_frozen_success_criteria"] else "did not meet"
        passed = sum(
            c["improved"] and c["beats_random_margin"] and c["state_sensitive"]
            for c in d["success_checks"]
        )
        lines.append(
            f"**{task.title()} {verdict} the frozen success criteria** ({passed}/3 seeds "
            f"met all components). Before {d['before']['mean']:.3f}, after "
            f"{biological['mean']:.3f} [{interval[0]:.3f}, {interval[1]:.3f}]; "
            f"seed scores {', '.join(f'{v:.3f}' for v in biological['seed_values'])}. "
            f"Fixed-history state-probe diversity averaged {d['state_probe']['mean']:.3f}.\n"
        )
        keys = {
            "flappy": ("obstacles_passed", "steps"),
            "pong": ("hits", "attempts"),
            "breakout": ("bricks_destroyed", "misses"),
            "snake": ("food", "steps"),
        }[task]
        available = [
            f"{k.replace('_', ' ')} {d['raw_biological'][k]['mean']:.3f}"
            for k in keys
            if k in d["raw_biological"]
        ]
        lines.append("Mean raw biological outcomes: " + ", ".join(available) + ".\n")
    lines += [
        "## Topology and perturbations\n",
        "The degree-preserving rewired control preserves in/out degree and outgoing "
        "count assignments, but not all biological properties or incoming strength; "
        "it is not a uniformly sampled random graph. These comparisons concern "
        "the artificial task, input mapping and learned readout used here.\n",
        "| Task | Bio−rewired | Intact | Noise SD 0.1 | Edge 10% | Neuron 10% |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task, d in summary["tasks"].items():
        vals = [d["contrasts"]["biological_minus_rewired"]["mean"]] + [
            d["perturbations"][k]["mean"]
            for k in ("after", "sensory_noise", "edge_ablation", "neuron_ablation")
        ]
        lines.append("| " + task + " | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    lines += [
        "\nPerturbations used fixed masks with no retraining. Their interpretation is "
        "limited when the intact policy fails the behavioral criteria. Neither "
        "resilience of a constant policy nor lesions that improve a poor policy "
        "demonstrate biologically meaningful robustness.\n",
        "## Post-hoc neural representations\n",
        "Matched alternating heuristic/random behavior produced identical sampled "
        "states across topologies. A ridge probe predicted heuristic-action labels "
        "using separate whole episodes for training and testing. The labels were "
        "never controller inputs. Raw activity supplied participation ratio, "
        "correlation and variance. Biological feature replicas are deterministic "
        "duplicates across these model seeds; they are not independent biological "
        "samples. Rewired replicas change graph topology.\n",
        "| Task | Bio probe | Rewired probe | Majority | Bio PR | Rewired PR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    rep = json.loads((ROOT / "representations.json").read_text())
    for task in ("flappy", "pong", "breakout", "snake"):

        def avg(condition, key):
            v = [r[key] for r in rep if r["task"] == task and r["condition"] == condition]
            return sum(v) / len(v)

        vals = [
            avg("biological", "linear_probe_accuracy"),
            avg("rewired", "linear_probe_accuracy"),
            avg("biological", "majority_accuracy"),
            avg("biological", "participation_ratio"),
            avg("rewired", "participation_ratio"),
        ]
        lines.append("| " + task + " | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    lines += [
        "\nProbe accuracy must be interpreted against the majority baseline. "
        "These post-hoc summaries do not establish a causal explanation of learning "
        "or a general ranking of biological and rewired circuitry.\n",
        "## Limitations and reproducibility\n",
        "Only about 15.05% of incoming and 7.12% of outgoing synaptic weight is "
        "retained in this selected circuit. Artificial background drive, simplified "
        "LIF dynamics, transmitter-sign assumptions and engineered inputs constrain "
        "interpretation. Development was deliberately limited to two candidates; "
        "Flappy and Snake feature-fit state coverage was small. Failure at these "
        "budgets is a result about this protocol, not proof of impossibility.\n",
        f"The scientific audit verified {audit['trials']} new trials and preserved "
        f"{audit['historical_files_preserved']} historical file hashes. It replayed "
        "clean and all three perturbation evaluations from representative checkpoints "
        "for every new task and repeated one full-budget Flappy training history "
        "exactly. All Stage A recurrent and frozen-control invariants passed. "
        f"Maximum trial segment duration was {audit['max_segment_seconds']:.2f} seconds; "
        f"maximum recorded trial RSS snapshot was {audit['max_rss_snapshot_mb']:.1f} MiB. "
        "These are cooperative guards and memory snapshots, not measured peaks.\n",
        "Reproduction commands are in `RUNBOOK.md`. Machine-readable primary, "
        "contrast, perturbation, raw-task and representation tables, complete result "
        "archives and five PNG/SVG figures are under `artifacts/v12/`. Checkpoints "
        "and the biological data remain in the local ignored run/data directories.\n",
    ]
    extension = "\n".join(lines)
    (ROOT / "manuscript_extension.md").write_text(extension)
    historical = Path("paper/manuscript_v11_historical.md").read_text()
    Path("paper/manuscript.md").write_text(historical + extension)
    print("Manuscript extension generated from audited artifacts; historical text preserved.")


if __name__ == "__main__":
    main()
