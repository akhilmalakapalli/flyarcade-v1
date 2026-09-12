# FLYARCADE-V1_MASTER.md

> **Purpose:** This is the single source of truth for building **flyarcade-v1**, a compact CPU-first NeuroAI project in which a real Drosophila connectome-derived subnetwork controls and learns simple arcade games using spiking neurons and reward-modulated plasticity.
>
> **Primary target:** Finish a rigorous, reproducible v1 locally on a Mac in VS Code, with no GPU or cloud dependency.
>
> **Agent rule:** Any coding agent working in this repository must read this entire file first, then read `STATE.md`, `HANDOFF.md`, and `DECISIONS.md` if they exist. Do not ask the user routine implementation questions. Make reasonable engineering decisions, document them, test them, and continue.

---

# 1. Project identity

## Name

**flyarcade-v1**

## One-sentence description

flyarcade-v1 is a Drosophila connectome-constrained spiking neural network project that learns multiple simple arcade tasks using reward-modulated synaptic plasticity, while testing whether biological topology affects learning, transfer, and robustness.

---

# 2. v1 deliverables

## Canonical biological dataset (user correction, 2026-09-12)

Use **HHMI Janelia MaleCNS v1.0** as the primary and canonical connectome.
Official project: <https://male-cns.janelia.org/>. Preferred access is
`neuprint-python`, server `https://neuprint.janelia.org`, dataset
`male-cns:v1.0`, explicitly pinned on every client/query path.

Select an anatomically justified sensory-to-motor population of approximately
1,000–5,000 neurons, within the existing 4,096-neuron guard. Query only needed
annotations and selected-population connectivity; do not acquire the full
connection graph or increase the 128 MiB download cap merely to pass acquisition.
Preserve original body IDs, direction, synapse counts, cell types, side/region
information and available neurotransmitter annotations. Record criteria, exact
IDs, query text, version and checksums. If the supported client requires a token
and none is available, stop cleanly with token setup instructions. FlyWire is
superseded; synthetic graphs are software fixtures only. Proceed to M2–M5 only
after authentic MaleCNS acquisition and validation. README.md remains untouched.

## Repository

Must include:

```text
README.md
FLYARCADE-V1_MASTER.md
STATE.md
DECISIONS.md
HANDOFF.md
CHANGELOG.md
requirements.txt or pyproject.toml
.gitignore
LICENSE
src/
tests/
scripts/
games/
experiments/
paper/
```

---

# 3. Repository layout

Use this unless there is a compelling reason to change it:

```text
flyarcade-v1/
├── FLYARCADE-V1_MASTER.md
├── STATE.md
├── DECISIONS.md
├── HANDOFF.md
├── CHANGELOG.md
├── README.md
├── pyproject.toml
├── .gitignore
├── src/
│   └── flyarcade/
│       ├── __init__.py
│       ├── config.py
│       ├── resources.py
│       ├── connectome/
│       ├── neural/
│       ├── sensory/
│       ├── motor/
│       ├── viz/
│       └── analysis/
├── games/
├── experiments/
├── tests/
├── scripts/
├── data/
├── runs/
├── cache/
├── artifacts/
└── paper/
```

**Important:** The repository is named `flyarcade-v1`, but the Python import/package directory remains `src/flyarcade/`. Do not rename the Python package to `flyarcade-v1`, because hyphens are not valid Python module identifiers.

---

# 12. Paper output

Create `paper/manuscript.md`.

Suggested title:

> **flyarcade-v1: Reward-Modulated Plasticity in a Drosophila Connectome-Constrained Spiking Network Across Simple Arcade Tasks**

Alternative conservative title:

> **flyarcade-v1: A Reproducible Benchmark for Reward-Modulated Learning in Drosophila Connectome-Derived Spiking Controllers**

---

# 14. Agent collaboration protocol

Codex and Claude Code should behave as a tag team through shared files.

### `STATE.md`

Always contain:

```markdown
# flyarcade-v1 State

## Overall status
IN_PROGRESS / BLOCKED / COMPLETE

## Current milestone
...

## Completed
- ...

## In progress
- ...

## Next
1. ...
2. ...
3. ...

## Tests
- ...

## Latest benchmark
- ...

## Resource status
- RAM:
- disk:
- runtime:

## Known issues
- ...

## Last agent
Codex / Claude Code

## Timestamp
...
```

## Tag-team loop

Every agent session must:

1. Read `FLYARCADE-V1_MASTER.md`.
2. Read `STATE.md`, `HANDOFF.md`, `DECISIONS.md`.
3. Run `python scripts/health_check.py` if it exists.
4. Run the fastest relevant tests.
5. Select the highest-priority incomplete milestone.
6. Implement it.
7. Test it.
8. Benchmark if performance-sensitive.
9. Update `STATE.md`.
10. Update `HANDOFF.md`.
11. Commit locally if appropriate.
12. Continue to the next milestone without asking for routine confirmation.

---

# 18. Bootstrap commands

These are intended for macOS Terminal inside VS Code.

From the desired parent directory:

```bash
mkdir -p flyarcade-v1
cd flyarcade-v1
git init
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

After the agent creates `pyproject.toml`:

```bash
pip install -e ".[dev]"
```

Verify:

```bash
python scripts/health_check.py
pytest -q
```

---

# 23. First prompt for Codex

```text
Read FLYARCADE-V1_MASTER.md completely and treat it as the authoritative project specification for flyarcade-v1. Work autonomously through the milestones in order. First create or inspect STATE.md, HANDOFF.md, and DECISIONS.md. Prioritize computer safety, reproducibility, sparse CPU execution, tests, and truthful scientific claims. Do not ask me routine questions. Make reasonable decisions and document them. Keep working through successive milestones until you hit a genuine external blocker, resource guard, usage limit, or completion. Before stopping for any reason, update STATE.md and HANDOFF.md so another coding agent can resume immediately.
```

---

# 24. First prompt for Claude Code

```text
Read FLYARCADE-V1_MASTER.md completely, then read STATE.md, HANDOFF.md, and DECISIONS.md. You are collaborating with another coding agent on the flyarcade-v1 repository. Resume from the current checkpoint rather than starting over. Work autonomously on the highest-priority incomplete milestone, preserve all safety/resource constraints, run tests, update documentation, and continue through later milestones when the current one passes. Do not ask me routine implementation questions. Before stopping for any reason, update STATE.md and HANDOFF.md with exact next actions and verification commands.
```

---

# 25. Agent review prompt

```text
Act as an independent reviewer of the current flyarcade-v1 repository. Read FLYARCADE-V1_MASTER.md and all coordination files. Do not rewrite working code unnecessarily. Look for scientific validity problems, leakage, unfair controls, numerical instability, performance bottlenecks, resource risks, missing tests, unsupported biological claims, and reproducibility gaps. Fix high-confidence issues, add regression tests, update DECISIONS.md, STATE.md, and HANDOFF.md, then continue with the next unfinished milestone.
```
