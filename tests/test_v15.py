"""v1.5 software tests: seeds, readouts, sources, selection gates (no scientific runs)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src_v15"))

from flyarcade_v15 import seeds as S  # noqa: E402
from flyarcade_v15.features import READOUTS, Fly15, Sensory15, Stacked  # noqa: E402
from flyarcade_v15.selection import select, still_rising, summarize  # noqa: E402

from flyarcade_v14 import study as v14  # noqa: E402
from flyarcade_v14.features import FlyFeatures  # noqa: E402

GRAPH = Path("data/malecns-v1.0/graph.npz")
needs_graph = pytest.mark.skipif(not GRAPH.exists(), reason="canonical graph not present")


def test_seed_namespace_disjoint_from_v14_history_and_confirmatory():
    lo = S.seed_for("catch", "software")
    hi = S.seed_for("flappy", "spare", S.MAX_SEEDS - 1, S.SEED_SLOT - 1) + S.BEHAVIOUR_OFFSET
    v14_hi = v14.seed_for("flappy", "validation_probe", 2, v14.SEED_SLOT - 1) + 2_000_000_000
    assert lo > v14_hi and lo > 12_000_000_000
    assert hi < S.SEED_BASE + len(S.TASK_ORDER) * S.TASK_BLOCK
    intervals = []
    for task in S.TASK_ORDER:
        for purpose in S.PURPOSES:
            start, end = S.purpose_interval(task, purpose)
            assert all(end < a or start > b for a, b in intervals)
            intervals.append((start, end))
        for k in S.CONFIRMATORY_RESERVED:
            a, b = S.confirmatory_interval(task, k)
            assert all(end < a or start > b for start, end in intervals)
    for bad in ("confirmatory", "conf_eval"):
        with pytest.raises(ValueError):
            S.seed_for("pong", bad)
    with pytest.raises(ValueError):
        S.seed_for("pong", "train", S.MAX_SEEDS)


def test_offsets_stay_inside_purpose_block():
    top = S.seed_for("snake", "train", S.MAX_SEEDS - 1, S.SEED_SLOT - 1)
    start, end = S.purpose_interval("snake", "train")
    assert start <= top + S.BEHAVIOUR_OFFSET <= end
    assert top - start < S.NEURAL_OFFSET


@needs_graph
@pytest.mark.parametrize("task", ["catch", "snake", "flappy"])
def test_descending_readout_bit_identical_to_v14(task):
    graph = v14.load_topology("biological")
    width = {"catch": 3, "snake": 21, "flappy": 6}[task]
    a = FlyFeatures(graph, width, 2, task=task, ticks=8, readout="descending")
    b = Fly15(graph, task, 2, ticks=8, readout="descending")
    rng = np.random.default_rng(0)
    for c in range(2):
        a.reset(c, 11 + c)
        b.reset(c, 11 + c)
    for _ in range(3):
        obs = rng.random((2, width))
        assert np.array_equal(a.raw_rates(obs), b.raw_rates(obs))


@needs_graph
def test_readouts_share_the_same_core_and_exclude_inputs_where_declared():
    graph = v14.load_topology("biological")
    full = Fly15(graph, "pong", 1, ticks=4, readout="all")
    inputs = set(full.inputs.tolist())
    for readout in READOUTS:
        src = Fly15(graph, "pong", 1, ticks=4, readout=readout)
        leaks = bool(set(src.select.tolist()) & inputs)
        assert leaks == (readout in ("all", "visual", "visual_cb"))
    obs = np.random.default_rng(1).random((1, 7))
    a = Fly15(graph, "pong", 1, ticks=4, readout="all")
    b = Fly15(graph, "pong", 1, ticks=4, readout="nonvisual")
    a.reset(0, 5)
    b.reset(0, 5)
    ra = a.raw_rates(obs)
    assert np.array_equal(ra[:, b.select], b.raw_rates(obs))


def test_sensory_and_stacked_sources():
    s = Sensory15("flappy", 2)
    x = s(np.full((2, 6), 0.25))
    assert x.shape == (2, 12) and np.all(np.abs(x) <= 1)
    st = Stacked(Sensory15("catch", 2), 3)
    st.reset(0, 1)
    st.reset(1, 2)
    first = st(np.array([[0.0, 0.5, 0.0], [1.0, 0.5, 0.0]]))
    assert first.shape == (2, 24) and np.all(first[:, 8:] == 0)
    second = st(np.array([[0.25, 0.5, 0.25], [1.0, 0.5, 0.25]]))
    assert np.array_equal(second[:, 8:16], first[:, :8])
    st.reset(0, 3)
    assert np.all(st.history[0] == 0) and not np.all(st.history[1] == 0)


def _result(score, probe=0.3, dom=0.6, chosen=100):
    return {
        "status": "COMPLETE",
        "finite": True,
        "score": score,
        "state_probe": {"score": probe},
        "dominant_action_fraction": dom,
        "chosen_checkpoint_transitions": chosen,
    }


def test_selection_gates_controls_and_consistency():
    cfg = {"arch": "gru", "hidden": 128, "ticks": 16, "readout": "descending", "transitions": 1}
    lucky = summarize(
        "lucky", "A", cfg, [_result(1.0), _result(0.2), _result(0.2)], role="headline"
    )
    steady = summarize("steady", "A", cfg, [_result(0.6)] * 3, role="headline")
    collapsed = summarize(
        "collapsed",
        "A",
        cfg,
        [_result(0.9), _result(0.9, probe=0.0), _result(0.9)],
        role="headline",
    )
    control = summarize("leak", "B", {**cfg, "readout": "all"}, [_result(0.99)] * 3, role="control")
    assert not collapsed["eligible"] and not control["eligible"] and control["gates_passed"]
    assert select([lucky, steady, collapsed, control])["name"] == "steady"
    simple = summarize(
        "simple", "A", {**cfg, "arch": "linear"}, [_result(0.6)] * 3, role="headline"
    )
    assert select([steady, simple])["name"] == "simple"
    assert still_rising([_result(1, chosen=90), _result(1, chosen=80), _result(1, chosen=10)], 100)
    assert not still_rising([_result(1, chosen=10)] * 3, 100)
