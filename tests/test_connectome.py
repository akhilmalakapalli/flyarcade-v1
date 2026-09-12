import hashlib
import json
import zipfile

import numpy as np
import pytest

from flyarcade.config import Limits
from flyarcade.connectome import (
    Graph,
    degree_preserving_control,
    import_csv,
    load_graph,
    save_graph,
    synthetic_graph,
)
from flyarcade.connectome.acquire import (
    EXPECTED_MD5,
    EXPECTED_SIZE,
    FILENAME,
    check_download_budget,
    inspect_record,
)
from flyarcade.resources import ResourceLimit


@pytest.fixture
def source(tmp_path):
    def write(rows):
        path = tmp_path / "edges.csv"
        path.write_text("pre_root_id,post_root_id,syn_count\n" + rows)
        manifest = {
            "kind": "synthetic",
            "source_url": "test-fixture",
            "release": "test-1",
            "license": "test-only",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        return path, manifest

    return write


def test_exact_large_ids_aggregation_threshold_and_autapses(source):
    path, manifest = source(
        "720575940625448968,720575940625448969,2\n"
        "720575940625448968,720575940625448969,3\n"
        "720575940625448969,720575940625448968,1\n"
        "720575940625448968,720575940625448968,100\n"
    )
    graph = import_csv(path, manifest, min_synapses=5)
    assert graph.neuron_ids.tolist() == [720575940625448968, 720575940625448969]
    assert graph.pre.tolist() == [0]
    assert graph.post.tolist() == [1]
    assert graph.counts.tolist() == [5]
    extraction = graph.provenance["extraction"]
    assert extraction["autapse_rows_removed"] == 1
    assert extraction["pairs_below_threshold"] == 1


def test_induced_selection_retains_isolates(source):
    path, manifest = source("1,2,5\n2,3,8\n")
    graph = import_csv(path, manifest, selected_ids=[1, 2, 4])
    assert graph.neuron_ids.tolist() == [1, 2, 4]
    assert graph.summary()["isolates"] == 1
    assert graph.e == 1
    assert graph.provenance["extraction"]["outside_selection_rows"] == 1


@pytest.mark.parametrize(
    "row",
    [
        "1.0,2,1\n",
        "0,2,1\n",
        "1,2,-1\n",
        "1,2,0\n",
        "1,2,nan\n",
        "18446744073709551616,2,1\n",
        "1,2,9223372036854775808\n",
        "1,2\n",
    ],
)
def test_invalid_csv_values_rejected(source, row):
    with pytest.raises(ValueError):
        import_csv(*source(row))


def test_hash_and_manifest_are_required(source):
    path, manifest = source("1,2,5\n")
    with pytest.raises(ValueError, match="requires"):
        import_csv(path, {})
    manifest["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256"):
        import_csv(path, manifest)


def test_graph_import_never_silently_truncates(source):
    with pytest.raises(ValueError, match="limits"):
        import_csv(*source("1,2,1\n2,3,1\n"), limits=Limits(max_neurons=2))
    with pytest.raises(ValueError, match="limits"):
        import_csv(*source("1,2,1\n2,3,1\n"), limits=Limits(max_edges=1))


@pytest.mark.parametrize(
    "pre,post,counts",
    [
        ([0], [0], [1]),
        ([0, 0], [1, 1], [1, 1]),
        ([0], [2], [1]),
        ([-1], [1], [1]),
        ([0], [1], [0]),
        ([0], [1], [1.5]),
        ([0], [], [1]),
    ],
)
def test_graph_invariants(pre, post, counts):
    with pytest.raises(ValueError):
        Graph(
            np.array([1, 2]), np.array(pre), np.array(post), np.array(counts), {"kind": "synthetic"}
        )


def test_synthetic_reproducibility_and_no_autapses():
    a, b = synthetic_graph(seed=13), synthetic_graph(seed=13)
    np.testing.assert_array_equal(a.pre, b.pre)
    np.testing.assert_array_equal(a.post, b.post)
    np.testing.assert_array_equal(a.counts, b.counts)
    assert a.provenance["kind"] == "synthetic"
    assert np.all(a.pre != a.post)
    assert synthetic_graph(n=1, edges=0).e == 0
    with pytest.raises(ValueError):
        synthetic_graph(n=2, edges=3)


def test_control_preserves_directed_degrees_and_outgoing_strength():
    graph = synthetic_graph(n=64, edges=512, seed=91)
    control = degree_preserving_control(graph, seed=9)
    assert control.provenance["accepted"] > 0
    assert np.any(graph.post != control.post)
    for endpoint in ("pre", "post"):
        np.testing.assert_array_equal(
            np.bincount(getattr(graph, endpoint), minlength=graph.n),
            np.bincount(getattr(control, endpoint), minlength=graph.n),
        )
    np.testing.assert_array_equal(control.counts, graph.counts)
    np.testing.assert_array_equal(control.neuron_ids, graph.neuron_ids)
    np.testing.assert_array_equal(control.pre, graph.pre)
    np.testing.assert_array_equal(
        degree_preserving_control(graph, seed=9).post,
        control.post,
    )
    assert control.provenance["parent"]["kind"] == "synthetic"


def test_no_swap_possible_is_reported():
    graph = synthetic_graph(n=3, edges=6)
    assert degree_preserving_control(graph).provenance["accepted"] == 0


def test_roundtrip_without_pickle(tmp_path):
    graph = degree_preserving_control(synthetic_graph())
    path = tmp_path / "graph.npz"
    save_graph(graph, path)
    loaded = load_graph(path)
    assert loaded.provenance == graph.provenance
    assert loaded.summary() == graph.summary()
    np.testing.assert_array_equal(loaded.post, graph.post)
    with pytest.raises(ValueError):
        loaded.post[0] = 999
    assert json.dumps(loaded.provenance)


def test_archive_expansion_guard(tmp_path):
    path = tmp_path / "large.npz"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("payload", b"0" * (32 * 2**20 + 1))
    with pytest.raises(ResourceLimit, match="expanded"):
        load_graph(path)


def test_pinned_source_size_preflight():
    record = {
        "metadata": {"license": {"id": "cc-by-4.0"}},
        "files": [
            {
                "key": FILENAME,
                "size": EXPECTED_SIZE,
                "checksum": "md5:" + EXPECTED_MD5,
                "links": {"self": "https://zenodo.org/example"},
            }
        ],
    }
    report = inspect_record(record)
    assert report["status"] == "RESOURCE_GUARD"
    assert report["downloaded"] is False
    with pytest.raises(ResourceLimit, match="budget"):
        check_download_budget(report)
    record["files"][0]["checksum"] = "md5:changed"
    with pytest.raises(ValueError, match="metadata differs"):
        inspect_record(record)
