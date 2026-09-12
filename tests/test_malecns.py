"""Mock API records are synthetic fixtures, never biological acquisition evidence."""

from types import SimpleNamespace

import numpy as np
import pytest

from flyarcade.connectome import load_graph, save_graph
from flyarcade.connectome.malecns import (
    DATASET,
    SERVER,
    TOKEN_ENV,
    AuthenticationRequired,
    adjacency_query,
    create_client,
    graph_from_records,
    metadata_query,
    read_token,
    records_from_response,
    selection_queries,
)
from flyarcade.resources import ResourceLimit


def test_missing_token_stops_before_client_creation(tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("must not construct client or touch network without credentials")

    with pytest.raises(AuthenticationRequired, match="Account"):
        create_client(environ={}, token_file=tmp_path / "missing", factory=forbidden)


def test_token_file_and_environment_precedence(tmp_path):
    path = tmp_path / "token"
    path.write_text(" fixture-token-file \n")
    assert read_token({}, path) == "fixture-token-file"
    assert read_token({TOKEN_ENV: "fixture-env"}, path) == "fixture-env"
    path.write_text(" " * 16385)
    with pytest.raises(ValueError, match="16 KiB"):
        read_token({}, path)


def test_client_pinned_no_default_dataset_or_flywire_fallback():
    calls = []

    def factory(server, **kwargs):
        calls.append((server, kwargs))
        return SimpleNamespace(server=server, dataset=kwargs["dataset"])

    create_client(environ={TOKEN_ENV: "fixture-only"}, token_file=None, factory=factory)
    assert calls == [
        (
            SERVER,
            {
                "dataset": DATASET,
                "token": "fixture-only",
                "verify": True,
                "progress": False,
            },
        )
    ]
    with pytest.raises(ValueError, match="canonical"):
        create_client(
            environ={TOKEN_ENV: "fixture-only"},
            token_file=None,
            factory=lambda *a, **k: SimpleNamespace(server=SERVER, dataset="male-cns:v0.9"),
        )


def test_anatomical_preview_is_bounded_and_count_is_separate():
    queries = selection_queries(["GNG", "LAL(R)"], known_rois=["GNG", "LAL(R)"])
    assert "count(n)" in queries["count"]
    assert "LIMIT 64" in queries["preview"]
    assert "n.`LAL(R)`" in queries["preview"]
    with pytest.raises(ValueError):
        selection_queries(["invented"], known_rois=["GNG"])
    with pytest.raises(ValueError):
        selection_queries(["bad`name"], known_rois=["bad`name"])


@pytest.mark.parametrize("ids", [[], [1.0], [True], [1, 1], [0], [2**64], ["1) MATCH"]])
def test_query_ids_are_exact_and_safe(ids):
    with pytest.raises(ValueError):
        metadata_query(ids)


def test_query_batches_and_both_endpoints_constrained():
    query = adjacency_query([11], [11, 13])
    assert "a.bodyId IN [11]" in query
    assert "b.bodyId IN [11, 13]" in query
    assert "-[e:ConnectsTo]->" in query
    assert "e.weight AS weight" in query
    assert "e.roiInfo AS roiInfo" in query
    with pytest.raises(ValueError):
        adjacency_query([15], [11, 13])
    with pytest.raises(ValueError):
        adjacency_query(list(range(1, 18)), list(range(1, 18)))
    with pytest.raises(ValueError):
        metadata_query(list(range(1, 66)))
    with pytest.raises(ValueError):
        adjacency_query([1], list(range(1, 4098)))


def test_response_schema_and_row_limit():
    response = {"columns": ["bodyId"], "data": [[2**53 + 1]]}
    assert records_from_response(response, ["bodyId"], 1)[0]["bodyId"] == 2**53 + 1
    with pytest.raises(ValueError):
        records_from_response(response, ["wrong"], 1)
    with pytest.raises(ResourceLimit):
        records_from_response(response, ["bodyId"], 0)
    with pytest.raises(ValueError):
        records_from_response({"columns": ["bodyId"], "data": [[]]}, ["bodyId"], 1)


@pytest.fixture
def records():
    neurons = [
        {
            "bodyId": 11,
            "type": "fixture-sensory",
            "somaSide": "L",
            "roiInfo": '{"GNG":{"pre":7}}',
            "predictedNt": "acetylcholine",
            "predictedNtConfidence": 0.9,
            "consensusNt": None,
        },
        {"bodyId": 13, "type": "fixture-motor", "rootSide": "R"},
        {"bodyId": 17, "type": None},
    ]
    edges = [
        {"pre": 11, "post": 13, "weight": 7, "roiInfo": '{"GNG":{"post":7},"Brain":{"post":7}}'}
    ]
    provenance = {
        "kind": "synthetic",
        "dataset": DATASET,
        "server": SERVER,
        "selected_ids": [11, 13, 17],
        "selection_criteria": "synthetic test fixture",
        "queries": [metadata_query([11, 13, 17]), adjacency_query([11], [11, 13, 17])],
    }
    return neurons, edges, provenance


def test_loader_preserves_annotations_ids_isolates_and_direction(records, tmp_path):
    neurons, edges, provenance = records
    graph = graph_from_records(*records)
    assert graph.neuron_ids.tolist() == [11, 13, 17]
    assert graph.summary()["isolates"] == 1
    assert graph.pre.tolist() == [0] and graph.post.tolist() == [1]
    assert graph.counts.tolist() == [7]  # Not 14 from overlapping ROI hierarchy.
    assert graph.provenance["neurons"] == neurons
    assert graph.provenance["edge_annotations"] == edges
    assert graph.provenance["kind"] == "synthetic"
    assert graph.provenance["synaptic_signs"] == "not-inferred"
    path = tmp_path / "fixture.npz"
    save_graph(graph, path)
    reloaded = load_graph(path)
    assert reloaded.provenance == graph.provenance
    np.testing.assert_array_equal(reloaded.neuron_ids, graph.neuron_ids)


@pytest.mark.parametrize("failure", ["dataset", "missing_neuron", "duplicate", "outside", "float"])
def test_loader_rejects_incomplete_or_misdeclared_records(records, failure):
    neurons, edges, provenance = records
    if failure == "dataset":
        provenance["dataset"] = "male-cns:v0.9"
    elif failure == "missing_neuron":
        neurons.pop()
    elif failure == "duplicate":
        edges.append(edges[0].copy())
    elif failure == "outside":
        edges[0]["post"] = 999
    else:
        edges[0]["weight"] = 7.0
    with pytest.raises(ValueError):
        graph_from_records(neurons, edges, provenance)


def test_bounded_client_transport_uses_timeout_and_cumulative_guard(monkeypatch):
    neuprint = pytest.importorskip("neuprint")
    requests_seen = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield b"{}"

    def request(*args, **kwargs):
        requests_seen.append((args, kwargs))
        return Response()

    def fake_init(self, server, **kwargs):
        self.server, self.dataset, self.verify = server, kwargs["dataset"], kwargs["verify"]
        self.session = SimpleNamespace(request=request)

    monkeypatch.setattr(neuprint.Client, "__init__", fake_init)
    client = create_client(environ={TOKEN_ENV: "fixture-only"}, token_file=None)
    response = client._fetch(SERVER + "/api/custom/custom", json={}, ispost=True)
    assert response._content == b"{}"
    assert requests_seen[0][1]["timeout"] == (5, 15)
    assert requests_seen[0][1]["stream"] is True
    assert requests_seen[0][1]["verify"] is True
    client.received_bytes = 128 * 2**20
    with pytest.raises(ResourceLimit):
        client._fetch(SERVER + "/api/custom/custom", json={}, ispost=True)


def test_external_annotations_keep_archive_small_and_verify_manifest(records, tmp_path):
    import hashlib
    import json
    import zipfile

    bundle = tmp_path / "manifest.json"
    bundle.write_text(json.dumps({"fixture": True}))
    ref = {"path": str(bundle), "sha256": hashlib.sha256(bundle.read_bytes()).hexdigest()}
    graph = graph_from_records(*records, annotation_bundle=ref)
    assert graph.provenance["edge_annotations"]["sha256"] == ref["sha256"]
    path = tmp_path / "graph.npz"
    save_graph(graph, path)
    with zipfile.ZipFile(path) as archive:
        # UTF-8 byte storage avoids NumPy Unicode's four-byte-per-character expansion.
        assert archive.getinfo("provenance.npy").file_size < 4096
    assert load_graph(path).provenance == graph.provenance
    bundle.write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        graph_from_records(*records, annotation_bundle=ref)
