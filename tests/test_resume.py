"""Restart-safety helpers (rfd.utils.resume) and the tolerant JSONL reader."""

import json

import numpy as np
import pytest

from rfd.utils.io import read_jsonl
from rfd.utils.resume import (
    ResultCache,
    ShardedArray,
    StageMarker,
    check_fp16_range,
    check_manifest,
    file_fingerprint,
    open_array,
    results_path,
)


def test_results_path_maps_only_results_prefix(monkeypatch, tmp_path):
    monkeypatch.delenv("RFD_RESULTS_DIR", raising=False)
    assert str(results_path("results/traces")) == "results/traces"
    monkeypatch.setenv("RFD_RESULTS_DIR", str(tmp_path / "drive"))
    assert results_path("results/traces") == tmp_path / "drive" / "traces"
    assert str(results_path("configs/x.yaml")) == "configs/x.yaml"
    assert results_path(tmp_path / "abs") == tmp_path / "abs"


def _fill(arr: ShardedArray, blocks: list[np.ndarray]) -> None:
    start = 0
    for k, b in enumerate(blocks):
        arr.write(k, start, b)
        start += len(b)


@pytest.mark.parametrize("scratch", [False, True])
def test_sharded_array_assembles_in_order(monkeypatch, tmp_path, scratch):
    if scratch:
        monkeypatch.setenv("RFD_SCRATCH_DIR", str(tmp_path / "scratch"))
    else:
        monkeypatch.delenv("RFD_SCRATCH_DIR", raising=False)
    rng = np.random.default_rng(0)
    blocks = [rng.standard_normal((n, 3, 2)).astype(np.float32) for n in (4, 0, 5)]
    arr = ShardedArray(tmp_path / "out", "acts", (3, 2), "float16", inputs="v1")
    _fill(arr, blocks[:2])
    assert arr.has(0) and arr.has(1) and not arr.has(2)
    with pytest.raises(RuntimeError):
        arr.complete(9, 3)  # shard 2 missing
    _fill(arr, blocks)
    arr.complete(9, 3)
    got = open_array(tmp_path / "out", "acts")
    np.testing.assert_array_equal(got, np.concatenate(blocks).astype(np.float16))
    assert (tmp_path / "scratch").exists() == scratch
    # a second open reuses the assembled file
    assert open_array(tmp_path / "out", "acts").shape == (9, 3, 2)


def test_sharded_array_discards_shards_from_other_inputs(tmp_path):
    a = ShardedArray(tmp_path, "acts", (2,), "float32", inputs="v1")
    a.write(0, 0, np.ones((3, 2)))
    assert ShardedArray(tmp_path, "acts", (2,), "float32", inputs="v1").has(0)
    assert not ShardedArray(tmp_path, "acts", (2,), "float32", inputs="v2").has(0)


def test_sharded_array_rejects_wrong_layout(tmp_path):
    a = ShardedArray(tmp_path, "acts", (2,), "float32")
    a.write(0, 0, np.ones((3, 2)))
    a.write(1, 4, np.ones((1, 2)))  # gap: rows 3..3 missing
    with pytest.raises(RuntimeError):
        a.complete(5, 2)
    with pytest.raises(ValueError):
        a.write(2, 5, np.ones((1, 3)))


def test_fp16_range_guard(tmp_path):
    assert check_fp16_range(np.array([1.0, -300.0]), "x") == 300.0
    with pytest.raises(ValueError):
        check_fp16_range(np.array([7e4]), "x")
    with pytest.raises(ValueError):
        ShardedArray(tmp_path, "a", (1,), "float16").write(0, 0, np.array([[np.inf]]))


def test_stage_marker_tracks_inputs(tmp_path):
    m = StageMarker(tmp_path, "probe", inputs="abc")
    assert not m.done() and not m.skip(force=False)
    m.mark({"n": 3})
    assert m.done() and m.skip(force=False) and not m.skip(force=True)
    assert not StageMarker(tmp_path, "probe", inputs="changed").done()


def test_result_cache_computes_once(tmp_path):
    c = ResultCache(tmp_path / "c")
    calls = []

    def fn():
        calls.append(1)
        return {"x": [1, 2], "nan": float("nan")}

    a = c.get_or_compute("test|add|0.5|random3", fn)
    b = ResultCache(tmp_path / "c").get_or_compute("test|add|0.5|random3", fn)
    assert a["x"] == b["x"] == [1, 2] and np.isnan(b["nan"]) and len(calls) == 1
    assert not c.has("test|add|0.5|random4") and c.n_cached() == 1


def test_file_fingerprint_changes_with_content(tmp_path):
    p = tmp_path / "f.jsonl"
    p.write_text("a\n")
    f1 = file_fingerprint(p)
    p.write_text("b\n")
    assert file_fingerprint(p) != f1


def test_read_jsonl_drops_only_a_truncated_last_line(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text(json.dumps({"a": 1}) + "\n" + '{"a": 2')
    assert list(read_jsonl(p)) == [{"a": 1}]
    p.write_text('{"a": 1\n' + json.dumps({"a": 2}) + "\n")
    with pytest.raises(json.JSONDecodeError):
        list(read_jsonl(p))


def test_manifest_refuses_a_second_model(tmp_path):
    gemma = {
        "model_name": "google/gemma-2-2b-it",
        "dtype": "float32",
        "attn_implementation": "eager",
    }
    check_manifest(tmp_path, gemma)
    check_manifest(tmp_path, dict(gemma))  # same model: fine
    with pytest.raises(RuntimeError):
        check_manifest(tmp_path, {"model_name": "Qwen/Qwen2.5-1.5B-Instruct", "dtype": "auto"})
