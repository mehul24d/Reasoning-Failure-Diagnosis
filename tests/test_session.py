"""Session helpers for time-limited notebook runs (rfd.utils.session)."""

import io
import json
import time
import zipfile

from rfd.utils.session import restore_results, run_targets, zip_results


def _results(root, stages):
    for st in stages:
        (root / "traces" / ".done").mkdir(parents=True, exist_ok=True)
        (root / "traces" / ".done" / f"{st}.json").write_text("{}")
    (root / "traces" / "run_manifest.json").write_text(json.dumps({"model_name": "m"}))
    (root / "traces" / "shards" / "acts").mkdir(parents=True, exist_ok=True)
    (root / "traces" / "shards" / "acts" / "00000.npy").write_bytes(b"x" * 10)
    return root


def test_restore_picks_most_complete_and_keeps_existing(tmp_path):
    inp = tmp_path / "input"
    _results(inp / "old" / "results", ["generate"])
    _results(inp / "new" / "results", ["generate", "label"])
    dest = tmp_path / "work" / "results"
    (dest / "traces").mkdir(parents=True)
    (dest / "traces" / "run_manifest.json").write_text("LOCAL")
    r = restore_results(inp, dest)
    assert r["restored_from"].endswith("new/results") and r["stage_markers"] == 2
    assert (dest / "traces" / ".done" / "label.json").exists()
    assert (dest / "traces" / "run_manifest.json").read_text() == "LOCAL"  # never overwritten
    assert restore_results(inp, dest)["copied"] == 0  # idempotent


def test_restore_from_zip_and_from_nothing(tmp_path):
    src = _results(tmp_path / "src" / "results", ["generate"])
    zip_results(src, tmp_path / "input" / "ds" / "results.zip")
    dest = tmp_path / "dest"
    r = restore_results(tmp_path / "input", dest)
    assert r["copied"] > 0 and (dest / "traces" / ".done" / "generate.json").exists()
    assert restore_results(tmp_path / "missing", tmp_path / "d2")["restored_from"] is None


def test_zip_light_mode_drops_shards(tmp_path):
    src = _results(tmp_path / "results", ["generate"])
    full = zip_results(src, tmp_path / "full.zip")
    light = zip_results(src, tmp_path / "light.zip", max_full_gb=0.0)
    names = zipfile.ZipFile(tmp_path / "light.zip").namelist()
    assert not full["light"] and light["light"]
    assert "results/traces/.done/generate.json" in names
    assert not any("/shards/" in n for n in names)


def test_run_targets_stops_on_failure_and_deadline():
    log = io.StringIO()
    sh = ["sh", "-c"]
    st = run_targets(["echo hi", "exit 3", "echo never"], time.time() + 60, sh, 0, log=log)
    assert [r["status"] for r in st] == ["complete", "FAILED", "not started"]
    assert "hi" in log.getvalue()
    t0 = time.time()
    st = run_targets(["sleep 30 & wait", "echo later"], time.time() + 1, sh, 0, grace=2, log=log)
    assert time.time() - t0 < 10  # the whole process group was killed
    assert st[0]["status"].startswith("stopped at session deadline")
    assert st[1]["status"] == "not started"
    st = run_targets(["echo x"], time.time() + 10, sh, min_seconds=60, log=log)
    assert st[0]["status"] == "not started (session time)"
