"""The Kaggle-output backup check: the store check and the failure report."""
import hashlib
import importlib.util
import json

import numpy as np

from harness.config import REPO_ROOT


def _mod():
    spec = importlib.util.spec_from_file_location("verify_kaggle_backup",
                                                  REPO_ROOT / "scripts" / "verify_kaggle_backup.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_store_check_catches_bad_and_unlisted_shards(tmp_path, monkeypatch):
    m = _mod()
    store, recs = tmp_path / "store", tmp_path / "records"
    store.mkdir()
    (recs / "s4").mkdir(parents=True)
    shards = {}
    for i, rows in enumerate((3, 5)):
        f = store / f"resid_w0_{i:05d}.npy"
        np.save(f, np.zeros((rows, 4, 2), dtype=np.float16))
        shards[f.name] = {"rows": rows, "bytes": f.stat().st_size, "index_entries": rows}
    (store / "index.jsonl").write_text('{"row": 0}\n')
    man = {"shards": shards, "bytes": sum(s["bytes"] for s in shards.values()),
           "index_sha256": hashlib.sha256((store / "index.jsonl").read_bytes()).hexdigest()}
    (store / "manifest.json").write_text(json.dumps(man))
    (recs / "s4" / "manifest.json").write_text(json.dumps(man))
    monkeypatch.setattr(m, "RECORDS", recs)
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)

    assert m.all_match(m.store_check(store))
    np.save(store / "resid_w0_00001.npy", np.zeros((4, 4, 2), dtype=np.float16))      # one row short
    np.save(store / "resid_w9_00000.npy", np.zeros((1, 4, 2), dtype=np.float16))      # not in the manifest
    rep = m.store_check(store)
    assert not rep["shards"]["match"]
    assert rep["shards"]["bad"][0]["shard"] == "resid_w0_00001.npy" and rep["shards"]["bad"][0]["rows"] == 4
    assert rep["shards"]["unlisted"] == ["resid_w9_00000.npy"]
    assert m.failures({"k": rep}) == ["k/shards"]


def test_missing_notebook_output_fails(tmp_path):
    m = _mod()
    for k in m.KERNELS:
        (tmp_path / k).mkdir()
    assert m.main([str(tmp_path)]) == 1
