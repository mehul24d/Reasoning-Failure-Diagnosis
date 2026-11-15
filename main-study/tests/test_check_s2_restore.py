"""The S2 restore check (DECISIONS D45): restored outputs must hash to the committed S2 records."""
import importlib.util
import json
import shutil

from harness.config import REPO_ROOT
from harness.pipeline import Run, sha256


def _mod():
    spec = importlib.util.spec_from_file_location("check_s2_restore",
                                                  REPO_ROOT / "scripts" / "check_s2_restore.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _fake(tmp_path):
    run = Run.study("m", root=tmp_path / "work")
    recs = tmp_path / "records"
    recs.mkdir(parents=True)
    for f, txt in ((run.problems, "p\n"), (run.traces, "t\n"), (run.scored, "s\n"), (run.positions, "x\n")):
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(txt)
    body = {"problems": {"sha256": sha256(run.problems)}, "generate": {"sha256": sha256(run.traces)},
            "score": {"sha256_scored": sha256(run.scored), "sha256_positions": sha256(run.positions)},
            "verify": {"pass": True}}
    for s, b in body.items():
        (recs / f"{s}.json").write_text(json.dumps(b))
        shutil.copy2(recs / f"{s}.json", run.status_file(s))
    return run, recs


def test_restore_check_passes_and_catches_changes(tmp_path):
    m = _mod()
    run, recs = _fake(tmp_path)
    assert m.check(run, recs)["all_match"]
    run.scored.write_text("changed\n")
    rep = m.check(run, recs)
    assert not rep["all_match"] and not rep["files"][run.scored.name]["match"]
    run, recs = _fake(tmp_path / "b")
    run.status_file("verify").write_text('{"pass": false}')
    assert not m.check(run, recs)["status_files"]["verify"]["match"]
    run.positions.unlink()
    assert m.check(run, recs)["files"][run.positions.name]["restored"] is None


def test_committed_s2_records_have_the_hashes():
    recs = REPO_ROOT / "reports" / "m1" / "s2"
    for s, k in (("problems", "sha256"), ("generate", "sha256"), ("score", "sha256_scored"),
                 ("score", "sha256_positions")):
        assert len(json.loads((recs / f"{s}.json").read_text())[k]) == 64
