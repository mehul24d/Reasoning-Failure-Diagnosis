"""Check restored S2 outputs against the committed S2 records before any later stage runs.

S3 was restored from the S2 output instead of S4's (DECISIONS D45). Before anything is
generated, the restored problems, traces, scored traces and read-out positions must hash to the
values committed in reports/m1/s2/, and the restored S2 stage-status files must match the
committed copies byte for byte. Any difference: exit 1, and nothing further runs.

    python scripts/check_s2_restore.py --model gemma-2-2b-it --out <report.json>
(BTP_DATA_ROOT points at the restored outputs; the committed records are read from the repo.)
"""
import argparse
import json
import sys
from pathlib import Path

from harness.config import REPO_ROOT
from harness.pipeline import Run, sha256

S2_STAGES = ("problems", "generate", "score", "verify")


def check(run: Run, records: Path) -> dict:
    rec = {s: json.loads((records / f"{s}.json").read_text()) for s in S2_STAGES}
    want = {
        run.problems: rec["problems"]["sha256"],
        run.traces: rec["generate"]["sha256"],
        run.scored: rec["score"]["sha256_scored"],
        run.positions: rec["score"]["sha256_positions"],
    }
    files = {}
    for f, h in want.items():
        got = sha256(f) if f.exists() else None
        files[f.name] = {"committed": h, "restored": got, "match": got == h}
    status = {}
    for s in S2_STAGES:
        f = run.base / "status" / f"{s}.json"
        got = sha256(f) if f.exists() else None
        status[s] = {"committed": sha256(records / f"{s}.json"), "restored": got}
        status[s]["match"] = got == status[s]["committed"]
    ok = all(v["match"] for v in files.values()) and all(v["match"] for v in status.values())
    return {"records": str(records), "files": files, "status_files": status, "all_match": ok}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-2-2b-it")
    ap.add_argument("--records", default=str(REPO_ROOT / "reports" / "m1" / "s2"))
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    rep = check(Run.study(a.model), Path(a.records))
    txt = json.dumps(rep, indent=2)
    print(txt)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(txt)
    if not rep["all_match"]:
        print("RESTORED S2 OUTPUTS DIFFER FROM THE COMMITTED S2 RECORDS: stopping.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
