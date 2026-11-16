"""Check a local download of the study's Kaggle notebook outputs against the committed records.

The S2-S4 outputs live in three Kaggle notebooks: btp-minor-s2 (problems, traces, scores),
btp-minor-s3-from-s2 (the same S2 files restored, plus the sampled runs; D45) and btp-minor-s4
(the same S2 files restored, plus the 13.67 GB activation store). Download them with

    kaggle kernels output mehuldahiya/<kernel> -p <backup>/<kernel>

then run

    python scripts/verify_kaggle_backup.py <backup> --out reports/m1/backup/verify.json \
        --sha-out reports/m1/backup/kaggle_outputs_sha256.json

Checks, per notebook (exit 1 if any fails):
- the S2 data files hash to the committed S2 records, and the S2 status files equal the
  committed copies byte for byte (the same check S3 ran before generating, scripts/check_s2_restore.py);
- the stage's own data and record files equal what was committed from that stage (sampled-run
  sha256 for S3; status files, session status, run summary, restore check);
- the activation store (S4): manifest.json equals the committed one; index.jsonl hashes to
  the manifest's index_sha256; every shard in the manifest is present with the manifest's
  byte size and a float16 .npy header whose row count matches; no unlisted shards.
The committed records have no per-shard sha256, so `--sha-out` writes the sha256 of every
downloaded file. Committing that file lets any later copy (e.g. an off-site backup) be checked
exactly with `--against`; add `--hf-repo <user>/<repo>` to check the off-site Hugging Face copy
against the same file without downloading it.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # works under `python -I`

import numpy as np                                                # noqa: E402

from harness.config import REPO_ROOT                              # noqa: E402
from harness.pipeline import Run                                  # noqa: E402

MODEL = "gemma-2-2b-it"
RECORDS = REPO_ROOT / "reports" / "m1"
KERNELS = ("btp-minor-s2", "btp-minor-s3-from-s2", "btp-minor-s4")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def same(got: Path, committed: Path) -> dict:
    g = sha(got) if got.exists() else None
    return {"file": str(got), "committed": str(committed.relative_to(REPO_ROOT)), "match": g == sha(committed)}


def s2_check(root: Path) -> dict:
    """The S2 restore check (check_s2_restore.check), without creating directories in the download."""
    run = Run.study(MODEL, root=root)
    base = run.base
    rec = {s: json.loads((RECORDS / "s2" / f"{s}.json").read_text())
           for s in ("problems", "generate", "score", "verify")}
    want = {run.problems: rec["problems"]["sha256"],
            base / "s2_traces.jsonl": rec["generate"]["sha256"],
            base / "s2_traces_scored.jsonl": rec["score"]["sha256_scored"],
            base / "s2_positions.jsonl": rec["score"]["sha256_positions"]}
    out = {f.name: {"match": (sha(f) if f.exists() else None) == h} for f, h in want.items()}
    for s in rec:
        out[f"status/{s}.json"] = same(base / "status" / f"{s}.json", RECORDS / "s2" / f"{s}.json")
    return out


def store_check(store: Path) -> dict:
    man_c = json.loads((RECORDS / "s4" / "manifest.json").read_text())
    out = {"manifest.json": same(store / "manifest.json", RECORDS / "s4" / "manifest.json"),
           "index.jsonl": {"match": (store / "index.jsonl").exists()
                           and sha(store / "index.jsonl") == man_c["index_sha256"]}}
    bad, total = [], 0
    for name, m in man_c["shards"].items():
        f = store / name
        if not f.exists():
            bad.append({"shard": name, "problem": "missing"})
            continue
        size = f.stat().st_size
        total += size
        with open(f, "rb") as fh:
            ver = np.lib.format.read_magic(fh)
            read = np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0
            shape, _, dtype = read(fh)
        if size != m["bytes"] or shape[0] != m["rows"] or dtype != np.float16:
            bad.append({"shard": name, "bytes": size, "want_bytes": m["bytes"], "rows": shape[0],
                        "want_rows": m["rows"], "dtype": str(dtype)})
    extra = sorted(p.name for p in store.glob("resid_*.npy") if p.name not in man_c["shards"])
    out["shards"] = {"n": len(man_c["shards"]), "bytes": total, "want_bytes": man_c["bytes"],
                     "bad": bad, "unlisted": extra,
                     "match": not bad and not extra and total == man_c["bytes"]}
    return out


def kernel_checks(k: str, root: Path) -> dict:
    out = {"s2_data": s2_check(root)}
    base = root / "data" / "generated" / MODEL
    if k == "btp-minor-s2":
        out["records"] = {"session_status.json": same(root / "session_status.json", RECORDS / "s2" / "session_status.json"),
                          "pip_freeze.txt": same(root / "pip_freeze.txt", RECORDS / "s2" / "pip_freeze.txt")}
    elif k == "btp-minor-s3-from-s2":
        want = json.loads((RECORDS / "s3" / "sampled.json").read_text())["sha256"]
        f = base / "s2_sampled.jsonl"
        out["records"] = {
            "s2_sampled.jsonl": {"match": f.exists() and sha(f) == want},
            **{f"status/{s}.json": same(base / "status" / f"{s}.json", RECORDS / "s3" / f"{s}.json")
               for s in ("sampled", "report")},
            **{n: same(root / n, RECORDS / "s3" / n) for n in ("session_status.json", "sessions.jsonl", "pip_freeze.txt")},
            "run_summary": same(root / "reports" / "m1" / f"s2_run_summary_{MODEL}.json",
                                RECORDS / "s3" / f"s2_run_summary_{MODEL}_kaggle.json"),
            "restore_check": same(root / "reports" / "m1" / f"s3_restore_check_{MODEL}.json",
                                  RECORDS / "s3" / f"s3_restore_check_{MODEL}.json")}
    elif k == "btp-minor-s4":
        out["records"] = {"status/cache.json": same(base / "status" / "cache.json", RECORDS / "s4" / "cache.json"),
                          "session_status.json": same(root / "session_status.json", RECORDS / "s4" / "session_status.json")}
        out["store"] = store_check(root / "cache" / "store" / MODEL)
    return out


def all_match(d) -> bool:
    if isinstance(d, dict):
        if "match" in d:
            return bool(d["match"])
        return all(all_match(v) for v in d.values())
    return True


def failures(d, path=""):
    if isinstance(d, dict):
        if "match" in d:
            return [] if d["match"] else [path]
        return [x for k, v in d.items() for x in failures(v, f"{path}/{k}" if path else k)]
    return []


def sha_manifest(backup: Path) -> dict:
    out = {}
    for k in KERNELS:
        files = sorted(p for p in (backup / k).rglob("*")       # .cache/: upload tools' own state
                       if p.is_file() and ".cache" not in p.relative_to(backup / k).parts)
        out[k] = {str(p.relative_to(backup / k)): {"bytes": p.stat().st_size, "sha256": sha(p)} for p in files}
    return out


def git_blob_sha1(path: Path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def hf_check(repo: str, backup: Path, want: dict, tree=None) -> dict:
    """Check an off-site copy in a Hugging Face dataset repo against the committed sha256 file.

    The Hub reports a sha256 for LFS files, and only a git blob id (sha1 of "blob <size>\\0" +
    content) for small files. LFS files are compared with the committed sha256 directly. A
    small file is compared through the local copy: its sha256 must be the committed one (checked
    here) and its git blob id must be the remote one. Nothing is downloaded."""
    if tree is None:
        from huggingface_hub import HfApi
        tree = [f for f in HfApi().list_repo_tree(repo, repo_type="dataset", recursive=True, expand=True)
                if hasattr(f, "size")]                                   # files, not folders
    remote = {f.path: f for f in tree if f.path != ".gitattributes"}
    out = {}
    for k in KERNELS:
        w = want.get(k, {})
        missing, changed, n_lfs = [], [], 0
        for rel, rec in w.items():
            f = remote.pop(f"{k}/{rel}", None)
            if f is None:
                missing.append(rel)
                continue
            lfs = getattr(f, "lfs", None)
            if f.size != rec["bytes"]:
                changed.append({"file": rel, "problem": "size", "remote": f.size, "want": rec["bytes"]})
            elif lfs is not None:
                n_lfs += 1
                if lfs.sha256 != rec["sha256"]:
                    changed.append({"file": rel, "problem": "lfs sha256"})
            else:
                local = backup / k / rel
                if not local.exists() or sha(local) != rec["sha256"] or git_blob_sha1(local) != f.blob_id:
                    changed.append({"file": rel, "problem": "git blob id"})
        out[k] = {"files": len(w), "lfs_files": n_lfs, "missing": missing[:20], "n_missing": len(missing),
                  "changed": changed[:20], "n_changed": len(changed), "match": not missing and not changed}
    out["unexpected"] = {"files": sorted(remote)[:20], "n": len(remote), "match": not remote}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("backup", help="folder holding one sub-folder per notebook output")
    ap.add_argument("--out")
    ap.add_argument("--sha-out", help="write the sha256 of every downloaded file here")
    ap.add_argument("--against", help="a committed --sha-out file: check this copy file by file against it")
    ap.add_argument("--hf-repo", help="with --against: also check this Hugging Face dataset repo (the off-site copy)")
    a = ap.parse_args(argv)
    if a.hf_repo and not a.against:
        ap.error("--hf-repo needs --against")
    backup = Path(a.backup)
    rep = {"backup": str(backup), "kernels": {}}
    for k in KERNELS:
        root = backup / k / "btp"
        rep["kernels"][k] = kernel_checks(k, root) if root.is_dir() else {"present": {"match": False}}
    if a.sha_out or a.against:
        man = sha_manifest(backup)
        if a.against:
            want = json.loads(Path(a.against).read_text())["files"]
            rep["against"] = {}
            for k in KERNELS:
                m, w = man.get(k, {}), want.get(k, {})
                rep["against"][k] = {"match": m == w,
                                     "missing": sorted(set(w) - set(m))[:20],
                                     "unexpected": sorted(set(m) - set(w))[:20],
                                     "changed": sorted(f for f in set(m) & set(w) if m[f] != w[f])[:20]}
            if a.hf_repo:
                rep["hf"] = {"repo": a.hf_repo, **hf_check(a.hf_repo, backup, want)}
        if a.sha_out:
            Path(a.sha_out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.sha_out).write_text(json.dumps(
                {"note": "sha256 of every file in the Kaggle notebook outputs, from the local download "
                         "checked by scripts/verify_kaggle_backup.py",
                 "totals": {k: {"files": len(v), "bytes": sum(x["bytes"] for x in v.values())}
                            for k, v in man.items()},
                 "files": man}, indent=1))
    rep["failures"] = (failures(rep["kernels"]) + failures(rep.get("against", {}))
                       + [f"hf/{x}" for x in failures({k: v for k, v in rep.get("hf", {}).items() if k != "repo"})])
    rep["all_match"] = not rep["failures"]
    txt = json.dumps(rep, indent=2)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(txt)
    print(json.dumps({"all_match": rep["all_match"], "failures": rep["failures"]}, indent=1))
    return 0 if rep["all_match"] else 1


if __name__ == "__main__":
    sys.exit(main())
