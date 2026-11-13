"""Provenance of the model weights the study runs on (prereg/S2.md amendment A12, DECISIONS D32).

Hashes every file of a model directory (sha256, and the git blob sha1 that the Hugging Face API
reports for files outside LFS), records the Kaggle Models handle and version taken from the
path, and compares the files with the published Hugging Face hashes
fetched from the HF API with --fetch-hf (on Kaggle: the local network of the project Mac masks
64-hex strings in API answers, so a locally saved reference is unusable).

    python -m harness.weights_provenance --dir /kaggle/input/.../gemma-2-2b-it/2 \
        --model gemma-2-2b-it --out reports/m1/weights_gemma-2-2b-it.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from .config import REPO_ROOT

# files that decide the computation; the rest (README, licence) are recorded but not required
REQUIRED = ("config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer.model",
            "tokenizer_config.json", "special_tokens_map.json")


def file_hashes(path: Path) -> dict:
    sha, size = hashlib.sha256(), path.stat().st_size
    blob = hashlib.sha1(f"blob {size}\0".encode())
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            sha.update(chunk)
            blob.update(chunk)
    return {"size": size, "sha256": sha.hexdigest(), "git_blob_sha1": blob.hexdigest()}


def kaggle_handle(path: str) -> dict:
    """/kaggle/input/<...>/<owner?>/gemma-2/transformers/gemma-2-2b-it/<version>."""
    m = re.search(r"/([^/]+)/(transformers|pytorch|jax|keras|flax)/([^/]+)/(\d+)/?$", path.rstrip("/"))
    if not m:
        return {"path": path}
    return {"path": path, "model": m.group(1), "framework": m.group(2), "variation": m.group(3),
            "version": int(m.group(4)),
            "handle": f"google/{m.group(1)}/{m.group(2)}/{m.group(3)}/{m.group(4)}"}


def compare_with_hf(files: dict, hf: dict) -> dict:
    out, ok = {}, True
    for name, ref in hf["files"].items():
        mine = files.get(name)
        if mine is None:
            out[name] = "absent from this copy"
            if name.endswith(".safetensors") or name in REQUIRED:
                ok = False
            continue
        if ref.get("sha256"):
            same = mine["sha256"] == ref["sha256"]
        else:
            same = mine["git_blob_sha1"] == ref.get("git_oid")
        out[name] = "identical" if same else "DIFFERENT"
        if not same and (name.endswith(".safetensors") or name in REQUIRED):
            ok = False
    weights = [n for n in hf["files"] if n.endswith(".safetensors")]
    return {"per_file": out, "weights_identical_to_hf": all(out.get(n) == "identical" for n in weights),
            "all_required_identical": ok, "hf_revision": hf.get("revision")}


def fetch_hf_reference(repo: str) -> dict:
    """Published file hashes from the Hugging Face API (public metadata, no token): LFS sha256
    for the weights and tokenizer, git blob sha1 for the rest."""
    import urllib.request

    def get(url):
        with urllib.request.urlopen(url, timeout=60) as r:
            return json.load(r)
    info = get(f"https://huggingface.co/api/models/{repo}")
    tree = get(f"https://huggingface.co/api/models/{repo}/tree/{info['sha']}")
    files = {f["path"]: {"size": f["size"], "sha256": (f.get("lfs") or {}).get("oid"),
                         "git_oid": f.get("oid")} for f in tree if f.get("type") == "file"}
    for n, v in files.items():
        if v["sha256"] is not None and not re.fullmatch(r"[0-9a-f]{64}", v["sha256"]):
            raise ValueError(f"malformed sha256 for {n} in the HF API answer")
    return {"repo": repo, "revision": info["sha"], "last_modified": info.get("lastModified"),
            "files": files}


def provenance(model_dir: str, model: str, hf: dict | None = None) -> dict:
    d = Path(model_dir)
    files = {p.name: file_hashes(p) for p in sorted(d.iterdir()) if p.is_file()}
    weights = sorted(n for n in files if n.endswith(".safetensors"))
    combined = hashlib.sha256("".join(files[n]["sha256"] for n in weights).encode()).hexdigest()
    rep = {"model": model, "kaggle": kaggle_handle(str(d)), "files": files,
           "weights_sha256_combined": combined,
           "combined_rule": "sha256 of the concatenated hex sha256 of the *.safetensors files, by name"}
    if hf is None:
        hf_ref = REPO_ROOT / "reports" / "m1" / f"hf_hub_files_{model}.json"
        hf = json.loads(hf_ref.read_text()) if hf_ref.exists() else None
    if hf is not None:
        rep["hf_reference"] = hf
        rep["vs_hugging_face"] = compare_with_hf(files, hf)
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--model", default="gemma-2-2b-it")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fetch-hf", default=None, metavar="REPO",
                    help="fetch the reference hashes from the HF API (e.g. google/gemma-2-2b-it)")
    a = ap.parse_args()
    rep = provenance(a.dir, a.model, fetch_hf_reference(a.fetch_hf) if a.fetch_hf else None)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=2))
    print(json.dumps({"kaggle": rep["kaggle"], "weights_sha256_combined": rep["weights_sha256_combined"],
                      "vs_hugging_face": rep.get("vs_hugging_face")}, indent=1))


if __name__ == "__main__":
    main()
