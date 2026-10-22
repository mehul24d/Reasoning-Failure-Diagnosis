"""Install requirements.txt into a preinstalled GPU image (Kaggle, Colab).

Rules (DECISIONS D17):
  - torch / torchvision / torchaudio / triton / nvidia-* are frozen at the
    image's versions in a constraints file (`pip list --format=freeze`), so the
    image's CUDA torch is never replaced: pip fails instead.
  - Wheels only (`--only-binary=:all:`): nothing is ever compiled. The one
    exception is a pure-Python, sdist-only dependency of transformer-lens.
  - Packages that determine results (CORE) are installed at their pinned
    versions. The other pinned packages (numpy, scipy, pandas, matplotlib, ...)
    keep the image's preinstalled version when there is one, because a pin
    without a wheel for the image's Python (e.g. scipy 1.13.1 on Python 3.13)
    would otherwise be compiled, and that fails. pip still upgrades such a
    package (wheel only) if a CORE pin requires it.
  - Every installed version is written to a JSON report.

Usage: python scripts/install_gpu_env.py [--requirements requirements.txt]
                                         [--report install_report.json] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path

TORCH_FAMILY = re.compile(r"^(torch|torchvision|torchaudio|triton|nvidia-[a-z0-9-]+)$")
# results-relevant libraries: always exactly the pinned version
CORE = {
    "transformers",
    "transformer-lens",
    "datasets",
    "huggingface-hub",
    "tokenizers",
    "safetensors",
    "accelerate",
    "scikit-learn",
}
SOURCE_OK = ["transformers-stream-generator"]  # sdist only; pure Python


def canon(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_requirements(path: str | Path) -> list[tuple[str, str]]:
    """(canonical name, requirement line) for every requirement in the file."""
    out = []
    for raw in Path(path).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)", line)
        if m:
            out.append((canon(m.group(1)), line))
    return out


def installed_versions() -> dict[str, str]:
    return {canon(d.metadata["Name"]): d.version for d in metadata.distributions()}


def plan(
    reqs: list[tuple[str, str]], installed: dict[str, str]
) -> tuple[list[str], list[str], dict[str, dict[str, str]]]:
    """-> (requirement lines to pass to pip, constraint lines, kept packages)."""
    constraints = [f"{n}=={v}" for n, v in sorted(installed.items()) if TORCH_FAMILY.match(n)]
    lines, kept = [], {}
    for name, line in reqs:
        if TORCH_FAMILY.match(name):
            continue  # never installed: the image's build is kept
        if name not in CORE and name in installed:
            lines.append(name)  # satisfied by the image unless a CORE pin needs more
            kept[name] = {"preinstalled": installed[name], "pinned": line}
        else:
            lines.append(line)
    return lines, constraints, kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requirements", default="requirements.txt")
    ap.add_argument("--report", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    reqs = parse_requirements(args.requirements)
    before = installed_versions()
    lines, constraints, kept = plan(reqs, before)
    Path("/tmp/rfd_constraints.txt").write_text("\n".join(constraints) + "\n")
    Path("/tmp/rfd_requirements.txt").write_text("\n".join(lines) + "\n")
    print("python", platform.python_version())
    print("frozen (constraints):", ", ".join(constraints) or "none")
    for n, k in kept.items():
        print(f"kept preinstalled {n}=={k['preinstalled']} (pin: {k['pinned']})")
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--progress-bar",
        "off",
        "--only-binary=:all:",
        *[f"--no-binary={p}" for p in SOURCE_OK],
        "-r",
        "/tmp/rfd_requirements.txt",
        "-c",
        "/tmp/rfd_constraints.txt",
    ]
    if args.dry_run:
        cmd.append("--dry-run")
    print("$", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd)
    after = installed_versions()
    report = {
        "python": platform.python_version(),
        "returncode": rc,
        "frozen": constraints,
        "kept_preinstalled": kept,
        "versions": {n: after.get(n) for n, _ in reqs} | {"torch": after.get("torch")},
        "changed": {
            n: [before.get(n), after.get(n)] for n in after if before.get(n) != after.get(n)
        },
    }
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2))
    torch_changed = before.get("torch") != after.get("torch")
    if rc or torch_changed:
        print("INSTALL FAILED" + (" (torch changed!)" if torch_changed else ""), file=sys.stderr)
        sys.exit(rc or 1)
    print("install OK")


if __name__ == "__main__":
    main()
