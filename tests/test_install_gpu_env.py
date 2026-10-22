"""Install plan for preinstalled GPU images (scripts/install_gpu_env.py)."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "install_gpu_env", Path(__file__).parents[1] / "scripts" / "install_gpu_env.py"
)
ige = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ige)


def test_plan_freezes_torch_pins_core_and_keeps_preinstalled(tmp_path):
    req = tmp_path / "r.txt"
    req.write_text(
        "# comment\ntorch==2.13.0\ntransformers==5.16.1\nscikit-learn==1.9.0\n"
        "scipy==1.13.1  # no cp313 wheel\nPyYAML==6.0.3\npytest==9.0.3\n"
    )
    installed = {
        "torch": "2.14.1+cu124",
        "triton": "3.4.0",
        "nvidia-cublas-cu12": "12.4.5.8",
        "scipy": "1.18.1",
        "pyyaml": "6.0.2",
        "scikit-learn": "1.6.1",
        "numpy": "2.5.4",
    }
    lines, constraints, kept = ige.plan(ige.parse_requirements(req), installed)
    assert constraints == [
        "nvidia-cublas-cu12==12.4.5.8",
        "torch==2.14.1+cu124",
        "triton==3.4.0",
    ]
    # torch is never requested; CORE pins exact even if preinstalled; others kept
    assert lines == [
        "transformers==5.16.1",
        "scikit-learn==1.9.0",
        "scipy",
        "pyyaml",
        "pytest==9.0.3",
    ]
    assert set(kept) == {"scipy", "pyyaml"} and kept["scipy"]["preinstalled"] == "1.18.1"


def test_requirements_split_keeps_legacy_out_of_gpu_installs():
    root = Path(__file__).parents[1]
    core = {n for n, _ in ige.parse_requirements(root / "requirements.txt")}
    assert ige.CORE <= core
    assert not core & {"python-louvain", "sae-lens", "hmmlearn", "ruptures", "networkx"}
    assert "-r requirements.txt" in (root / "requirements-legacy.txt").read_text()
