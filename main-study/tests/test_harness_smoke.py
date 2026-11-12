"""Smoke test of the harness on a tiny model (Qwen2.5-0.5B-Instruct, ~1 GB download).

Skipped unless BTP_SMOKE=1, so the fast tests stay fast:
    BTP_SMOKE=1 pytest tests/test_harness_smoke.py -s
"""
import json
import os

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("BTP_SMOKE") != "1", reason="set BTP_SMOKE=1")


def test_equivalence_checks_pass_on_tiny_model(tmp_path):
    from harness import equivalence

    out = tmp_path / "eq.json"
    report = equivalence.run("qwen2.5-0.5b-instruct", str(out), device="cpu", max_new_tokens=48)
    print(json.dumps(report["checks"], indent=1))
    assert report["all_pass"], report["checks"]


def test_generation_writes_and_resumes(tmp_path):
    from harness import generate
    from harness.prompts import PromptSpec

    probs = tmp_path / "p.jsonl"
    probs.write_text(json.dumps({"problem_id": "t1", "template_id": "T", "variant": "base",
                                 "question": "Tom has 3 apples and buys 4 more. How many now?",
                                 "answer": 7}) + "\n")
    out = tmp_path / "g.jsonl"
    specs = [PromptSpec("marked", 0)]
    generate.run("qwen2.5-0.5b-instruct", str(probs), str(out), specs=specs, device="cpu")
    generate.run("qwen2.5-0.5b-instruct", str(probs), str(out), specs=specs, device="cpu")
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 1  # second run resumed, wrote nothing
    assert rows[0]["output_ids"] and "Answer" in rows[0]["text"]
