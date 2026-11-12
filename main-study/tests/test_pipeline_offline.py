"""The minor-project pipeline (prereg/S2.md amendments A2-A6) on a tiny random model, offline."""
import copy
import json

import numpy as np
import pytest
import torch

from cache import verify_cache
from cache.positions import answer_char, char_to_token, token_char_ends, trace_positions
from harness import batched, generate, pipeline
from harness.config import experiment_config
from harness.prompts import PromptSpec, encode_chat

from .tiny_model import build, build_tl


@pytest.fixture(scope="module")
def tiny():
    hf, tok = build()
    return hf, tok, build_tl(hf, tok)


class TinyLoader:
    def __init__(self, tiny):
        self.hf_, self.tok, self.tl_ = tiny

    def tokenizer(self):
        return self.tok

    def hf(self, lock_dir):
        return self.hf_

    def tl(self, lock_dir, n_devices=1):
        return self.tl_


@pytest.fixture
def small_cfg(monkeypatch):
    """max_new_tokens 24 (the tiny model's context is 512) and a 2-row shard size."""
    cfg = copy.deepcopy(experiment_config())
    cfg["generation"]["greedy"]["max_new_tokens"] = 24
    cfg["generation"]["batched"]["batch_size"] = 3
    cfg["cache"]["rows_per_shard"] = 5
    cfg["cache"]["fp16_check_traces"] = 2
    cfg["verification"]["per_template"] = 2
    cfg["verification"]["min_sample"] = 3
    for mod in (pipeline, verify_cache):
        monkeypatch.setattr(mod, "experiment_config", lambda c=cfg: c)
    return cfg


# ----------------------------------------------------------------------------- stop rule
def test_answer_line_end():
    assert batched.answer_line_end("Step 1: 2 + 2 = 4\nAnswer: 4\nmore") == len("Step 1: 2 + 2 = 4\nAnswer: 4\n")
    assert batched.answer_line_end("Answer: 4") is None                 # line not finished yet
    assert batched.answer_line_end("Answer:\n4\n") is None               # no number on the line
    assert batched.answer_line_end("**Final Answer:** $12\n") is not None
    assert batched.answer_line_end("The answer is 4.\n") is None          # not at a line start
    assert batched.answer_line_end("Step 1: answer 3 + 4 = 7\n") is None  # 'answer' inside a step


def test_trim_prefers_the_earliest_stop():
    eos = {9}
    assert batched._trim([1, 2, 9, 0, 0], eos, None, 5) == ([1, 2, 9], "eos")
    assert batched._trim([1, 2, 3, 0, 0], eos, 1, 5) == ([1, 2], "answer_line")
    assert batched._trim([1, 9, 3, 0, 0], eos, 3, 5) == ([1, 9], "eos")
    assert batched._trim([1, 2, 3, 4, 5], eos, None, 5) == ([1, 2, 3, 4, 5], "max_new_tokens")


def test_answer_line_stopping_criterion(tiny):
    _, tok, _ = tiny
    prompt = tok("Problem 3", add_special_tokens=False)["input_ids"]
    gen = tok.convert_tokens_to_ids(["Answer", "7", "\n", "Tom"])   # "\n" is a vocabulary word
    stop = batched.AnswerLineStop(tok, len(prompt), 1)
    flags = [bool(stop(torch.tensor([prompt + gen[: k + 1]]), None)[0]) for k in range(len(gen))]
    # WordLevel decodes with spaces: "Answer 7 \n" ends the answer line at the newline token
    assert flags == [False, False, True, True]
    assert stop.stop_step[0] == 2


# ----------------------------------------------------------------------------- batched generation
def test_batched_greedy_equals_batch_one(tiny):
    hf, tok, _ = tiny
    qs = ["Tom has 3 apples", "Liam buys 4 boxes of pencils . What is 4 + 5 ?", "What is 2 ?"]
    ids = [encode_chat(tok, q, PromptSpec("marked", 0)) for q in qs]
    assert len({len(i) for i in ids}) > 1          # left padding is really exercised
    g = generate.greedy_generation_config(tok, 20)
    single = [generate.generate_one(hf, tok, i, g)["output_ids"] for i in ids]
    res = batched.generate_batch(hf, tok, ids, g, record_top2=True)
    for s, r in zip(single, res):
        n = len(r.output_ids)
        assert r.output_ids == s[:n]
        assert r.top2["id1"] == r.output_ids          # greedy: the recorded top-1 is the token
        assert all(a >= b for a, b in zip(r.top2["l1"], r.top2["l2"]))


def test_tl_fallback_engine_matches_hf(tiny):
    hf, tok, tl = tiny
    ids = encode_chat(tok, "Tom has 3 apples", PromptSpec("terse", 0))
    g = generate.greedy_generation_config(tok, 16)
    ref = generate.generate_one(hf, tok, ids, g)["output_ids"]
    r = batched.tl_generate_one(tl, tok, ids, 16, batched.eos_ids(tok))
    assert r.output_ids == ref[: len(r.output_ids)]


# ----------------------------------------------------------------------------- positions
def test_char_to_token_and_answer_char(tiny):
    _, tok, _ = tiny
    ids = tok.convert_tokens_to_ids("Step 1 : 3 + 4 = 7 \n Answer 7".split(" "))
    ends = token_char_ends(tok, ids)
    text = tok.decode(ids)
    assert ends[-1] == len(text)
    for k in range(len(ids)):
        start = 0 if k == 0 else ends[k - 1]
        assert char_to_token(ends, max(start, ends[k] - 1)) == k
    assert text[answer_char(text)] == "7"
    assert answer_char("no marker here 5") is None
    assert answer_char("Answer: -$12.\n") is not None and "Answer: -$12.\n"[answer_char("Answer: -$12.\n")] == "-"


def test_trace_positions_marked_steps(tiny):
    _, tok, _ = tiny
    out = tok.convert_tokens_to_ids("Step 1 : 3 + 4 = 7 \n Step 2 : 7 + 2 = 9 \n Answer 9".split(" "))
    rec = {"trace_id": "x", "prompt": "marked_p0", "prompt_ids": [1, 2, 3], "output_ids": out,
           "text": tok.decode(out, skip_special_tokens=True)}
    p = trace_positions(tok, rec)
    kinds = [(q["kind"], q["step"]) for q in p["positions"]]
    assert kinds == [("step_first", 0), ("step_last", 0), ("step_first", 1), ("step_last", 1),
                     ("final_answer", None)]
    ids = rec["prompt_ids"] + out
    word = lambda q: tok.convert_ids_to_tokens(ids[q["tok"]])  # noqa: E731
    assert [word(q) for q in p["positions"]] == ["Step", "7", "Step", "9", "9"]
    assert p["positions"][-1]["tok"] == len(ids) - 1


# ----------------------------------------------------------------------------- TL pass
def test_tl_pass_matches_full_forward_and_softcap(tiny):
    hf, tok, tl = tiny
    ids = encode_chat(tok, "Tom has 3 apples", PromptSpec("marked", 0))
    g = generate.greedy_generation_config(tok, 12)
    r = batched.generate_batch(hf, tok, [ids], g, record_top2=True)[0]
    full = ids + r.output_ids
    rows = [0, len(ids) - 1, len(full) - 1]
    resid, rec = verify_cache.tl_pass(tl, full, len(ids), rows, r.top2, chunk=5)
    assert rec["n_match"] == rec["n_tokens"] == len(r.output_ids)
    assert rec["max_abs_logit_diff"] < 1e-4 and rec["hf_top1_is_output"] == rec["n_tokens"]
    _, cache = tl.run_with_cache(torch.tensor([full]))
    names = verify_cache.hook_names(tl.cfg.n_layers)
    for j, h in enumerate(names):
        assert torch.allclose(torch.from_numpy(resid[:, j]), cache[h][0, rows], atol=1e-6)
    # with a soft-cap, the chunked logits equal TL's own forward
    tl.cfg.output_logits_soft_cap = 0.05
    try:
        ref = tl(torch.tensor([full]))[0, len(ids) - 1: -1].argmax(-1).tolist()
        _, rec2 = verify_cache.tl_pass(tl, full, len(ids), [], None, chunk=4)
        assert rec2["n_tokens"] - rec2["n_match"] == sum(a != b for a, b in zip(ref, r.output_ids))
    finally:
        tl.cfg.output_logits_soft_cap = 0.0


# ----------------------------------------------------------------------------- whole pipeline
def _setup_run(tmp_path, tok):
    run = pipeline.Run("qwen2.5-0.5b-instruct", tmp_path / "gen", tmp_path / "store",
                       tmp_path / "reports", "smoke", tmp_path / "problems.jsonl", tmp_path / "split.json")
    probs = []
    for t in range(3):
        for i in range(2):
            probs.append({"problem_id": f"p{t}-{i}", "template_id": f"T{t}", "variant": "base",
                          "source": "gsm8k", "question": f"Tom has {t + i + 2} apples",
                          "answer": str(t + i + 2), "solution": f"He has {t + i + 2}.\n#### {t + i + 2}"})
    pipeline.write_jsonl(probs, run.problems)
    run.split.write_text(json.dumps({"train": ["T0"], "dev": ["T1"], "test": ["T2"]}))
    return run


def test_pipeline_end_to_end_two_workers_and_resume(tiny, tmp_path, small_cfg, monkeypatch):
    hf, tok, tl = tiny
    loader = TinyLoader(tiny)
    run = _setup_run(tmp_path, tok)
    log = pipeline.run_stages(run, ["generate", "score"], loader=loader, n_workers=2)
    assert [x["status"] for x in log] == ["complete", "complete"]
    traces = pipeline.read_jsonl(run.traces)
    assert len(traces) == 6 * 6 and len({t["trace_id"] for t in traces}) == 36
    assert {t["worker"] for t in traces} == {0, 1}
    assert all(t["stop_reason"] in ("eos", "answer_line", "max_new_tokens") for t in traces)
    # batched traces equal batch-1 greedy on the same prompt
    g = generate.greedy_generation_config(tok, 24)
    t0 = traces[0]
    assert generate.generate_one(hf, tok, t0["prompt_ids"], g)["output_ids"][: t0["n_new_tokens"]] == t0["output_ids"]
    # all wrong on a random model; give a few synthetic positions so the cache has rows
    pos = pipeline.read_jsonl(run.positions)
    for p in pos:
        n = p["n_prompt"] + p["n_output"]
        p["positions"] = [{"kind": "step_first", "step": 0, "char": 0, "tok": p["n_prompt"], "token_id": 0},
                          {"kind": "step_last", "step": 0, "char": 1, "tok": n - 1, "token_id": 0},
                          {"kind": "final_answer", "step": None, "char": 2, "tok": n - 1, "token_id": 0}]
    pipeline.write_jsonl(pos, run.positions)

    log = pipeline.run_stages(run, ["verify", "cache"], loader=loader, n_workers=2)
    assert [x["status"] for x in log] == ["complete", "complete"], log
    gate = json.loads((run.reports / "smoke_equivalence_batched_qwen2.5-0.5b-instruct.json").read_text())
    assert gate["pass"] and gate["reproduction"] == 1.0 and gate["templates"] == 3
    assert gate["sample_size"] == 6 and gate["max_abs_logit_diff"] < 1e-4
    assert gate["fp16_check"]["traces"] == 2 and gate["fp16_check"]["max_rel_err"] < 1e-3
    man = json.loads((run.store / "manifest.json").read_text())
    assert man["n_rows"] == 36 * 2 and man["n_index_entries"] == 36 * 3 and not man["orphan_shards"]
    assert len(man["hooks"]) == tl.cfg.n_layers + 1
    # every index entry reads back the residual of its token, within float16 rounding
    idx = [json.loads(line) for line in (run.store / "index.jsonl").read_text().splitlines()]
    e = idx[4]
    arr = np.load(run.store / e["shard"], mmap_mode="r")[e["row"]]
    tr = {t["trace_id"]: t for t in traces}[e["trace_id"]]
    _, cache = tl.run_with_cache(torch.tensor([tr["prompt_ids"] + tr["output_ids"]]))
    ref = cache["blocks.1.hook_resid_post"][0, e["tok"]].numpy()
    assert np.allclose(arr[2].astype(np.float32), ref, rtol=2e-3, atol=2e-3)
    from cache import store

    assert np.array_equal(store.residuals(run.store, [e], "blocks.1.hook_resid_post")[0],
                          arr[2].astype(np.float32))
    vrecs = [r for f in sorted((run.base / "parts").glob("verify.part*.jsonl"))
             for r in pipeline.read_jsonl(f)]
    W = np.random.default_rng(0).standard_normal((tl.cfg.d_model, 8)).astype(np.float32)
    chk = store.fp16_sae_check(run.store, vrecs, "blocks.1.hook_resid_post",
                               lambda x: np.maximum(x @ W, 0))
    assert chk["traces"] == 2 and chk["max_abs_feature_diff"] < 1e-2

    # sampled: test-split failures only (template T2: 2 problems x 6 prompts), k = 5
    log = pipeline.run_stages(run, ["sampled", "report"], loader=loader, n_workers=2)
    assert [x["status"] for x in log] == ["complete", "complete"]
    s = pipeline.read_jsonl(run.sampled)
    assert len(s) == 2 * 6 * 5 and {r["template_id"] for r in s} == {"T2"}
    assert {r["mode"] for r in s} == {"sampled"} and all(r["seed"] is not None for r in s)

    # every stage is now skipped
    log = pipeline.run_stages(run, pipeline.STAGES[1:], loader=loader, n_workers=2)
    assert all(x["status"] == "complete (skipped)" for x in log)


def test_partial_stage_resumes_without_regenerating(tiny, tmp_path, small_cfg, monkeypatch):
    loader = TinyLoader(tiny)
    run = _setup_run(tmp_path, tiny[1])
    calls = {"n": 0}
    real = batched.generate_batch

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt   # the session dies mid-stage
        return real(*a, **k)

    monkeypatch.setattr(batched, "generate_batch", flaky)
    with pytest.raises(KeyboardInterrupt):
        pipeline.run_stages(run, ["generate"], loader=loader, n_workers=1)
    first = pipeline.read_jsonl(run.part("greedy", 0))
    assert len(first) == 2 * 3                    # two finished batches of 3 were kept
    monkeypatch.setattr(batched, "generate_batch", real)
    log = pipeline.run_stages(run, ["generate"], loader=loader, n_workers=1)
    assert log[0]["status"] == "complete"
    allr = pipeline.read_jsonl(run.traces)
    assert len(allr) == 36
    kept = {r["trace_id"]: r for r in first}
    assert all(kept[r["trace_id"]]["batch_keys_crc"] == r["batch_keys_crc"] for r in allr if r["trace_id"] in kept)


def test_deadline_makes_a_stage_partial(tiny, tmp_path, small_cfg, monkeypatch):
    import time

    run = _setup_run(tmp_path, tiny[1])
    ticks = {"n": 0}

    def near(self):           # the deadline arrives after the first batch
        ticks["n"] += 1
        return ticks["n"] > 1

    monkeypatch.setattr(pipeline._Deadline, "near", near)
    log = pipeline.run_stages(run, ["generate", "score"], loader=TinyLoader(tiny), n_workers=1,
                              deadline=time.time() + 3600)
    assert [x["status"] for x in log] == ["partial"]          # score is not attempted
    assert len(pipeline.read_jsonl(run.part("greedy", 0))) == 3
    assert not run.traces.exists() or not run.traces.read_text()
    monkeypatch.undo()
    log = pipeline.run_stages(run, ["score"], loader=TinyLoader(tiny), deadline=time.time() + 10)
    assert log[0]["status"] == "not started (deadline)"


def test_study_stages_refuse_while_s2_is_a_draft(tmp_path):
    run = pipeline.Run.study("gemma-2-2b-it", root=tmp_path)
    if pipeline.s2_is_binding():
        pytest.skip("S2 is binding")
    with pytest.raises(RuntimeError, match="DRAFT"):
        pipeline.run_stages(run, ["generate"])


def test_design_effect_rule():
    rows = []
    for t in range(10):     # train/dev: failure rate differs by template -> rho > 0
        for i in range(10):
            rows.append({"split": "train", "template_id": f"a{t}", "correct": i >= t, "s0": 0})
    for t in range(4):
        for i in range(10):
            rows.append({"split": "test", "template_id": f"b{t}", "correct": False, "s0": 0})
    d = pipeline.design_effect(rows)
    assert 0 < d["rho_failure_icc_train_dev"] < 1
    assert d["deff"] == pytest.approx(1 + 9 * d["rho_failure_icc_train_dev"])
    assert d["required_test_failures_with_s0"] >= 320 and d["top_up_needed"]


def test_estimate_projection_and_budget_flag():
    from harness import estimate

    meas = {
        "model": "gemma-2-2b-it",
        "greedy": {"tokens_per_gpu_second": 200.0, "mean_new_tokens": 150.0, "traces": 384},
        "greedy_summary": {"hit_max_tokens_fraction": 0.0, "stop_reasons": {"eos": 384}},
        "sampled": {"tokens_per_gpu_second": 250.0, "mean_new_tokens": 160.0},
        "tl_fallback": {"tokens_per_gpu_second": 20.0},
        "verify": {"seconds_per_seq_token": 0.001, "mean_seq_tokens": 300.0, "seconds_per_trace": 0.3,
                   "reproduction": 1.0, "max_abs_logit_diff": 1e-4, "fp16_check": {"max_rel_err": 3e-4}},
        "rows_per_trace": 11.0, "bytes_per_greedy_trace": 8000.0, "bytes_per_sampled_trace": 4000.0,
        "load_seconds_per_stage": {"generation": 120.0, "verification_and_caching": 300.0, "sampled_runs": 120.0},
        "session_setup_minutes": 15.0,
    }
    p = estimate.project(meas, 2)
    n = 97 * 8 * 6
    assert p["sizes"]["greedy_traces"] == n == 4656
    assert p["sizes"]["sampled_runs"] == math_ceil(40 * 8 * 6 * 0.29) * 5
    assert p["t4_gpu_hours"]["generation"] == pytest.approx(n * 150 / 200 / 3600 + 2 * 120 / 3600, abs=1e-3)
    assert p["disk_gb"]["activation_store"] == pytest.approx(n * 11 * 27 * 2304 * 2 / 1e9, abs=1e-3)
    assert not p["exceeds_budget"] and p["kaggle_sessions"] == 1
    md = estimate.markdown({"model": "gemma-2-2b-it", "date": "2026-11-13", "gpus": ["Tesla T4"] * 2,
                            "commit": "abc", "batch_size": 16, "measured": meas, "projection": p})
    assert "Within budget" in md
    meas["greedy"]["tokens_per_gpu_second"] = 5.0
    assert estimate.project(meas, 2)["exceeds_budget"]


def math_ceil(x):
    import math

    return math.ceil(x)


def test_failed_gate_and_tl_fallback(tiny, tmp_path, small_cfg, monkeypatch):
    loader = TinyLoader(tiny)
    run = _setup_run(tmp_path, tiny[1])
    pipeline.run_stages(run, ["generate", "score"], loader=loader, n_workers=1)
    small_cfg["equivalence_checks"]["teacher_forced_reproduction_min"] = 1.01   # cannot pass
    log = pipeline.run_stages(run, ["verify", "cache"], loader=loader, n_workers=1)
    assert [x["status"] for x in log] == ["failed"] and "fall back" in log[0]["next"]
    info = pipeline.fallback_to_tl(run)
    assert pipeline.engine_of(run) == "tl" and "smoke_traces.jsonl" in info["moved"]
    assert run.status("generate")["status"] == "pending"
    small_cfg["equivalence_checks"]["teacher_forced_reproduction_min"] = 0.99
    small_cfg["generation"]["greedy"]["max_new_tokens"] = 6
    log = pipeline.run_stages(run, ["generate", "score", "verify"], loader=loader, n_workers=1)
    assert [x["status"] for x in log] == ["complete"] * 3
    assert {r["engine"] for r in pipeline.read_jsonl(run.traces)} == {"tl"}
