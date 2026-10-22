"""Quick smoke test of a GPU machine before the long runs (a few minutes).

Checks that the gated checkpoint loads (HF_TOKEN set; its value is never
printed), that the effective generation config is pure greedy, that chat
prompts carry one BOS and fold the system text into the first user turn when
the template has no system role, that the worst-case batches fit in memory
(longest CoT prompts generating the full token budget; teacher-forced passes
capturing every layer; Baseline A arithmetic batches), that stored activations
fit the configured dtype, and that a zero-strength steering edit changes no
trace. Outputs go to <results>/smoke/ and are never reported.

Usage: python -m rfd.experiments.smoke [--harness CFG]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from rfd.harness.model import load  # sets MPS env before torch use

import torch  # noqa: E402

from rfd.experiments.collect_traces import build_items  # noqa: E402
from rfd.experiments.phase0_equivalence import generation_config_report  # noqa: E402
from rfd.harness.generate import forward, generate  # noqa: E402
from rfd.harness.prompts import COT_SYSTEM, arith_messages, chat_ids, cot_messages  # noqa: E402
from rfd.utils.config import load_run_config  # noqa: E402
from rfd.utils.resume import FP16_MAX, atomic_write_json, results_path  # noqa: E402


def _peak_gb(device: str) -> float | None:
    return torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None


def _reset_peak(device: str) -> None:
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", default="configs/gemma2_2b.yaml")
    ap.add_argument("--traces-config", default="configs/traces.yaml")
    ap.add_argument("--baseline-a-config", default="configs/baseline_a.yaml")
    ap.add_argument("--out", default="results/smoke")
    args = ap.parse_args()
    cfg = load_run_config(args.traces_config, args.harness)
    acfg = load_run_config(args.baseline_a_config, args.harness)
    h = cfg["harness"]
    out = results_path(args.out)
    report: dict = {"harness": args.harness, "model": h["model"]["model_name"], "checks": {}}
    checks = report["checks"]
    report["hf_token_set"] = bool(os.environ.get("HF_TOKEN"))

    t0 = time.time()
    loaded = load(h["model"])
    model, tok, device = loaded.model, loaded.tokenizer, loaded.device
    report.update(
        device=device,
        dtype=str(model.dtype),
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        load_seconds=round(time.time() - t0, 1),
        weights_gb=_peak_gb(device),
    )
    if device == "cuda":
        report["gpu"] = torch.cuda.get_device_name(0)
        report["gpu_total_gb"] = torch.cuda.get_device_properties(0).total_memory / 2**30
    want_attn = h["model"].get("attn_implementation")
    checks["attn_implementation"] = want_attn is None or report["attn_implementation"] == want_attn
    gcr = generation_config_report(model)
    report["generation_config"] = gcr
    checks["generation_config_greedy"] = gcr["greedy_neutral"]

    # ---- prompts
    items, _ = build_items(cfg)
    for it in items:
        it["prompt_ids"] = chat_ids(tok, cot_messages(it["question"]))
    p0 = items[0]["prompt_ids"]
    head = tok.decode(p0[:80])
    report["prompt_head"] = head
    if tok.bos_token_id is not None:
        checks["single_bos"] = p0.count(tok.bos_token_id) == 1
    if not h["model"].get("system_role", True):
        first_user = head.find(COT_SYSTEM[:40])
        checks["system_text_in_first_user_turn"] = first_user >= 0 and "system" not in head.lower()
    lens = sorted(len(it["prompt_ids"]) for it in items)
    report["prompt_tokens"] = {"min": lens[0], "median": lens[len(lens) // 2], "max": lens[-1]}

    # ---- worst-case generation batch: longest prompts, full token budget
    bs, max_new = h["generation"]["batch_size"], h["generation"]["max_new_tokens"]
    longest = sorted(items, key=lambda i: -len(i["prompt_ids"]))[:bs]
    _reset_peak(device)
    t0 = time.time()
    n = max(len(i["prompt_ids"]) for i in longest)
    ids = torch.full((bs, n), tok.pad_token_id, dtype=torch.long)
    mask = torch.zeros((bs, n), dtype=torch.long)
    for k, it in enumerate(longest):
        ids[k, n - len(it["prompt_ids"]) :] = torch.tensor(it["prompt_ids"])
        mask[k, n - len(it["prompt_ids"]) :] = 1
    with torch.no_grad():
        g = model.generate(
            input_ids=ids.to(device),
            attention_mask=mask.to(device),
            max_new_tokens=max_new,
            min_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
    dt = time.time() - t0
    report["gen_worst_case"] = {
        "batch_size": bs,
        "prompt_len": n,
        "new_tokens": max_new,
        "peak_gb": _peak_gb(device),
        "seconds": round(dt, 1),
        "tokens_per_s": round(bs * max_new / dt, 1),
    }
    checks["gen_worst_case_fits"] = True
    full = [g[k, n - len(it["prompt_ids"]) :].tolist() for k, it in enumerate(longest)]

    # ---- worst-case teacher-forced pass: all layers captured + full-vocab argmax
    fbs = h["forward_batch_size"]
    _reset_peak(device)
    t0 = time.time()
    seqs = full[:fbs]
    layers = list(range(model.config.num_hidden_layers))
    pos = [list(range(len(s) - 12, len(s))) for s in seqs]
    res = forward(model, tok, seqs, layers, pos, return_argmax=True)
    absmax = float(res["resid"].abs().max())
    report["forward_worst_case"] = {
        "batch_size": fbs,
        "seq_len": max(len(s) for s in seqs),
        "peak_gb": _peak_gb(device),
        "seconds": round(time.time() - t0, 1),
        "resid_absmax": absmax,
    }
    checks["forward_worst_case_fits"] = True
    if h["acts_dtype"] == "float16":
        checks["acts_fit_float16"] = absmax < FP16_MAX / 4  # margin for other positions

    # ---- Baseline A arithmetic batch (generation + all-layer capture)
    abs_ = acfg["gen_batch_size"]
    _reset_peak(device)
    demos = [(271, 318), (402, 197)]
    pre = [
        chat_ids(tok, arith_messages(10**7 + k, 2 * 10**7 + k, "+", demos), f"<<{10**7 + k}+")
        for k in range(abs_)
    ]
    outs = generate(model, tok, pre, acfg["max_new_tokens"], stop_at_final_answer=False)
    forward(
        model, tok, [p + o.gen_ids for p, o in zip(pre, outs)], layers, [[len(p) - 1] for p in pre]
    )
    report["arith_batch"] = {"batch_size": abs_, "peak_gb": _peak_gb(device)}
    checks["arith_batch_fits"] = True

    # ---- tiny end to end: 4 real traces, then a zero-strength steering edit
    from rfd.generation.answer_extraction import extract_predicted_answer, is_correct
    from rfd.steering.engine import SteerSpec, run_condition

    few = items[:4]
    gens = generate(model, tok, [i["prompt_ids"] for i in few], max_new)
    traces = []
    for it, o in zip(few, gens):
        pred = extract_predicted_answer(o.text)
        traces.append(
            {
                **it,
                "gen_ids": o.gen_ids,
                "text": o.text,
                "pred": pred,
                "correct": is_correct(pred, str(it["gold_answer"])),
                "has_final_marker": "Final Answer:" in o.text,
            }
        )
    report["sample_traces"] = [
        {"uid": t["uid"], "correct": t["correct"], "text": t["text"][:400]} for t in traces
    ]
    checks["final_answer_marker_present"] = any(t["has_final_marker"] for t in traces)
    d = model.config.hidden_size
    zero = SteerSpec(len(layers) // 2, "add", torch.ones(d), 0.0)
    rr = run_condition(model, tok, traces, zero, max_new, batch_size=fbs)
    checks["zero_strength_no_divergence"] = not any(r.diverged for r in rr)

    report["all_pass"] = all(checks.values())
    atomic_write_json(out / "smoke.json", report)
    for k, v in checks.items():
        print(f"{'PASS' if v else 'FAIL'}  {k}")
    for k in ("gen_worst_case", "forward_worst_case", "arith_batch"):
        print(k, report[k])
    print("prompt head:", repr(head[:300]))
    print("wrote", out / "smoke.json")
    if not report["all_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
