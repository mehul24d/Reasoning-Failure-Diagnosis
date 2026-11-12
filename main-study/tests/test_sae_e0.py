"""SAE validation and E0 logic with fake SAEs on the tiny model."""
import numpy as np
import pandas as pd
import pytest
import torch

from analysis import sae_validation as sv
from analysis import sparsity_e0 as e0
from harness import generate
from harness.prompts import PromptSpec, encode_chat

from .tiny_model import build, build_tl


class IdentitySAE(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
    def encode(self, x):           # non-negative "features": split into positive / negative parts
        return torch.cat([x.clamp_min(0), (-x).clamp_min(0)], -1)
    def decode(self, f):
        d = f.shape[-1] // 2
        return f[..., :d] - f[..., d:]


class ZeroSAE(IdentitySAE):
    def decode(self, f):
        return torch.zeros(*f.shape[:-1], f.shape[-1] // 2)


class MeanSAE(IdentitySAE):
    """Reconstructs every token as a fixed vector -> FVU >= 1 by construction."""
    def __init__(self, mean):
        super().__init__(len(mean)); self.mean = mean
    def decode(self, f):
        return self.mean.expand(*f.shape[:-1], -1)


@pytest.fixture(scope="module")
def setup():
    hf, tok = build()
    tl = build_tl(hf, tok)
    g = generate.greedy_generation_config(tok, 12)
    recs = []
    for i, q in enumerate(["Tom has 3 apples", "Liam buys 4 boxes of pencils"]):
        ids = encode_chat(tok, q, PromptSpec("marked", 0))
        r = generate.generate_one(hf, tok, ids, g)
        r.update(problem_id=f"p{i}", template_id=f"T{i}", correct=bool(i % 2))
        recs.append(r)
    return tl, tok, recs


def test_identity_sae_is_perfect(setup):
    tl, tok, recs = setup
    r = sv.validate_layer(tl, IdentitySAE(64), recs, layer=1, bos_id=tok.bos_token_id)
    assert r["fvu"] == pytest.approx(0.0, abs=1e-10)
    assert abs(r["delta_ce"]) < 1e-5
    assert r["n_tokens"] == sum(len(x["output_ids"]) for x in recs)
    assert sv.judge(r, 0.4, 0.2)["pass"]


def test_bad_saes_fail(setup):
    tl, tok, recs = setup
    z = sv.validate_layer(tl, ZeroSAE(64), recs, layer=1, bos_id=tok.bos_token_id)
    assert z["fvu"] > 1.0                      # ||h||^2 / ||h - mean||^2 >= 1
    m = sv.validate_layer(tl, MeanSAE(torch.ones(64)), recs, layer=1, bos_id=tok.bos_token_id)
    assert m["fvu"] >= 1.0
    assert not sv.judge(z, 0.4, 0.2)["pass_fvu"]


def test_sae_ids_prefer_it_layers():
    from harness.config import model_config

    cfg = model_config("gemma-2-9b-it")
    assert sv.sae_ids_for(cfg, 20) == ("gemma-scope-9b-it-res-canonical", "layer_20/width_131k/canonical")
    assert sv.sae_ids_for(cfg, 21) == ("gemma-scope-9b-pt-res-canonical", "layer_21/width_131k/canonical")
    cfg2 = model_config("gemma-2-2b-it")
    assert sv.sae_ids_for(cfg2, 5) == ("gemma-scope-2b-pt-res-canonical", "layer_5/width_65k/canonical")


def test_l0_and_entropy():
    a = torch.tensor([[0.0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]])
    l0, h = e0.l0_and_entropy(a)
    assert l0.tolist() == [0, 1, 2, 4]
    assert h.tolist() == pytest.approx([0, 0, np.log(2), np.log(4)])


def test_depth_bands_and_layer_pick():
    assert [e0.depth_band(l, 26) for l in (0, 8, 9, 17, 18, 25)] == \
        ["early", "early", "middle", "middle", "late", "late"]
    picked = e0.pick_layers(list(range(26)), 26, 3)
    assert len(picked) == 9 and picked[0] == 0 and picked[-1] == 25


def test_collect_runs(setup):
    tl, tok, recs = setup
    df = e0.collect(tl, lambda l: IdentitySAE(64), [0, 2], recs)
    assert set(df["layer"]) == {0, 2}
    assert len(df) == 2 * sum(len(r["output_ids"]) for r in recs)
    assert (df["recon_err"] < 1e-10).all() and df["correct"].nunique() == 2


def _synthetic(effect: float, confound: bool, n_templates=60, seed=0):
    """Tokens where L0 depends on token identity (and on failure by `effect`).
    With confound=True failed traces use mostly 'hard' tokens, which have high L0 by themselves.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_templates):
        correct = t % 2 == 0
        for _ in range(60):
            p_hard = 0.8 if (confound and not correct) else 0.3
            tok = int(rng.random() < p_hard)
            l0 = 50 + 30 * tok + (0 if correct else effect) + rng.normal(0, 5)
            rows.append(dict(template_id=f"T{t}", problem_id=f"T{t}", prompt="m", correct=correct,
                             layer=3, token_id=tok, rel_pos=rng.random(), resid_norm=rng.random(),
                             recon_err=rng.random(), l0=l0, entropy=l0 / 10))
    return pd.DataFrame(rows)


def test_matching_removes_token_confound():
    df = _synthetic(effect=0.0, confound=True)
    raw = df[~df.correct].l0.mean() - df[df.correct].l0.mean()
    assert raw > 10                                     # naive comparison is badly confounded
    res = e0.analyse(df, n_layers=26, n_resamples=300)
    est = res["early"]["delta_L0"]
    assert abs(est["estimate"]) < 2 and est["ci_low"] < 0 < est["ci_high"]


def test_matching_recovers_true_effect():
    res = e0.analyse(_synthetic(effect=6.0, confound=True), n_layers=26, n_resamples=300)
    est = res["early"]["delta_L0"]
    assert est["estimate"] == pytest.approx(6.0, abs=1.5) and est["ci_low"] > 0
    # d uses the pooled SD of the matched samples, which includes between-token variation:
    # SD ~ sqrt(5^2 + 30^2 * 0.8 * 0.2) ~ 13.0, so d ~ 6 / 13.0 ~ 0.46
    assert res["early"]["d_L0"]["estimate"] == pytest.approx(0.46, abs=0.08)
    assert "p_holm" in est and res["middle"]["n_tokens"] == 0


def test_drop_sae_download(tmp_path, monkeypatch):
    from huggingface_hub import constants
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    repo = tmp_path / "models--google--gemma-scope-9b-pt-res"
    other = tmp_path / "models--google--gemma-2-9b-it"
    (repo / "blobs").mkdir(parents=True); other.mkdir()
    monkeypatch.delenv("BTP_DROP_SAE_CACHE", raising=False)
    sv.drop_sae_download("gemma-scope-9b-pt-res-canonical")
    assert repo.exists()                       # off by default
    monkeypatch.setenv("BTP_DROP_SAE_CACHE", "1")
    sv.drop_sae_download("gemma-scope-9b-pt-res-canonical")
    assert not repo.exists() and other.exists()   # only the SAE repo, never the model


def test_sae_validation_resumes_finished_layers(tmp_path):
    """Layers in the .partial.jsonl are reused only for the same trace file and count."""
    import json as _json
    out = str(tmp_path / "v.json")
    row = {"layer": 3, "fvu": 0.1, "delta_ce": 0.01, "pass": True}
    with sv._partial_path(out).open("w") as f:
        f.write(_json.dumps({"traces": "t.jsonl", "n_traces": 200, "row": row}) + "\n")
        f.write(_json.dumps({"traces": "other.jsonl", "n_traces": 200,
                             "row": {**row, "layer": 4}}) + "\n")
    assert sv.finished_layers(out, "t.jsonl", 200) == {3: row}
    assert sv.finished_layers(out, "t.jsonl", 100) == {}
    assert sv.finished_layers(str(tmp_path / "none.json"), "t.jsonl", 200) == {}
