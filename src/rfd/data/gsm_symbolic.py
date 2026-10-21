"""GSM-Symbolic loading with reconstructed gold programs.

The HF instances (`apple/GSM-Symbolic`) carry no `<<a op b = c>>` calculator
annotations. The gold program of an instance is therefore parsed from its
solution text with the evaluation-validated parser in `rfd.parsing.equations`
(the same parser used on model outputs); an instance is kept only if its last
parsed step equals the `####` gold answer.

As an independent cross-check, the templates in github.com/apple/ml-gsm-symbolic
DO carry annotations (`answer_annotated`): for instances whose question matches
the template regex and whose rendered answer reproduces the HF text exactly,
we compare the annotation-derived steps with the text-parsed ones
(`template_crosscheck`). See docs/DECISIONS.md D2.
"""

from __future__ import annotations

import json
import math
import re
import ssl
import urllib.request
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from rfd.parsing.equations import BinStep, close, decompose, program_from_text

TEMPLATE_URL = (
    "https://raw.githubusercontent.com/apple/ml-gsm-symbolic/main/templates/symbolic/{idx:04d}.json"
)
N_TEMPLATES = 100

_SLOT_RE = re.compile(r"\{(\w+)\s*,\s*([^{}]*?)\s*\}")
_EXPR_RE = re.compile(r"\{([^{}]+)\}")
_ANNOT_RE = re.compile(r"<<([^<>]*)>>")
_GOLD_RE = re.compile(r"####\s*(.+)")
_NUM_RE = re.compile(r"^-?[\d,]*\.?\d+$")


@dataclass
class SymbolicProblem:
    template_id: int
    instance: int
    question: str
    answer_text: str
    gold_answer: float
    variables: dict[str, Any]
    gold_steps: list[BinStep] = field(default_factory=list)
    question_numbers: list[float] = field(default_factory=list)

    @property
    def uid(self) -> str:
        return f"t{self.template_id:03d}-i{self.instance:02d}"


# --------------------------------------------------------------------------- templates


def fetch_templates(cache_dir: str | Path, n: int = N_TEMPLATES) -> list[dict[str, Any]]:
    """Download (once) and load the GSM-Symbolic templates. Network on first call only."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    templates = []
    context = None
    for idx in range(n):
        path = cache_dir / f"{idx:04d}.json"
        if not path.exists():
            if context is None:
                import certifi  # python.org builds ship without system CA certs

                context = ssl.create_default_context(cafile=certifi.where())
            url = TEMPLATE_URL.format(idx=idx)
            with urllib.request.urlopen(url, timeout=30, context=context) as resp:
                path.write_bytes(resp.read())
        templates.append(json.loads(path.read_text()))
    return templates


def template_question_text(question_annotated: str) -> str:
    return question_annotated.split("#init:")[0].strip()


def _value_pattern(default: str) -> str:
    if _NUM_RE.match(default.strip()):
        return r"-?[\d,]*\.?\d+"
    return r".+?"


def build_question_regex(question_annotated: str) -> re.Pattern[str]:
    """Regex with one named group per template variable (repeats -> backrefs)."""
    text = template_question_text(question_annotated)
    parts: list[str] = []
    seen: set[str] = set()
    pos = 0
    for m in _SLOT_RE.finditer(text):
        parts.append(re.escape(text[pos : m.start()]))
        name, default = m.group(1), m.group(2)
        if name in seen:
            parts.append(f"(?P={name})")
        else:
            parts.append(f"(?P<{name}>{_value_pattern(default)})")
            seen.add(name)
        pos = m.end()
    parts.append(re.escape(text[pos:]))
    return re.compile("".join(parts), re.DOTALL)


def _coerce(value: str) -> Any:
    v = value.replace(",", "")
    if _NUM_RE.match(v):
        f = float(v)
        return int(f) if f.is_integer() and "." not in v else f
    return value


_SAFE_FUNCS = {
    "int": int,
    "float": float,
    "round": round,
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "math": math,
    "Fraction": Fraction,
}


def _eval_expr(expr: str, variables: dict[str, Any]) -> Any:
    return eval(expr, {"__builtins__": {}}, {**_SAFE_FUNCS, **variables})  # noqa: S307


def _fmt(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def render_annotated_answer(answer_annotated: str, variables: dict[str, Any]) -> str:
    """Substitute every `{expr}` in the template answer with its value."""
    return _EXPR_RE.sub(lambda m: _fmt(_eval_expr(m.group(1), variables)), answer_annotated)


def strip_annotations(text: str) -> str:
    return _ANNOT_RE.sub("", text)


def _norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------- gold program


def annotated_program(rendered_answer: str) -> list[BinStep]:
    """Binary steps from `<<expr=result>>` annotations of a rendered template answer."""
    steps: list[BinStep] = []
    for annot in _ANNOT_RE.findall(rendered_answer):
        lhs = annot.split("=")[0].strip().replace("%", "*0.01").replace(",", "")
        if lhs and any(op in lhs for op in "+-*/"):
            steps.extend(decompose(lhs))
    return steps


def numbers_in(text: str) -> list[float]:
    return [float(x.replace(",", "")) for x in re.findall(r"(?<![\w.])-?\d[\d,]*\.?\d*", text)]


# --------------------------------------------------------------------------- instances


def parse_instance(row: dict[str, Any]) -> SymbolicProblem | None:
    """Instance with its text-parsed gold program, or None if the program does
    not end at the gold answer."""
    body, _, tail = row["answer"].partition("####")
    try:
        gold = float(tail.strip().replace(",", ""))
    except ValueError:
        return None
    steps = program_from_text(body)
    if not steps or not close(steps[-1].result, gold):
        return None
    return SymbolicProblem(
        template_id=int(row["id"]),
        instance=int(row["instance"]),
        question=row["question"].strip(),
        answer_text=row["answer"],
        gold_answer=gold,
        variables={},
        gold_steps=steps,
        question_numbers=numbers_in(row["question"]),
    )


def template_crosscheck(template: dict[str, Any], row: dict[str, Any]) -> list[BinStep] | None:
    """Annotation-derived program, or None when the template cannot reproduce
    this instance exactly (unpublished generator semantics: fractions, list
    variables, float formatting)."""
    m = build_question_regex(template["question_annotated"]).fullmatch(row["question"].strip())
    if m is None:
        return None
    variables = {k: _coerce(v) for k, v in m.groupdict().items()}
    try:
        rendered = render_annotated_answer(template["answer_annotated"], variables)
        steps = annotated_program(rendered)
    except Exception:  # noqa: BLE001 - any failure means "not reconstructable"
        return None
    if _norm_ws(strip_annotations(rendered)) != _norm_ws(row["answer"]):
        return None
    return steps


def load_gsm_symbolic(
    cache_dir: str | Path, hf_path: str = "apple/GSM-Symbolic", hf_config: str = "main"
) -> tuple[list[SymbolicProblem], dict[str, Any]]:
    """All usable instances plus a parsing / cross-check report."""
    from datasets import load_dataset

    templates = fetch_templates(Path(cache_dir) / "gsm_symbolic_templates")
    rows = load_dataset(hf_path, hf_config)["test"]
    problems, dropped = [], []
    n_checked = n_agree = 0
    for row in rows:
        prob = parse_instance(row)
        if prob is None:
            dropped.append(f"t{int(row['id']):03d}-i{int(row['instance']):02d}")
            continue
        problems.append(prob)
        annotated = template_crosscheck(templates[int(row["id"])], row)
        if annotated:
            n_checked += 1
            # annotations omit some computations (e.g. template 0's last line),
            # so agreement = every annotated result also appears in the text program
            text_results = [s.result for s in prob.gold_steps]
            n_agree += all(any(close(a.result, r) for r in text_results) for a in annotated)
    report = {
        "n_rows": len(rows),
        "n_usable": len(problems),
        "n_dropped": len(dropped),
        "dropped_by_template": {
            t: sum(d.startswith(t) for d in dropped) for t in sorted({d[:4] for d in dropped})
        },
        "n_templates_usable": len({p.template_id for p in problems}),
        "template_crosscheck_n": n_checked,
        "template_crosscheck_agree": n_agree,
    }
    return problems, report


# --------------------------------------------------------------------------- splits


def template_splits(
    template_ids: list[int], fractions: dict[str, float], seed: int
) -> dict[str, list[int]]:
    """Fixed template-grouped split, e.g. {"train": .5, "dev": .2, "test": .3}."""
    ids = sorted(set(template_ids))
    rng = np.random.default_rng(seed)
    perm = [ids[i] for i in rng.permutation(len(ids))]
    out: dict[str, list[int]] = {}
    start = 0
    names = list(fractions)
    for i, name in enumerate(names):
        n = len(perm) - start if i == len(names) - 1 else int(round(fractions[name] * len(perm)))
        out[name] = sorted(perm[start : start + n])
        start += n
    return out
