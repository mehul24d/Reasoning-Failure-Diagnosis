"""Evaluation-validated equation parsing for free-text math solutions.

Used for BOTH gold solutions (GSM-Symbolic `answer` text) and model-generated
steps, so gold and model programs are extracted by the same procedure.

An "equation" is `<lhs> = <number>`. The lhs is tokenized into numbers,
operators and parentheses (unit words, currency symbols and LaTeX wrappers
are dropped). For gold text, the longest token suffix whose value equals the
stated rhs is taken as the expression (so leading prose numbers such as "In
the second 30 seconds, 4 * 60 = 240" do not leak in). For model text, the
stated rhs may be WRONG (an arithmetic slip), so `parse_equations(...,
validate=False)` instead takes the longest syntactically valid suffix that
starts at a number and keeps the stated result as-is.
"""

from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass, field

OPS = ("+", "-", "*", "/")

_LATEX = [
    (r"\\times", "*"),
    (r"\\cdot", "*"),
    (r"\\div", "/"),
    (r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)"),
    (r"\\left|\\right|\\\(|\\\)|\\\[|\\\]|\\,|\\!|\\%|\\text\{[^{}]*\}|\\mathrm\{[^{}]*\}", " "),
    (r"\\boxed\{([^{}]*)\}", r"\1"),
]
# number: 1,234.5 | .5 | 12 ; a trailing '%' is kept as a marker
_TOKEN_RE = re.compile(
    r"(?P<num>(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?%?|(?<![\w.])\d*\.?\d+%?)"
    r"|(?P<op>[+\-*/×÷x−–])"
    r"|(?P<lp>\()|(?P<rp>\))"
    r"|(?P<eq>=)"
    r"|(?P<word>[A-Za-z_]+)"
)
_PY_OPS = {
    ast.Add: ("+", operator.add),
    ast.Sub: ("-", operator.sub),
    ast.Mult: ("*", operator.mul),
    ast.Div: ("/", operator.truediv),
}


@dataclass
class BinStep:
    op: str
    a: float
    b: float
    result: float  # value as STATED (model) or computed (gold intermediate)
    computed: float  # op(a, b)

    @property
    def arithmetic_ok(self) -> bool:
        return close(self.result, self.computed)


@dataclass
class Equation:
    expr: str  # normalized python expression, e.g. "72*7"
    stated: float
    steps: list[BinStep] = field(default_factory=list)
    char_start: int = 0  # offset of the expression's first number in the input text
    char_end: int = 0  # offset just past the stated result


def close(x: float, y: float, rel: float = 1e-4, abs_: float = 1e-2) -> bool:
    return abs(x - y) <= max(abs_, rel * max(abs(x), abs(y)))


def normalize_text(text: str) -> str:
    for pat, rep in _LATEX:
        text = re.sub(pat, rep, text)
    return text.replace("$", " ").replace("\u2212", "-")


def _num(tok: str) -> tuple[float, bool]:
    pct = tok.endswith("%")
    return float(tok.rstrip("%").replace(",", "")), pct


def _tokenize(text: str) -> list[tuple[str, str, int, int]]:
    out = []
    for m in _TOKEN_RE.finditer(text):
        kind = m.lastgroup
        tok = m.group(kind)
        if kind == "op":
            tok = {"×": "*", "x": "*", "÷": "/", "−": "-", "–": "-"}.get(tok, tok)
        out.append((kind, tok, m.start(), m.end()))
    return out


def _clean_lhs(tokens: list[tuple[str, str, int, int]]) -> list[tuple[str, str, int, int]]:
    """Drop words; an op between a word and a number (`inches/remora * 7`)
    collapses to the LAST op before the next number/paren."""
    toks = [t for t in tokens if t[0] != "word"]
    # "70 (building blocks) + 46" leaves an empty "( )" once words are dropped
    changed = True
    while changed:
        changed = False
        for i in range(len(toks) - 1):
            if toks[i][0] == "lp" and toks[i + 1][0] == "rp":
                del toks[i : i + 2]
                changed = True
                break
    # 'x' used as a word (e.g. "x-ray") is rare; treat x as multiply only between numbers
    out: list[tuple[str, str, int, int]] = []
    for t in toks:
        if t[0] == "op" and out and out[-1][0] == "op":
            out[-1] = t
        else:
            out.append(t)
    while out and out[-1][0] == "op":  # "12 inches/foot =" leaves a dangling '/'
        out.pop()
    return out


def _to_python(tokens: list[tuple[str, str, int, int]], pct_mode: str) -> str:
    parts = []
    for kind, tok, _, _ in tokens:
        if kind == "num":
            v, pct = _num(tok)
            parts.append(repr(v * 0.01) if (pct and pct_mode == "fraction") else repr(v))
        else:
            parts.append(tok)
    return "".join(parts)


def decompose(expr: str) -> list[BinStep]:
    """Binary steps of a python arithmetic expression in evaluation order."""
    tree = ast.parse(expr, mode="eval")
    steps: list[BinStep] = []

    def visit(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -visit(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in _PY_OPS:
            a, b = visit(node.left), visit(node.right)
            sym, fn = _PY_OPS[type(node.op)]
            if sym == "/" and b == 0:
                raise ZeroDivisionError
            r = float(fn(a, b))
            steps.append(BinStep(sym, a, b, r, r))
            return r
        raise ValueError("unsupported")

    visit(tree.body)
    return steps


def _try_expr(tokens, pct_mode: str) -> tuple[str, list[BinStep]] | None:
    if not tokens or tokens[0][0] not in ("num", "lp") or tokens[-1][0] not in ("num", "rp"):
        return None
    if not any(t[0] == "op" for t in tokens):
        return None
    expr = _to_python(tokens, pct_mode)
    try:
        steps = decompose(expr)
    except (SyntaxError, ValueError, ZeroDivisionError):
        return None
    return (expr, steps) if steps else None


def parse_equations(text: str, validate: bool = True) -> list[Equation]:
    """All `expr = number` equations in `text`, in order of appearance."""
    norm = normalize_text(text)
    tokens = _tokenize(norm)
    equations: list[Equation] = []
    seg_start = 0
    for i, (kind, _, _, _) in enumerate(tokens):
        if kind != "eq":
            continue
        # rhs: first number after '=' (skipping words like units/currency)
        rhs = None
        for k in range(i + 1, min(i + 4, len(tokens))):
            if tokens[k][0] == "num":
                rhs = tokens[k]
                break
            if tokens[k][0] in ("eq", "op", "lp"):
                # allow a leading minus on rhs
                if tokens[k][1] == "-" and k + 1 < len(tokens) and tokens[k + 1][0] == "num":
                    rhs = ("num", "-" + tokens[k + 1][1], tokens[k][2], tokens[k + 1][3])
                break
        lhs = _clean_lhs(tokens[seg_start:i])
        seg_start = i + 1
        if rhs is None:
            continue
        stated, stated_pct = _num(rhs[1].lstrip("-"))
        stated = -stated if rhs[1].startswith("-") else stated
        found = None
        for s in range(len(lhs)):  # longest suffix first
            cand = lhs[s:]
            for pct_mode in ("number", "fraction"):
                got = _try_expr(cand, pct_mode)
                if got is None:
                    continue
                expr, steps = got
                value = steps[-1].computed
                if (
                    not validate
                    or close(value, stated)
                    or (stated_pct and close(value, stated * 0.01))
                ):
                    found = (expr, steps, cand[0][2])
                    break
            if found:
                break
        if found is None:
            continue
        expr, steps, start = found
        steps[-1].result = stated  # the stated value is what the text carries forward
        equations.append(Equation(expr, stated, steps, start, rhs[3]))
    return equations


def program_from_text(text: str) -> list[BinStep]:
    """Concatenated binary steps of every validated equation in a gold solution."""
    return [s for eq in parse_equations(text, validate=True) for s in eq.steps]
