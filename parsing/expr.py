"""Arithmetic equations in free text (used by the gold extractor and the step parser).

An *equation* is `<expression> = <number>`, where the expression contains only numbers,
+ - * / (also x, ×, ÷, \\times, \\div, \\cdot) and parentheses. Expressions with more than one
operator are *compound*; `decompose` turns them into binary steps in evaluation order.

Everything is exact (fractions.Fraction). A percent sign, a unit word or any other token inside
the expression makes it unparseable: the equation is then skipped and counted, never guessed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction

OPS = {"+": "+", "-": "-", "−": "-", "–": "-", "*": "*", "x": "*", "×": "*", "·": "*",
       "/": "/", "÷": "/"}

_NORMALISE = [
    (re.compile(r"\\times|\\cdot"), "*"),
    (re.compile(r"\\div"), "/"),
    (re.compile(r"\\[()\[\]]"), " "),     # LaTeX math delimiters \( \) \[ \]
    (re.compile(r"\*\*"), " "),           # Markdown bold (never a power in these problems)
    (re.compile(r"[\u2212\u2013]"), "-"),
]

NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_TOKEN = re.compile(rf"\s*(?:(?P<num>{NUM})|(?P<op>[+\-*/x×÷·])|(?P<par>[()]))")
# characters an expression may contain; scanned backwards from each '='
_EXPR_CHARS = set("0123456789.,$ ()+-*/x×÷·\t")
_RHS = re.compile(rf"\s*\$?\s*(?P<num>-?(?:{NUM}))")


def normalise(text: str) -> str:
    for pat, rep in _NORMALISE:
        text = pat.sub(rep, text)
    return text


def to_fraction(s: str) -> Fraction:
    return Fraction(s.replace(",", ""))


@dataclass(frozen=True)
class BinStep:
    op: str                 # one of + - * /
    a: Fraction
    b: Fraction
    result: Fraction        # exact value of a op b
    a_sub: int | None = None  # index of the sub-step that produced a (compound expressions)
    b_sub: int | None = None


@dataclass
class Equation:
    start: int              # character span of the whole equation in the normalised text
    end: int
    lhs: str
    rhs: Fraction           # stated result
    steps: list[BinStep] = field(default_factory=list)

    @property
    def compound(self) -> bool:
        return len(self.steps) > 1

    @property
    def value(self) -> Fraction:
        return self.steps[-1].result

    @property
    def consistent(self) -> bool:
        """Stated result equals the exact value of the expression."""
        return self.value == self.rhs


# --------------------------------------------------------------------------- parsing
class _Parser:
    """Recursive descent over tokens; builds binary steps in evaluation order."""

    def __init__(self, toks: list[tuple[str, str]]):
        self.toks, self.i, self.steps = toks, 0, []

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else (None, None)

    def take(self):
        t = self.peek()
        self.i += 1
        return t

    def expr(self):
        left = self.term()
        while self.peek()[0] == "op" and OPS[self.peek()[1]] in "+-":
            op = OPS[self.take()[1]]
            left = self._emit(op, left, self.term())
        return left

    def term(self):
        left = self.atom()
        while self.peek()[0] == "op" and OPS[self.peek()[1]] in "*/":
            op = OPS[self.take()[1]]
            left = self._emit(op, left, self.atom())
        return left

    def atom(self):
        kind, val = self.take()
        if kind == "num":
            return (to_fraction(val), None)
        if kind == "par" and val == "(":
            inner = self.expr()
            if self.take() != ("par", ")"):
                raise ValueError("unbalanced")
            return inner
        raise ValueError(f"unexpected token {val!r}")

    def _emit(self, op, left, right):
        (a, ai), (b, bi) = left, right
        if op == "/" and b == 0:
            raise ValueError("division by zero")
        r = {"+": a + b, "-": a - b, "*": a * b, "/": a / b if b else None}[op]
        self.steps.append(BinStep(op, a, b, r, ai, bi))
        return (r, len(self.steps) - 1)


def _tokenise(s: str) -> list[tuple[str, str]] | None:
    s = s.replace("$", " ")
    toks, pos = [], 0
    while pos < len(s):
        if s[pos:].strip() == "":
            break
        m = _TOKEN.match(s, pos)
        if not m or m.end() == pos:
            return None
        kind = m.lastgroup
        toks.append((kind, m.group(kind)))
        pos = m.end()
    return toks


def parse_expression(lhs: str) -> list[BinStep] | None:
    """Binary steps of an expression, or None if it is not a pure arithmetic expression with
    at least one operator."""
    toks = _tokenise(lhs)
    if not toks:
        return None
    # strip leading operators / unmatched ")" picked up by the backward scan (e.g. "box" -> "x")
    while toks and (toks[0][0] == "op" or toks[0] == ("par", ")")):
        toks = toks[1:]
    # drop a leading "(" that is never closed
    while toks and toks[0] == ("par", "(") and \
            sum(1 if t == ("par", "(") else -1 if t == ("par", ")") else 0 for t in toks) > 0:
        toks = toks[1:]
    if not toks:
        return None
    p = _Parser(toks)
    try:
        p.expr()
    except (ValueError, ZeroDivisionError):
        return None
    if p.i != len(toks) or not p.steps:
        return None
    return p.steps


def find_equations(text: str) -> tuple[list[Equation], int]:
    """All parseable equations in order, and the number of '=' signs that did not parse."""
    text = normalise(text)
    out, skipped = [], 0
    for m in re.finditer(r"(?<![<>!=])=(?!=)", text):
        j = m.start()
        i = j
        while i > 0 and text[i - 1] in _EXPR_CHARS:
            i -= 1
        lhs = text[i:j]
        # a comma followed by a space is a list separator, and a period not followed by a digit
        # ends a sentence or a step label ("Step 1."): keep only the text after the last one
        lhs = re.split(r",\s|\.(?!\d)", lhs)[-1]
        r = _RHS.match(text, m.end())
        steps = None
        if lhs.strip():
            steps = parse_expression(lhs)
            # a label such as "2)" can precede the expression: drop leading tokens until the rest
            # parses completely (the remainder must still be a full expression with an operator)
            cut = lhs
            while steps is None:
                nxt = re.match(r"\s*(?:\d[\d,.]*|[^\d\s])\s*", cut)
                if not nxt or nxt.end() >= len(cut):
                    break
                cut = cut[nxt.end():]
                steps = parse_expression(cut)
            lhs = cut.lstrip(" \t)+-*/x×÷·") if steps is not None else lhs
        if steps is None or r is None:
            skipped += 1
            continue
        lhs_start = j - len(lhs)
        out.append(Equation(lhs_start + (len(lhs) - len(lhs.lstrip())), r.end(), lhs.strip(),
                            to_fraction(r.group("num")), steps))
    return out, skipped


def first_equation(text: str) -> Equation | None:
    eqs, _ = find_equations(text)
    return eqs[0] if eqs else None
