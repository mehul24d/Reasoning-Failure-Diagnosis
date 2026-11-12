"""Arithmetic equations in free text (used by the gold extractor and the step parser).

An *equation* is `<expression> = <number>`. The expression is read backwards from the '=' to
the start of its clause (a line break, ':', ';', '?', '!', another '=' or a sentence end).
Inside that clause:
  * numbers may carry currency symbols ($ € £ ¥ ₹ ₺ ₣), thousands separators, '%' (n% = n/100)
    and unit words ("10 baskets * $9/basket"): unit words and "/unit" are dropped;
  * operators are + - * / and also x, ×, ·, ÷, \\times, \\cdot, \\div; LaTeX \\frac{a}{b} is a/b;
  * "a/b" written without spaces is a fraction literal when a < b or the expression uses ÷
    (so 21/4 ÷ 3/4 and 60 * 20/100 read as intended), a mixed number "11 1/4" is 11 + 1/4, and
    "2(3 + 4)" is 2 * (3 + 4);
  * a number glued to a single letter ("2m", "3y") is an algebra variable: the clause is
    rejected rather than misread.
The expression is the LONGEST suffix of the clause that parses completely and has at least one
operator. Compound expressions (more than one operator) are split into binary steps in
evaluation order. Everything is exact (fractions.Fraction).

`normalise` keeps the text length unchanged, so every span refers to the original text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction

OPS = {"+": "+", "-": "-", "*": "*", "x": "*", "X": "*", "×": "*", "·": "*", "/": "/", "÷": "/"}
CURRENCY = "$€£¥₹₺₣"

NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"


def _same_len(rep: str):
    return lambda m: (rep + " " * len(m.group(0)))[:len(m.group(0))]


def _frac(m):
    s = f"({m.group(1)}/{m.group(2)})"
    return s + " " * (len(m.group(0)) - len(s))


_NORMALISE = [
    (re.compile(r"\\[dt]?frac\{\s*(\d+)\s*\}\{\s*(\d+)\s*\}"), _frac),
    (re.compile(r"\\times|\\cdot"), _same_len("*")),
    (re.compile(r"\\div"), _same_len("/")),
    (re.compile(r"\\[()\[\]]|\\,|\\!"), _same_len(" ")),
    (re.compile(r"\*\*"), _same_len(" ")),               # Markdown bold
    (re.compile(r"[\u2212\u2013\u2014]"), _same_len("-")),
    (re.compile(f"[{CURRENCY}]"), _same_len(" ")),
    (re.compile(r"(?<=[A-Za-z])-(?=[A-Za-z])"), _same_len("'")),   # "time-outs" is one word
]


def normalise(text: str) -> str:
    """Length-preserving normalisation (spans in the result are spans in the input)."""
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
    start: int              # character span of the equation in the (original) text
    end: int
    lhs: str
    rhs: Fraction           # stated result
    steps: list[BinStep] = field(default_factory=list)
    eq_pos: int = -1        # position of the '=' sign
    chained: bool = False   # an intermediate link of a chain "a = b = c" (see _mark_chains)

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


# --------------------------------------------------------------------------- tokens
@dataclass
class _Tok:
    kind: str               # num | op | lp | rp | word | var | junk
    text: str
    start: int
    end: int
    value: Fraction | None = None


_LEX = re.compile(
    rf"(?P<num>(?:{NUM})%?)(?P<glue>[A-Za-z]+)?"
    r"|(?P<op>[+\-*/×·÷])"
    r"|(?P<lp>\()|(?P<rp>\))"
    r"|(?P<word>[A-Za-z][A-Za-z'.]*)"
    r"|(?P<ws>\s+)"
    r"|(?P<junk>.)")


def _lex(s: str, base: int) -> list[_Tok]:
    out: list[_Tok] = []
    for m in _LEX.finditer(s):
        if m.group("ws"):
            continue
        st, en = base + m.start(), base + m.end()
        if m.group("num") is not None:
            t = m.group("num")
            pct = t.endswith("%")
            v = to_fraction(t.rstrip("%")) / (100 if pct else 1)
            glue = m.group("glue")
            if glue and glue.lower() == "x":          # "3x4" is 3 * 4
                out.append(_Tok("num", t, st, base + m.start("glue"), v))
                out.append(_Tok("op", "*", base + m.start("glue"), base + m.end("glue")))
            elif glue and len(glue) == 1:             # "2m": an algebra variable
                out.append(_Tok("var", m.group(0), st, en))
            else:                                     # "5kg", "4th": unit / ordinal dropped
                out.append(_Tok("num", t, st, base + m.end("num"), v))
                if glue:
                    out.append(_Tok("word", glue, base + m.start("glue"), en))
        elif m.group("op"):
            out.append(_Tok("op", OPS[m.group("op")], st, en))
        elif m.group("lp"):
            out.append(_Tok("lp", "(", st, en))
        elif m.group("rp"):
            out.append(_Tok("rp", ")", st, en))
        elif m.group("word"):
            w = m.group("word")
            out.append(_Tok("word", w, st, en))
        else:
            out.append(_Tok("junk", m.group(0), st, en))
    return out


_UNIT_LETTERS = set("mgshlLc")     # metres, grams, seconds, hours, litres, cents


def _clean(toks: list[_Tok]) -> list[_Tok] | None:
    """Words -> operators or dropped; literals merged. None if the clause holds a variable."""
    var = [t for t in toks if t.kind == "var"]
    if var:
        # "3m * 2m": one unit letter on every number is a unit, not an algebra variable
        letters = {t.text[-1] for t in var}
        if len(letters) == 1 and letters <= _UNIT_LETTERS and \
                not any(t.kind == "num" for t in toks):
            toks = [x for t in toks for x in (
                (_Tok("num", t.text[:-1], t.start, t.end - 1, to_fraction(t.text[:-1])),
                 _Tok("word", t.text[-1], t.end - 1, t.end)) if t.kind == "var" else (t,))]
        else:
            return None
    out: list[_Tok] = []
    i = 0
    while i < len(toks):
        t = toks[i]
        prev = out[-1] if out else None
        nxt = toks[i + 1] if i + 1 < len(toks) else None
        if t.kind == "word":
            # the operand before "x" / "times" may carry up to two unit words: "3 mph x 2"
            back = len(out) - 1
            while back >= 0 and back >= len(out) - 2 and out[back].kind == "gap" and \
                    out[back].text[:1].isalpha():
                back -= 1
            left = out[back] if back >= 0 else None
            if t.text in ("x", "X") and left is not None and left.kind in ("num", "rp") \
                    and nxt is not None and nxt.kind in ("num", "lp"):
                out.append(_Tok("op", "*", t.start, t.end))
            elif t.text.lower() in ("times",) and left is not None and left.kind == "num" \
                    and nxt is not None and nxt.kind == "num":
                out.append(_Tok("op", "*", t.start, t.end))
            elif prev is not None and prev.kind == "op" and prev.text == "/" and \
                    (len(out) < 2 or out[-2].kind in ("num", "rp") or
                     (out[-2].kind == "gap" and len(out) >= 3 and out[-3].kind == "num")):
                out.pop()        # "$9/basket", "14 strawberries/plant": drop "/" and the unit
            else:
                out.append(_Tok("gap", t.text, t.start, t.end))   # a word between items
            i += 1
            continue
        if t.kind == "junk":
            out.append(_Tok("gap", t.text, t.start, t.end))
            i += 1
            continue
        out.append(t)
        i += 1
    return _merge_literals(_drop_word_parens(out))


def _drop_word_parens(toks: list[_Tok]) -> list[_Tok]:
    """Remove parenthesised asides without numbers: "150 km (each way) * 2"."""
    out: list[_Tok] = []
    for t in toks:
        out.append(t)
        if t.kind == "rp":
            k = len(out) - 2
            while k >= 0 and out[k].kind == "gap":
                k -= 1
            if k >= 0 and out[k].kind == "lp" and k < len(out) - 2:
                del out[k:]
    return out


def _merge_literals(toks: list[_Tok]) -> list[_Tok]:
    """Mark tight fractions "a/b" (no spaces); _resolve_fracs decides how to read them."""
    out: list[_Tok] = []
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.kind == "num" and i + 2 < len(toks) and toks[i + 1].kind == "op" and \
                toks[i + 1].text == "/" and toks[i + 2].kind == "num" and \
                t.end == toks[i + 1].start and toks[i + 1].end == toks[i + 2].start and \
                t.value.denominator == 1 and toks[i + 2].value.denominator == 1 and \
                "%" not in t.text + toks[i + 2].text and toks[i + 2].value != 0:
            out.append(_Tok("frac", f"{t.value}/{toks[i + 2].value}", t.start, toks[i + 2].end))
            i += 3
            continue
        out.append(t)
        i += 1
    return out


def _resolve_fracs(toks: list[_Tok], raw: str) -> list[_Tok]:
    """Decide for each tight fraction whether it is a literal or a division."""
    uses_obelus = "÷" in raw
    n_ops = sum(t.kind == "op" for t in toks)
    out: list[_Tok] = []
    for k, t in enumerate(toks):
        if t.kind != "frac":
            out.append(t)
            continue
        num_s, den_s = t.text.split("/")
        a, b = Fraction(num_s), Fraction(den_s)
        prev = out[-1] if out else None
        literal = (a < b or uses_obelus) and (n_ops >= 1 or prev is not None and
                                             prev.kind == "num")
        if prev is not None and prev.kind == "num" and a < b and prev.value.denominator == 1 \
                and prev.end < t.start:
            # mixed number "11 1/4"
            out[-1] = _Tok("num", f"{prev.text} {t.text}", prev.start, t.end, prev.value + a / b)
            continue
        if literal:
            out.append(_Tok("num", t.text, t.start, t.end, a / b))
        else:
            mid = t.start + len(num_s)
            out += [_Tok("num", num_s, t.start, mid, a), _Tok("op", "/", mid, mid + 1),
                    _Tok("num", den_s, mid + 1, t.end, b)]
    # implicit multiplication: num ( | ) num | ) (
    res: list[_Tok] = []
    for t in out:
        if res and ((res[-1].kind == "num" and t.kind == "lp") or
                    (res[-1].kind == "rp" and t.kind in ("num", "lp"))):
            res.append(_Tok("op", "*", t.start, t.start))
        res.append(t)
    return res


# --------------------------------------------------------------------------- parsing
class _Parser:
    """Recursive descent over tokens; builds binary steps in evaluation order."""

    def __init__(self, toks: list[_Tok]):
        self.toks, self.i, self.steps = toks, 0, []

    def peek(self) -> _Tok | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def expr(self):
        left = self.term()
        while (t := self.peek()) is not None and t.kind == "op" and t.text in "+-":
            self.i += 1
            left = self._emit(t.text, left, self.term())
        return left

    def term(self):
        left = self.atom()
        while (t := self.peek()) is not None and t.kind == "op" and t.text in "*/":
            self.i += 1
            left = self._emit(t.text, left, self.atom())
        return left

    def atom(self):
        t = self.peek()
        if t is None:
            raise ValueError("end")
        self.i += 1
        if t.kind == "num":
            return (t.value, None)
        if t.kind == "op" and t.text == "-" and self.i == 1:      # leading unary minus
            nxt = self.peek()
            if nxt is not None and nxt.kind == "num":
                self.i += 1
                return (-nxt.value, None)
        if t.kind == "lp":
            inner = self.expr()
            c = self.peek()
            if c is None or c.kind != "rp":
                raise ValueError("unbalanced")
            self.i += 1
            return inner
        raise ValueError(f"unexpected {t.text!r}")

    def _emit(self, op, left, right):
        (a, ai), (b, bi) = left, right
        if op == "/" and b == 0:
            raise ValueError("division by zero")
        r = {"+": a + b, "-": a - b, "*": a * b, "/": a / b if b else None}[op]
        self.steps.append(BinStep(op, a, b, r, ai, bi))
        return (r, len(self.steps) - 1)


def _parse_full(toks: list[_Tok]) -> list[BinStep] | None:
    if not toks or any(t.kind == "gap" for t in toks):
        return None
    p = _Parser(toks)
    try:
        p.expr()
    except (ValueError, ZeroDivisionError):
        return None
    if p.i != len(toks) or not p.steps:
        return None
    return p.steps


def _without_unit_gaps(toks: list[_Tok]) -> list[_Tok]:
    """Drop words that sit next to a number inside an expression ("10 baskets * 9 = 90",
    "100 horses + 29 sheep"). A word between two numbers with no operator stays a gap, so it
    still separates unrelated numbers ("3 dogs and 10 * 9")."""
    out = []
    for k, t in enumerate(toks):
        if t.kind != "gap":
            out.append(t)
            continue
        # look at the nearest non-gap neighbours
        left = next((x for x in reversed(toks[:k]) if x.kind != "gap"), None)
        right = next((x for x in toks[k + 1:] if x.kind != "gap"), None)
        between_operands = left is not None and right is not None and \
            left.kind in ("num", "rp") and right.kind in ("num", "lp")
        if between_operands:
            out.append(t)          # keeps "3 dogs and 10" separated
        # otherwise (next to an operator, a paren, or the clause edge) it is a unit: drop it
    return out


def parse_tokens(toks: list[_Tok], raw: str) -> tuple[list[BinStep], int, bool] | None:
    """Longest suffix of the clause that is a complete expression with an operator.
    Returns (steps, start offset, cut) where `cut` says that an operator, a parenthesis or a
    number-free gap directly before the suffix was left out (the expression may continue to
    the left, so an inconsistent result must not be trusted)."""
    toks = _resolve_fracs(_without_unit_gaps(toks), raw)
    for i in range(len(toks)):
        unary = toks[i].kind == "op" and toks[i].text == "-" and \
            (i == 0 or toks[i - 1].kind not in ("num", "rp"))
        if toks[i].kind not in ("num", "lp") and not unary:
            continue
        steps = _parse_full(toks[i:])
        if steps is not None:
            cut = i > 0 and toks[i - 1].kind in ("op", "lp", "rp", "num")
            return steps, toks[i].start, cut
    return None


def parse_expression(lhs: str) -> list[BinStep] | None:
    """Binary steps of an expression (the whole string must be one expression)."""
    text = normalise(lhs)
    toks = _clean(_lex(text, 0))
    if toks is None:
        return None
    toks2 = _resolve_fracs(_without_unit_gaps(toks), text)
    if any(t.kind == "gap" for t in toks2):
        return None
    return _parse_full(toks2)


# a sentence end ('.' before a space), but not the '.' of a title such as "Ms." or "Dr."
_STOP = re.compile(r"[\n:;?!=<>]|(?<!\bMr)(?<!\bMs)(?<!\bDr)(?<!\bSt)(?<!\bvs)(?<!\bMrs)"
                   r"\.(?=\s|$)|,\s")
_RHS = re.compile(rf"[ \t]*(?P<neg>-)?[ \t]*(?P<num>{NUM})"
                  r"(?:(?P<frac>/\d+)|[ \t]+(?P<mixed>\d+/\d+))?"
                  r"(?P<pct>%)?(?P<glue>[A-Za-z]+)?")


def _rhs_value(r) -> Fraction | None:
    v = to_fraction(r.group("num"))
    if r.group("frac"):
        d = int(r.group("frac")[1:])
        if d == 0 or "." in r.group("num"):
            return None
        v = v / d
    elif r.group("mixed"):
        a, b = map(int, r.group("mixed").split("/"))
        if b == 0 or a >= b or v.denominator != 1:
            return None
        v = v + Fraction(a, b)
    if r.group("pct"):
        v = v / 100
    return -v if r.group("neg") else v


def find_equations(text: str) -> tuple[list[Equation], int]:
    """All parseable equations in order, and the number of '=' whose left side has an
    operator but could not be parsed (signs with no operator before them, such as
    "Total = 48", are not counted)."""
    norm = normalise(text)
    out, skipped = [], 0
    for m in re.finditer(r"(?<![<>!=])=(?!=)", norm):
        j = m.start()
        # clause start: last stop character before '=' (max 240 chars back)
        lo = max(0, j - 240)
        stops = [s.end() for s in _STOP.finditer(norm, lo, j)]
        cs = stops[-1] if stops else lo
        clause = norm[cs:j]
        r = _RHS.match(norm, m.end())
        has_op = bool(re.search(r"\d\s*[A-Za-z%$]*\s*[-+*/×·÷x(]|[)]\s*[-+*/×·÷x(]", clause))
        if r is None or (r.group("glue") and len(r.group("glue")) == 1
                         and r.group("glue").lower() != "x"
                         and r.group("glue") not in _UNIT_LETTERS):
            skipped += has_op
            continue
        if "^" in clause:                 # powers are outside the step model
            skipped += has_op
            continue
        toks = _clean(_lex(clause, cs))
        got = parse_tokens(toks, clause) if toks else None
        if got is None:
            skipped += has_op
            continue
        steps, start, cut = got
        rhs = _rhs_value(r)
        if rhs is None:
            skipped += 1
            continue
        if cut and steps[-1].result != rhs:
            skipped += 1                  # a truncated, inconsistent left side: do not guess
            continue
        end = max(e for e in (r.end("num"), r.end("frac"), r.end("mixed"), r.end("pct")))
        out.append(Equation(start, end, text[start:j].strip(), rhs, steps, eq_pos=j))
    _mark_chains(out)
    return out, skipped


def _mark_chains(eqs: list[Equation]) -> None:
    """Partial evaluation "8*5 + 90 = 40 + 90 = 130": the first '=' states an intermediate form,
    not a result. When an inconsistent equation's right side starts the next equation and that
    one is consistent with the same value, the first equation is a chain link: its stated
    result is replaced by the exact value."""
    for a, b in zip(eqs, eqs[1:]):
        if not a.consistent and b.start <= a.end and b.start >= a.eq_pos and b.consistent \
                and b.rhs == a.value:
            a.rhs = a.value
            a.chained = True
    # "Time at 3 mph + Time at 1 mph = 2 hours + 4 hours = 6 hours": a left side whose numbers
    # are labels (a word follows an operator) and whose right side starts a consistent
    # equation is a description, not a computation (even when the labels happen to add up,
    # "age in 3 years + age in 3 years + age in 3 years = 9 + 6 + 21 = 36"): drop it
    drop = {i for i, (a, b) in enumerate(zip(eqs, eqs[1:]))
            if not a.chained and b.start <= a.end and b.start >= a.eq_pos
            and b.consistent and re.search(r"[-+*/×÷]\s*[A-Za-z]{2,}", a.lhs)}
    eqs[:] = [e for i, e in enumerate(eqs) if i not in drop]


def first_equation(text: str) -> Equation | None:
    eqs, _ = find_equations(text)
    return eqs[0] if eqs else None
