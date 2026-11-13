"""Independent solvers for the templates whose distractor clause was rewritten (prereg/S2.md A18).

Each solver reads the quantities it needs from the question text (anchored on the template's
fixed wording, never on the clause's span) and computes the answer. `distractor_review.py` runs
it on the clean and the with-clause text of every instance 1-16 and requires both to equal the
dataset's answer: the clause changes neither the answer nor what the solver reads.
"""
from __future__ import annotations

import math
import re
from fractions import Fraction as F

from .instance_checks import _UNITS, _WORDNUM, to_num

D = r"(\d[\d,]*(?:\.\d+)?)"                       # a number written with digits
M = r"\D*?(\d[\d,]*(?:\.\d+)?)"                   # an amount after a currency sign
NW = rf"(\d[\d,]*(?:\.\d+)?|{_WORDNUM})"           # digits or a number word
FR = r"([\w/ -]+?)"                                # a fraction or multiplier phrase
_ORD = {"half": 2, "halves": 2, "third": 3, "quarter": 4, "fourth": 4, "fifth": 5, "sixth": 6,
        "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10}
_MUL = {"twice": 2, "double": 2, "thrice": 3, "triple": 3, "quadruple": 4, "quintuple": 5}


def num(tok: str) -> F:
    return to_num(tok.replace(",", ""))


def ratio(tok: str) -> F:
    """'twice', 'three times', '1/3', 'two-thirds', 'a quarter', 'one-half', 'half', '5'."""
    t = tok.strip().lower()
    if t in _MUL:
        return F(_MUL[t])
    if t in ("half", "a half", "one half"):
        return F(1, 2)
    m = re.fullmatch(r"(\d+)/(\d+)", t)
    if m:
        return F(int(m.group(1)), int(m.group(2)))
    m = re.fullmatch(r"(.+?) times", t)
    if m:
        return num(m.group(1))
    m = re.fullmatch(r"(a|an|one|\w+)[ -](\w+?)s?", t)
    if m and (m.group(2) in _ORD or m.group(2) + "s" in _ORD):
        k = 1 if m.group(1) in ("a", "an") else num(m.group(1))
        return F(k) / _ORD.get(m.group(2), _ORD.get(m.group(2) + "s", 0))
    return num(t)


def g(q: str, pat: str, *k: int):
    m = re.search(pat, q)
    if not m:
        raise LookupError(pat)
    return [m.group(i) for i in (k or (1,))] if len(k) > 1 else m.group(k[0] if k else 1)


def gn(q: str, pat: str, *k: int):
    v = g(q, pat, *k)
    return [num(x) for x in v] if isinstance(v, list) else num(v)


def _t7(q):
    loads = re.findall(rf"buys {NW} (\d+)-\w+ (?:boxes|bags|packs|sacks)", q)
    total = sum(num(n) * num(w) for n, w in loads)
    return math.ceil(total / gn(q, rf"carry {D} \w+ at a time"))


def _t16(q):
    a, b, c, w = gn(q, rf"sold {D} boxes of [\w ]+? and {D} fewer boxes.*?sold {D} more boxes.*?and {D} boxes",
                    1, 2, 3, 4)
    p1, p2 = gn(q, rf"If the [\w ]+? cost {M} and the [\w ]+? cost {M}", 1, 2)
    return (2 * a + c) * p1 + (2 * w - b) * p2


def _t40(q):
    money = gn(q, rf"gave her {M} ")
    buy, price, get, off = g(q, rf"Buy {D} \w+ at {M} per \w+, get {D} \w+ ([\w/-]+) off", 1, 2, 3, 4)
    buy, price, get, off = num(buy), num(price), num(get), ratio(off)
    took = gn(q, rf"scooped up {D} ")
    candy = buy * price + get * price * (1 - off) + (took - buy - get) * price
    bag, n, each = gn(q, rf"for {M} and {D} [\w ]+? that were {M} each", 1, 2, 3)
    return money - candy - bag - n * each


def _t45(q):
    length = gn(q, rf"that is {D} \w+ long")
    gap = gn(q, rf"leave {D} \w+ between")
    plants = math.floor(length / gap)
    return (plants - gn(q, rf"already owns {D} flowers")) * gn(q, rf"costs {M} at")


def _t50(q):
    area, au = g(q, rf"is {D} square (\w+)", 1, 2)
    length, lu = g(q, rf"length of his room is {D} (\w+)", 1, 2)
    ans_u = g(q, r"perimeter of the room in (\w+)")
    to_ft = {"feet": F(1), "foot": F(1), "yards": F(3), "yard": F(3), "meters": F(1), "meter": F(1)}
    length = num(length) * to_ft[lu] / to_ft[au]
    perim = 2 * (length + num(area) / length)
    return perim * to_ft[au] / to_ft[ans_u]


def _t51(q):
    prices = {k.lower(): F(v) for k, v in re.findall(r"(\w+) for \D*?(\d+\.\d+) per \w+", q)}
    sales = re.findall(r"(\d+) \w+ of (\w+)", q.split("If ")[-1])
    return sum(math.floor(prices[k.lower()] + F(1, 2)) * int(n) for n, k in sales)


def _t75(q):
    counts = re.findall(r"(\d+) (\w+) balls", q.split(". ")[0])
    prices = {c: F(p) for c, p in re.findall(r"(\w+) balls cost \D*?(\d+(?:\.\d+)?)", q, re.I)}
    prices = {c.lower(): p for c, p in prices.items()}
    return sum(int(n) * prices[c.lower()] for n, c in counts)


def _t88(q):
    m = ratio(g(q, r"run ([\w ]+?) faster"))
    f = ratio(g(q, r"that is ([\w/-]+) as fast"))
    skip = gn(q, rf"skip at {D} miles")
    hours, fr, fw = g(q, rf"in {D} hours if she spends ([\w/-]+) of the time running and ([\w/-]+) of the time walking",
                      1, 2, 3)
    run = skip / f
    return num(hours) * (ratio(fr) * run + ratio(fw) * run / m)


def _t89(q):
    h1, h2 = gn(q, rf"from {D} PM to {D} PM", 1, 2)
    paid = h2 - h1 - gn(q, rf"gets {D} hours free")
    c = gn(q, rf"hour is {M} ")
    return c + (paid - 1) * ratio(g(q, r"is (\w+(?: times)?) the cost")) * c


def _t96(q):
    budget = gn(q, rf"had {M} budgeted")
    per = (gn(q, rf"A round of [\w -]+? is {M}\.") + gn(q, rf"have {M} in")
           + num(g(q, rf"ride the [\w -]+? {NW} times")) * gn(q, rf"cost {M} a ride"))
    return math.floor(budget / per) - 1


def _t47(q):
    a = gn(q, rf"\. \w+ finds {D} eggs")
    b = a * ratio(g(q, rf"\. \w+ finds {FR} as many as"))
    c = b - gn(q, rf"finds {D} less than")
    d = c * ratio(g(q, rf"and \w+ finds {FR} as many"))
    return 12 * gn(q, rf"hid {D} dozen") - a - b - c - d


def _t61(q):
    n = gn(q, rf"with {D} inside")
    fake = num(re.search(rf"\. ((?i:{NW[1:-1]})) of them he", q).group(1).lower())
    return n - n * ratio(g(q, r"dropped ([\w/-]+) of them")) + gn(q, rf"came up with {D}") - fake


SOLVERS = {
    3: lambda q: (lambda t, a, b: t - a - b - ratio(g(q, r"is (\w+(?: times)?) the number that prefer")) * b)(
        *gn(q, rf"of {D} students has various hobbies\. {D} like to [\w ]+?, {D} like to", 1, 2, 3)),
    5: lambda q: (lambda a, b, t: t * a / (a + b))(*gn(q, rf"ratio of {D}:{D}\..*?a total of {D} ", 1, 2, 3)),
    6: lambda q: gn(q, rf"has {D} [\w ]+? of [\w ]+? field") * gn(q, rf"There are {D} \w+ per ")
                 * 12 / gn(q, rf"every {D} months"),
    7: _t7,
    12: lambda q: (lambda per, n, k: n - per * k)(gn(q, rf"can hold {D} "), *gn(q, rf"If he has {D} \w+ and {D} ", 1, 2)),
    14: lambda q: gn(q, rf"have {D} minutes") - gn(q, rf"takes them {D} minutes") - gn(q, rf"another {D} minutes"),
    16: _t16,
    20: lambda q: (lambda a, b: a + b + ratio(g(q, r"he picks (\w+(?: times)?) the number")) * a)(
        *gn(q, rf"picks {D} \w+ on \w+\. Then he picks {D} ", 1, 2)),
    22: lambda q: gn(q, rf"total of {D} ") + num(g(q, rf"used {NW} of the")) - gn(q, rf"package of {D} new"),
    26: lambda q: {"week": 7, "month": 30}[g(q, r"in one (week|month)")] * sum(num(x) for x in re.findall(rf"eats {D} [\w ]+? per day", q)),
    28: lambda q: (lambda c, f: c - c * f - gn(q, rf"{D} patients are admitted") * 7 * gn(q, rf"after {D} weeks"))(
        gn(q, rf"capacity of {D} "), ratio(g(q, r"with ([\w/-]+) occupied"))),
    30: lambda q: gn(q, rf"has {M} and wants") - gn(q, rf"wants to buy {D} ") * gn(q, rf"costs {M}\."),
    31: lambda q: gn(q, rf"is {D} years old") + gn(q, rf"currently {D} years apart") / 2,
    32: lambda q: gn(q, rf"selling price of {M}\.") * (1 + (gn(q, rf"which is {D}%") + gn(q, rf"that is {D}%")) / 100)
                  - gn(q, rf"beyond her {M} budget"),
    33: lambda q: gn(q, rf"reach {D} signatures") - sum(gn(q, rf"has {D} signatures in her book, and \w+ has {D}", 1, 2)),
    34: lambda q: (lambda s: s - s * gn(q, rf"amount to {D}%") / 100 - s * gn(q, rf"is {D}% of the selling") / 100
                   - gn(q, rf"paid {M} for the remaining"))(gn(q, rf"for {M}\. He paid")),
    35: lambda q: (lambda c: c - c * ratio(g(q, rf"cover {FR} of the cost")) - gn(q, rf"he has {M} and"))(
        gn(q, rf"costs {M}\?")),
    36: lambda q: sum(num(n) * num(p) for n, p in re.findall(rf"{D} dozen [\w ]+? (?:which cost|for) {M} per dozen", q)),
    37: lambda q: (lambda p, a: p / (a + a * ratio(g(q, r"while \w+ can eat ([\w ]+?) as much"))))(
        gn(q, rf"has {D} "), gn(q, rf"If \w+ can eat {D} ")),
    38: lambda q: (gn(q, rf"How much more do {D} ") * gn(q, rf"at the \w+ costs {M}\.")
                   - gn(q, rf"cost than {D} ") * gn(q, rf"cost {M} each\.")),
    39: lambda q: (lambda f, d, m, h: f * m / d / (h * 60) * 100)(
        *gn(q, rf"with {D} \w+\. They have {D} days.*?takes them {D} minutes.*?work {D} hours", 1, 2, 3, 4)),
    40: _t40,
    41: lambda q: (lambda a, m: gn(q, rf"there are {D} \w+ in the") - a - a * m)(
        gn(q, rf"made out of {D} "), ratio(g(q, r"\w+, ([\w ]+?) as many"))),
    42: lambda q: 7 * gn(q, rf"uses {D} \w+ of") * gn(q, rf"costs {M} per \w+\.")
                  * (1 + gn(q, rf"costs {D}% more") / 100) + gn(q, rf"himself an? [\w -]+? for {M}"),
    43: lambda q: (lambda b: b + b * ratio(g(q, rf"found {FR} as many")))(gn(q, rf"there were {D} ")),
    44: lambda q: (lambda n: n + n * ratio(g(q, r"there are ([\w/ -]+?) the number of boys")))(gn(q, rf"There are {D} girls")),
    45: _t45,
    46: lambda q: (lambda s, a, b: (s - a - b) / a)(
        gn(q, rf"space for {D} people"), *gn(q, rf"accounting for {D} seats for graduates and {D} seats", 1, 2)),
    47: _t47,
    48: lambda q: (lambda p, s, t, n: n / p + n / s * t)(
        *gn(q, rf"peel {D} \w+ a minute and saute {D} \w+ in {D} minutes", 1, 2, 3), gn(q, rf"saute {D} \w+\?")),
    50: _t50,
    51: _t51,
    52: lambda q: (lambda k, v: k * v - gn(q, rf"buys {D} ") * gn(q, rf"buys {D} [\w ]+? for {M} each", 2)
                   - gn(q, rf"also buys {D} packs") * gn(q, rf"packs of [\w ]+? for {M} each"))(
        *gn(q, rf"has {D} \D+?{D} bills", 1, 2)),
    53: lambda q: (lambda a: a + gn(q, rf"scored {D} points more") + a * ratio(g(q, rf"points more than {FR} as many")))(
        gn(q, rf"scored {D} points in")),
    56: lambda q: gn(q, rf"is {D} \w+ of") * gn(q, rf"drinks {D} ") * gn(q, rf"has {D} calories"),
    57: lambda q: (lambda b, c: b * ratio(g(q, rf"has {FR} as many tubes of \w+ paint as \w+, and"))
                   + c * ratio(g(q, rf", and {FR} as many tubes")))(
        *gn(q, rf"has {D} tubes of \w+ paint and {D} tubes", 1, 2)),
    58: lambda q: (gn(q, rf"mine {D} \w+ of ore per day") * ratio(g(q, r"mine ([\w ]+?) as much"))
                   * (1 + gn(q, rf"{D}% more") / 100) * gn(q, rf"can {D} \w+ with") * gn(q, rf"with {D} days")),
    59: lambda q: (lambda a, fa, b, fb: (a - gn(q, rf"spill {D} liters")) * ratio(fa) + num(b) * ratio(fb))(
        gn(q, rf"have {D} liters"), g(q, r"that are ([\w/-]+) water"), g(q, rf"to {D} liters"),
        g(q, r"that is ([\w/-]+) water")),
    61: _t61,
    62: lambda q: (lambda a, p: 2 * (a + a * (1 + p / 100)))(*gn(q, rf"there are {D} \w+ cards, and {D}% more", 1, 2)),
    63: lambda q: (lambda a, k: 2 * a - k)(*gn(q, rf"has {D} \w+\. His \w+ has {D} fewer", 1, 2)),
    64: lambda q: gn(q, rf"has {D} jars") * gn(q, rf"decorate {D} ") / gn(q, rf"holds {D} "),
    65: lambda q: gn(q, rf"do {D} fewer hours") * gn(q, rf"losing {M}/hour") - gn(q, rf"charges {M}\."),
    67: lambda q: (gn(q, rf"weighs {D} ") + gn(q, rf"weighing {D} ") + (lambda n, w: n * w)(
        *gn(q, rf"his {D} young \w+ who weigh {D} ", 1, 2))) * gn(q, rf"is {D}% of the weight") / 100,
    68: lambda q: (lambda a, b, n, m: a * n + b * m)(
        *gn(q, rf"takes \w+ {D} minutes to finish an? [\w ]+? and {D} minutes", 1, 2),
        *gn(q, rf"solved {D} [\w ]+? and {D} ", 1, 2)),
    70: lambda q: 12 * gn(q, rf"for {D} weeks every month") * gn(q, rf"for {D} days every week") * gn(q, rf"paid {M} every"),
    71: lambda q: (lambda a, k, na, nb: na * a + nb * (2 * a - k))(
        *gn(q, rf"needs {D} \w+, and each \w+ needs {D} ", 1, 2), *gn(q, rf"there are {D} \w+ and {D} ", 1, 2)),
    72: lambda q: ((gn(q, rf"with {D} people") - gn(q, rf"with {D} of them")) * gn(q, rf"send {D} letters")
                   * gn(q, rf"that are {D} pages") * gn(q, rf"every {D} minutes") / 60),
    75: _t75,
    77: lambda q: (lambda p: p - p * ratio(g(q, r"writing ([\w/-]+) of the")))(gn(q, rf"submit an? {D}-page")),
    81: lambda q: gn(q, rf"hid {D} ") - 2 * gn(q, rf"each found {D} ") - gn(q, rf"except {D} "),
    82: lambda q: (lambda a, b: (a + b) * (1 - ratio(g(q, r", ([\w/-]+) of the total"))) - gn(q, rf"if {D} \w+ stayed"))(
        *gn(q, rf"there were {D} \w+ and {D} ", 1, 2)),
    83: lambda q: gn(q, rf"made {D} \w+\.") - sum(gn(q, rf"have {D} [\w ]+?, {D} [\w ]+?, and some", 1, 2)),
    86: lambda q: (lambda j: j + j + gn(q, rf", a pair of [\w -]+? for {M} more")
                   + gn(q, rf"originally {M} but") * ratio(g(q, r"on sale for ([\w/-]+) price"))
                   + (lambda k, h: h - k)(*gn(q, rf"coupon for {M} off the package of {M} ", 1, 2)))(
        gn(q, rf"She bought an? [\w -]+? for {M},")),
    88: _t88,
    89: _t89,
    90: lambda q: (gn(q, rf"has {D} friends") / ratio(g(q, r"has ([\w ]+?) as many \w+ friends as"))
                   * ratio(g(q, r"has ([\w ]+?) as many friends on"))),
    91: lambda q: (lambda b, k, l: b + ratio(g(q, r"there are ([\w ]+?) as many")) * b + k + (k - l))(
        gn(q, rf"total of {D} "), *gn(q, rf"bought {D} more \w+ at the store and {D} fewer", 1, 2)),
    92: lambda q: gn(q, rf"slew {D} ") * math.prod(ratio(x) for x in re.findall(rf"slew {FR} as many", q)),
    93: lambda q: (lambda c: gn(q, rf"swim {D} ") * (ratio(g(q, r"minutes more than (\w+(?: times)?) as long")) * c
                                                  + gn(q, rf"in {D} minutes more than") - c))(
        gn(q, rf"in {D} minutes\. When")),
    95: lambda q: (lambda a: a + a * ratio(g(q, r"and ([\w ]+?) that number")) + sum(
        gn(q, rf"There are {D} boys and {D} girls", 1, 2)))(gn(q, rf"There are {D} students playing")),
    96: _t96,
    98: lambda q: gn(q, rf"infects {D} people") * (1 + gn(q, rf"infects {D} others")) ** int(gn(q, rf"after {D} days")),
    99: lambda q: (lambda a, d1, b, d2: (a * d1 + b * d2) / 12)(
        *gn(q, rf"eats {D} eggs a day for {D} days", 1, 2), *gn(q, rf"to {D} eggs a day for {D} days", 1, 2)),
}


def solve(template_id: int, question: str) -> F:
    return F(SOLVERS[template_id](question))
