"""Instance-level checks of the distractor clauses (prereg/S2.md A18, DECISIONS D40).

GSM-Symbolic varies names, objects, places, days and units from instance to instance, but a
clause is one fixed sentence (only d changes). These checks run on every instance a clause can
be inserted into (instances 1-16: the study's 8 and the top-up instances 9-16).

1. Partitive check (`partitive_check`). A partitive clause ("{d} of the X ...") names a subset
of a group. It contradicts the problem when d is not smaller than the size of that group in the
instance (e.g. "18 of the coloring books" when the problem says he buys 6), or when the
instance has no such group at all (the size cannot be read). For every study template whose
clause is partitive, `REFERENTS` gives the group the clause refers to and how its size is read
from (or implied by) the instance's question; the check requires d < size on every instance.
Where the clause could refer to more than one group (e.g. Viktor's or Alejandro's black paint
tubes), the smallest candidate is used. A group whose size the problem never states or implies
(e.g. "his coworkers") is not a counted story quantity; the check passes it and says so. A
partitive clause without an entry in `REFERENTS` fails, so a new partitive clause cannot pass
unexamined.

2. Grounding checks (`grounding_check`): see below.
"""
from __future__ import annotations

import re
from fractions import Fraction

PARTITIVE_RE = re.compile(r"\{d\} of\b")

_UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * (i + 2) for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
_WORDNUM = (r"(?:(?:" + "|".join(_TENS) + r")(?:[ -](?:" + "|".join(list(_UNITS)[1:10]) + r"))?|"
            + "|".join(sorted(_UNITS, key=len, reverse=True)) + r")")
N = rf"(\d[\d,]*(?:\.\d+)?|{_WORDNUM}|a dozen|a|an)"          # one number token, as a group
_MULT = {"twice": 2, "double": 2, "thrice": 3, "triple": 3, "quadruple": 4, "half": Fraction(1, 2)}


def to_num(tok: str) -> Fraction:
    t = tok.strip().lower().replace(",", "")
    if re.fullmatch(r"\d+(?:\.\d+)?", t):
        return Fraction(t)
    if t in ("a", "an"):
        return Fraction(1)
    if t == "a dozen":
        return Fraction(12)
    if t in _UNITS:
        return Fraction(_UNITS[t])
    parts = re.split(r"[ -]", t)
    if parts[0] in _TENS:
        return Fraction(_TENS[parts[0]] + (_UNITS[parts[1]] if len(parts) > 1 else 0))
    raise ValueError(f"not a number: {tok!r}")


def mult(tok: str) -> Fraction:
    """'twice' -> 2, 'three times' -> 3, '1/3' -> 1/3, 'two-thirds' etc. are not needed here."""
    t = tok.strip().lower()
    if t in _MULT:
        return Fraction(_MULT[t])
    m = re.fullmatch(rf"{N} times", t)
    if m:
        return to_num(m.group(1))
    m = re.fullmatch(r"(\d+)/(\d+)", t)
    if m:
        return Fraction(int(m.group(1)), int(m.group(2)))
    raise ValueError(f"not a multiplier: {tok!r}")


def grab(q: str, pattern: str, k: int = 1) -> Fraction:
    m = re.search(pattern, q, re.I)
    if not m:
        raise LookupError(f"pattern not found: {pattern!r}")
    return to_num(m.group(k))


def grab_all(q: str, pattern: str) -> tuple[str, ...]:
    m = re.search(pattern, q, re.I)
    if not m:
        raise LookupError(f"pattern not found: {pattern!r}")
    return m.groups()


MULT = r"(twice|double|thrice|triple|quadruple|\w+ times|\d+/\d+)"

# template -> (the group the clause names, size of that group in the instance; None = never stated)
REFERENTS: dict[int, tuple[str, object]] = {
    3: ("the students", lambda q: grab(q, rf"of {N} students")),
    4: ("the kernels (at least the kernels that pop)", lambda q, a: a),
    6: ("the oranges on each square yard", lambda q: grab(q, rf"{N} oranges per square yard")),
    7: ("the alpacas (number never stated)", None),
    8: ("the balls", lambda q: grab(q, rf"juggle {N} balls")),
    12: ("the crackers", lambda q: grab(q, rf"has {N} crackers")),
    13: ("the plants (new plants + plants on the ledges)",
         lambda q: grab(q, rf"received {N} new") + grab(q, rf"has {N} potted plants on each")
         * grab(q, rf"each of the {N} window")),
    16: ("the boxes sold on Tuesday (brownie + orange pie)",
         lambda q: (lambda g: to_num(g[0]) + to_num(g[3]) - to_num(g[1]))(
             grab_all(q, rf"On Tuesday, he sold {N} boxes of [\w ]+? and {N} fewer boxes .*?"
                         rf"On Wednesday, he sold {N} more boxes of [\w ]+? than on Tuesday and {N} boxes"))),
    20: ("the pineapples picked on Thursday", lambda q: grab(q, rf"picks {N} \w+ on Thursday")),
    22: ("the spoons in Emma's new package (the unknown asked for)", lambda q, a: a),
    27: ("the emails that need a response",
         lambda q: grab(q, rf"receives {N} emails") * (1 - grab(q, r"(\d+(?:\.\d+)?)% of those") / 100)),
    33: ("the signatures in the smaller of the two books",
         lambda q: min(grab(q, rf"has {N} signatures in her book"), grab(q, rf"and \w+ has {N}"))),
    36: ("the tarts", lambda q: 12 * grab(q, rf"{N} dozen tarts")),
    37: ("the pieces of lasagna", lambda q: grab(q, rf"has {N} pieces")),
    39: ("the floors", lambda q: grab(q, rf"with {N} floors")),
    41: ("the orange tiles",
         lambda q: (lambda g: to_num(g[0]) * mult(g[1]))(grab_all(q, rf"made out of {N} \w+ tiles, (\w+(?: times)?) as many"))),
    43: ("the grasshoppers", lambda q: grab(q, rf"there were {N} grasshoppers")),
    44: ("the girls", lambda q: grab(q, rf"There are {N} girls")),
    45: ("the flowers she already owns", lambda q: grab(q, rf"already owns {N} flowers")),
    47: ("the eggs that Quinn found", lambda q: grab(q, rf"Quinn finds {N} eggs")),
    48: ("the mushrooms (the smallest stated batch)",
         lambda q: min(grab(q, rf"peel {N} mushrooms"), grab(q, rf"saute {N} mushrooms"), grab(q, rf"saute {N} mushrooms\?"))),
    51: ("the rose bundles", lambda q: grab(q, rf"{N} bundles of roses")),
    52: ("the coloring books", lambda q: grab(q, rf"buys {N} coloring books")),
    54: ("the slices", lambda q: grab(q, rf"into {N} pieces")),
    56: ("the cartons", lambda q: grab(q, rf"drinks {N} cartons")),
    57: ("the black paint tubes (the smaller holding: Alejandro's)",
         lambda q: min(grab(q, rf"has {N} tubes of black paint"),
                       grab(q, rf"has {N} tubes of black paint") * mult(grab_all(q, rf"has {MULT} as many tubes of black")[0]))),
    58: ("the trolls", lambda q: grab(q, rf"can {N} trolls")),
    60: ("his coworkers (number never stated)", None),
    61: ("the pebbles in the bag", lambda q: grab(q, rf"with {N} inside")),
    62: ("the purple cards", lambda q: grab(q, rf"there are {N} purple cards")),
    63: ("his marbles", lambda q: grab(q, rf"has {N} marbles")),
    64: ("the jars", lambda q: grab(q, rf"has {N} jars")),
    68: ("the chess puzzles", lambda q: grab(q, rf"and {N} chess puzzles")),
    71: ("the alpacas", lambda q: grab(q, rf"there are {N} alpacas")),
    72: ("his pen pals (the ones he still has)",
         lambda q: grab(q, rf"pal with {N} people") - grab(q, rf"with {N} of them")),
    74: ("the girls", lambda q: grab(q, rf"{N} of them are girls")),
    79: ("the pieces in the first puzzle", lambda q: grab(q, rf"of a {N} piece puzzle")),
    81: ("the coins", lambda q: grab(q, rf"hid {N} coins")),
    82: ("the engineers", lambda q: grab(q, rf"and {N} engineers")),
    83: ("the lemon brownies", lambda q: grab(q, rf"have {N} lemon brownies")),
    91: ("the bananas (the bananas first in the house)", lambda q: grab(q, rf"total of {N} bananas")),
    92: ("the goblins that Jin slew", lambda q: grab(q, rf"slew {N} goblins")),
    94: ("the applicants", lambda q: grab(q, rf"^{N} people apply")),
    95: ("the badminton players",
         lambda q: grab(q, rf"There are {N} students playing") * mult(grab_all(q, rf"and {MULT} that number playing")[0])),
    99: ("the eggs she buys (total eggs eaten)",
         lambda q: grab(q, rf"eats {N} eggs a day for {N}", 1) * grab(q, rf"eats {N} eggs a day for {N}", 2)),
}


def referent_size(template_id: int, question: str, answer) -> Fraction | None:
    """Size of the group a partitive clause names, in this instance (None: never stated)."""
    desc, fn = REFERENTS[template_id]
    if fn is None:
        return None
    return Fraction(fn(question, Fraction(answer))) if fn.__code__.co_argcount == 2 else Fraction(fn(question))


def partitive_check(template_id: int, clause_template: str, question: str, answer, d: int) -> dict:
    """{'partitive', 'group', 'size', 'ok', 'note'} for one instance."""
    if not PARTITIVE_RE.search(clause_template):
        return {"partitive": False, "group": None, "size": None, "ok": True, "note": "not partitive"}
    if template_id not in REFERENTS:
        return {"partitive": True, "group": None, "size": None, "ok": False,
                "note": "partitive clause with no referent entry in harness/partitive.py"}
    desc = REFERENTS[template_id][0]
    try:
        size = referent_size(template_id, question, answer)
    except (LookupError, ValueError) as e:
        return {"partitive": True, "group": desc, "size": None, "ok": False, "note": f"size not read: {e}"}
    if size is None:
        return {"partitive": True, "group": desc, "size": None, "ok": True, "note": "group size never stated"}
    return {"partitive": True, "group": desc, "size": size, "ok": d < size,
            "note": "" if d < size else f"d={d} is not smaller than {desc} = {size}"}


# 2. Grounding checks (`grounding_check`), on all instances of a template at once:
#    variable_entity   no content word of the clause appears in some instances but not others
#                      (such a word is a template variable: "the oranges" in a pomegranate
#                      instance names something the problem does not have)
#    pronoun_gender    he/his/him (she/her) in the clause needs he/his/him (she/her) in the
#                      question of every instance
#    no_forward_ref    a content word the clause shares with the question appears before the
#                      insertion point in every instance (the clause does not mention a story
#                      entity before the problem introduces it)
STOP = set("""the a an of on in at to is are was were has have had and or for with that this its it
his her their she he they them him also other some each than more less bit slightly little still same
different new old kind kinds total all by from can be will been there these those own owns one ones
""".split())
_WORD = re.compile(r"[A-Za-z]+(?:'[a-z]+)?")
_HE, _SHE = {"he", "his", "him", "himself"}, {"she", "her", "hers", "herself"}


def stem(w: str) -> str:
    """Crude plural/possessive stem: pieces -> piece, berries -> berry, boxes -> box."""
    w = w.lower()
    if w.endswith("'s"):
        w = w[:-2]
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if re.search(r"(?:ss|x|z|ch|sh)es$", w):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def content_words(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.replace("{d}", "")) if w.lower() not in STOP]


def _stems(text: str) -> set[str]:
    return {stem(w) for w in _WORD.findall(text)}


def grounding_check(clause_template: str, questions: list[str], insert) -> dict:
    """`questions`: the instances' clean questions; `insert(q, clause)`: the insertion rule.
    Returns {check: ok} and notes."""
    n = len(questions)
    q_stems = [_stems(q) for q in questions]
    pre_stems = [_stems(insert(q, "\x00").split("\x00")[0]) for q in questions]
    words = content_words(clause_template)
    present = {w: sum(stem(w) in s for s in q_stems) for w in words}
    variable = sorted({w for w, k in present.items() if 0 < k < n})
    forward = sorted({w for w, k in present.items() if k == n and not all(stem(w) in s for s in pre_stems)})
    cw = {w.lower() for w in _WORD.findall(clause_template)}
    gender_bad = []
    for i, q in enumerate(questions):
        qw = {w.lower() for w in _WORD.findall(q)}
        if (cw & _HE and not qw & _HE) or (cw & _SHE and not qw & _SHE):
            gender_bad.append(i)
    notes = []
    if variable:
        notes.append("names instance-specific words " + ", ".join(f"{w} ({present[w]}/{n})" for w in variable))
    if forward:
        notes.append("mentions " + ", ".join(forward) + " before the problem introduces it")
    if gender_bad:
        notes.append(f"pronoun has no same-gender antecedent in {len(gender_bad)}/{n} instances")
    return {"variable_entity": not variable, "pronoun_gender": not gender_bad, "no_forward_ref": not forward,
            "note": "; ".join(notes)}
