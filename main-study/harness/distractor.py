"""The no-op distractor condition of the S2 population (prereg/S2.md A14, A17; DECISIONS D35, D39).

Primary population (A14): GSM-Symbolic distractor failures, i.e. a problem the model solves
without the clause and fails with it. Every study instance is generated in both conditions:
`clean` (the GSM-Symbolic `main` question) and `distractor` (the same question with one no-op
clause inserted before its final sentence).

The clause table and the insertion rule are the baseline's (`src/rfd/data/distractor.py`, D6),
copied here so the Kaggle run needs nothing outside main-study/; `tests/test_distractor.py`
checks that both give identical text. Template 50 had no clause in the baseline; its clause was
drafted on 2026-11-13 (A17) and goes through the same automated checks and human review.

The pipeline never reads `DRAFT_CLAUSES` directly. It reads the reconciled table written by
`scripts/review_agreement.py` after the two-person review (`data/distractor_clauses_reviewed.json`):
kept clauses, rewritten clauses (re-checked), and dropped templates. Without that file the
`problems` stage refuses to run.
"""
from __future__ import annotations

import json
import re

import numpy as np

from .config import REPO_ROOT

BASELINE_SEED = 42                          # seed rule: 42 * 1000 + template * 100 + instance
REVIEWED = REPO_ROOT / "data" / "distractor_clauses_reviewed.json"

# one clause per study template; {d} is the distractor quantity
DRAFT_CLAUSES: dict[int, str] = {
    1: 'At its densest point, the fog is {d} feet thick.',
    3: '{d} of the students also play chess on weekends.',
    4: '{d} of the kernels are slightly larger than the others.',
    5: 'She keeps {d} spare cups in the cupboard.',
    6: '{d} of the oranges on each square yard are a little smaller than average.',
    7: '{d} of the alpacas are older than ten years.',
    8: '{d} of the balls are slightly heavier than the rest.',
    12: '{d} of the crackers are slightly chipped but can still be placed.',
    13: '{d} of the plants have purple flowers.',
    14: 'The cinema is showing {d} different movies today.',
    16: '{d} of the boxes sold on Tuesday were wrapped in red paper.',
    18: 'The store had {d} other customers that day.',
    19: 'The quiz was written by {d} different teachers.',
    20: '{d} of the pineapples picked on Thursday were slightly smaller than the rest.',
    21: 'His favorite show has {d} seasons.',
    22: '{d} of the spoons in the new package were blue.',
    23: 'She has knitted {d} scarves for other relatives in the past.',
    24: 'The label on the bag is printed in {d} languages.',
    26: 'The family owns {d} plates.',
    27: '{d} of the emails that need a response come from the same client.',
    28: 'The clinic has {d} doctors on staff.',
    29: 'Their family dog is {d} years old.',
    30: 'The store has {d} other kinds of washers on display.',
    31: 'Their grandmother has {d} grandchildren in total.',
    32: 'The townhouse has {d} windows.',
    33: '{d} of the signatures in the first book were signed in blue ink.',
    34: 'The cottage had {d} rooms.',
    35: 'The conference will last {d} hours.',
    36: '{d} of the tarts had chocolate on top.',
    37: '{d} of the pieces have extra cheese on them.',
    38: 'The cafe has {d} tables.',
    39: '{d} of the floors have carpets instead of tiles.',
    40: 'The store had {d} other kinds of candy for sale.',
    41: '{d} of the orange tiles are slightly cracked.',
    42: 'The store sells {d} different brands of tea.',
    43: '{d} of the grasshoppers were bright green.',
    44: '{d} of the girls are wearing hats.',
    45: '{d} of the flowers she already owns are red.',
    46: 'The auditorium has {d} exits.',
    47: '{d} of the eggs that Quinn found were painted yellow.',
    48: '{d} of the mushrooms are a bit smaller than the rest.',
    49: 'Each complex has {d} parking spots for visitors.',
    50: "The room has {d} windows.",                  # drafted 2026-11-13 (A17); needs the review
    51: '{d} of the rose bundles are wrapped in paper.',
    52: '{d} of the coloring books have animals on the cover.',
    53: 'The dartboard they played on is {d} years old.',
    54: '{d} of the slices were thicker than the others.',
    55: 'Her pencil is {d} centimeters long.',
    56: '{d} of the cartons were bought at a different store.',
    57: '{d} of the black paint tubes are almost empty.',
    58: '{d} of the trolls have red beards.',
    59: 'The bottle of grape drink has {d} stickers on it.',
    60: '{d} of his coworkers also work the same shift.',
    61: '{d} of the pebbles in the bag were white.',
    62: '{d} of the purple cards are slightly bent.',
    63: '{d} of his marbles are blue.',
    64: '{d} of the jars have red lids.',
    65: 'The bookkeeper has {d} other clients.',
    67: 'The van is {d} years old.',
    68: '{d} of the chess puzzles came from an old book.',
    69: 'He listens to {d} songs while he exercises.',
    70: 'He uses {d} brushes for each job.',
    71: '{d} of the alpacas are brown.',
    72: '{d} of his pen pals live in another country.',
    73: "The comet's tail can be {d} million kilometers long.",
    74: '{d} of the girls ride the bus to school.',
    75: 'The shop also sells {d} kinds of rackets.',
    77: 'Her thesis has {d} chapters.',
    78: 'His school has {d} classrooms.',
    79: '{d} of the pieces in the first puzzle were edge pieces.',
    80: 'The vending machines also sell {d} kinds of candy.',
    81: '{d} of the coins were shiny new ones.',
    82: '{d} of the engineers wore glasses.',
    83: '{d} of the lemon brownies have nuts on top.',
    84: 'The company that issued the stock has {d} offices.',
    86: 'The store had {d} other jerseys on display.',
    87: 'The park also has {d} smaller waterslides.',
    88: 'She owns {d} pairs of running shoes.',
    89: 'The car has {d} seats.',
    90: 'Zahra has posted {d} times on LinkedIn this month.',
    91: '{d} of the bananas were still a little green.',
    92: '{d} of the goblins that Jin slew were wearing helmets.',
    93: 'The pool is {d} meters deep.',
    94: '{d} of the applicants live in another city.',
    95: '{d} of the badminton players are left-handed.',
    96: 'The party will last {d} hours.',
    97: 'The store has {d} other cash registers.',
    98: 'The virus was discovered by a team of {d} scientists.',
    99: '{d} of the eggs she buys are brown.',
}

_RESERVED = {7.0, 10.0, 12.0}               # unit constants that appear in solutions
_D_RANGE = range(2, 20)
_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_ABBREV_RE = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Prof|St|Jr|Sr)\.$")


def close(x: float, y: float, rel: float = 1e-4, abs_: float = 1e-2) -> bool:
    return abs(x - y) <= max(abs_, rel * max(abs(x), abs(y)))


def insert_clause(question: str, clause: str) -> str:
    """Insert `clause` immediately before the final sentence of `question` (baseline rule)."""
    cuts = [m.end() for m in _BOUNDARY_RE.finditer(question)
            if not _ABBREV_RE.search(question[: m.start()])]
    if not cuts:
        return f"{clause} {question}"
    return f"{question[:cuts[-1]]}{clause} {question[cuts[-1]:]}"


def sample_distractor_value(forbidden: list[float], rng: np.random.Generator,
                            value_range: range = _D_RANGE) -> int:
    cand = [v for v in value_range
            if float(v) not in _RESERVED and not any(close(float(v), f) for f in forbidden)]
    if not cand:
        cand = [v for v in range(21, 100) if not any(close(float(v), f) for f in forbidden)]
    return int(rng.choice(cand))


def forbidden_values(row: dict) -> list[float]:
    """Question numbers and gold operands and results: d never equals one of them."""
    from parsing.gold import gold_program

    prog, sl = gold_program(row["question"], row["solution"], row["answer"], "gsm_symbolic")
    return [float(s.value) for s in sl] + [float(v) for st in prog.steps for v in (st.a, st.b, st.result)]


def make_variant(template_id: int, instance: int, row: dict, clause_template: str) -> dict:
    """The distractor condition of one problem: question, d, rendered clause and its char span."""
    rng = np.random.default_rng(BASELINE_SEED * 1000 + template_id * 100 + instance)
    d = sample_distractor_value(forbidden_values(row), rng)
    clause = clause_template.format(d=d)
    q = insert_clause(row["question"], clause)
    start = q.index(clause)
    return {"question": q, "d": d, "clause": clause, "distractor_span": [start, start + len(clause)]}


def reviewed_clauses(path=REVIEWED) -> dict[int, str]:
    """Clause per template after the two-person review (kept or rewritten; dropped ones absent)."""
    if not path.exists():
        raise RuntimeError(f"{path} is missing: the distractor clauses must be reviewed and reconciled "
                           "(scripts/review_agreement.py, DECISIONS D39) before any study problem is built")
    tab = json.loads(path.read_text())
    return {int(t): c["clause_template"] for t, c in tab["clauses"].items()
            if c["action"] in ("keep", "rewrite")}


def pair_rows(rows: list[dict], clauses: dict[int, str]) -> list[dict]:
    """Both conditions of every instance, clean first. `pair_id` links the two; a template with
    no reviewed clause has the clean condition only."""
    out = []
    for r in rows:
        t = int(r["template_id"].split("-")[1])
        inst = int(r["problem_id"].rsplit("-", 1)[1])
        out.append({**r, "variant": "clean", "condition": "clean", "pair_id": r["problem_id"]})
        if t in clauses:
            v = make_variant(t, inst, r, clauses[t])
            out.append({**r, "problem_id": r["problem_id"].replace("gsms-main-", "gsms-noop-", 1),
                        "variant": "distractor", "condition": "distractor",
                        "pair_id": r["problem_id"], "question": v["question"], "distractor": v["d"],
                        "clause": v["clause"], "distractor_span": v["distractor_span"]})
    return out
