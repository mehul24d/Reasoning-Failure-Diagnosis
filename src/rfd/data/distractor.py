"""No-op distractor variants of GSM-Symbolic problems (GSM-NoOp style).

GSM-NoOp was never released (Cox et al. 2603.01437 also had to re-create it),
so we insert one inconsequential clause per template, immediately before the
final (question) sentence. Every clause carries an explicit distractor-quantity
slot `{d}`; the gold program and answer are unchanged by construction.

The clauses below were written by the AI assistant (Claude) in this project
session, one per template, aiming for "plausible but irrelevant" (the NoOp
recipe); they have NOT yet been independently reviewed by a human — see
docs/DECISIONS.md D6.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from rfd.parsing.equations import close

NOOP_CLAUSES: dict[int, str] = {
    0: "The day before, {d} of the remoras had briefly swum away and come back.",
    1: "At its densest point, the fog is {d} feet thick.",
    2: "The die had already been rolled {d} times earlier that day.",
    3: "{d} of the students also play chess on weekends.",
    4: "{d} of the kernels are slightly larger than the others.",
    5: "She keeps {d} spare cups in the cupboard.",
    6: "{d} of the oranges on each square yard are a little smaller than average.",
    7: "{d} of the alpacas are older than ten years.",
    8: "{d} of the balls are slightly heavier than the rest.",
    9: "{d} of the blocks are painted blue.",
    10: "The hotel has {d} floors.",
    12: "{d} of the crackers are slightly chipped but can still be placed.",
    13: "{d} of the plants have purple flowers.",
    14: "The cinema is showing {d} different movies today.",
    15: "{d} of the spotted puppies have brown spots.",
    16: "{d} of the boxes sold on Tuesday were wrapped in red paper.",
    17: "There are {d} benches along the beach part of the trail.",
    18: "The store had {d} other customers that day.",
    19: "The quiz was written by {d} different teachers.",
    20: "{d} of the pineapples picked on Thursday were slightly smaller than the rest.",
    21: "His favorite show has {d} seasons.",
    22: "{d} of the spoons in the new package were blue.",
    23: "She has knitted {d} scarves for other relatives in the past.",
    24: "The label on the bag is printed in {d} languages.",
    25: "The class has {d} computers at the back of the room.",
    26: "The family owns {d} plates.",
    27: "{d} of the emails that need a response come from the same client.",
    28: "The clinic has {d} doctors on staff.",
    29: "Their family dog is {d} years old.",
    30: "The store has {d} other kinds of washers on display.",
    31: "Their grandmother has {d} grandchildren in total.",
    32: "The townhouse has {d} windows.",
    33: "{d} of the signatures in the first book were signed in blue ink.",
    34: "The cottage had {d} rooms.",
    35: "The conference will last {d} hours.",
    36: "{d} of the tarts had chocolate on top.",
    37: "{d} of the pieces have extra cheese on them.",
    38: "The cafe has {d} tables.",
    39: "{d} of the floors have carpets instead of tiles.",
    40: "The store had {d} other kinds of candy for sale.",
    41: "{d} of the orange tiles are slightly cracked.",
    42: "The store sells {d} different brands of tea.",
    43: "{d} of the grasshoppers were bright green.",
    44: "{d} of the girls are wearing hats.",
    45: "{d} of the flowers she already owns are red.",
    46: "The auditorium has {d} exits.",
    47: "{d} of the eggs that Quinn found were painted yellow.",
    48: "{d} of the mushrooms are a bit smaller than the rest.",
    49: "Each complex has {d} parking spots for visitors.",
    51: "{d} of the rose bundles are wrapped in paper.",
    52: "{d} of the coloring books have animals on the cover.",
    53: "The dartboard they played on is {d} years old.",
    54: "{d} of the slices were thicker than the others.",
    55: "Her pencil is {d} centimeters long.",
    56: "{d} of the cartons were bought at a different store.",
    57: "{d} of the black paint tubes are almost empty.",
    58: "{d} of the trolls have red beards.",
    59: "The bottle of grape drink has {d} stickers on it.",
    60: "{d} of his coworkers also work the same shift.",
    61: "{d} of the pebbles in the bag were white.",
    62: "{d} of the purple cards are slightly bent.",
    63: "{d} of his marbles are blue.",
    64: "{d} of the jars have red lids.",
    65: "The bookkeeper has {d} other clients.",
    66: "{d} of the light white spools are almost used up.",
    67: "The van is {d} years old.",
    68: "{d} of the chess puzzles came from an old book.",
    69: "He listens to {d} songs while he exercises.",
    70: "He uses {d} brushes for each job.",
    71: "{d} of the alpacas are brown.",
    72: "{d} of his pen pals live in another country.",
    73: "The comet's tail can be {d} million kilometers long.",
    74: "{d} of the girls ride the bus to school.",
    75: "The shop also sells {d} kinds of rackets.",
    76: "{d} of the trouts had spots on their backs.",
    77: "Her thesis has {d} chapters.",
    78: "His school has {d} classrooms.",
    79: "{d} of the pieces in the first puzzle were edge pieces.",
    80: "The vending machines also sell {d} kinds of candy.",
    81: "{d} of the coins were shiny new ones.",
    82: "{d} of the engineers wore glasses.",
    83: "{d} of the lemon brownies have nuts on top.",
    84: "The company that issued the stock has {d} offices.",
    86: "The store had {d} other jerseys on display.",
    87: "The park also has {d} smaller waterslides.",
    88: "She owns {d} pairs of running shoes.",
    89: "The car has {d} seats.",
    90: "Zahra has posted {d} times on LinkedIn this month.",
    91: "{d} of the bananas were still a little green.",
    92: "{d} of the goblins that Jin slew were wearing helmets.",
    93: "The pool is {d} meters deep.",
    94: "{d} of the applicants live in another city.",
    95: "{d} of the badminton players are left-handed.",
    96: "The party will last {d} hours.",
    97: "The store has {d} other cash registers.",
    98: "The virus was discovered by a team of {d} scientists.",
    99: "{d} of the eggs she buys are brown.",
}

# values that legitimately appear as unit constants in solutions
_RESERVED = {7.0, 10.0, 12.0}
_D_RANGE = range(2, 20)
_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_ABBREV_RE = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Prof|St|Jr|Sr)\.$")


@dataclass
class DistractorVariant:
    question: str
    d: int
    clause: str


def insert_clause(question: str, clause: str) -> str:
    """Insert `clause` immediately before the final sentence of `question`."""
    boundaries = [
        m.end()
        for m in _BOUNDARY_RE.finditer(question)
        if not _ABBREV_RE.search(question[: m.start()])
    ]
    if not boundaries:
        return f"{clause} {question}"
    cut = boundaries[-1]
    return f"{question[:cut]}{clause} {question[cut:]}"


def sample_distractor_value(
    forbidden: list[float], rng: np.random.Generator, value_range: range = _D_RANGE
) -> int:
    candidates = [
        v
        for v in value_range
        if float(v) not in _RESERVED and not any(close(float(v), f) for f in forbidden)
    ]
    if not candidates:
        candidates = [v for v in range(21, 100) if not any(close(float(v), f) for f in forbidden)]
    return int(rng.choice(candidates))


def make_variant(
    template_id: int,
    question: str,
    forbidden: list[float],
    seed: int,
) -> DistractorVariant | None:
    """Distractor variant, or None if the template has no clause."""
    template = NOOP_CLAUSES.get(template_id)
    if template is None:
        return None
    rng = np.random.default_rng(seed)
    d = sample_distractor_value(forbidden, rng)
    clause = template.format(d=d)
    return DistractorVariant(question=insert_clause(question, clause), d=d, clause=clause)
