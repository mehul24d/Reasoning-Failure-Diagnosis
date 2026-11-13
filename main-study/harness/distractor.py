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

# Pre-review rewrites (2026-11-13, prereg/S2.md A18, DECISIONS D40): template -> (clause, reason).
# Reasons: "prescreen" = FAIL/UNSURE on irrelevant, natural or no_hint in the AI pre-screen
# (docs/review/distractor_clauses_ai_prescreen.csv, note quoted); "partitive" = d is not smaller
# than the group the clause names on some instance 1-16; "grounding" = the clause names a word
# that is a template variable (absent from some instances), a pronoun without a same-gender
# antecedent, or a story word before the problem introduces it (harness/instance_checks.py);
# "plausibility" = implausible for some d in 2-19. Failing partitive clauses become non-partitive.
PREREVIEW_REWRITES: dict[int, tuple[str, str]] = {
    3: ("{d} of the students are left-handed.",
        "prescreen: chess could be read as another hobby group, shrinking 'the rest'"),
    5: ("The sugar comes from a farm {d} kilometers away.",
        "prescreen: 'cups' is also the unit of water; spare cups might be read into the total"),
    6: ("The field is {d} kilometers from the nearest town.",
        "partitive + grounding: the fruit varies (oranges in 2/16 instances); 19 > 6 per square yard in one"),
    7: ("He lives {d} kilometers from the feed store.",
        "grounding: the animals vary (alpacas in 4/16); the clause also had a second number ('ten')"),
    12: ("He is expecting {d} guests tonight.",
         "prescreen: 'but can still be placed' hints that d is irrelevant; grounding: crackers in 2/16"),
    14: ("They pass {d} traffic lights on the way.", "grounding: the landmark varies (cinema in 9/16)"),
    16: ("He has helped with this fundraiser for {d} years.", "grounding: the days vary (Tuesday in 7/16)"),
    20: ("He started this job {d} weeks ago.",
         "prescreen: tense mismatch; grounding: the fruit (2/16) and the days (7/16) vary"),
    22: ("She had invited {d} friends.",
         "prescreen: 'the new package' is ambiguous and bounds the unknown; partitive: d >= the answer on 2/16"),
    26: ("Their kitchen has {d} cupboards.", "grounding: 'family' first appears in the question"),
    28: ("Each new patient waits about {d} minutes to be seen.", "grounding: the facility varies (clinic in 3/16)"),
    30: ("The store opened {d} years ago.", "grounding: the item varies (washers in 4/16)"),
    31: ("They live {d} kilometers apart.",
         "prescreen: 'their grandmother' assumes the two are related, which the problem never says"),
    32: ("She has been searching for {d} months.", "grounding: the property type varies (townhouse in 6/16)"),
    33: ("Their school is {d} kilometers from their home.",
         "prescreen: 'the first book' is undefined; partitive: d >= the smaller book on 2/16"),
    34: ("He had lived there for {d} years.", "grounding: the property type varies (cottage in 5/16)"),
    35: ("He started raising money {d} days ago.", "grounding: the event varies (conference in 3/16)"),
    36: ("She was there for {d} minutes.", "grounding: the venue and the treats vary (tarts in 6/16)"),
    37: ("It was made {d} hours ago.", "grounding: the dish and its pieces vary (pieces in 5/16)"),
    38: ("The baker has worked there for {d} years.", "grounding: the venue varies (cafe in 3/16)"),
    39: ("They have done this job for {d} years.", "grounding: floors or rooms vary (floors in 10/16)"),
    40: ("Her vacation lasted {d} days.", "grounding: the candy (8/16) and the store (9/16) vary"),
    41: ("It took {d} minutes to build.",
         "partitive + grounding: colours and objects vary (orange tiles in 7/16); 19 > 14 orange tiles in one"),
    42: ("The store is {d} minutes from his house.", "grounding: the product varies (tea in 4/16)"),
    43: ("It took them {d} minutes to get there.", "grounding: the insects vary (grasshoppers in 7/16)"),
    44: ("There are also {d} parents watching.", "partitive: d >= the number of girls on 2/16"),
    45: ("She has had the flower bed for {d} years.",
         "prescreen + partitive: d >= the flowers she owns on 8/16 instances"),
    46: ("The venue has {d} entrances.", "grounding: the venue varies (auditorium in 3/16)"),
    47: ("The hunt started {d} minutes ago.",
         "prescreen: tense mismatch; grounding: Quinn appears in 1/16 instances"),
    48: ("She has been cooking for {d} years.",
         "prescreen: 'the mushrooms' before they are introduced; grounding: mushrooms in 3/16; partitive on 2/16"),
    50: ("It has {d} windows.", "grounding: 'room' first appears in the question in some instances"),
    51: ("He has run the stand for {d} years.", "grounding: the bundles vary (bundles in 5/16)"),
    52: ("He has been saving for {d} weeks.",
         "prescreen + partitive: d >= the coloring books bought on 4/16; grounding: the item varies (5/16)"),
    53: ("They have known each other for {d} years.", "grounding: the game varies (darts or basketball)"),
    56: ("He bought the drinks {d} days ago.",
         "prescreen + partitive: d >= the cartons drunk on 6/16; 'a different store' has no first store"),
    57: ("They both started painting {d} years ago.",
         "prescreen: unclear whose black tubes; grounding: the colours vary (black in 4/16)"),
    58: ("The mine is {d} kilometers from the nearest village.",
         "prescreen: 'the trolls' before any group is introduced; grounding: trolls in 7/16"),
    59: ("I bought both drinks {d} days ago.",
         "prescreen: no bottle is mentioned; grounding: the drink varies (grape in 11/16)"),
    61: ("The walk home took {d} minutes.", "grounding: the objects vary (pebbles in 6/16)"),
    62: ("The set was printed {d} years ago.", "grounding: the colours vary (purple in 10/16)"),
    63: ("They live {d} kilometers from school.", "grounding: the objects vary (marbles in 2/16)"),
    64: ("She bought the jars {d} days ago.", "prescreen + partitive: d >= the jars on 10/16 instances"),
    65: ("She has worked for herself for {d} years.", "grounding: the professional varies (bookkeeper in 5/16)"),
    67: ("He is {d} kilometers from home.", "grounding: the vehicle varies (van in 5/16)"),
    68: ("Her best friend played for {d} hours over the same period.",
         "prescreen + partitive: d >= the chess puzzles on 5/16; grounding: the puzzles vary"),
    70: ("He has {d} coworkers.", "grounding: the trade varies (painter or electrician: 'brushes')"),
    71: ("He has kept livestock for {d} years.", "grounding: the animals vary (alpacas in 4/16)"),
    72: ("He lives {d} kilometers from the nearest post office.",
         "prescreen + partitive: d >= his remaining pen pals on 14/16 instances"),
    75: ("The balls arrived {d} days ago.", "grounding: 'the shop' is absent from some instances (6/16)"),
    77: ("It is due in {d} days.", "grounding: the document varies (thesis in 7/16)"),
    81: ("The twins arrived {d} minutes late.", "grounding: the objects vary (coins in 2/16)"),
    82: ("The venue had {d} windows.", "grounding: the professions vary (engineers in 4/16)"),
    83: ("They started baking {d} hours ago.", "grounding: the baked goods vary (brownies in 3/16)"),
    86: ("Her team practices {d} hours a week.", "grounding: no store is mentioned; jerseys in 8/16"),
    88: ("She owns {d} pairs of sneakers.", "grounding: 'running' first appears in the question"),
    89: ("He drives {d} kilometers in that time.", "grounding: the vehicle varies (car in 3/16)"),
    90: ("Each of them posts {d} times a week.", "grounding: the names and the platform vary (3/16)"),
    91: ("He walks {d} minutes to get to the store.", "grounding: the fruit varies (bananas in 7/16)"),
    92: ("They fought for {d} hours.", "grounding: the names and creatures vary (goblins in 1/16)"),
    93: ("The swimming spot is a {d}-minute bike ride from home.",
         "prescreen: no pool is mentioned and 16 m is implausibly deep"),
    95: ("The school employs {d} coaches.",
         "partitive: d >= the badminton players on 1/16; grounding: the sports vary (9/16)"),
    96: ("The party venue is {d} kilometers away.", "plausibility: a party lasting up to 19 hours"),
    98: ("It was first reported by a team of {d} scientists.", "grounding: virus or infection varies (4/16)"),
    99: ("The local farm has {d} hens.",
         "prescreen: the clause opens the problem, so 'she' and 'the eggs' precede Nia; no 'she' in 16/16"),
}

# A19 (D41): temptation rating of the rewritten clauses, on the pre-screen's scale (PASS = the
# number has a plausible use in the problem, WEAK = little pull, UNSURE). The pre-screen rated the
# DRAFT clauses; these rate the clauses actually used. A covariate only, never an exclusion rule.
REWRITE_TEMPTATION_AI: dict[int, str] = {
    3: "PASS", 5: "WEAK", 6: "WEAK", 7: "UNSURE", 12: "PASS", 14: "WEAK", 16: "WEAK", 20: "WEAK",
    22: "UNSURE", 26: "WEAK", 28: "WEAK", 30: "WEAK", 31: "UNSURE", 32: "WEAK", 33: "WEAK", 34: "WEAK",
    35: "WEAK", 36: "WEAK", 37: "UNSURE", 38: "WEAK", 39: "WEAK", 40: "WEAK", 41: "WEAK", 42: "WEAK",
    43: "WEAK", 44: "PASS", 45: "WEAK", 46: "WEAK", 47: "WEAK", 48: "WEAK", 50: "WEAK", 51: "WEAK",
    52: "UNSURE", 53: "WEAK", 56: "WEAK", 57: "WEAK", 58: "WEAK", 59: "WEAK", 61: "WEAK", 62: "WEAK",
    63: "WEAK", 64: "WEAK", 65: "WEAK", 67: "UNSURE", 68: "PASS", 70: "UNSURE", 71: "WEAK", 72: "WEAK",
    75: "WEAK", 77: "UNSURE", 81: "WEAK", 82: "WEAK", 83: "UNSURE", 86: "WEAK", 88: "WEAK", 89: "UNSURE",
    90: "UNSURE", 91: "WEAK", 92: "WEAK", 93: "PASS", 95: "PASS", 96: "WEAK", 98: "PASS", 99: "UNSURE",
}
TEMPTATION = REPO_ROOT / "data" / "distractor_temptation.json"

# the clauses the human reviewers see (docs/review/human_review_final.csv)
FINAL_CLAUSES: dict[int, str] = {t: PREREVIEW_REWRITES.get(t, (c, ""))[0] for t, c in DRAFT_CLAUSES.items()}

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
