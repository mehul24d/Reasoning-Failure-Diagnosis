"""Add the extracted answer and correctness to a traces JSONL (rule: parsing/answers.py)."""
from __future__ import annotations

import argparse
import json
from collections import Counter

from parsing.answers import extract_final_answer, is_correct


def score_record(rec: dict) -> dict:
    pred = extract_final_answer(rec["text"])
    rec = dict(rec)
    rec["pred_answer"] = None if pred is None else str(pred)
    rec["answer_extracted"] = pred is not None
    rec["correct"] = is_correct(pred, rec.get("gold_answer"))
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    stats = Counter()
    with open(a.traces) as fin, open(a.out, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            r = score_record(json.loads(line))
            stats["n"] += 1
            stats["correct"] += r["correct"]
            stats["no_answer"] += not r["answer_extracted"]
            stats["hit_max_tokens"] += bool(r.get("hit_max_tokens"))
            fout.write(json.dumps(r) + "\n")
    n = max(stats["n"], 1)
    print(f"traces {stats['n']}  accuracy {stats['correct']/n:.3f}  "
          f"no answer {stats['no_answer']/n:.3f}  hit max tokens {stats['hit_max_tokens']/n:.3f}")


if __name__ == "__main__":
    main()
