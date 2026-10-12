"""Shared data structures for problems loaded from any benchmark."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Problem:
    id: str
    dataset: str
    split: str
    question: str
    gold_answer: str
