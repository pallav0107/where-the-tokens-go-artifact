"""Official HotpotQA answer normalisation, exact match and token F1."""
from __future__ import annotations

import re
import string
from collections import Counter

_YES_NO = ("yes", "no", "noanswer")


def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in string.punctuation)

    return " ".join(remove_articles(remove_punc(str(s).lower())).split())


def exact_match(pred: str, gold: str) -> int:
    return int(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred), normalize_answer(gold)
    if p in _YES_NO or g in _YES_NO:  # yes/no/noanswer mismatch -> 0
        return 1.0 if p == g else 0.0
    pt, gt = p.split(), g.split()
    same = sum((Counter(pt) & Counter(gt)).values())
    if same == 0:
        return 0.0
    precision, recall = same / len(pt), same / len(gt)
    return 2 * precision * recall / (precision + recall)
