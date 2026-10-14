"""WER, keyword metrics and latency."""
from __future__ import annotations

from dataclasses import dataclass

import jiwer

from . import lexicon as L
from .normalize import normalize
from .scripts import find_alerts


def wer(refs: list[str], hyps: list[str]) -> float:
    pairs = [(normalize(r), normalize(h)) for r, h in zip(refs, hyps)]
    pairs = [(r, h) for r, h in pairs if r]  # jiwer rejects empty references
    if not pairs:
        return float("nan")
    r, h = zip(*pairs)
    return jiwer.wer(list(r), [x if x else "<empty>" for x in h])


@dataclass
class KeywordScore:
    tp: int = 0
    fn: int = 0
    fp: int = 0
    hours: float = 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else float("nan")

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else float("nan")

    @property
    def false_alerts_per_hour(self) -> float:
        return self.fp / self.hours if self.hours else float("nan")


def keyword_score(ref_alerts: list[list[str]], hyps: list[str], durations_s: list[float],
                  phrases: list[str] = L.ALERT_PHRASES) -> KeywordScore:
    """Utterance-level detection: a phrase is 'detected' if it occurs in the hypothesis.

    A phrase in the hypothesis that is not in the reference is a false alert; a miss is a
    reference phrase absent from the hypothesis. They are reported separately because
    their operational costs differ.
    """
    s = KeywordScore(hours=sum(durations_s) / 3600)
    for ref, hyp in zip(ref_alerts, hyps):
        got = set(find_alerts(hyp, phrases))
        want = set(ref)
        s.tp += len(got & want)
        s.fn += len(want - got)
        s.fp += len(got - want)
    return s
