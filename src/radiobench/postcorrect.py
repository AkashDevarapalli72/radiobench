"""Fuzzy post-correction of ASR output against the domain lexicon.

Slides 1-3 word windows over the normalised hypothesis; if a window is close
(but not identical) to a lexicon term it is replaced. Conservative by design:
short windows are never touched, because every wrong "correction" into an alert
phrase is a false alarm.
"""
from __future__ import annotations

from rapidfuzz import fuzz, process

from . import lexicon as L
from .normalize import normalize


class PostCorrector:
    def __init__(self, terms: list[str] | None = None, threshold: float = 86.0, min_chars: int = 7):
        self.terms = sorted({normalize(t) for t in (terms or L.biasing_terms())})
        self.known = set(self.terms)
        self.by_len = {n: [t for t in self.terms if len(t.split()) == n] for n in (1, 2, 3)}
        self.threshold = threshold
        self.min_chars = min_chars

    def _match(self, span: str) -> str | None:
        """Return the replacement for ``span`` (the span itself if already a lexicon term), else None."""
        if span in self.known:
            return span
        if len(span) < self.min_chars:
            return None
        # compare only against terms with the same word count so a window can't swallow extra words
        hit = process.extractOne(span, self.by_len.get(len(span.split()), []), scorer=fuzz.ratio, score_cutoff=self.threshold)
        return hit[0] if hit else None

    def correct(self, text: str) -> str:
        words = normalize(text).split()
        out, i = [], 0
        while i < len(words):
            for n in (3, 2, 1):  # prefer the longest span
                if i + n > len(words):
                    continue
                repl = self._match(" ".join(words[i: i + n]))
                if repl is not None:
                    out.append(repl)
                    i += n
                    break
            else:
                out.append(words[i])
                i += 1
        return " ".join(out)
