"""Text normalisation shared by references and hypotheses.

Whisper emits digits ("10-4", "Unit 12"); scripts are written in spoken form
("ten four", "unit twelve"). Both sides go through ``normalize`` so the WER
measures recognition, not formatting.
"""
from __future__ import annotations

import re

_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
         "fourteen fifteen sixteen seventeen eighteen nineteen").split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def _below_100(n: int) -> str:
    if n < 20:
        return _ONES[n]
    t, o = divmod(n, 10)
    return _TENS[t] + (" " + _ONES[o] if o else "")


def number_to_words(s: str) -> str:
    """1-3 digit numbers read as cardinals; longer strings (house numbers, plates) digit by digit."""
    if len(s) == 4 and s.startswith("10") and s[2:] != "00":  # run-together ten-code: 1099 -> ten ninety nine
        return "ten " + _below_100(int(s[2:]))  # (scripts never put a 0 in house numbers, so no clash)
    if len(s) > 3 or (len(s) > 1 and s.startswith("0")):
        return " ".join(_ONES[int(c)] for c in s)
    n = int(s)
    if n < 100:
        return _below_100(n)
    h, r = divmod(n, 100)
    return _ONES[h] + " hundred" + (" " + _below_100(r) if r else "")


def normalize(text: str) -> str:
    t = text.lower().replace("&", " and ")
    t = re.sub(r"(?<=\d),(?=\d{3})", "", t)  # 1,200 -> 1200
    t = re.sub(r"\d+", lambda m: " " + number_to_words(m.group()) + " ", t)
    t = re.sub(r"[-/]", " ", t)
    t = re.sub(r"[^a-z' ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()
