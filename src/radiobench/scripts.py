"""Generate public-safety-style radio / 911 scripts with keyword labels."""
from __future__ import annotations

import json
import random
from pathlib import Path

from . import lexicon as L
from .normalize import normalize

_NUMS = "one two three four five six seven eight nine ten eleven twelve fifteen twenty".split()


def _unit(r):  # "Adam twelve"
    return f"{r.choice(L.UNIT_PREFIXES)} {r.choice(_NUMS)}"


def _addr(r):
    return f"{' '.join(r.choice(_NUMS[:9]) for _ in range(r.choice([3, 4])))} {r.choice(L.STREETS)}"


def _cross(r):
    a, b = r.sample(L.STREETS, 2)
    return f"{a} and {b}"


ALERT_TEMPLATES = [
    "{unit} respond to {addr}, report of {alert}",
    "{unit} {code}, {alert} at {cross}, request backup",
    "all units, {alert} reported near {cross}, use caution",
    "dispatch, {unit}, we have {alert} at {addr}, send additional units",
    "{unit} to dispatch, {code}, {alert} on {street}",
    "caller reports {alert} at {addr}, please hurry",
    "nine one one what is your emergency, there has been {alert} at {addr}",
]
ROUTINE_TEMPLATES = [
    "{unit} {code}, {routine} at {addr}",
    "dispatch to {unit}, what is your {code}",
    "{unit} show me {code} at {cross}",
    "{unit} responding to {routine} on {street}, {code}",
    "nine one one what is your emergency, I would like to report a {routine} at {addr}",
]


def generate(n: int, seed: int = 0, alert_fraction: float = 0.55) -> list[dict]:
    r = random.Random(seed)
    out = []
    for i in range(n):
        has_alert = r.random() < alert_fraction
        alerts = [r.choice(L.ALERT_PHRASES)] if has_alert else []
        if has_alert and r.random() < 0.15:  # occasional second alert
            alerts.append(r.choice([a for a in L.ALERT_PHRASES if a not in alerts]))
        slots = dict(unit=_unit(r), addr=_addr(r), cross=_cross(r), street=r.choice(L.STREETS),
                     code=r.choice(list(L.TEN_CODES)), routine=r.choice(L.ROUTINE_PHRASES))
        if has_alert:
            text = r.choice(ALERT_TEMPLATES).format(alert=alerts[0], **slots)
            if len(alerts) > 1:
                text += f", also {alerts[1]}"
        else:
            text = r.choice(ROUTINE_TEMPLATES).format(**slots)
        out.append({"id": f"rs{i:04d}", "text": text, "alerts": alerts,
                    "kind": "911" if "nine one one" in text else "dispatch"})
    return out


def find_alerts(text: str, phrases: list[str] = L.ALERT_PHRASES) -> list[str]:
    t = f" {normalize(text)} "
    return [p for p in phrases if f" {p} " in t]


def write_jsonl(rows: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
