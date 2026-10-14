"""Domain lexicon: the 'gazetteer' an agency would know in advance.

All strings are in spoken form (what a person says into the radio).
"""
ALERT_PHRASES = [
    "shots fired", "officer down", "structure fire", "man down", "armed robbery",
    "active shooter", "vehicle pursuit", "gas leak", "medical emergency", "suspect on foot",
]

TEN_CODES = {  # spoken form -> meaning
    "ten four": "acknowledged", "ten twenty": "location", "ten eight": "in service",
    "ten seven": "out of service", "ten thirty three": "emergency traffic",
    "ten fifty": "accident", "ten seventy one": "shooting", "ten ninety nine": "officer needs help",
}

STREETS = [
    "Oak Street", "Maple Avenue", "Cedar Lane", "Pinecrest Drive", "Willow Court",
    "Harrison Boulevard", "Kensington Road", "Marquette Avenue", "Fairmount Street",
    "Lakeshore Drive", "Sycamore Way", "Dearborn Street", "Alder Place", "Montrose Avenue",
    "Wabash Street", "Ravenswood Road", "Lincoln Highway", "Jefferson Terrace",
    "Cottonwood Lane", "Beacon Hill Road", "Chestnut Street", "Elmhurst Boulevard",
    "Granville Avenue", "Hawthorne Circle", "Ingleside Drive", "Juniper Street",
]

UNIT_PREFIXES = ["Adam", "Baker", "Charlie", "David", "Edward", "Frank", "Henry", "Lincoln", "Mary", "Tango"]

ROUTINE_PHRASES = [
    "traffic stop", "welfare check", "noise complaint", "parking violation",
    "suspicious vehicle", "disabled vehicle", "alarm activation", "lost child",
]


def biasing_terms() -> list[str]:
    """Terms used for the Whisper prompt and the fuzzy post-corrector."""
    return ALERT_PHRASES + list(TEN_CODES) + STREETS + ROUTINE_PHRASES


def build_prompt(max_streets: int = 12) -> str:
    return ("Police and fire radio dispatch. Ten-codes, unit numbers, street names. "
            + ", ".join(p.capitalize() for p in ALERT_PHRASES[:6]) + ". "
            + "Streets: " + ", ".join(STREETS[:max_streets]) + ".")
