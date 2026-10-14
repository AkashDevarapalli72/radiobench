from radiobench.metrics import keyword_score, wer
from radiobench.normalize import normalize
from radiobench.postcorrect import PostCorrector
from radiobench import scripts


def test_normalize_numbers_and_codes():
    assert normalize("10-4, Unit 12.") == "ten four unit twelve"
    assert normalize("1234 Oak St") == "one two three four oak st"
    assert normalize("10-33") == "ten thirty three"
    assert normalize("1099") == normalize("10-99") == "ten ninety nine"


def test_wer_ignores_formatting():
    assert wer(["ten four unit twelve"], ["10-4, Unit 12"]) == 0.0


def test_scripts_labels_consistent():
    rows = scripts.generate(200, seed=1)
    assert len({r["id"] for r in rows}) == 200
    for r in rows:
        assert set(scripts.find_alerts(r["text"])) == set(r["alerts"])
    assert any(r["alerts"] for r in rows) and any(not r["alerts"] for r in rows)


def test_keyword_scoring_counts_miss_and_false_alarm_separately():
    refs = [["shots fired"], [], ["officer down"]]
    hyps = ["shots fired on oak", "we have shots fired", "officer dawn"]
    s = keyword_score(refs, hyps, [3600, 3600, 3600])
    assert (s.tp, s.fn, s.fp) == (1, 1, 1)
    assert s.recall == 0.5 and s.false_alerts_per_hour == 1 / 3


def test_postcorrect_fixes_near_misses_and_leaves_short_words():
    pc = PostCorrector()
    assert pc.correct("report of shot fired at harrison boulevard") == "report of shots fired at harrison boulevard"
    assert "structure fire" in pc.correct("structure fyre on oak street")
    assert pc.correct("the cat sat") == "the cat sat"
