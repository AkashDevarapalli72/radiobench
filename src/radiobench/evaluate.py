"""Run every system across every condition and write results.csv.

Config (YAML) example: configs/default.yaml. Transcripts are cached per system
so interrupted runs resume and re-scoring is instant.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from . import lexicon
from .metrics import keyword_score, wer
from .postcorrect import PostCorrector
from .scripts import read_jsonl


def load_manifest(cfg: dict) -> list[dict]:
    rows = []
    for m in cfg["manifests"]:
        rows += read_jsonl(m)
    conds = set(cfg["conditions"])
    rows = [r for r in rows if r["condition"] in conds]
    lim = cfg.get("limit")
    if lim:  # first `lim` utterances per source, keeping every condition of each
        keep: dict[str, list[str]] = {}
        for r in rows:
            u = keep.setdefault(r["source"], [])
            if r["utt"] not in u and len(u) < lim:
                u.append(r["utt"])
        rows = [r for r in rows if r["utt"] in keep[r["source"]]]
    return rows


def transcribe_all(system: dict, rows: list[dict], out_dir: Path) -> dict[str, dict]:
    cache = out_dir / "transcripts" / f"{system['name']}.jsonl"
    cache.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if cache.exists():
        done = {r["id"]: r for r in map(json.loads, cache.read_text(encoding="utf-8").splitlines())}
    todo = [r for r in rows if r["id"] not in done]
    if todo:
        from .asr import Recognizer
        rec = Recognizer(system["size"], system.get("device", "cpu"), system.get("compute_type"),
                         prompt=lexicon.build_prompt() if system.get("prompt") else None,
                         beam_size=system.get("beam_size", 5))
        with open(cache, "a", encoding="utf-8") as f:
            for k, r in enumerate(todo, 1):
                t = rec.transcribe(r["audio"])
                rec_row = {"id": r["id"], "hyp": t.text, "latency_s": t.latency_s, "duration_s": t.duration_s}
                done[r["id"]] = rec_row
                f.write(json.dumps(rec_row) + "\n")
                f.flush()
                if k % 25 == 0:
                    print(f"  {system['name']}: {k}/{len(todo)}")
    return done


def score(rows: list[dict], hyps: dict[str, dict], system: str, post: PostCorrector | None) -> list[dict]:
    df = pd.DataFrame(rows)
    out = []
    for (source, cond), g in df.groupby(["source", "condition"], sort=False):
        recs = [hyps[i] for i in g["id"]]
        texts = [r["hyp"] for r in recs]
        if post:
            texts = [post.correct(t) for t in texts]
        entry = {"system": system + ("+fuzzy" if post else ""), "source": source, "condition": cond,
                 "n_utts": len(g), "wer": wer(list(g["text"]), texts),
                 "rtf": sum(r["latency_s"] for r in recs) / sum(r["duration_s"] for r in recs),
                 "latency_s": sum(r["latency_s"] for r in recs) / len(recs)}
        if source == "radio":
            ks = keyword_score(list(g["alerts"]), texts, list(g["duration"]))
            entry.update(kw_recall=ks.recall, kw_precision=ks.precision,
                         false_alerts_per_hour=ks.false_alerts_per_hour, kw_tp=ks.tp, kw_fn=ks.fn, kw_fp=ks.fp)
        out.append(entry)
    return out


def run(cfg_path: str) -> Path:
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(Path(cfg_path).read_text())
    rows = load_manifest(cfg)
    results = []
    for system in cfg["systems"]:
        print(f"== {system['name']}")
        hyps = transcribe_all(system, rows, out_dir)
        results += score(rows, hyps, system["name"], None)
        if cfg.get("fuzzy_post", True):
            results += score(rows, hyps, system["name"], PostCorrector())
    path = out_dir / "results.csv"
    pd.DataFrame(results).to_csv(path, index=False)
    print(f"wrote {path}")
    return path
