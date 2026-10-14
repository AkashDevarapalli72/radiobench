"""Dataset acquisition and construction.

* LibriSpeech test-clean  -> clean sanity benchmark
* ESC-50                  -> real siren / wind / traffic-like noise clips
* synthetic radio set     -> scripts -> Piper TTS -> degradation simulator
"""
from __future__ import annotations

import csv
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

from . import scripts, tts
from .degrade import ChannelConfig, NOISE_KINDS, degrade, resample

LIBRISPEECH_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"   # ~346 MB
ESC50_URL = "https://github.com/karoldvl/ESC-50/archive/master.zip"           # ~600 MB
ESC50_MAP = {"siren": "siren", "wind": "wind", "engine": "traffic", "car_horn": "traffic", "train": "traffic"}

# name -> (snr_db or None). "clean" is undegraded TTS; "channel" is the radio channel with no acoustic noise.
CONDITIONS: dict[str, float | None] = {"clean": None, "channel": None,
                                       "snr20": 20, "snr10": 10, "snr5": 5, "snr0": 0}


def channel_for(condition: str, utt_index: int, seed: int = 0) -> ChannelConfig | None:
    """Channel config for a named condition. Noise type rotates across utterances."""
    if condition == "clean":
        return None
    base = dict(band=(300, 3000), codec="gsm", clip_level=0.5, squelch_tail_ms=150,
                dropout_rate=0.3, seed=seed + utt_index)
    snr = CONDITIONS[condition]
    if snr is not None:
        base.update(noise_kind=NOISE_KINDS[utt_index % len(NOISE_KINDS)], snr_db=float(snr))
    return ChannelConfig(**base)


# ---------------------------------------------------------------- downloads

def fetch(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    print(f"downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    return dest


def download_librispeech(root: Path) -> Path:
    out = root / "LibriSpeech" / "test-clean"
    if not out.exists():
        tgz = fetch(LIBRISPEECH_URL, root / "test-clean.tar.gz")
        with tarfile.open(tgz) as t:
            t.extractall(root, filter="data")
    return out


def download_esc50(root: Path) -> Path:
    out = root / "ESC-50-master"
    if not out.exists():
        z = fetch(ESC50_URL, root / "esc50.zip")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(root)
    return out


def download_voice(voice_dir: Path) -> None:
    voice_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "piper.download_voices", tts.VOICE,
                    "--data-dir", str(voice_dir)], check=True)


# ----------------------------------------------------------------- loaders

def load_noise_bank(esc_root: Path, sr: int = 8000) -> dict[str, list[np.ndarray]]:
    bank: dict[str, list[np.ndarray]] = {k: [] for k in NOISE_KINDS}
    with open(esc_root / "meta" / "esc50.csv", newline="") as f:
        for row in csv.DictReader(f):
            kind = ESC50_MAP.get(row["category"])
            if kind:
                x, s = sf.read(esc_root / "audio" / row["filename"], dtype="float32")
                bank[kind].append(resample(x, s, sr))
    return bank


def librispeech_manifest(root: Path, n: int, max_seconds: float = 15.0) -> list[dict]:
    rows = []
    for trans in sorted(root.rglob("*.trans.txt")):
        for line in trans.read_text().splitlines():
            uid, text = line.split(" ", 1)
            flac = trans.parent / f"{uid}.flac"
            info = sf.info(flac)
            if info.duration <= max_seconds:
                rows.append({"id": f"ls_{uid}@clean", "utt": f"ls_{uid}", "condition": "clean",
                             "source": "librispeech", "audio": str(flac), "text": text.lower(),
                             "alerts": [], "duration": info.duration})
    rng = np.random.default_rng(0)
    idx = sorted(rng.choice(len(rows), size=min(n, len(rows)), replace=False))
    return [rows[i] for i in idx]


# ------------------------------------------------------------ radio dataset

def build_radio_dataset(out_dir: Path, voice_dir: Path, n: int = 300, seed: int = 0,
                        conditions: list[str] | None = None,
                        noise_bank: dict[str, list[np.ndarray]] | None = None) -> list[dict]:
    conditions = conditions or list(CONDITIONS)
    clean_dir = out_dir / "_tts"
    clean_dir.mkdir(parents=True, exist_ok=True)
    items = scripts.generate(n, seed)
    speakers = tts.pick_speakers(n, seed=seed)
    manifest = []
    for i, (item, spk) in enumerate(zip(items, speakers)):
        clean = clean_dir / f"{item['id']}.wav"
        if not clean.exists():
            tts.synthesize(item["text"], clean, voice_dir, spk)
        x, sr = sf.read(clean, dtype="float32")
        for cond in conditions:
            cdir = out_dir / cond
            cdir.mkdir(exist_ok=True)
            path = cdir / f"{item['id']}.wav"
            cfg = channel_for(cond, i, seed)
            if cfg is None:
                shutil.copy(clean, path)
                dur = len(x) / sr
            else:
                y, sro = degrade(x, sr, cfg, noise_bank)
                sf.write(path, y, sro, subtype="PCM_16")
                dur = len(y) / sro
            manifest.append({"id": f"{item['id']}@{cond}", "utt": item["id"], "condition": cond,
                             "source": "radio", "audio": str(path), "text": item["text"],
                             "alerts": item["alerts"], "kind": item["kind"], "speaker": spk,
                             "duration": dur})
    scripts.write_jsonl(manifest, out_dir / "manifest.jsonl")
    return manifest
