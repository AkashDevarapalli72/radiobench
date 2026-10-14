"""Multi-voice synthetic speech with Piper (https://github.com/OHF-Voice/piper1-gpl).

Uses the multi-speaker ``en_US-libritts_r-medium`` voice so one model gives many
distinct speakers. Voice files (.onnx + .onnx.json) are fetched with
``python -m piper.download_voices`` (see README / `radiobench download`).
"""
from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path

VOICE = "en_US-libritts_r-medium"


def synthesize(text: str, out_wav: str | Path, voice_dir: str | Path, speaker: int = 0) -> None:
    model = Path(voice_dir) / f"{VOICE}.onnx"
    if not model.exists():
        raise FileNotFoundError(f"{model} missing; run `radiobench download --tts`")
    subprocess.run(
        [sys.executable, "-m", "piper", "-m", str(model), "-f", str(out_wav), "-s", str(speaker)],
        input=text, text=True, check=True)


def pick_speakers(n_utts: int, n_speakers: int = 40, seed: int = 0, max_id: int = 900) -> list[int]:
    r = random.Random(seed)
    pool = r.sample(range(max_id), n_speakers)
    return [pool[i % n_speakers] for i in range(n_utts)]
