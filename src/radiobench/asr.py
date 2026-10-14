"""faster-whisper wrapper with optional vocabulary biasing and timing."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import soundfile as sf

from .degrade import resample


@dataclass
class Transcript:
    text: str
    latency_s: float
    duration_s: float


class Recognizer:
    def __init__(self, size: str = "tiny", device: str = "cpu", compute_type: str | None = None,
                 prompt: str | None = None, beam_size: int = 5):
        from faster_whisper import WhisperModel  # heavy import, keep lazy
        compute_type = compute_type or ("int8" if device == "cpu" else "float16")
        self.model = WhisperModel(size, device=device, compute_type=compute_type)
        self.prompt = prompt
        self.beam_size = beam_size

    def transcribe(self, path: str) -> Transcript:
        x, sr = sf.read(path, dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        dur = len(x) / sr
        x16 = resample(x, sr, 16000)  # Whisper is 16 kHz; the 8 kHz band-limit is preserved
        t0 = time.perf_counter()
        segs, _ = self.model.transcribe(
            np.ascontiguousarray(x16), language="en", beam_size=self.beam_size,
            initial_prompt=self.prompt, condition_on_previous_text=False, temperature=0.0)
        text = " ".join(s.text.strip() for s in segs)
        return Transcript(text, time.perf_counter() - t0, dur)
