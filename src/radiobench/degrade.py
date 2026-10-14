"""Radio channel degradation simulator.

Chain (all steps optional / configurable):
    resample -> band-pass -> acoustic noise at target SNR -> clipping
    -> push-to-talk squelch tail -> low-bitrate codec -> dropouts

The codec (GSM / AMR-NB through ffmpeg) is a *stand-in* for a P25 IMBE/AMBE
vocoder, which cannot be reproduced here. Noise comes from an ESC-50 bank when
supplied, otherwise from the synthetic generators below.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

NOISE_KINDS = ("siren", "wind", "traffic")
SR = 8000


@dataclass
class ChannelConfig:
    sr_out: int = SR
    band: tuple[float, float] | None = (300.0, 3000.0)
    codec: str | None = "gsm"            # "gsm" | "amr_nb" | None
    noise_kind: str | None = None        # one of NOISE_KINDS, or None
    snr_db: float | None = None          # required if noise_kind is set
    clip_level: float | None = None      # fraction of peak at which the signal hard-clips (e.g. 0.4)
    squelch_tail_ms: int = 0             # PTT release burst appended after speech
    dropout_rate: float = 0.0            # expected dropouts per second
    dropout_ms: tuple[int, int] = (40, 120)
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- primitives

def find_ffmpeg() -> str | None:
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32)
    g = int(np.gcd(sr_in, sr_out))
    return signal.resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def bandpass(x: np.ndarray, sr: int, lo: float, hi: float, order: int = 4) -> np.ndarray:
    hi = min(hi, sr / 2 * 0.99)
    sos = signal.butter(order, [lo, hi], btype="band", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, x).astype(np.float32)


def active_power(x: np.ndarray, sr: int, frame_ms: int = 20) -> float:
    """Mean power over frames that carry speech (silence would otherwise understate the signal)."""
    n = max(1, int(sr * frame_ms / 1000))
    m = len(x) // n
    if m == 0:
        return float(np.mean(x ** 2)) + 1e-12
    p = np.mean(x[: m * n].reshape(m, n) ** 2, axis=1)
    active = p[p > 0.05 * p.max()] if p.max() > 0 else p
    return float(np.mean(active)) + 1e-12


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float, sr: int) -> np.ndarray:
    ps, pn = active_power(speech, sr), float(np.mean(noise ** 2)) + 1e-12
    gain = np.sqrt(ps / (pn * 10 ** (snr_db / 10)))
    return (speech + gain * noise).astype(np.float32)


# ------------------------------------------------------------------- noise

def synth_noise(kind: str, n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / sr
    if kind == "siren":  # two-tone wail with harmonics
        f = 850 + 350 * np.sin(2 * np.pi * rng.uniform(0.15, 0.3) * t + rng.uniform(0, 6.28))
        ph = 2 * np.pi * np.cumsum(f) / sr
        x = np.sin(ph) + 0.5 * np.sin(2 * ph) + 0.25 * np.sin(3 * ph)
    elif kind == "wind":  # gusty low-passed noise
        w = rng.standard_normal(n)
        sos = signal.butter(2, 500, btype="low", fs=sr, output="sos")
        gust = np.interp(t, np.linspace(0, t[-1] + 1e-9, 12), rng.uniform(0.2, 1.0, 12))
        x = signal.sosfilt(sos, w) * gust
    elif kind == "traffic":  # rumble + broadband road noise + occasional horn
        w = rng.standard_normal(n)
        sos = signal.butter(2, 1200, btype="low", fs=sr, output="sos")
        x = signal.sosfilt(sos, w) + 0.8 * np.sin(2 * np.pi * rng.uniform(60, 110) * t)
        for _ in range(int(rng.integers(0, 3))):
            s = int(rng.uniform(0, max(1, n - sr // 2)))
            e = min(n, s + int(sr * 0.4))
            x[s:e] += 0.6 * np.sin(2 * np.pi * rng.uniform(380, 520) * t[s:e])
    else:
        raise ValueError(f"unknown noise kind {kind!r}; expected one of {NOISE_KINDS}")
    return (x / (np.std(x) + 1e-9)).astype(np.float32)


def get_noise(kind: str, n: int, sr: int, rng: np.random.Generator,
              bank: dict[str, list[np.ndarray]] | None = None) -> np.ndarray:
    clips = (bank or {}).get(kind)
    if not clips:
        return synth_noise(kind, n, sr, rng)
    c = clips[int(rng.integers(len(clips)))]
    c = np.tile(c, int(np.ceil(n / len(c))) + 1)
    off = int(rng.integers(0, len(c) - n + 1))
    return c[off: off + n].astype(np.float32)


# --------------------------------------------------------------- impairments

def hard_clip(x: np.ndarray, level: float) -> np.ndarray:
    peak = np.max(np.abs(x)) + 1e-9
    y = np.clip(x / peak, -level, level)
    return (y / level * 0.9).astype(np.float32)


def squelch_tail(x: np.ndarray, sr: int, tail_ms: int, rng: np.random.Generator) -> np.ndarray:
    n = int(sr * tail_ms / 1000)
    if n == 0:
        return x
    env = np.exp(-np.arange(n) / (0.35 * n))
    burst = rng.standard_normal(n) * env
    sos = signal.butter(2, [300, 2800], btype="band", fs=sr, output="sos")
    burst = signal.sosfilt(sos, burst)
    burst *= 0.3 * np.sqrt(active_power(x, sr)) / (np.std(burst) + 1e-9)
    return np.concatenate([x, burst.astype(np.float32)])


def dropouts(x: np.ndarray, sr: int, rate: float, ms: tuple[int, int],
             rng: np.random.Generator) -> np.ndarray:
    y = x.copy()
    k = int(rng.poisson(rate * len(x) / sr))
    fade = int(0.005 * sr)
    for _ in range(k):
        d = int(sr * rng.integers(ms[0], ms[1] + 1) / 1000)
        s = int(rng.integers(0, max(1, len(x) - d)))
        seg = np.zeros(d, dtype=np.float32)
        y[s: s + d] = seg
        if s > fade:  # soften the edge so it sounds like a packet loss, not a click
            y[s - fade: s] *= np.linspace(1, 0, fade, dtype=np.float32)
    return y


_CODECS = {
    "gsm": ("gsm", ["-c:a", "gsm", "-ar", "8000", "-ac", "1"]),
    "amr_nb": ("amr", ["-c:a", "libopencore_amrnb", "-ar", "8000", "-ac", "1", "-b:a", "4.75k"]),
}


def codec_available(codec: str) -> bool:
    ff = find_ffmpeg()
    if not ff or codec not in _CODECS:
        return False
    out = subprocess.run([ff, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    return _CODECS[codec][1][1] in out


def apply_codec(x: np.ndarray, sr: int, codec: str) -> np.ndarray:
    """Round-trip through a low-bitrate speech codec with ffmpeg."""
    ff = find_ffmpeg()
    if ff is None:
        raise RuntimeError("ffmpeg not found (install it or `pip install imageio-ffmpeg`)")
    if codec == "amr_nb" and not codec_available("amr_nb"):
        warnings.warn("this ffmpeg build has no AMR-NB encoder; falling back to GSM 06.10")
        codec = "gsm"
    if sr != 8000:
        raise ValueError("speech codecs here are narrowband: resample to 8 kHz first")
    ext, args = _CODECS[codec]
    with tempfile.TemporaryDirectory() as d:
        src, enc, dst = Path(d) / "in.wav", Path(d) / f"enc.{ext}", Path(d) / "out.wav"
        sf.write(src, x, sr, subtype="PCM_16")
        base = [ff, "-y", "-loglevel", "error"]
        subprocess.run([*base, "-i", str(src), *args, str(enc)], check=True)
        subprocess.run([*base, "-i", str(enc), "-ar", str(sr), "-ac", "1", str(dst)], check=True)
        y, _ = sf.read(dst, dtype="float32")
    return np.pad(y, (0, max(0, len(x) - len(y))))[: max(len(x), len(y))]


# ------------------------------------------------------------------ pipeline

def degrade(x: np.ndarray, sr: int, cfg: ChannelConfig,
            noise_bank: dict[str, list[np.ndarray]] | None = None) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(cfg.seed)
    if x.ndim > 1:
        x = x.mean(axis=1)
    y = resample(x.astype(np.float32), sr, cfg.sr_out)
    if cfg.band:
        y = bandpass(y, cfg.sr_out, *cfg.band)
    if cfg.noise_kind:
        if cfg.snr_db is None:
            raise ValueError("snr_db is required when noise_kind is set")
        y = mix_at_snr(y, get_noise(cfg.noise_kind, len(y), cfg.sr_out, rng, noise_bank),
                       cfg.snr_db, cfg.sr_out)
    if cfg.clip_level:
        y = hard_clip(y, cfg.clip_level)
    if cfg.squelch_tail_ms:
        y = squelch_tail(y, cfg.sr_out, cfg.squelch_tail_ms, rng)
    if cfg.codec:
        y = apply_codec(y, cfg.sr_out, cfg.codec)
    if cfg.dropout_rate:
        y = dropouts(y, cfg.sr_out, cfg.dropout_rate, cfg.dropout_ms, rng)
    peak = np.max(np.abs(y))
    if peak > 0.99:
        y = y / peak * 0.99
    return y.astype(np.float32), cfg.sr_out


def degrade_file(src: str | Path, dst: str | Path, cfg: ChannelConfig,
                 noise_bank: dict[str, list[np.ndarray]] | None = None) -> float:
    x, sr = sf.read(src, dtype="float32")
    y, sr_out = degrade(x, sr, cfg, noise_bank)
    sf.write(dst, y, sr_out, subtype="PCM_16")
    return len(y) / sr_out
