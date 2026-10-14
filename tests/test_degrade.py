import numpy as np
import pytest

from radiobench import degrade as D

SR = 16000


def speechlike(sec=3.0, sr=SR, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(sec * sr)) / sr
    env = (np.sin(2 * np.pi * 3 * t) > -0.3).astype(float)  # syllable-ish on/off
    x = sum(np.sin(2 * np.pi * f * t) for f in (150, 450, 900, 1800, 5000)) * env
    return (0.3 * x / np.abs(x).max() + 0.002 * rng.standard_normal(len(t))).astype(np.float32)


def band_energy(x, sr, lo, hi):
    X = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / sr)
    return X[(f >= lo) & (f < hi)].sum() / X.sum()


def cfg(**kw):
    base = dict(codec=None, band=None)
    base.update(kw)
    return D.ChannelConfig(**base)


def test_output_rate_and_length():
    x = speechlike()
    y, sr = D.degrade(x, SR, cfg())
    assert sr == 8000 and abs(len(y) - len(x) // 2) <= 2


def test_bandpass_removes_out_of_band():
    x = speechlike()
    y, sr = D.degrade(x, SR, cfg(band=(300, 3000)))
    assert band_energy(x, SR, 3500, 8000) > 0.01
    assert band_energy(y, sr, 3400, 4000) < 0.005


@pytest.mark.parametrize("snr", [20, 10, 5, 0])
@pytest.mark.parametrize("kind", D.NOISE_KINDS)
def test_snr_is_accurate(snr, kind):
    x = D.resample(speechlike(), SR, 8000)
    noise = D.get_noise(kind, len(x), 8000, np.random.default_rng(1))
    mixed = D.mix_at_snr(x, noise, snr, 8000)
    resid = mixed - x
    measured = 10 * np.log10(D.active_power(x, 8000) / np.mean(resid ** 2))
    assert abs(measured - snr) < 0.1


def test_clipping_flattens_peaks():
    x = D.resample(speechlike(), SR, 8000)
    y = D.hard_clip(x, 0.3)
    assert np.mean(np.abs(y) > 0.89) > 0.05 and np.max(np.abs(y)) <= 0.9 + 1e-6


def test_squelch_tail_extends_and_decays():
    x = D.resample(speechlike(), SR, 8000)
    y = D.squelch_tail(x, 8000, 200, np.random.default_rng(0))
    assert len(y) == len(x) + 1600
    tail = y[len(x):]
    assert np.mean(tail[:400] ** 2) > np.mean(tail[-400:] ** 2)


def test_dropouts_zero_segments():
    x = np.ones(8000 * 10, dtype=np.float32)
    y = D.dropouts(x, 8000, rate=2.0, ms=(50, 100), rng=np.random.default_rng(0))
    assert (y == 0).sum() > 0 and (y == 0).sum() < len(y)


def test_deterministic_given_seed():
    x = speechlike()
    c = cfg(noise_kind="wind", snr_db=5, dropout_rate=1.0, squelch_tail_ms=100, seed=7)
    a, _ = D.degrade(x, SR, c)
    b, _ = D.degrade(x, SR, c)
    assert np.array_equal(a, b)


def test_noise_requires_snr():
    with pytest.raises(ValueError):
        D.degrade(speechlike(), SR, cfg(noise_kind="siren"))


@pytest.mark.skipif(not D.codec_available("gsm"), reason="ffmpeg with GSM encoder not available")
def test_gsm_roundtrip_preserves_length_and_alters_signal():
    x = D.resample(speechlike(), SR, 8000)
    y = D.apply_codec(x, 8000, "gsm")
    assert abs(len(y) - len(x)) <= 400 and not np.allclose(x[: len(y)], y[: len(x)], atol=1e-3)
