"""Command line entry point: `radiobench <command>`."""
from __future__ import annotations

import argparse
from pathlib import Path

from . import data, scripts
from .degrade import ChannelConfig, degrade_file


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="radiobench")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="fetch datasets / TTS voice into --root")
    d.add_argument("--root", default="data")
    d.add_argument("--librispeech", action="store_true")
    d.add_argument("--esc50", action="store_true")
    d.add_argument("--tts", action="store_true")

    b = sub.add_parser("build-dataset", help="synthesise scripts and apply every condition")
    b.add_argument("--root", default="data")
    b.add_argument("-n", type=int, default=300)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--librispeech-n", type=int, default=100)

    g = sub.add_parser("degrade", help="degrade one audio file")
    g.add_argument("src")
    g.add_argument("dst")
    g.add_argument("--snr", type=float)
    g.add_argument("--noise", default="siren", choices=["siren", "wind", "traffic"])
    g.add_argument("--codec", default="gsm", choices=["gsm", "amr_nb", "none"])
    g.add_argument("--seed", type=int, default=0)

    e = sub.add_parser("evaluate", help="run all systems x conditions")
    e.add_argument("--config", default="configs/default.yaml")

    r = sub.add_parser("report", help="tables + plots from results.csv")
    r.add_argument("results_csv")

    a = p.parse_args(argv)
    root = Path(getattr(a, "root", "data"))

    if a.cmd == "download":
        if a.librispeech:
            data.download_librispeech(root)
        if a.esc50:
            data.download_esc50(root)
        if a.tts:
            data.download_voice(root / "voices")
    elif a.cmd == "build-dataset":
        esc = root / "ESC-50-master"
        bank = data.load_noise_bank(esc) if esc.exists() else None
        print("noise source:", "ESC-50" if bank else "synthetic generators")
        rows = data.build_radio_dataset(root / "radio", root / "voices", a.n, a.seed, noise_bank=bank)
        ls = root / "LibriSpeech" / "test-clean"
        if ls.exists():
            scripts.write_jsonl(data.librispeech_manifest(ls, a.librispeech_n), root / "librispeech.jsonl")
        print(f"{len(rows)} radio utterances written")
    elif a.cmd == "degrade":
        cfg = ChannelConfig(codec=None if a.codec == "none" else a.codec, seed=a.seed,
                            noise_kind=a.noise if a.snr is not None else None, snr_db=a.snr)
        print(f"wrote {a.dst} ({degrade_file(a.src, a.dst, cfg):.1f}s)")
    elif a.cmd == "evaluate":
        from . import evaluate, report
        report.make_report(evaluate.run(a.config))
    elif a.cmd == "report":
        from . import report
        print(report.make_report(a.results_csv))


if __name__ == "__main__":
    main()
